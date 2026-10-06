"""Point-in-time access: what was public at a decision time (objective 2).

    features_available_at(decision_time)  every table row with published_at <= decision_time
    bid_inputs(bid_day)                   the rows a model may use for delivery day D+1,
                                          decided at 05:00 New York time on D
    gfs_rule(weather, d1)                 the three-day weather rule: one GFS value per point and
                                          D+1 hour (day2 up to 21:00, day3 for 22:00 and 23:00)
    assert_no_lookahead(frames, t)        raises LookaheadError if any row has published_at > t
    rule_violations(table, df)            counts rows whose published_at breaks the documented rule

Candidate windows deliberately reach PAST the decision time (e.g. the target day's own day-ahead
prices, the isolf file named D+1), so it is the published_at filter, not a date window, that
keeps the future out; `Frames.excluded` counts what the filter removed.
No price statistic is computed here.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from common import (DA_POST_BOUND, GFS_DAY2_LAST_LOCAL_HOUR, GFS_LEAD_HOURS, PARQUET, RT_LAG, TZ,
                    decision_time, gfs_published_at, local_at, local_midnight)

TABLES = ["prices_zone", "load_forecast", "outages", "weather_gfs"]
DA_COLS = ["delivery_hour", "zone", "ptid", "da_lbmp", "da_loss", "da_congestion_raw", "da_congestion",
           "da_energy", "da_published_at"]
RT_COLS = ["delivery_hour", "zone", "ptid", "rt_lbmp", "rt_loss", "rt_congestion_raw", "rt_congestion",
           "rt_energy", "rt_published_at", "rt_revised"]


class LookaheadError(AssertionError):
    pass


class Frames(dict):
    """dict of DataFrames, plus .excluded = rows the published_at filter removed per frame."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.excluded: dict[str, int] = {}


def _utc_naive(s: pd.Series) -> np.ndarray:
    return s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()


def _ts64(t: pd.Timestamp) -> np.datetime64:
    return np.datetime64(pd.Timestamp(t).tz_convert("UTC").tz_localize(None))


class Store:
    """Loads the parquet tables once and serves time windows. `tables` overrides (for tests)."""

    def __init__(self, root: Path = PARQUET, tables: dict | None = None):
        self.root = Path(root)
        self._t: dict[str, pd.DataFrame] = {}
        self._key: dict[str, np.ndarray] = {}
        for name, df in (tables or {}).items():
            self._install(name, df)

    def _install(self, name: str, df: pd.DataFrame):
        df = df.copy()
        if name in ("load_forecast",):
            df["issue_date"] = pd.to_datetime(df["issue_date"])
            key = df["issue_date"]
        elif name == "outages":
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
            key = df["snapshot_date"]
        elif name == "weather_gfs":
            key = df["target_hour"]
        else:
            key = df["delivery_hour"]
        order = np.argsort(_utc_naive(key) if hasattr(key.dtype, "tz") else key.to_numpy(), kind="stable")
        df = df.iloc[order].reset_index(drop=True)
        key = key.iloc[order].reset_index(drop=True)
        self._t[name] = df
        self._key[name] = _utc_naive(key) if hasattr(key.dtype, "tz") else key.to_numpy()

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._t:
            self._install(name, pd.read_parquet(self.root / f"{name}.parquet"))
        return self._t[name]

    def window(self, name: str, lo, hi) -> pd.DataFrame:
        """Rows whose time key (delivery/target hour, or issue/snapshot date) is in [lo, hi)."""
        self.table(name)
        k = self._key[name]
        if np.issubdtype(k.dtype, np.datetime64) and getattr(lo, "tzinfo", None) is not None:
            lo, hi = _ts64(lo), _ts64(hi)
        else:
            lo, hi = np.datetime64(pd.Timestamp(lo)), np.datetime64(pd.Timestamp(hi))
        i, j = np.searchsorted(k, lo, "left"), np.searchsorted(k, hi, "left")
        return self._t[name].iloc[i:j]

    def gen_window(self, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
        """prices_gen rows with delivery_hour in [lo, hi), reading only the month files needed."""
        months = pd.period_range(pd.Timestamp(lo.tz_convert(TZ).date()).to_period("M"),
                                 pd.Timestamp(hi.tz_convert(TZ).date()).to_period("M"), freq="M")
        files = [str(self.root / "prices_gen" / f"{p.strftime('%Y%m')}.parquet") for p in months]
        files = [f for f in files if Path(f).exists()]
        if not files:
            return pd.DataFrame()
        d = ds.dataset(files, format="parquet")
        typ = d.schema.field("delivery_hour").type
        flt = (ds.field("delivery_hour") >= pa.scalar(lo, type=typ)) & \
              (ds.field("delivery_hour") < pa.scalar(hi, type=typ))
        return d.to_table(filter=flt).to_pandas()


_DEFAULT: Store | None = None


def default_store() -> Store:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Store()
    return _DEFAULT


def _published(frames: Frames, name: str, df: pd.DataFrame, pub_col: str, t: pd.Timestamp):
    pub = df[pub_col]
    ok = pub.notna() & (pub <= t)
    out = df[ok]
    if pub_col != "published_at":
        out = out.rename(columns={pub_col: "published_at"})
    frames[name] = out
    frames.excluded[name] = int((~ok).sum())


def features_available_at(decision: pd.Timestamp, store: Store | None = None, lookback_days: int = 400,
                          include_gen: bool = False, gen_lookback_days: int = 3) -> Frames:
    """Every row public at `decision` (tz-aware) within the lookback, one frame per source.
    Day-ahead and real-time prices are split into separate frames because they are published at
    different times (da_published_at, rt_published_at)."""
    s = store or default_store()
    t = pd.Timestamp(decision)
    if t.tzinfo is None:
        raise ValueError("decision time must be timezone-aware")
    t = t.tz_convert(TZ)
    ahead = pd.Timedelta(days=4)              # candidates reach past t on purpose
    f = Frames()
    pz = s.window("prices_zone", t - pd.Timedelta(days=lookback_days), t + ahead)
    _published(f, "da_prices_zone", pz[DA_COLS], "da_published_at", t)
    _published(f, "rt_prices_zone", pz[RT_COLS], "rt_published_at", t)
    day = pd.Timestamp(t.date())
    lf = s.window("load_forecast", day - pd.Timedelta(days=10), day + ahead)
    _published(f, "load_forecast", lf, "published_at", t)
    og = s.window("outages", day - pd.Timedelta(days=30), day + ahead)
    _published(f, "outages", og, "published_at", t)
    wx = s.window("weather_gfs", t - pd.Timedelta(days=30), t + ahead)
    _published(f, "weather_gfs", wx[wx["temperature_2m_c"].notna()], "published_at", t)
    if include_gen:
        g = s.gen_window(t - pd.Timedelta(days=gen_lookback_days), t + ahead)
        if len(g):
            gda = g[["delivery_hour", "ptid", "name", "point_type", "da_lbmp", "da_loss",
                     "da_congestion_raw", "da_congestion", "da_energy", "da_published_at"]]
            grt = g[["delivery_hour", "ptid", "name", "point_type", "rt_lbmp", "rt_loss",
                     "rt_congestion_raw", "rt_congestion", "rt_energy", "rt_published_at", "rt_revised"]]
            _published(f, "da_prices_gen", gda, "da_published_at", t)
            _published(f, "rt_prices_gen", grt, "rt_published_at", t)
    return f


def bid_inputs(bid_day: dt.date, store: Store | None = None, **kw) -> Frames:
    """Inputs a model may use for delivery day D+1, decided at 05:00 on D = bid_day:
    price history, the newest public load-forecast vintage for every D+1 hour, the newest public
    outage snapshot, and the GFS forecasts for D+1 hours that were already public."""
    t = decision_time(bid_day)
    f = features_available_at(t, store, **kw)
    d1 = pd.Timestamp(bid_day + dt.timedelta(days=1))
    out = Frames()
    out.excluded = dict(f.excluded)
    for k in ("da_prices_zone", "rt_prices_zone", "da_prices_gen", "rt_prices_gen"):
        if k in f:
            out[k] = f[k]
    lf = f["load_forecast"]
    lf = lf[local_midnight(lf["target_hour"]) == d1]
    lf = lf.sort_values("published_at").drop_duplicates(["zone", "target_hour"], keep="last")
    out["load_forecast_d1"] = lf
    og = f["outages"]
    out["outages_latest"] = og[og["snapshot_date"] == og["snapshot_date"].max()] if len(og) else og
    out["weather_d1"] = gfs_rule(f["weather_gfs"], d1)
    return out


DAY2, DAY3 = GFS_LEAD_HOURS[2], GFS_LEAD_HOURS[3]


def gfs_rule(wx: pd.DataFrame, d1: pd.Timestamp) -> pd.DataFrame:
    """Three-day weather rule (OBJECTIVES addendum, 6 Oct evening). `wx` must already be filtered
    to rows public at the decision time with a non-null value (features_available_at does both).
    For each point and hour of delivery day d1 (naive local midnight) keep one row:
      * previous_day2 (run_lead_hours 48) for hours beginning up to 21:00 New York time,
      * previous_day3 (72) for 22:00 and 23:00, and wherever day2 is absent.
    Day2 is absent in two cases only: the archive's null hours, and 21:00 on the fall-back day,
    whose day2 value is published at 21:00 EST on D+1 minus 40 h = 06:00 EDT on D, after the
    deadline, so the published_at filter has already removed it and day3 is used. On the
    spring-forward day the day2 value for 22:00 EDT is public exactly at the deadline, but the rule
    still uses day3 there."""
    if len(wx) == 0:
        return wx
    wx = wx[local_midnight(wx["target_hour"]) == d1]
    hour = wx["target_hour"].dt.tz_convert(TZ).dt.hour
    lead = wx["run_lead_hours"]
    keep = ((lead == DAY2) & (hour <= GFS_DAY2_LAST_LOCAL_HOUR)) | (lead == DAY3)
    wx = wx[keep].sort_values(["point", "target_hour", "run_lead_hours"], kind="stable")
    return wx.drop_duplicates(["point", "target_hour"], keep="first")


def assert_no_lookahead(frames: dict, decision: pd.Timestamp):
    """Raise LookaheadError naming each frame that holds a row published after `decision`."""
    t = pd.Timestamp(decision)
    bad = {}
    for name, df in frames.items():
        if df is None or len(df) == 0:
            continue
        if "published_at" not in df.columns:
            raise LookaheadError(f"{name}: no published_at column")
        pub = df["published_at"]
        n = int((pub.isna() | (pub > t)).sum())
        if n:
            bad[name] = n
    if bad:
        raise LookaheadError(f"rows published after {t}: {bad}")


def rule_violations(table: str, df: pd.DataFrame) -> dict[str, int]:
    """Count rows whose published_at does not follow the documented rule (README.md)."""
    v = {}
    if table in ("prices_zone", "prices_gen"):
        dd = local_midnight(df["delivery_hour"])
        da = df["da_published_at"].dropna()
        bound = local_at(dd[da.index] - pd.Timedelta(days=1), DA_POST_BOUND)
        v["da_before_11h_bound"] = int((da < bound).sum())
        v["da_before_file_written"] = int((da < df.loc[da.index, "da_file_written_at"]).sum())
        rt = df["rt_published_at"].dropna()
        v["rt_before_hour_end_plus_lag"] = int((rt < df.loc[rt.index, "delivery_hour"]
                                                + pd.Timedelta(hours=1) + RT_LAG).sum())
        rev = df.loc[rt.index, "rt_revised"].astype(bool)
        v["rt_revised_before_rewrite"] = int((rt[rev] < df.loc[rt.index[rev], "rt_file_written_at"]).sum())
        both = df[["da_published_at", "rt_published_at"]]
        latest = both["da_published_at"].where(both["rt_published_at"].isna()
                                               | (both["da_published_at"] >= both["rt_published_at"]),
                                               both["rt_published_at"])
        v["published_at_not_latest_component"] = int((df["published_at"] != latest).fillna(False).sum())
    elif table in ("load_forecast", "outages"):
        v["published_not_file_written"] = int((df["published_at"] != df["file_written_at"]).sum())
        v["published_missing"] = int(df["published_at"].isna().sum())
    elif table == "weather_gfs":
        lead_ok = df["run_lead_hours"].isin(list(GFS_LEAD_HOURS.values()))
        v["run_lead_not_48_or_72"] = int((~lead_ok).sum())
        exp = gfs_published_at(df.loc[lead_ok, "target_hour"], df.loc[lead_ok, "run_lead_hours"])
        v["published_not_target_minus_lead_plus_8h"] = int((df.loc[lead_ok, "published_at"] != exp).sum())
    return {k: n for k, n in v.items() if n}
