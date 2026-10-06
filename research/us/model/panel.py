"""The panel: one row per (bid day D, zone, delivery hour of D+1), with the base features a trader
had at 05:00 New York time on D (CONTRACT.md, "The panel").

Every feature for D is computed from timing.features_available_at(05:00 on D) on a LockedStore,
which never loads a row for a delivery date on or after the holdout start (lock.py). The label
`gap` = rt_lbmp - da_lbmp of the delivery hour is kept apart (columns y_*, gap) and is never a feature.

    build_panel(start, end, include_gen=False)  delivery dates start..end inclusive
    BASE_FEATURES, GEN_FEATURES                 feature names (base set; zone-plus-generator ablation)
    load_panel / save_panel                     parquet cache, both pass through the lock

What a trader has at 05:00 on D (README.md, published_at rules):
  * day-ahead prices for D (posted by 11:00 on D-1) and every earlier day; not D+1 (posted on D);
  * real-time prices for every hour that ended by 04:45 on D, except days whose file NYISO rewrote
    after midnight (those count only from the rewrite);
  * the newest load-forecast vintage published by 05:00 (normally the file named D, written ~07:05 on D-1);
  * generator and border-point prices on the same rules.
"""
from __future__ import annotations

import datetime as dt
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lock  # noqa: F401  (imported first: the lock is in place before any data code)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

PIPELINE_DIR = lock.STUDY_DIR / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import timing as T  # noqa: E402
from common import HOME, PARQUET, TZ, ZONES, decision_time  # noqa: E402

CACHE = HOME / "cache"
BUILD_START = dt.date(2020, 1, 1)
BUILD_END = dt.date(2023, 12, 31)
LOOKBACK_DAYS = 370                 # the trailing 365-day mean needs 365 days before 05:00 on D
SPIKE = 50.0                        # USD/MWh, |gap| counted as a spike
ZONE_CODE = {z: i for i, z in enumerate(ZONES)}
EXTERNAL = ["H Q", "NPX", "O H", "PJM"]

KEY_COLS = ["bid_date", "delivery_date", "delivery_hour", "zone", "hour"]
LABEL_COLS = ["gap", "y_da_lbmp", "y_rt_lbmp", "label_published_at"]

ZH_FEATURES = ["da_d0_h", "da_d0_cong_h", "da_d0_loss_h", "da_d0_energy_h", "da_dm1_h", "da_7d_h",
               "gap_dm1_h", "rt_dm1_h", "cgap_dm1_h", "gap_7d_h", "cgap_7d_h",
               "gap_28d_h", "gap_28d_h_med", "gap_28d_h_std", "spike_28d_h", "negspike_28d_h",
               "gap_365d_h", "gap_365d_h_med", "n_365d_h"]
Z_FEATURES = ["gap_dm1_mean", "n_rt_dm1", "gap_7d_all", "gap_28d_all", "spike_7d_all",
              "gap_d0_early", "rt_d0_early", "da_d0_mean", "da_d0_max", "rt_last", "hours_since_rt"]
LF_FEATURES = ["lf_h", "lf_peak", "lf_h_rel", "lf_d1_minus_d0_h", "lf_rev_h", "lf_lead",
               "lf_nyiso_h", "lf_nyiso_peak"]
CAL_FEATURES = ["zone_code", "hour", "dow", "month", "doy_sin", "doy_cos", "holiday", "weekend"]
BASE_FEATURES = ZH_FEATURES + Z_FEATURES + LF_FEATURES + CAL_FEATURES
GEN_FEATURES = ([f"ext_da_d0_h_{n.replace(' ', '')}" for n in EXTERNAL]
                + [f"ext_gap_dm1_h_{n.replace(' ', '')}" for n in EXTERNAL]
                + ["gen_dacong_d0_h_p05", "gen_dacong_d0_h_p50", "gen_dacong_d0_h_p95", "gen_dacong_d0_h_std",
                   "gen_gap_dm1_h_p05", "gen_gap_dm1_h_p50", "gen_gap_dm1_h_p95", "gen_rtcong_dm1_h_absmean",
                   "n_gen_points"])
CATEGORICAL = ["zone_code"]
INT_FEATURES = {"zone_code", "hour", "dow", "month", "holiday", "weekend"}
BASELINE_SIGNAL = "gap_365d_h"      # the baseline takes the side of this trailing 365-day mean


# ---------------------------------------------------------------- locked data access

GEN_COLS = ["delivery_hour", "ptid", "name", "point_type", "da_lbmp", "da_loss", "da_congestion_raw",
            "da_congestion", "da_energy", "da_published_at", "rt_lbmp", "rt_loss", "rt_congestion_raw",
            "rt_congestion", "rt_energy", "rt_published_at", "rt_revised"]


def _bound_scalar(field_type: pa.DataType, end: dt.date):
    """`end` as a pyarrow scalar comparable with a column of this type."""
    if pa.types.is_timestamp(field_type):
        ts = pd.Timestamp(end, tz=TZ) if field_type.tz else pd.Timestamp(end)
        return pa.scalar(ts, type=field_type)
    if pa.types.is_date(field_type):
        return pa.scalar(end, type=field_type)
    raise TypeError(f"cannot bound a column of type {field_type}")


def read_locked(path: Path, col: str, end: dt.date, columns: list[str] | None = None) -> pd.DataFrame:
    """Rows of a parquet file with col < end, filtered inside pyarrow, then checked by the lock."""
    d = ds.dataset(str(path), format="parquet")
    flt = ds.field(col) < _bound_scalar(d.schema.field(col).type, end)
    df = d.to_table(filter=flt, columns=columns).to_pandas()
    lock.assert_build_only(df[col])
    return df


class LockedStore(T.Store):
    """timing.Store that can only hold rows for delivery dates before `end` (2024-01-01 while locked)."""
    TIME_COL = {"prices_zone": "delivery_hour", "load_forecast": "target_hour",
                "outages": "snapshot_date", "weather_gfs": "target_hour"}

    def __init__(self, root: Path = PARQUET, end: dt.date | None = None, env: dict | None = None,
                 gen_cache_months: int = 3):
        self.end = lock.read_end(end, env=env)
        self.end_ts = pd.Timestamp(self.end, tz=TZ)
        self._gen: OrderedDict = OrderedDict()
        self._gen_n = gen_cache_months
        super().__init__(root)

    def table(self, name: str) -> pd.DataFrame:
        if name not in self._t:
            col = self.TIME_COL[name]
            # An outage snapshot dated S serves bids on S for delivery S+1: keep S + 1 < end.
            end = self.end - dt.timedelta(days=1) if name == "outages" else self.end
            self._install(name, read_locked(self.root / f"{name}.parquet", col, end))
        return self._t[name]

    def _gen_month(self, period: pd.Period) -> pd.DataFrame | None:
        if period in self._gen:
            self._gen.move_to_end(period)
            return self._gen[period]
        f = self.root / "prices_gen" / f"{period.strftime('%Y%m')}.parquet"
        df = read_locked(f, "delivery_hour", self.end, GEN_COLS) if f.exists() else None
        self._gen[period] = df
        while len(self._gen) > self._gen_n:
            self._gen.popitem(last=False)
        return df

    def gen_window(self, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
        hi = min(pd.Timestamp(hi), self.end_ts)
        if lo >= hi:
            return pd.DataFrame()
        last = (hi - pd.Timedelta(microseconds=1)).tz_convert(TZ)
        months = pd.period_range(pd.Timestamp(lo.tz_convert(TZ).date()).to_period("M"),
                                 pd.Timestamp(last.date()).to_period("M"), freq="M")
        parts = []
        for p in months:
            df = self._gen_month(p)
            if df is not None and len(df):
                parts.append(df[(df["delivery_hour"] >= lo) & (df["delivery_hour"] < hi)])
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------- per-day features

def _local(ts: pd.Series) -> tuple[pd.Series, pd.Series]:
    loc = ts.dt.tz_convert(TZ)
    return loc.dt.tz_localize(None).dt.normalize(), loc.dt.hour


def newest_vintage(lf: pd.DataFrame, target_day: pd.Timestamp) -> pd.DataFrame:
    """The bid_inputs rule: for each (zone, target hour) of the day, the row published last."""
    ld, _ = _local(lf["target_hour"])
    x = lf[ld == target_day]
    return x.sort_values("published_at", kind="stable").drop_duplicates(["zone", "target_hour"], keep="last")


def _calendar(d1: pd.Timestamp) -> dict:
    from pandas.tseries.holiday import USFederalHolidayCalendar
    hol = USFederalHolidayCalendar().holidays(d1 - pd.Timedelta(days=1), d1 + pd.Timedelta(days=1))
    doy = d1.dayofyear
    return {"dow": d1.dayofweek, "month": d1.month, "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25), "holiday": int(d1 in hol),
            "weekend": int(d1.dayofweek >= 5)}


def day_features(D: dt.date, f: dict, skeleton: pd.DataFrame) -> pd.DataFrame:
    """Base features for the delivery rows of D+1 (skeleton: delivery_hour, zone), from frames f
    returned by features_available_at(05:00 on D)."""
    t = decision_time(D)
    d0 = pd.Timestamp(D)
    dm1, d1 = d0 - pd.Timedelta(days=1), d0 + pd.Timedelta(days=1)
    da = f["da_prices_zone"]
    rt = f["rt_prices_zone"]
    da = da[da["zone"].isin(ZONES)].copy()
    rt = rt[rt["zone"].isin(ZONES)].copy()
    da["ldate"], da["h"] = _local(da["delivery_hour"])
    g = da[["delivery_hour", "zone", "ldate", "h", "da_lbmp", "da_congestion"]].merge(
        rt[["delivery_hour", "zone", "rt_lbmp", "rt_congestion"]], on=["delivery_hour", "zone"], how="inner")
    g["gap"] = g["rt_lbmp"] - g["da_lbmp"]
    g["cgap"] = g["rt_congestion"] - g["da_congestion"]
    g["spike"] = (g["gap"] > SPIKE).astype(float)
    g["negspike"] = (g["gap"] < -SPIKE).astype(float)

    zh = pd.DataFrame(index=pd.MultiIndex.from_product([ZONES, range(24)], names=["zone", "h"]))
    x = da[da["ldate"] == d0].groupby(["zone", "h"])[["da_lbmp", "da_congestion", "da_loss", "da_energy"]].mean()
    zh = zh.join(x.set_axis(["da_d0_h", "da_d0_cong_h", "da_d0_loss_h", "da_d0_energy_h"], axis=1))
    zh = zh.join(da[da["ldate"] == dm1].groupby(["zone", "h"])["da_lbmp"].mean().rename("da_dm1_h"))
    w7 = (da["ldate"] >= d0 - pd.Timedelta(days=7)) & (da["ldate"] <= dm1)
    zh = zh.join(da[w7].groupby(["zone", "h"])["da_lbmp"].mean().rename("da_7d_h"))
    x = g[g["ldate"] == dm1].groupby(["zone", "h"])[["gap", "rt_lbmp", "cgap"]].mean()
    zh = zh.join(x.set_axis(["gap_dm1_h", "rt_dm1_h", "cgap_dm1_h"], axis=1))
    g7 = g[(g["ldate"] >= d0 - pd.Timedelta(days=7)) & (g["ldate"] <= dm1)]
    zh = zh.join(g7.groupby(["zone", "h"])[["gap", "cgap"]].mean().set_axis(["gap_7d_h", "cgap_7d_h"], axis=1))
    g28 = g[(g["ldate"] >= d0 - pd.Timedelta(days=28)) & (g["ldate"] <= dm1)]
    x = g28.groupby(["zone", "h"]).agg(gap_28d_h=("gap", "mean"), gap_28d_h_med=("gap", "median"),
                                       gap_28d_h_std=("gap", "std"), spike_28d_h=("spike", "mean"),
                                       negspike_28d_h=("negspike", "mean"))
    zh = zh.join(x)
    g365 = g[g["delivery_hour"] >= t - pd.Timedelta(days=365)]
    x = g365.groupby(["zone", "h"]).agg(gap_365d_h=("gap", "mean"), gap_365d_h_med=("gap", "median"),
                                        n_365d_h=("gap", "size"))
    zh = zh.join(x)

    z = pd.DataFrame(index=pd.Index(ZONES, name="zone"))
    gd = g[g["ldate"] == dm1].groupby("zone")["gap"]
    z = z.join(gd.mean().rename("gap_dm1_mean")).join(gd.size().rename("n_rt_dm1"))
    z["n_rt_dm1"] = z["n_rt_dm1"].fillna(0)
    z = z.join(g7.groupby("zone")["gap"].mean().rename("gap_7d_all"))
    z = z.join(g28.groupby("zone")["gap"].mean().rename("gap_28d_all"))
    z = z.join(g7.groupby("zone")["spike"].mean().rename("spike_7d_all"))
    x = g[g["ldate"] == d0].groupby("zone")[["gap", "rt_lbmp"]].mean()
    z = z.join(x.set_axis(["gap_d0_early", "rt_d0_early"], axis=1))
    x = da[da["ldate"] == d0].groupby("zone")["da_lbmp"].agg(["mean", "max"])
    z = z.join(x.set_axis(["da_d0_mean", "da_d0_max"], axis=1))
    if len(rt):
        last = rt.sort_values("delivery_hour", kind="stable").groupby("zone").tail(1).set_index("zone")
        z = z.join(last["rt_lbmp"].rename("rt_last"))
        hrs = (t - (last["delivery_hour"] + pd.Timedelta(hours=1))) / pd.Timedelta(hours=1)
        z = z.join(hrs.astype(float).rename("hours_since_rt"))
    else:
        z["rt_last"], z["hours_since_rt"] = np.nan, np.nan

    out = skeleton[["delivery_hour", "zone"]].copy()
    out["hour"] = out["delivery_hour"].dt.tz_convert(TZ).dt.hour
    out = out.join(zh, on=["zone", "hour"]).join(z, on="zone")

    # load forecast: newest vintage public at 05:00 for D+1, and for D (for the day-on-day change)
    lf = f["load_forecast"]
    v1 = newest_vintage(lf, d1)
    v0 = newest_vintage(lf, d0)
    ld1 = v1.set_index(["zone", "target_hour"])["load_forecast_mw"]
    out["lf_h"] = ld1.reindex(pd.MultiIndex.from_arrays([out["zone"], out["delivery_hour"]])).to_numpy()
    nyiso = v1[v1["zone"] == "NYISO"].set_index("target_hour")["load_forecast_mw"]
    out["lf_nyiso_h"] = nyiso.reindex(out["delivery_hour"]).to_numpy()
    out["lf_nyiso_peak"] = nyiso.max() if len(nyiso) else np.nan
    out["lf_peak"] = out.groupby("zone")["lf_h"].transform("max")
    out["lf_h_rel"] = out["lf_h"] / out["lf_peak"]
    _, h0 = _local(v0["target_hour"]) if len(v0) else (None, pd.Series(dtype=int))
    l0 = v0.assign(h=h0).groupby(["zone", "h"])["load_forecast_mw"].mean() if len(v0) else pd.Series(dtype=float)
    out["lf_d1_minus_d0_h"] = out["lf_h"] - l0.reindex(pd.MultiIndex.from_arrays([out["zone"], out["hour"]])).to_numpy()
    if len(v1):
        lead = (d1 - pd.to_datetime(v1["issue_date"]).max()).days
        ld1_all = newest_vintage(lf, d1)
        prev = lf[(_local(lf["target_hour"])[0] == d1) & (pd.to_datetime(lf["issue_date"]) < pd.to_datetime(ld1_all["issue_date"]).max())]
        prev = prev.sort_values("published_at", kind="stable").drop_duplicates(["zone", "target_hour"], keep="last")
        pv = prev.set_index(["zone", "target_hour"])["load_forecast_mw"]
        out["lf_rev_h"] = out["lf_h"] - pv.reindex(pd.MultiIndex.from_arrays([out["zone"], out["delivery_hour"]])).to_numpy()
    else:
        lead = np.nan
        out["lf_rev_h"] = np.nan
    out["lf_lead"] = lead

    for k, v in _calendar(d1).items():
        out[k] = v
    out["zone_code"] = out["zone"].map(ZONE_CODE).astype(int)
    return out


def day_gen_features(D: dt.date, f: dict, skeleton: pd.DataFrame) -> pd.DataFrame:
    """Zone-plus-generator summary (secondary ablation): border proxies by name and system-wide
    distributions over generator points, per clock hour, from frames with include_gen=True."""
    d0 = pd.Timestamp(D)
    dm1 = d0 - pd.Timedelta(days=1)
    h = pd.DataFrame(index=pd.Index(range(24), name="h"))
    gda, grt = f.get("da_prices_gen"), f.get("rt_prices_gen")
    if gda is None or not len(gda):
        out = skeleton[["delivery_hour"]].copy()
        for c in GEN_FEATURES:
            out[c] = np.nan
        return out.drop(columns="delivery_hour")
    gda = gda.copy()
    gda["ldate"], gda["h"] = _local(gda["delivery_hour"])
    gg = gda[["delivery_hour", "ptid", "name", "point_type", "ldate", "h", "da_lbmp", "da_congestion"]].merge(
        grt[["delivery_hour", "ptid", "rt_lbmp", "rt_congestion"]], on=["delivery_hour", "ptid"], how="inner")
    gg["gap"] = gg["rt_lbmp"] - gg["da_lbmp"]
    ext0 = gda[(gda["point_type"] == "external") & (gda["ldate"] == d0)]
    ext1 = gg[(gg["point_type"] == "external") & (gg["ldate"] == dm1)]
    for n in EXTERNAL:
        k = n.replace(" ", "")
        h = h.join(ext0[ext0["name"] == n].groupby("h")["da_lbmp"].mean().rename(f"ext_da_d0_h_{k}"))
        h = h.join(ext1[ext1["name"] == n].groupby("h")["gap"].mean().rename(f"ext_gap_dm1_h_{k}"))
    gen0 = gda[(gda["point_type"] == "gen") & (gda["ldate"] == d0)].groupby("h")["da_congestion"]
    h = h.join(gen0.quantile(0.05).rename("gen_dacong_d0_h_p05")).join(gen0.median().rename("gen_dacong_d0_h_p50"))
    h = h.join(gen0.quantile(0.95).rename("gen_dacong_d0_h_p95")).join(gen0.std().rename("gen_dacong_d0_h_std"))
    h = h.join(gen0.size().rename("n_gen_points"))
    gen1 = gg[(gg["point_type"] == "gen") & (gg["ldate"] == dm1)]
    q = gen1.groupby("h")["gap"]
    h = h.join(q.quantile(0.05).rename("gen_gap_dm1_h_p05")).join(q.median().rename("gen_gap_dm1_h_p50"))
    h = h.join(q.quantile(0.95).rename("gen_gap_dm1_h_p95"))
    h = h.join(gen1.assign(a=gen1["rt_congestion"].abs()).groupby("h")["a"].mean().rename("gen_rtcong_dm1_h_absmean"))
    hour = skeleton["delivery_hour"].dt.tz_convert(TZ).dt.hour
    out = h.reindex(hour.to_numpy())
    out.index = skeleton.index
    return out[GEN_FEATURES]


# ---------------------------------------------------------------- the panel

_STORE: LockedStore | None = None


def _labels(store: T.Store, start: dt.date, end: dt.date) -> pd.DataFrame:
    pz = store.table("prices_zone")
    lo, hi = pd.Timestamp(start, tz=TZ), pd.Timestamp(end + dt.timedelta(days=1), tz=TZ)
    x = pz[(pz["delivery_hour"] >= lo) & (pz["delivery_hour"] < hi) & pz["zone"].isin(ZONES)]
    x = x[["delivery_hour", "zone", "da_lbmp", "rt_lbmp", "published_at"]].rename(
        columns={"da_lbmp": "y_da_lbmp", "rt_lbmp": "y_rt_lbmp", "published_at": "label_published_at"})
    x = x.copy()
    x["delivery_date"], _ = _local(x["delivery_hour"])
    x["gap"] = x["y_rt_lbmp"] - x["y_da_lbmp"]
    return x


def _build_days(args) -> list[pd.DataFrame]:
    days, include_gen = args
    store = _STORE
    lab = _LABELS
    out = []
    for d1 in days:
        D = (d1 - pd.Timedelta(days=1)).date()
        t = decision_time(D)
        f = T.features_available_at(t, store, lookback_days=LOOKBACK_DAYS, include_gen=include_gen,
                                    gen_lookback_days=2)
        T.assert_no_lookahead(f, t)                       # every input row published by 05:00 on D
        sk = lab.get(d1)
        if sk is None or not len(sk):
            continue
        x = day_features(D, f, sk)
        if include_gen:
            x = pd.concat([x, day_gen_features(D, f, sk)], axis=1)
        x["bid_date"] = pd.Timestamp(D)
        x["delivery_date"] = d1
        x = x.join(sk[[c for c in LABEL_COLS]])
        out.append(x)
    return out


_LABELS: dict = {}


def build_panel(start: dt.date = BUILD_START, end: dt.date = BUILD_END, store: LockedStore | None = None,
                include_gen: bool = False, workers: int = 1) -> pd.DataFrame:
    """Panel rows for delivery dates start..end (inclusive). Raises before reading if end is in the
    holdout and the study is locked."""
    global _STORE, _LABELS
    lock.assert_build_only([start, end])
    _STORE = store or LockedStore()
    lab = _labels(_STORE, start, end)
    _LABELS = {d: g for d, g in lab.groupby("delivery_date")}
    days = sorted(_LABELS)
    if include_gen or workers <= 1:
        chunks = [days]
        parts = _build_days((days, include_gen))     # generator months are cached in order: one process
    else:
        _STORE.table("load_forecast"), _STORE.table("outages"), _STORE.table("weather_gfs")
        n = max(1, len(days) // (workers * 4))
        chunks = [days[i:i + n] for i in range(0, len(days), n)]
        import multiprocessing as mp
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
            parts = [p for res in ex.map(_build_days, [(c, include_gen) for c in chunks]) for p in res]
    panel = pd.concat(parts, ignore_index=True)
    feats = BASE_FEATURES + (GEN_FEATURES if include_gen else [])
    cols = list(dict.fromkeys(KEY_COLS + LABEL_COLS + feats))          # "hour" is a key and a feature
    for c in feats:
        if c not in INT_FEATURES:
            panel[c] = panel[c].astype("float64")                       # one dtype whatever the source stores
    panel = panel[cols].sort_values(["delivery_hour", "zone"], kind="stable")
    panel = panel.reset_index(drop=True)
    lock.assert_build_only(panel["delivery_hour"])
    return panel


def build_gen(start: dt.date, end: dt.date, store: LockedStore | None = None, log=None) -> pd.DataFrame:
    """Only the zone-plus-generator summary columns (GEN_FEATURES) for delivery dates start..end,
    keyed by delivery_hour and zone, from features_available_at(include_gen=True) per bid day."""
    lock.assert_build_only([start, end])
    store = store or LockedStore()
    lab = _labels(store, start, end)
    parts = []
    for i, (d1, sk) in enumerate(lab.groupby("delivery_date")):
        D = (d1 - pd.Timedelta(days=1)).date()
        t = decision_time(D)
        f = T.features_available_at(t, store, lookback_days=3, include_gen=True, gen_lookback_days=2)
        T.assert_no_lookahead(f, t)
        x = day_gen_features(D, f, sk)
        x["delivery_hour"], x["zone"] = sk["delivery_hour"], sk["zone"]
        parts.append(x)
        if log and i % 100 == 0:
            log(f"  gen features {d1.date()}")
    out = pd.concat(parts, ignore_index=True)
    lock.assert_build_only(out["delivery_hour"])
    return out[["delivery_hour", "zone"] + GEN_FEATURES]


def save_panel(panel: pd.DataFrame, path: Path) -> None:
    lock.assert_build_only(panel["delivery_hour"])
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path)


def load_panel(path: Path, end: dt.date | None = None) -> pd.DataFrame:
    """Read a cached panel, keeping only delivery dates before the lock's read end."""
    end = lock.read_end(end)
    p = read_locked(path, "delivery_hour", end)
    return p


def bid_days(start: dt.date, end: dt.date) -> list[dt.date]:
    lock.assert_build_only([start, end])
    return [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
