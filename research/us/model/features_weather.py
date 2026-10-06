"""Idea B features: GFS weather forecasts next to the operator's own load forecast.

    features(panel_index) -> DataFrame aligned to the panel index

Panel row = (bid_date D, zone, delivery_hour h of D+1), decided at 05:00 New York time on D.
Every input comes from timing.bid_inputs(D), i.e. timing.features_available_at(05:00 on D), and the
rows actually used are passed through timing.assert_no_lookahead before any feature is computed.

Weather (GFS 2 m temperature at the zone's primary point, three-day rule: day2 run for hours up to
21:00, day3 run for 22:00 and 23:00; see pipeline/timing.gfs_rule):
    wx_temp_f              forecast temperature for the hour, Fahrenheit
    wx_run_lead_hours      48 (day2) or 72 (day3), which run the value came from
    wx_d1_max_f/min_f/mean_f   daily max, min and mean over the hours of D+1
    wx_d1_hdh65, wx_d1_cdh65   heating and cooling degree hours, base 65 F, scaled to a 24-hour day
                           (24 x mean over the day's hours, so 23- and 25-hour DST days compare)
Operator (NYISO isolf, the newest vintage public at 05:00 on D, normally the file named D):
    lf_mw                  forecast load for the zone-hour, MW
    lf_d1_peak_mw/min_mw/mean_mw   daily peak, minimum and mean of that forecast over D+1
    lf_vintage_age_days    D minus the vintage's issue date (0 normally)
Week-over-week ("_wow"): the same feature minus its value for the same weekday last week (D+1-7),
as it was known at 05:00 on D-7 under the same rules, so weather and operator changes are compared
like for like. Hourly changes match the same local wall-clock hour. "_wow_pct" = percent change.
The contrast between them (weather says demand moves, the operator's forecast moves less or more)
is left to the model: the pairs wx_*_wow and lf_*_wow are the inputs for it.

Daily features need at least MIN_HOURS hours of D+1, else NaN. GFS values exist from 25 March 2021,
so weather columns are NaN before that (idea B's building window starts there).
No price is read here. The default store reads through the holdout lock (lock.py), and features()
refuses panel rows delivered on or after 2024-01-01 while the study is locked.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lock  # noqa: E402  (holdout lock: no row for a delivery date on or after 2024-01-01 while locked)
import timing as T  # noqa: E402
from common import PARQUET, TZ, ZONES, decision_time  # noqa: E402
from points import POINTS  # noqa: E402

PRIMARY_POINT = {zone: name for name, (zone, _lat, _lon, primary) in POINTS.items() if primary}
BASE_F = 65.0
MIN_HOURS = 20
WEEK = pd.Timedelta(days=7)
USED = ("weather_d1", "load_forecast_d1")

HOURLY = ["wx_temp_f", "wx_run_lead_hours", "wx_temp_f_wow", "lf_mw", "lf_mw_wow", "lf_mw_wow_pct"]
WX_DAILY = ["wx_d1_max_f", "wx_d1_min_f", "wx_d1_mean_f", "wx_d1_hdh65", "wx_d1_cdh65"]
LF_DAILY = ["lf_d1_peak_mw", "lf_d1_min_mw", "lf_d1_mean_mw"]
COLUMNS = (HOURLY + WX_DAILY + [c + "_wow" for c in WX_DAILY] + LF_DAILY
           + [c.replace("_mw", "_wow_pct") for c in LF_DAILY] + ["lf_vintage_age_days"])


# ------------------------------------------------------------------------------- panel index
def utc_ns(ts: pd.Series) -> pd.Series:
    """tz-aware instants as naive UTC datetime64[ns]: a merge key immune to unit and zone mismatches."""
    return ts.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")


def wall_clock(ts: pd.Series) -> pd.Series:
    """Naive local wall-clock time, used to match the same hour a week earlier across DST."""
    return ts.dt.tz_convert(TZ).dt.tz_localize(None).astype("datetime64[ns]")


def panel_frame(panel_index: pd.MultiIndex) -> pd.DataFrame:
    """Columns bid_date (naive midnight, datetime64[ns]), zone, delivery_hour (tz-aware New York),
    one row per index entry in order. Levels are found by name (bid_date, zone, delivery_hour),
    else by position 0, 1, 2."""
    names = list(panel_index.names)

    def level(cands, pos):
        for c in cands:
            if c in names:
                return pd.Series(panel_index.get_level_values(names.index(c)))
        return pd.Series(panel_index.get_level_values(pos))

    hour = level(("delivery_hour", "target_hour", "hour"), 2)
    if not isinstance(hour.dtype, pd.DatetimeTZDtype):
        raise TypeError("panel delivery_hour must be tz-aware timestamps (hour beginning, New York)")
    p = pd.DataFrame({"bid_date": pd.to_datetime(level(("bid_date", "bid_day", "D"), 0)).dt.normalize()
                                    .astype("datetime64[ns]"),
                      "zone": level(("zone",), 1).astype(str),
                      "delivery_hour": hour.dt.tz_convert(TZ)})
    if not (wall_clock(p["delivery_hour"]).dt.normalize() == p["bid_date"] + pd.Timedelta(days=1)).all():
        raise ValueError("every delivery_hour must fall on bid_date + 1 (New York time)")
    p["hour_utc"] = utc_ns(p["delivery_hour"])
    return p


# ------------------------------------------------------------------------------------ inputs
_STORES: dict = {}


def locked_store(needed: dict[str, str], end: dt.date | None = None) -> T.Store:
    """A timing.Store holding the `needed` tables ({name: time column}) read through the holdout
    lock (rows before lock.read_end(end), 2024-01-01 while locked; outage snapshots one day
    earlier, as panel.LockedStore), and every other source as an empty frame."""
    import pyarrow.parquet as pq
    from panel import read_locked
    end = lock.read_end(end)
    key = (tuple(sorted(needed)), end)
    if key not in _STORES:
        tables = {}
        for name in T.TABLES:
            path = PARQUET / f"{name}.parquet"
            if name in needed:
                e = end - dt.timedelta(days=1) if name == "outages" else end
                tables[name] = read_locked(path, needed[name], e)
            else:
                tables[name] = pq.read_schema(path).empty_table().to_pandas()
        _STORES[key] = T.Store(tables=tables)
    return _STORES[key]


def default_store(end: dt.date | None = None) -> T.Store:
    """The weather and load-forecast tables through the holdout lock; other sources empty."""
    return locked_store({"weather_gfs": "target_hour", "load_forecast": "target_hour"}, end)


def day_inputs(bid_day: dt.date, store: T.Store) -> dict[str, pd.DataFrame]:
    """The rows used for bid day D, checked against the 05:00 deadline (raises LookaheadError)."""
    x = T.bid_inputs(bid_day, store)
    used = {k: x[k] for k in USED}
    T.assert_no_lookahead(used, decision_time(bid_day))
    return used


def gather(bid_days, store: T.Store) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long frames over bid days, keyed by bid_date, zone, hour_utc (naive UTC) and wall (naive local):
    weather (wx_temp_f, wx_run_lead_hours) at each zone's primary point, and the operator's load
    forecast (lf_mw, issue_date)."""
    wx_parts, lf_parts = [], []
    for D in sorted(bid_days):
        used = day_inputs(D, store)
        bd = pd.Timestamp(D)
        w = used["weather_d1"]
        w = w[w["point"].isin(PRIMARY_POINT.values())]
        wx_parts.append(pd.DataFrame({
            "bid_date": bd, "zone": w["zone"].astype(str), "hour_utc": utc_ns(w["target_hour"]),
            "wall": wall_clock(w["target_hour"]),
            "wx_temp_f": w["temperature_2m_c"].astype(float) * 9 / 5 + 32,
            "wx_run_lead_hours": w["run_lead_hours"].astype(float)}))
        lf = used["load_forecast_d1"]
        lf = lf[lf["zone"].isin(ZONES)]
        lf_parts.append(pd.DataFrame({
            "bid_date": bd, "zone": lf["zone"].astype(str), "hour_utc": utc_ns(lf["target_hour"]),
            "wall": wall_clock(lf["target_hour"]), "lf_mw": lf["load_forecast_mw"].astype(float),
            "issue_date": pd.to_datetime(lf["issue_date"]).astype("datetime64[ns]")}))
    wx = pd.concat(wx_parts, ignore_index=True)
    lf = pd.concat(lf_parts, ignore_index=True)
    for f in (wx, lf):
        f["bid_date"] = f["bid_date"].astype("datetime64[ns]")
    return wx, lf


# ---------------------------------------------------------------------------------- features
def daily_weather(wx: pd.DataFrame) -> pd.DataFrame:
    t = wx["wx_temp_f"]
    g = wx.assign(hdh=(BASE_F - t).clip(lower=0), cdh=(t - BASE_F).clip(lower=0)).groupby(["bid_date", "zone"])
    d = pd.DataFrame({"wx_d1_max_f": g["wx_temp_f"].max(), "wx_d1_min_f": g["wx_temp_f"].min(),
                      "wx_d1_mean_f": g["wx_temp_f"].mean(), "wx_d1_hdh65": 24 * g["hdh"].mean(),
                      "wx_d1_cdh65": 24 * g["cdh"].mean()})
    return d.where(g["wx_temp_f"].count() >= MIN_HOURS, axis=0)


def daily_load(lf: pd.DataFrame) -> pd.DataFrame:
    g = lf.groupby(["bid_date", "zone"])
    d = pd.DataFrame({"lf_d1_peak_mw": g["lf_mw"].max(), "lf_d1_min_mw": g["lf_mw"].min(),
                      "lf_d1_mean_mw": g["lf_mw"].mean()})
    d = d.where(g["lf_mw"].count() >= MIN_HOURS, axis=0)
    bid = pd.Series(d.index.get_level_values("bid_date"), index=d.index)
    d["lf_vintage_age_days"] = (bid - g["issue_date"].max()).dt.days.astype(float)
    return d


def week_ago_daily(d: pd.DataFrame) -> pd.DataFrame:
    """Daily frame re-keyed so the row for bid day D-7 lines up with bid day D."""
    w = d.reset_index()
    w["bid_date"] = w["bid_date"] + WEEK
    return w.set_index(["bid_date", "zone"])


def week_ago_hourly(long: pd.DataFrame, col: str) -> pd.DataFrame:
    """Hourly values of bid day D-7, keyed by (bid_date D, zone, wall-clock hour one week later).
    The repeated 01:00 of a fall-back day is averaged; the missing 02:00 of a spring day stays NaN."""
    w = long[["bid_date", "zone", "wall", col]].copy()
    w["wall"] = w["wall"] + WEEK
    w["bid_date"] = w["bid_date"] + WEEK
    return w.groupby(["bid_date", "zone", "wall"])[col].mean().rename(col + "_wk").reset_index()


def features(panel_index: pd.MultiIndex, store: T.Store | None = None) -> pd.DataFrame:
    """Idea B feature columns (COLUMNS) for every panel row, index = panel_index, same order."""
    p = panel_frame(panel_index)
    lock.assert_build_only(p["delivery_hour"])
    store = store or default_store()
    days = set(p["bid_date"].dt.date)
    wx, lf = gather(days | {d - dt.timedelta(days=7) for d in days}, store)

    out = p.copy()
    out["wall"] = wall_clock(out["delivery_hour"])
    keys = ["bid_date", "zone", "hour_utc"]
    out = out.merge(wx.drop(columns="wall"), on=keys, how="left", validate="many_to_one")
    out = out.merge(lf.drop(columns=["wall", "issue_date"]), on=keys, how="left", validate="many_to_one")
    out = out.merge(week_ago_hourly(wx, "wx_temp_f"), on=["bid_date", "zone", "wall"], how="left")
    out = out.merge(week_ago_hourly(lf, "lf_mw"), on=["bid_date", "zone", "wall"], how="left")
    out["wx_temp_f_wow"] = out["wx_temp_f"] - out["wx_temp_f_wk"]
    out["lf_mw_wow"] = out["lf_mw"] - out["lf_mw_wk"]
    out["lf_mw_wow_pct"] = 100 * out["lf_mw_wow"] / out["lf_mw_wk"]

    dw, dl = daily_weather(wx), daily_load(lf)
    dw_wk, dl_wk = week_ago_daily(dw), week_ago_daily(dl)
    daily = dw.join(dl, how="outer")
    for c in WX_DAILY:
        daily[c + "_wow"] = dw[c] - dw_wk[c].reindex(dw.index)
    for c in LF_DAILY:
        wk = dl_wk[c].reindex(dl.index)
        daily[c.replace("_mw", "_wow_pct")] = 100 * (dl[c] - wk) / wk
    out = out.merge(daily.reset_index(), on=["bid_date", "zone"], how="left", validate="many_to_one")

    res = out[COLUMNS].astype(float)
    res.index = panel_index
    return res
