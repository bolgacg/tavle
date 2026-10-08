"""V4: idea B (weather forecast against the operator's load forecast) run before 2021 on archived
GEFS reforecasts (never reanalysis). Same columns as v1 features_weather.py so the v1 B model reads them
unchanged; the operator side (lf_*) already sits in the v2 panel (LF_FEATURES).

Source: parquet_v2/weather_reforecast.parquet (data agent). Columns read (REFC map below):
    init_time (tz-aware UTC), valid_time (tz-aware), zone, temperature_2m_c, available_at (tz-aware).
A forecast counts for bid day D only if available_at <= 05:00 New York on D; of those, the newest
init wins (normally 00Z on D, i.e. 19:00 or 20:00 New York on D-1), and every valid hour of D+1 must
come from that one init, as an operational trader would have it.

Columns:
    wx_temp_f, wx_lead_hours                 forecast temperature for the hour, the lead it came from
    wx_d1_max_f/min_f/mean_f/hdh65/cdh65     daily values over D+1 (>= 20 hours, else NaN)
    wx_temp_f_wow, wx_d1_*_wow               minus the same feature known at 05:00 on D-7
    wx_lf_contrast                           weather week change (F) minus the operator's peak-forecast week
                                             change (%): weather says demand moves, operator not
                                             (only when panel is passed; lf_peak from the panel)
Rows with no reforecast stay NaN; the runner scores V4 only on quarters whose test rows have weather.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ideas_common as C

NAME = "V4"
REFC = dict(table="weather_reforecast", init="init_utc", valid="target_hour", zone="zone",
            temp="temperature_2m_c", avail="published_at")
# data agent schema (7 Oct): GEFS v12 reforecast control member c00, 00 UTC run of D, leads 27-54 h
# 3-hourly, 12 points (v1 points.py; `primary` marks each zone's point), 2010-2019,
# published_at = init + 8 h (03:00 EST / 04:00 EDT on D). Hourly values are interpolated here.
BASE_F = 65.0
MIN_HOURS = 20
COLUMNS = ["wx_temp_f", "wx_lead_hours", "wx_d1_max_f", "wx_d1_min_f", "wx_d1_mean_f", "wx_d1_hdh65",
           "wx_d1_cdh65", "wx_temp_f_wow", "wx_d1_max_f_wow", "wx_d1_min_f_wow", "wx_d1_mean_f_wow",
           "wx_d1_hdh65_wow", "wx_d1_cdh65_wow", "wx_lf_contrast"]
ROWS = {"V4_B_reforecast": dict(features="base+V4", rule="two_sided", note="v1 B before 2021")}


def load(store=None, table: str | None = None) -> pd.DataFrame:
    table = table or REFC["table"]                    # V16 passes weather_gefs_joined; same columns
    if store is not None and table in store:
        df = store[table]
    else:
        df = C.read_table(table, REFC["valid"])
    check_forecast_archive(df)
    if "primary" in df:
        df = df[df["primary"].astype(bool)]
    df = df.rename(columns={REFC[k]: k for k in ("init", "valid", "zone", "temp", "avail")})
    return hourly(df[["init", "valid", "zone", "temp", "avail"]])


FORBIDDEN = ("reanalysis", "era5", "era-5", "merra", "observ", "metar", "isd", "analysis_only", "actual")


def check_forecast_archive(df: pd.DataFrame) -> None:
    """Binding rule (7 Oct): archived forecasts only. Refuse a weather table without an issue time, a
    lead and a `source` column that names a forecast archive (and never reanalysis or observations)."""
    need = [REFC["init"], "lead_hours", "source"]
    miss = [c for c in need if c not in df]
    if miss:
        raise ValueError(f"V4 refuses the weather table: missing {miss} (issue time, lead, forecast source)")
    src = df["source"].astype(str).str.lower().unique()
    bad = [s for s in src if any(f in s for f in FORBIDDEN) or "forecast" not in s and "gefs" not in s
           and "gfs" not in s]
    if bad:
        raise ValueError(f"V4 refuses the weather table: source {bad} is not a forecast archive")
    if df[REFC["init"]].isna().any() or df["lead_hours"].isna().any():
        raise ValueError("V4 refuses the weather table: rows without issue time or lead")


def hourly(df: pd.DataFrame) -> pd.DataFrame:
    """3-hourly to hourly by linear interpolation in time, within one (init, zone) run only."""
    out = []
    for (init, zone), g in df.groupby(["init", "zone"], sort=False):
        g = g.sort_values("valid").set_index("valid")
        idx = pd.date_range(g.index.min(), g.index.max(), freq="h")
        t = g["temp"].reindex(g.index.union(idx)).interpolate(method="time").reindex(idx)
        out.append(pd.DataFrame({"init": init, "valid": idx, "zone": zone, "temp": t.to_numpy(),
                                 "avail": g["avail"].max()}))
    return pd.concat(out, ignore_index=True) if out else df


def per_bid_day(wx: pd.DataFrame, bid_days) -> pd.DataFrame:
    """For each bid day D: the newest init public by 05:00 on D, its hours on D+1 (wall clock)."""
    wx = wx.copy()
    wx["avail"] = pd.to_datetime(wx["avail"]).dt.tz_convert(C.TZ)
    wx["valid"] = pd.to_datetime(wx["valid"]).dt.tz_convert(C.TZ)
    wx["wall"] = wx["valid"].dt.tz_localize(None).astype("datetime64[ns]")
    inits = wx.groupby("init")["avail"].max().reset_index()
    # first bid day whose 05:00 deadline is at or after the init's publication
    inits["first"] = (inits["avail"] - pd.Timedelta(hours=C.DECISION_HOUR)).dt.tz_localize(None) \
        .dt.ceil("D").astype("datetime64[ns]")
    inits = inits.sort_values(["first", "avail"])
    days = pd.DataFrame({"bid_date": pd.DatetimeIndex(sorted(set(bid_days))).astype("datetime64[ns]")})
    pick = pd.merge_asof(days, inits[["first", "init"]], left_on="bid_date", right_on="first", direction="backward")
    pick = pick.dropna(subset=["init"])
    f = wx.merge(pick[["bid_date", "init"]], on="init", how="inner")
    f = f[f["wall"].dt.normalize() == f["bid_date"] + pd.Timedelta(days=1)]
    lead = f["valid"].dt.tz_convert("UTC") - pd.to_datetime(f["init"], utc=True)
    f = f.assign(whour=f["wall"].dt.hour, wx_temp_f=f["temp"].astype(float) * 9 / 5 + 32,
                 wx_lead_hours=lead.dt.total_seconds() / 3600)
    return f.groupby(["bid_date", "zone", "whour"], as_index=False).agg(
        wx_temp_f=("wx_temp_f", "mean"), wx_lead_hours=("wx_lead_hours", "max"), avail=("avail", "max"))


def daily(h: pd.DataFrame) -> pd.DataFrame:
    t = h["wx_temp_f"]
    g = h.assign(hdh=(BASE_F - t).clip(lower=0), cdh=(t - BASE_F).clip(lower=0)).groupby(["bid_date", "zone"])
    d = pd.DataFrame({"wx_d1_max_f": g["wx_temp_f"].max(), "wx_d1_min_f": g["wx_temp_f"].min(),
                      "wx_d1_mean_f": g["wx_temp_f"].mean(), "wx_d1_hdh65": 24 * g["hdh"].mean(),
                      "wx_d1_cdh65": 24 * g["cdh"].mean()})
    return d.where(g["wx_temp_f"].count() >= MIN_HOURS, axis=0)


def features(panel_index: pd.MultiIndex, panel: pd.DataFrame | None = None, store=None,
             table: str | None = None) -> pd.DataFrame:
    p = C.panel_frame(panel_index)
    days = set(p["bid_date"])
    days |= {d - pd.Timedelta(days=7) for d in days}
    h = per_bid_day(load(store, table), days)
    # audit: every value used must be public at 05:00 on its bid day
    if len(h):
        late = pd.to_datetime(h["avail"]).dt.tz_convert(C.TZ) > h["bid_date"].map(C.decision_time)
        assert not late.any(), "V4 lookahead: reforecast used before its available_at"
    d = daily(h) if len(h) else pd.DataFrame(columns=COLUMNS[2:7])
    wk_h = h.assign(bid_date=h["bid_date"] + pd.Timedelta(days=7))[["bid_date", "zone", "whour", "wx_temp_f"]]
    wk_d = d.reset_index().assign(bid_date=lambda x: x["bid_date"] + pd.Timedelta(days=7)).set_index(["bid_date", "zone"]) \
        if len(d) else d
    out = p.merge(h.drop(columns="avail"), on=["bid_date", "zone", "whour"], how="left")
    out = out.merge(wk_h.rename(columns={"wx_temp_f": "wk"}), on=["bid_date", "zone", "whour"], how="left")
    out["wx_temp_f_wow"] = out["wx_temp_f"] - out["wk"]
    dd = d.copy()
    for c in list(d.columns):
        dd[c + "_wow"] = d[c] - wk_d[c].reindex(d.index) if len(d) else np.nan
    out = out.merge(dd.reset_index(), on=["bid_date", "zone"], how="left") if len(dd) else out
    out["wx_lf_contrast"] = np.nan
    if panel is not None and "lf_peak" in panel:
        # operator side, week change in percent, beside the weather week change: no scaling fitted on
        # the panel (a z-score over the whole panel would read the test quarter's distribution)
        lf = pd.DataFrame({"bid_date": p["bid_date"].to_numpy(), "zone": p["zone"].to_numpy(),
                           "lf_peak": panel["lf_peak"].to_numpy(float)})
        lfp = lf.groupby(["bid_date", "zone"])["lf_peak"].first()
        wk = lfp.copy()
        wk.index = pd.MultiIndex.from_arrays([wk.index.get_level_values(0) + pd.Timedelta(days=7),
                                              wk.index.get_level_values(1)], names=wk.index.names)
        pct = (100 * (lfp - wk.reindex(lfp.index)) / wk.reindex(lfp.index)).rename("lfpct").reset_index()
        out = out.merge(pct, on=["bid_date", "zone"], how="left")
        out["wx_lf_contrast"] = out.get("wx_d1_mean_f_wow", np.nan) - out["lfpct"]
    for c in COLUMNS:
        if c not in out:
            out[c] = np.nan
    res = out[COLUMNS].astype(float)
    res.index = panel_index
    return res


rule = C.two_sided
