"""Independent re-implementation for the audit of side/storm_value.py (storm-day filter and flip).

Written from the description of the strategy, not from the script's code. Build years only.
HOLDOUT: every table is filtered AT LOAD (pyarrow filter) to delivery or target hours before
2024-01-01 New York time, and every loader asserts it. Nothing here reads, computes or prints
anything for later dates.

Runs on gene: ~/nyiso-us/side_audit/ with ~/nyiso-us/.venv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

HOME = Path.home() / "nyiso-us"
PQ = HOME / "parquet"
OUT = HOME / "side_audit"
TZ = "America/New_York"
END = pd.Timestamp("2024-01-01", tz=TZ)
ZONES = ["CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL", "LONGIL",
         "MHK VL", "MILLWD", "N.Y.C.", "NORTH", "WEST"]

# Per cleared MWh, from model/fees.py (cost audit): RS1 + FERC on both sides, supply adds the uplift bound.
RS1 = {2020: 0.0862, 2021: 0.0757, 2022: 0.0853, 2023: 0.1066}
FERC = {2020: 0.010, 2021: 0.015, 2022: 0.016, 2023: 0.017}
UPLIFT = {2020: 0.007, 2021: 0.004, 2022: 0.003, 2023: 0.026}
LOAD_FEE = {y: RS1[y] + FERC[y] for y in RS1}
SUPPLY_FEE = {y: RS1[y] + FERC[y] + UPLIFT[y] for y in RS1}

BID_DAYS = pd.date_range("2020-01-02", "2023-12-30", freq="D")   # D; delivery D+1 (same span as the script)
SCRIPT_CUTS = [0.80, 0.90, 0.95, 0.98]
WIDE_CUTS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]
YEARS = [2021, 2022, 2023]


# ---------------------------------------------------------------- loading (holdout enforced here)
def _read(name: str, timecol: str, columns=None, extra=None) -> pd.DataFrame:
    d = ds.dataset(PQ / f"{name}.parquet", format="parquet")
    flt = ds.field(timecol) < pa.scalar(END, type=d.schema.field(timecol).type)
    if extra is not None:
        flt = flt & extra
    df = d.to_table(filter=flt, columns=columns).to_pandas()
    assert len(df) > 0 and df[timecol].max() < END, f"{name}: holdout filter failed"
    return df


def load_prices() -> pd.DataFrame:
    cols = ["delivery_hour", "zone", "da_lbmp", "rt_lbmp", "da_published_at", "rt_published_at",
            "rt_revised", "rt_file_written_at"]
    px = _read("prices_zone", "delivery_hour", cols, ds.field("zone").isin(ZONES))
    px["ddate"] = px["delivery_hour"].dt.tz_localize(None).dt.normalize()    # local delivery date
    px["gap"] = px["rt_lbmp"] - px["da_lbmp"]                                   # RT minus DA
    assert px["ddate"].max() < pd.Timestamp("2024-01-01")
    return px


def load_lf() -> pd.DataFrame:
    lf = _read("load_forecast", "target_hour", None, ds.field("zone") == "NYISO")
    lf["issue_date"] = pd.to_datetime(lf["issue_date"])
    return lf


def load_wx() -> pd.DataFrame:
    return _read("weather_gfs", "target_hour", None)


def assert_holdout(*frames):
    for f, col in frames:
        assert f[col].max() < END, f"holdout breach in {col}"


# ---------------------------------------------------------------- deadline
def deadlines(bid_days, mode: str = "wallclock") -> pd.Series:
    """05:00 New York time on D. 'wallclock' = the true 05:00 local; 'script' = local midnight plus
    5 elapsed hours (what storm_value.py does), which is 06:00 EDT on spring-forward days and
    04:00 EST on fall-back days."""
    naive = pd.DatetimeIndex(bid_days)
    if mode == "wallclock":
        t = (naive + pd.Timedelta(hours=5)).tz_localize(TZ)
    elif mode == "script":
        t = naive.tz_localize(TZ) + pd.Timedelta(hours=5)
    else:
        raise ValueError(mode)
    return pd.Series(t, index=naive)


# ---------------------------------------------------------------- the four day-level inputs
def _bid_day_of_target(ts: pd.Series) -> pd.Series:
    """Bid day D whose delivery day D+1 contains this (tz-aware) target hour."""
    return ts.dt.tz_localize(None).dt.normalize() - pd.Timedelta(days=1)


def used_rows(bid_days, px, lf, wx, deadline="wallclock", gfs="all", lfsel="max_issue"):
    """The rows each input uses, with their bid day D, for inspection and for the inputs."""
    bid_days = pd.DatetimeIndex(bid_days)
    t05 = deadlines(bid_days, deadline)

    l = lf.assign(D=_bid_day_of_target(lf["target_hour"]))
    l = l[l["D"].isin(bid_days)]
    l = l.assign(t05=l["D"].map(t05))
    l = l[l["published_at"].notna() & (l["published_at"] <= l["t05"])]
    if lfsel == "max_issue":            # newest vintage that is public, as one block
        l = l[l["issue_date"] == l.groupby("D")["issue_date"].transform("max")]
    elif lfsel == "per_hour":           # newest public value for every target hour (pipeline bid_inputs)
        l = l.sort_values("published_at", kind="stable").drop_duplicates(["D", "target_hour"], keep="last")
    else:
        raise ValueError(lfsel)

    w = wx[wx["primary"] & wx["temperature_2m_c"].notna()]
    w = w.assign(D=_bid_day_of_target(w["target_hour"]))
    w = w[w["D"].isin(bid_days)]
    w = w.assign(t05=w["D"].map(t05))
    w = w[w["published_at"] <= w["t05"]]
    if gfs == "rule":                   # three-day rule: day2 up to 21:00, day3 for 22:00 and 23:00 and gaps
        hr = w["target_hour"].dt.hour
        w = w[((w["run_lead_hours"] == 48) & (hr <= 21)) | (w["run_lead_hours"] == 72)]
        w = w.sort_values(["point", "target_hour", "run_lead_hours"], kind="stable")
        w = w.drop_duplicates(["point", "target_hour"], keep="first")
    elif gfs != "all":                  # 'all' = every public row of either lead (what the script does)
        raise ValueError(gfs)

    r = px[px["ddate"].isin(bid_days)]
    r = r.assign(D=r["ddate"], t05=r["ddate"].map(t05))
    r = r[r["rt_published_at"].notna() & (r["rt_published_at"] <= r["t05"])]
    return l, w, r


def inputs(bid_days, px, lf, wx, **kw) -> pd.DataFrame:
    l, w, r = used_rows(bid_days, px, lf, wx, **kw)
    out = pd.DataFrame(index=pd.DatetimeIndex(bid_days))
    out["peak"] = l.groupby("D")["load_forecast_mw"].max()
    out["tmin"] = w.groupby("D")["temperature_2m_c"].min()
    out["tmax"] = w.groupby("D")["temperature_2m_c"].max()
    out["rt"] = r.groupby("D")["gap"].mean()
    return out


# ---------------------------------------------------------------- storm score
def trailing_rank(x: np.ndarray, window: int = 365, min_obs: int = 60) -> np.ndarray:
    """Share of the previous `window` days (strictly earlier, NaN skipped) below today's value."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    pad = np.concatenate([np.full(window, np.nan), x])
    win = np.lib.stride_tricks.sliding_window_view(pad, window)[:n]      # row i = x[i-window : i]
    cnt = (~np.isnan(win)).sum(1)
    with np.errstate(invalid="ignore"):
        less = (win < x[:, None]).sum(1)
    out = np.full(n, np.nan)
    ok = (cnt >= min_obs) & ~np.isnan(x)
    out[ok] = less[ok] / cnt[ok]
    return out


COMPONENTS = {"load": lambda d: d["peak"], "cold": lambda d: -d["tmin"],
              "heat": lambda d: d["tmax"], "rt": lambda d: d["rt"]}


def storm_score(days: pd.DataFrame, comps=("load", "cold", "heat", "rt"), window: int = 365):
    """Returns (storm indexed by DELIVERY date D+1, component ranks indexed by bid day D)."""
    days = days.reindex(pd.date_range(days.index.min(), days.index.max(), freq="D"))   # positions = days
    ranks = pd.DataFrame({c: trailing_rank(COMPONENTS[c](days).to_numpy(float), window) for c in comps},
                         index=days.index)
    s = ranks.max(axis=1, skipna=True)
    s.index = s.index + pd.Timedelta(days=1)
    return s, ranks


# ---------------------------------------------------------------- book and strategies
def daily_book(px: pd.DataFrame) -> pd.DataFrame:
    """Per delivery date: sum of RT minus DA over the 11 zones x hours, rows with a price, and the
    daily net of 1 MW supply / 1 MW load in every zone-hour at base cost."""
    g = px.groupby("ddate")
    b = pd.DataFrame({"sum_gap": g["gap"].sum(), "n": g["gap"].count(), "rows": g["gap"].size()})
    yr = b.index.year
    b["sup"] = -b["sum_gap"] - b["n"] * yr.map(SUPPLY_FEE).to_numpy()
    b["load"] = b["sum_gap"] - b["n"] * yr.map(LOAD_FEE).to_numpy()
    return b


def legs(book: pd.DataFrame, cost=None):
    if cost is None:
        return book["sup"], book["load"]
    return -book["sum_gap"] - book["n"] * cost, book["sum_gap"] - book["n"] * cost


def strategy(book, storm, cut, kind, cost=None) -> pd.Series:
    sup, load = legs(book, cost)
    on = (storm.reindex(book.index) > cut).to_numpy()          # NaN > cut is False: trade supply
    if kind == "supply":
        return sup
    if kind == "filter":
        return sup.where(~on, 0.0)
    if kind == "flip":
        return sup.where(~on, load)
    raise ValueError(kind)


def pick_cut(book, storm, kind, year, cuts=SCRIPT_CUTS):
    prior = book.index.year == year - 1
    return max(cuts, key=lambda c: strategy(book, storm, c, kind)[prior].sum())


def year_stats(daily: pd.Series) -> dict:
    s = daily.sort_values()
    return {"net": round(float(daily.sum())), "best_day": round(float(daily.max())),
            "best_day_date": str(daily.idxmax().date()), "worst_day": round(float(daily.min())),
            "no_best1": round(float(s.iloc[:-1].sum())), "no_best3": round(float(s.iloc[:-3].sum())),
            "no_best10": round(float(s.iloc[:-10].sum()))}


def run_protocol(book, storm, kind, cuts=SCRIPT_CUTS, cost=None) -> dict:
    """The script's protocol: cut chosen on the prior year's net (base cost), applied to the year."""
    out = {}
    for y in YEARS:
        c = pick_cut(book, storm, kind, y, cuts)
        d = strategy(book, storm, c, kind, cost)
        d = d[d.index.year == y]
        on = (storm.reindex(d.index) > c)
        out[y] = {"cut": c, "days_on": int(on.sum()), **year_stats(d)}
    out["sum"] = sum(out[y]["net"] for y in YEARS)
    return out
