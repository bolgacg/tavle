"""Independent re-implementation for the audit of idea 10 (weather surprise) in side/lab.py.

Written from the registered description and the lab's one-line rule, not by calling lab.py:
for delivery day D+1, hours up to 21:00 New York time, at the 11 primary points, the GFS value
predicted 48 hours before the valid time (Open-Meteo previous_day2) against the value predicted
72 hours before (previous_day3), both filtered on published_at <= 05:00 New York clock time on D
(pipeline common.decision_time). Surprise toward the extreme = |T48 - 18.3| - |T72 - 18.3|, the
day's mean over point-hours with both values (at least 100), ranked against the previous 365 days
(at least 60 values, strictly earlier, NaN skipped). Virtual load in every zone-hour when the rank
is above a cut, else virtual supply. The cut for year Y is the best of (0.80, 0.90, 0.95, 0.98) on
year Y-1 at base cost, ties to the first; 0.95 when Y-1 has no rank.

HOLDOUT: every table is filtered AT LOAD to delivery or target times before 2024-01-01 New York
time and every loader asserts it. Nothing here reads, computes or prints anything for later dates.
Runs on gene: ~/nyiso-us/side_audit/ with ~/nyiso-us/.venv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

HOME = Path.home() / "nyiso-us"
sys.path.insert(0, str(HOME / "pipeline"))
sys.path.insert(0, str(HOME / "side_audit"))
import common                       # noqa: E402  decision_time = 05:00 New York clock time on D
import storm_audit_lib as S         # noqa: E402  price loader, daily book, fee tables (storm audit)

TZ = S.TZ
END = S.END
END_NAIVE = pd.Timestamp("2024-01-01")
PQ = S.PQ
COMFORT = 18.3
CUTS = (0.80, 0.90, 0.95, 0.98)
DEFAULT_CUT = 0.95
YEARS = [2021, 2022, 2023]
MIN_PAIRS = 100
DELIV = pd.date_range("2020-01-01", "2023-12-31", freq="D")         # naive delivery dates D+1


def assert_naive_build(idx, what=""):
    idx = pd.DatetimeIndex(idx)
    assert len(idx) == 0 or idx.max() < END_NAIVE, f"HOLDOUT BREACH {what}: {idx.max()}"


# ---------------------------------------------------------------- loading (holdout enforced here)
def load_wx() -> pd.DataFrame:
    d = ds.dataset(PQ / "weather_gfs.parquet", format="parquet")
    flt = ds.field("target_hour") < pa.scalar(END, type=d.schema.field("target_hour").type)
    w = d.to_table(filter=flt, columns=["point", "zone", "primary", "target_hour", "temperature_2m_c",
                                        "run_lead_hours", "published_at"]).to_pandas()
    assert len(w) > 0 and w["target_hour"].max() < END, "weather holdout filter failed"
    loc = w["target_hour"].dt.tz_convert(TZ)
    w["ddate"] = loc.dt.tz_localize(None).dt.normalize()          # local delivery date of the target hour
    w["lhour"] = loc.dt.hour                                      # local clock hour (fall-back 01:00 twice)
    assert_naive_build(w["ddate"], "weather ddate")
    return w


def load_book() -> pd.DataFrame:
    px = S.load_prices()
    assert px["delivery_hour"].max() < END
    b = S.daily_book(px)
    assert_naive_build(b.index, "book")
    assert (b.index == DELIV).all(), "book days not 2020-01-01 .. 2023-12-31"
    return b


def deadline_series(shift_h: float = 0.0) -> pd.Series:
    """05:00 New York clock time on bid day D, indexed by the naive DELIVERY date D+1."""
    t = [common.decision_time((d - pd.Timedelta(days=1)).date()) for d in DELIV]
    t = pd.DatetimeIndex(t)
    assert (t.hour == 5).all() and (t.minute == 0).all()
    return pd.Series(t + pd.Timedelta(hours=shift_h), index=DELIV)


# ---------------------------------------------------------------- surprise
def pairs(w: pd.DataFrame, last_hour: int = 21, deadline: pd.Series | None = None) -> pd.DataFrame:
    """One row per primary point and target hour of D+1 (local hour <= last_hour) with both the
    48-hour and the 72-hour value, each public at 05:00 on D."""
    dl = deadline_series() if deadline is None else deadline
    x = w[w["primary"] & w["temperature_2m_c"].notna() & (w["lhour"] <= last_hour) & w["ddate"].isin(DELIV)]
    x = x[x["published_at"] <= x["ddate"].map(dl)]
    a = x[x["run_lead_hours"] == 48].set_index(["point", "target_hour"])
    b = x[x["run_lead_hours"] == 72].set_index(["point", "target_hour"])
    assert a.index.is_unique and b.index.is_unique
    j = a[["ddate", "lhour", "temperature_2m_c", "published_at"]].rename(
        columns={"temperature_2m_c": "t48", "published_at": "pub48"}).join(
        b[["temperature_2m_c", "published_at"]].rename(columns={"temperature_2m_c": "t72", "published_at": "pub72"}),
        how="inner")
    j["s"] = (j["t48"] - COMFORT).abs() - (j["t72"] - COMFORT).abs()
    return j.reset_index()


def day_surprise(j: pd.DataFrame, min_pairs: int = MIN_PAIRS) -> pd.DataFrame:
    g = j.groupby("ddate")["s"]
    d = pd.DataFrame({"surprise": g.mean(), "pairs": g.size()}).reindex(DELIV)
    d.loc[d["pairs"].fillna(0) < min_pairs, "surprise"] = np.nan
    return d


def rank_loop(x, window: int = 365, min_obs: int = 60) -> np.ndarray:
    """Share of the previous `window` calendar days with a value strictly below today's (plain loop)."""
    v = np.asarray(x, dtype=float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if np.isnan(v[i]):
            continue
        prev = v[max(0, i - window):i]
        prev = prev[~np.isnan(prev)]
        if len(prev) >= min_obs:
            out[i] = float((prev < v[i]).sum()) / len(prev)
    return out


def score(w, last_hour=21, deadline=None, min_pairs=MIN_PAIRS) -> pd.DataFrame:
    d = day_surprise(pairs(w, last_hour, deadline), min_pairs)
    d["rank"] = rank_loop(d["surprise"].to_numpy())
    return d


# ---------------------------------------------------------------- money
def legs(book: pd.DataFrame, cost: float | None = None):
    """Daily net of 1 MW supply and 1 MW load in all 11 zones x all hours of the delivery day."""
    if cost is None:
        return book["sup"], book["load"]
    return -book["sum_gap"] - book["n"] * cost, book["sum_gap"] - book["n"] * cost


def rule_daily(book, rank, cut_by_year: dict, cost=None) -> pd.Series:
    sup, load = legs(book, cost)
    r = pd.Series(np.asarray(rank, dtype=float), index=book.index)
    c = pd.Series(book.index.year, index=book.index).map(cut_by_year).astype(float)
    on = (r > c).fillna(False).astype(bool)                      # NaN rank or no cut: supply
    return load.where(on, sup), on


def choose_cut(book, rank, Y, cuts=CUTS, default=DEFAULT_CUT, cost=None, drop_dates=()):
    r = pd.Series(np.asarray(rank, dtype=float), index=book.index)
    prior = book.index.year == Y - 1
    if not np.isfinite(r[prior]).any():
        return default
    vals = []
    for c in cuts:
        d, _ = rule_daily(book, rank, {Y - 1: c}, cost)
        d = d[prior & ~book.index.isin(pd.DatetimeIndex(list(drop_dates)))]
        vals.append(d.sum())
    return cuts[int(np.argmax(vals))]


def walk(book, rank, cuts=CUTS, default=DEFAULT_CUT, choose_cost=None, eval_cost=None, drop_dates=()):
    ch = {Y: choose_cut(book, rank, Y, cuts, default, choose_cost, drop_dates) for Y in YEARS}
    d, on = rule_daily(book, rank, ch, eval_cost)
    return ch, d, on


def year_nets(d: pd.Series) -> dict:
    out = {str(Y): round(float(d[d.index.year == Y].sum())) for Y in YEARS}
    out["sum"] = sum(out[str(Y)] for Y in YEARS)
    return out


def sharpe(d: pd.Series) -> float:
    d = d[d.index.year.isin(YEARS)]
    return round(float(d.mean() / d.std() * np.sqrt(365)), 2)


def without_best(d: pd.Series, k: int) -> float:
    return round(float(np.sort(d.to_numpy())[:-k].sum()))
