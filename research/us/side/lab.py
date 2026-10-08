"""Strategy lab: every registered strategy scored the same way over 2021, 2022 and 2023 (build years only).

Governing text: research/us/OBJECTIVES.md, addendum "ten registered ideas" (6 Oct 2026, 22:30, commit 42ad84b):
the ideas, Bo's bar and the safeguards. Runs on gene: ~/nyiso-us/side/lab.py with ~/nyiso-us/.venv.

    python lab.py            score everything whose inputs exist, write side/lab_results.json and .md

HOLDOUT. Every table and prediction file is filtered AT LOAD to delivery (or target) times before
2024-01-01 New York time, and every loader asserts it. Nothing here reads, computes or prints anything for
later dates.

Scoring (identical for every strategy; storm_value.py's score function and cost tables):
  * Per zone-hour, signed MW: -1 virtual supply, +1 virtual load, 0 none. P&L = mw * (RT - DA) - |mw| * cost,
    cost = Schedule 1 + FERC + uplift bound for supply legs, Schedule 1 + FERC for load legs (COST, LOADCOST).
  * Daily P&L = sum over the delivery day's zone-hours; every day counts, days without a position count as 0.
  * Sharpe = mean / std (ddof 1) of daily P&L * sqrt(365). Return on bankroll = net / 500,000 * 365 / days.
  * Max drawdown from the start of the period (a zero is put before the first day).
  * Stress: 0.50 USD per MWh replaces the cost on both sides.
  * Bar (Bo's): average net >= 50,000 USD a year, positive in at least 2 of 3 years, three-year Sharpe > 0.42,
    positive three-year total at the 0.50 stress.

Walk-forward rule (one rule for the whole lab, the same as storm_value.py and model/evaluate.py):
  * Every cut, threshold, zone list or setting used in year Y is chosen on year Y-1 alone, by the strategy's
    own net P&L at full cost; ties go to the first grid value. Model predictions come from monthly refits
    trained only on data up to two days before each month.
  * Declared before any result, for a year whose previous year has no scores at all:
      trailing-rank cuts (idea 10): 0.95 (the lead modeller's DEFAULT_STORM);
      probability cuts (ideas 3, 4, 6 day gate): 0.20, i.e. twice the 10 percent base rate of the target;
      idea 4 pair: (0.20, 0.50); hourly spike cuts (ideas 6, 9): 0.05 (evaluate.py DEFAULT_SPIKE p*);
      idea 5 load margin: the load cost of a zone-day (symmetric with the supply side).
  * Grids: storm and weather cuts (0.80, 0.90, 0.95, 0.98) on trailing 365-day percentile ranks;
    probability cuts (0.10, 0.15, 0.20, 0.30, 0.40, 0.50); hourly spike cuts (0.02, 0.05) = gbm.P_STAR_GRID.

Safeguards (binding, every idea):
  (a) injected lookahead: values published after 05:00 on D are corrupted; with the 05:00 filter every
      decision for D+1 must stay the same, and with the filter opened to midnight they must change.
      Where the lab builds the score itself (baseline, hourly mean, storm score, weather surprise, the
      gradient-boosting labels) this is done on the raw tables; for every strategy the rule layer is also
      tested (corrupting later days' scores never changes today's position) and so is the choice layer
      (corrupting the scored year's outcomes never changes the setting chosen for it).
  (b) placebos: the same rule, same walk-forward, fed (1) its score one day late and (2) its score with the
      dates permuted within each year (20 permutations, fixed seeds).
  (c) "too good": a year with Sharpe above 3, or one day above half the year's profit, is flagged and
      checked in writing.
"""
from __future__ import annotations

import datetime as dt
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

HOME = Path.home() / "nyiso-us"
sys.path.insert(0, str(HOME / "pipeline"))
import common                                   # noqa: E402  pipeline/common.py: decision_time (05:00 New York clock)

PQ = HOME / "parquet"
RES = HOME / "results"
SIDE = HOME / "side"
OPT = Path(os.environ.get("LAB_OPT_DIR", str(RES)))   # other agents' files; overridden only for synthetic tests
OUT_TAG = os.environ.get("LAB_OUT_TAG", "")
TZ = "America/New_York"
END = pd.Timestamp("2024-01-01", tz=TZ)
END_NAIVE = pd.Timestamp("2024-01-01")

# storm_value.py, exactly
COST = {2020: 0.0862 + 0.010 + 0.007, 2021: 0.0757 + 0.015 + 0.004, 2022: 0.0853 + 0.016 + 0.003,
        2023: 0.1066 + 0.017 + 0.026}
LOADCOST = {y: c - u for (y, c), u in zip(COST.items(), [0.007, 0.004, 0.003, 0.026])}   # load legs pay no uplift
ZONES = ["CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL", "LONGIL", "MHK VL", "MILLWD", "N.Y.C.", "NORTH", "WEST"]
UPSTATE = ["WEST", "GENESE", "CENTRL", "NORTH", "MHK VL", "CAPITL"]     # zones A to F
BANK = 500_000
STRESS = 0.50
YEARS = [2021, 2022, 2023]
BAR = {"avg_net_per_year": 50_000, "positive_years": 2, "sharpe_3y": 0.42, "stress_total": 0.0}

CUTS = (0.80, 0.90, 0.95, 0.98)
DEFAULT_RANK_CUT = 0.95
P_GRID = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
DEFAULT_P = 0.20
DEFAULT_P_PAIR = (0.20, 0.50)
PSTAR_GRID = (0.02, 0.05)
DEFAULT_PSTAR = 0.05
COMFORT_C = 18.3            # degree-day base (65 F): "toward the extreme" = further from it
MIN_WX_PAIRS = 100          # a day's weather surprise needs at least 100 point-hours with two runs
N_SHUFFLE = 20
SEED = 20261006
N_TEST_DAYS = 20
SIZE_DD = 100_000           # sizing view: drawdown target, 20 percent of the bankroll
SIZE_CAP = 5.0
REGISTERED = ("Registered so far: ideas 1 to 12, the three basic strategies (always supply, the usual-side baseline, "
              "the hourly-mean version of A) and ideas A to D, 19 strategies in all; A to D are scored in "
              "model/evaluate.py, the other 15 here. Idea 14 (registered 7 Oct after the full table was seen) is "
              "scored here too; idea 13 in robust/summary.py. ")
TRIED_BEFORE = (REGISTERED + "Before this lab, on the same build years: storm_value.py tried 4 rules beside always supply (storm "
                "filter, storm flip, deep risk sizing, a six-zone subset), spike_value.py 2 (deep and history spike "
                "filters, 6 cuts each), the storm audit 22 diagnostic variants of ideas 1 and 2, and model/evaluate.py "
                "the registered ideas A to D with both models. Ideas 1 and 2 were found in that exploration (disclosed "
                "in OBJECTIVES.md).")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ================================================================== loading (holdout enforced here)
def _bound(typ):
    if pa.types.is_timestamp(typ):
        return pa.scalar(END if typ.tz else END_NAIVE, type=typ)
    if pa.types.is_date(typ):
        return pa.scalar(dt.date(2024, 1, 1), type=typ)
    raise TypeError(f"no holdout bound for {typ}")


def assert_build(values, what=""):
    s = pd.Series(values).dropna()
    if not len(s):
        return
    mx = pd.Timestamp(s.max())
    assert (mx < END) if mx.tzinfo is not None else (mx < END_NAIVE), f"HOLDOUT BREACH {what}: {mx}"


def read_build(path, timecol, columns=None, extra=None) -> pd.DataFrame:
    d = ds.dataset(str(path), format="parquet")
    flt = ds.field(timecol) < _bound(d.schema.field(timecol).type)
    if extra is not None:
        flt = flt & extra
    df = d.to_table(filter=flt, columns=columns).to_pandas()
    assert_build(df[timecol], f"{path}:{timecol}")
    return df


def load_tables():
    px = read_build(PQ / "prices_zone.parquet", "delivery_hour",
                    ["delivery_hour", "zone", "da_lbmp", "rt_lbmp", "rt_published_at", "published_at"],
                    ds.field("zone").isin(ZONES))
    lf = read_build(PQ / "load_forecast.parquet", "target_hour", None, ds.field("zone") == "NYISO")
    lf["issue_date"] = pd.to_datetime(lf["issue_date"])
    wx = read_build(PQ / "weather_gfs.parquet", "target_hour", None)
    wx = wx[wx["primary"] & wx["temperature_2m_c"].notna()].reset_index(drop=True)
    return px, lf, wx


# ================================================================== the hourly frame H
class Frame:
    """Every zone-hour of delivery dates 2020-01-01 to 2023-12-31, with day index, zone index and slot
    (local hour; 24 = the second 01:00 of the fall-back day)."""

    def __init__(self, px: pd.DataFrame):
        h = px.sort_values(["delivery_hour", "zone"], kind="stable").reset_index(drop=True)
        h["ddate"] = h["delivery_hour"].dt.tz_localize(None).dt.normalize()
        h["hour"] = h["delivery_hour"].dt.hour
        occ = h.groupby(["ddate", "zone", "hour"]).cumcount()
        h["slot"] = np.where(occ > 0, 24, h["hour"])
        h["gap"] = h["rt_lbmp"] - h["da_lbmp"]
        assert h["gap"].notna().all()
        self.h = h
        self.days = pd.DatetimeIndex(sorted(h["ddate"].unique()))
        assert (np.diff(self.days.values).astype("timedelta64[D]").astype(int) == 1).all(), "days not contiguous"
        assert_build(self.days, "frame days")
        self.ND = len(self.days)
        self.bid = self.days - pd.Timedelta(days=1)
        self.day_year = self.days.year.to_numpy()
        self.DI = self.days.get_indexer(h["ddate"])
        self.ZI = pd.Index(ZONES).get_indexer(h["zone"])
        self.SL = h["slot"].to_numpy()
        self.YEAR = h["ddate"].dt.year.to_numpy()
        self.GAP = h["gap"].to_numpy(float)
        self.SUPC = pd.Series(self.YEAR).map(COST).to_numpy(float)
        self.LOADC = pd.Series(self.YEAR).map(LOADCOST).to_numpy(float)
        self.NH = len(h)
        self.zone = h["zone"].to_numpy()

    def t05(self, shift_h: float = 0.0) -> pd.DatetimeIndex:
        """05:00 New York clock time on each bid day D (aligned with self.days = D+1), from
        pipeline common.decision_time; shift_h opens the filter later (lookahead tests only)."""
        if getattr(self, "_t05", None) is None:
            t = pd.DatetimeIndex([common.decision_time(d.date()) for d in self.bid]).as_unit("ns")
            assert (t == (self.bid + pd.Timedelta(hours=5)).tz_localize(TZ).as_unit("ns")).all()
            assert (t.hour == 5).all()
            self._t05 = t
        return self._t05 + pd.Timedelta(hours=shift_h)

    # ---- money
    def day_pnl(self, mw, cost=None, gap=None):
        g = self.GAP if gap is None else gap
        if cost is None:
            c = np.where(mw < 0, self.SUPC, np.where(mw > 0, self.LOADC, 0.0))
        else:
            c = cost
        pnl = mw * g - np.abs(mw) * c
        return np.bincount(self.DI, weights=pnl, minlength=self.ND)

    def day_mwh(self, mw):
        return np.bincount(self.DI, weights=np.abs(mw), minlength=self.ND)

    def rows(self, arr):
        """Day-level (ND,), zone-day (ND, 11) or cube (ND, 11, 25) array to one value per H row."""
        a = np.asarray(arr)
        if a.ndim == 1:
            return a[self.DI]
        if a.ndim == 2:
            return a[self.DI, self.ZI]
        return a[self.DI, self.ZI, self.SL]

    def to_cube(self, values) -> np.ndarray:
        c = np.full((self.ND, len(ZONES), 25), np.nan)
        c[self.DI, self.ZI, self.SL] = values
        return c


# ================================================================== statistics
def stats(daily: np.ndarray, mwh: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    s = pd.Series(daily, index=dates)
    sd = s.std()
    cum = np.concatenate([[0.0], np.cumsum(daily)])
    mdd = float((cum - np.maximum.accumulate(cum)).min())
    srt = np.sort(daily)
    tot_mwh = float(mwh.sum())
    return {"net_usd": round(float(s.sum())), "return_on_500k_pct": round(100 * float(s.sum()) / BANK * 365 / len(s), 1),
            "sharpe": round(float(s.mean() / sd * np.sqrt(365)), 2) if sd > 0 else None,
            "max_drawdown_usd": round(mdd), "worst_day_usd": round(float(s.min())), "worst_day": str(s.idxmin().date()),
            "best_day_usd": round(float(s.max())), "best_day": str(s.idxmax().date()),
            "net_without_best_3_days": round(float(srt[:-3].sum())), "net_without_best_5_days": round(float(srt[:-5].sum())),
            "mwh_per_day": round(tot_mwh / len(s), 1),
            "usd_per_mwh": round(float(s.sum()) / tot_mwh, 3) if tot_mwh else None, "days": len(s)}


def evaluate(F: Frame, mw: np.ndarray) -> dict:
    full, stress, mwh = F.day_pnl(mw), F.day_pnl(mw, STRESS), F.day_mwh(mw)
    out = {"years": {}}
    for Y in YEARS:
        m = F.day_year == Y
        out["years"][str(Y)] = {**stats(full[m], mwh[m], F.days[m]), "stress_0.50_usd": round(float(stress[m].sum()))}
    m3 = np.isin(F.day_year, YEARS)
    out["three_year"] = {**stats(full[m3], mwh[m3], F.days[m3]), "stress_0.50_usd": round(float(stress[m3].sum()))}
    nets = [out["years"][str(Y)]["net_usd"] for Y in YEARS]
    avg = float(np.mean(nets))
    out["avg_net_per_year"] = round(avg)
    out["avg_return_on_500k_pct"] = round(100 * avg / BANK, 1)
    chk = {"avg_net_per_year": avg >= BAR["avg_net_per_year"],
           "positive_years": sum(n > 0 for n in nets) >= BAR["positive_years"],
           "sharpe_3y": (out["three_year"]["sharpe"] or -9) > BAR["sharpe_3y"],
           "stress_total": out["three_year"]["stress_0.50_usd"] > BAR["stress_total"]}
    out["bar"] = chk
    out["PASS"] = all(chk.values())
    out["best_5_days_three_year"] = {str(F.days[m3][j].date()): round(float(full[m3][j]))
                                     for j in np.argsort(full[m3])[::-1][:5]}
    tg = []
    for Y in YEARS:
        y = out["years"][str(Y)]
        if y["sharpe"] is not None and y["sharpe"] > 3:
            tg.append({"year": Y, "why": f"Sharpe {y['sharpe']} above 3"})
        if y["net_usd"] > 0 and y["best_day_usd"] > 0.5 * y["net_usd"]:
            tg.append({"year": Y, "why": f"best day {y['best_day']} earned {y['best_day_usd']:,} of the year's "
                                         f"{y['net_usd']:,} ({100 * y['best_day_usd'] / y['net_usd']:.0f} percent)"})
    out["too_good"] = tg
    # sizing view (coordinator, 6 Oct 21:40): NOT part of the registered verdict
    d3 = full[m3]
    down = np.sqrt(np.mean(np.minimum(d3, 0.0) ** 2))
    mdd = -float(out["three_year"]["max_drawdown_usd"])
    scale = min(SIZE_CAP, SIZE_DD / mdd) if mdd > 0 else SIZE_CAP
    out["sizing_view"] = {"sharpe": out["three_year"]["sharpe"],
                          "sortino": round(float(d3.mean() / down * np.sqrt(365)), 2) if down > 0 else None,
                          "return_over_max_drawdown": round(avg / mdd, 2) if mdd > 0 else None,
                          "scale_for_100k_drawdown": round(scale, 2),
                          "return_on_500k_at_that_scale_pct": round(100 * scale * avg / BANK, 1),
                          "note": "sizing view, not part of the registered verdict: P&L, costs and drawdown scale "
                                  "linearly with the book; scale = MW per position, capped at 5"}
    out["_daily"] = full
    return out


# ================================================================== strategies and the walk-forward
class Strat:
    def __init__(self, name, idea, line, inputs, pos, grid=None, default=None, has_prior=None, chooser=None,
                 placebo=True, source="lab", lookahead_inputs=None, notes=None, neigh=None):
        self.name, self.idea, self.line, self.inputs, self.pos = name, idea, line, inputs, pos
        self.neigh = neigh
        self.fixed_choices = False          # True: settings were chosen by another agent's procedure
        self.incomplete = None              # text when a scored year has no positions at all
        self.grid, self.default, self.chooser, self.placebo = grid, default, chooser, placebo
        self.has_prior = has_prior or (lambda F, inp, yr: any(np.isfinite(a[F.day_year == yr]).any()
                                                               for a in inp.values()))
        self.source, self.lookahead_inputs, self.notes = source, lookahead_inputs, notes or []


def choose(F, st: Strat, inp, Y, gap=None, on_year=None):
    yr = Y - 1 if on_year is None else on_year
    if st.chooser is not None:
        return st.chooser(F, inp, Y, gap, yr)
    if st.grid is None:
        return None, "no setting"
    if not st.has_prior(F, inp, yr):
        return st.default, "declared default (no scores in the year before)"
    m = choice_mask(F, yr, Y)
    vals = [F.day_pnl(st.pos(F, inp, q), gap=gap)[m].sum() for q in st.grid]
    i = int(np.argmax(vals))
    return st.grid[i], f"best of {len(st.grid)} on {yr}"


def choice_mask(F, yr, Y):
    """Delivery days of year yr whose outcomes are public when year Y's setting is first used (05:00 on 31
    December of Y-1, bidding for 1 January of Y): for yr = Y-1 that drops 31 December."""
    m = F.day_year == yr
    if yr < Y:
        m &= F.days < pd.Timestamp(f"{Y - 1}-12-31")
    return m


def walk(F, st: Strat, inp, gap=None, in_sample=False):
    mw = np.zeros(F.NH)
    ch = {}
    for Y in YEARS:
        p, how = choose(F, st, inp, Y, gap, on_year=Y if in_sample else None)
        m = st.pos(F, inp, p)
        sel = F.YEAR == Y
        mw[sel] = m[sel]
        ch[Y] = (p, how)
    return mw, ch


def shift_inputs(inp):
    out = {}
    for k, a in inp.items():
        b = np.full(a.shape, np.nan)
        b[1:] = a[:-1]
        out[k] = b
    return out


def perm_within_year(F, seed):
    rng = np.random.default_rng(seed)
    perm = np.arange(F.ND)
    for y in np.unique(F.day_year):
        idx = np.where(F.day_year == y)[0]
        perm[idx] = rng.permutation(idx)
    return perm


def shuffle_inputs(F, inp, seed):
    perm = perm_within_year(F, seed)
    return {k: np.asarray(a)[perm] for k, a in inp.items()}


# ================================================================== scores built by the lab
def trailing_rank(x, window=365, min_obs=60):
    """Share of the previous `window` days (strictly earlier, NaN skipped) below today's value (storm_value)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    pad = np.concatenate([np.full(window, np.nan), x])
    win = np.lib.stride_tricks.sliding_window_view(pad, window)[:n]
    cnt = (~np.isnan(win)).sum(1)
    with np.errstate(invalid="ignore"):
        less = (win < x[:, None]).sum(1)
    out = np.full(n, np.nan)
    ok = (cnt >= min_obs) & ~np.isnan(x)
    out[ok] = less[ok] / cnt[ok]
    return out


def _bid_of_target(ts: pd.Series) -> pd.Series:
    return ts.dt.tz_localize(None).dt.normalize() - pd.Timedelta(days=1)


def storm_inputs(F: Frame, px, lf, wx, shift_h=0.0) -> pd.DataFrame:
    """The four day-level inputs of storm_value.py for every bid day D, each filtered on published_at <= 05:00
    New York wall-clock time on D (storm_value.py used local midnight + 5 elapsed hours; see notes)."""
    T = pd.Series(F.t05(shift_h), index=F.bid)
    l = lf.assign(D=_bid_of_target(lf["target_hour"]))
    l = l[l["D"].isin(F.bid)]
    l = l[l["published_at"] <= l["D"].map(T)]
    l = l[l["issue_date"] == l.groupby("D")["issue_date"].transform("max")]
    w = wx.assign(D=_bid_of_target(wx["target_hour"]))
    w = w[w["D"].isin(F.bid)]
    w = w[w["published_at"] <= w["D"].map(T)]
    r = px.assign(D=px["delivery_hour"].dt.tz_localize(None).dt.normalize())
    r = r[r["D"].isin(F.bid)]
    r = r[r["rt_published_at"] <= r["D"].map(T)]
    out = pd.DataFrame(index=F.bid)
    out["peak"] = l.groupby("D")["load_forecast_mw"].max()
    out["tmin"] = w.groupby("D")["temperature_2m_c"].min()
    out["tmax"] = w.groupby("D")["temperature_2m_c"].max()
    out["rt"] = (r["rt_lbmp"] - r["da_lbmp"]).groupby(r["D"]).mean()
    return out


def storm_score(inp: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    ranks = pd.DataFrame({"p_load": trailing_rank(inp["peak"]), "p_cold": trailing_rank(-inp["tmin"]),
                          "p_heat": trailing_rank(inp["tmax"]), "p_rt": trailing_rank(inp["rt"])}, index=inp.index)
    return ranks.max(axis=1, skipna=True).to_numpy(float), ranks       # aligned with F.days (= bid + 1)


def weather_surprise(F: Frame, wx, shift_h=0.0) -> tuple[np.ndarray, pd.DataFrame]:
    """Idea 10. For D+1 hours up to 21:00 at every primary point: the newest and the previous forecast public by
    05:00 on D (in the archive these are the 48-hour and 72-hour leads). Surprise toward the extreme =
    |T_new - 18.3| - |T_prev - 18.3| in C; the day's mean over point-hours with both runs."""
    T = pd.Series(F.t05(shift_h), index=F.bid)
    loc = wx["target_hour"].dt.tz_localize(None)
    w = wx[loc.dt.hour <= 21].assign(D=loc[loc.dt.hour <= 21].dt.normalize() - pd.Timedelta(days=1))
    w = w[w["D"].isin(F.bid)]
    w = w[w["published_at"] <= w["D"].map(T)]
    w = w.sort_values(["point", "target_hour", "published_at"], kind="stable")
    w["k"] = w.groupby(["point", "target_hour"]).cumcount(ascending=False)
    new = w[w["k"] == 0].set_index(["point", "target_hour"])
    prev = w[w["k"] == 1].set_index(["point", "target_hour"])["temperature_2m_c"].rename("t_prev")
    j = new.join(prev, how="inner")
    j["s"] = (j["temperature_2m_c"] - COMFORT_C).abs() - (j["t_prev"] - COMFORT_C).abs()
    g = j.groupby("D")["s"]
    day = pd.DataFrame({"surprise": g.mean(), "pairs": g.size()}).reindex(F.bid)
    day.loc[day["pairs"] < MIN_WX_PAIRS, "surprise"] = np.nan
    day["rank"] = trailing_rank(day["surprise"])
    return day["rank"].to_numpy(float), day


def baseline_cube(F: Frame, px, shift_h=0.0, gap_override=None) -> np.ndarray:
    """Trailing 365-day mean gap per zone and local hour, as panel.py's gap_365d_h: rows with delivery_hour in
    [t - 365 days, t) whose row published_at <= t, t = 05:00 on D. Returned as a cube for delivery day D+1."""
    T = F.t05(shift_h).as_unit("ns").asi8
    Y365 = np.int64(365 * 86400 * 10**9)
    cube = np.full((F.ND, len(ZONES), 25), np.nan)
    d = px[["delivery_hour", "zone", "published_at"]].copy()
    d["gap"] = (px["rt_lbmp"] - px["da_lbmp"]).to_numpy() if gap_override is None else gap_override
    d["hour"] = d["delivery_hour"].dt.hour
    d["dh"] = d["delivery_hour"].dt.tz_convert("UTC").astype("int64") if False else d["delivery_hour"].astype("datetime64[ns, America/New_York]").dt.tz_convert("UTC").dt.tz_localize(None).astype("int64")
    d["pub"] = d["published_at"].astype("datetime64[ns, America/New_York]").dt.tz_convert("UTC").dt.tz_localize(None).astype("int64")
    for (z, hr), g in d.groupby(["zone", "hour"], sort=False):
        g = g.sort_values("dh", kind="stable")
        dh, pub, gap = g["dh"].to_numpy(), g["pub"].to_numpy(), g["gap"].to_numpy(float)
        cs = np.concatenate([[0.0], np.cumsum(gap)])
        lo = np.searchsorted(dh, T - Y365, "left")
        hi = np.searchsorted(dh, T, "left")
        S = cs[hi] - cs[lo]
        N = (hi - lo).astype(float)
        # rows inside the window but not yet published at T: T in (dh, min(pub, dh + 365 d)]
        a = np.searchsorted(T, dh, "right")
        b = np.minimum(np.searchsorted(T, pub, "left"), np.searchsorted(T, dh + Y365, "right"))
        k = b > a
        dS = np.zeros(F.ND + 1)
        dN = np.zeros(F.ND + 1)
        np.add.at(dS, a[k], gap[k]); np.add.at(dS, b[k], -gap[k])
        np.add.at(dN, a[k], 1.0); np.add.at(dN, b[k], -1.0)
        S -= np.cumsum(dS)[:-1]
        N -= np.cumsum(dN)[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(N > 0, S / np.where(N > 0, N, 1), np.nan)
        zi = ZONES.index(z)
        cube[:, zi, hr] = mean
        if hr == 1:
            cube[:, zi, 24] = mean
    return cube


# ================================================================== rules
def mk_basics(F, sig):
    sup_cost = F.SUPC

    def pos_always(F, inp, p):
        return np.full(F.NH, -1.0)

    def pos_base(F, inp, p):
        s = F.rows(inp["sig"])
        return np.where(np.isnan(s), 0.0, np.where(s < 0, -1.0, np.where(s > 0, 1.0, 0.0)))

    def pos_mean(F, inp, p):
        s = F.rows(inp["sig"])
        return np.where(np.isnan(s), 0.0, np.where(s <= -sup_cost, -1.0, 0.0))

    return [
        Strat("always_supply", "basic", "Virtual supply in every zone-hour, every day.", {}, pos_always,
              placebo=False),
        Strat("baseline_usual_side", "basic",
              "Each zone-hour takes the side of its average gap over the trailing 365 days (supply if real time "
              "usually settles below day ahead, load if above).", {"sig": sig}, pos_base),
        Strat("A_hourly_mean", "basic (idea A, simple)",
              "Supply only in the zone-hours whose trailing 365-day average gap is at or below minus the supply "
              "cost; no model.", {"sig": sig}, pos_mean),
    ]


def mk_storm(F, storm):
    def pos_filter(F, inp, c):
        return np.where(F.rows(inp["storm"] > c), 0.0, -1.0)

    def pos_flip(F, inp, c):
        return np.where(F.rows(inp["storm"] > c), 1.0, -1.0)

    pairs = list(itertools.combinations(UPSTATE, 2))
    grid8 = [(c, pr) for c in CUTS for pr in pairs]
    is_dn = np.isin(F.ZI, [ZONES.index("N.Y.C."), ZONES.index("LONGIL")])

    def pos_pairs(F, inp, p):
        c, (u1, u2) = p
        on = F.rows(inp["storm"] > c)
        mw = np.full(F.NH, -1.0)
        mw[on] = 0.0
        mw[on & is_dn] = 1.0
        mw[on & ((F.ZI == ZONES.index(u1)) | (F.ZI == ZONES.index(u2)))] = -1.0
        return mw

    def neigh8(p):
        c, pr = p
        i = CUTS.index(c)
        out = {}
        if i > 0:
            out["lower cut"] = (CUTS[i - 1], pr)
        if i < len(CUTS) - 1:
            out["higher cut"] = (CUTS[i + 1], pr)
        return out

    return [
        Strat("idea1_storm_filter", "1",
              "Sit out the whole day when the hand-made storm score (highest of four trailing-year percentile "
              "ranks: peak load forecast, coldest and hottest forecast temperature, the gap already published "
              "for D) is above a cut; otherwise supply everywhere.", {"storm": storm}, pos_filter, list(CUTS),
              DEFAULT_RANK_CUT),
        Strat("idea2_storm_flip", "2",
              "Same storm score; virtual load in every zone-hour on flagged days, supply on the others.",
              {"storm": storm}, pos_flip, list(CUTS), DEFAULT_RANK_CUT),
        Strat("idea8_storm_zone_pairs", "8",
              "On flagged storm days, load in N.Y.C. and LONGIL against supply in two upstate zones (cut and the "
              "two zones chosen on the year before); supply everywhere on the other days.", {"storm": storm},
              pos_pairs, grid8, (DEFAULT_RANK_CUT, ("WEST", "CENTRL")), neigh=neigh8),
    ]


def mk_weather(F, wrank):
    def pos(F, inp, c):
        return np.where(F.rows(inp["wrank"] > c), 1.0, -1.0)

    return Strat("idea10_weather_surprise", "10",
                 "Virtual load all day when the latest GFS run, against the run before it, moves D+1's temperatures "
                 "further from 18.3 C than on most days of the trailing year (rank above a cut); otherwise supply.",
                 {"wrank": wrank}, pos, list(CUTS), DEFAULT_RANK_CUT)


def mk_spike_rule(name, idea, line, cube_key, cube, extra=None):
    """Idea A's rule: supply unless P(spike) > p*; a missing probability is not traded (strategies.idea_A)."""
    def pos(F, inp, ps):
        p = F.rows(inp[cube_key])
        return np.where(np.isnan(p), 0.0, np.where(p > ps, 0.0, -1.0))

    return Strat(name, idea, line, {cube_key: cube, **(extra or {})}, pos, list(PSTAR_GRID), DEFAULT_PSTAR)


def mk_prob_gate(name, idea, line, key, arr, mode="filter"):
    if mode == "filter":
        def pos(F, inp, c):
            p = F.rows(inp[key])
            return np.where(p > c, 0.0, -1.0)            # NaN > c is False: supply
        return Strat(name, idea, line, {key: arr}, pos, list(P_GRID), DEFAULT_P)
    # three-way: supply below c1, sit out between, load at or above c2
    grid = [(a, b) for a in P_GRID for b in P_GRID if b > a]

    def pos3(F, inp, p):
        c1, c2 = p
        x = F.rows(inp[key])
        return np.where(x >= c2, 1.0, np.where(x > c1, 0.0, -1.0))

    def neigh3(p):
        c1, c2 = p
        i, j = P_GRID.index(c1), P_GRID.index(c2)
        out = {}
        for lab, (a, b) in (("lower sit-out cut", (i - 1, j)), ("higher sit-out cut", (i + 1, j)),
                            ("lower load cut", (i, j - 1)), ("higher load cut", (i, j + 1))):
            if 0 <= a < len(P_GRID) and 0 <= b < len(P_GRID) and a < b:
                out[lab] = (P_GRID[a], P_GRID[b])
        return out
    return Strat(name, idea, line, {key: arr}, pos3, grid, DEFAULT_P_PAIR, neigh=neigh3)


# ================================================================== optional inputs (files of other agents)
def load_tail_cube(F):
    """Idea 9: the lead modeller's walk-forward S = 100 spike classifier (model/evaluate.py tail_pred)."""
    fs = [HOME / "cache" / "wf" / f"spike_S100_{y}.parquet" for y in YEARS]
    if not all(f.exists() for f in fs):
        return None, "pending: " + ", ".join(str(f) for f in fs if not f.exists())
    d = pd.concat([read_build(f, "delivery_hour") for f in fs], ignore_index=True)
    m = F.h[["delivery_hour", "zone"]].merge(d, on=["delivery_hour", "zone"], how="left", validate="one_to_one")
    v = m["pred"].to_numpy(float)
    cov = {str(y): float(np.isfinite(v[F.YEAR == y]).mean()) for y in YEARS}
    return F.to_cube(v), f"cache/wf/spike_S100_2021..2023.parquet, coverage {cov}"


def load_deep_spike_cube(F):
    f = RES / "deep_wf_2021_2023_spike.parquet"
    if not f.exists():
        return None, None, "pending"
    d = read_build(f, "delivery_hour")
    j = json.loads((RES / "deep_wf_2021_2023_spike.json").read_text())
    # timing metadata: each row predicted by the refit of its own month, trained to <= 2 days before it
    rm = pd.to_datetime(d["refit_month"])
    month_ok = bool((rm == d["delivery_date"].dt.to_period("M").dt.start_time).all())
    tl = {r["month"]: r["train_last"] for r in j["refits"]}
    train_ok = all(pd.Timestamp(v) <= pd.Timestamp(k) - pd.Timedelta(days=2) for k, v in tl.items())
    m = F.h[["delivery_hour", "zone"]].merge(d[["delivery_hour", "zone", "p_spike"]], on=["delivery_hour", "zone"],
                                             how="left", validate="one_to_one")
    meta = {"rows_in_own_month_refit": month_ok, "train_last_at_least_2_days_before_month": train_ok,
            "refits": len(tl), "config": j.get("config_name"), "threshold": j.get("threshold")}
    return F.to_cube(m["p_spike"].to_numpy(float)), meta, "results/deep_wf_2021_2023_spike.parquet"


# ================================================================== safeguard (a): lookahead tests
def test_days(F):
    rng = np.random.default_rng(SEED)
    idx = np.where(np.isin(F.day_year, YEARS))[0]
    return np.sort(rng.choice(idx, N_TEST_DAYS, replace=False))


def la_storm(F, px, lf, wx, storm, choices_by_strat):
    """Corrupt everything published after 05:00 on D (load forecast x3, temperatures +40 C, real-time gap
    +1000); recompute the storm score. Filter at 05:00: unchanged. Filter opened to 24:00 on D: changes."""
    out = []
    t05 = F.t05()
    for i in test_days(F):
        T = t05[i]
        lf2 = lf.copy(); m = lf2["published_at"] > T; lf2.loc[m, "load_forecast_mw"] *= 3
        wx2 = wx.copy(); m = wx2["published_at"] > T; wx2.loc[m, "temperature_2m_c"] += 40
        px2 = px.copy(); m = px2["rt_published_at"] > T; px2.loc[m, "rt_lbmp"] += 1000
        s_f, _ = storm_score(storm_inputs(F, px2, lf2, wx2))
        s_o, _ = storm_score(storm_inputs(F, px2, lf2, wx2, shift_h=19))
        s_r, _ = storm_score(storm_inputs(F, px, lf, wx, shift_h=19))   # real data, filter opened
        y = int(F.day_year[i])
        rec = {"delivery_day": str(F.days[i].date()), "storm": _r(storm[i]), "storm_filtered_corrupt": _r(s_f[i]),
               "storm_opened_corrupt": _r(s_o[i]), "storm_opened_real": _r(s_r[i])}
        for name, ch in choices_by_strat.items():
            c = ch[y][0][0] if isinstance(ch[y][0], tuple) else ch[y][0]
            flag = lambda v: bool(v > c) if np.isfinite(v) else False
            rec[name] = {"cut": c, "decision": flag(storm[i]), "filtered_corrupt": flag(s_f[i]),
                         "opened_corrupt": flag(s_o[i]), "opened_real": flag(s_r[i])}
        out.append(rec)
    return out


def _r(v):
    return None if not np.isfinite(v) else round(float(v), 4)


def la_summary_storm(recs, name):
    n = len(recs)
    same_f = sum(r["storm_filtered_corrupt"] == r["storm"] for r in recs)
    dec_f = sum(r[name]["filtered_corrupt"] != r[name]["decision"] for r in recs)
    dec_o = sum(r[name]["opened_corrupt"] != r[name]["decision"] for r in recs)
    dec_r = sum(r[name]["opened_real"] != r[name]["decision"] for r in recs)
    val_o = sum(r["storm_opened_corrupt"] != r["storm"] for r in recs)
    return {"test_days": n, "score_unchanged_with_filter": f"{same_f} of {n}",
            "decisions_changed_with_filter": dec_f, "score_changed_filter_opened": f"{val_o} of {n}",
            "decisions_changed_filter_opened_corrupt": dec_o, "decisions_changed_filter_opened_real_data": dec_r,
            "pass": same_f == n and dec_f == 0 and dec_o > 0}


def la_weather(F, wx, wrank, ch):
    """Inject a fake newer GFS run (24-hour lead, published 16 hours before its target hour, 60 C) and corrupt
    every weather row published after 05:00 on D. Filter at 05:00: unchanged. Opened to 24:00: changes."""
    recs = []
    t05 = F.t05()
    for i in test_days(F):
        T = t05[i]
        d1 = wx[wx["target_hour"].dt.tz_localize(None).dt.normalize() == F.days[i]].copy()
        fake = d1[d1["run_lead_hours"] == 48].copy()
        fake["run_lead_hours"] = 24
        fake["published_at"] = fake["target_hour"] - pd.Timedelta(hours=16)
        fake["temperature_2m_c"] = 60.0
        wx2 = wx.copy(); m = wx2["published_at"] > T; wx2.loc[m, "temperature_2m_c"] += 40
        wx2 = pd.concat([wx2, fake], ignore_index=True)
        r_f, _ = weather_surprise(F, wx2)
        r_o, _ = weather_surprise(F, wx2, shift_h=19)
        y = int(F.day_year[i])
        c = ch[y][0]
        fl = lambda v: bool(v > c) if np.isfinite(v) else False
        recs.append({"delivery_day": str(F.days[i].date()), "rank": _r(wrank[i]), "rank_filtered_corrupt": _r(r_f[i]),
                     "rank_opened_corrupt": _r(r_o[i]), "cut": c, "decision": fl(wrank[i]),
                     "filtered_corrupt": fl(r_f[i]), "opened_corrupt": fl(r_o[i])})
    n = len(recs)
    same = sum((r["rank_filtered_corrupt"] == r["rank"]) for r in recs)
    return {"test_days": n, "score_unchanged_with_filter": f"{same} of {n}",
            "decisions_changed_with_filter": sum(r["filtered_corrupt"] != r["decision"] for r in recs),
            "score_changed_filter_opened": f"{sum(r['rank_opened_corrupt'] != r['rank'] for r in recs)} of {n}",
            "decisions_changed_filter_opened_corrupt": sum(r["opened_corrupt"] != r["decision"] for r in recs),
            "pass": same == n and all(r["filtered_corrupt"] == r["decision"] for r in recs)
                    and any(r["opened_corrupt"] != r["decision"] for r in recs),
            "note": "days before the weather archive has 60 days of history have no rank and trade supply; they "
                    "count as unchanged", "days": recs}


def la_baseline(F, px, sig, strats):
    """Corrupt every real-time gap published after 05:00 on D (+1000). The trailing mean for D+1 must not move
    with the 05:00 filter, and must move with the filter opened to 24:00."""
    out = {}
    t05 = F.t05()
    recs = []
    for i in test_days(F)[:8]:          # each recompute is the full 4-year cube
        T = t05[i]
        g2 = (px["rt_lbmp"] - px["da_lbmp"]).to_numpy(float).copy()
        g2[(px["published_at"] > T).to_numpy()] += 1000
        c_f = baseline_cube(F, px, gap_override=g2)
        c_o = baseline_cube(F, px, shift_h=19, gap_override=g2)
        sel = F.DI == i
        rec = {"delivery_day": str(F.days[i].date()),
               "signal_max_abs_change_with_filter": float(np.nanmax(np.abs(c_f[i] - sig[i]))),
               "signal_max_abs_change_opened": float(np.nanmax(np.abs(c_o[i] - sig[i])))}
        for st in strats:
            p0 = st.pos(F, {"sig": sig}, None)[sel]
            rec[st.name] = {"changed_with_filter": int((st.pos(F, {"sig": c_f}, None)[sel] != p0).sum()),
                            "changed_opened": int((st.pos(F, {"sig": c_o}, None)[sel] != p0).sum()),
                            "zone_hours": int(sel.sum())}
        recs.append(rec)
    for st in strats:
        out[st.name] = {"test_days": len(recs),
                        "zone_hour_positions_changed_with_filter": sum(r[st.name]["changed_with_filter"] for r in recs),
                        "zone_hour_positions_changed_filter_opened": sum(r[st.name]["changed_opened"] for r in recs),
                        "pass": all(r["signal_max_abs_change_with_filter"] < 1e-9 for r in recs)
                                and sum(r[st.name]["changed_with_filter"] for r in recs) == 0
                                and sum(r[st.name]["changed_opened"] for r in recs) > 0,
                        "note": "signal compared to 1e-9 (cumulative sums over corrupted later rows move the last "
                                "digits); positions compared exactly"}
    return out, recs


def la_generic(F, st: Strat, inp, ch):
    """Rule layer: corrupting the score of every later day never changes today's positions. Choice layer:
    corrupting the scored year's outcomes (and all later ones) never changes the setting chosen for it;
    choosing on the scored year itself (lookahead allowed in) does change settings or positions."""
    rng = np.random.default_rng(SEED)
    rule_changed = 0
    for i in test_days(F):
        y = int(F.day_year[i])
        p = ch[y][0]
        inp2 = {}
        for k, a in inp.items():
            b = np.array(a, dtype=float, copy=True)
            b[i + 1:] = rng.uniform(-1e3, 1e3, size=b[i + 1:].shape)
            inp2[k] = b
        sel = F.DI == i
        rule_changed += int((st.pos(F, inp2, p)[sel] != st.pos(F, inp, p)[sel]).sum())
    if st.fixed_choices:
        return {"rule_positions_changed_by_later_scores": rule_changed,
                "choices_changed_by_corrupting_scored_year_outcomes": "chosen by the lead (not re-run here)",
                "choices_changed_when_chosen_on_scored_year": "chosen by the lead",
                "pass": rule_changed == 0}
    choice_changed = 0
    for Y in YEARS:
        g2 = F.GAP.copy()
        m = F.YEAR >= Y
        g2[m] = g2[m] * -3 + rng.normal(0, 500, m.sum())
        if choose(F, st, inp, Y, gap=g2)[0] != ch[Y][0]:
            choice_changed += 1
    ins_mw, ins_ch = walk(F, st, inp, in_sample=True)
    opened = sum(ins_ch[Y][0] != ch[Y][0] for Y in YEARS)
    return {"rule_positions_changed_by_later_scores": rule_changed,
            "choices_changed_by_corrupting_scored_year_outcomes": choice_changed,
            "choices_changed_when_chosen_on_scored_year": opened if st.grid is not None else "no setting",
            "pass": rule_changed == 0 and choice_changed == 0}


# ================================================================== placebos
def placebos(F, st: Strat, inp, real_total):
    if not st.placebo or not inp:
        return {"shifted_one_day_total": None, "shuffled_mean_total": None, "note": "no score to perturb"}
    mw, ch = walk(F, st, shift_inputs(inp))
    e = evaluate(F, mw)
    tots = []
    for s in range(N_SHUFFLE):
        mws, _ = walk(F, st, shuffle_inputs(F, inp, SEED + s))
        tots.append(evaluate(F, mws)["three_year"]["net_usd"])
    tots = np.array(tots)
    return {"shifted_one_day_total": e["three_year"]["net_usd"],
            "shifted_one_day_years": {Y: e["years"][str(Y)]["net_usd"] for Y in YEARS},
            "shuffled_mean_total": round(float(tots.mean())), "shuffled_p95_total": round(float(np.quantile(tots, 0.95))),
            "shuffled_max_total": round(float(tots.max())),
            "shuffled_share_at_or_above_real": round(float((tots >= real_total).mean()), 2), "n_shuffles": N_SHUFFLE}


# ================================================================== fragility (coordinator, 6 Oct 21:00)
def neighbours(st: Strat, p) -> dict:
    if st.grid is None:
        return {}
    if st.neigh is not None:
        return st.neigh(p)
    g = list(st.grid)
    i = g.index(p)
    out = {}
    if i > 0:
        out["lower"] = g[i - 1]
    if i < len(g) - 1:
        out["higher"] = g[i + 1]
    return out


def settings_walk(F, st, inp, settings):
    mw = np.zeros(F.NH)
    for Y in YEARS:
        m = st.pos(F, inp, settings[Y])
        sel = F.YEAR == Y
        mw[sel] = m[sel]
    return mw


class RandomDays:
    """Month-matched random-days placebo: the rule's own day position vectors (every zone-hour of a day,
    moved as a block) are dealt to random days of the same calendar month, 1,000 draws. For a day-level
    rule this is the same number of sit-out, flip or pair days drawn at random within each month.
    Gross P&L of source day a placed on target day b = sum(mw_a * gap_b); costs follow the rows that
    exist on b. Verified: the identity draw reproduces the rule's own net to the cent."""

    def __init__(self, F: Frame, draws=1000):
        self.F, self.draws = F, draws
        self.G = np.zeros((F.ND, len(ZONES), 25))
        self.G[F.DI, F.ZI, F.SL] = F.GAP
        self.E = np.zeros((F.ND, len(ZONES), 25))
        self.E[F.DI, F.ZI, F.SL] = 1.0
        rng = np.random.default_rng(SEED + 7)
        per = F.days.to_period("M")
        self.months = []
        for mth in sorted(set(per[np.isin(F.day_year, YEARS)])):
            idx = np.where(per == mth)[0]
            k = len(idx)
            perms = np.argsort(rng.random((draws, k)), axis=1)
            self.months.append((mth, idx, perms))

    def cube(self, mw):
        F = self.F
        c = np.full((F.ND, len(ZONES), 25), np.nan)
        c[F.DI, F.ZI, F.SL] = mw
        for s in range(25):                        # slots absent on a day (DST) take the 01:00 value
            miss = np.isnan(c[:, :, s])
            c[:, :, s] = np.where(miss, c[:, :, 1], c[:, :, s])
        return np.nan_to_num(c)

    def run(self, mw, real_years):
        F = self.F
        c = self.cube(mw)
        tot = np.zeros(self.draws)
        by_year = {Y: np.zeros(self.draws) for Y in YEARS}
        diag = {Y: 0.0 for Y in YEARS}
        for mth, idx, perms in self.months:
            Y = mth.year
            A = c[idx].reshape(len(idx), -1)
            B = self.G[idx].reshape(len(idx), -1)
            Ex = self.E[idx].reshape(len(idx), -1)
            M = A @ B.T - COST[Y] * ((A < 0) * -A) @ Ex.T - LOADCOST[Y] * ((A > 0) * A) @ Ex.T
            val = M[perms, np.arange(len(idx))].sum(1)      # target b receives source perms[:, b]
            by_year[Y] += val
            tot += val
            diag[Y] += float(np.trace(M))
        for Y in YEARS:                                      # identity check
            assert abs(diag[Y] - real_years[Y]) < 1.0, (Y, diag[Y], real_years[Y])
        real_tot = sum(real_years.values())
        return {"draws": self.draws,
                "percentile_of_rule": round(100 * float((tot < real_tot).mean()), 1),
                "share_random_beats_rule": round(float((tot > real_tot).mean()), 3),
                "random_median_total": round(float(np.median(tot))),
                "years": {str(Y): {"percentile_of_rule": round(100 * float((by_year[Y] < real_years[Y]).mean()), 1),
                                   "share_random_beats_rule": round(float((by_year[Y] > real_years[Y]).mean()), 3),
                                   "random_median": round(float(np.median(by_year[Y])))} for Y in YEARS}}


class Boot:
    """Stationary bootstrap over days (mean block 7), 5,000 draws, one index matrix shared by every strategy."""

    def __init__(self, n, draws=5000, block=7):
        rng = np.random.default_rng(SEED + 11)
        new = rng.random((draws, n)) < 1 / block
        start = rng.integers(0, n, size=(draws, n))
        idx = np.empty((draws, n), dtype=np.int64)
        idx[:, 0] = start[:, 0]
        for t in range(1, n):
            idx[:, t] = np.where(new[:, t], start[:, t], (idx[:, t - 1] + 1) % n)
        self.idx = idx

    def ci_mean(self, daily):
        m = daily[self.idx].mean(1)
        return [round(float(np.quantile(m, 0.025)), 1), round(float(np.quantile(m, 0.975)), 1)]


def fragility(F, st, inp, ch, res, RD: RandomDays, BT: Boot):
    daily = res["_daily"]
    m3 = np.isin(F.day_year, YEARS)
    total = float(daily[m3].sum())
    out = {}
    # neighbours on the grid
    labs = sorted(set().union(*[neighbours(st, ch[Y][0]).keys() for Y in YEARS])) if st.grid else []
    nb = {}
    for lab in labs:
        sets = {Y: neighbours(st, ch[Y][0]).get(lab, ch[Y][0]) for Y in YEARS}
        d = F.day_pnl(settings_walk(F, st, inp, sets))
        nb[lab] = {"settings": {str(Y): _js(sets[Y]) for Y in YEARS},
                   "years": {str(Y): round(float(d[F.day_year == Y].sum())) for Y in YEARS},
                   "three_year": round(float(d[m3].sum()))}
    out["neighbours"] = nb if st.grid else "no setting"
    flip = any(np.sign(v["three_year"]) != np.sign(total) for v in nb.values())
    # best 10 days of each year
    b10 = {}
    for Y in YEARS:
        dy = np.sort(daily[F.day_year == Y])
        b10[str(Y)] = {"net_without_best_10": round(float(dy[:-10].sum())),
                       "three_year_total_without_this_years_best_10": round(total - float(dy[-10:].sum()))}
    out["without_best_10_days"] = b10
    b10_neg = any(v["three_year_total_without_this_years_best_10"] < 0 for v in b10.values())
    # random-days placebo
    if st.name == "always_supply":
        out["random_days"] = "not applicable: the same book every day"
        rd_bad = False
    else:
        mw, _ = walk(F, st, inp)
        real_years = {Y: float(daily[F.day_year == Y].sum()) for Y in YEARS}
        out["random_days"] = RD.run(mw, real_years)
        rd_bad = out["random_days"]["share_random_beats_rule"] > 0.20
    # bootstrap
    ci = BT.ci_mean(daily[m3])
    out["bootstrap_ci95_daily_mean_usd"] = ci
    out["bootstrap_ci95_per_year_usd"] = [round(ci[0] * 365), round(ci[1] * 365)]
    reasons = []
    if flip:
        reasons.append("three-year total changes sign at a neighbouring setting")
    if b10_neg:
        reasons.append("three-year total turns negative without one year's best 10 days")
    if rd_bad:
        reasons.append("random days in the same months beat the rule more than 20 percent of the time")
    out["FRAGILE"] = bool(reasons)
    out["fragile_reasons"] = reasons
    return out


# ================================================================== too-good checks (written, see md)
def too_good_diag(F, st, inp, res, ch, extra_day_info=None):
    out = []
    daily = res["_daily"]
    for tg in res["too_good"]:
        Y = tg["year"]
        y = res["years"][str(Y)]
        i = F.days.get_loc(pd.Timestamp(y["best_day"]))
        sel = F.DI == i
        mw, _ = walk(F, st, inp)
        d = {"year": Y, "flag": tg["why"], "best_day": y["best_day"], "best_day_usd": y["best_day_usd"],
             "year_net_usd": y["net_usd"], "year_net_without_best_day": y["net_usd"] - y["best_day_usd"],
             "mean_gap_that_day_usd_per_mwh": round(float(F.GAP[sel].mean()), 1),
             "max_gap_that_day": round(float(F.GAP[sel].max()), 1),
             "always_supply_that_day_usd": round(float(F.day_pnl(np.full(F.NH, -1.0))[i])),
             "position_mix_that_day": {"load": int((mw[sel] > 0).sum()), "supply": int((mw[sel] < 0).sum()),
                                       "none": int((mw[sel] == 0).sum())}}
        for k, a in inp.items():
            if np.asarray(a).ndim == 1:
                d[f"{k}_that_day"] = _r(a[i]); d[f"{k}_day_before"] = _r(a[i - 1])
        if extra_day_info is not None:
            d.update(extra_day_info(i))
        out.append(d)
    return out


# ================================================================== main
def main():
    t0 = time.time()
    SIDE.mkdir(exist_ok=True)
    log("loading (holdout filter at load)")
    px, lf, wx = load_tables()
    F = Frame(px)
    assert_build(F.h["delivery_hour"], "H"); assert_build(lf["target_hour"], "lf"); assert_build(wx["target_hour"], "wx")
    log(f"H rows {F.NH}, days {F.ND} ({F.days[0].date()} to {F.days[-1].date()})")

    strats, notes = [], {}
    # ---- basics
    log("baseline cube")
    sig = baseline_cube(F, px)
    pb = pd.read_parquet(HOME / "cache" / "panel_base.parquet", columns=["delivery_hour", "zone", "gap_365d_h"])
    pb = pb[pb["delivery_hour"] < END]
    assert_build(pb["delivery_hour"], "panel_base")
    m = F.h[["delivery_hour", "zone"]].merge(pb, on=["delivery_hour", "zone"], how="left", validate="one_to_one")
    mine = F.rows(sig)
    ok = np.isfinite(mine) & m["gap_365d_h"].notna().to_numpy()
    sc = np.isin(F.YEAR, YEARS)
    notes["baseline_crosscheck_vs_panel_base"] = {
        "rows_compared_2021_2023": int((ok & sc).sum()),
        "max_abs_diff": float(np.abs(mine - m["gap_365d_h"].to_numpy())[ok & sc].max()),
        "sign_agreement": float((np.sign(mine) == np.sign(m["gap_365d_h"].to_numpy()))[ok & sc].mean())}
    log("baseline crosscheck", notes["baseline_crosscheck_vs_panel_base"])
    basics = mk_basics(F, sig)
    strats += basics

    # ---- storm score (ideas 1, 2, 8)
    log("storm score")
    s_in = storm_inputs(F, px, lf, wx)
    storm, ranks = storm_score(s_in)
    # reproduction of storm_value.py's deadline (local midnight + 5 elapsed hours) to quantify the difference
    T_script = pd.Series(F.bid.tz_localize(TZ) + pd.Timedelta(hours=5), index=F.bid)
    diff_days = [str(d.date()) for d, a, b in zip(F.bid, F.t05(), T_script) if a != b]
    notes["storm_deadline"] = {"rule": "05:00 wall clock New York on D", "days_where_storm_value_deadline_differs":
                               len(diff_days), "examples": diff_days[:8]}
    storm_strats = mk_storm(F, storm)
    strats += storm_strats

    # ---- idea 10
    log("weather surprise")
    wrank, wday = weather_surprise(F, wx)
    notes["weather_surprise"] = {"first_day_with_rank": str(F.days[np.where(np.isfinite(wrank))[0][0]].date()),
                                 "days_with_rank_2021_2023": int(np.isfinite(wrank[np.isin(F.day_year, YEARS)]).sum()),
                                 "runs_used": "newest and previous public run per point-hour; in the archive the "
                                              "48-hour and 72-hour leads"}
    strats.append(mk_weather(F, wrank))

    # ---- idea 9 (lead modeller's tail classifier)
    tail, tail_src = load_tail_cube(F)
    pending = {}
    if tail is not None:
        st9 = mk_spike_rule("idea9_tail_spike_S100", "9",
                                    "Idea A with spikes defined as real time at least 100 USD above day ahead: supply "
                                    "in every zone-hour except where the gradient-boosting probability of such a spike "
                                    "exceeds p* (0.02 or 0.05, chosen on the year before).", "p_tail", tail)
        st9.lookahead_inputs = {"pass": None, "note": "inputs not tested here: the lead modeller's walk-forward "
                                "(model/walkforward.py, model/evaluate.py tail_pred) owns the feature and label timing; "
                                "its prediction file carries no refit dates to check"}
        st9.source = "lead modeller (cache/wf/spike_S100_*.parquet)"
        strats.append(st9)
        notes["idea9_source"] = tail_src
    else:
        pending["idea9_tail_spike_S100"] = tail_src

    # ---- deep and gbm day ideas (3 to 7), when their files exist
    dsp_cube, dsp_meta, dsp_src = load_deep_spike_cube(F)
    notes["deep_spike_file"] = {"source": dsp_src, "timing_metadata": dsp_meta}
    extra = deep_day_strats(F, px, dsp_cube, notes, pending)
    strats += extra

    # ---- ideas 11 and 12 (policy agent)
    strats += policy_strats(F, notes, pending)
    strats += idea14_strats(F, notes, pending)

    # ---- the side past-spike-rate filter, and the lead modeller's strategies (master table)
    strats.append(mk_hist(F))
    lead, lead_same = load_lead(F, notes)
    strats += lead

    # ---- run everything
    RD = RandomDays(F)
    BT = Boot(int(np.isin(F.day_year, YEARS).sum()))
    results = {}
    choices_storm = {}
    for st in strats:
        log("walk", st.name)
        inp = st.inputs
        mw, ch = walk(F, st, inp)
        res = evaluate(F, mw)
        res["choices"] = {str(Y): {"setting": _js(ch[Y][0]), "how": ch[Y][1]} for Y in YEARS}
        res["placebo"] = placebos(F, st, inp, res["three_year"]["net_usd"])
        res["lookahead_rule_and_choice"] = la_generic(F, st, inp, ch) if inp else {"pass": True, "note": "no inputs"}
        res["too_good_checks"] = too_good_diag(F, st, inp, res, ch)
        res["fragility"] = fragility(F, st, inp, ch, res, RD, BT)
        res["FRAGILE"] = res["fragility"]["FRAGILE"]
        res["idea"], res["line"], res["source"], res["notes"] = st.idea, st.line, st.source, st.notes
        if st.incomplete:
            res["incomplete"] = st.incomplete
            res["PASS"] = None
        lt = getattr(st, "lead_totals", None) or lead_same.get(st.name)
        if lt:
            res["vs_lead"] = {Y: {"lab": res["years"][Y]["net_usd"], "lead": round(lt[Y]),
                                  "diff_pct": round(100 * (res["years"][Y]["net_usd"] - lt[Y]) / max(abs(lt[Y]), 1), 2)}
                              for Y in map(str, YEARS) if Y in lt}
        if st.name in ("idea1_storm_filter", "idea2_storm_flip", "idea8_storm_zone_pairs"):
            choices_storm[st.name] = ch
        results[st.name] = res
        if st.name == "idea10_weather_surprise":
            results[st.name]["lookahead_inputs"] = la_weather(F, wx, wrank, ch)

    log("lookahead: baseline inputs")
    la_b, la_b_days = la_baseline(F, px, sig, basics[1:])
    for k, v in la_b.items():
        results[k]["lookahead_inputs"] = v
    results["always_supply"]["lookahead_inputs"] = {"pass": True, "note": "no inputs: the position never changes"}
    log("lookahead: storm inputs")
    la_s = la_storm(F, px, lf, wx, storm, choices_storm)
    for k in choices_storm:
        results[k]["lookahead_inputs"] = la_summary_storm(la_s, k)
    for st in strats:
        if st.lookahead_inputs is not None:
            results[st.name]["lookahead_inputs"] = st.lookahead_inputs

    # ---- reproduction check against storm_value.py's published numbers
    try:
        sv = json.loads((SIDE.parent / "results" / "side_storm_value.json").read_text())
        rep = {}
        for k_lab, k_sv in [("always_supply", "always_supply"), ("idea1_storm_filter", "storm_day_filter"),
                            ("idea2_storm_flip", "storm_day_flip_to_load")]:
            rep[k_lab] = {str(Y): {"lab": results[k_lab]["years"][str(Y)]["net_usd"], "storm_value": sv[str(Y)][k_sv]["net_usd"],
                                   "lab_cut": results[k_lab]["choices"][str(Y)]["setting"],
                                   "storm_value_cut": sv[str(Y)][k_sv].get("cut")} for Y in YEARS}
        notes["reproduction_vs_storm_value_json"] = rep
    except Exception as ex:      # noqa: BLE001
        notes["reproduction_vs_storm_value_json"] = f"not available: {ex}"

    m3 = np.isin(F.day_year, YEARS)
    pd.DataFrame({k: v["_daily"][m3] for k, v in results.items()}, index=F.days[m3]).rename_axis("delivery_date") \
        .to_parquet(SIDE / f"lab_daily{OUT_TAG}.parquet")
    for v in results.values():
        v.pop("_daily", None)
    n_settings = sum((len(st.grid) if st.grid else 1) for st in strats)
    meta = {"written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "seconds": round(time.time() - t0),
            "holdout": "every table filtered at load to delivery/target times before 2024-01-01 New York, asserted",
            "years": YEARS, "bankroll_usd": BANK, "costs_supply": COST, "costs_load": LOADCOST, "stress_usd_per_mwh": STRESS,
            "bar": BAR, "grids": {"rank_cuts": CUTS, "prob_cuts": P_GRID, "pstar": PSTAR_GRID},
            "defaults": {"rank_cut": DEFAULT_RANK_CUT, "prob_cut": DEFAULT_P, "prob_pair": DEFAULT_P_PAIR,
                         "pstar": DEFAULT_PSTAR},
            "strategies_scored": len(strats), "settings_searched": n_settings, "pending": pending,
            "n_shuffles": N_SHUFFLE, "test_days": [str(F.days[i].date()) for i in test_days(F)],
            "tried_before": TRIED_BEFORE}
    out = {"meta": meta, "notes": notes, "strategies": results,
           "lookahead_storm_days": la_s, "lookahead_baseline_days": la_b_days}
    (SIDE / f"lab_results{OUT_TAG}.json").write_text(json.dumps(out, indent=1, default=_js))
    write_md(json.loads(json.dumps(out, default=_js)), SIDE / f"lab_results{OUT_TAG}.md")
    log("wrote", SIDE / f"lab_results{OUT_TAG}.json", f"{time.time() - t0:.0f} s")


def _js(o):
    if isinstance(o, tuple):
        return [_js(x) for x in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, dt.date)):
        return str(o)
    return o


# ================================================================== ideas 3 to 7 (filled when files exist)
DAY_FEATS = OPT / "day_features_2020_2023.parquet"
DEEP_DAY = OPT / "deep_day_wf_2021_2023.parquet"
GBM_PARAMS = dict(objective="binary", n_estimators=300, learning_rate=0.03, num_leaves=7, min_data_in_leaf=20,
                  feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED,
                  n_jobs=1, deterministic=True, force_row_wise=True)
GBM_FIRST_MONTH = dt.date(2020, 4, 1)       # first refit: trained on January to March 2020 (2020 predictions only
                                            # serve to choose the 2021 cut)
TRAIN_GAP_DAYS = 2
STORM_Q = 0.10


def month_starts(lo: dt.date, hi: dt.date):
    m = dt.date(lo.year, lo.month, 1)
    while m <= hi:
        yield m
        m = dt.date(m.year + (m.month == 12), m.month % 12 + 1, 1)


def load_day_features():
    df = read_build(DAY_FEATS, "delivery_date")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"])
    assert_build(df["delivery_date"], "day features")
    return df


def gbm_fit_predict(tr, te, cols, label_override=None):
    import lightgbm as lgb
    book = tr["y__book_supply_pnl"].to_numpy(float) if label_override is None else label_override
    L = float(np.quantile(book, STORM_Q))
    y = (book < L).astype(int)
    if y.sum() < 2 or y.sum() == len(y):
        return np.full(len(te), np.nan), L
    m = lgb.LGBMClassifier(**GBM_PARAMS)
    m.fit(tr[cols].to_numpy(float), y)
    return m.predict_proba(te[cols].to_numpy(float))[:, 1], L


def gbm_train_rows(df, ms: dt.date, through=None):
    """Training rows for the refit of month ms: delivery dates up to two days before the month whose
    labels were public by 05:00 on the month's first bid day (train_deep_day.walk's rule).
    `through` replaces the cut (lookahead test only)."""
    cut = pd.Timestamp(ms - dt.timedelta(days=TRAIN_GAP_DAYS)) if through is None else through
    known = common.decision_time(ms - dt.timedelta(days=1))
    sel = (df["delivery_date"] <= cut) & (df["y__n_rows"] > 0)
    if through is None:
        sel &= df["y__label_published_at"] <= known
    return df[sel]


def gbm_idea7(F, df):
    cols = [c for c in df.columns if "__" in c and not c.startswith("y__")]
    p = pd.Series(np.nan, index=F.days)
    recs = []
    for ms in month_starts(GBM_FIRST_MONTH, dt.date(2023, 12, 1)):
        me = pd.Timestamp(ms) + pd.offsets.MonthEnd(0)
        tr = gbm_train_rows(df, ms)
        te = df[(df["delivery_date"] >= pd.Timestamp(ms)) & (df["delivery_date"] <= me)]
        assert tr["delivery_date"].max() < te["delivery_date"].min()
        pr, L = gbm_fit_predict(tr, te, cols)
        p.loc[pd.DatetimeIndex(te["delivery_date"])] = pr
        recs.append({"month": str(ms), "train_days": len(tr), "train_last": str(tr["delivery_date"].max().date()),
                     "L": round(L, 1), "positives": int((tr["y__book_supply_pnl"] < L).sum())})
    return p.to_numpy(float), recs, cols


def la_gbm(F, df, cols, p7):
    """Idea 7 inputs: labels after the training cut are corrupted (x -5 and noise). With the cut, the month's
    predictions must not move; with the label window opened to the month's own last day they must."""
    rng = np.random.default_rng(SEED)
    out = []
    for ms in (dt.date(2021, 7, 1), dt.date(2022, 12, 1), dt.date(2023, 2, 1)):
        me = pd.Timestamp(ms) + pd.offsets.MonthEnd(0)
        te = df[(df["delivery_date"] >= pd.Timestamp(ms)) & (df["delivery_date"] <= me)]
        cut = pd.Timestamp(ms - dt.timedelta(days=TRAIN_GAP_DAYS))
        d2 = df.copy()
        late = d2["delivery_date"] > cut
        d2.loc[late, "y__book_supply_pnl"] = d2.loc[late, "y__book_supply_pnl"] * -5 + rng.normal(0, 5e4, late.sum())
        pr_f, _ = gbm_fit_predict(gbm_train_rows(d2, ms), te, cols)
        tr_o = gbm_train_rows(d2, ms, through=me)
        pr_o, _ = gbm_fit_predict(tr_o, te, cols)
        base = p7[F.days.get_indexer(pd.DatetimeIndex(te["delivery_date"]))]
        out.append({"month": str(ms), "max_abs_change_with_cut": float(np.nanmax(np.abs(pr_f - base))),
                    "max_abs_change_opened": float(np.nanmax(np.abs(pr_o - base)))})
    return {"months": out, "pass": all(o["max_abs_change_with_cut"] < 1e-12 for o in out)
            and all(o["max_abs_change_opened"] > 1e-6 for o in out),
            "features": "the day-feature matrix is the deep agent's (deep_day.build_day_features, every value from "
                        "timing.features_available_at at 05:00 on D, tested in model/test_deep_day.py); the lab "
                        "tests the label window it controls"}


def tri(*xs):
    """True if every check ran and passed, False if any ran and failed, None if any did not run."""
    if any(x is False for x in xs):
        return False
    if any(x is None for x in xs):
        return None
    return True


def load_deep_day(F):
    if not DEEP_DAY.exists():
        return None
    d = read_build(DEEP_DAY, "delivery_date")
    d["delivery_date"] = pd.to_datetime(d["delivery_date"])
    assert_build(d["delivery_date"], "deep day")
    day = d[d["zone"] == "ALL"].set_index("delivery_date")
    p_storm = day["p_storm"].reindex(F.days).to_numpy(float)
    z = d[d["zone"] != "ALL"].pivot(index="delivery_date", columns="zone", values="pred_zone_supply_pnl")
    zp = z.reindex(index=F.days, columns=ZONES).to_numpy(float)
    meta = {"rows": len(d), "first": str(d["delivery_date"].min().date()), "last": str(d["delivery_date"].max().date())}
    rm = pd.to_datetime(d["refit_month"])
    meta["rows_in_own_month_refit"] = bool((rm == d["delivery_date"].dt.to_period("M").dt.start_time).all())
    jf = DEEP_DAY.with_suffix(".json")
    if jf.exists():
        j = json.loads(jf.read_text())
        meta["train_last_at_least_2_days_before_month"] = all(
            pd.Timestamp(r["train_last"]) <= pd.Timestamp(r["month"]) - pd.Timedelta(days=2) for r in j["refits"])
        meta["refits"] = len(j["refits"])
        meta["eval_by_year"] = j.get("eval_by_year")
    return p_storm, zp, meta


def deep_day_strats(F, px, dsp_cube, notes, pending):
    strats = []
    # ---- idea 7: gradient boosting on the deep agent's day features
    if DAY_FEATS.exists():
        log("idea 7: LightGBM walk-forward on the day features")
        df = load_day_features()
        p7, recs, cols = gbm_idea7(F, df)
        notes["idea7_gbm"] = {"features": len(cols), "refits": recs, "params": GBM_PARAMS,
                              "first_refit": str(GBM_FIRST_MONTH)}
        la7 = la_gbm(F, df, cols, p7)
        st7 = mk_prob_gate("idea7_gbm_storm_filter", "7",
                           "LightGBM reads the deep agent's day features and gives the probability that the next "
                           "day's always-supply book loses more than the worst-tenth day of its training window; "
                           "sit out the whole day when it is above a cut chosen on the year before, otherwise supply.",
                           "p7", p7)
        st7.lookahead_inputs = la7
        st7.source = "lab (LightGBM, monthly refits from April 2020)"
        strats.append(st7)
    else:
        pending["idea7_gbm_storm_filter"] = f"waiting for {DAY_FEATS}"
    # ---- ideas 3 to 6: deep agent's walk-forward predictions
    dd = load_deep_day(F)
    if dd is None:
        for k in ("idea3_deep_storm_filter", "idea4_deep_storm_three_way", "idea5_deep_zone_day", "idea6_deep_two_gates"):
            pending[k] = f"waiting for {DEEP_DAY}"
        return strats
    p_storm, zp, meta = dd
    notes["deep_day_file"] = meta
    la_deep = {"pass": tri(meta.get("rows_in_own_month_refit"), meta.get("train_last_at_least_2_days_before_month")),
               "checked": "every prediction comes from the refit of its own month, trained on delivery dates at "
                          "least two days before that month (deep_day_wf json); feature timing is the deep agent's "
                          "test (model/test_deep_day.py)", **{k: meta.get(k) for k in
                                                              ("rows_in_own_month_refit", "train_last_at_least_2_days_before_month")}}
    st3 = mk_prob_gate("idea3_deep_storm_filter", "3",
                       "The deep day model's probability that the next day's always-supply book has a worst-tenth "
                       "loss; sit out the whole day when it is above a cut, otherwise supply.", "p_storm", p_storm)
    st4 = mk_prob_gate("idea4_deep_storm_three_way", "4",
                       "Same deep probability with two cuts: supply below the first, sit out between, virtual load "
                       "in every zone-hour at or above the second.", "p_storm", p_storm, mode="three")
    for st in (st3, st4):
        st.lookahead_inputs, st.source = la_deep, "deep agent (deep_day_wf_2021_2023)"
        strats.append(st)

    # ---- idea 5: per-zone day profit
    M_GRID = (0.0, 10.0, 25.0, 50.0, 100.0, 200.0)
    be = {y: 24 * (COST[y] + LOADCOST[y]) for y in COST}

    def pos5(F, inp, M):
        x = F.rows(inp["zp"])
        BE = pd.Series(F.YEAR).map(be).to_numpy(float)
        return np.where(x > 0, -1.0, np.where(x < -(BE + M), 1.0, 0.0))    # NaN: none
    st5 = Strat("idea5_deep_zone_day", "5",
                "The deep model predicts each zone's next-day supply profit after cost; supply that zone all day "
                "when the prediction is above zero, virtual load when it is below minus (the load break-even plus "
                "a margin chosen on the year before), otherwise nothing.", {"zp": zp}, pos5, list(M_GRID), 0.0,
                source="deep agent (deep_day_wf_2021_2023)", lookahead_inputs=la_deep)
    strats.append(st5)

    # ---- idea 6: idea 3's day gate, then the deep hourly spike gate
    if dsp_cube is not None:
        def pos6(F, inp, p):
            c3, ps = p
            day_out = F.rows(inp["p_storm"] > c3)
            h = F.rows(inp["p_spike"])
            return np.where(day_out | np.isnan(h) | (h > ps), 0.0, -1.0)

        def chooser6(F, inp, Y, gap, yr):
            c3, how3 = choose(F, st3, {"p_storm": inp["p_storm"]}, Y, gap, on_year=yr)
            m = choice_mask(F, yr, Y)
            if not (np.isfinite(inp["p_storm"][m]).any() and np.isfinite(inp["p_spike"][m]).any()):
                return (c3, DEFAULT_PSTAR), f"day cut: {how3}; p*: declared default"
            vals = [F.day_pnl(pos6(F, inp, (c3, ps)), gap=gap)[m].sum() for ps in PSTAR_GRID]
            return (c3, PSTAR_GRID[int(np.argmax(vals))]), f"day cut: {how3}; p*: best of {len(PSTAR_GRID)} on {yr}"

        def neigh6(p):
            c3, ps = p
            i, j = P_GRID.index(c3), PSTAR_GRID.index(ps)
            out = {}
            if i > 0: out["lower day cut"] = (P_GRID[i - 1], ps)
            if i < len(P_GRID) - 1: out["higher day cut"] = (P_GRID[i + 1], ps)
            if j > 0: out["lower p*"] = (c3, PSTAR_GRID[j - 1])
            if j < len(PSTAR_GRID) - 1: out["higher p*"] = (c3, PSTAR_GRID[j + 1])
            return out
        grid6 = [(a, b) for a in P_GRID for b in PSTAR_GRID]
        st6 = Strat("idea6_deep_two_gates", "6",
                    "Idea 3's day gate (sit out days the deep day model flags), then inside the days that pass, the "
                    "deep hourly spike model: supply every zone-hour except where its spike probability is above p*.",
                    {"p_storm": p_storm, "p_spike": dsp_cube}, pos6, grid6, (DEFAULT_P, DEFAULT_PSTAR),
                    chooser=chooser6, neigh=neigh6, source="deep agent (deep_day_wf and deep_wf_spike)",
                    lookahead_inputs=la_deep)
        strats.append(st6)
    else:
        pending["idea6_deep_two_gates"] = "waiting for results/deep_wf_2021_2023_spike.parquet"
    return strats


# ================================================================== ideas 11 and 12 (policy agent's file)
POLICY = OPT / "deep_policy_wf_2021_2023.parquet"
POLICY_NAMES = {"11": ("idea11_learned_allocator",
                       "A deep gate network weights four simple experts per zone and day (always supply, sit out, "
                       "virtual load, the storm-day zone pair), trained on their realised profit in earlier days; "
                       "the position is the weighted mix, held all day."),
                "12": ("idea12_policy_network",
                       "A deep network outputs each zone's position for the next day directly, from minus 1 (supply) "
                       "to plus 1 (load), trained on next-day profit after costs minus a penalty on losses; held all "
                       "day.")}


def policy_strats(F, notes, pending):
    if not POLICY.exists():
        for k, (name, _) in POLICY_NAMES.items():
            pending[name] = f"waiting for {POLICY}"
        return []
    d = read_build(POLICY, "delivery_date")
    d["delivery_date"] = pd.to_datetime(d["delivery_date"])
    assert_build(d["delivery_date"], "policy")
    pcol = next((c for c in ("position", "pos", "mw", "position_mw") if c in d.columns), None)
    if pcol is None or "idea" not in d.columns or "zone" not in d.columns:
        for k, (name, _) in POLICY_NAMES.items():
            pending[name] = f"policy file has no position column I can read: {list(d.columns)}"
        return []
    out = []
    meta = {"rows": len(d), "columns": list(d.columns), "position_column": pcol}
    if "refit_month" in d.columns:
        rm = pd.to_datetime(d["refit_month"])
        meta["rows_in_own_month_refit"] = bool((rm == d["delivery_date"].dt.to_period("M").dt.start_time).all())
    jf = POLICY.with_suffix(".json")
    if jf.exists():
        j = json.loads(jf.read_text())
        recs = j.get("refits") or []
        if isinstance(recs, dict):
            recs = [r for v in recs.values() for r in (v if isinstance(v, list) else [v])]
        tl = [r for r in recs if isinstance(r, dict) and "train_last" in r and "month" in r]
        if tl:
            meta["train_last_at_least_2_days_before_month"] = all(
                pd.Timestamp(r["train_last"]) <= pd.Timestamp(r["month"]) - pd.Timedelta(days=2) for r in tl)
    notes["policy_file"] = meta
    for v, g in d.groupby(d["idea"].astype(str)):
        key = next((k for k in POLICY_NAMES if k in v), None)
        name, line = POLICY_NAMES[key] if key else (f"idea_{v}", f"Positions from the policy file, idea {v}.")
        z = g.pivot_table(index="delivery_date", columns="zone", values=pcol, aggfunc="first")
        arr = np.clip(z.reindex(index=F.days, columns=ZONES).to_numpy(float), -1, 1)

        def pos(F, inp, p):
            return np.nan_to_num(F.rows(inp["pos"]))
        la = {"pass": tri(meta.get("rows_in_own_month_refit"), meta.get("train_last_at_least_2_days_before_month")),
              "checked": "refit dates in the policy file and its json (each row from its own month's refit, trained "
                         "to at least two days before the month); feature timing is the policy agent's test",
              **{k: meta.get(k) for k in ("rows_in_own_month_refit", "train_last_at_least_2_days_before_month")}}
        st = Strat(name, key or v, line, {"pos": arr}, pos, source="policy agent (deep_policy_wf_2021_2023)",
                   lookahead_inputs=la)
        cov = {str(Y): float(np.isfinite(arr[F.day_year == Y]).mean()) for Y in YEARS}
        st.notes = [f"coverage of zone-days with a position: {cov}"]
        out.append(st)
    return out


# ================================================================== past spike rate filter (spike_value.py)
HIST_S = 25.0
ROLL_K = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)       # spike_value.py and evaluate.py grid
DEFAULT_ROLL_K = 0.10                                 # evaluate.py DEFAULT_ROLL_K
ROLL_DAYS = 30


def hist_cube(F):
    """Per year Y: each zone-hour's spike frequency (gap >= 25) over delivery dates from 2020-01-01 to 30
    December of Y-1 (public by 05:00 on 31 December, the first bid of Y). 2020 has no earlier data: NaN."""
    c = np.full((F.ND, len(ZONES), 25), np.nan)
    h = F.h
    for Y in sorted(set(F.day_year)):
        hist = h[h["ddate"] <= pd.Timestamp(f"{Y - 1}-12-30")]
        if not len(hist):
            continue
        rate = (hist["gap"] >= HIST_S).groupby([hist["zone"], hist["hour"]]).mean()
        sel = np.where(F.YEAR == Y)[0]
        key = pd.MultiIndex.from_arrays([F.zone[sel], h["hour"].to_numpy()[sel]])
        c[F.DI[sel], F.ZI[sel], F.SL[sel]] = rate.reindex(key).to_numpy(float)
    return c


def rolling_cut_rows(F, cube, k):
    """Per delivery day: the (1-k) quantile of the pooled scores of the previous 30 delivery days (strictly
    earlier); no earlier scores: never sit out (spike_value.rolling_cut)."""
    flat = cube.reshape(F.ND, -1)
    thr = np.full(F.ND, np.inf)
    for i in range(F.ND):
        pool = flat[max(0, i - ROLL_DAYS):i].ravel()
        pool = pool[np.isfinite(pool)]
        if len(pool):
            thr[i] = np.quantile(pool, 1 - k)
    return thr


def mk_hist(F):
    cache = {}

    def pos(F, inp, k):
        arr = inp["p_hist"]
        key = (id(arr), k)
        hit = cache.get(key)
        if hit is None or hit[0] is not arr:          # the cache holds the array, so its id cannot be reused
            if len(cache) > 64:
                cache.clear()
            hit = cache[key] = (arr, rolling_cut_rows(F, arr, k))
        thr = hit[1]
        p = F.rows(inp["p_hist"])
        return np.where(np.isfinite(p) & (p > thr[F.DI]), 0.0, -1.0)
    return Strat("side_past_spike_rate_filter", "side (spike_value.py history filter)",
                 "Supply everywhere except zone-hours whose spike frequency (real time 25 USD or more above day "
                 "ahead) in all earlier years is in the top k of the previous 30 days' scores; k chosen on the year "
                 "before.", {"p_hist": hist_cube(F)}, pos, list(ROLL_K), DEFAULT_ROLL_K)


# ================================================================== the lead modeller's strategies
LEAD_POS = SIDE / "lead_positions_2021_2023.parquet"
LEAD_TOT = SIDE / "lead_ledger_totals.json"
LEAD_STORM = SIDE / "lead_storm_score.parquet"
LEAD_SAME_AS_LAB = {"always_supply": "always_supply", "baseline": "baseline_usual_side", "A_hourly_mean": "A_hourly_mean"}
LEAD_IDEA = {"A_spike_gbm": "A (main, gbm)", "A_spike_deep": "A (main, deep)", "A_regression_v1": "A v1 (superseded)",
             "A_regression_v1_deep": "A v1 (deep)", "B_gbm": "B (gbm)", "B_deep": "B (deep)", "C_gbm": "C (gbm)",
             "C_deep": "C (deep)", "D_gbm": "D (gbm)", "D_deep": "D (deep)", "storm_day_filter": "1 (lead's build)",
             "storm_day_flip": "2 (lead's build)", "A_spike_tail_S100_gbm": "9 (lead's choice of p*)",
             "risk_sized_supply_gbm": "side (risk sizing)", "zone_subset_supply": "side (zone subset)",
             "A_spike_deep_rolling_cut": "side (deep rolling cut)"}


def load_lead(F, notes):
    if not LEAD_POS.exists():
        return [], {}
    d = read_build(LEAD_POS, "delivery_hour")
    tot = json.loads(LEAD_TOT.read_text())
    lead_desc = {}
    try:
        lj = json.loads((Path.home() / "nyiso-us" / "results" / "strategy_list_2021_2023.json").read_text())
        lead_desc = {x["strategy"]: x["what"] for x in lj["strategies"]}
    except Exception:            # noqa: BLE001
        pass
    storm = None
    if LEAD_STORM.exists():
        ls = pd.read_parquet(LEAD_STORM)
        ls["delivery_date"] = pd.to_datetime(ls["delivery_date"])
        ls = ls[ls["delivery_date"] < END_NAIVE]
        assert_build(ls["delivery_date"], "lead storm")
        storm = ls.set_index("delivery_date")["storm"].reindex(F.days).to_numpy(float)
    key = F.h[["delivery_hour", "zone"]]
    out = []
    for name, g in d.groupby("strategy"):
        if name in LEAD_SAME_AS_LAB:
            continue
        m = key.merge(g[["delivery_hour", "zone", "mw", "pred"]], on=["delivery_hour", "zone"], how="left",
                      validate="one_to_one")
        mw = m["mw"].to_numpy(float)
        pred = m["pred"].to_numpy(float)
        cube_pos, cube_pred = F.to_cube(mw), F.to_cube(pred)
        yrs = sorted(g["year"].unique())
        st = None
        ideal = lead_desc.get(name, name)
        # settings inferred from the exported positions, so neighbours on the grid can be scored
        if name.startswith("A_spike") and name != "A_spike_deep_rolling_cut":
            st = mk_spike_rule(name, LEAD_IDEA.get(name, "lead"), ideal, "pred", cube_pred)
            inferred = {}
            for Y in YEARS:
                sel = F.YEAR == Y
                hits = [ps for ps in PSTAR_GRID if np.array_equal(np.nan_to_num(mw[sel]), st.pos(F, st.inputs, ps)[sel])]
                inferred[Y] = hits[0] if hits else None
            if any(v is None for v in inferred.values()):
                st = None
        elif name in ("storm_day_filter", "storm_day_flip") and storm is not None:
            flip = name == "storm_day_flip"

            def pos_s(F, inp, c, flip=flip):
                return np.where(F.rows(inp["storm"] > c), 1.0 if flip else 0.0, -1.0)
            st = Strat(name, LEAD_IDEA[name], ideal, {"storm": storm}, pos_s, list(CUTS), DEFAULT_RANK_CUT)
            inferred = {}
            for Y in YEARS:
                sel = F.YEAR == Y
                hits = [c for c in CUTS if np.array_equal(np.nan_to_num(mw[sel]), st.pos(F, st.inputs, c)[sel])]
                inferred[Y] = hits[0] if hits else None
            if any(v is None for v in inferred.values()):
                st = None
        if st is not None:
            st.chooser = (lambda inf: (lambda F, inp, Y, gap, yr: (inf[Y], "the lead's choice")))(inferred)
        else:
            def pos_fixed(F, inp, p):
                return np.nan_to_num(F.rows(inp["pos"]))
            st = Strat(name, LEAD_IDEA.get(name, "lead"), ideal, {"pos": cube_pos}, pos_fixed)
        st.fixed_choices = True
        st.source = "lead modeller (model/evaluate.py positions)"
        st.lookahead_inputs = {"pass": None, "note": "inputs and choices are the lead's (timing test and the "
                               "real-data noise test in model/); the lab tests only the rule layer"}
        missing = [Y for Y in YEARS if Y not in yrs]
        if missing:
            st.incomplete = f"no positions for {', '.join(map(str, missing))}"
        st.lead_totals = tot["totals"].get(name, {})
        st.name = f"{name} [lead]" if name in ("storm_day_filter", "storm_day_flip", "A_spike_tail_S100_gbm") else name
        out.append(st)
    same = {}
    for lname, lab_name in LEAD_SAME_AS_LAB.items():
        same[lab_name] = tot["totals"].get(lname, {})
    notes["lead_export"] = {"strategies": int(d["strategy"].nunique()), "rows": len(d),
                            "same_strategies_scored_once": LEAD_SAME_AS_LAB}
    return out, same


# ================================================================== idea 14 (model/deep_alloc14.py)
IDEA14 = OPT / "idea14_wf_2021_2023.parquet"
IDEA14_CANDS = ("C_deep", "B_gbm", "C_gbm", "B_deep")


def idea14_strats(F, notes, pending):
    """The input (the score the placebos shift and shuffle) is the gate's day weights [ND, 4]; the candidates'
    exported MW and the refit's prior-only scales are fixed: position = sum_c w_c * scale_c * mw_c."""
    name = "idea14_learned_allocator_candidates"
    if not IDEA14.exists() or not LEAD_POS.exists():
        pending[name] = f"waiting for {IDEA14}"
        return []
    d = read_build(IDEA14, "delivery_date")
    d["delivery_date"] = pd.to_datetime(d["delivery_date"])
    assert_build(d["delivery_date"], "idea14")
    d = d.set_index("delivery_date").reindex(F.days)
    W = d[[f"w_{c}" for c in IDEA14_CANDS]].to_numpy(float)
    S = np.nan_to_num(d[[f"scale_{c}" for c in IDEA14_CANDS]].to_numpy(float))[F.DI]
    lp = read_build(LEAD_POS, "delivery_hour", ["delivery_hour", "zone", "mw", "strategy"],
                    ds.field("strategy").isin(list(IDEA14_CANDS)))
    key = F.h[["delivery_hour", "zone"]]
    Hm = np.stack([np.nan_to_num(key.merge(lp[lp["strategy"] == c][["delivery_hour", "zone", "mw"]],
                                           on=["delivery_hour", "zone"], how="left", validate="one_to_one")["mw"]
                                 .to_numpy(float)) for c in IDEA14_CANDS], 1)

    def pos(F, inp, p):
        return (np.nan_to_num(inp["w"])[F.DI] * S * Hm).sum(1)
    meta = {"rows_in_own_month_refit": bool((pd.to_datetime(d["refit_month"].dropna())
                                             == d["refit_month"].dropna().index.to_period("M").start_time).all())}
    jf = IDEA14.with_suffix(".json")
    if jf.exists():
        recs = [r for r in json.loads(jf.read_text()).get("refits", []) if "train_last" in r]
        meta["train_last_at_least_2_days_before_month"] = all(
            pd.Timestamp(r["train_last"]) <= pd.Timestamp(r["month"]) - pd.Timedelta(days=2) for r in recs)
        meta["kappa_choice"] = {k: v.get("kappa") for k, v in json.loads(jf.read_text()).get("kappa_choice", {}).items()}
    notes["idea14_file"] = meta
    la = {"pass": tri(meta.get("rows_in_own_month_refit"), meta.get("train_last_at_least_2_days_before_month")),
          "checked": "refit dates in the idea 14 file and json; feature and price timing is model/test_deep_alloc14.py",
          **meta}
    st = Strat(name, "14", "A deep gate splits one unit of risk per day between C deep, B gbm, C gbm and B deep, "
               "each pre-scaled to a 100,000 USD drawdown with scales from data before each refit; trained on "
               "earlier days' realised profits, pulled toward equal weights (registered after the table was seen).",
               {"w": W}, pos, source="model/deep_alloc14.py (idea14_wf_2021_2023)", lookahead_inputs=la)
    st.notes = [f"days with weights: {dict((str(Y), int(np.isfinite(W[F.day_year == Y]).all(1).sum())) for Y in YEARS)}"]
    return [st]


# ================================================================== the report
RECON_NOTES: dict = {           # strategy or (strategy, year) -> reason for a difference with the lead
    "C_gbm": "Same gross profit. Where two pairs put opposite legs in the same zone-hour, the lab nets them into one "
             "position before charging costs, so it pays fewer fees than the lead's leg-by-leg ledger.",
    "C_deep": "Same gross profit. Where two pairs put opposite legs in the same zone-hour, the lab nets them into one "
              "position before charging costs, so it pays fewer fees than the lead's leg-by-leg ledger.",
}
RECON_TEXT: list = [
    "**Storm-day filter: lab 107,141 against the lead's 144,018.** Two things differ, separated in side/recon.py "
    "(recon.json). First, the score's inputs: the lab reads the GFS forecast as storm_value.py did (every value of "
    "either run that is public at 05:00); the lead reads one value per point and hour by the pipeline's three-day "
    "rule and takes the load forecast hour by hour. Both are legal at 05:00 (storm audit, section 1). The two scores "
    "are equal on 861 of 1,400 days, and the flag at 0.80 differs on 16 days. That accounts for 2021 (41,620 against "
    "39,476). Second, the 2022 cut. On the lead's own score, 0.80 wins only because the choice counts 31 December "
    "2021: 0.80 beat 0.90 on 2021 by 0.08 USD a day, about 28 USD over the year. The 2022 setting is first used at "
    "05:00 on 31 December 2021, before that day's prices are public, so that day has to be left out. Without it "
    "the cut is 0.90 and 2022 earns 45,535 instead of 89,357. The right figures are therefore 100,196 on the "
    "lead's score (39,476, 45,535 and 15,185) and 107,141 on storm_value's inputs. The lead's 144,018 rests on one "
    "day of lookahead in the cut choice and should read 100,196.",
    "**Storm-day flip: lab 320,492 against the lead's 306,733.** Here only the score's inputs differ: both choose "
    "0.80, 0.90 and 0.80, with or without 31 December. Built with the three-day GFS rule, the lab's score matches "
    "the lead in 2022 and 2023 to the dollar (282,065 and -49,753); 2021 still differs (76,325 against 74,421) "
    "through the remaining input details. Neither has lookahead. The lead's version follows the pipeline's weather "
    "rule, so 306,733 is the figure for the list.",
    "**Idea 9, 2022: lab 377 against the lead's -28,960.** The predictions are the same file; p* is not. The lab "
    "picks p* on the 2021 walk-forward predictions it actually traded (0.02 earned 14.59 USD a day, 0.05 lost 5.99); "
    "the lead picks it on a separate model trained once before 2021 and scored on 2021 (0.05 lost 12.11 a day, 0.02 "
    "lost 13.62). At p* = 0.05 the lab reproduces -28,960 exactly. Neither uses 2022 data. The lab's choice is "
    "closer to what would have been traded, but the main fact is that a near coin-flip between two cuts moves 2022 "
    "by 29,337, and idea 9 fails the bar in both versions (negative at the 0.50 stress).",
    "**Choice window.** Every lab choice for year Y now uses the year before only up to 30 December, because "
    "the setting is first used at 05:00 on 31 December. No lab choice changed because of it.",
    "**B starts on 1 May 2021** (its weather archive starts on 25 March). The lab counts 1 January to 30 April "
    "2021 as days without positions, so B's net is the lead's, while its 2021 and three-year Sharpe and return per "
    "year are lower than the lead's (which count only the days B could trade).",
]
ELLIOTT = ("Written check: Winter Storm Elliott. On 24 December 2022 real time settled on average 561 USD/MWh above "
           "day ahead, and the rule held virtual load in every zone-hour because the storm score was just above the "
           "0.90 cut (0.9014 in the lab's build, through the cold component alone). The storm audit (side/storm_audit.md) reproduced this to the "
           "dollar and found no lookahead; outside Elliott's two days the flip's load legs earned 8,515 USD over "
           "three years. The flag stands: the 2022 profit is one storm.")
TOO_GOOD_NOTES: dict = {        # (strategy, year) -> written check, after reading the diagnostics
    ("B_gbm", 2022): "Written check (side/b_audit.py, b_audit.json). Inputs: all 11,605 sampled hourly weather and "
                     "load-forecast values in B's training matrix (44 delivery days, Elliott included) equal a rebuild "
                     "from the raw tables at 05:00; with the filter opened to midnight only 398 of the load-forecast "
                     "values would still match. Placebos: one day late 92,197, dates shuffled -16,950 on average (none "
                     "of 20 at or above the real rule), random days in the same months never beat it. So the 2022 "
                     "result is read from data public at 05:00. It is still one exceptional year: December 2022 earned "
                     "300,208 and Elliott's two days 180,520 (on 24 December B held load in 261 of 264 zone-hours with "
                     "a predicted gap of +34 USD; the realised gap was +561). Without those two days 2022 earns "
                     "374,601. B's forecast correlates 0.12 with the hourly gap in 2022, 0.04 in 2021 and -0.03 in "
                     "2023, when its load legs lost 51,023 in 10 of 11 zones.",
    ("idea2_storm_flip", 2022): ELLIOTT,
    ("storm_day_flip [lead]", 2022): ELLIOTT,
    ("C_gbm", 2023): "Written check: the year earned only 4,403 USD, so an ordinary good pair day exceeds half of it. "
                     "The flag says 2023 was thin, not that the day is suspect.",
}
SUMMARY_NOTES: list = [
    "**Bottom line.** Seven rows meet Bo's bar at 1 MW: B with gradient boosting (583,946 over three years), idea 10 "
    "weather surprise (352,130), the storm-day flip in both builds (320,492 here, 306,733 in the lead's), B deep "
    "(310,453), C gradient boosting (169,889) and C deep (165,407). The two storm flips are also FRAGILE. Every "
    "passing row except C makes most of its money in 2022: Winter Storm Elliott's two days give B 180,520, idea 10 "
    "204,836 and the flip 204,836, and in 2023 B, the flip and B deep lose money while idea 10 earns less than "
    "always supply. B's inputs pass an independent rebuild at 05:00 and its date placebos earn far less than it "
    "does, so its 2022 is read from public data, but it is one year. Among the passing rows, only C is positive "
    "in all three years; its shuffled-date placebos earn most of what it earns (C deep: 142,531 of 165,407), so C's profit is a "
    "steady pair stance, not day-by-day timing.",
]


def _f(x, nd=0):
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    return f"{x:,.{nd}f}"


def write_md(out: dict, path: Path):
    S = out["strategies"]
    meta = out["meta"]
    order = sorted(S, key=lambda k: -S[k]["three_year"]["net_usd"])
    L = []
    a = L.append
    a("# Strategy lab: every strategy scored the same way on 2021 to 2023")
    a("")
    a(f"Written {meta['written']} on gene by `side/lab.py`. Build years only: every table and prediction file was "
      "filtered at load to delivery times before 1 January 2024 New York time, and every loader asserts it. "
      "Nothing was committed, pushed or sent.")
    a("")
    n_pass = sum(bool(S[k]["PASS"]) for k in S)
    n_pf = sum(bool(S[k]["PASS"]) and S[k]["FRAGILE"] for k in S)
    a(f"{n_pass} of {len(S)} scored strategies meet Bo's bar; {n_pf} of those {n_pass} are also flagged FRAGILE. "
      + (f"Still pending: {', '.join(meta['pending'])}." if meta["pending"] else "Nothing is pending."))
    for t in SUMMARY_NOTES:
        a("")
        a(t)
    a("")
    a("## Ranking by three-year net (USD, full costs, 1 MW per zone-hour)")
    a("")
    a("The bar (registered, unchanged): average at least 50,000 a year, positive in at least 2 of 3 years, three-year "
      "Sharpe above 0.42, positive three-year total at a cost of 0.50 per MWh. FRAGILE (coordinator's flag, shown "
      "beside PASS): the total changes sign at a neighbouring setting, or turns negative without one year's best 10 "
      "days, or random days in the same months beat the rule more than 20 percent of the time.")
    a("")
    a("| # | Strategy | From | 2021 | 2022 | 2023 | Three-year | Without best 5 days | Average a year | Sharpe, 3 years "
      "| Total at 0.50 stress | Placebo: score one day late | Placebo: dates shuffled (mean of 20; share at or above real) "
      "| Random days beat rule | PASS | FRAGILE |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, k in enumerate(order, 1):
        v = S[k]; y = v["years"]; pl = v["placebo"]; fr = v["fragility"]
        rd = fr["random_days"]
        rds = "n/a" if isinstance(rd, str) else f"{100 * rd['share_random_beats_rule']:.0f}%"
        shuf = "n/a" if pl.get("shuffled_mean_total") is None else \
            f"{_f(pl['shuffled_mean_total'])} ({pl['shuffled_share_at_or_above_real']:.2f})"
        src = v["source"].split(" ")[0] if v["source"] != "lab" else "lab"
        src = {"lead": "lead", "deep": "deep agent", "policy": "policy agent"}.get(src, src)
        ps = "not assessed" if v["PASS"] is None else ("PASS" if v["PASS"] else "no")
        a(f"| {i} | {k} | {src} | {_f(y['2021']['net_usd'])} | {_f(y['2022']['net_usd'])} | {_f(y['2023']['net_usd'])} "
          f"| {_f(v['three_year']['net_usd'])} | {_f(v['three_year']['net_without_best_5_days'])} "
          f"| {_f(v['avg_net_per_year'])} | {_f(v['three_year']['sharpe'], 2)} "
          f"| {_f(v['three_year']['stress_0.50_usd'])} | {_f(pl.get('shifted_one_day_total'))} | {shuf} | {rds} "
          f"| {ps} | {'FRAGILE' if v['FRAGILE'] else 'no'} |")
    a("")
    a("## What each strategy does")
    a("")
    for k in order:
        inc = f" Incomplete: {S[k]['incomplete']}; not assessed against the bar." if S[k].get("incomplete") else ""
        a(f"- **{k}** (idea {S[k]['idea']}; from {S[k]['source']}). {S[k]['line']}{inc}")
    for k, why in meta["pending"].items():
        a(f"- **{k}**: not scored, {why}.")
    a("")
    a("## Each year")
    a("")
    a("Return is net over the 500,000 USD bankroll. Sharpe is daily, times the square root of 365, with zero days "
      "included. Drawdown runs from the start of the year.")
    a("")
    a("| Strategy | Year | Net | Return | Sharpe | Max drawdown | Worst day | Best day | Without best 3 days "
      "| Without best 5 days | Without best 10 days | MWh a day | At 0.50 stress |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in order:
        v = S[k]
        for Y in YEARS:
            y = v["years"][str(Y)]
            b10 = v["fragility"]["without_best_10_days"][str(Y)]["net_without_best_10"]
            a(f"| {k} | {Y} | {_f(y['net_usd'])} | {y['return_on_500k_pct']}% | {_f(y['sharpe'], 2)} "
              f"| {_f(y['max_drawdown_usd'])} | {_f(y['worst_day_usd'])} ({y['worst_day']}) "
              f"| {_f(y['best_day_usd'])} ({y['best_day']}) | {_f(y['net_without_best_3_days'])} "
              f"| {_f(y['net_without_best_5_days'])} | {_f(b10)} "
              f"| {y['mwh_per_day']} | {_f(y['stress_0.50_usd'])} |")
        t = v["three_year"]
        a(f"| {k} | 2021 to 2023 | {_f(t['net_usd'])} | {v['avg_return_on_500k_pct']}% a year | {_f(t['sharpe'], 2)} "
          f"| {_f(t['max_drawdown_usd'])} | {_f(t['worst_day_usd'])} ({t['worst_day']}) "
          f"| {_f(t['best_day_usd'])} ({t['best_day']}) | {_f(t['net_without_best_3_days'])} "
          f"| {_f(t['net_without_best_5_days'])} | | {t['mwh_per_day']} "
          f"| {_f(t['stress_0.50_usd'])} |")
    a("")
    a("## Fragility")
    a("")
    a("Neighbours: each year's chosen setting moved one step on its grid, in every year at once. The bootstrap is "
      "stationary over days (mean block 7, 5,000 draws) on the three-year daily net, shown per year. Random days: "
      "the rule's own daily books dealt to random days of the same month, 1,000 draws; percentile of the real rule.")
    a("")
    a("| Strategy | Settings chosen 2021 / 2022 / 2023 | Three-year total at neighbouring settings | 95% interval, "
      "USD a year | Random days: percentile of rule | Without each year's best 10 days, three-year total | FRAGILE because |")
    a("|---|---|---|---|---|---|---|")
    for k in order:
        v = S[k]; fr = v["fragility"]
        ch = " / ".join(_set(v["choices"][str(Y)]["setting"]) for Y in YEARS)
        nb = fr["neighbours"]
        nbs = nb if isinstance(nb, str) else ("; ".join(f"{lab} {_f(x['three_year'])}" for lab, x in nb.items()) or "none")
        rd = fr["random_days"]
        rds = rd if isinstance(rd, str) else f"{rd['percentile_of_rule']}"
        b10 = " / ".join(_f(fr["without_best_10_days"][str(Y)]["three_year_total_without_this_years_best_10"]) for Y in YEARS)
        ci = fr["bootstrap_ci95_per_year_usd"]
        a(f"| {k} | {ch} | {nbs} | {_f(ci[0])} to {_f(ci[1])} | {rds} | {b10} | {'; '.join(fr['fragile_reasons']) or 'no'} |")
    a("")
    a("## Sizing view (not part of the registered verdict)")
    a("")
    a("From the 2021 to 2023 walk-forward only. Sortino uses the downside deviation of daily net (zero days "
      "included). Return over max drawdown = average net a year over the three-year max drawdown. Scale = the MW per "
      "position that would make the three-year max drawdown 100,000 USD (20 percent of the bankroll), capped at 5; "
      "profit, costs and drawdown all scale linearly with it. This is a sizing view only; PASS above is computed at "
      "1 MW as registered.")
    a("")
    a("| Strategy | Sharpe | Sortino | Return over max drawdown | Max drawdown at 1 MW | Scale for a 100,000 drawdown "
      "| Return on 500,000 a year at that scale |")
    a("|---|---|---|---|---|---|---|")
    for k in order:
        z = S[k]["sizing_view"]
        a(f"| {k} | {_f(z['sharpe'], 2)} | {_f(z['sortino'], 2)} | {_f(z['return_over_max_drawdown'], 2)} "
          f"| {_f(S[k]['three_year']['max_drawdown_usd'])} | {z['scale_for_100k_drawdown']} "
          f"| {z['return_on_500k_at_that_scale_pct']}% |")
    a("")
    a("## Against the lead modeller's list")
    a("")
    a("The lead's strategies are scored here from their own hourly positions (model/evaluate.py ledgers, exported "
      "read-only by side/lead_export.py), so every row has the same columns. Every year that differs from the "
      "lead's own figure by more than 1 percent is listed with the reason.")
    a("")
    a("| Strategy | Year | Lab | Lead | Difference | Why |")
    a("|---|---|---|---|---|---|")
    nd = 0
    for k in order:
        for Y, x in (S[k].get("vs_lead") or {}).items():
            if abs(x["diff_pct"]) > 1:
                nd += 1
                a(f"| {k} | {Y} | {_f(x['lab'])} | {_f(x['lead'])} | {x['diff_pct']:+.1f}% "
                  f"| {RECON_NOTES.get((k, Y), RECON_NOTES.get(k, ''))} |")
    if not nd:
        a("| none | | | | | |")
    for t in RECON_TEXT:
        a("")
        a(t)
    a("")
    a("## Safeguards")
    a("")
    a("**(a) Injected lookahead.** Inputs: values published after 05:00 on D are corrupted (load forecast times 3, "
      "temperatures plus 40 C, real-time prices plus 1,000, a fake newer GFS run at 60 C, labels after the training "
      "cut scrambled). With the 05:00 filter the decision for D+1 must not change; with the filter opened to "
      "midnight it must. Rule: corrupting every later day's score never changes today's positions. Choice: "
      "scrambling the outcomes of the scored year and after never changes the setting chosen for it; choosing on "
      "the scored year itself does change settings (shown as a count of years).")
    a("")
    a("| Strategy | Input test | Decisions changed, filter at 05:00 | Decisions changed, filter opened | Rule test "
      "| Choice test | Years whose setting moves if chosen in-sample | Pass |")
    a("|---|---|---|---|---|---|---|---|")
    for k in order:
        v = S[k]; li = v.get("lookahead_inputs", {}); lr = v["lookahead_rule_and_choice"]
        if "decisions_changed_with_filter" in li:
            f05, fo, kind = li["decisions_changed_with_filter"], li["decisions_changed_filter_opened_corrupt"], \
                f"raw tables, {li['test_days']} days"
        elif "zone_hour_positions_changed_with_filter" in li:
            f05, fo, kind = li["zone_hour_positions_changed_with_filter"], li["zone_hour_positions_changed_filter_opened"], \
                f"raw tables, {li['test_days']} days (zone-hours)"
        elif "months" in li:
            f05 = "max change " + _f(max(m["max_abs_change_with_cut"] for m in li["months"]), 3)
            fo = "max change " + _f(max(m["max_abs_change_opened"] for m in li["months"]), 3)
            kind = "label window, 3 months"
        elif "checked" in li:
            f05, fo, kind = "n/a", "n/a", "refit dates checked: " + ("yes" if li.get("pass") else "could not confirm")
        else:
            f05, fo, kind = "n/a", "n/a", li.get("note", "n/a")
        lp = li.get("pass", True)
        ok = "FAIL" if (lp is False or lr.get("pass", True) is False) else \
            ("pass (rule, choice); inputs not tested here" if lp is None else "pass")
        a(f"| {k} | {kind} | {f05} | {fo} | {lr.get('rule_positions_changed_by_later_scores', 'n/a')} changed "
          f"| {lr.get('choices_changed_by_corrupting_scored_year_outcomes', 'n/a')} changed "
          f"| {lr.get('choices_changed_when_chosen_on_scored_year', 'n/a')} | {ok} |")
    a("")
    a("**(b) Placebos** are in the ranking table: the same rule and walk-forward fed its score one day late, and fed "
      "its score with dates permuted within each year (20 permutations; the share of them at or above the real "
      "rule is in brackets). A rule whose placebos earn about what it earns is not reading the day.")
    a("")
    a("**(c) Too good.** A year is flagged when its Sharpe is above 3 or one day earns more than half its profit.")
    a("")
    any_tg = False
    for k in order:
        for tg in S[k]["too_good_checks"]:
            any_tg = True
            note = TOO_GOOD_NOTES.get((k, tg["year"]), "")
            if not note and "best day" in tg["flag"] and tg["year_net_usd"] < 25_000 \
                    and tg["best_day_usd"] <= tg["always_supply_that_day_usd"] + 1:
                note = ("Written check: a small year, not an outsized day. On that day the strategy earned no more "
                        "than always supply did, from ordinary supply positions; nothing about the day suggests "
                        "lookahead.")
            a(f"- {k}, {tg['year']}: {tg['flag']}. Year without that day: {_f(tg['year_net_without_best_day'])}. "
              f"That day the mean real-time minus day-ahead gap was {tg['mean_gap_that_day_usd_per_mwh']} USD/MWh "
              f"and always supply made {_f(tg['always_supply_that_day_usd'])}. {note}")
    if not any_tg:
        a("- No year is flagged.")
    a("")
    a("## How many strategies were tried")
    a("")
    n_rows, pend = meta["strategies_scored"], meta["pending"]
    reg_scored = 19 - len(pend)
    a(f"Registered so far: ideas 1 to 12, the three basic strategies and ideas A to D, 19 strategies. {reg_scored} "
      f"of them are scored in this table" + (f"; {len(pend)} are pending ({', '.join(pend)})." if pend else ".")
      + f" The table also has {n_rows - reg_scored} further rows: the deep-model versions of A to D (the registered "
      "secondary comparison), A's first rule (A v1) in both models, the lead's builds of ideas 1, 2 and 9 beside the "
      "lab's, and four side rules (risk-sized supply, the zone subset, the deep rolling cut, the past spike rate "
      f"filter). Choices inside the rows searched {meta['settings_searched']} grid settings in total, counting each "
      "grid value of each row once. " + meta.get("tried_before", "").replace(REGISTERED, ""))
    a("")
    a("## Notes")
    a("")
    for k, v in out["notes"].items():
        if k in ("idea7_gbm",):
            v = {kk: vv for kk, vv in v.items() if kk != "refits"}
        a(f"- {k}: `{json.dumps(v, default=str)[:600]}`")
    path.write_text("\n".join(L) + "\n")


def _set(x):
    if x is None:
        return "none"
    if isinstance(x, (list, tuple)):
        return "(" + ", ".join(_set(y) for y in x) + ")"
    return str(x)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "md":          # rebuild the report from the saved JSON only
        write_md(json.loads((SIDE / f"lab_results{OUT_TAG}.json").read_text()), SIDE / f"lab_results{OUT_TAG}.md")
    else:
        main()
