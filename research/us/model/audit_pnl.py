"""Independent second implementation of the 2023 rehearsal P&L (audit written 6 Oct 2026).

Recomputes, from the zone price table and the saved predictions, every zone-hour position, the daily
P&L and the headline numbers that rehearsal.py writes to results/rehearsal_2023.json, then lists every
difference over 0.5 percent. It imports nothing from the official model code (fees, strategies, score,
panel, walkforward, rehearsal): the gap, the trailing 365-day baseline signal, the fee, the positions,
the day calendar, the bootstrap, Holm and the verdict words are written again here from CONTRACT.md
and OBJECTIVES.md, in a different form where that is possible:
  * idea C is booked as two explicit legs (load in i, supply in j), each with its own fee;
  * days come from a fixed calendar, not from the rows present;
  * the stationary bootstrap draws geometric block lengths instead of a per-day restart coin, so its
    intervals differ from the official ones by Monte Carlo noise only; the noise is measured with
    several seeds and reported beside every interval difference.

HOLDOUT: every read is filtered inside pyarrow to delivery hours before 2024-01-01 New York time and
checked again after reading; any later date raises HoldoutError. Nothing for 2024 onward is computed.

    python audit_pnl.py --prices prices_zone_2022_2023.parquet --preds preds_gbm_2023.parquet \\
        --official rehearsal_2023.json [--panel panel_2023.parquet] \\
        [--deep-base f] [--deep-weather f] [--deep-outages f] --out audit_2023.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

NY = "America/New_York"
HOLDOUT = pd.Timestamp("2024-01-01", tz=NY)
YEAR = 2023
# NYISO "2023 Schedule One Posting": "Virtual Resources $0.1066 per cleared MWh" (the only year scored here)
FEE = {2022: 0.0853, 2023: 0.1066}
STRESS = (0.50, 1.00)
ZONES = ("CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL", "LONGIL", "MHK VL", "MILLWD", "N.Y.C.", "NORTH", "WEST")
ZCODE = {z: i for i, z in enumerate(ZONES)}
N_BOOT, MEAN_BLOCK, ALPHA = 10_000, 7, 0.05
SECONDARY = ("B", "C", "D")
WORDS = ("pays", "doesn't pay", "inconclusive", "not run")
TOL = 0.005
SEEDS = tuple(range(101, 111))                    # ten bootstrap streams for the Monte Carlo band
PRED_OF = {"A": "base", "B": "weather", "C": "base", "D": "outages", "A_gen": "gen"}


class HoldoutError(RuntimeError):
    pass


# ----------------------------------------------------------------- holdout-safe reading

def guard(values) -> None:
    """Raise if any value is a delivery time or date on or after 2024-01-01 New York time."""
    s = values if isinstance(values, pd.Series) else pd.Series(values)
    if not len(s):
        return
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        late = s.dt.tz_convert(NY) >= HOLDOUT
    else:
        late = pd.to_datetime(s) >= HOLDOUT.tz_localize(None)
    if bool(late.any()):
        raise HoldoutError(f"{int(late.sum())} value(s) on or after 2024-01-01; the audit never reads the holdout")


def read_pre2024(path, col: str = "delivery_hour", columns: list[str] | None = None,
                 start: pd.Timestamp | None = None) -> pd.DataFrame:
    d = ds.dataset(str(path), format="parquet")
    typ = d.schema.field(col).type

    def bound(ts: pd.Timestamp):
        return pa.scalar(ts if typ.tz else ts.tz_convert(NY).tz_localize(None), type=typ)
    flt = ds.field(col) < bound(HOLDOUT)
    if start is not None:
        flt = flt & (ds.field(col) >= bound(start))
    if columns is not None:
        columns = [c for c in columns if c in d.schema.names]
    df = d.to_table(filter=flt, columns=columns).to_pandas()
    guard(df[col])
    return df


# ----------------------------------------------------------------- zone-hours, gap, trailing signal

def _utc64(s: pd.Series) -> np.ndarray:
    return s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]")


def hours_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """One row per (delivery hour instant, load zone): gap = rt - da, local date and local clock hour.
    The two 01:00 hours of a fall-back day are distinct instants and stay two rows."""
    p = prices[prices["zone"].isin(ZONES)].copy()
    guard(p["delivery_hour"])
    if p.duplicated(["delivery_hour", "zone"]).any():
        raise ValueError(f"{int(p.duplicated(['delivery_hour', 'zone']).sum())} duplicated zone-hours in prices")
    loc = p["delivery_hour"].dt.tz_convert(NY)
    p["ldate"] = loc.dt.tz_localize(None).dt.normalize()
    p["lhour"] = loc.dt.hour.astype(int)
    p["zc"] = p["zone"].map(ZCODE).astype(int)
    p["gap"] = p["rt_lbmp"].astype(float) - p["da_lbmp"].astype(float)
    return p.sort_values(["delivery_hour", "zc"], kind="stable").reset_index(drop=True)


def decision(bid_day: dt.date) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime(bid_day.year, bid_day.month, bid_day.day, 5, 0), tz=NY)


def trailing_signal(H: pd.DataFrame, delivery_days) -> np.ndarray:
    """Baseline signal per row of H for the given delivery days (NaN elsewhere): mean gap of the same
    zone and local clock hour over rows public at 05:00 on the bid day D (both the day-ahead and the
    real-time price published by then) whose delivery hour is within the 365 days before 05:00 on D."""
    dh = _utc64(H["delivery_hour"])
    dap, rtp = _utc64(H["da_published_at"]), _utc64(H["rt_published_at"])
    gap = H["gap"].to_numpy(float)
    key = (H["zc"].to_numpy() * 24 + H["lhour"].to_numpy()).astype(np.int64)
    ld = H["ldate"].to_numpy()
    out = np.full(len(H), np.nan)
    for d in pd.DatetimeIndex(delivery_days):
        t = decision((d - pd.Timedelta(days=1)).date())
        t64 = np.datetime64(t.tz_convert("UTC").tz_localize(None), "ns")
        lo64 = np.datetime64((t - pd.Timedelta(days=365)).tz_convert("UTC").tz_localize(None), "ns")
        i0 = np.searchsorted(dh, lo64, "left")
        i1 = np.searchsorted(dh, t64 + np.timedelta64(4, "D"), "left")   # reach past t: the filter decides
        sl = slice(i0, i1)
        ok = (dap[sl] <= t64) & (rtp[sl] <= t64) & np.isfinite(gap[sl])
        k = key[sl][ok]
        s = np.bincount(k, weights=gap[sl][ok], minlength=len(ZONES) * 24)
        c = np.bincount(k, minlength=len(ZONES) * 24)
        mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        rows = np.flatnonzero(ld == np.datetime64(d.to_datetime64()))
        out[rows] = mean[key[rows]]
    return out


# ----------------------------------------------------------------- positions and money

def fee_rows(ldate: pd.Series | np.ndarray, override: float | None = None) -> np.ndarray:
    years = pd.DatetimeIndex(ldate).year
    if override is not None:
        return np.full(len(years), float(override))
    missing = sorted(set(years) - set(FEE))
    if missing:
        raise KeyError(f"no audited fee for {missing}")
    return np.array([FEE[y] for y in years], float)


def book(pos: np.ndarray, gap: np.ndarray, fee: np.ndarray) -> dict:
    """1 MW per non-zero position. load (+1) earns gap - fee, supply (-1) earns -gap - fee, none 0.
    An hour without a settled gap is not traded and not charged."""
    traded = (pos != 0) & np.isfinite(gap)
    g = np.where(traded, gap, 0.0)
    p = np.where(traded, pos, 0.0)
    return {"pnl": p * g - np.abs(p) * fee, "gross": p * g, "mwh": np.abs(p)}


def pos_baseline(sig: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(sig), np.sign(sig), 0.0)


def pos_supply_only(pred: np.ndarray, fee: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(pred) & (pred <= -fee), -1.0, 0.0)


def pos_two_sided(pred: np.ndarray, fee: np.ndarray) -> np.ndarray:
    ok = np.isfinite(pred)
    return np.where(ok & (pred <= -fee), -1.0, np.where(ok & (pred >= fee), 1.0, 0.0))


def zone_ledger(T: pd.DataFrame, pos: np.ndarray, fee: np.ndarray) -> pd.DataFrame:
    b = book(pos, T["gap"].to_numpy(float), fee)
    return pd.DataFrame({"ldate": T["ldate"].to_numpy(), "zone": T["zone"].to_numpy(), **b,
                         "absgap": np.abs(np.nan_to_num(T["gap"].to_numpy(float)))})


def pair_ledger(T: pd.DataFrame, pred: np.ndarray, pairs: list[tuple[str, str]], fee_override=None) -> pd.DataFrame:
    """Idea C as two legs per pair-hour: s = +1 is load in i and supply in j, s = -1 the reverse.
    Each leg is booked on its own zone's gap with its own fee."""
    W = T.assign(pred=pred)
    gw = W.pivot(index="delivery_hour", columns="zone", values="gap")
    pw = W.pivot(index="delivery_hour", columns="zone", values="pred")
    ld = gw.index.tz_convert(NY).tz_localize(None).normalize()
    fee = fee_rows(ld, fee_override)
    parts = []
    for i, j in pairs:
        gi, gj = gw[i].to_numpy(float), gw[j].to_numpy(float)
        d = pw[i].to_numpy(float) - pw[j].to_numpy(float)
        s = np.where(np.isfinite(d) & (d >= 2 * fee), 1.0, np.where(np.isfinite(d) & (d <= -2 * fee), -1.0, 0.0))
        s = np.where(np.isfinite(gi) & np.isfinite(gj), s, 0.0)
        li, lj = book(s, gi, fee), book(-s, gj, fee)
        parts.append(pd.DataFrame({"ldate": ld, "zone": f"{i}|{j}",
                                   "pnl": li["pnl"] + lj["pnl"], "gross": li["gross"] + lj["gross"],
                                   "mwh": li["mwh"] + lj["mwh"],
                                   "absgap": np.fmax(np.abs(gi), np.abs(gj))}))
    if not parts:
        return pd.DataFrame(columns=["ldate", "zone", "pnl", "gross", "mwh", "absgap"])
    out = pd.concat(parts, ignore_index=True)
    out["absgap"] = out["absgap"].fillna(0.0)
    return out


def ledgers(T: pd.DataFrame, sig: np.ndarray, preds: dict, pairs, fee_override=None) -> dict:
    fee = fee_rows(T["ldate"], fee_override)
    L = {"baseline": zone_ledger(T, pos_baseline(sig), fee)}
    if "base" in preds:
        L["A"] = zone_ledger(T, pos_supply_only(preds["base"], fee), fee)
        L["C"] = pair_ledger(T, preds["base"], pairs, fee_override)
    if "weather" in preds:
        L["B"] = zone_ledger(T, pos_two_sided(preds["weather"], fee), fee)
    if "outages" in preds:
        L["D"] = zone_ledger(T, pos_two_sided(preds["outages"], fee), fee)
    if "gen" in preds:
        L["A_gen"] = zone_ledger(T, pos_supply_only(preds["gen"], fee), fee)
    return L


# ----------------------------------------------------------------- days, bootstrap, words

def calendar(year: int = YEAR) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")


def per_day(L: pd.DataFrame, days: pd.DatetimeIndex, col: str = "pnl") -> np.ndarray:
    """Sum over each calendar day; a day without a trade is a zero, never dropped."""
    if not len(L):
        return np.zeros(len(days))
    pos = days.get_indexer(pd.DatetimeIndex(L["ldate"]))
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} ledger rows fall outside the scored days")
    return np.bincount(pos, weights=L[col].to_numpy(float), minlength=len(days))


def boot_index(n: int, n_boot: int = N_BOOT, mean_block: float = MEAN_BLOCK, seed: int = SEEDS[0]) -> np.ndarray:
    """Stationary bootstrap (Politis and Romano 1994), circular: blocks of geometric length (mean
    `mean_block` days) starting at uniformly drawn days, laid end to end until n days are filled."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, n), dtype=np.int64)
    ar = np.arange(n)
    for b in range(n_boot):
        L = rng.geometric(1.0 / mean_block, n)
        S = rng.integers(0, n, n)
        c = np.cumsum(L)
        k = int(np.searchsorted(c, n, "left")) + 1
        L = L[:k].copy()
        L[-1] -= int(c[k - 1] - n)
        first = np.concatenate([[0], np.cumsum(L)[:-1]])
        out[b] = (np.repeat(S[:k], L) + ar - np.repeat(first, L)) % n
    return out


def bmeans(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.asarray(x, float)[idx].mean(axis=1)


def pct_interval(bm: np.ndarray, level: float) -> list[float]:
    a = (1.0 - level) / 2.0
    return [float(np.quantile(bm, a)), float(np.quantile(bm, 1.0 - a))]


def pval(bm: np.ndarray) -> float:
    n = len(bm)
    return float(min(1.0, 2.0 * min((np.sum(bm <= 0) + 1) / (n + 1), (np.sum(bm >= 0) + 1) / (n + 1))))


def word(lo: float, hi: float, reached: bool = True) -> str:
    if not reached:
        return "inconclusive"
    return "pays" if lo > 0 else ("doesn't pay" if hi < 0 else "inconclusive")


def holm_levels(p: dict[str, float]) -> dict[str, dict]:
    keys = sorted(p, key=lambda k: (p[k], k))
    m, out, alive = len(keys), {}, True
    for r, k in enumerate(keys, start=1):
        a = ALPHA / (m - r + 1)
        rej = alive and p[k] < a
        out[k] = {"rank": r, "alpha": a, "level": 1 - a, "rejected": rej, "reached": alive}
        alive = rej
    return out


# ----------------------------------------------------------------- summaries

def money(L: pd.DataFrame) -> dict:
    mwh, pnl, gross = float(L["mwh"].sum()), float(L["pnl"].sum()), float(L["gross"].sum())
    return {"pnl_usd": round(pnl, 2), "mwh": mwh, "profit_per_mwh": pnl / mwh if mwh else None,
            "break_even_fee": gross / mwh if mwh else None}


def groups(L: pd.DataFrame, key) -> dict:
    out = {}
    for k, g in L.groupby(key):
        m = float(g["mwh"].sum())
        out[str(k)] = {"pnl_usd": round(float(g["pnl"].sum()), 2), "mwh": m,
                       "profit_per_mwh": float(g["pnl"].sum()) / m if m else None}
    return out


def summary(L: pd.DataFrame, Lb: pd.DataFrame, days: pd.DatetimeIndex, idx: np.ndarray, cut: float) -> dict:
    di, db = per_day(L, days), per_day(Lb, days)
    diff = di - db
    bm = bmeans(diff, idx)
    months = np.asarray(days.strftime("%Y-%m"))
    mi = pd.Series(di).groupby(months).sum()
    md = pd.Series(diff).groupby(months).sum()
    keep, keepb = L[L["absgap"] <= cut], Lb[Lb["absgap"] <= cut]
    dd = per_day(keep, days) - per_day(keepb, days)
    traded = per_day(L, days, "mwh") > 0
    return {"n_days": int(len(days)), "mean_daily_idea": float(di.mean()), "mean_daily_baseline": float(db.mean()),
            "mean_daily_diff": float(diff.mean()), "interval_95": pct_interval(bm, 0.95), "_bm": bm,
            "p_two_sided": pval(bm), "se_daily_diff": float(bm.std(ddof=1)),
            "idea_alone_interval_95": pct_interval(bmeans(di, idx), 0.95),
            "idea_money": money(L), "baseline_money": money(Lb), "mwh_per_day": float(L["mwh"].sum() / len(days)),
            "worst_month_idea": {"month": str(mi.idxmin()), "pnl_usd": float(mi.min())},
            "worst_month_diff": {"month": str(md.idxmin()), "usd": float(md.min())},
            "months_positive_idea": int((mi > 0).sum()), "months_positive_diff": int((md > 0).sum()),
            "months": int(len(mi)),
            "monthly": {k: {"idea_usd": round(float(mi[k]), 2), "diff_usd": round(float(md[k]), 2)} for k in mi.index},
            "per_zone": groups(L, L["zone"]), "per_year": groups(L, pd.DatetimeIndex(L["ldate"]).year),
            "without_extreme_1pct": {"abs_gap_cut_usd": float(cut), "mean_daily_diff": float(dd.mean()),
                                     "interval_95": pct_interval(bmeans(dd, idx), 0.95),
                                     "idea_money": money(keep), "baseline_money": money(keepb)},
            "days_without_trade": int((~traded).sum()),
            "mean_daily_idea_if_no_trade_days_were_dropped": float(di[traded].mean()) if traded.any() else None}


def score(T: pd.DataFrame, sig: np.ndarray, preds: dict, pairs, days: pd.DatetimeIndex, idx: np.ndarray) -> dict:
    cut = float(np.quantile(np.abs(T["gap"].dropna().to_numpy(float)), 0.99))
    L = ledgers(T, sig, preds, pairs)
    ideas = {k: summary(L[k], L["baseline"], days, idx, cut) for k in ("A", "B", "C", "D") if k in L}
    if "A" in ideas:
        a = ideas["A"]
        a["interval_level"], a["interval"] = 0.95, pct_interval(a["_bm"], 0.95)
        a["verdict"] = word(*a["interval"])
    h = holm_levels({k: ideas[k]["p_two_sided"] for k in SECONDARY if k in ideas})
    for k, info in h.items():
        r = ideas[k]
        r["holm"] = info
        r["interval_level"] = info["level"]
        r["interval"] = pct_interval(r["_bm"], info["level"])
        r["verdict"] = word(*r["interval"], reached=info["reached"])
    for k in SECONDARY:
        ideas.setdefault(k, {"verdict": "not run"})
    stress = {}
    for f in STRESS:
        Ls = ledgers(T, sig, preds, pairs, fee_override=f)
        stress[str(f)] = {}
        for k in Ls:
            if k == "baseline":
                continue
            d = per_day(Ls[k], days) - per_day(Ls["baseline"], days)
            stress[str(f)][k] = {"mean_daily_diff": float(d.mean()), "interval_95": pct_interval(bmeans(d, idx), 0.95),
                                 "idea_money": money(Ls[k]), "baseline_money": money(Ls["baseline"])}
    return {"scored_days": int(len(days)), "extreme_cut_abs_gap_usd": cut,
            "baseline": {**money(L["baseline"]), "mean_daily": float(per_day(L["baseline"], days).mean())},
            "ideas": ideas, "fee_stress": stress, "_ledgers": L}


def deep_vs_gbm(Ld: pd.DataFrame, Lg: pd.DataFrame, days: pd.DatetimeIndex, idx: np.ndarray) -> dict:
    d = per_day(Ld, days) - per_day(Lg, days)
    lo, hi = pct_interval(bmeans(d, idx), 0.95)
    margin = float(fee_rows(days).mean()) * float(Lg["mwh"].sum() / len(days))
    w = ("equivalent" if (-margin <= lo and hi <= margin) else
         "better" if lo > 0 else "worse" if hi < 0 else "inconclusive")
    return {"mean_daily_diff": float(d.mean()), "interval_95": [lo, hi], "margin_usd_per_day": margin, "verdict": w}


# ----------------------------------------------------------------- comparison

MC_FIELDS = ("interval", "p_two_sided", "se_daily_diff", "mde")


def flat(x, path="") -> dict:
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            if str(k).startswith("_") or k in ("calibration", "boot_means"):
                continue
            out.update(flat(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(x, (list, tuple)) and x and all(isinstance(v, (int, float)) for v in x):
        for i, v in enumerate(x):
            out[f"{path}[{i}]"] = v
    elif isinstance(x, (bool, np.bool_)):
        out[path] = bool(x)
    elif isinstance(x, (int, float, np.integer, np.floating, str)) or x is None:
        out[path] = x.item() if hasattr(x, "item") else x
    return out


def compare(official: dict, mine: dict, mc: tuple[dict, dict] | None = None, prefix: str = "") -> list[dict]:
    """Every shared leaf whose relative difference exceeds TOL (strings: any difference).
    Cent-rounded money fields need more than 0.01 USD as well. Bootstrap fields (intervals, p, se) are
    compared with the mean over SEEDS and marked within_mc when inside 3 Monte Carlo sd (p: at least 0.002)."""
    fo, fm = flat(official), flat(mine)
    mc_mean, mc_sd = mc if mc is not None else ({}, {})
    rows = []
    for k in sorted(set(fo) & set(fm)):
        a, b = fo[k], fm[k]
        if isinstance(a, str) or isinstance(b, str) or isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
            if a != b:
                rows.append({"field": prefix + k, "official": a, "audit": b, "rel": None})
            continue
        a, b = float(a), float(mc_mean.get(k, b))
        if np.isnan(a) and np.isnan(b):
            continue
        rel = abs(a - b) / max(abs(a), 1e-12)
        cents = k.endswith(("pnl_usd", "idea_usd", "diff_usd"))
        if rel > TOL and (not cents or abs(a - b) > 0.01):
            r = {"field": prefix + k, "official": a, "audit": b, "rel": rel}
            if k in mc_sd:
                band = 3 * mc_sd[k] * np.sqrt(1 + 1 / len(SEEDS))
                if k.endswith("p_two_sided"):
                    band = max(band, 0.002)
                r["mc_sd"], r["within_mc"] = mc_sd[k], bool(abs(a - b) <= band)
            rows.append(r)
    return rows


def mc_runs(T, sig, preds, pairs, days) -> tuple[dict, dict]:
    """The score under each seed; returns (mean of numeric leaves, sd of numeric leaves)."""
    runs = [flat(score(T, sig, preds, pairs, days, boot_index(len(days), seed=s))) for s in SEEDS]
    keys = [k for k, v in runs[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    arr = {k: np.array([r[k] for r in runs], float) for k in keys}
    return {k: float(v.mean()) for k, v in arr.items()}, {k: float(v.std(ddof=1)) for k, v in arr.items()}


# ----------------------------------------------------------------- data checks on the real files

def panel_checks(P: pd.DataFrame, T: pd.DataFrame, sig: np.ndarray) -> dict:
    key = ["delivery_hour", "zone"]
    guard(P["delivery_hour"])
    P = P[P["delivery_hour"].dt.tz_convert(NY) >= pd.Timestamp(f"{YEAR}-01-01", tz=NY)]
    out = {"rows": int(len(P)), "duplicated_keys": int(P.duplicated(key).sum())}
    ld = P["delivery_hour"].dt.tz_convert(NY).dt.tz_localize(None).dt.normalize()
    out["delivery_date_matches_local_date"] = bool((pd.to_datetime(P["delivery_date"]) == ld).all())
    per = P.groupby(ld).size()
    out["days"] = int(len(per))
    out["rows_per_day"] = {str(k): int(v) for k, v in per.value_counts().items()}
    out["dst_days"] = {d: int(per.get(pd.Timestamp(d), 0)) for d in (f"{YEAR}-03-12", f"{YEAR}-11-05")}
    if "bid_date" in P:
        out["bid_date_is_day_before"] = bool((pd.to_datetime(P["bid_date"]) == ld - pd.Timedelta(days=1)).all())
    m = T[key + ["gap"]].assign(sig=sig).merge(P.drop_duplicates(key), on=key, how="outer",
                                               suffixes=("_audit", "_panel"), indicator=True, validate="one_to_one")
    out["rows_only_in_prices"] = int((m["_merge"] == "left_only").sum())
    out["rows_only_in_panel"] = int((m["_merge"] == "right_only").sum())
    both = m[m["_merge"] == "both"]
    dg = (both["gap_audit"] - both["gap_panel"]).abs()
    out["gap_max_abs_diff"] = float(dg.max())
    out["gap_nan_mismatch"] = int((both["gap_audit"].isna() != both["gap_panel"].isna()).sum())
    if "gap_365d_h" in both:
        ds_ = (both["sig"] - both["gap_365d_h"]).abs()
        out["signal_max_abs_diff"] = float(ds_.max())
        out["signal_rows_diff_over_1e-6"] = int((ds_ > 1e-6).sum())
        out["signal_sign_disagreements"] = int((np.sign(both["sig"].fillna(0)) != np.sign(both["gap_365d_h"].fillna(0))).sum())
    return out


def align_preds(T: pd.DataFrame, pr: pd.DataFrame, cols: list[str]) -> tuple[dict, dict]:
    key = ["delivery_hour", "zone"]
    info = {"rows": int(len(pr)), "duplicated_keys": int(pr.duplicated(key).sum())}
    m = T[key].merge(pr[key + cols], on=key, how="left", validate="one_to_one")
    preds = {}
    for c in cols:
        v = m[c].to_numpy(float)
        info[f"{c}_nan_or_missing"] = int((~np.isfinite(v)).sum())
        preds[c] = v
    return preds, info


# ----------------------------------------------------------------- main

def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prices", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--official", required=True)
    ap.add_argument("--panel")
    ap.add_argument("--deep-base")
    ap.add_argument("--deep-weather")
    ap.add_argument("--deep-outages")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    cols = ["delivery_hour", "zone", "da_lbmp", "rt_lbmp", "da_published_at", "rt_published_at", "rt_revised"]
    H = hours_frame(read_pre2024(a.prices, columns=cols, start=pd.Timestamp(f"{YEAR - 2}-12-25", tz=NY)))
    days = calendar()
    T = H[H["ldate"].dt.year == YEAR].reset_index(drop=True)
    sig_all = trailing_signal(H, days)
    sig = sig_all[(H["ldate"].dt.year == YEAR).to_numpy()]
    off = json.loads(Path(a.official).read_text())
    pairs = [tuple(p.split("|")) for p in off.get("pairs", {}).get("chosen", [])]

    pr = read_pre2024(a.preds)
    gcols = [c for c in ("base", "gen", "weather", "outages") if c in pr.columns and pr[c].notna().any()]
    preds, pinfo = align_preds(T, pr, gcols)
    report = {"generated": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
              "holdout": "every read filtered to delivery hours before 2024-01-01 New York time and checked",
              "inputs": {"prices_rows_2023": int(len(T)), "pairs": ["|".join(p) for p in pairs], "preds": pinfo,
                         "days_in_calendar": int(len(days)),
                         "calendar_days_without_price_rows": int(len(set(days) - set(T["ldate"].unique())))}}
    if a.panel:
        report["panel"] = panel_checks(read_pre2024(a.panel), T, sig)

    mine = score(T, sig, preds, pairs, days, boot_index(len(days)))
    mc = mc_runs(T, sig, preds, pairs, days)
    report["audit"] = {k: v for k, v in mine.items() if not k.startswith("_")}
    report["audit_mc_mean"], report["audit_mc_sd"] = mc
    report["diffs_gbm"] = compare({k: off.get(k) for k in ("scored_days", "extreme_cut_abs_gap_usd", "baseline",
                                                           "ideas", "fee_stress")}, mine, mc)

    # Variant with the panel's own gap and baseline signal: isolates the data layer from the money logic.
    if a.panel:
        P = read_pre2024(a.panel, columns=["delivery_hour", "zone", "gap", "gap_365d_h"])
        m = T[["delivery_hour", "zone"]].merge(P, on=["delivery_hour", "zone"], how="left", validate="one_to_one")
        Tp = T.assign(gap=m["gap"].to_numpy(float))
        mp = score(Tp, m["gap_365d_h"].to_numpy(float), preds, pairs, days, boot_index(len(days)))
        report["diffs_gbm_with_panel_inputs"] = compare(
            {k: off.get(k) for k in ("scored_days", "extreme_cut_abs_gap_usd", "baseline", "ideas", "fee_stress")},
            mp, mc)

    if "deep" in off:
        dp, dinfo = {}, {}
        for fs, f in (("base", a.deep_base), ("weather", a.deep_weather), ("outages", a.deep_outages)):
            if f:
                p, info = align_preds(T, read_pre2024(f), ["pred_gap"])
                dp[fs], dinfo[fs] = p["pred_gap"], info
        report["inputs"]["deep_preds"] = dinfo
        if dp:
            md = score(T, sig, dp, pairs, days, boot_index(len(days)))
            mc_d = mc_runs(T, sig, dp, pairs, days)
            report["audit_deep"] = {k: v for k, v in md.items() if not k.startswith("_")}
            report["diffs_deep"] = compare({k: off["deep"].get("scores", {}).get(k) for k in
                                            ("scored_days", "extreme_cut_abs_gap_usd", "baseline", "ideas", "fee_stress")},
                                           md, mc_d, "deep.")
            idx = boot_index(len(days))
            comp = {k: deep_vs_gbm(md["_ledgers"][k], mine["_ledgers"][k], days, idx)
                    for k in ("A", "B", "C", "D") if k in md["_ledgers"] and k in mine["_ledgers"]}
            report["audit_deep_vs_gbm"] = comp
            report["diffs_deep_vs_gbm"] = compare(off["deep"].get("deep_vs_gbm", {}), comp, None, "deep_vs_gbm.")

    Path(a.out).write_text(json.dumps(report, indent=1, default=str))
    rows_ = [r for k in ("diffs_gbm", "diffs_deep", "diffs_deep_vs_gbm") for r in report.get(k, [])]
    real = [r["field"] for r in rows_ if not r.get("within_mc")]
    print(f"wrote {a.out}: {len(rows_)} differences over {TOL:.1%}, {len(real)} outside Monte Carlo noise: {real[:20]}")
    return report


def extract(path: str, col: str, lo: str, cols: str | None = None) -> None:
    """Stream rows with lo <= col < 2024-01-01 (New York) as parquet on stdout, writing nothing where it runs:
    ssh gene 'nyiso-us/.venv/bin/python - extract nyiso-us/parquet/prices_zone.parquet delivery_hour 2021-12-25' \\
        < audit_pnl.py > prices.parquet"""
    import io
    import pyarrow.parquet as pq
    df = read_pre2024(path, col, cols.split(",") if cols else None, start=pd.Timestamp(lo, tz=NY))
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf)
    sys.stdout.buffer.write(buf.getvalue())
    print(f"{path}: {len(df)} rows, max {df[col].max()}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        extract(*sys.argv[2:])
        sys.exit(0)
    sys.exit(0 if main() else 1)
