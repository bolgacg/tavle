"""Scoring (CONTRACT.md, "Scoring"; OBJECTIVES.md items 5 and 6).

The unit is the day: daily P&L = sum of the day's hourly ledger rows. For each idea the tested series
is (idea daily P&L - baseline daily P&L). Intervals come from a stationary bootstrap over days
(Politis and Romano; mean block 7 days, 10,000 replicates, fixed seed); the same resampled days are
used for every idea, so the ideas are compared on identical draws.

Verdict words (objective 6 and the addendum of 6 Oct, late evening): pays = lower end of the
(corrected) interval above zero; doesn't pay = upper end below zero; inconclusive otherwise.
Holm (B, C, D): two-sided bootstrap p-values (percentile inversion, consistent with the intervals);
the hypothesis ranked k-th smallest (k = 1..m) gets the interval at level 1 - 0.05 / (m - k + 1).
A word other than inconclusive needs the Holm step to have rejected that hypothesis.
Deep against gradient boosting: better if the 95% interval of (deep - gbm) daily P&L is above zero,
worse if below; equivalent only if it contains zero and lies inside +- the gradient-boosting version's
mean daily cost (one fee per MWh it traded, at each side's cost); else inconclusive.
Context figures shown beside the verdicts (own profit per year, risk-adjusted ratios) live in context.py.
"""
from __future__ import annotations

from statistics import NormalDist

import lock

import numpy as np
import pandas as pd


N_BOOT = 10_000
MEAN_BLOCK = 7
SEED = 20261006
ALPHA = 0.05
POWER = 0.80
SECONDARY = ("B", "C", "D")


# ----------------------------------------------------------------- bootstrap

def stationary_indices(n: int, n_boot: int = N_BOOT, mean_block: float = MEAN_BLOCK, seed: int = SEED) -> np.ndarray:
    """(n_boot, n) day indices: blocks start at a uniform day, continue circularly, and end with
    probability 1/mean_block after each day."""
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    new = rng.random((n_boot, n)) < p
    starts = rng.integers(0, n, (n_boot, n))
    for k in range(1, n):
        idx[:, k] = np.where(new[:, k], starts[:, k], (idx[:, k - 1] + 1) % n)
    return idx


def boot_means(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.asarray(x, float)[idx].mean(axis=1)


def interval(bm: np.ndarray, level: float) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(bm, a)), float(np.quantile(bm, 1 - a))


def p_two_sided(bm: np.ndarray) -> float:
    n = len(bm)
    lo = (np.sum(bm <= 0) + 1) / (n + 1)
    hi = (np.sum(bm >= 0) + 1) / (n + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def holm(pvals: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    order = sorted(pvals, key=lambda k: (pvals[k], k))
    m = len(order)
    out, still = {}, True
    for r, k in enumerate(order):
        a = alpha / (m - r)
        rej = still and pvals[k] < a
        out[k] = {"rank": r + 1, "alpha": a, "level": 1 - a, "p": pvals[k], "rejected": bool(rej), "reached": still}
        still = rej
    return out


def verdict(lo: float, hi: float, reached: bool = True) -> str:
    if not reached:
        return "inconclusive"
    if lo > 0:
        return "pays"
    if hi < 0:
        return "doesn't pay"
    return "inconclusive"


def mde(se: float, alpha: float, power: float = POWER) -> float:
    z = NormalDist()
    return float((z.inv_cdf(1 - alpha / 2) + z.inv_cdf(power)) * se)


# ----------------------------------------------------------------- daily series and summaries

def daily(led: pd.DataFrame, days: pd.DatetimeIndex, col: str = "pnl") -> pd.Series:
    return led.groupby("delivery_date")[col].sum().reindex(days, fill_value=0.0)


def money(led: pd.DataFrame) -> dict:
    mwh = float(led["mwh"].sum())
    pnl = float(led["pnl"].sum())
    gross = float(led["gross"].sum())
    return {"pnl_usd": round(pnl, 2), "mwh": mwh, "profit_per_mwh": (pnl / mwh) if mwh else None,
            "break_even_fee": (gross / mwh) if mwh else None}


def calibration(pred: pd.Series, gap: pd.Series, bins: int = 10) -> dict:
    d = pd.DataFrame({"pred": pred, "gap": gap}).dropna()
    if len(d) < bins * 10:
        return {}
    d["bin"] = pd.qcut(d["pred"].rank(method="first"), bins, labels=False)
    t = d.groupby("bin").agg(pred_mean=("pred", "mean"), gap_mean=("gap", "mean"), gap_median=("gap", "median"),
                             n=("gap", "size"))
    slope = float(np.polyfit(d["pred"], d["gap"], 1)[0]) if d["pred"].std() > 0 else None
    return {"deciles": [{k: (round(float(v), 4) if k != "n" else int(v)) for k, v in r.items()}
                        for r in t.to_dict("records")],
            "slope_gap_on_pred": slope, "corr": float(d["pred"].corr(d["gap"])),
            "mean_pred": float(d["pred"].mean()), "mean_gap": float(d["gap"].mean())}


def _groups(led: pd.DataFrame, key: pd.Series) -> dict:
    out = {}
    for k, g in led.groupby(key):
        out[str(k)] = {"pnl_usd": round(float(g["pnl"].sum()), 2), "mwh": float(g["mwh"].sum()),
                       "profit_per_mwh": float(g["pnl"].sum() / g["mwh"].sum()) if g["mwh"].sum() else None}
    return out


def summarize(name: str, led: pd.DataFrame, base: pd.DataFrame, days: pd.DatetimeIndex, idx: np.ndarray,
              level: float = 1 - ALPHA, extreme_cut: float | None = None, pred: pd.Series | None = None,
              gap: pd.Series | None = None) -> dict:
    """Everything score.py reports for one idea against the baseline on the same days."""
    lock.assert_build_only(pd.Series(days))
    di, db = daily(led, days), daily(base, days)
    diff = (di - db).to_numpy()
    bm = boot_means(diff, idx)
    lo, hi = interval(bm, level)
    lo95, hi95 = interval(bm, 0.95)
    alone = boot_means(di.to_numpy(), idx)                 # the idea's own money, not against the baseline
    month = pd.Series(pd.DatetimeIndex(days).to_period("M").astype(str), index=days)
    mi, md = di.groupby(month).sum(), (di - db).groupby(month).sum()
    r = {"idea": name, "n_days": int(len(days)), "mean_daily_idea": float(di.mean()),
         "mean_daily_baseline": float(db.mean()), "mean_daily_diff": float(diff.mean()),
         "interval_level": level, "interval": [lo, hi], "interval_95": [lo95, hi95],
         "p_two_sided": p_two_sided(bm), "se_daily_diff": float(bm.std(ddof=1)),
         "idea_alone_interval_95": list(interval(alone, 0.95)),
         "boot_means": bm,
         "idea_money": money(led), "baseline_money": money(base),
         "mwh_per_day": float(led["mwh"].sum() / len(days)),
         "worst_month_idea": {"month": mi.idxmin(), "pnl_usd": float(mi.min())},
         "worst_month_diff": {"month": md.idxmin(), "usd": float(md.min())},
         "months_positive_idea": int((mi > 0).sum()), "months_positive_diff": int((md > 0).sum()),
         "months": int(len(mi)),
         "monthly": {k: {"idea_usd": round(float(mi[k]), 2), "diff_usd": round(float(md[k]), 2)} for k in mi.index},
         "per_zone": _groups(led, led["zone"]),
         "per_year": _groups(led, pd.DatetimeIndex(led["delivery_date"]).year)}
    if extreme_cut is not None:
        li, lb = led[led["absgap"] <= extreme_cut], base[base["absgap"] <= extreme_cut]
        dd = (daily(li, days) - daily(lb, days)).to_numpy()
        b2 = boot_means(dd, idx)
        r["without_extreme_1pct"] = {"abs_gap_cut_usd": float(extreme_cut), "mean_daily_diff": float(dd.mean()),
                                     "interval_95": list(interval(b2, 0.95)), "idea_money": money(li),
                                     "baseline_money": money(lb)}
    if pred is not None and gap is not None:
        r["calibration"] = calibration(pred, gap)
    return r


def stress(name: str, led_s: pd.DataFrame, base_s: pd.DataFrame, days, idx) -> dict:
    di, db = daily(led_s, days), daily(base_s, days)
    bm = boot_means((di - db).to_numpy(), idx)
    return {"mean_daily_diff": float((di - db).mean()), "interval_95": list(interval(bm, 0.95)),
            "idea_money": money(led_s), "baseline_money": money(base_s)}


def apply_holm(results: dict[str, dict], idx_level_fn=None) -> None:
    """Fill interval/verdict: A at 95%; B, C, D with Holm-adjusted levels. Mutates results."""
    if "A" in results:
        a = results["A"]
        a["interval_level"], a["interval"] = 0.95, list(interval(a["boot_means"], 0.95))
        a["verdict"] = verdict(*a["interval"])
        a["mde_80pct_power"] = mde(a["se_daily_diff"], ALPHA)
    sec = {k: results[k]["p_two_sided"] for k in SECONDARY if k in results}
    h = holm(sec)
    for k, info in h.items():
        r = results[k]
        r["holm"] = info
        r["interval_level"] = info["level"]
        r["interval"] = list(interval(r["boot_means"], info["level"]))
        r["verdict"] = verdict(*r["interval"]) if info["rejected"] else "inconclusive"   # audit patch
        r["mde_80pct_power"] = mde(r["se_daily_diff"], ALPHA / len(sec))      # worst-case Holm level


def deep_vs_gbm(led_deep: pd.DataFrame, led_gbm: pd.DataFrame, days, idx) -> dict:
    d = (daily(led_deep, days) - daily(led_gbm, days)).to_numpy()
    bm = boot_means(d, idx)
    lo, hi = interval(bm, 0.95)
    margin = float((led_gbm["gross"] - led_gbm["pnl"]).sum() / len(days))   # the costs it paid: one fee per MWh
    if lo > 0:
        word = "better"
    elif hi < 0:
        word = "worse"
    elif -margin <= lo <= 0 <= hi <= margin:
        word = "equivalent"
    else:
        word = "inconclusive"
    return {"mean_daily_diff": float(d.mean()), "interval_95": [lo, hi], "margin_usd_per_day": margin,
            "verdict": word}


def compare(led_x: pd.DataFrame, led_y: pd.DataFrame, days, idx, level: float = 0.95) -> dict:
    """Daily P&L of x minus y on the same days and resamples: mean and interval (context, no test)."""
    d = (daily(led_x, days) - daily(led_y, days)).to_numpy()
    bm = boot_means(d, idx)
    return {"mean_daily_diff": float(d.mean()), "interval_95": list(interval(bm, level)),
            "p_two_sided": p_two_sided(bm)}


def strip(results):
    """Drop the bootstrap arrays before writing JSON."""
    if isinstance(results, dict):
        return {k: strip(v) for k, v in results.items() if k != "boot_means"}
    if isinstance(results, list):
        return [strip(v) for v in results]
    if isinstance(results, (np.floating, np.integer)):
        return results.item()
    return results
