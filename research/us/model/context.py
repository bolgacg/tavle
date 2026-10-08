"""Context figures shown beside every verdict (OBJECTIVES addendum, 6 Oct late evening). None of these
is a test, and none feeds a verdict word; score.py, strategies.py and fees.py stay free of any
annualisation.

    block(ledgers, days, idx)   for every ledger (each idea, the baseline, always-supply):
                                own net profit per day with its 95% interval; annual net = mean daily
                                net x 365; return on a fixed bankroll of 500,000 USD = annual net /
                                500,000 (cost audit addendum, commit c3adae3); annualised Sharpe = mean /
                                std of daily net P&L x sqrt(365), days without a trade counted as zero;
                                maximum drawdown in USD of cumulative daily net P&L; return over maximum
                                drawdown = annual net / maximum drawdown.
Benchmarks (S&P 500, Danish rates) are added at page time, not here.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import score as SC

DAYS_PER_YEAR = 365
BANKROLL_USD = 500_000.0


def max_drawdown(daily_net: pd.Series) -> float:
    """Largest fall of cumulative net P&L from its running peak (the peak starts at zero), in USD."""
    c = np.r_[0.0, daily_net.cumsum().to_numpy()]
    return float(np.max(np.maximum.accumulate(c) - c))


def sharpe_annualised(daily_net: pd.Series) -> float | None:
    sd = float(daily_net.std(ddof=1))
    if not np.isfinite(sd) or sd == 0:
        return None
    return float(daily_net.mean()) / sd * math.sqrt(DAYS_PER_YEAR)


def own(led: pd.DataFrame, days, idx) -> dict:
    d = SC.daily(led, days)
    bm = SC.boot_means(d.to_numpy(), idx)
    annual = float(d.mean()) * DAYS_PER_YEAR
    mdd = max_drawdown(d)
    return {"mean_daily_net_usd": float(d.mean()), "interval_95": list(SC.interval(bm, 0.95)),
            "annual_net_usd": annual, "sharpe_annualised": sharpe_annualised(d),
            "max_drawdown_usd": mdd, "return_over_max_drawdown": (annual / mdd) if mdd > 0 else None,
            "days_with_trades": int((led.groupby("delivery_date")["mwh"].sum().reindex(days, fill_value=0) > 0).sum()),
            "bankroll_usd": BANKROLL_USD, "return_on_bankroll": annual / BANKROLL_USD}


def block(ledgers: dict[str, pd.DataFrame], days, idx) -> dict:
    return {k: own(v, days, idx) for k, v in ledgers.items()}


def fill_bankroll(blk: dict, bankroll_usd: float) -> dict:
    for v in blk.values():
        v["bankroll_usd"] = float(bankroll_usd)
        v["return_on_bankroll"] = v["annual_net_usd"] / float(bankroll_usd)
    return blk
