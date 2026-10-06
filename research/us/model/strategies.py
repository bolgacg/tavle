"""Positions and money for the registered strategies (CONTRACT.md, "Positions and money", "Strategies").

Per zone-hour, 1 MW, assumed to clear at the market price. Position +1 = virtual load (earns gap - fee),
-1 = virtual supply (earns -gap - fee), 0 = none. gap = rt_lbmp - da_lbmp. fee = that year's
NYISO Schedule 1 non-physical rate (fees.py), or a stress cost for every MWh.

    baseline(panel)                  side of the trailing 365-day mean gap (panel.BASELINE_SIGNAL)
    idea_A(panel, pred)              supply when pred <= -fee, else none
    idea_B / idea_D(panel, pred)     supply when pred <= -fee, load when pred >= +fee, else none
    idea_C(panel, pred, pairs)       per pair (i, j): load i + supply j when pred_i - pred_j >= 2 fee,
                                     the reverse when <= -2 fee, else none
    choose_pairs(panel, pred)        up to 5 pairs, from build-year predictions only
    weather_features / outage_features   idea B and D inputs (features_weather.py, features_outages.py)

Every function returns an hourly ledger: delivery_date, delivery_hour, zone (or "i|j"), pos, mwh, pnl,
gross, absgap (largest |gap| of the legs, for the extreme-hours check), pred.
"""
from __future__ import annotations

import itertools

import lock

import numpy as np
import pandas as pd

import fees
from panel import BASELINE_SIGNAL, ZONES

SUPPLY, NONE, LOAD = -1, 0, 1
MAX_PAIRS = 5
IDEAS = {
    "A": dict(features="base", rule="supply_only"),
    "B": dict(features="weather", rule="two_sided"),
    "C": dict(features="base", rule="pairs"),
    "D": dict(features="outages", rule="two_sided"),
}


def _fee(panel: pd.DataFrame, fee_override: float | None) -> np.ndarray:
    return fees.fee_for(panel["delivery_date"], fee_override)


def ledger(panel: pd.DataFrame, pos: np.ndarray, fee: np.ndarray, pred: pd.Series | None = None) -> pd.DataFrame:
    lock.assert_build_only(panel["delivery_hour"])
    gap = panel["gap"].to_numpy(float)
    pos = np.asarray(pos, dtype=float)
    pos = np.where(np.isnan(gap), 0.0, pos)          # an hour without a settled gap is not traded
    g = np.nan_to_num(gap)
    mwh = np.abs(pos)
    return pd.DataFrame({"delivery_date": panel["delivery_date"].to_numpy(),
                         "delivery_hour": panel["delivery_hour"].to_numpy(),
                         "zone": panel["zone"].to_numpy(), "pos": pos, "mwh": mwh,
                         "gross": pos * g, "pnl": pos * g - mwh * fee, "absgap": np.abs(g),
                         "pred": (pred.reindex(panel.index).to_numpy() if pred is not None else np.nan)},
                        index=panel.index)


def baseline(panel: pd.DataFrame, fee_override: float | None = None) -> pd.DataFrame:
    s = panel[BASELINE_SIGNAL].to_numpy(float)
    pos = np.where(s < 0, SUPPLY, np.where(s > 0, LOAD, NONE))
    pos = np.where(np.isnan(s), NONE, pos)
    return ledger(panel, pos, _fee(panel, fee_override))


def idea_A(panel: pd.DataFrame, pred: pd.Series, fee_override: float | None = None) -> pd.DataFrame:
    fee = _fee(panel, fee_override)
    p = pred.reindex(panel.index).to_numpy(float)
    pos = np.where(p <= -fee, SUPPLY, NONE)
    return ledger(panel, pos, fee, pred)


def two_sided(panel: pd.DataFrame, pred: pd.Series, fee_override: float | None = None) -> pd.DataFrame:
    fee = _fee(panel, fee_override)
    p = pred.reindex(panel.index).to_numpy(float)
    pos = np.where(p <= -fee, SUPPLY, np.where(p >= fee, LOAD, NONE))
    return ledger(panel, pos, fee, pred)


idea_B = two_sided
idea_D = two_sided


def _pair_frame(panel: pd.DataFrame, pred: pd.Series, i: str, j: str) -> pd.DataFrame:
    cols = ["delivery_date", "delivery_hour", "gap"]
    a = panel.loc[panel["zone"] == i, cols].assign(pred=pred.reindex(panel.index[panel["zone"] == i]))
    b = panel.loc[panel["zone"] == j, cols].assign(pred=pred.reindex(panel.index[panel["zone"] == j]))
    return a.merge(b, on=["delivery_date", "delivery_hour"], suffixes=("_i", "_j"), how="inner")


def idea_C(panel: pd.DataFrame, pred: pd.Series, pairs: list[tuple[str, str]],
           fee_override: float | None = None) -> pd.DataFrame:
    lock.assert_build_only(panel["delivery_hour"])
    out = []
    for i, j in pairs:
        m = _pair_frame(panel, pred, i, j)
        fee = fees.fee_for(m["delivery_date"], fee_override)
        d = (m["pred_i"] - m["pred_j"]).to_numpy(float)
        leg = np.where(d >= 2 * fee, LOAD, np.where(d <= -2 * fee, SUPPLY, NONE)).astype(float)
        ok = m["gap_i"].notna() & m["gap_j"].notna()
        leg = np.where(ok, leg, 0.0)
        spread = np.nan_to_num((m["gap_i"] - m["gap_j"]).to_numpy(float))
        mwh = 2 * np.abs(leg)
        out.append(pd.DataFrame({"delivery_date": m["delivery_date"].to_numpy(),
                                 "delivery_hour": m["delivery_hour"].to_numpy(), "zone": f"{i}|{j}",
                                 "pos": leg, "mwh": mwh, "gross": leg * spread, "pnl": leg * spread - mwh * fee,
                                 "absgap": np.maximum(m["gap_i"].abs(), m["gap_j"].abs()).fillna(0).to_numpy(),
                                 "pred": d}))
    if not out:
        return pd.DataFrame(columns=["delivery_date", "delivery_hour", "zone", "pos", "mwh", "gross", "pnl",
                                     "absgap", "pred"])
    return pd.concat(out, ignore_index=True)


def choose_pairs(panel: pd.DataFrame, pred: pd.Series, max_pairs: int = MAX_PAIRS) -> tuple[list, list[dict]]:
    """All 55 zone pairs scored by the pair rule's net P&L on the given (build-year, out-of-sample)
    predictions; the best `max_pairs` with positive P&L are chosen. Returns (pairs, table)."""
    lock.assert_build_only(panel["delivery_hour"])
    table = []
    for i, j in itertools.combinations(ZONES, 2):
        led = idea_C(panel, pred, [(i, j)])
        table.append({"pair": f"{i}|{j}", "pnl": float(led["pnl"].sum()), "mwh": float(led["mwh"].sum()),
                      "per_mwh": float(led["pnl"].sum() / led["mwh"].sum()) if led["mwh"].sum() else None})
    table.sort(key=lambda r: -r["pnl"])
    chosen = [tuple(r["pair"].split("|")) for r in table if r["pnl"] > 0][:max_pairs]
    return chosen, table


def _index(panel: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(panel[["bid_date", "zone", "delivery_hour"]])


def weather_features(panel: pd.DataFrame, store) -> pd.DataFrame:
    """Idea B columns from features_weather.py (features agent), on the locked store."""
    import features_weather as FW
    lock.assert_build_only(panel["delivery_hour"])
    f = FW.features(_index(panel), store=store)
    f.index = panel.index
    return f


def outage_features(panel: pd.DataFrame, store) -> pd.DataFrame:
    """Idea D columns from features_outages.py (features agent), on the locked store."""
    import features_outages as FO
    lock.assert_build_only(panel["delivery_hour"])
    f = FO.features(_index(panel), store=store)
    f.index = panel.index
    return f
