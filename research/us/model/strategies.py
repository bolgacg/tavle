"""Positions and money for the registered strategies (CONTRACT.md, "Positions and money", "Strategies").

Per zone-hour, 1 MW, assumed to clear at the market price. Position +1 = virtual load (earns gap - load
cost), -1 = virtual supply (earns -gap - supply cost), 0 = none. gap = rt_lbmp - da_lbmp. Costs per
cleared MWh (fees.py, cost audit): load = Rate Schedule 1 + FERC charge; supply = the same + the uplift
upper bound; a stress cost replaces both. Every "fee" threshold in the rules is the cost of the side
being taken (B, D: supply when pred <= -supply cost, load when pred >= load cost; C: the predicted
difference must cover one load leg plus one supply leg).

    baseline(panel)                  side of the trailing 365-day mean gap (panel.BASELINE_SIGNAL)
    always_supply(panel)             supply in every zone-hour (context: the value of A's spike filter)
    idea_A(panel, p_spike, p_star)   supply in every zone-hour except where the predicted probability of
                                     a spike (gap >= S) exceeds p_star (addendum of 6 Oct, late evening)
    idea_A_hourly_mean(panel)        supply where the trailing 365-day zone-hour mean gap <= -supply cost
    idea_A_v1(panel, pred)           the superseded v1 rule: supply when pred <= -fee, else none. Kept so
                                     the v1 rehearsal stays reproducible; it also still selects the base
                                     regression model that idea C uses (C unchanged)
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
    "A": dict(features="base", rule="supply_except_spike_risk", model="classifier P(gap >= S)"),
    "B": dict(features="weather", rule="two_sided"),
    "C": dict(features="base", rule="pairs"),
    "D": dict(features="outages", rule="two_sided"),
}


def _load_cost(panel: pd.DataFrame, fee_override: float | None) -> np.ndarray:
    return fees.fee_for(panel["delivery_date"], fee_override)


def _supply_cost(panel: pd.DataFrame, fee_override: float | None) -> np.ndarray:
    return fees.supply_fee_for(panel["delivery_date"], fee_override)


def ledger(panel: pd.DataFrame, pos: np.ndarray, fee: np.ndarray | None = None, pred: pd.Series | None = None,
           fee_override: float | None = None) -> pd.DataFrame:
    """`fee` = an explicit per-row cost per MWh; None = each row's cost by side (fees.cost_for)."""
    lock.assert_build_only(panel["delivery_hour"])
    if fee is None:
        fee = fees.cost_for(panel["delivery_date"], pos, fee_override)
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
    return ledger(panel, pos, fee_override=fee_override)


def always_supply(panel: pd.DataFrame, fee_override: float | None = None) -> pd.DataFrame:
    return ledger(panel, np.full(len(panel), SUPPLY), fee_override=fee_override)


def idea_A(panel: pd.DataFrame, p_spike: pd.Series, p_star: float, fee_override: float | None = None) -> pd.DataFrame:
    """Supply unless P(spike) > p_star. A missing probability is not traded. The fee does not move the
    rule, only the money."""
    p = p_spike.reindex(panel.index).to_numpy(float)
    pos = np.where(np.isnan(p), NONE, np.where(p > p_star, NONE, SUPPLY))
    return ledger(panel, pos, None, p_spike, fee_override)


def idea_A_hourly_mean(panel: pd.DataFrame, fee_override: float | None = None) -> pd.DataFrame:
    """The simple version of A, no model: supply where the zone-hour's trailing 365-day mean gap (the
    baseline's signal, public at 05:00 on D) is at or below minus the supply cost, else none."""
    s = panel[BASELINE_SIGNAL].to_numpy(float)
    pos = np.where(np.isnan(s), NONE, np.where(s <= -_supply_cost(panel, fee_override), SUPPLY, NONE))
    return ledger(panel, pos, None, None, fee_override)


def idea_A_v1(panel: pd.DataFrame, pred: pd.Series, fee_override: float | None = None) -> pd.DataFrame:
    p = pred.reindex(panel.index).to_numpy(float)
    pos = np.where(p <= -_supply_cost(panel, fee_override), SUPPLY, NONE)
    return ledger(panel, pos, None, pred, fee_override)


def two_sided(panel: pd.DataFrame, pred: pd.Series, fee_override: float | None = None) -> pd.DataFrame:
    p = pred.reindex(panel.index).to_numpy(float)
    pos = np.where(p <= -_supply_cost(panel, fee_override), SUPPLY,
                   np.where(p >= _load_cost(panel, fee_override), LOAD, NONE))
    return ledger(panel, pos, None, pred, fee_override)


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
        pair_cost = fees.fee_for(m["delivery_date"], fee_override) + fees.supply_fee_for(m["delivery_date"], fee_override)
        d = (m["pred_i"] - m["pred_j"]).to_numpy(float)
        leg = np.where(d >= pair_cost, LOAD, np.where(d <= -pair_cost, SUPPLY, NONE)).astype(float)
        ok = m["gap_i"].notna() & m["gap_j"].notna()
        leg = np.where(ok, leg, 0.0)
        spread = np.nan_to_num((m["gap_i"] - m["gap_j"]).to_numpy(float))
        mwh = 2 * np.abs(leg)
        out.append(pd.DataFrame({"delivery_date": m["delivery_date"].to_numpy(),
                                 "delivery_hour": m["delivery_hour"].to_numpy(), "zone": f"{i}|{j}",
                                 "pos": leg, "mwh": mwh, "gross": leg * spread,
                                 "pnl": leg * spread - np.abs(leg) * pair_cost,
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


def outage_features(panel: pd.DataFrame, store, sites_through=None) -> pd.DataFrame:
    """Idea D columns from features_outages.py (features agent), on the locked store. `sites_through`
    (a date) recounts the 20 outage sites on snapshots up to that date (rehearsal: 2022-12-30);
    None keeps the frozen list counted on 2020 to 2023 (held-out run)."""
    import features_outages as FO
    lock.assert_build_only(panel["delivery_hour"])
    top = None
    if sites_through is not None:
        top = FO.top_sites(store.table("outages"), last_snapshot=pd.Timestamp(sites_through))
    f = FO.features(_index(panel), store=store, top=top)
    f.index = panel.index
    return f
