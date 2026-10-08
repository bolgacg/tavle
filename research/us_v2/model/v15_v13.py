"""V15b: the V13 all-inputs deep model traded on both sides and with limit prices (OBJECTIVES.md, V15).

Predictions: V13's saved full-menu gap predictions (preds/v13_<arch>.parquet), the architecture per quarter as V13
chose it on earlier quarters (results/v2/v13/arch_choice.json, else v13_rolling.chosen_by_quarter). No retraining.
Rows:
  V15b_V13_both    two-sided v1 rule on the chosen gap: supply when pred <= -supply cost, load when >= load cost
  V15b_V13_limit   V2 limit rule; the deep model gives the level, the base gradient-boosting conditional model
                   (preds/gbm.parquet cond_k, V2) gives the shape over the 9 day-ahead levels:
                   cond'_k = pred_v13 + cond_k - cond_at_today's_DA. Both inputs are bid-time; the limit is chosen
                   from the grid alone (strategies_v2.v2_limits).
Settings tried: 3 each (the architecture menu). Run after results/v13.done.

    python v15_v13.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import v15_common as V
from v15_common import R, SV


def compose_limit_grid(pred: np.ndarray, gbm_cond: np.ndarray) -> np.ndarray:
    """[n, K] grid: the deep prediction placed at today's day-ahead level, the GBM curve's shape around it."""
    return pred[:, None] + gbm_cond - gbm_cond[:, [V.I_D0]]


def main():
    V.check_da_mult()
    import v13_rolling as V13
    import v13_model as VM
    names = {a: V13.name_of(a) for a in VM.ARCHS}
    miss = [n for n in names.values() if not V13.finished(n)]
    if miss:
        R.log(f"V15b NOT RUN: V13 full-menu predictions missing or incomplete: {miss}")
        V.V15DIR.mkdir(parents=True, exist_ok=True)
        (V.V15DIR / "V15b_not_run.json").write_text(json.dumps({"missing": miss}, indent=1))
        return
    panel = R.load_panel(columns=["delivery_hour", "zone", "delivery_date", "gap", "y_da_lbmp", "gap_365d_h"])
    f = V13.V13DIR / "arch_choice.json"
    if f.exists():
        choice = {qn: v["arch"] for qn, v in json.loads(f.read_text()).items()}
        how = "results/v2/v13/arch_choice.json"
    else:
        choice = {qn: a for qn, (a, _) in V13.chosen_by_quarter(panel).items()}
        how = "v13_rolling.chosen_by_quarter"
    g = V13.merged(panel, names)
    sel = np.full(len(g), np.nan)
    for qn, a in choice.items():
        m = (g["quarter"] == qn).to_numpy()
        sel[m] = g.loc[m, f"pred__{a}"].to_numpy()
    g["pred_sel"] = sel
    g = SV.attach(g.drop(columns=["gap"]), panel)
    meta = {"idea": "V15", "settings_tried": len(VM.ARCHS), "arch_choice_source": how, "arch_choice": choice}
    SV.out("V15b_V13_both", g, V.two_sided(g, g["pred_sel"].to_numpy(float)),
           {**meta, "line": "V15b: V13 all-inputs deep gap, two-sided rule (supply <= -cost, load >= cost)"})
    K = len(V.DA_MULT)
    gb = R.load_preds("gbm")[["delivery_hour", "zone"] + [f"dag_{k}" for k in range(K)] + [f"cond_{k}" for k in range(K)]]
    h = g[["delivery_hour", "zone", "delivery_date", "gap", "y_da_lbmp", "pred_sel"]].merge(
        gb, on=["delivery_hour", "zone"], how="inner", validate="one_to_one")
    C = np.stack([h[f"cond_{k}"].to_numpy(float) for k in range(K)], 1)
    G = compose_limit_grid(h["pred_sel"].to_numpy(float), C)
    for k in range(K):
        h[f"cond_{k}"] = G[:, k]
    SV.out("V15b_V13_limit", h, SV.v2_pos(h),
           {**meta, "line": "V15b: V2 limit rule, level from the V13 deep gap, shape over 9 day-ahead levels from the "
                            "V2 gradient-boosting conditional model", "levels": "da_d0_h x " + str(V.DA_MULT)})


if __name__ == "__main__":
    main()
