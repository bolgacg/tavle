"""V15a: the V2 limit-price rule on the V4 weather, V5 border and V8 error-mining models (OBJECTIVES.md, V15).

The limit rule needs the gap predicted at 9 hypothetical day-ahead levels, which the saved V4, V5 and V8 predictions
(one gap per row) do not hold, so the conditional model is retrained on each idea's own feature set
(ideas/run_ideas.build: same columns, same fixed learner, same rolling windows) with the V2 construction
(v15_common.make_cond_fp). V8 reuses its saved predictions: its second-stage correction (pred_<setting> - pred_base,
the setting V8 chose per quarter on earlier quarters, from pos/V8_error_mining.json) is added at every grid level.
V4 is scored on 2013 to 2019 only, as V4 (the reforecast ends in 2019).

    python v15_limit.py V4|V5|V8 [--threads 2]
Rows: V15a_V4_limit, V15a_V5_limit, V15a_V8_limit -> results/v2/pos/ (score_v2.py).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import v15_common as V
from v15_common import R, SV

ROW = {"V4": "V4_B_reforecast", "V5": "V5_border_inputs", "V8": "V8_error_mining"}


def v8_correction(g: pd.DataFrame) -> tuple[np.ndarray, dict]:
    p = R.load_preds("V8_error_mining")
    ch = json.loads((R.POS / "V8_error_mining.json").read_text()).get("choices", {})
    x = g[["delivery_hour", "zone", "quarter"]].merge(
        p[["delivery_hour", "zone", "pred_base", "pred_cells", "pred_gbm2"]], on=["delivery_hour", "zone"],
        how="left", validate="one_to_one")
    st = x["quarter"].map(lambda qn: (ch.get(qn) or {}).get("setting") or "cells")
    sel = np.where(st == "gbm2", x["pred_gbm2"], x["pred_cells"])
    corr = np.nan_to_num(sel - x["pred_base"].to_numpy(float))
    return corr, {qn: v.get("setting") for qn, v in ch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=sorted(ROW))
    ap.add_argument("--threads", type=int, default=V.THREADS)
    a = ap.parse_args()
    V.check_da_mult()
    import run_ideas as RI
    _, spec, mod, panel, extra, cols = RI.build(ROW[a.which])
    if a.which == "V5":
        V.save_v5_cache(panel, extra[[c for c in extra.columns if c.startswith("bx_")]])
    if a.which == "V4":
        V.save_cache(V.v4_cache_path(), panel, extra[[c for c in extra.columns if c.startswith("wx_")]])
    R.log(f"V15a {a.which}: {len(cols)} features, panel {panel.shape}")
    name = f"v15a_{a.which}_cond"
    preds = R.run_rows(name, V.make_cond_fp(cols, a.threads), panel=panel, extra=extra)
    g = SV.attach(preds, panel)
    meta = {"idea": "V15", "line": f"V15a: V2 limit-price rule on the {ROW[a.which]} model (conditional refit on its "
            "features, 9 day-ahead levels); a bid counts only when day-ahead clears at or past the limit",
            "settings_tried": 1, "features": len(cols), "levels": "da_d0_h x " + str(V.DA_MULT)}
    if a.which == "V8":
        corr, chosen = v8_correction(g)
        for k in range(len(V.DA_MULT)):
            g[f"cond_{k}"] = g[f"cond_{k}"] + corr
        meta.update({"settings_tried": 2, "v8_setting_by_quarter": chosen,
                     "notes": "V8 second stage (cells or gbm2, V8's own per-quarter choice) added at every level"})
    if a.which == "V4":
        g = g[pd.to_datetime(g["delivery_date"]) < V.REFORECAST_END].reset_index(drop=True)
        meta["notes"] = "scored 2013-2019 only (reforecast ends 2019), as V4_B_reforecast"
    mw = SV.v2_pos(g)
    SV.out(f"V15a_{a.which}_limit", g, mw, meta)


if __name__ == "__main__":
    main()
