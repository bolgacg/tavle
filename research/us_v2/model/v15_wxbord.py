"""V15c: one gradient-boosting model with weather and border inputs together (OBJECTIVES.md, V15).

Inputs: the v1 base features, the panel's bord__ border block and V5's bx_ border features (the V5_border_inputs
set, ideas/run_ideas.py), plus weather forecasts (v15_common): GEFS reforecast to 2019 through V4's path, GFS
archive from 25 March 2021 from the pipeline's day matrix, NaN in between (masked; the model still trades on its
price inputs there), never observed weather. Learner: ideas/learner.py (fixed LightGBM), rolling 3-year windows,
quarterly, as v2. Rule: two-sided v1 rule. Settings tried: 1.

Before training, a real-data lookahead check of the day-matrix path: the GFS columns of rows delivered on or before a
cut date must not move when every day-matrix row delivered after the cut is overwritten (three cuts).

    python v15_wxbord.py [--threads 2]
Row: V15c_wx_border_gbm -> results/v2/pos/.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import v15_common as V
from v15_common import R, SV, L

CUTS = ("2021-06-14", "2022-02-02", "2023-08-20")          # GFS days (the day-matrix path); the reforecast path
#                                                           has its own tests (ideas/test_lookahead.py, V4 audit)


def real_data_check(day: pd.DataFrame, keys: pd.DataFrame, feats: pd.DataFrame) -> dict:
    res = {}
    wcols = [c for c in day.columns if c.startswith(("wx__", "wxr__"))]
    dd = pd.to_datetime(day["delivery_date"]).dt.normalize()
    for cut in CUTS:
        c = pd.Timestamp(cut)
        sub = keys[(keys["delivery_date"] <= c) & (keys["delivery_date"] > c - pd.Timedelta(days=10))]
        bad = day.copy()
        bad.loc[(dd > c).to_numpy(), wcols] = 1e6
        a = feats.loc[sub.index]
        b = V.weather_from_day(bad, sub, use_reforecast=False)
        if not a.equals(b):
            raise V.LookaheadError(f"V15c weather moved when days after {cut} were overwritten")
        ctrl = day.copy()
        ctrl.loc[((dd <= c) & (dd > c - pd.Timedelta(days=10))).to_numpy(), wcols] = 1e6
        moved = not a.equals(V.weather_from_day(ctrl, sub, use_reforecast=False))
        has = bool(a["w15_temp_f"].notna().any())
        if has and not moved:
            raise V.LookaheadError(f"V15c positive control did not move at {cut}")
        res[cut] = {"rows": int(len(sub)), "unchanged_after_poison": True, "control_moved": moved, "has_weather": has}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=V.THREADS)
    a = ap.parse_args()
    panel = R.load_panel()
    keys = V.panel_keys(panel)
    day = V.read_day_weather()
    G = V.weather_from_day(day, keys, use_reforecast=False)
    chk = real_data_check(day, keys, G)
    W = V.combine_weather(V.v4_weather(panel), G)
    R.log(f"V15c weather lookahead check passed: {chk}")
    src = W["w15_src"].groupby(pd.to_datetime(panel["delivery_date"]).dt.year).agg(lambda s: s.value_counts().to_dict())
    R.log(f"V15c weather source by year (1 reforecast, 2 GFS, 0 none): {src.to_dict()}")
    B = V.v5_features(panel)
    extra = pd.concat([B, W], axis=1)
    cols = [c for c in R.base_features() if c in panel] + [c for c in panel.columns if c.startswith("bord__")] \
        + list(extra.columns)
    L.check_features(cols)
    V.V15DIR.mkdir(parents=True, exist_ok=True)
    (V.V15DIR / "v15c_lookahead_check.json").write_text(json.dumps({"checks": chk, "features": cols}, indent=1))
    params = dict(L.PARAMS, num_threads=a.threads)
    L.PARAMS.update(params)

    def fp(tr, te, q):
        return pd.Series(L.base_fit_predict(tr, te, "gap", cols), index=te.index)

    preds = R.run_rows("v15c_wx_border", fp, panel=panel, extra=extra)
    g = SV.attach(preds, panel)
    SV.out("V15c_wx_border_gbm", g, V.two_sided(g, g["pred"].to_numpy(float)),
           {"idea": "V15", "settings_tried": 1, "features": len(cols),
            "line": "V15c: one LightGBM with weather forecasts (GEFS reforecast to 2019, GFS archive from 2021-03-25, "
                    "masked between) and border inputs (bord__, V5 bx_), two-sided rule",
            "notes": "weather NaN 2020-01-01..2021-03-24 (no archived forecast); the model trades on price inputs there"})


if __name__ == "__main__":
    main()
