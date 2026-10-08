"""Run one V4 to V9 strategy row through the v2 rolling runner (us_v2/model/rolling.py on gene).

    python run_ideas.py <ROW>          ROW in REGISTRY (below); one queue line per row
    python run_ideas.py --list

Every row: features() of its module joined as `extra`; R.run_rows fits per quarter (train = previous
3 years, test = the quarter; 2012 warm-up only to choose 2013 settings); predictions to positions with
the module's rule and R.costs; settings chosen per quarter by R.choose_setting on earlier out-of-sample
quarters only; R.write_positions -> results/v2/pos/<ROW>.parquet + .json (settings_tried counted).
Queue: price rows wait for results/v2_price_data.done; V4 waits for results/v2_weather_data.done and runs
last. After every row: results/v2_ideas.done (the V12 allocator waits for it).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("V2_MODEL_DIR", str(Path.home() / "nyiso-us" / "v2" / "model")))

import ideas_common as C  # noqa: E402
from learner import base_fit_predict, check_features  # noqa: E402
import v4_reforecast as V4, v5_border as V5, v6_specialists as V6  # noqa: E401,E402
import v7_regimes as V7, v8_error_mining as V8, v9_decompose as V9  # noqa: E401,E402
import v16_gefs_joined as V16  # noqa: E402

KEYS = {"bid_date", "delivery_date", "delivery_hour", "zone", "hour_key", "quarter", "warmup"}
RESULTS = Path(os.environ.get("V2_RESULTS", str(Path.home() / "nyiso-us" / "results" / "v2")))

REGISTRY = {
    "V4_B_reforecast": dict(mod=V4, kind="single", notes="scored 2013-2019 only (reforecast ends 2019)"),
    "V5_border_inputs": dict(mod=V5, kind="single"),
    "V5_zone_border_spread": dict(mod=V5, kind="spread"),
    "V6_NYC": dict(mod=V6, kind="specialist", zone="N.Y.C."),
    "V6_LI": dict(mod=V6, kind="specialist", zone="LONGIL"),
    "V7_flags": dict(mod=V7, kind="single"),
    "V7_recency": dict(mod=V7, kind="multi", settings=["h365", "h730", "flat"]),
    "V8_error_mining": dict(mod=V8, kind="multi", settings=["cells", "gbm2"]),
    "V9_decompose": dict(mod=V9, kind="decompose"),
    "V16_B_gefs_joined": dict(mod=V16, kind="single",
                              notes="V4 on GEFS v12: reforecast runs to 2019-12-31 + live archive from 2020-09-23; "
                                    "delivery 2020-01-01..2020-09-23 not traded"),
}


# ------------------------------------------------------------------ runner API (rolling.py, 7 Oct)
def runner():
    import rolling as R
    return R


def costs(R, frame):
    """(load_cost, supply_cost) per row; R.row_costs returns supply first."""
    sup, lod = R.row_costs(pd.to_datetime(frame["delivery_date"]).dt.year)
    return lod, sup


def base_features(R, panel, row):
    cols = [c for c in R.base_features() if c in panel]
    if row == "V5_border_inputs":
        cols += [c for c in panel.columns if c.startswith("bord__")]
    return cols


# ----------------------------------------------------------------------------- rows
def build(row: str):
    R = runner()
    spec = REGISTRY[row]
    mod = spec["mod"]
    panel = R.load_panel()
    C.assert_pre_holdout(panel["delivery_hour"])
    idx = pd.MultiIndex.from_frame(panel[["bid_date", "zone", "delivery_hour"]])
    extra = mod.features(idx, panel=panel)
    extra.index = panel.index
    if spec["kind"] == "decompose":
        extra = extra.join(V9.targets(panel))
    extra = extra[[c for c in extra.columns if c not in panel.columns]]
    cols = base_features(R, panel, row) + [c for c in extra.columns if not c.startswith("y_")]
    check_features(cols)
    return R, spec, mod, panel, extra, cols


def fit_predict_for(row, spec, mod, cols):
    kind = spec["kind"]
    if kind == "single":
        return lambda tr, te, q: pd.Series(base_fit_predict(tr, te, "gap", cols), index=te.index)
    if kind == "specialist":
        z = spec["zone"]

        def fp(tr, te, q):
            t = te[te["zone"] == z]
            return pd.Series(base_fit_predict(tr[tr["zone"] == z], t, "gap", cols), index=t.index)
        return fp
    if kind == "decompose":
        return lambda tr, te, q: V9.fit_predict(tr, te, base_fit_predict, cols)
    if row == "V7_recency":
        def fp(tr, te, q):
            end = pd.to_datetime(tr["bid_date"]).max()
            out = {}
            for name, hl in zip(spec["settings"], V7.HALF_LIVES):
                out[f"pred_{name}"] = base_fit_predict(tr, te, "gap", cols, sample_weight=V7.sample_weight(tr, end, hl))
            return pd.DataFrame(out, index=te.index)
        return fp
    if row == "V8_error_mining":
        return lambda tr, te, q: V8.fit_predict(tr, te, base_fit_predict, cols)
    raise KeyError(row)


def run(row: str):
    import strategies_v2 as S
    R, spec, mod, panel, extra, cols = build(row)
    if spec["kind"] == "spread":
        run_spread(R, panel, extra)
        return
    preds = R.run_rows(row, fit_predict_for(row, spec, mod, cols), panel=panel, extra=extra)
    rows = S.attach(preds, panel)
    if "zone" in spec:
        rows = rows[rows["zone"] == spec["zone"]].reset_index(drop=True)
    if row == "V4_B_reforecast":       # reforecast ends 2019; nothing covers 2020-01-01..2021-03-24
        rows = rows[pd.to_datetime(rows["delivery_date"]) < pd.Timestamp("2020-01-01")].reset_index(drop=True)
    if row == "V16_B_gefs_joined":     # no forecast of either source for Jan-Sep 2020: not traded
        rows = rows[V16.traded(rows["delivery_date"]).to_numpy()].reset_index(drop=True)
    settings = spec.get("settings")

    def pos_fn(df, st):
        col = f"pred_{st}" if st else "pred"
        return mod.rule(df, df[col].to_numpy(float), costs(R, df)).astype(float)

    if settings:
        mw, choices = S.per_quarter(rows, settings, pos_fn)
    else:
        mw, choices = pos_fn(rows, None), {}
    R.write_positions(row, rows.assign(mw=mw)[["delivery_hour", "zone", "mw"]],
                      meta=dict(idea=mod.NAME, line=row, settings_tried=len(settings or [1]), choices=choices,
                                features=len(cols), learner="ideas/learner.py fixed LightGBM",
                                notes=spec.get("notes", "")))


def run_spread(R, panel, extra):
    """V5 zone-versus-border spread: no model; ledger with border leg, outside results/v2/pos/."""
    lab = V5.border_labels(panel)
    load_c, sup_c = costs(R, panel)
    rows = []
    for b in C.BORDERS:
        legs = V5.spread_legs(panel, extra, (load_c, sup_c), b)
        bgap = lab[V5.TAG[b]].to_numpy(float)
        zgap = panel["gap"].to_numpy(float)
        ok = ~np.isnan(zgap) & ~np.isnan(bgap)
        pos = np.where(ok, legs["mw"].to_numpy(float), 0.0)
        spread = np.nan_to_num(zgap - bgap)
        rows.append(legs.assign(mw=pos, border_mw=-pos, gross=pos * spread,
                                pnl=pos * spread - np.abs(pos) * (load_c + sup_c)))
    led = pd.concat(rows, ignore_index=True)
    led = led[led["delivery_hour"] >= pd.Timestamp("2013-01-01", tz=C.TZ)]
    C.assert_pre_holdout(led["delivery_hour"])
    out = RESULTS / "ideas"
    out.mkdir(parents=True, exist_ok=True)
    led.to_parquet(out / "V5_zone_border_spread.parquet")
    led["year"] = led["delivery_hour"].dt.year
    summ = led.assign(mwh=led["mw"].abs()).groupby(["year", "border"])[["pnl", "mwh"]].sum()
    (out / "V5_zone_border_spread.json").write_text(json.dumps(dict(
        flag="not_tradable_as_virtual", settings_tried=1, rule="side of trailing 28-day zone-minus-border gap",
        by_year_border={f"{k[0]}|{k[1]}": {kk: float(vv) for kk, vv in v.items()}
                        for k, v in summ.to_dict("index").items()}), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("row", nargs="?")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.row:
        print("\n".join(REGISTRY))
    else:
        run(a.row)
