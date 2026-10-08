"""V14a, CPU lane: the v2 mean gradient-boosting model (gbm_rolling BASE_CFG, v1 base features) trained on 1, 2, 3
and 5-year windows and an expanding window from 2010-01-01, each with recency weights (half-life 12 months,
declared); columns b1, b2, b3, b5, bexp and their mean gbm_win -> results/v2/preds/v14a_gbm.parquet.

    python v14_gbm.py [--threads 2] [--smoke] [--selftest-only]

Self-test first (results/v2/v14/selftest_gbm_a.json): the prediction for one test day must not move when every
label after the window cut and every input of later bid days are replaced by noise, and a label column offered
as an input must raise.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import rolling as R
import v14_core as C

os.environ.setdefault("US_LGBM_THREADS", "2")
import lightgbm as lgb  # noqa: E402

import gbm as G1  # noqa: E402
from gbm_rolling import BASE_CFG  # noqa: E402

WINDOWS = (("b1", 1), ("b2", 2), ("b3", 3), ("b5", 5), ("bexp", None))


def window_rows(panel, dd, q, yrs):
    lo, hi = R.window(q[0], yrs) if yrs else (R.DATA_START, R.window(q[0])[1])
    m = (dd >= pd.Timestamp(lo)) & (dd <= pd.Timestamp(hi)) & panel["gap"].notna()
    return panel[m]


def make_fp(panel: pd.DataFrame, feats: list[str], threads: int, rounds: int, windows=WINDOWS):
    C.assert_inputs_public(feats)
    cats = [c for c in G1.CATEGORICAL if c in feats]
    dd = pd.to_datetime(panel["delivery_date"])

    def fp(tr, te, q):
        out = pd.DataFrame(index=te.index)
        Xt = te[feats].astype(np.float32)
        seen = {}
        for col, yrs in windows:
            lo = R.window(q[0], yrs)[0] if yrs else R.DATA_START
            if lo in seen:                                       # same rows as a shorter window (early years)
                out[col] = out[seen[lo]]
                continue
            seen[lo] = col
            trw = window_rows(panel, dd, q, yrs)
            R.assert_pre2024(trw["delivery_hour"], "v14a gbm window")
            assert trw["delivery_date"].max() <= C.label_cut(q)
            w = C.recency_weights(trw["delivery_date"], C.label_cut(q))
            y = np.clip(trw["gap"].to_numpy(float), -BASE_CFG["target_clip"], BASE_CFG["target_clip"])
            p = {**G1.FIXED, "num_threads": threads, "num_leaves": BASE_CFG["num_leaves"],
                 "min_data_in_leaf": BASE_CFG["min_data_in_leaf"], "deterministic": True}
            b = lgb.train(p, lgb.Dataset(trw[feats].astype(np.float32), y, weight=w, categorical_feature=cats), rounds)
            out[col] = b.predict(Xt)
        out["gbm_win"] = out[[c for c, _ in windows]].mean(axis=1)
        return out

    return fp


def selftest(panel, feats, threads, log) -> dict:
    q = R.quarters()[16]
    t = C.test_day(q)
    res = {"written": C.stamp(), "tests": []}
    try:
        C.assert_inputs_public(feats + ["gap"])
        raise C.LeakError("label column accepted as an input")
    except C.LeakError as e:
        if "accepted" in str(e):
            raise
        res["tests"].append({"test": "label_input_refused", "ok": True})
    win = (("b1", 1),)
    dd = pd.to_datetime(panel["delivery_date"])
    te = panel[dd == t]
    p0 = make_fp(panel, feats, threads, 30, win)(None, te, q)["b1"].to_numpy()
    noisy = C.noise_after(panel, C.label_cut(q), t - pd.Timedelta(days=1), feats)
    te1 = noisy[dd == t]
    p1 = make_fp(noisy, feats, threads, 30, win)(None, te1, q)["b1"].to_numpy()
    res["tests"].append(C.check_same(p0, p1, f"gbm_a_injected_lookahead {t.date()}"))
    d = R.RESULTS / "v14"
    d.mkdir(parents=True, exist_ok=True)
    (d / "selftest_gbm_a.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"self-test gbm a: {len(res['tests'])} checks passed")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=G1.N_ROUNDS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest-only", action="store_true")
    a = ap.parse_args()
    feats = R.base_features()
    panel = R.load_panel(columns=list(dict.fromkeys(["delivery_hour", "zone", "delivery_date", "gap"] + feats)))
    for c in feats:
        if panel[c].dtype.kind == "f":
            panel[c] = panel[c].astype(np.float32)
    selftest(panel, feats, a.threads, R.log)
    if a.selftest_only:
        return
    name = "v14a_gbm" + ("_smoke" if a.smoke else "")
    qs = R.quarters()[3:5] if a.smoke else None
    R.run_rows(name, make_fp(panel, feats, a.threads, 30 if a.smoke else a.rounds), panel, qs=qs)
    if not a.smoke:
        R.done("v14a_gbm")


if __name__ == "__main__":
    main()
