"""CPU lane: every LightGBM model of v2 on the rolling quarters, features loaded once (rolling.py).

Per quarter, on the 3-year window (v1 base features, panel.BASE_FEATURES):
  mean      regression of the gap, the frozen v1 base configuration (15 leaves, 200 rows per leaf, clip 250)
            -> C GBM (V1) and the side + expected profit of V3
  ens4      mean of the last 4 windows' mean models on this quarter (this one and the 3 before) -> V11
  p_s25, p_s50, p_s100   spike classifiers P(gap >= S), v1 SpikeClassifier -> A spike (V1)
  q10, q50, q90          quantile regressions of the gap -> V3 tail-risk sizing
  cond_k, dag_k          V2: a regression of the gap that also sees the delivery hour's day-ahead price
            (cond_da, cond_da_ratio = DA / da_d0_h); its prediction at 9 hypothetical day-ahead levels
            dag_k = da_d0_h x DA_MULT[k] (the same hour's day-ahead price on D, public at 05:00 on D).
            The realised day-ahead price of D+1 is an INPUT of the conditional model only in training;
            at bid time the limit price is chosen from the grid alone (strategies_v2.v2_limits).

    python gbm_rolling.py [--threads 2] [--smoke]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

import rolling as R

os.environ.setdefault("US_LGBM_THREADS", "2")
import lightgbm as lgb  # noqa: E402

import gbm as G1  # noqa: E402  (v1)

BASE_CFG = dict(num_leaves=15, min_data_in_leaf=200, target_clip=250.0)      # v1 frozen_choices.json "base"
S_GRID = G1.S_GRID                                                            # (25, 50, 100)
QUANTS = (0.10, 0.50, 0.90)
DA_MULT = (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0, 3.0)
ENS = 4


def _params(threads, **kw):
    return {**G1.FIXED, "num_threads": threads, **kw}


def make_fit_predict(features: list[str], threads: int, n_rounds: int, models_dir):
    hist: list[str] = []                       # model files of earlier quarters (for ens4), in order
    models_dir.mkdir(parents=True, exist_ok=True)
    cats = [c for c in G1.CATEGORICAL if c in features]

    def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, q) -> pd.DataFrame:
        R.assert_pre2024(tr["delivery_hour"], "gbm train")
        R.assert_pre2024(te["delivery_hour"], "gbm test")
        X, Xt = tr[features].astype(float), te[features].astype(float)
        y = tr["gap"].to_numpy(float)
        out = pd.DataFrame(index=te.index)
        t = time.time()
        # mean model (C, V3, V11)
        p = _params(threads, num_leaves=BASE_CFG["num_leaves"], min_data_in_leaf=BASE_CFG["min_data_in_leaf"])
        b = lgb.train(p, lgb.Dataset(X, np.clip(y, -BASE_CFG["target_clip"], BASE_CFG["target_clip"]),
                                     categorical_feature=cats), n_rounds)
        f = models_dir / f"mean_{q[2]}.txt"
        b.save_model(str(f))
        out["mean"] = b.predict(Xt)
        prev = [m for m in sorted(models_dir.glob("mean_*.txt")) if m.stem.split("_")[1] < q[2]][-(ENS - 1):]
        ens = [out["mean"].to_numpy()] + [lgb.Booster(model_file=str(m)).predict(Xt) for m in prev]
        out["ens4"] = np.mean(ens, 0)
        out["ens_n"] = len(ens)
        # spike classifiers (A)
        for S in S_GRID:
            pb = _params(threads, objective="binary", **G1.SPIKE_FIXED)
            bs = lgb.train(pb, lgb.Dataset(X, (y >= S).astype(float), categorical_feature=cats), n_rounds)
            out[f"p_s{S}"] = bs.predict(Xt)
        # quantiles (V3)
        for a in QUANTS:
            pq = _params(threads, objective="quantile", alpha=a, num_leaves=15, min_data_in_leaf=200)
            bq = lgb.train(pq, lgb.Dataset(X, y, categorical_feature=cats), n_rounds)
            out[f"q{int(round(a * 100)):02d}"] = bq.predict(Xt)
        # conditional on the delivery hour's day-ahead price (V2)
        def cond_X(df, da):
            Z = df[features].astype(float).copy()
            Z["cond_da"] = da
            Z["cond_da_ratio"] = da / df["da_d0_h"].astype(float).where(df["da_d0_h"].abs() > 1, np.nan)
            return Z
        bc = lgb.train(p, lgb.Dataset(cond_X(tr, tr["y_da_lbmp"].to_numpy(float)),
                                      np.clip(y, -BASE_CFG["target_clip"], BASE_CFG["target_clip"]),
                                      categorical_feature=cats), n_rounds)
        base = te["da_d0_h"].to_numpy(float)
        for k, m in enumerate(DA_MULT):
            lvl = base * m
            out[f"dag_{k}"] = lvl
            out[f"cond_{k}"] = bc.predict(cond_X(te, lvl))
        out["gbm_seconds"] = round(time.time() - t, 1)
        return out

    return fit_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=int(os.environ.get("US_LGBM_THREADS", "2")))
    ap.add_argument("--rounds", type=int, default=G1.N_ROUNDS)
    ap.add_argument("--smoke", action="store_true", help="two quarters, 30 rounds, name gbm_smoke")
    a = ap.parse_args()
    feats = R.base_features()
    need = ["delivery_hour", "zone", "delivery_date", "gap", "y_da_lbmp"] + feats
    panel = R.load_panel(columns=list(dict.fromkeys(need)))
    R.log(f"panel {panel.shape}, {panel['delivery_date'].min().date()}..{panel['delivery_date'].max().date()}")
    name = "gbm_smoke" if a.smoke else "gbm"
    qs = R.quarters()[3:5] if a.smoke else None
    fp = make_fit_predict(feats, a.threads, 30 if a.smoke else a.rounds, R.PREDS / f"{name}_models")
    R.run_rows(name, fp, panel, qs=qs)
    if not a.smoke:
        R.done("v2_gbm")


if __name__ == "__main__":
    main()
