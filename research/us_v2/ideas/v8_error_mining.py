"""V8: error mining. A second model trained on where the first model's errors cluster.

Inside each training window only (never the test quarter): the base model's out-of-fold residuals are
made by splitting the 3-year window into its 3 years and predicting each year from a model fit on the
other two. Two second stages, each a setting chosen per quarter on earlier out-of-sample quarters:
    cells   residual mean per cell (zone x 4-hour block x season x load-forecast tercile, tercile cuts
            from the training rows), shrunk n/(n+K) toward 0 and kept only when |t| >= 2
    gbm2    a shallow LightGBM on the residual with cluster inputs only (zone_code, hour, month,
            lf_h_rel, and wx_temp_f when present), 150 rounds
pred = base (fit on the whole window) + second-stage correction. features() adds nothing: the cluster
keys are base columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ideas_common as C

NAME = "V8"
K = 200.0
T_MIN = 2.0
SETTINGS = ["cells", "gbm2"]
ROWS = {"V8_error_mining": dict(features="base", rule="two_sided", second_stage=SETTINGS)}
CLUSTER_COLS = ["zone_code", "hour", "month", "lf_h_rel", "wx_temp_f"]


def features(panel_index, panel=None, store=None) -> pd.DataFrame:
    return pd.DataFrame(index=panel_index)


def _year_folds(train: pd.DataFrame) -> np.ndarray:
    d = pd.to_datetime(train["delivery_date"])
    start = d.min()
    return np.minimum(((d - start).dt.days // 365).to_numpy(), 2)


def oof_residuals(train, base_fp, feature_cols, target="gap") -> np.ndarray:
    fold = _year_folds(train)
    pred = np.full(len(train), np.nan)
    for k in np.unique(fold):
        tr, te = train[fold != k], train[fold == k]
        pred[fold == k] = base_fp(tr, te, target, feature_cols)
    return train[target].to_numpy(float) - pred


def _cells(df: pd.DataFrame, cuts) -> pd.Series:
    blk = (df["hour"].to_numpy() // 4).astype(int)
    season = ((df["month"].to_numpy() % 12) // 3).astype(int)
    terc = np.digitize(df["lf_h_rel"].to_numpy(float), cuts) if "lf_h_rel" in df else np.zeros(len(df), int)
    return pd.Series([f"{z}|{b}|{s}|{t}" for z, b, s, t in zip(df["zone"], blk, season, terc)], index=df.index)


def cell_correction(train, resid, test) -> np.ndarray:
    cuts = np.nanquantile(train["lf_h_rel"].to_numpy(float), [1 / 3, 2 / 3]) if "lf_h_rel" in train else []
    r = pd.DataFrame({"cell": _cells(train, cuts).to_numpy(), "r": np.clip(resid, -250, 250)}).dropna()
    s = r.groupby("cell")["r"].agg(["mean", "std", "count"])
    t = s["mean"] / (s["std"] / np.sqrt(s["count"])).replace(0, np.nan)
    corr = (s["mean"] * s["count"] / (s["count"] + K)).where(t.abs() >= T_MIN, 0.0)
    return _cells(test, cuts).map(corr).fillna(0.0).to_numpy(float)


def gbm2_correction(train, resid, test) -> np.ndarray:
    import lightgbm as lgb
    from learner import PARAMS
    cols = [c for c in CLUSTER_COLS if c in train]
    ok = ~np.isnan(resid)
    params = dict(PARAMS, num_leaves=8, min_data_in_leaf=1000, learning_rate=0.05)
    m = lgb.train(params, lgb.Dataset(train.loc[ok, cols].astype(float), np.clip(resid[ok], -250, 250)),
                  num_boost_round=150)
    return m.predict(test[cols].astype(float))


def fit_predict(train, test, base_fp, feature_cols, target="gap") -> pd.DataFrame:
    resid = oof_residuals(train, base_fp, feature_cols, target)
    base = base_fp(train, test, target, feature_cols)
    return pd.DataFrame({"pred_base": base,
                         "pred_cells": base + cell_correction(train, resid, test),
                         "pred_gbm2": base + gbm2_correction(train, resid, test)}, index=test.index)


rule = C.two_sided
