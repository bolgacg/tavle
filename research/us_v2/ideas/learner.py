"""The base learner the ideas share: one fixed LightGBM regression (v1 gbm.py FIXED settings, 400 rounds,
target clipped to +-250 as v1's grid option). Fixed, so it adds no settings to the tries count.

    base_fit_predict(train, test, target, feature_cols, sample_weight=None) -> np.ndarray
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

SEED = 20261007
PARAMS = dict(objective="regression", learning_rate=0.03, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, max_bin=255, seed=SEED, deterministic=True,
              force_row_wise=True, num_threads=int(os.environ.get("US_LGBM_THREADS", "4")), verbose=-1,
              min_data_in_leaf=200)
N_ROUNDS = 400
CLIP = 250.0
LABELS = {"gap", "y_da_lbmp", "y_rt_lbmp", "label_published_at", "y_energy", "y_cong", "y_loss"}


def check_features(cols) -> None:
    bad = [c for c in cols if c in LABELS or c.startswith("y_")]
    if bad:
        raise ValueError(f"label columns used as features: {bad}")


def base_fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, feature_cols,
                     sample_weight=None, n_rounds: int = N_ROUNDS) -> np.ndarray:
    import lightgbm as lgb
    feature_cols = list(feature_cols)
    check_features(feature_cols)
    y = train[target].to_numpy(float)
    ok = ~np.isnan(y)
    X = train.loc[ok, feature_cols].astype(float)
    w = None if sample_weight is None else np.asarray(sample_weight, float)[ok]
    if ok.sum() < 1000:
        return np.full(len(test), np.nan)
    ds = lgb.Dataset(X, np.clip(y[ok], -CLIP, CLIP), weight=w, free_raw_data=True)
    m = lgb.train(PARAMS, ds, num_boost_round=n_rounds)
    return m.predict(test[feature_cols].astype(float))
