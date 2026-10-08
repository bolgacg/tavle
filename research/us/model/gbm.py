"""Gradient boosting model (LightGBM), the verdict model inside every idea (CONTRACT.md, "Model interface").

    m = GBMModel(features, config)     config: one entry of GRID (at most 8, CONTRACT.md)
    m.fit(panel_train)                 rows with a known gap; the label is `gap` (USD/MWh)
    m.predict(panel) -> Series         predicted gap aligned to panel.index

Squared error is the objective because the trading rules compare the EXPECTED gap with the fee; a
robust loss would estimate something nearer the median, which for this right-skewed gap is too
optimistic for supply. The grid's `target_clip` option (train on the gap clipped to +-250) is the one
concession to the spikes, and the inner validation decides whether it helps.
"""
from __future__ import annotations

import os

import lock

import lightgbm as lgb
import numpy as np
import pandas as pd

from panel import CATEGORICAL

SEED = 20261006
# Threads: 4 by default; US_LGBM_THREADS lowers it when gene's cores are shared (4 threads on 2 free cores
# made refits 25 times slower). Results are deterministic for a given thread count.
FIXED = dict(objective="regression", learning_rate=0.03, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=1.0, max_bin=255, seed=SEED, deterministic=True, force_row_wise=True,
             num_threads=int(os.environ.get("US_LGBM_THREADS", "4")), verbose=-1)
N_ROUNDS = 400
GRID = [dict(num_leaves=nl, min_data_in_leaf=md, target_clip=tc)
        for nl in (15, 63) for md in (200, 2000) for tc in (None, 250.0)]
assert len(GRID) <= 8


def config_name(c: dict) -> str:
    return f"leaves{c['num_leaves']}_minleaf{c['min_data_in_leaf']}_clip{c['target_clip'] or 'none'}"


class GBMModel:
    def __init__(self, features: list[str], config: dict | None = None, name: str = "gbm",
                 n_rounds: int | None = None, num_threads: int | None = None):
        self.features = list(features)
        self.config = dict(config or GRID[0])
        self.name = name
        self.n_rounds = n_rounds or N_ROUNDS
        self.params = {**FIXED, "num_leaves": self.config["num_leaves"],
                       "min_data_in_leaf": self.config["min_data_in_leaf"]}
        if num_threads:
            self.params["num_threads"] = num_threads
        self.booster: lgb.Booster | None = None
        self.n_train = 0

    def _X(self, panel: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.features if c not in panel.columns]
        if missing:
            raise KeyError(f"panel lacks features {missing[:5]}")
        return panel[self.features].astype(float)

    def fit(self, panel_train: pd.DataFrame) -> None:
        lock.assert_build_only(panel_train["delivery_hour"])
        tr = panel_train[panel_train["gap"].notna()]
        y = tr["gap"].to_numpy(float)
        if self.config.get("target_clip"):
            c = float(self.config["target_clip"])
            y = np.clip(y, -c, c)
        cats = [c for c in CATEGORICAL if c in self.features]
        ds = lgb.Dataset(self._X(tr), y, categorical_feature=cats, free_raw_data=True)
        self.booster = lgb.train(self.params, ds, num_boost_round=self.n_rounds)
        self.n_train = len(tr)

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        lock.assert_build_only(panel["delivery_hour"])
        if self.booster is None:
            raise RuntimeError("fit first")
        return pd.Series(self.booster.predict(self._X(panel)), index=panel.index, name="pred")

    def importance(self) -> dict[str, float]:
        g = self.booster.feature_importance("gain")
        tot = g.sum() or 1.0
        return dict(sorted(((f, float(v / tot)) for f, v in zip(self.features, g)), key=lambda kv: -kv[1]))


# ----------------------------------------------------------------- idea A: the spike classifier
# Addendum of 6 Oct (late evening): A supplies in every zone-hour except where P(spike) > p_star,
# spike = gap >= S. Configuration budget for this model (at most 8, CONTRACT.md): 3 thresholds S x 2 cuts
# p_star = 6, with the LightGBM settings fixed (15 leaves, 200 rows per leaf, 400 trees at 0.03, the
# base regression's 2022 choice without its clip, which has no meaning for a classifier).
# Why p_star near 2 to 5%: a normal hour earns about 1 to 2 USD/MWh for supply and a spike costs at
# least S (25 to 100) and often several times that, so sitting out pays once P(spike) is a few percent.
S_GRID = (25, 50, 100)
P_STAR_GRID = (0.02, 0.05)
SPIKE_FIXED = dict(num_leaves=15, min_data_in_leaf=200)


class SpikeClassifier:
    """LightGBM binary model of P(gap >= S). predict() returns that probability (not a gap)."""

    def __init__(self, features: list[str], S: float, name: str = "gbm_spike", n_rounds: int | None = None,
                 num_threads: int | None = None):
        self.features = list(features)
        self.S = float(S)
        self.name = name
        self.n_rounds = n_rounds or N_ROUNDS
        self.params = {**FIXED, **SPIKE_FIXED, "objective": "binary"}
        if num_threads:
            self.params["num_threads"] = num_threads
        self.booster: lgb.Booster | None = None
        self.base_rate = None

    def fit(self, panel_train: pd.DataFrame) -> None:
        lock.assert_build_only(panel_train["delivery_hour"])
        tr = panel_train[panel_train["gap"].notna()]
        y = (tr["gap"].to_numpy(float) >= self.S).astype(float)
        self.base_rate = float(y.mean())
        cats = [c for c in CATEGORICAL if c in self.features]
        ds = lgb.Dataset(tr[self.features].astype(float), y, categorical_feature=cats, free_raw_data=True)
        self.booster = lgb.train(self.params, ds, num_boost_round=self.n_rounds)

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        lock.assert_build_only(panel["delivery_hour"])
        if self.booster is None:
            raise RuntimeError("fit first")
        return pd.Series(self.booster.predict(panel[self.features].astype(float)), index=panel.index, name="p_spike")

    def importance(self) -> dict[str, float]:
        return GBMModel.importance(self)
