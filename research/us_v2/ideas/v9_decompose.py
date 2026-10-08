"""V9: gap decomposition. NYISO LBMP = energy + loss + congestion parts, so the gap (rt - da) splits
into an energy gap, a loss gap and a congestion gap. Each part gets its own model (same learner and
features); the predicted gap is their sum, traded with the v1 two-sided rule.

targets(panel) builds y_energy, y_cong, y_loss for each row's delivery hour from parquet_v2/prices_zone
(sign-corrected congestion, as the panel's da_d0_cong_h). They are labels: learner.check_features
refuses them as inputs. The identity gap == y_energy + y_cong + y_loss is checked (|diff| <= 0.05).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ideas_common as C

NAME = "V9"
PARTS = {"y_energy": ("rt_energy", "da_energy"), "y_cong": ("rt_congestion", "da_congestion"),
         "y_loss": ("rt_loss", "da_loss")}
ROWS = {"V9_decompose": dict(features="base", rule="two_sided", targets=list(PARTS))}


def features(panel_index, panel=None, store=None) -> pd.DataFrame:
    return pd.DataFrame(index=panel_index)


def targets(panel: pd.DataFrame, store=None) -> pd.DataFrame:
    C.assert_pre_holdout(panel["delivery_hour"])
    cols = ["delivery_hour", "zone"] + [c for pair in PARTS.values() for c in pair]
    pz = store["prices_zone"][cols] if store is not None else C.read_table("prices_zone", "delivery_hour", columns=cols)
    pz = pz[pz["zone"].isin(C.ZONES)].copy()
    pz["k"] = pz["delivery_hour"].dt.tz_convert("UTC")
    pz = pz.groupby(["k", "zone"]).mean(numeric_only=True)
    key = pd.MultiIndex.from_arrays([panel["delivery_hour"].dt.tz_convert("UTC"), panel["zone"]])
    out = pd.DataFrame(index=panel.index)
    for y, (rt, da) in PARTS.items():
        out[y] = (pz[rt] - pz[da]).reindex(key).to_numpy()
    if "gap" in panel:
        diff = (panel["gap"] - out.sum(axis=1, min_count=3)).abs()
        bad = (diff > 0.05).sum()
        if bad > 0.001 * diff.notna().sum():
            raise ValueError(f"V9: gap != energy + congestion + loss on {bad} rows (sign convention?)")
    return out


def fit_predict(train, test, base_fp, feature_cols) -> pd.DataFrame:
    parts = {y: base_fp(train, test, y, feature_cols) for y in PARTS}
    out = pd.DataFrame({f"pred_{y[2:]}": v for y, v in parts.items()}, index=test.index)
    out["pred"] = out.sum(axis=1)
    return out


rule = C.two_sided
