"""V7: market-change awareness. Dated regime flags (v7_regimes.csv: NYISO rule and fleet changes
2010 to 2023, each with a source URL and a confidence) plus recency-weighted training.

A flag is on for bid day D only when the change took effect on or before D (effective_date <= D), so
a change effective on the delivery day D+1 is not yet on. Columns:
    rg_<id>                  1 if the change is in force at D, else 0 (one per csv row)
    rg_days_since_last       days since the most recent change in force at D (capped at 1500)
    rg_n_changes             changes in force at D
Recency weighting: sample_weight(train, train_end, half_life_days) = 0.5 ** (age / half_life),
age = train_end - bid_date in days. Settings tried per quarter: half-life 365, 730 and none (flat),
chosen by R.choose_setting on earlier out-of-sample quarters only.
Rows: V7_flags (base + flags, flat weights), V7_recency (base + flags, half-life chosen per quarter).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import ideas_common as C

NAME = "V7"
CSV = Path(__file__).resolve().parent / "v7_regimes.csv"
HALF_LIVES = [365.0, 730.0, None]
ROWS = {"V7_flags": dict(features="base+V7", rule="two_sided", weights="flat"),
        "V7_recency": dict(features="base+V7", rule="two_sided", weights="half-life " + str(HALF_LIVES))}


def regimes() -> pd.DataFrame:
    r = pd.read_csv(CSV, parse_dates=["effective_date"])
    assert (r["effective_date"] < pd.Timestamp(C.HOLDOUT)).all(), "regime list must stop before 2024"
    return r.sort_values("effective_date").reset_index(drop=True)


def columns() -> list[str]:
    return [f"rg_{i}" for i in regimes()["id"]] + ["rg_days_since_last", "rg_n_changes"]


def features(panel_index: pd.MultiIndex, panel: pd.DataFrame | None = None, store=None) -> pd.DataFrame:
    p = C.panel_frame(panel_index)
    r = regimes()
    D = p["bid_date"].to_numpy("datetime64[ns]")
    eff = r["effective_date"].to_numpy("datetime64[ns]")
    on = D[:, None] >= eff[None, :]
    out = pd.DataFrame(on.astype(float), columns=[f"rg_{i}" for i in r["id"]])
    last = np.where(on, eff[None, :], np.datetime64("NaT")).max(axis=1)
    since = (D - last).astype("timedelta64[D]").astype(float)
    out["rg_days_since_last"] = np.where(on.any(axis=1), np.minimum(since, 1500), 1500)
    out["rg_n_changes"] = on.sum(axis=1).astype(float)
    out.index = panel_index
    return out


def sample_weight(train: pd.DataFrame, train_end, half_life_days: float | None) -> np.ndarray:
    if half_life_days is None:
        return np.ones(len(train))
    age = (pd.Timestamp(train_end) - pd.to_datetime(train["bid_date"]).dt.tz_localize(None)
           if getattr(pd.to_datetime(train["bid_date"]).dt, "tz", None) is not None
           else pd.Timestamp(train_end) - pd.to_datetime(train["bid_date"])).dt.days.to_numpy(float)
    if (age < 0).any():
        raise ValueError("a training row is after train_end")
    return 0.5 ** (age / half_life_days)


rule = C.two_sided
