"""End-to-end smoke run of rehearsal.stage_gbm on the synthetic tables, with dates shifted into the
synthetic range, 2 configurations and 30 trees. Checks the JSON has every section and that the
holdout rows in the synthetic files never reached the panel."""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd

import gbm
import panel as P
import rehearsal as R


def test_stage_gbm_on_synthetic(synth_root, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "BUILD", (dt.date(2023, 5, 1), dt.date(2023, 12, 31)))
    monkeypatch.setattr(R, "TUNE_VALID", (dt.date(2023, 8, 1), dt.date(2023, 9, 30)))
    monkeypatch.setattr(R, "REH", (dt.date(2023, 10, 1), dt.date(2023, 12, 31)))
    monkeypatch.setattr(R, "B_START", dt.date(2023, 5, 1))
    monkeypatch.setattr(R, "SETS", {"base": ("A", None), "gen": ("A_gen", None),
                                    "weather": ("B", dt.date(2023, 5, 1)), "outages": ("D", None)})
    monkeypatch.setattr(R, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(R, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(R, "OUT", tmp_path / "results" / "rehearsal_2023.json")
    monkeypatch.setattr(R, "WORKERS", 2)
    monkeypatch.setattr(gbm, "GRID", gbm.GRID[:2])
    monkeypatch.setattr(gbm, "N_ROUNDS", 30)
    out = R.stage_gbm(P.LockedStore(synth_root))
    res = json.loads((tmp_path / "results" / "rehearsal_2023.json").read_text())
    for k in ("panel", "tuning", "pairs", "walk_forward", "ideas", "fee_stress", "mde", "baseline"):
        assert k in res, k
    assert res["ideas"]["A"]["verdict"] in ("pays", "doesn't pay", "inconclusive")
    for k in ("B", "C", "D"):
        assert res["ideas"][k]["verdict"] in ("pays", "doesn't pay", "inconclusive", "not run")
    a = res["ideas"]["A"]
    for k in ("interval", "profit_per_mwh", "break_even_fee"):
        assert k in a or k in a["idea_money"], k
    assert a["n_days"] == 92 and res["scored_days"] == 92
    assert len(res["walk_forward"]["base"]) == 3
    p = pd.read_parquet(tmp_path / "cache" / "panel_base.parquet")
    assert p["delivery_hour"].max() < pd.Timestamp("2024-01-01", tz="America/New_York")
    assert out["seconds"] >= 0
    R.stage_rescore(P.LockedStore(synth_root))                         # scores again from cached predictions
    res2 = json.loads((tmp_path / "results" / "rehearsal_2023.json").read_text())
    assert "idea_alone_interval_95" in res2["ideas"]["A"] and "rescored" in res2
    assert res2["ideas"]["A"]["mean_daily_diff"] == res["ideas"]["A"]["mean_daily_diff"]
    assert res2["tuning"] == res["tuning"]
