"""End-to-end smoke runs of rehearsal.py on the synthetic tables (dates shifted into the synthetic
range, 2 regression configurations, 30 trees): the v2 gbm stage, rescore, and the held-out stage's code
path. Checks every section of the JSON, that the holdout rows in the synthetic files never reached the
panel, and that the real held-out stage refuses to run while the study is locked."""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

import gbm
import lock
import panel as P
import rehearsal as R


@pytest.fixture
def shifted(synth_root, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "FIRST_DAY", dt.date(2023, 5, 1))
    monkeypatch.setattr(R, "BUILD", (dt.date(2023, 5, 1), dt.date(2023, 12, 31)))
    monkeypatch.setattr(R, "TUNE_VALID", (dt.date(2023, 8, 1), dt.date(2023, 9, 30)))
    monkeypatch.setattr(R, "REH", (dt.date(2023, 10, 1), dt.date(2023, 12, 31)))
    monkeypatch.setattr(R, "B_START", dt.date(2023, 5, 1))
    monkeypatch.setattr(R, "SITES_THROUGH", dt.date(2023, 9, 30))
    monkeypatch.setattr(R, "SETS", {"base": ("A", None), "weather": ("B", dt.date(2023, 5, 1)), "outages": ("D", None)})
    monkeypatch.setattr(R, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(R, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(R, "OUT", tmp_path / "results" / "rehearsal_2023_v2.json")
    monkeypatch.setattr(R, "SPIKE_CONFIG", tmp_path / "spike_config.json")
    monkeypatch.setattr(R, "CHOICES", tmp_path / "frozen_choices.json")
    monkeypatch.setattr(R, "WORKERS", 2)
    monkeypatch.setattr(gbm, "GRID", gbm.GRID[:2])
    monkeypatch.setattr(gbm, "N_ROUNDS", 30)
    return tmp_path


def test_stage_gbm_v2_rescore_and_heldout_path(synth_root, shifted, monkeypatch):
    tmp_path = shifted
    out = R.stage_gbm(P.LockedStore(synth_root))
    res = json.loads((tmp_path / "results" / "rehearsal_2023_v2.json").read_text())
    for k in ("panel", "tuning", "pairs", "walk_forward", "ideas", "fee_stress", "mde", "baseline", "always_supply",
              "context", "value_of_the_spike_filter", "spike_rule", "ablation_gen"):
        assert k in res, k
    t = res["tuning"]["spike"]
    assert t["configurations"] == len(gbm.S_GRID) * len(gbm.P_STAR_GRID) <= 8
    a = res["ideas"]["A"]
    assert a["verdict"] in ("pays", "doesn't pay", "inconclusive") and a["rule"]["S"] in gbm.S_GRID
    assert 0 <= a["rule"]["sat_out_share"] <= 1
    for k in ("A", "B", "C", "D", "baseline", "always_supply"):
        c = res["context"][k]
        assert c["bankroll_usd"] == 500_000 and {"sharpe_annualised", "max_drawdown_usd", "return_on_bankroll",
                                                 "return_over_max_drawdown"} <= set(c)
    assert res["scored_days"] == 92 and len(res["walk_forward"]["spike"]) == 3
    sc = json.loads((tmp_path / "spike_config.json").read_text())
    ch = json.loads((tmp_path / "frozen_choices.json").read_text())
    assert sc["S_usd_per_mwh"] == res["spike_rule"]["S"] == ch["spike"]["S"]
    assert set(ch["regression_configs"]) == {"base", "weather", "outages"} and ch["pairs"] == res["pairs"]["chosen"]
    p = pd.read_parquet(tmp_path / "cache" / "panel_base_20231231.parquet")
    assert p["delivery_hour"].max() < pd.Timestamp("2024-01-01", tz="America/New_York")
    assert out["seconds"] >= 0

    R.stage_rescore(P.LockedStore(synth_root))                         # scores again from cached predictions
    res2 = json.loads((tmp_path / "results" / "rehearsal_2023_v2.json").read_text())
    assert "rescored" in res2 and res2["tuning"] == res["tuning"]
    assert res2["ideas"]["A"]["mean_daily_diff"] == pytest.approx(res["ideas"]["A"]["mean_daily_diff"])
    assert res2["value_of_the_spike_filter"] == res["value_of_the_spike_filter"]

    # the held-out stage's code path, on synthetic months inside 2023 (the real one is locked, below)
    monkeypatch.setattr(R, "HOLDOUT", (dt.date(2023, 11, 1), dt.date(2023, 12, 31)))
    monkeypatch.setattr(P, "PARQUET", synth_root)
    monkeypatch.setattr(P.LockedStore.__init__, "__defaults__", (synth_root, None, None, 3))
    R.stage_heldout()
    files = list((tmp_path / "results").glob("heldout_202311_202312_nofreeze_*.json"))
    assert len(files) == 1
    h = json.loads(files[0].read_text())
    assert h["run"] == "held-out" and h["scored_days"] == 61 and h["ideas"]["A"]["verdict"] != "not run"
    assert h["walk_forward"]["spike"][0]["train_last"] == "2023-10-30"


def test_real_heldout_stage_is_locked(monkeypatch):
    monkeypatch.delenv(lock.ENV_FLAG, raising=False)
    with pytest.raises(lock.HoldoutLocked):
        R.stage_heldout()
