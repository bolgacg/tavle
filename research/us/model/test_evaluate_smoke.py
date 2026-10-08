"""evaluate.py end to end on the synthetic tables, with three two-month periods standing in for 2021 to
2023: every choice for a period is made on the period before (defaults for the first), every strategy
gets per-period and total figures, and the bar flags are consistent."""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

import evaluate as E
import gbm
import panel as P
import rehearsal as R


def test_evaluate_on_synthetic(synth_root, tmp_path, monkeypatch):
    monkeypatch.setattr(R, "BUILD", (dt.date(2023, 3, 1), dt.date(2023, 12, 31)))
    monkeypatch.setattr(R, "SITES_THROUGH", dt.date(2023, 9, 30))
    monkeypatch.setattr(R, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(R, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(R, "OUT", tmp_path / "results" / "no_v2.json")             # no v2 rehearsal to reuse here
    monkeypatch.setattr(R, "WORKERS", 2)
    monkeypatch.setattr(gbm, "GRID", gbm.GRID[:2])
    monkeypatch.setattr(gbm, "N_ROUNDS", 20)
    monkeypatch.setattr(E, "PERIODS", [("p1", dt.date(2023, 7, 1), dt.date(2023, 8, 31)),
                                       ("p2", dt.date(2023, 9, 1), dt.date(2023, 10, 31)),
                                       ("p3", dt.date(2023, 11, 1), dt.date(2023, 12, 31))])
    monkeypatch.setattr(E, "PRE", (dt.date(2023, 5, 1), dt.date(2023, 6, 30)))
    monkeypatch.setattr(E, "B_FIRST", dt.date(2023, 7, 15))
    monkeypatch.setattr(E, "B_TRAIN", dt.date(2023, 4, 1))
    monkeypatch.setattr(E, "OUT", tmp_path / "results" / "strategy_list")
    (tmp_path / "results").mkdir(parents=True)
    panel, cols, _ = E.load(P.LockedStore(synth_root))
    assert set(cols["outages_by_period"]) == {"p1", "p2", "p3"}
    import walkforward as W
    W.CHOICE_LOG.clear()
    E.compute(panel, cols)
    res = E.score(panel, cols)
    # every choice went through the guard, which raises on any outcome not public at the period's first bid
    kinds = {e["what"].split()[0] for e in W.CHOICE_LOG}
    assert {"select", "storm", "pairs", "zones"} <= kinds
    starts = {"p1": "2023-07-01", "p2": "2023-09-01", "p3": "2023-11-01"}
    for e in W.CHOICE_LOG:
        if e["last_delivery_day"]:
            assert pd.Timestamp(e["last_delivery_day"]) <= pd.Timestamp(e["period_start"]) - pd.Timedelta(days=2)
    assert all(e["period_start"] in starts.values() for e in W.CHOICE_LOG)
    names = {r["strategy"] for r in res["strategies"]}
    ch = res["choices"]
    assert {"always_supply", "baseline", "A_hourly_mean", "A_regression_v1", "A_spike_gbm", "B_gbm", "C_gbm",
            "D_gbm", "storm_day_filter", "A_spike_tail_S100_gbm", "risk_sized_supply_gbm", "zone_subset_supply"} <= names
    assert res["tries"]["strategies_in_this_list"] == len(res["strategies"])
    assert "storm_day_flip" in names
    for lab, lo in (("p1", "2023-07-01"), ("p2", "2023-09-01"), ("p3", "2023-11-01")):
        how = ch[lab]["storm"]["how"]                  # the year before, ending two days before the period
        assert how.startswith("default") or how.rstrip().endswith(str((pd.Timestamp(lo) - pd.Timedelta(days=2)).date()))
    assert len(ch["p2"]["zones"]["zones"]) <= E.MAX_ZONES
    assert ch["p2"]["zones"]["chosen_on"].endswith("2023-08-30")             # ends two days before the period
    rs = next(r for r in res["strategies"] if r["strategy"] == "risk_sized_supply_gbm")
    al = next(r for r in res["strategies"] if r["strategy"] == "always_supply")
    assert 0 < rs["total"]["mwh"] < al["total"]["mwh"]                                 # sized below 1 MW
    assert ch["p1"]["base"]["how"].startswith("default") and ch["p1"]["spike"]["S"] == E.DEFAULT_SPIKE[0]
    assert "p1" in ch["p2"]["base"]["how"] and "p2" in ch["p3"]["spike"]["how"]       # chosen on the period before
    assert ch["p1"]["pairs"]["chosen_on"].startswith("2023-05-01")
    done = [r for r in res["strategies"] if "total" in r]
    assert [r["rank"] for r in done] == list(range(1, len(done) + 1))
    assert all(done[i]["total"]["net_usd"] >= done[i + 1]["total"]["net_usd"] for i in range(len(done) - 1))
    for r in done:
        assert set(r["years"]) <= {"p1", "p2", "p3"}
        t = r["total"]
        assert t["net_usd"] == pytest.approx(sum(y["net_usd"] for y in r["years"].values()), abs=0.05)
        assert r["passes_bar"] == (r["status"] == "complete" and all(r["bar"].values()))
        for k in ("profit_per_mwh", "sharpe_annualised", "max_drawdown_usd", "return_on_bankroll_annualised",
                  "worst_day", "months_positive", "net_usd_at_0_50_stress"):
            assert k in t
    b = next(r for r in done if r["strategy"] == "B_gbm")
    assert b["years"]["p1"]["first"] == "2023-07-15"
    E.write_md(res, tmp_path / "t.md")
    assert "| 1 |" in (tmp_path / "t.md").read_text()
    # the cached predictions are reused: a second score gives the same numbers
    res2 = E.score(panel, cols)
    assert json.dumps(res2["strategies"], default=str) == json.dumps(res["strategies"], default=str)
    assert pd.read_parquet(tmp_path / "cache" / "panel_base_20231231.parquet")["delivery_hour"].max() < \
        pd.Timestamp("2024-01-01", tz="America/New_York")
