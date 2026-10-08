"""Costs at 1.00 USD per MWh, and results by season and by half-year, for the four strongest rows of the lab's
master table (C deep, B gradient boosting, C gradient boosting, B deep). Robustness check, OBJECTIVES.md addendum
with idea 13 (commit b31b516). Idea 13 gets the same treatment in summary.py, which builds it.

    .venv/bin/python robust/stress_seasons.py          (on gene from ~/nyiso-us; `smoke` for synthetic data)

Positions are the lead modeller's, exported for the lab after the choice fix (side/lead_positions_2021_2023.parquet,
read by side/lab.py load_lead); scoring is side/lab.py's (evaluate, stats, day_pnl). Build years only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rb  # noqa: E402

import numpy as np  # noqa: E402


def row(F, mw, lab_net=None) -> dict:
    res = rb.score(F, mw)
    out = {**rb.brief(res), **rb.stress_and_seasons(F, mw)}
    out["lab_three_year_net"] = lab_net
    out["reproduces_lab"] = None if lab_net is None else bool(round(out["three_year_net"]) == round(lab_net))
    return out


def main():
    F, *_ = rb.frame()
    labres = rb.lab_results() or {"strategies": {}}
    strats = rb.lead_strats(F)
    out = {}
    for name in rb.TOP4:
        if name not in strats:
            out[name] = {"pending": "not in side/lead_positions_2021_2023.parquet"}
            continue
        lab_net = labres["strategies"].get(name, {}).get("three_year", {}).get("net_usd")
        out[name] = row(F, rb.mw_of(F, strats[name]), lab_net)
        rb.log(name, out[name]["three_year_net"], "at 1.00:", out[name]["stress_1.00"]["three_year"]["net_usd"])
    rb.write_json(rb.OUT / "stress_seasons.json", {
        "what": "costs at 1.00 USD per MWh and results by season and half-year, 2021 to 2023 walk-forward, 1 MW",
        "lab_results_written": (labres.get("meta") or {}).get("written"), "strategies": out})
    rb.log("wrote", rb.OUT / "stress_seasons.json")


def smoke():
    import pandas as pd

    import lab as LB
    rng = np.random.default_rng(2)
    days = pd.date_range("2021-01-01", "2023-12-31", freq="D")
    hours = pd.date_range("2021-01-01", "2024-01-01", freq="h", tz="America/New_York", inclusive="left")
    px = pd.DataFrame({"delivery_hour": np.repeat(hours, 11), "zone": np.tile(LB.ZONES, len(hours))})
    px["da_lbmp"] = 30.0
    px["rt_lbmp"] = 30.0 + rng.normal(-1, 8, len(px))
    F = LB.Frame(px)
    assert F.ND == len(days)
    r = row(F, -np.ones(F.NH))
    assert set(r["seasons"]) == set(rb.SEASONS) and len(r["half_years"]) == 6
    assert r["stress_1.00"]["three_year"]["net_usd"] < r["three_year_net"]
    rb.log("smoke ok: three-year", r["three_year_net"], "at 1.00", r["stress_1.00"]["three_year"]["net_usd"])


if __name__ == "__main__":
    smoke() if len(sys.argv) > 1 and sys.argv[1] == "smoke" else main()
