"""Shared helpers for the robustness checks registered on 6 Oct 2026 (OBJECTIVES.md, the addendum with idea 13,
commit b31b516): 20 random starts for the deep models, the settings next to each chosen setting, costs at
1.00 USD per MWh, results by season and half-year, and idea 13 (portfolio of the survivors).

Canonical code: laptop ~/projects/tavle/research/us/robust/, run copy gene ~/nyiso-us/robust/, outputs gene
~/nyiso-us/results/robust/. Nothing here writes outside results/robust/.

HOLDOUT. Every table is read through the existing locked loaders (side/lab.py read_build and load_tables,
model/panel.read_locked, model/lock.py), which filter at read time to delivery dates before 2024-01-01 and
assert it. Nothing on or after 2024-01-01 is read.

Scoring is side/lab.py's, by import (Frame, evaluate, stats, walk, load_lead): signed MW per zone-hour,
the lab's cost tables, Bo's bar and the sizing view. Nothing of it is copied here.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path(os.environ.get("ROBUST_HOME", str(Path.home() / "nyiso-us")))
for _sub in ("pipeline", "model", "side"):
    if str(HOME / _sub) not in sys.path:
        sys.path.insert(0, str(HOME / _sub))

RESULTS = HOME / "results"
SIDE = HOME / "side"
OUT = Path(os.environ.get("ROBUST_OUT", str(RESULTS / "robust")))
END = pd.Timestamp("2024-01-01", tz="America/New_York")

TOP4 = ["C_deep", "B_gbm", "C_gbm", "B_deep"]          # the strongest rows named for the checks
# Treated as fragile per their audits (coordinator's instruction, 6 Oct night), whatever the lab's flag says.
FRAGILE_BY_AUDIT = {"idea10_weather_surprise": "idea 10 audit (side/idea10_audit.md)",
                    "idea2_storm_flip": "storm audit (side/storm_audit.md)",
                    "storm_day_flip [lead]": "storm audit (side/storm_audit.md)"}
SIZING_BAR_PCT = 10.0                                   # Bo's 10 percent a year, judged at the sizing view
SEASONS = {"winter (Dec to Feb)": (12, 1, 2), "spring (Mar to May)": (3, 4, 5), "summer (Jun to Aug)": (6, 7, 8),
           "autumn (Sep to Nov)": (9, 10, 11)}
STRESS_HIGH = 1.00


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def lab():
    import lab as LB                                   # side/lab.py
    return LB


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_js))
    os.replace(tmp, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def read_json(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


# ------------------------------------------------------------------ the lab's frame and scoring
def frame():
    """(F, px, lf, wx) exactly as side/lab.py main() loads them (holdout filter at load, asserted)."""
    LB = lab()
    px, lf, wx = LB.load_tables()
    F = LB.Frame(px)
    LB.assert_build(F.h["delivery_hour"], "robust H")
    return F, px, lf, wx


def ledger_to_mw(F, led: pd.DataFrame) -> np.ndarray:
    """A model/strategies.py ledger to signed MW per row of the lab's frame. Pair rows ("i|j", idea C) are split
    into +pos in zone i and -pos in zone j, and legs in the same zone-hour are summed, as side/lead_export.py
    exports the lead's positions for the lab."""
    d = led[["delivery_hour", "zone", "pos"]].copy()
    d["zone"] = d["zone"].astype(str)
    pair = d["zone"].str.contains("|", regex=False)
    if pair.any():
        a = d[pair]
        ij = a["zone"].str.split("|", regex=False, expand=True)
        legs = pd.concat([a.assign(zone=ij[0], mw=a["pos"]), a.assign(zone=ij[1], mw=-a["pos"])])
        d = pd.concat([d[~pair].assign(mw=d.loc[~pair, "pos"]), legs])
    else:
        d["mw"] = d["pos"]
    assert d["delivery_hour"].max() < END
    g = d.groupby(["delivery_hour", "zone"], as_index=False)["mw"].sum()
    m = F.h[["delivery_hour", "zone"]].merge(g, on=["delivery_hour", "zone"], how="left", validate="one_to_one")
    assert int(m["mw"].notna().sum()) == len(g), "positions without a row in the lab's frame"
    return np.nan_to_num(m["mw"].to_numpy(float))


def sizing_pass(res: dict) -> bool:
    """Bo's bar judged at the sizing view (decision of 6 Oct): positive in at least 2 of 3 years, three-year Sharpe
    above 0.42, positive at the 0.50 stress, and at least 10 percent a year on 500,000 at the drawdown-sized scale."""
    if res.get("incomplete") or res.get("PASS") is None:
        return False
    b, sv = res["bar"], res["sizing_view"]
    return bool(b["positive_years"] and b["sharpe_3y"] and b["stress_total"]
                and (sv.get("return_on_500k_at_that_scale_pct") or -1e9) >= SIZING_BAR_PCT)


def score(F, mw: np.ndarray) -> dict:
    """side/lab.py evaluate (full costs, 0.50 stress, the bar, the sizing view) plus the bar at the sizing view."""
    res = lab().evaluate(F, mw)
    res["pass_at_sizing_view"] = sizing_pass(res)
    return res


def brief(res: dict) -> dict:
    """The numbers the summary tables use."""
    t = res["three_year"]
    return {"years": {y: v["net_usd"] for y, v in res["years"].items()}, "three_year_net": t["net_usd"],
            "sharpe": t["sharpe"], "max_drawdown": t["max_drawdown_usd"], "stress_0.50": t["stress_0.50_usd"],
            "avg_net_per_year": res["avg_net_per_year"], "PASS_1MW": res["PASS"],
            "pass_at_sizing_view": res.get("pass_at_sizing_view", sizing_pass(res)),
            "sizing_scale": res["sizing_view"]["scale_for_100k_drawdown"],
            "sizing_return_pct": res["sizing_view"]["return_on_500k_at_that_scale_pct"]}


def stress_and_seasons(F, mw: np.ndarray) -> dict:
    """Costs at 1.00 USD per MWh on both sides, and the full-cost result by season (pooled over 2021 to 2023 and
    per year) and by half-year, each through side/lab.py stats on the daily P&L."""
    LB = lab()
    full, hi, mwh = F.day_pnl(mw), F.day_pnl(mw, STRESS_HIGH), F.day_mwh(mw)
    m3 = np.isin(F.day_year, LB.YEARS)
    out = {"stress_1.00": {"years": {str(Y): round(float(hi[F.day_year == Y].sum())) for Y in LB.YEARS},
                           "three_year": LB.stats(hi[m3], mwh[m3], F.days[m3])}}
    mon = F.days.month.to_numpy()
    sea = {}
    for name, months in SEASONS.items():
        sel = m3 & np.isin(mon, months)
        sea[name] = {"pooled": LB.stats(full[sel], mwh[sel], F.days[sel]),
                     "years": {str(Y): round(float(full[sel & (F.day_year == Y)].sum())) for Y in LB.YEARS}}
    out["seasons"] = sea
    half = {}
    for Y in LB.YEARS:
        for h, months in (("H1", range(1, 7)), ("H2", range(7, 13))):
            sel = (F.day_year == Y) & np.isin(mon, list(months))
            half[f"{Y} {h}"] = LB.stats(full[sel], mwh[sel], F.days[sel])
    out["half_years"] = half
    out["note"] = ("winter pools the December, January and February days of 2021 to 2023 (January and February "
                   "2024 are held out and not read); half-years are January to June and July to December")
    return out


# ------------------------------------------------------------------ other agents' outputs (read only)
def lab_results() -> dict | None:
    return read_json(SIDE / "lab_results.json")


def lead_choices() -> tuple[dict, str]:
    """The lead modeller's choices per year (model/evaluate.py score), with the file's time for the record."""
    f = RESULTS / "strategy_list_2021_2023.json"
    j = json.loads(f.read_text())
    return j["choices"], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f.stat().st_mtime))


def lead_strats(F) -> dict:
    """The lead's strategies as the lab scores them (side/lead_positions_2021_2023.parquet via lab.load_lead)."""
    LB = lab()
    lst, _ = LB.load_lead(F, {})
    return {s.name: s for s in lst}


def mw_of(F, st) -> np.ndarray:
    return lab().walk(F, st, st.inputs)[0]


def label_rows() -> pd.DataFrame:
    """Keys and labels of the build panel (model cache panel_base, read through model/panel.read_locked)."""
    import lock
    import panel as P
    d = P.read_locked(P.CACHE / "panel_base.parquet", "delivery_hour", lock.read_end(),
                      columns=["delivery_hour", "zone", "delivery_date", "gap"])
    assert d["delivery_hour"].max() < END
    return d.reset_index(drop=True)
