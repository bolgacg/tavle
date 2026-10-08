"""Neighbouring settings for B and C with gradient boosting (robustness check, OBJECTIVES.md addendum with idea 13,
commit b31b516). Run on gene from ~/nyiso-us:  .venv/bin/python robust/gbm_neighbours.py   (or `smoke`)

The registered grid (model/gbm.py GRID) has three two-valued settings: leaves 15 or 63, rows per leaf 200 or
2,000, target clip none or 250. The settings next to a chosen configuration are the three that differ from it
in exactly one of them. As in the lab's fragility test, each year's chosen configuration (model/evaluate.py,
read from results/strategy_list_2021_2023.json after the choice fix) is moved one step on one setting in every
year at once, walked forward through 2021 to 2023 with the same monthly refits (evaluate.reg_pred), turned into
positions by the registered rules (strategies.idea_B; strategies.idea_C with each year's registered pairs, which
stay fixed) and scored by side/lab.py. The chosen configuration is run through the same path as a check that
it reproduces the lab's row.

Walk-forward predictions already cached by the lead modeller (cache/wf) are read, never written; any other
configuration is computed here and cached in results/robust/cache_wf/. Nothing is fitted for a choice.
Holdout: evaluate.load and every loader go through model/lock.py; nothing on or after 2024-01-01 is read.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rb  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evaluate as E  # noqa: E402
import gbm  # noqa: E402
import panel as P  # noqa: E402
import rehearsal as R  # noqa: E402
import strategies as S  # noqa: E402

DIMS = {"num_leaves": (15, 63), "min_data_in_leaf": (200, 2000), "target_clip": (None, 250.0)}
STRATS = {"B_gbm": "weather", "C_gbm": "base"}
LEAD_WF = E.wf_dir()
_ORIG_WF_DIR = E.wf_dir
CFG = {gbm.config_name(c): c for c in gbm.GRID}


def _refuse(*a, **k):
    raise RuntimeError("robust: a lead cache is missing; refusing to build or write it")


def read_only_caches():
    """Like side/lead_export.py: never build or write the lead's feature caches."""
    orig = R._cached

    def cached(name, build):
        if not (R.CACHE / f"{name}.parquet").exists():
            _refuse()
        return orig(name, build)
    R._cached = cached
    P.save_panel = _refuse


def flip(cfg: dict, dim: str) -> dict:
    a, b = DIMS[dim]
    new = dict(cfg)
    new[dim] = b if cfg[dim] == a else a
    assert new in gbm.GRID and new != cfg
    return new


def pred(panel, cols, kind, cfg, lab, my_wf: Path):
    """evaluate.reg_pred, reading the lead's cached walk-forward when it exists, else computing into my_wf."""
    have = (LEAD_WF / f"{kind}_{gbm.config_name(cfg)}_{lab}.parquet").exists()
    E.wf_dir = (lambda: LEAD_WF) if have else (lambda: my_wf)
    try:
        return E.reg_pred(panel, cols, kind, cfg, lab), ("lead cache" if have else "computed here")
    finally:
        E.wf_dir = _ORIG_WF_DIR


def ledger(strat, panel, p, lab, choices):
    if strat == "B_gbm":
        return S.idea_B(R.rows(panel, *E.period(lab, "B")), p)
    pairs = [tuple(x.split("|")) for x in choices[str(lab)]["pairs"]["pairs"]]
    return S.idea_C(R.rows(panel, *E.period(lab)), p, pairs)


def run(panel, cols, F, choices, labres, my_wf: Path, labels=None) -> dict:
    labels = labels or E.labels()
    out = {}
    for strat, kind in STRATS.items():
        chosen = {lab: CFG[choices[str(lab)][kind]["config"]] for lab in labels}
        variants = {"chosen": chosen}
        for dim in DIMS:
            variants[f"{dim} one step"] = {lab: flip(chosen[lab], dim) for lab in labels}
        rows = {}
        for vname, cfgs in variants.items():
            leds, src = [], {}
            for lab in labels:
                p, src[str(lab)] = pred(panel, cols, kind, cfgs[lab], lab, my_wf)
                leds.append(ledger(strat, panel, p, lab, choices))
            res = rb.score(F, rb.ledger_to_mw(F, pd.concat(leds, ignore_index=True)))
            rows[vname] = {"settings": {str(lab): gbm.config_name(cfgs[lab]) for lab in labels}, "source": src,
                           **rb.brief(res)}
            rb.log(f"{strat} {vname}: three-year {rows[vname]['three_year_net']:,}, Sharpe {rows[vname]['sharpe']}")
        base = rows["chosen"]["three_year_net"]
        lab_net = (labres or {}).get("strategies", {}).get(strat, {}).get("three_year", {}).get("net_usd")
        out[strat] = {"kind": kind, "variants": rows,
                      "sign_kept_at_every_neighbour": all(np.sign(v["three_year_net"]) == np.sign(base)
                                                          for k, v in rows.items() if k != "chosen"),
                      "reproduces_lab": None if lab_net is None else bool(round(base) == round(lab_net)),
                      "lab_three_year_net": lab_net,
                      "pairs_note": "C's pairs stay as registered for each year" if strat == "C_gbm" else None}
    return out


def main():
    read_only_caches()
    my_wf = rb.OUT / "cache_wf"
    my_wf.mkdir(parents=True, exist_ok=True)
    choices, when = rb.lead_choices()
    labres = rb.lab_results()
    rb.log(f"choices from results/strategy_list_2021_2023.json written {when}")
    panel, cols, _ = E.load()
    assert panel["delivery_hour"].max() < rb.END
    F, *_ = rb.frame()
    res = run(panel, cols, F, choices, labres, my_wf)
    rb.write_json(rb.OUT / "gbm_neighbours.json", {
        "what": "B and C with gradient boosting at the settings next to each year's chosen configuration "
                "(one setting moved one step, every year at once), 2021 to 2023 walk-forward, scored by side/lab.py",
        "choices_file_written": when, "lgbm_threads": gbm.FIXED["num_threads"],
        "note": "configurations computed here use the thread count above; LightGBM is deterministic for a given "
                "thread count", "strategies": res})
    rb.log("wrote", rb.OUT / "gbm_neighbours.json")


def smoke():
    """Synthetic panel and frame: the same run() with a tiny walk-forward over two months of 2023."""
    import tempfile

    import lab as LB
    rng = np.random.default_rng(1)
    days = pd.date_range("2022-11-10", "2023-02-28", freq="D")
    hours = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in range(24)]).tz_localize(
        "America/New_York")                                  # no clock change between these dates
    zones = LB.ZONES
    n = len(hours) * len(zones)
    panel = pd.DataFrame({"delivery_hour": np.repeat(hours, len(zones)), "zone": np.tile(zones, len(hours))})
    panel["delivery_date"] = panel["delivery_hour"].dt.tz_localize(None).dt.normalize()
    panel["f1"], panel["f2"] = rng.normal(size=n), rng.normal(size=n)
    panel["x_w"] = rng.normal(size=n)
    panel["gap"] = 2 * panel["f1"] + rng.normal(scale=5, size=n) - 1
    panel["label_published_at"] = panel["delivery_hour"] + pd.Timedelta(hours=2)
    panel["hour"] = panel["delivery_hour"].dt.hour
    cols = {"base": ["f1", "f2"], "weather": ["f1", "f2", "x_w"]}
    px = pd.DataFrame({"delivery_hour": panel["delivery_hour"], "zone": panel["zone"], "da_lbmp": 30.0,
                       "rt_lbmp": 30.0 + panel["gap"]})
    px = px[px["delivery_hour"] >= pd.Timestamp("2022-12-25", tz="America/New_York")]
    F = LB.Frame(px)
    old = (E.PERIODS, E.B_FIRST, E.B_TRAIN, LB.YEARS)
    E.PERIODS = [(2023, dt.date(2023, 1, 1), dt.date(2023, 2, 28))]
    E.B_FIRST, E.B_TRAIN = dt.date(2023, 1, 1), dt.date(2022, 11, 10)
    LB.YEARS = [2023]
    E._v2 = lambda: None
    gbm.N_ROUNDS = 20
    choices = {"2023": {"base": {"config": "leaves15_minleaf200_clipnone"},
                        "weather": {"config": "leaves63_minleaf2000_clip250.0"},
                        "pairs": {"pairs": ["CAPITL|NORTH", "LONGIL|WEST"]}}}
    with tempfile.TemporaryDirectory() as tmp:
        global LEAD_WF
        LEAD_WF = Path(tmp) / "lead_wf_absent"
        res = run(panel, cols, F, choices, None, Path(tmp) / "wf", labels=[2023])
        rb.write_json(Path(tmp) / "gbm_neighbours.json", res)
        assert set(res) == {"B_gbm", "C_gbm"} and len(res["C_gbm"]["variants"]) == 4
        assert len(list((Path(tmp) / "wf").glob("*.parquet"))) == 8
    E.PERIODS, E.B_FIRST, E.B_TRAIN, LB.YEARS = old
    rb.log("smoke ok: 2 strategies x 4 settings walked forward, scored by lab.evaluate")


if __name__ == "__main__":
    smoke() if len(sys.argv) > 1 and sys.argv[1] == "smoke" else main()
