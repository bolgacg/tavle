"""Export the lead modeller's 2021 to 2023 strategy positions for the strategy lab (read-only use of model/).

Calls model/evaluate.ledgers_for for each year exactly as `evaluate.py score` does, from the cached walk-forward
predictions. Nothing is fitted and nothing is written inside model/, results/ or cache/: every fitting and
saving entry point is replaced by a function that raises, so a missing cache stops the export instead of
refitting. Build years only: evaluate's loaders go through model/lock.py, and every exported row is asserted
to be before 2024-01-01.

Output (gene ~/nyiso-us/side/): lead_positions_2021_2023.parquet (strategy, year, delivery_hour, zone, mw, pred),
lead_storm_score.parquet (delivery_date, storm), lead_ledger_totals.json (the lead's own net per strategy and
year, from the same ledgers, for the comparison).
Pair ledgers (idea C, "i|j") are split into their two legs: +leg MW in zone i, -leg MW in zone j.
"""
import json
import sys
from pathlib import Path

HOME = Path.home() / "nyiso-us"
sys.path.insert(0, str(HOME / "pipeline"))
sys.path.insert(0, str(HOME / "model"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evaluate as E  # noqa: E402
import gbm  # noqa: E402
import panel as P  # noqa: E402
import rehearsal as R  # noqa: E402
import walkforward as W  # noqa: E402

END = pd.Timestamp("2024-01-01", tz="America/New_York")
OUT = HOME / "side"


def _refuse(*a, **k):
    raise RuntimeError("lead_export: a cache is missing; refusing to fit or write")


W.walk_forward = _refuse
gbm.GBMModel.fit = _refuse
gbm.SpikeClassifier.fit = _refuse
P.save_panel = _refuse
_orig_cached = R._cached


def _cached_read_only(name, build):
    if not (R.CACHE / f"{name}.parquet").exists():
        _refuse()
    return _orig_cached(name, build)


R._cached = _cached_read_only
_orig_sel = E._sel_fit


def _sel_read_only(panel, kind, key, lab, make, train_start=None):
    if not E._name(f"sel_{kind}", key, lab).exists():
        p0 = E.prev(lab)
        va = R.rows(panel, *E.period(p0, "B" if kind == "weather" else ""))
        tr = panel[W.train_mask(panel, va["delivery_date"].min().date(), train_start)]
        if tr["delivery_date"].nunique() < E.MIN_TRAIN_DAYS:
            return None, va                      # evaluate's own "too little data" path, no fit
        _refuse()
    return _orig_sel(panel, kind, key, lab, make, train_start)


E._sel_fit = _sel_read_only


def main():
    panel, cols, _ = E.load()
    assert panel["delivery_hour"].max() < END
    parts, totals, choices = [], {}, {}
    for lab in E.labels():
        L, ch = E.ledgers_for(panel, cols, lab)
        choices[str(lab)] = {k: v[1] for k, v in ch.items()}
        for name, led in L.items():
            assert led["delivery_hour"].max() < END
            totals.setdefault(name, {})[str(lab)] = round(float(led["pnl"].sum()), 2)
            d = led[["delivery_hour", "zone", "pos", "pred"]].copy()
            pair = d["zone"].astype(str).str.contains("|", regex=False)
            if pair.any():
                a = d[pair].copy()
                ij = a["zone"].astype(str).str.split("|", regex=False, expand=True)
                legs = pd.concat([a.assign(zone=ij[0], mw=a["pos"]), a.assign(zone=ij[1], mw=-a["pos"])])
                d = pd.concat([d[~pair].assign(mw=d.loc[~pair, "pos"]), legs])
            else:
                d["mw"] = d["pos"]
            g = d.groupby(["delivery_hour", "zone"], as_index=False).agg(mw=("mw", "sum"), pred=("pred", "first"),
                                                                       legs=("mw", "size"))
            g["strategy"], g["year"] = name, int(lab)
            parts.append(g)
        print(lab, sorted(L), flush=True)
    out = pd.concat(parts, ignore_index=True)
    assert out["delivery_hour"].max() < END
    out.to_parquet(OUT / "lead_positions_2021_2023.parquet")
    G = E._shared(panel, cols)
    st = G["storm"].rename("storm").rename_axis("delivery_date").reset_index()
    assert pd.to_datetime(st["delivery_date"]).max() < END.tz_localize(None)
    st.to_parquet(OUT / "lead_storm_score.parquet")
    (OUT / "lead_ledger_totals.json").write_text(json.dumps({"totals": totals, "choices": choices}, indent=1,
                                                            default=str))
    print("wrote", len(out), "rows,", out["strategy"].nunique(), "strategies", flush=True)


if __name__ == "__main__":
    main()
