"""Walk-forward run of idea 14 (deep_alloc14.py). On gene from ~/nyiso-us/model:

    ../.venv/bin/python train_deep_alloc14.py all --device cuda     refits 2021-01 .. 2023-12 (every kappa, 5 seeds),
                                                                    saved per year (a restart skips done years), then
                                                                    merge: kappa per year, the scored file and json
    ../.venv/bin/python train_deep_alloc14.py merge                 merge only

Output results/idea14_wf_2021_2023.parquet: one row per delivery date 2021-01-01 .. 2023-12-31 with the chosen
kappa's weights w_<cand> (sum 1, or 0 when no candidate takes part) and the refit's prior-only scales scale_<cand>.
The lab's position per zone-hour is sum_c w_c * scale_c * (candidate c's exported MW).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import deep_alloc14 as A
import deep_policy as DP
import lock

PARTS = DP.RESULTS / "idea14_parts"
FIRST, LAST = dt.date(2021, 1, 1), dt.date(2023, 12, 31)
YEARS = (2021, 2022, 2023)


def log(msg: str):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def part(Y: int, suffix: str = ""):
    b = PARTS / f"idea14_{Y}{suffix}"
    return b.with_suffix(".parquet"), b.with_suffix(".json")


def inputs():
    px = DP.load_prices()
    tg = DP.Targets(px)
    cp = A.CandProfits(px, A.load_candidates(), tg)
    return px, tg, cp


def run(a, px, tg, cp):
    import torch
    torch.set_num_threads(a.threads)
    PARTS.mkdir(parents=True, exist_ok=True)
    feats = DP.load_day_features(DP.FEATS)
    log(f"features {feats.shape}; candidate days with profits {dict(zip(A.CANDS, np.isfinite(cp.P).sum(0).tolist()))}")
    cfg = {"max_epochs": a.max_epochs} if a.max_epochs else None
    first, last = dt.date.fromisoformat(a.first), dt.date.fromisoformat(a.last)
    for Y in range(first.year, last.year + 1):
        pq, js = part(Y, a.suffix)
        if pq.exists() and js.exists():
            log(f"{Y}: done before, skipped")
            continue
        t0 = time.time()
        w, r = A.walk_forward(feats, tg, cp, max(first, dt.date(Y, 1, 1)), min(last, dt.date(Y, 12, 31)), A.KAPPAS,
                              a.seeds, cfg, a.device, log)
        w.to_parquet(pq)
        js.write_text(json.dumps({"refits": r, "seconds": round(time.time() - t0, 1), "config": {**A.CONFIG, **(cfg or {})},
                                  "seeds": list(a.seeds)}, default=str))
        log(f"{Y}: {len(r)} refits, {time.time() - t0:.0f} s -> {pq}")


def merge(a, px, tg, cp):
    missing = [str(p) for Y in YEARS for p in part(Y, a.suffix) if not p.exists()]
    if missing and not a.allow_partial:
        raise SystemExit(f"walk-forward incomplete, nothing written: missing {missing}")
    ws, recs = [], []
    for Y in YEARS:
        pq, js = part(Y, a.suffix)
        if pq.exists():
            w = pd.read_parquet(pq)
            w["delivery_date"] = pd.to_datetime(w["delivery_date"])
            ws.append(w)
            recs += json.loads(js.read_text())["refits"]
    allw = pd.concat(ws, ignore_index=True)
    DP.assert_build(allw["delivery_date"], "idea14 parts")
    lock.assert_build_only(allw["delivery_date"])
    choice = A.choose_kappas(allw, tg, cp, YEARS)
    final = A.final_rows(allw, choice)
    out = Path(str(A.OUT).replace(".parquet", a.suffix + ".parquet"))
    final.to_parquet(out)
    allw.to_parquet(Path(str(DP.RESULTS / "idea14_wf_all_kappas") + a.suffix + ".parquet"))
    quick = {}
    for Y in YEARS:
        f = final[final["delivery_date"].dt.year == Y]
        eq = f.copy()
        part_ = np.stack([eq[f"scale_{c}"] > 0 for c in A.CANDS], 1)
        n = np.maximum(part_.sum(1), 1)
        for j, c in enumerate(A.CANDS):
            eq[f"w_{c}"] = part_[:, j] / n
        quick[str(Y)] = {"idea14_unnetted": round(float(A.day_net(f, tg, cp).sum())),
                         "equal_weights_same_scales_unnetted": round(float(A.day_net(eq, tg, cp).sum())),
                         "mean_weights": {c: round(float(f[f"w_{c}"].mean()), 3) for c in A.CANDS}}
    summ = {"what": "idea 14 (deep_alloc14.py); see module docstrings", "kappas": list(A.KAPPAS),
            "loss_lam": A.LOSS_LAM, "min_hist": A.MIN_HIST, "dd_budget": A.DD_BUDGET, "scale_cap": A.SCALE_CAP,
            "config": A.CONFIG, "kappa_choice": {str(k): v for k, v in choice.items()}, "refits": recs,
            "quick_net": quick, "note": "quick net sums candidates' own profits (not netted); the lab's numbers are "
                                        "the official ones"}
    out.with_suffix(".json").write_text(json.dumps(summ, indent=1, default=str))
    log(f"kappa {json.dumps({str(k): v['kappa'] for k, v in choice.items()})}; quick {json.dumps(quick)}")
    log(f"-> {out} ({len(final)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["all", "run", "merge"])
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--first", default=str(FIRST))
    ap.add_argument("--last", default=str(LAST))
    ap.add_argument("--max-epochs", type=int, default=0, help="smoke runs only")
    ap.add_argument("--suffix", default="", help="output name suffix (smoke runs only)")
    ap.add_argument("--allow-partial", action="store_true", help="merge: smoke runs only")
    a = ap.parse_args()
    a.seeds = tuple(int(x) for x in a.seeds.split(","))
    log(f"train_deep_alloc14 {a.mode} {vars(a)}")
    px, tg, cp = inputs()
    if a.mode in ("all", "run"):
        run(a, px, tg, cp)
    if a.mode in ("all", "merge"):
        merge(a, px, tg, cp)


if __name__ == "__main__":
    main()
