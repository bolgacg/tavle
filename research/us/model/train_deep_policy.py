"""Walk-forward runs of ideas 11 and 12 (deep_policy.py; owner: policy agent). On gene from ~/nyiso-us/model:

    ../.venv/bin/python train_deep_policy.py run --ideas 11     monthly refits July 2020 .. December 2023, every
    ../.venv/bin/python train_deep_policy.py run --ideas 12     lambda; one process per idea, one CPU thread each;
                                                                saved after every year (a restart skips done years)
    ../.venv/bin/python train_deep_policy.py merge              lambda per year, the scored file and quick yearly net
                                                                -> results/deep_policy_wf_2021_2023.parquet and .json
    ../.venv/bin/python train_deep_policy.py score              quick yearly net again from the saved file

Refits from July 2020 exist only to choose the 2021 lambda (on July to December 2020); the scored file holds
delivery dates 2021-01-01 to 2023-12-31, each from its own month's refit, with the lambda chosen on the year before.
Output columns: delivery_date, bid_date, zone, idea (11 or 12), position (signed MW held in every hour of the
zone-day: -1 supply, +1 load), w_supply, w_out, w_load, w_pair (idea 11: effective expert weights, sum 1),
pair_up1, pair_up2 (idea 11: the pair's upstate supply legs; N.Y.C. and LONGIL are its load legs), lam, refit_month.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import deep_policy as DP
import lock

OUT = DP.RESULTS / "deep_policy_wf_2021_2023"
PARTS = DP.RESULTS / "deep_policy_parts"
FIRST, LAST = dt.date(2020, 7, 1), dt.date(2023, 12, 31)
YEARS = (2021, 2022, 2023)


def log(msg: str):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def wait_for(path: Path, minutes: int, every: int = 10):
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > minutes * 60:
            raise SystemExit(f"{path} not there after {minutes} minutes")
        log(f"waiting for {path}")
        time.sleep(every * 60)


def part_paths(idea: int, year: int, suffix: str = ""):
    base = PARTS / f"idea{idea}_{year}{suffix}"
    return base.with_suffix(".parquet"), base.with_suffix(".json")


def run(a):
    import torch
    torch.set_num_threads(a.threads)
    wait_for(DP.FEATS, a.wait_minutes)
    PARTS.mkdir(parents=True, exist_ok=True)
    feats = DP.load_day_features(DP.FEATS)
    tg = DP.Targets(DP.load_prices())
    log(f"features {feats.shape}, delivery {feats['delivery_date'].min().date()}..{feats['delivery_date'].max().date()}; "
        f"price days {len(tg.days)}; threads {torch.get_num_threads()}")
    cfg = {"max_epochs": a.max_epochs} if a.max_epochs else None
    first, last = dt.date.fromisoformat(a.first), dt.date.fromisoformat(a.last)
    for idea in a.ideas:
        for Y in range(first.year, last.year + 1):
            pq, js = part_paths(idea, Y, a.suffix)
            if pq.exists() and js.exists():
                log(f"idea {idea} {Y}: done before, skipped")
                continue
            ty = time.time()
            p, r = DP.walk_forward(feats, tg, idea, max(first, dt.date(Y, 1, 1)), min(last, dt.date(Y, 12, 31)),
                                   DP.LAMBDAS, a.seeds, cfg, a.device, log)
            p.to_parquet(pq)
            js.write_text(json.dumps({"refits": r, "seconds": round(time.time() - ty, 1),
                                      "config": {**DP.CONFIG, **(cfg or {})}, "seeds": list(a.seeds),
                                      "threads": a.threads}, default=str))
            log(f"idea {idea} {Y}: {len(r)} refits, {time.time() - ty:.0f} s -> {pq}")


def load_parts(ideas, suffix=""):
    parts, recs, secs = [], [], {}
    for idea in ideas:
        secs[idea] = []
        for Y in range(FIRST.year, LAST.year + 1):
            pq, js = part_paths(idea, Y, suffix)
            if not pq.exists():
                continue
            p = pd.read_parquet(pq)
            p["delivery_date"] = pd.to_datetime(p["delivery_date"])
            DP.assert_build(p["delivery_date"], str(pq))
            parts.append(p)
            j = json.loads(js.read_text())
            recs += j["refits"]
            secs[idea].append(j["seconds"])
    allpos = pd.concat(parts, ignore_index=True)
    lock.assert_build_only(allpos["delivery_date"])
    return allpos, recs, secs


def final_positions(allpos: pd.DataFrame, choice: dict) -> pd.DataFrame:
    parts = []
    for idea, ch in choice.items():
        for Y in YEARS:
            parts.append(allpos[(allpos["idea"] == idea) & (allpos["delivery_date"].dt.year == Y)
                                & (allpos["lam"] == ch[Y]["lam"])])
    out = pd.concat(parts, ignore_index=True).sort_values(["idea", "delivery_date", "zone"], kind="stable")
    DP.assert_build(out["delivery_date"], "final")
    lock.assert_build_only(out["delivery_date"])
    assert (pd.to_datetime(out["refit_month"]) == out["delivery_date"].dt.to_period("M").dt.start_time).all()
    assert not out.duplicated(["idea", "delivery_date", "zone"]).any()
    return out.reset_index(drop=True)


def quick_net(final: pd.DataFrame, allpos: pd.DataFrame | None, px: pd.DataFrame) -> dict:
    h = DP.hourly_frame(px)
    res = {}
    for Y in YEARS:
        g = h[h["year"] == Y]
        r = {"always_supply": DP.score(g, -np.ones(len(g)))}
        for idea in sorted(final["idea"].unique()):
            f = final[(final["idea"] == idea) & (final["delivery_date"].dt.year == Y)]
            if len(f):
                mw = DP.hourly_mw(g, f)
                r[f"idea{idea}"] = {**DP.score(g, mw),
                                    "stress_0.50": round(float((mw * g["gap"] - np.abs(mw) * 0.5).sum())),
                                    "lam": float(f["lam"].iloc[0]),
                                    "mean_abs_position": round(float(f["position"].abs().mean()), 3),
                                    "share_supply_side": round(float((f["position"] < 0).mean()), 3)}
            if allpos is not None:
                for l in sorted(allpos["lam"].unique()):
                    a = allpos[(allpos["idea"] == idea) & (allpos["lam"] == l) & (allpos["delivery_date"].dt.year == Y)]
                    if len(a):
                        r[f"idea{idea}_lam{l:g}"] = DP.score(g, DP.hourly_mw(g, a))["net_usd"]
        res[str(Y)] = r
    return res


def merge(a):
    missing = [str(pp) for i in a.ideas for Y in range(FIRST.year, LAST.year + 1) for pp in part_paths(i, Y, a.suffix)
               if not pp.exists()]
    if missing and not a.allow_partial:
        raise SystemExit(f"walk-forward incomplete, nothing written: missing {missing}")
    allpos, recs, secs = load_parts(a.ideas, a.suffix)
    px = DP.load_prices()
    tg = DP.Targets(px)
    choice = {idea: DP.choose_lambdas(allpos[allpos["idea"] == idea], tg, YEARS) for idea in a.ideas}
    final = final_positions(allpos, choice)
    out = Path(str(OUT) + a.suffix)
    final.to_parquet(out.with_suffix(".parquet"))
    allpos.to_parquet(Path(str(DP.RESULTS / "deep_policy_wf_all_lambdas") + a.suffix + ".parquet"))
    qn = quick_net(final, allpos, px)
    summ = {"what": "ideas 11 and 12 (deep_policy.py); positions held every hour of the zone-day; see module docstrings",
            "span_refits": [str(FIRST), str(LAST)], "scored_years": list(YEARS), "lambdas": list(DP.LAMBDAS),
            "config": DP.CONFIG, "lambda_choice": {str(k): {str(y): v for y, v in c.items()} for k, c in choice.items()},
            "refits": recs,
            "seconds_by_idea": {str(k): round(float(sum(v)), 1) for k, v in secs.items()},
            "seconds_per_refit": {str(i): round(float(np.mean([r["seconds"] for r in recs if r["idea"] == i])), 1)
                                  for i in a.ideas},
            "quick_net_storm_value_score": qn,
            "note": "quick net by storm_value.score at full cost; the lab's numbers are the official ones"}
    out.with_suffix(".json").write_text(json.dumps(summ, indent=1, default=str))
    for y, r in qn.items():
        log(f"{y}: " + ", ".join(f"{k} {v['net_usd'] if isinstance(v, dict) else v:,}" for k, v in r.items()))
    log(f"lambda choice {json.dumps({str(k): {str(y): v['lam'] for y, v in c.items()} for k, c in choice.items()})}")
    log(f"-> {out.with_suffix('.parquet')} ({len(final)} rows)")


def score_only(a):
    out = Path(str(OUT) + a.suffix)
    final = pd.read_parquet(out.with_suffix(".parquet"))
    final["delivery_date"] = pd.to_datetime(final["delivery_date"])
    DP.assert_build(final["delivery_date"], "final")
    print(json.dumps(quick_net(final, None, DP.load_prices()), indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "merge", "score"])
    ap.add_argument("--ideas", default="11,12")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--first", default=str(FIRST))
    ap.add_argument("--last", default=str(LAST))
    ap.add_argument("--max-epochs", type=int, default=0, help="smoke runs only")
    ap.add_argument("--suffix", default="", help="output name suffix (smoke runs only)")
    ap.add_argument("--wait-minutes", type=int, default=120)
    ap.add_argument("--allow-partial", action="store_true", help="merge: smoke runs only")
    a = ap.parse_args()
    a.ideas = [int(x) for x in a.ideas.split(",")]
    a.seeds = tuple(int(x) for x in a.seeds.split(","))
    log(f"train_deep_policy {a.mode} {vars(a)}")
    {"run": run, "merge": merge, "score": score_only}[a.mode](a)


if __name__ == "__main__":
    main()
