"""The held-out run of version 1: every row of research/us/HELDOUT-LIST.md (7 candidates, 27 comparisons) walked
forward over the test window exactly as the build years were, then scored by side/lab.py's scoring. Written 8 Oct 2026.

Runs only in the dry-run or held-out mode of heldout_mode.py (US_RUN_MODE=dryrun or heldout, US_RUN_DIR set):

    dryrun   test window 2023-01-01 to 2023-12-31, every choice made on 2022: must reproduce the build's 2023
             positions, predictions and daily net for every row (stage `compare` checks it). Reads nothing after
             2023-12-31.
    heldout  test window 2024-01-01 to 2026-09-30; opens only through the lock on the frozen clone (heldout.sh).

    python heldout_v1.py all          every stage below in order (restartable: a finished stage is skipped)
    python heldout_v1.py <stage> ...  seed panel gbm dayfeat policy11 policy12 deep_base deep_weather deep_outages
                                      deep_spike deepday policy_merge export alloc14 lab compare (a stage another
                                      process is running is waited for, not run twice)
    python heldout_v1.py plan         print the window, the paths and the stages, read nothing

Rules (the build's, unchanged; this file only points the build code at the window and at US_RUN_DIR):
  * seed: model outputs of the years before the window (walk-forward and selection predictions, deep per-year
    predictions, the policy and idea 14 per-year parts, the deep day rows) are COPIED from ~/nyiso-us into
    US_RUN_DIR; nothing for a window year is copied and nothing under ~/nyiso-us is ever written. Data products
    (panel, features, day features) are rebuilt in US_RUN_DIR from the raw tables.
  * gbm (model/evaluate.py compute): each window year Y gets its choices on Y-1 (configurations, S and p*, pairs,
    the tail cut) and monthly refits on data up to two days before each month. Build years are read from the
    copied caches only: a missing build-year cache stops the run instead of refitting. In the dry run the
    rehearsal's 2023 choices and predictions are not reused, so the generic year-Y path makes them again.
  * deep (model/train_deep.py wf, configuration c2, seeds 0 to 2, S from model/spike_config.json, GPU), deepday
    (model/train_deep_day.py walk, 5 seeds, CPU with 2 threads as the build ran it), policy (model/deep_policy.py, ideas 11 and 12, every lambda, CPU,
    one thread), alloc14 (model/deep_alloc14.py, every kappa): monthly refits over the window; each year's
    p*, lambda and kappa chosen on the year before from the combined (build plus window) walk-forward.
  * export: model/evaluate.ledgers_for for every year as side/lead_export.py does (lead positions).
  * lab: side/lab.py's rules and scoring for the window years (per year, total, Sharpe, 0.50 stress, placebos,
    fragility, too-good checks, lookahead tests), idea 13 with the build's component weights, and the sizing
    view at the build's scales. Costs per year from fees.py (2026: RS1 unverified, see fees.SOURCE).
Outputs in US_RUN_DIR: side/lab_results_heldout.json and .md (window only), side/lab_daily_heldout.parquet
(window days, the lab_daily format plus idea13_portfolio), side/lab_daily_2021_<end>.parquet and
side/lab_results_2021_<end>.json (build years from the build files plus the window, for
chartdata/build_chartdata.py --include-heldout --v1-results/--v1-daily), timing.json, and in the dry run
dryrun_equality.json and .md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
for _p in (HERE, STUDY / "pipeline", STUDY / "side", STUDY / "robust"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import heldout_mode as HM  # noqa: E402
import lock  # noqa: E402

MODE = HM.mode()
if MODE == "build":
    raise SystemExit("heldout_v1.py runs only with US_RUN_MODE=dryrun or heldout (and US_RUN_DIR)")
RUN = HM.run_dir()
W0, W1 = HM.window()
WY = list(range(W0.year, W1.year + 1))                 # window years
READ_END = HM.read_end()
BUILD_HOME = Path.home() / "nyiso-us"                   # build outputs: read and copied, never written
os.environ.setdefault("US_CACHE_DIR", str(RUN / "cache"))
os.environ.setdefault("US_LGBM_THREADS", "4")            # the build's thread count (gbm.FIXED; results depend on it)

import common  # noqa: E402
if os.environ.get("US_PARQUET_DIR"):                     # raw tables of the window, if not ~/nyiso-us/parquet
    common.PARQUET = Path(os.environ["US_PARQUET_DIR"]).expanduser()
PARQUET = common.PARQUET

_lock_read_end = lock.read_end


def _read_end(requested_end=None, study_dir=lock.STUDY_DIR, env=None):
    """Loaders' default end becomes this mode's (2024-01-01 in the dry run); the lock still checks it."""
    return _lock_read_end(requested_end or READ_END, study_dir, env)


lock.read_end = _read_end

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fees  # noqa: E402
import panel as P  # noqa: E402

CACHE = P.CACHE
RES = RUN / "results"
SIDE = RUN / "side"
CKPT = RUN / "ckpt"
DONE = RUN / "done"
KEY = ["delivery_hour", "zone"]
BUILD_YEARS = [2021, 2022, 2023]
FIRST_YEAR = 2021
END_TAG = f"2021_{W1:%Y%m}"
TIMING = RUN / "timing.json"


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _done(stage: str) -> Path:
    return DONE / f"{stage}.json"


def _record(stage: str, seconds: float, extra: dict | None = None):
    DONE.mkdir(parents=True, exist_ok=True)
    _done(stage).write_text(json.dumps({"stage": stage, "seconds": round(seconds, 1), "mode": MODE,
                                        "window": [str(W0), str(W1)], "finished": time.strftime("%F %T"),
                                        **(extra or {})}, indent=1, default=str))
    t = json.loads(TIMING.read_text()) if TIMING.exists() else {}
    t[stage] = round(seconds, 1)
    TIMING.write_text(json.dumps(t, indent=1))


def _wait_gpu(min_free_mib: int = 2500, every: int = 60):
    while True:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30).stdout.split()
            free = int(out[0])
        except Exception:                                   # noqa: BLE001  no GPU tool: go on (CPU fallback)
            return
        if free >= min_free_mib:
            log(f"GPU: {free} MiB free")
            return
        log(f"GPU: {free} MiB free, waiting for {min_free_mib}")
        time.sleep(every)


def assert_window(dates, what: str):
    """Nothing later than the window's last day, ever."""
    s = pd.Series(dates).dropna()
    if not len(s):
        return
    mx = pd.Timestamp(s.max())
    if mx.tzinfo is not None:
        mx = mx.tz_convert(common.TZ).tz_localize(None)
    assert mx.normalize() <= pd.Timestamp(W1), f"{what}: {mx} is after {W1}"


# ====================================================================== seed (copies of build outputs)
def _year_of(stem: str) -> str:
    return stem.rsplit("_", 1)[1]


def seed():
    """Copy the build's model outputs for years before the window. Never a window year, never a data product."""
    copied = []

    def cp(src: Path, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        copied.append({"from": str(src), "to": str(dst.relative_to(RUN)), "sha256": _sha(dst)})

    for f in sorted((BUILD_HOME / "cache" / "wf").glob("*.parquet")):
        y = _year_of(f.stem)
        if y == "pre" or int(y) < W0.year:
            cp(f, CACHE / "wf" / f.name)
    if W0.year > 2023:                                    # 2023 as the build made it: the v2 rehearsal's
        cp(BUILD_HOME / "cache" / "preds_gbm_2023_v2.parquet", CACHE / "preds_gbm_2023_v2.parquet")
        cp(BUILD_HOME / "results" / "rehearsal_2023_v2.json", RES / "rehearsal_2023_v2.json")
    cp(BUILD_HOME / "results" / "deep_tune_2022.json", RES / "deep_tune_2022.json")   # c2, chosen on 2022
    for kind in ("base", "weather", "outages", "spike"):
        for y in BUILD_YEARS:
            if y < W0.year:
                cp(BUILD_HOME / "results" / f"deep_wf_{kind}_{y}.parquet", RES / "seed" / f"deep_wf_{kind}_{y}.parquet")
    for x in ("base", "B", "D", "spike"):
        cp(BUILD_HOME / "results" / f"deep_wf_2021_2023_{x}.json", RES / "seed" / f"deep_wf_2021_2023_{x}.json")
    for idea in (11, 12):
        for y in (2020, *BUILD_YEARS):
            if y < W0.year:
                for suf in (".parquet", ".json"):
                    cp(BUILD_HOME / "results" / "deep_policy_parts" / f"idea{idea}_{y}{suf}",
                       RES / "deep_policy_parts" / f"idea{idea}_{y}{suf}")
    for y in BUILD_YEARS:
        if y < W0.year:
            for suf in (".parquet", ".json"):
                cp(BUILD_HOME / "results" / "idea14_parts" / f"idea14_{y}{suf}", RES / "idea14_parts" / f"idea14_{y}{suf}")
    # the deep day rows: one file for 2021 to 2023, so only its rows before the window are written
    dd = pd.read_parquet(BUILD_HOME / "results" / "deep_day_wf_2021_2023.parquet")
    dd["delivery_date"] = pd.to_datetime(dd["delivery_date"])
    dd = dd[dd["delivery_date"] < pd.Timestamp(W0)].reset_index(drop=True)
    (RES / "seed").mkdir(parents=True, exist_ok=True)
    dd.to_parquet(RES / "seed" / "deep_day_wf_before_window.parquet")
    j = json.loads((BUILD_HOME / "results" / "deep_day_wf_2021_2023.json").read_text())
    j["refits"] = [r for r in j["refits"] if pd.Timestamp(r["month"]) < pd.Timestamp(W0)]
    j.pop("eval_by_year", None)
    (RES / "seed" / "deep_day_wf_before_window.json").write_text(json.dumps(j, indent=1, default=str))
    copied.append({"from": str(BUILD_HOME / "results" / "deep_day_wf_2021_2023.parquet"),
                   "to": "results/seed/deep_day_wf_before_window.parquet", "rows": len(dd),
                   "note": f"rows with delivery date before {W0} only"})
    (RUN / "seed_manifest.json").write_text(json.dumps({"mode": MODE, "window": [str(W0), str(W1)],
                                                         "copied": copied}, indent=1))
    log(f"seed: {len(copied)} files from {BUILD_HOME}")
    return {"files": len(copied)}


# ====================================================================== the lead's pipeline (model/evaluate.py)
_E = None


def ev():
    """model/evaluate.py pointed at the window and at US_RUN_DIR (module globals only; no code path changed)."""
    global _E
    if _E is not None:
        return _E
    import evaluate as E
    import rehearsal as R
    R.RESULTS = RES
    R.OUT = RES / "rehearsal_2023_v2.json"
    R.CACHE = CACHE
    R.READ_END = READ_END
    R.BUILD = (R.FIRST_DAY, W1)
    R.SITES_THROUGH = dt.date(W0.year - 1, 12, 30)       # the first window year's 20 sites (x_ columns)
    E.PERIODS = [(y, dt.date(y, 1, 1), min(dt.date(y, 12, 31), W1)) for y in range(FIRST_YEAR, W1.year + 1)]
    E.OUT = RES / "strategy_list_heldout"
    if MODE == "dryrun":
        E._v2 = lambda: None                              # 2023 is the window: its generic path, not the reuse
    orig_wf, orig_sel = E.wf, E._sel_fit

    def wf_guard(panel, kind, key, lab, lo, hi, make, train_start=None):
        if not (lab in WY) and not E._name(kind, key, lab).exists():
            raise RuntimeError(f"build-year prediction missing from the seed: {E._name(kind, key, lab).name}; "
                               "refusing to refit a build year")
        return orig_wf(panel, kind, key, lab, lo, hi, make, train_start)

    def sel_guard(panel, kind, key, lab, make, train_start=None):
        if lab not in WY and not E._name(f"sel_{kind}", key, lab).exists():
            p0 = E.prev(lab)
            va = R.rows(panel, *E.period(p0, "B" if kind == "weather" else ""))
            tr = panel[E.W.train_mask(panel, va["delivery_date"].min().date(), train_start)]
            if tr["delivery_date"].nunique() >= E.MIN_TRAIN_DAYS:
                raise RuntimeError(f"build-year selection fit missing from the seed: sel_{kind} {key} {lab}")
        return orig_sel(panel, kind, key, lab, make, train_start)
    E.wf, E._sel_fit = wf_guard, sel_guard
    _E = E
    return E


_PANEL = None


def load_panel():
    global _PANEL
    if _PANEL is None:
        E = ev()
        store = P.LockedStore(end=READ_END)
        panel, cols, _ = E.load(store)
        assert_window(panel["delivery_date"], "panel")
        assert panel["delivery_date"].max() == pd.Timestamp(W1), "the panel does not reach the window's last day"
        _PANEL = (panel, cols)
    return _PANEL


def stage_panel():
    panel, cols = load_panel()
    return {"rows": len(panel), "first": str(panel["delivery_date"].min().date()),
            "last": str(panel["delivery_date"].max().date()), "feature_sets": {k: len(v) for k, v in cols.items()
                                                                                if isinstance(v, list)}}


def stage_gbm():
    E = ev()
    panel, cols = load_panel()
    E.compute(panel, cols)
    ch = {}
    for lab in WY:
        ch[str(lab)] = {k: v[1] for k, v in {
            "base": E.choose_reg(panel, cols, "base", lab), "weather": E.choose_reg(panel, cols, "weather", lab),
            "outages": E.choose_reg(panel, cols, "outages", lab), "spike": E.choose_spike(panel, cols, lab),
            "tail": E.choose_tail(panel, cols, lab)}.items()}
    (RES / "gbm_choices_window.json").write_text(json.dumps(ch, indent=1, default=str))
    return {"choices": "results/gbm_choices_window.json"}


# ====================================================================== deep walk-forwards (model/train_deep.py)
DEEP_KINDS = ("base", "weather", "outages", "spike")


def stage_deep(kind: str):
    import argparse as _ap
    import train_deep as TD
    TD.HOME = RUN                                          # its outage feature cache: US_RUN_DIR/cache/deep
    ev()                                                   # rehearsal globals (load_all for B and D columns)
    _wait_gpu()
    tmp = RES / "deep_wf_window" / kind
    tmp.mkdir(parents=True, exist_ok=True)
    args = _ap.Namespace(kind=kind, config="auto", seeds=(0, 1, 2), device=None, override={})
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    tlog = TD.Log(RUN / "logs" / f"deep_{kind}.log")
    shutil.copy2(RES / "deep_tune_2022.json", tmp / "deep_tune_2022.json")
    summ = TD.wf(args, tlog, span=(W0, W1), start=TD.BUILD_START, results=tmp, ckpt_dir=CKPT / "deep_wf",
                 root=PARQUET)
    _merge_deep(kind, TD)
    return {"refits": len(summ["refits"]), "seconds_per_refit_mean": summ["seconds_per_refit_mean"]}


def _merge_deep(kind: str, TD):
    """Build-year per-year files (seed) plus the window's, as the build's combined file, per-year files and JSON;
    for the spike model the p* per year (train_deep.choose_p_star on the combined predictions)."""
    x = TD.WF_NAME[kind]
    tmp = RES / "deep_wf_window" / kind
    parts = []
    for y in range(FIRST_YEAR, W0.year):
        d = pd.read_parquet(RES / "seed" / f"deep_wf_{kind}_{y}.parquet")
        parts.append(d)
    new = pd.read_parquet(tmp / f"deep_wf_2021_2023_{x}.parquet")
    parts.append(new)
    out = pd.concat(parts)
    out.index = out["panel_index"].to_numpy()
    out = out.sort_index()
    assert not out.index.duplicated().any()
    assert_window(out["delivery_date"], f"deep {kind}")
    lab = TD.label_panel(TD.BUILD_START, W1, PARQUET)
    m = lab.loc[out.index]
    assert (m["delivery_hour"].to_numpy() == out["delivery_hour"].to_numpy()).all() \
        and (m["zone"].to_numpy() == out["zone"].to_numpy()).all(), f"deep {kind}: panel index does not line up"
    out.to_parquet(RES / f"deep_wf_2021_2023_{x}.parquet")
    for y, part in out.groupby(out["delivery_date"].dt.year):
        part.to_parquet(RES / f"deep_wf_{kind}_{y}.parquet")
    js = json.loads((RES / "seed" / f"deep_wf_2021_2023_{x}.json").read_text())
    jn = json.loads((tmp / f"deep_wf_2021_2023_{x}.json").read_text())
    js["refits"] = [r for r in js["refits"] if pd.Timestamp(r["month"]) < pd.Timestamp(W0)] + jn["refits"]
    js["span"] = [js["span"][0], str(W1)]
    js["window_run"] = {"mode": MODE, "window": [str(W0), str(W1)], "seconds_total": jn["seconds_total"],
                        "seconds_per_refit_mean": jn["seconds_per_refit_mean"]}
    for k in ("auc_by_year", "metrics_by_year", "p_star_choice"):
        js.pop(k, None)
    if kind == "spike":
        js["p_star_choice"] = TD.choose_p_star(out, lab, RES)
    (RES / f"deep_wf_2021_2023_{x}.json").write_text(json.dumps(js, indent=1, default=str))
    log(f"deep {kind}: {len(out)} rows in the combined file")


# ====================================================================== day features, deep day, policy, idea 14
FEATS = RES / "day_features_2020_2023.parquet"            # the build's file name; here it runs to the window's end


def stage_dayfeat():
    import deep_day as DY
    bid = [dt.date(2019, 12, 31) + dt.timedelta(days=i) for i in range((W1 - dt.timedelta(days=1)
                                                                          - dt.date(2019, 12, 31)).days + 1)]
    df = DY.build_day_features(bid, P.LockedStore(end=READ_END), log=log)
    assert_window(df["delivery_date"], "day features")
    RES.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATS)
    return {"days": len(df), "columns": int(df.shape[1])}


def stage_deepday():
    import torch
    import deep_day as DY
    import train_deep_day as TDD
    torch.set_num_threads(2)                               # as the build (train_deep_day.py --threads 2)
    df = pd.read_parquet(FEATS)
    # On the CPU, as the build's run was: its 2023-01 refit's best epochs are reproduced on the CPU only (the GPU
    # gives other, equally repeatable, epochs), so the registered walk-forward is the CPU one.
    out, recs = TDD.walk(df, (W0, W1), DY.SEEDS, "cpu", log=log)
    seed_rows = pd.read_parquet(RES / "seed" / "deep_day_wf_before_window.parquet")
    allr = pd.concat([seed_rows, out], ignore_index=True)
    allr["delivery_date"] = pd.to_datetime(allr["delivery_date"])
    assert_window(allr["delivery_date"], "deep day")
    allr.to_parquet(RES / "deep_day_wf_2021_2023.parquet")
    j = json.loads((RES / "seed" / "deep_day_wf_before_window.json").read_text())
    j["refits"] = j["refits"] + recs
    j["span"] = [j["span"][0], str(W1)]
    j["window_run"] = {"mode": MODE, "window": [str(W0), str(W1)], "refits": len(recs),
                       "seconds_per_refit_mean": float(np.mean([r["seconds"] for r in recs]))}
    (RES / "deep_day_wf_2021_2023.json").write_text(json.dumps(j, indent=1, default=str))
    return {"refits": len(recs)}


def _dp():
    import deep_policy as DP
    DP.RESULTS, DP.PARQUET, DP.FEATS = RES, PARQUET, FEATS
    DP.END = pd.Timestamp(READ_END, tz=common.TZ)
    DP.END_NAIVE = pd.Timestamp(READ_END)
    for y in range(2024, W1.year + 1):                    # storm_value's tables stop at 2023; fees.py has the rest
        DP.COST.setdefault(y, fees.SUPPLY_RATES[y])
        DP.LOADCOST.setdefault(y, fees.RATES[y])
    return DP


def stage_policy(idea: int):
    """One idea's monthly refits over the window, every lambda, one CPU thread (as the build)."""
    import torch
    torch.set_num_threads(1)
    DP = _dp()
    feats = DP.load_day_features(FEATS)
    tg = DP.Targets(DP.load_prices(PARQUET / "prices_zone.parquet"))
    parts = RES / "deep_policy_parts"
    parts.mkdir(parents=True, exist_ok=True)
    n = 0
    for Y in WY:
        pq, js = parts / f"idea{idea}_{Y}.parquet", parts / f"idea{idea}_{Y}.json"
        if pq.exists() and js.exists():
            continue
        t0 = time.time()
        p, r = DP.walk_forward(feats, tg, idea, max(W0, dt.date(Y, 1, 1)), min(W1, dt.date(Y, 12, 31)), DP.LAMBDAS,
                               DP.SEEDS, None, "cpu", log)
        p.to_parquet(pq)
        js.write_text(json.dumps({"refits": r, "seconds": round(time.time() - t0, 1), "config": DP.CONFIG,
                                  "seeds": list(DP.SEEDS), "threads": 1}, default=str))
        n += len(r)
    return {"refits": n}


def stage_policy_merge():
    DP = _dp()
    tg = DP.Targets(DP.load_prices(PARQUET / "prices_zone.parquet"))
    years = list(range(FIRST_YEAR, W1.year + 1))
    parts, recs = [], []
    for idea in (11, 12):
        for Y in range(2020, W1.year + 1):
            pq = RES / "deep_policy_parts" / f"idea{idea}_{Y}.parquet"
            p = pd.read_parquet(pq)
            p["delivery_date"] = pd.to_datetime(p["delivery_date"])
            parts.append(p)
            recs += json.loads(pq.with_suffix(".json").read_text())["refits"]
    allpos = pd.concat(parts, ignore_index=True)
    assert_window(allpos["delivery_date"], "policy")
    choice = {idea: DP.choose_lambdas(allpos[allpos["idea"] == idea], tg, years) for idea in (11, 12)}
    fin = []
    for idea, ch in choice.items():
        for Y in years:
            fin.append(allpos[(allpos["idea"] == idea) & (allpos["delivery_date"].dt.year == Y)
                              & (allpos["lam"] == ch[Y]["lam"])])
    final = pd.concat(fin, ignore_index=True).sort_values(["idea", "delivery_date", "zone"], kind="stable")
    assert (pd.to_datetime(final["refit_month"]) == final["delivery_date"].dt.to_period("M").dt.start_time).all()
    assert not final.duplicated(["idea", "delivery_date", "zone"]).any()
    final.reset_index(drop=True).to_parquet(RES / "deep_policy_wf_2021_2023.parquet")
    (RES / "deep_policy_wf_2021_2023.json").write_text(json.dumps(
        {"what": "ideas 11 and 12, build years from the build's parts, window years refitted here",
         "lambda_choice": {str(k): {str(y): v for y, v in c.items()} for k, c in choice.items()},
         "refits": recs, "lambdas": list(DP.LAMBDAS), "config": DP.CONFIG}, indent=1, default=str))
    return {"lambda": {str(k): {str(y): v["lam"] for y, v in c.items()} for k, c in choice.items()}}


def stage_alloc14():
    import torch
    torch.set_num_threads(2)
    DP = _dp()
    import deep_alloc14 as A
    A.LEAD_POS = SIDE / "lead_positions_2021_2023.parquet"
    A.OUT = RES / "idea14_wf_2021_2023.parquet"
    _wait_gpu()
    px = DP.load_prices(PARQUET / "prices_zone.parquet")
    tg = DP.Targets(px)
    cp = A.CandProfits(px, A.load_candidates(A.LEAD_POS), tg)
    feats = DP.load_day_features(FEATS)
    pdir = RES / "idea14_parts"
    pdir.mkdir(parents=True, exist_ok=True)
    for Y in WY:
        pq, js = pdir / f"idea14_{Y}.parquet", pdir / f"idea14_{Y}.json"
        if pq.exists() and js.exists():
            continue
        t0 = time.time()
        w, r = A.walk_forward(feats, tg, cp, max(W0, dt.date(Y, 1, 1)), min(W1, dt.date(Y, 12, 31)), A.KAPPAS,
                              DP.SEEDS, None, "cuda" if torch.cuda.is_available() else "cpu", log)
        w.to_parquet(pq)
        js.write_text(json.dumps({"refits": r, "seconds": round(time.time() - t0, 1), "config": A.CONFIG,
                                  "seeds": list(DP.SEEDS)}, default=str))
    years = list(range(FIRST_YEAR, W1.year + 1))
    ws, recs = [], []
    for Y in years:
        pq = pdir / f"idea14_{Y}.parquet"
        w = pd.read_parquet(pq)
        w["delivery_date"] = pd.to_datetime(w["delivery_date"])
        ws.append(w)
        recs += json.loads(pq.with_suffix(".json").read_text())["refits"]
    allw = pd.concat(ws, ignore_index=True)
    assert_window(allw["delivery_date"], "idea14")
    choice = A.choose_kappas(allw, tg, cp, years)
    final = A.final_rows(allw, choice)
    final.to_parquet(A.OUT)
    A.OUT.with_suffix(".json").write_text(json.dumps(
        {"what": "idea 14, build years from the build's parts, window years refitted here",
         "kappa_choice": {str(k): v for k, v in choice.items()}, "refits": recs, "kappas": list(A.KAPPAS)},
        indent=1, default=str))
    return {"kappa": {str(k): v["kappa"] for k, v in choice.items()}}


# ====================================================================== the lead's positions (side/lead_export.py)
LEAD_ROWS = ["always_supply", "baseline", "A_hourly_mean", "A_regression_v1", "C_gbm", "A_spike_gbm", "B_gbm", "D_gbm",
             "A_regression_v1_deep", "C_deep", "B_deep", "D_deep", "A_spike_deep", "storm_day_filter", "storm_day_flip",
             "A_spike_tail_S100_gbm", "risk_sized_supply_gbm", "zone_subset_supply", "A_spike_deep_rolling_cut"]


def stage_export():
    """side/lead_export.py main(), every year 2021 to the window's end, from cached predictions only."""
    E = ev()
    import gbm
    W = E.W

    def refuse(*a, **k):
        raise RuntimeError("export: a prediction is missing; refusing to fit")
    saved = (W.walk_forward, gbm.GBMModel.fit, gbm.SpikeClassifier.fit)
    W.walk_forward, gbm.GBMModel.fit, gbm.SpikeClassifier.fit = refuse, refuse, refuse
    try:
        panel, cols = load_panel()
        parts, totals, choices = [], {}, {}
        for lab in E.labels():
            L, ch = E.ledgers_for(panel, cols, lab)
            missing = sorted(set(LEAD_ROWS) - set(L))
            assert not missing, f"{lab}: no ledger for {missing}"
            choices[str(lab)] = {k: v[1] for k, v in ch.items()}
            for name, led in L.items():
                assert_window(led["delivery_hour"], name)
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
            log(f"export {lab}: {len(L)} strategies")
        out = pd.concat(parts, ignore_index=True)
        SIDE.mkdir(parents=True, exist_ok=True)
        out.to_parquet(SIDE / "lead_positions_2021_2023.parquet")
        G = E._shared(panel, cols)
        st = G["storm"].rename("storm").rename_axis("delivery_date").reset_index()
        st.to_parquet(SIDE / "lead_storm_score.parquet")
        (SIDE / "lead_ledger_totals.json").write_text(json.dumps({"totals": totals, "choices": choices}, indent=1,
                                                                default=str))
    finally:
        W.walk_forward, gbm.GBMModel.fit, gbm.SpikeClassifier.fit = saved
    return {"rows": len(out), "strategies": int(out["strategy"].nunique())}


# ====================================================================== the lab's scoring over the window
def lab_module():
    import lab as LB
    sys.path[:] = [p for p in sys.path if p != str(BUILD_HOME / "pipeline")]   # lab.py adds the build copy's
    LB.HOME = RUN                                          # load_tail_cube reads HOME/cache/wf
    LB.PQ, LB.RES, LB.SIDE, LB.OPT = PARQUET, RES, SIDE, RES
    LB.END = pd.Timestamp(READ_END, tz=common.TZ)
    LB.END_NAIVE = pd.Timestamp(READ_END)
    LB.DAY_FEATS, LB.DEEP_DAY = FEATS, RES / "deep_day_wf_2021_2023.parquet"
    LB.POLICY, LB.IDEA14 = RES / "deep_policy_wf_2021_2023.parquet", RES / "idea14_wf_2021_2023.parquet"
    LB.LEAD_POS, LB.LEAD_TOT = SIDE / "lead_positions_2021_2023.parquet", SIDE / "lead_ledger_totals.json"
    LB.LEAD_STORM = SIDE / "lead_storm_score.parquet"
    for y in range(2024, W1.year + 1):
        LB.COST.setdefault(y, fees.SUPPLY_RATES[y])
        LB.LOADCOST.setdefault(y, fees.RATES[y])
    LB.YEARS = list(WY)

    def gbm_idea7(F, df):                                  # lab.gbm_idea7 with the window's last month
        cols = [c for c in df.columns if "__" in c and not c.startswith("y__")]
        p = pd.Series(np.nan, index=F.days)
        recs = []
        for ms in LB.month_starts(LB.GBM_FIRST_MONTH, dt.date(W1.year, W1.month, 1)):
            me = pd.Timestamp(ms) + pd.offsets.MonthEnd(0)
            tr = LB.gbm_train_rows(df, ms)
            te = df[(df["delivery_date"] >= pd.Timestamp(ms)) & (df["delivery_date"] <= me)]
            assert tr["delivery_date"].max() < te["delivery_date"].min()
            pr, L = LB.gbm_fit_predict(tr, te, cols)
            p.loc[pd.DatetimeIndex(te["delivery_date"])] = pr
            recs.append({"month": str(ms), "train_days": len(tr), "train_last": str(tr["delivery_date"].max().date()),
                         "L": round(L, 1), "positives": int((tr["y__book_supply_pnl"] < L).sum())})
        return p.to_numpy(float), recs, cols
    LB.gbm_idea7 = gbm_idea7

    def load_tail_cube(F):                                 # lab.load_tail_cube over every year, not only the scored
        fs = [CACHE / "wf" / f"spike_S100_{y}.parquet" for y in range(FIRST_YEAR, W1.year + 1)]
        if not all(f.exists() for f in fs):
            return None, "pending: " + ", ".join(str(f) for f in fs if not f.exists())
        d = pd.concat([LB.read_build(f, "delivery_hour") for f in fs], ignore_index=True)
        m = F.h[KEY].merge(d, on=KEY, how="left", validate="one_to_one")
        v = m["pred"].to_numpy(float)
        return F.to_cube(v), f"cache/wf/spike_S100_{FIRST_YEAR}..{W1.year}.parquet"
    LB.load_tail_cube = load_tail_cube
    return LB


def build_lab_strats(LB, px, lf, wx, F):
    """side/lab.py main()'s strategies, in its order, with its inputs and notes."""
    strats, notes, pending = [], {}, {}
    sig = LB.baseline_cube(F, px)
    basics = LB.mk_basics(F, sig)
    strats += basics
    storm, _ = LB.storm_score(LB.storm_inputs(F, px, lf, wx))
    strats += LB.mk_storm(F, storm)
    wrank, _ = LB.weather_surprise(F, wx)
    strats.append(LB.mk_weather(F, wrank))
    tail, tail_src = LB.load_tail_cube(F)
    if tail is not None:
        st9 = LB.mk_spike_rule("idea9_tail_spike_S100", "9", "idea 9 (see side/lab.py)", "p_tail", tail)
        st9.source = "lead modeller (cache/wf/spike_S100_*.parquet)"
        strats.append(st9)
    else:
        pending["idea9_tail_spike_S100"] = tail_src
    dsp_cube, dsp_meta, dsp_src = LB.load_deep_spike_cube(F)
    notes["deep_spike_file"] = {"source": dsp_src, "timing_metadata": dsp_meta}
    strats += LB.deep_day_strats(F, px, dsp_cube, notes, pending)
    strats += LB.policy_strats(F, notes, pending)
    strats += LB.idea14_strats(F, notes, pending)
    strats.append(LB.mk_hist(F))
    lead, lead_same = LB.load_lead(F, notes)
    strats += lead
    return strats, notes, pending, {"sig": sig, "storm": storm, "wrank": wrank, "basics": basics,
                                    "lead_same": lead_same}


def _scaled(F, LB, mw, scale):
    """Window figures at a carried size: daily net, cost and drawdown scale linearly with the book."""
    d = F.day_pnl(mw)
    m = np.isin(F.day_year, LB.YEARS)
    dd = scale * d[m]
    cum = np.concatenate([[0.0], np.cumsum(dd)])
    mdd = float((cum - np.maximum.accumulate(cum)).min())
    n = int(m.sum())
    return {"scale_mw": scale, "net_usd": round(float(dd.sum())),
            "return_on_500k_annualised_pct": round(100 * float(dd.sum()) / LB.BANK * 365 / n, 1),
            "max_drawdown_usd": round(mdd), "days": n,
            "years": {str(Y): round(float(scale * d[F.day_year == Y].sum())) for Y in LB.YEARS}}


def stage_lab():
    LB = lab_module()
    t0 = time.time()
    px, lf, wx = LB.load_tables()
    F = LB.Frame(px)
    assert_window(F.h["delivery_hour"], "lab frame")
    assert F.days[-1] == pd.Timestamp(W1), "the lab's frame does not reach the window's last day"
    strats, notes, pending, aux = build_lab_strats(LB, px, lf, wx, F)
    build_lab = json.loads((BUILD_HOME / "side" / "lab_results.json").read_text())["strategies"]
    robust = json.loads((BUILD_HOME / "results" / "robust" / "robustness_summary.json").read_text())
    RD = LB.RandomDays(F)
    BT = LB.Boot(int(np.isin(F.day_year, LB.YEARS).sum()))
    results, mws, choices_storm = {}, {}, {}
    for st in strats:
        log("walk", st.name)
        inp = st.inputs
        mw, ch = LB.walk(F, st, inp)
        mws[st.name] = mw
        res = LB.evaluate(F, mw)
        res["choices"] = {str(Y): {"setting": LB._js(ch[Y][0]), "how": ch[Y][1]} for Y in LB.YEARS}
        res["placebo"] = LB.placebos(F, st, inp, res["three_year"]["net_usd"])
        res["lookahead_rule_and_choice"] = LB.la_generic(F, st, inp, ch) if inp else {"pass": True, "note": "no inputs"}
        res["too_good_checks"] = LB.too_good_diag(F, st, inp, res, ch)
        res["fragility"] = LB.fragility(F, st, inp, ch, res, RD, BT)
        res["FRAGILE"] = res["fragility"]["FRAGILE"]
        res["idea"], res["line"], res["source"], res["notes"] = st.idea, st.line, st.source, st.notes
        if st.incomplete:
            res["incomplete"] = st.incomplete
            res["PASS"] = None
        if st.name in ("idea1_storm_filter", "idea2_storm_flip", "idea8_storm_zone_pairs"):
            choices_storm[st.name] = ch
        if st.lookahead_inputs is not None:
            res["lookahead_inputs"] = st.lookahead_inputs
        b = build_lab.get(st.name, {}).get("sizing_view", {}).get("scale_for_100k_drawdown")
        res["sizing_view_carried"] = _scaled(F, LB, mw, b) if b else {"note": "no build-year scale"}
        results[st.name] = res
        if st.name == "idea10_weather_surprise":
            res["lookahead_inputs"] = LB.la_weather(F, wx, aux["wrank"], ch)
    la_b, _ = LB.la_baseline(F, px, aux["sig"], aux["basics"][1:])
    for k, v in la_b.items():
        results[k]["lookahead_inputs"] = v
    results["always_supply"]["lookahead_inputs"] = {"pass": True, "note": "no inputs: the position never changes"}
    la_s = LB.la_storm(F, px, lf, wx, aux["storm"], choices_storm)
    for k in choices_storm:
        results[k]["lookahead_inputs"] = LB.la_summary_storm(la_s, k)

    # idea 13: the build's survivors with the build's weights (MW per position), netted per zone-hour
    i13 = robust["idea13"]
    w13 = idea13_weights(i13)
    mw13 = np.zeros(F.NH)
    for k, w in w13.items():
        mw13 += w * mws[k]
    r13 = LB.evaluate(F, mw13)
    real_years = {Y: float(r13["_daily"][F.day_year == Y].sum()) for Y in LB.YEARS}
    r13["fragility"] = {"random_days": RD.run(mw13, real_years),
                        "bootstrap_ci95_daily_mean_usd": BT.ci_mean(r13["_daily"][np.isin(F.day_year, LB.YEARS)])}
    r13["components_weight_mw"] = w13
    k13 = i13["sizing"]["multiplier_for_100k_drawdown_capped_at_5MW_per_zone_hour"]
    r13["sizing_view_carried"] = _scaled(F, LB, mw13, k13)
    r13["idea"], r13["line"], r13["source"] = "13", "Equal-risk portfolio of the five survivors (build weights).", \
        "robust/summary.py (weights from results/robust/robustness_summary.json)"
    results["idea13_portfolio"] = r13
    mws["idea13_portfolio"] = mw13

    m = np.isin(F.day_year, LB.YEARS)
    daily = pd.DataFrame({k: v["_daily"][m] for k, v in results.items()}, index=F.days[m]).rename_axis("delivery_date")
    daily.to_parquet(SIDE / "lab_daily_heldout.parquet")
    for v in results.values():
        v.pop("_daily", None)
    meta = {"written": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "seconds": round(time.time() - t0), "mode": MODE,
            "window": [str(W0), str(W1)], "years": LB.YEARS, "costs_supply": {str(k): v for k, v in LB.COST.items()},
            "costs_load": {str(k): v for k, v in LB.LOADCOST.items()}, "cost_sources": fees.SOURCE,
            "stress_usd_per_mwh": LB.STRESS, "pending": pending, "n_shuffles": LB.N_SHUFFLE,
            "note": ("'three_year' in each row is the whole window (the lab's field name); 'years' are calendar years "
                     "of the window" + ("; 2026 runs to 30 September" if W1.year == 2026 else "")
                     + "; sizing_view is recomputed on the window, sizing_view_carried uses the build's scale")}
    out = {"meta": meta, "notes": notes, "strategies": results, "lookahead_storm_days": la_s}
    (SIDE / "lab_results_heldout.json").write_text(json.dumps(out, indent=1, default=LB._js))
    write_md(json.loads(json.dumps(out, default=LB._js)), SIDE / "lab_results_heldout.md")
    _combined(daily, results, build_lab, i13)
    _build_years_check(LB, F, strats, mws, i13)
    return {"rows": len(results), "pending": pending}


def idea13_weights(i13: dict) -> dict:
    """robust/summary.py idea13's weights, unrounded: scale = min(5, 100,000 / the component's three-year max
    drawdown at 1 MW, which the lab rounds to the dollar), divided by the number of components (the JSON's
    weight_mw is rounded to 4 decimals)."""
    comps = i13["components"]
    LB = lab_module()
    return {k: min(LB.SIZE_CAP, LB.SIZE_DD / -v["max_drawdown_1MW"]) / len(comps) for k, v in comps.items()}


def _combined(daily, results, build_lab, i13):
    """Build-year daily (side/lab_daily.parquet) plus the window, and a results file whose rows keep the build's
    flags and sizing view (the selection was fixed on the build years) with the window's years added."""
    bd = pd.read_parquet(BUILD_HOME / "side" / "lab_daily.parquet")
    bd.index = pd.DatetimeIndex(bd.index)
    bd = bd[bd.index < pd.Timestamp(W0)]
    i13d = pd.read_parquet(SIDE / "idea13_build_daily.parquet") if (SIDE / "idea13_build_daily.parquet").exists() \
        else None
    if i13d is not None:
        bd["idea13_portfolio"] = i13d["idea13_portfolio"].reindex(bd.index)
    comb = pd.concat([bd[[c for c in daily.columns if c in bd.columns]], daily])
    comb.index.name = "delivery_date"
    comb.to_parquet(SIDE / f"lab_daily_{END_TAG}.parquet")
    rows = {}
    for k, r in results.items():
        b = build_lab.get(k)
        if b is None and k == "idea13_portfolio":
            b = {"years": {y: {"net_usd": v} for y, v in i13["as_built"]["years"].items()},
                 "three_year": {"net_usd": i13["as_built"]["three_year_net"], "sharpe": i13["as_built"]["sharpe"],
                                "max_drawdown_usd": i13["as_built"]["max_drawdown"]},
                 "sizing_view": {"scale_for_100k_drawdown": i13["sizing"]["multiplier_for_100k_drawdown_capped_at_5MW_per_zone_hour"],
                                 "return_on_500k_at_that_scale_pct": i13["sizing"]["return_on_500k_pct"]},
                 "bar": {"positive_years": True, "sharpe_3y": True, "stress_total": True}, "FRAGILE": False,
                 "PASS": None, "idea": "13", "line": r["line"]}
        b = json.loads(json.dumps(b or {}, default=str))
        yrs = {y: v for y, v in (b.get("years") or {}).items() if int(y) < W0.year}
        yrs.update(r["years"])
        rows[k] = {**b, "years": yrs, "heldout": {"window": [str(W0), str(W1)], "total": r["three_year"],
                                                    "sizing_view_carried": r.get("sizing_view_carried"),
                                                    "FRAGILE_on_window": r.get("FRAGILE")}}
    (SIDE / f"lab_results_{END_TAG}.json").write_text(json.dumps(
        {"meta": {"what": "build-year rows (side/lab_results.json, idea 13 from robustness_summary.json) with the "
                          "window's years added; flags and sizing view are the build's", "window": [str(W0), str(W1)],
                  "mode": MODE}, "strategies": rows}, indent=1, default=str))


def _build_years_check(LB, F, strats, mws_window, i13):
    """Every row walked over the build years before the window with the same code and inputs: its daily net must
    equal the build's side/lab_daily.parquet. Also writes idea 13's build-year daily for the combined table."""
    years_before = [y for y in BUILD_YEARS if y < W0.year]
    if not years_before:
        return
    bd = pd.read_parquet(BUILD_HOME / "side" / "lab_daily.parquet")
    bd.index = pd.DatetimeIndex(bd.index)
    keep = LB.YEARS
    LB.YEARS = years_before
    try:
        lead = {s.name: s for s in LB.load_lead(F, {})[0]}   # the lead rows' settings are read per scored year
        strats = [lead.get(s.name, s) for s in strats]
        m = np.isin(F.day_year, years_before)
        out, mws = {}, {}
        for st in strats:
            mw, _ = LB.walk(F, st, st.inputs)
            mws[st.name] = mw
            d = pd.Series(F.day_pnl(mw)[m], index=F.days[m])
            if st.name in bd.columns:
                diff = (d - bd[st.name].reindex(d.index)).abs()
                out[st.name] = {"max_abs_daily_diff_usd": float(diff.max()),
                                "net_run": round(float(d.sum()), 2), "net_build": round(float(bd[st.name].reindex(d.index).sum()), 2)}
        mw13 = sum(w * mws[k] for k, w in idea13_weights(i13).items())
        d13 = pd.Series(F.day_pnl(mw13)[m], index=F.days[m])
        pd.DataFrame({"idea13_portfolio": d13}).rename_axis("delivery_date").to_parquet(SIDE / "idea13_build_daily.parquet")
        out["idea13_portfolio"] = {"net_run_by_year": {str(y): round(float(d13[d13.index.year == y].sum())) for y in years_before},
                                   "net_build_by_year": {y: v for y, v in i13["as_built"]["years"].items()
                                                         if int(y) in years_before}}
    finally:
        LB.YEARS = keep
    (SIDE / "build_years_check.json").write_text(json.dumps(out, indent=1, default=str))
    worst = max(v["max_abs_daily_diff_usd"] for v in out.values() if "max_abs_daily_diff_usd" in v)
    log(f"build years {years_before} recomputed: worst daily difference against the build {worst:.6f} USD")
    # the combined table was written before this check: refresh it with idea 13's build-year days
    j = json.loads((SIDE / "lab_results_heldout.json").read_text())
    daily = pd.read_parquet(SIDE / "lab_daily_heldout.parquet")
    _combined(daily, j["strategies"], json.loads((BUILD_HOME / "side" / "lab_results.json").read_text())["strategies"], i13)


def _f(x, nd=0):
    if x is None:
        return "n/a"
    if isinstance(x, str):
        return x
    return f"{x:,.{nd}f}"


def write_md(out: dict, path: Path):
    S, meta = out["strategies"], out["meta"]
    ys = [str(y) for y in meta["years"]]
    order = sorted(S, key=lambda k: -S[k]["three_year"]["net_usd"])
    L = [f"# Version 1, {meta['mode']} run: {meta['window'][0]} to {meta['window'][1]}", "",
         f"Written {meta['written']} by `model/heldout_v1.py`. Every row walked forward as in the build years, "
         "scored by side/lab.py's scoring (full costs, 1 MW per zone-hour). " + meta["note"] + ".", "",
         "| # | Strategy | " + " | ".join(ys) + " | Window total | Sharpe | Max drawdown | At 0.50 stress "
         "| One day late | Shuffled (share at or above) | Random days beat it | Build scale (MW) | Window net at that scale "
         "| Return a year at that scale | FRAGILE on the window |",
         "|" + "---|" * (13 + len(ys))]
    for i, k in enumerate(order, 1):
        v = S[k]
        t, pl = v["three_year"], v.get("placebo") or {}
        rd = (v.get("fragility") or {}).get("random_days")
        rds = "n/a" if not isinstance(rd, dict) else f"{100 * rd['share_random_beats_rule']:.0f}%"
        shuf = "n/a" if pl.get("shuffled_mean_total") is None else \
            f"{_f(pl['shuffled_mean_total'])} ({pl['shuffled_share_at_or_above_real']:.2f})"
        sv = v.get("sizing_view_carried") or {}
        L.append(f"| {i} | {k} | " + " | ".join(_f(v["years"][y]["net_usd"]) for y in ys)
                 + f" | {_f(t['net_usd'])} | {_f(t['sharpe'], 2)} | {_f(t['max_drawdown_usd'])} | {_f(t['stress_0.50_usd'])}"
                 f" | {_f(pl.get('shifted_one_day_total'))} | {shuf} | {rds} | {sv.get('scale_mw', 'n/a')}"
                 f" | {_f(sv.get('net_usd'))} | {sv.get('return_on_500k_annualised_pct', 'n/a')}%"
                 f" | {'FRAGILE' if v.get('FRAGILE') else 'no'} |")
    L += ["", "## Settings chosen on the year before", "", "| Strategy | " + " | ".join(ys) + " |",
          "|" + "---|" * (1 + len(ys))]
    for k in order:
        c = S[k].get("choices") or {}
        L.append(f"| {k} | " + " | ".join(str((c.get(y) or {}).get("setting")) for y in ys) + " |")
    if meta.get("pending"):
        L += ["", "Pending (not scored): " + "; ".join(f"{k}: {v}" for k, v in meta["pending"].items())]
    path.write_text("\n".join(L) + "\n")


# ====================================================================== dry-run equality
def stage_compare():
    """Dry run only: the 2023 the generic path made against the 2023 the build made, row by row."""
    assert MODE == "dryrun", "compare is the dry run's proof"
    Y = W0.year
    rep: dict = {}
    # daily net per lab row
    bd = pd.read_parquet(BUILD_HOME / "side" / "lab_daily.parquet")
    bd.index = pd.DatetimeIndex(bd.index)
    rd = pd.read_parquet(SIDE / "lab_daily_heldout.parquet")
    rd.index = pd.DatetimeIndex(rd.index)
    bd = bd[bd.index.year == Y]
    rows = {}
    for c in sorted(set(bd.columns) | set(rd.columns)):
        if c not in bd.columns or c not in rd.columns:
            rows[c] = {"status": "missing in " + ("build" if c not in bd.columns else "dry run")}
            continue
        a, b = rd[c].reindex(bd.index), bd[c]
        diff = (a - b).abs()
        rows[c] = {"days": int(len(b)), "days_exactly_equal": int((a == b).sum()), "max_abs_daily_diff_usd": float(diff.max()),
                   "net_dryrun": round(float(a.sum()), 2), "net_build": round(float(b.sum()), 2),
                   "net_diff_usd": round(float(a.sum() - b.sum()), 6)}
    rep["lab_daily_2023"] = rows
    i13 = json.loads((BUILD_HOME / "results" / "robust" / "robustness_summary.json").read_text())["idea13"]
    rep["idea13_2023"] = {"net_dryrun": round(float(rd["idea13_portfolio"].sum())),
                          "net_build": i13["as_built"]["years"][str(Y)]}
    # lab choices
    bl = json.loads((BUILD_HOME / "side" / "lab_results.json").read_text())["strategies"]
    rl = json.loads((SIDE / "lab_results_heldout.json").read_text())["strategies"]
    rep["lab_choices_2023"] = {k: {"dryrun": rl[k]["choices"][str(Y)]["setting"], "build": bl[k]["choices"][str(Y)]["setting"],
                                   "equal": rl[k]["choices"][str(Y)]["setting"] == bl[k]["choices"][str(Y)]["setting"]}
                               for k in rl if k in bl and "choices" in rl[k]}
    # lead positions
    bp = pd.read_parquet(BUILD_HOME / "side" / "lead_positions_2021_2023.parquet")
    rp = pd.read_parquet(SIDE / "lead_positions_2021_2023.parquet")
    pos = {}
    for name in sorted(set(bp["strategy"]) | set(rp["strategy"])):
        out = {}
        for y in BUILD_YEARS:
            a = bp[(bp["strategy"] == name) & (bp["year"] == y)].set_index(KEY)
            b = rp[(rp["strategy"] == name) & (rp["year"] == y)].set_index(KEY)
            if not len(a) or not len(b):
                out[str(y)] = {"status": f"rows build {len(a)}, dry run {len(b)}"}
                continue
            j = a[["mw", "pred"]].join(b[["mw", "pred"]], how="outer", lsuffix="_b", rsuffix="_r")
            eqp = (j["pred_b"] == j["pred_r"]) | (j["pred_b"].isna() & j["pred_r"].isna())
            out[str(y)] = {"rows_build": len(a), "rows_dryrun": len(b),
                           "mw_exactly_equal_share": float((j["mw_b"] == j["mw_r"]).mean()),
                           "mw_max_abs_diff": float((j["mw_b"] - j["mw_r"]).abs().max()),
                           "pred_exactly_equal_share": float(eqp.mean()),
                           "pred_max_abs_diff": float((j["pred_b"] - j["pred_r"]).abs().max())}
        pos[name] = out
    rep["lead_positions"] = pos
    # evaluate's choices for 2023
    bc = json.loads((BUILD_HOME / "side" / "lead_ledger_totals.json").read_text())["choices"][str(Y)]
    rc = json.loads((SIDE / "lead_ledger_totals.json").read_text())["choices"][str(Y)]

    def core(c):
        return {k: {kk: vv for kk, vv in v.items() if kk not in ("how", "scores_mean_daily_net", "top", "table", "chosen_on")}
                for k, v in c.items()}
    rep["evaluate_choices_2023"] = {"build": core(bc), "dryrun": core(rc), "equal": core(bc) == core(rc),
                                    "build_how": {k: v.get("how") for k, v in bc.items()},
                                    "dryrun_how": {k: v.get("how") for k, v in rc.items()}}
    # deep prediction files
    dp = {}
    for kind, col in (("base", "pred_gap"), ("weather", "pred_gap"), ("outages", "pred_gap"), ("spike", "p_spike")):
        a = pd.read_parquet(BUILD_HOME / "results" / f"deep_wf_{kind}_{Y}.parquet").set_index(KEY)[col]
        b = pd.read_parquet(RES / f"deep_wf_{kind}_{Y}.parquet").set_index(KEY)[col]
        j = pd.concat([a.rename("b"), b.rename("r")], axis=1)
        dp[kind] = {"rows": len(j), "exactly_equal_share": float((j["b"] == j["r"]).mean()),
                    "max_abs_diff": float((j["b"] - j["r"]).abs().max()),
                    "mean_abs_diff": float((j["b"] - j["r"]).abs().mean())}
    bj = json.loads((BUILD_HOME / "results" / "deep_wf_spike_choice.json").read_text())
    rj = json.loads((RES / "deep_wf_spike_choice.json").read_text())
    dp["spike_p_star"] = {y: {"build": bj.get(y), "dryrun": rj.get(y)} for y in ("2022", "2023")}
    rep["deep_predictions_2023"] = dp
    # deep day
    a = pd.read_parquet(BUILD_HOME / "results" / "deep_day_wf_2021_2023.parquet")
    b = pd.read_parquet(RES / "deep_day_wf_2021_2023.parquet")
    for d in (a, b):
        d["delivery_date"] = pd.to_datetime(d["delivery_date"])
    a, b = (d[d["delivery_date"].dt.year == Y].set_index(["delivery_date", "zone"]) for d in (a, b))
    j = a[["p_storm", "pred_zone_supply_pnl"]].join(b[["p_storm", "pred_zone_supply_pnl"]], lsuffix="_b", rsuffix="_r")
    rep["deep_day_2023"] = {c: {"exactly_equal_share": float((j[c + "_b"] == j[c + "_r"]).mean()),
                                "max_abs_diff": float((j[c + "_b"] - j[c + "_r"]).abs().max())}
                            for c in ("p_storm", "pred_zone_supply_pnl")}
    # policy and idea 14
    for name, fn, keys, cols in (("policy_2023", "deep_policy_wf_2021_2023.parquet", ["idea", "delivery_date", "zone"],
                                  ["position", "lam"]),
                                 ("idea14_2023", "idea14_wf_2021_2023.parquet", ["delivery_date"],
                                  ["w_C_deep", "w_B_gbm", "w_C_gbm", "w_B_deep", "scale_C_deep", "scale_B_gbm",
                                   "scale_C_gbm", "scale_B_deep", "kappa"])):
        a = pd.read_parquet(BUILD_HOME / "results" / fn)
        b = pd.read_parquet(RES / fn)
        for d in (a, b):
            d["delivery_date"] = pd.to_datetime(d["delivery_date"])
        a, b = (d[d["delivery_date"].dt.year == Y].set_index(keys) for d in (a, b))
        j = a[cols].join(b[cols], lsuffix="_b", rsuffix="_r")
        rep[name] = {c: {"exactly_equal_share": float((j[c + "_b"] == j[c + "_r"]).mean()),
                         "max_abs_diff": float((j[c + "_b"] - j[c + "_r"]).abs().max())} for c in cols}
    # data products rebuilt here
    a = pd.read_parquet(BUILD_HOME / "results" / "day_features_2020_2023.parquet")
    b = pd.read_parquet(FEATS)
    same_cols = list(a.columns) == list(b.columns)
    eq = None
    if same_cols and len(a) == len(b):
        x, y_ = a.drop(columns=["bid_date", "delivery_date"]), b.drop(columns=["bid_date", "delivery_date"])
        eq = bool(((x == y_) | (x.isna() & y_.isna())).all().all())
    rep["day_features_rebuilt"] = {"same_columns": same_cols, "rows_build": len(a), "rows_run": len(b), "all_equal": eq}
    a = pd.read_parquet(BUILD_HOME / "cache" / "panel_base.parquet")
    b = pd.read_parquet(CACHE / "panel_base.parquet")
    eq = None
    if list(a.columns) == list(b.columns) and len(a) == len(b):
        eq = bool(all(((a[c] == b[c]) | (a[c].isna() & b[c].isna())).all() for c in a.columns))
    rep["panel_base_rebuilt"] = {"rows_build": len(a), "rows_run": len(b), "all_equal": eq}
    (RUN / "dryrun_equality.json").write_text(json.dumps(rep, indent=1, default=str))
    _compare_md(rep)
    return {"written": "dryrun_equality.json"}


def _compare_md(rep: dict):
    L = ["# Dry run of the version 1 held-out runner: 2023 against the build", "",
         "Window 2023-01-01 to 2023-12-31, every choice made on 2022 by the generic year-Y path, nothing after "
         "2023-12-31 read. Daily net per row (side/lab_daily.parquet, 2023):", "",
         "| Row | Days equal | Max daily difference, USD | 2023 net, dry run | 2023 net, build | Difference, USD |",
         "|---|---|---|---|---|---|"]
    for k, v in rep["lab_daily_2023"].items():
        if "status" in v:
            L.append(f"| {k} | {v['status']} | | | | |")
        else:
            L.append(f"| {k} | {v['days_exactly_equal']} of {v['days']} | {v['max_abs_daily_diff_usd']:.6g} "
                     f"| {v['net_dryrun']:,.2f} | {v['net_build']:,.2f} | {v['net_diff_usd']:,.6g} |")
    L += ["", f"Idea 13, 2023: dry run {rep['idea13_2023']['net_dryrun']:,}, build {rep['idea13_2023']['net_build']:,}.", "",
          "Deep predictions, 2023:", "", "| Model | Rows | Exactly equal | Max abs difference | Mean abs difference |",
          "|---|---|---|---|---|"]
    for k, v in rep["deep_predictions_2023"].items():
        if k == "spike_p_star":
            continue
        L.append(f"| {k} | {v['rows']} | {100 * v['exactly_equal_share']:.2f}% | {v['max_abs_diff']:.3g} | {v['mean_abs_diff']:.3g} |")
    L += ["", f"Deep spike p*: {rep['deep_predictions_2023']['spike_p_star']}",
          f"Deep day: {rep['deep_day_2023']}", f"Policy: {rep['policy_2023']}", f"Idea 14: {rep['idea14_2023']}",
          f"Evaluate's 2023 choices equal: {rep['evaluate_choices_2023']['equal']}",
          f"Lab choices equal: {all(v['equal'] for v in rep['lab_choices_2023'].values())}",
          f"Day features rebuilt: {rep['day_features_rebuilt']}", f"Panel rebuilt: {rep['panel_base_rebuilt']}"]
    (RUN / "dryrun_equality.md").write_text("\n".join(L) + "\n")


# ====================================================================== driver
def _policy_subprocesses() -> list:
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    (RUN / "logs").mkdir(parents=True, exist_ok=True)
    procs = []
    for idea in (11, 12):
        if _done(f"policy{idea}").exists():
            continue
        f = open(RUN / "logs" / f"policy{idea}.log", "a")
        procs.append((idea, subprocess.Popen([sys.executable, str(Path(__file__)), f"policy{idea}"], env=env,
                                             stdout=f, stderr=subprocess.STDOUT)))
    return procs


STAGES = {"seed": seed, "panel": stage_panel, "gbm": stage_gbm, "dayfeat": stage_dayfeat,
          "policy11": lambda: stage_policy(11), "policy12": lambda: stage_policy(12),
          "deep_base": lambda: stage_deep("base"), "deep_weather": lambda: stage_deep("weather"),
          "deep_outages": lambda: stage_deep("outages"), "deep_spike": lambda: stage_deep("spike"),
          "deepday": stage_deepday, "policy_merge": stage_policy_merge, "export": stage_export,
          "alloc14": stage_alloc14, "lab": stage_lab, "compare": stage_compare}


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def run_stage(name: str, force: bool = False):
    """Run a stage once; a second process asking for a stage another live process is running waits for it."""
    DONE.mkdir(parents=True, exist_ok=True)
    running = DONE / f"{name}.running"
    while True:
        if _done(name).exists() and not force:
            log(f"{name}: done before, skipped")
            return
        try:
            fd = os.open(running, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                pid = int(running.read_text() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and _alive(pid):
                time.sleep(30)
                continue
            running.unlink(missing_ok=True)                 # left by a process that died
    try:
        log(f"{name}: start ({MODE}, window {W0}..{W1}, run dir {RUN})")
        t0 = time.time()
        extra = STAGES[name]()
        _record(name, time.time() - t0, extra if isinstance(extra, dict) else None)
        log(f"{name}: done in {time.time() - t0:.0f} s")
    finally:
        running.unlink(missing_ok=True)


def run_all():
    for s in ("seed", "panel", "gbm", "dayfeat"):
        run_stage(s)
    procs = _policy_subprocesses()                        # CPU, one thread each, beside the GPU stages
    for s in ("deep_base", "deep_weather", "deep_outages", "deep_spike", "deepday"):
        run_stage(s)
    for idea, p in procs:
        rc = p.wait()
        if rc != 0:
            raise SystemExit(f"policy idea {idea} failed (exit {rc}); see {RUN / 'logs'}")
    for s in ("policy_merge", "export", "alloc14", "lab"):
        run_stage(s)
    if MODE == "dryrun":
        run_stage("compare")
    log(f"version 1 {MODE} run written under {RUN}")


def plan():
    print(json.dumps({"mode": MODE, "window": [str(W0), str(W1)], "run_dir": str(RUN), "cache": str(CACHE),
                      "parquet": str(PARQUET), "build_home (read only)": str(BUILD_HOME),
                      "lgbm_threads": os.environ.get("US_LGBM_THREADS"), "stages": list(STAGES),
                      "done": sorted(p.stem for p in DONE.glob("*.json")) if DONE.exists() else []}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="+", choices=["all", "plan", *STAGES])
    ap.add_argument("--force", action="store_true", help="rerun a stage that finished before")
    a = ap.parse_args()
    for st in a.stage:
        if st == "plan":
            plan()
        elif st == "all":
            run_all()
        else:
            run_stage(st, a.force)
