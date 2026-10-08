"""Held-out runner for every version 2 row (research/us_v2/HELDOUT-LIST.md: 26 candidates, 12 comparisons).

The rolling windows continue exactly as in the build: train on the previous 3 years up to 2 days before each
quarter, test on the next quarter, settings on the trailing 4 out-of-sample quarters, same fee table, same code.
Nothing of the build is changed: the run mode (research/us/model/heldout_mode.py) sends every output to
$US_RUN_DIR/results/v2, which is first SEEDED by copying the build's per-quarter predictions and fitted models of the
quarters before the window (~/nyiso-us/results/v2 is only read). Each module then refits only the window's quarters
(rolling.run_rows refuses any other) and writes positions over every quarter, the earlier ones from the seeded
predictions. V12 is carried as it ran (alloc_v12.BUILD_CANDIDATES).

    US_RUN_MODE=dryrun  US_RUN_DIR=<dir> python heldout_v2.py all      window 2023 (proof against the build)
    US_RUN_MODE=heldout US_RUN_DIR=<dir> V2_PARQUET=<dir>/parquet_v2 python heldout_v2.py all
                                                                        window 2024-01-01 .. 2026-09-30
    python heldout_v2.py seed | run | score | check | status

Steps run in two lanes (CPU and GPU, each one step at a time; a GPU step waits for 2.5 GB free GPU memory), in
dependency order. A finished step leaves steps/<step>.json (times, exit code) and is skipped on a restart; inside a
step the modules resume per quarter. Logs: $US_RUN_DIR/logs/<step>.log. Use nohup:
    nohup python heldout_v2.py all > $US_RUN_DIR/heldout_v2.log 2>&1 &
check (dry run only): every row's 2023 positions against the build's, per zone-hour, and the scorer's 2023 net
against the build's -> results/v2/dryrun_check.{json,md}.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IDEAS = HERE.parent / "ideas"
sys.path.insert(0, str(HERE))

import rolling as R  # noqa: E402

PY = os.environ.get("V2_PYTHON", sys.executable)
ROWS_LIST = HERE.parent / "HELDOUT-LIST.md"
DATA_FLAGS = ("v2_data.done", "v2_price_data.done", "v2_weather_data.done")
GPU_FREE_MB = 2500


def ideas(row):
    return (IDEAS, [PY, "-I", "run_ideas.py", row])


def model(*args):
    return (HERE, [PY, *args])


# name: (lane, (cwd, argv), extra env, dependencies). The order is the build's, the dependencies its inputs.
STEPS = {
    "gbm":           ("cpu", model("gbm_rolling.py", "--threads", "2"), {}, []),
    "deep_c":        ("gpu", model("deep_rolling.py", "c", "--device", "cuda"), {"OMP_NUM_THREADS": "1"}, []),
    "deep_v10":      ("gpu", model("deep_rolling.py", "v10", "--device", "cuda"), {"OMP_NUM_THREADS": "1"}, []),
    "V5_border_inputs": ("cpu", ideas("V5_border_inputs"), {}, []),
    "V6_NYC":        ("cpu", ideas("V6_NYC"), {}, []),
    "V6_LI":         ("cpu", ideas("V6_LI"), {}, []),
    "V7_flags":      ("cpu", ideas("V7_flags"), {}, []),
    "V9_decompose":  ("cpu", ideas("V9_decompose"), {}, []),
    "V7_recency":    ("cpu", ideas("V7_recency"), {}, []),
    "V8_error_mining": ("cpu", ideas("V8_error_mining"), {}, []),
    "V4_B_reforecast": ("cpu", ideas("V4_B_reforecast"), {}, []),
    "V16_B_gefs_joined": ("cpu", ideas("V16_B_gefs_joined"), {}, []),
    "v14a_gbm":      ("cpu", model("v14_gbm.py", "--threads", "2"), {}, []),
    "v14d_gru":      ("gpu", model("v14_gru.py", "d", "--device", "cuda"), {}, []),
    "v14b_gru":      ("gpu", model("v14_gru.py", "b", "--device", "cuda"), {}, []),
    "v14a_gru":      ("gpu", model("v14_gru.py", "a", "--device", "cuda"), {}, []),
    "v13":           ("gpu", model("v13_rolling.py", "rows", "--device", "cuda"), {}, []),
    "v14e_mlp":      ("gpu", model("v14_mlp.py", "e", "--device", "cuda"), {}, []),
    "v14f_mlp":      ("gpu", model("v14_mlp.py", "f", "--device", "cuda"), {}, []),
    "v15a_V4":       ("cpu", model("v15_limit.py", "V4", "--threads", "2"), {}, ["V4_B_reforecast"]),
    "v15a_V5":       ("cpu", model("v15_limit.py", "V5", "--threads", "2"), {}, ["V5_border_inputs"]),
    "v15a_V8":       ("cpu", model("v15_limit.py", "V8", "--threads", "2"), {}, ["V8_error_mining"]),
    "v15d_V4_pairs": ("cpu", model("v15_pairs.py"), {}, ["V4_B_reforecast"]),
    "v15c_wx_border": ("cpu", model("v15_wxbord.py", "--threads", "2"), {}, ["v15a_V4", "v15a_V5"]),
    "strategies":    ("cpu", model("strategies_v2.py"), {}, ["gbm", "deep_c", "deep_v10"]),
    "v15b_V13":      ("cpu", model("v15_v13.py"), {}, ["v13", "gbm"]),
    "v14_positions": ("cpu", model("v14_positions.py"), {},
                      ["gbm", "deep_c", "v14a_gbm", "v14a_gru", "v14b_gru", "v14d_gru", "v14e_mlp", "v14f_mlp"]),
    "v12":           ("gpu", model("alloc_v12.py", "--device", "cuda", "--wait-ideas-hours", "0"),
                      {"OMP_NUM_THREADS": "1"},
                      ["strategies", "v14_positions", "v15b_V13", "v15c_wx_border", "v15d_V4_pairs", "v13",
                       "V5_border_inputs", "V6_NYC", "V6_LI", "V7_flags", "V7_recency", "V8_error_mining",
                       "V9_decompose", "V4_B_reforecast", "V16_B_gefs_joined"]),
    "comparisons":   ("cpu", model("strategies_v2.py", "--only", "comparisons"), {}, ["strategies"]),
}
SCORE = model("score_v2.py")


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def run_dir() -> Path:
    if R.MODE == "build":
        raise SystemExit("set US_RUN_MODE=dryrun or heldout (and US_RUN_DIR)")
    return R.RUN_DIR


def env() -> dict:
    e = dict(os.environ)
    e.update({"US_RUN_MODE": R.MODE, "US_RUN_DIR": str(R.RUN_DIR), "V2_RESULTS": str(R.RESULTS),
              "V2_PARQUET": str(R.PARQUET), "NYISO_V2_PARQUET": str(R.PARQUET), "V2_PANEL": str(R.PANEL),
              "V2_DAY": str(R.DAYFEATS), "V2_MODEL_DIR": str(HERE), "V2_IDEAS_DIR": str(IDEAS),
              "V1_MODEL_DIR": str(R.V1_MODEL), "OMP_NUM_THREADS": "2", "US_LGBM_THREADS": "2"})
    return e


# ============================================================================ seed
def first_window_quarter() -> str:
    return next(q[2] for q in R.quarters() if R.in_window(q))


def seed() -> dict:
    """Copy (never move) the build's per-quarter parts and fitted models of the quarters before the window, the
    build's positions (to build_pos/, read by V12 and the check only) and the data flags. Idempotent."""
    src = R.BUILD_RESULTS
    if src.resolve() == R.RESULTS.resolve():
        raise SystemExit("run results must not be the build results")
    q0 = first_window_quarter()
    y0 = int(q0[:4])
    n = {"parts": 0, "models": 0, "pos": 0}
    for d in sorted((src / "preds").iterdir()):
        if not d.is_dir():
            continue
        out = R.PREDS / d.name
        if d.name.endswith("_parts"):
            for f in sorted(d.glob("*.parquet")):
                if f.stem < q0:
                    n["parts"] += _copy(f, out / f.name)
        elif d.name.endswith("_models"):
            for f in sorted(d.iterdir()):
                st = f.stem
                key = st[4:] if st.startswith("pre_") else st.split("_")[-1]
                early = int(key) < y0 if st.startswith("pre_") else key < q0
                if early:
                    n["models"] += _copy(f, out / f.name)
    for f in sorted((src / "pos").glob("*")):
        if f.is_file():
            n["pos"] += _copy(f, R.RESULTS / "build_pos" / f.name)
    for fl in DATA_FLAGS:
        if (src.parent / fl).exists():
            _copy(src.parent / fl, R.RESULTS.parent / fl)
    rec = {"mode": R.MODE, "window": [str(x) for x in R.TEST_WINDOW], "first_quarter": q0, "copied": n,
           "source": str(src), "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (R.RESULTS / "seed.json").write_text(json.dumps(rec, indent=1))
    log(f"seeded from {src} (quarters before {q0}): {n}")
    return rec


def _copy(a: Path, b: Path) -> int:
    if b.exists() and b.stat().st_size == a.stat().st_size:
        return 0
    b.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a, b)
    return 1


# ============================================================================ run
class Runner:
    def __init__(self, only: set | None = None):
        self.dir = run_dir() / "steps"
        self.logs = run_dir() / "logs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.only = only
        self.lock = threading.Lock()
        self.failed: set = set()

    def ok(self, s) -> bool:
        f = self.dir / f"{s}.json"
        return f.exists() and json.loads(f.read_text()).get("exit") == 0

    def ready(self, s) -> bool:
        return all(self.ok(d) for d in STEPS[s][3])

    def blocked(self, s) -> bool:
        return any(d in self.failed or self.blocked(d) for d in STEPS[s][3])

    def gpu_wait(self):
        while True:
            try:
                free = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                          capture_output=True, text=True).stdout.split()[0])
            except Exception:
                return
            if free >= GPU_FREE_MB:
                return
            log(f"GPU: {free} MB free, waiting for {GPU_FREE_MB}")
            time.sleep(60)

    def run_step(self, s):
        lane, (cwd, argv), extra, _ = STEPS[s]
        if lane == "gpu":
            self.gpu_wait()
        e = env()
        e.update(extra)
        t0 = time.time()
        log(f"start {s}: {' '.join(argv)}")
        with open(self.logs / f"{s}.log", "a") as fh:
            fh.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv)}\n")
            fh.flush()
            rc = subprocess.run(argv, cwd=str(cwd), env=e, stdout=fh, stderr=subprocess.STDOUT).returncode
        rec = {"step": s, "lane": lane, "argv": argv, "exit": rc, "seconds": round(time.time() - t0, 1),
               "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
               "end": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (self.dir / f"{s}.json").write_text(json.dumps(rec, indent=1))
        log(f"end {s}: exit {rc}, {rec['seconds'] / 60:.1f} min")
        if rc != 0:
            with self.lock:
                self.failed.add(s)

    def lane(self, name):
        order = [s for s in STEPS if STEPS[s][0] == name and (self.only is None or s in self.only)]
        while True:
            left = [s for s in order if not self.ok(s) and s not in self.failed and not self.blocked(s)]
            if not left:
                return
            nxt = next((s for s in left if self.ready(s)), None)
            if nxt is None:
                time.sleep(30)
                continue
            self.run_step(nxt)

    def run(self):
        th = [threading.Thread(target=self.lane, args=(ln,)) for ln in ("cpu", "gpu")]
        for t in th:
            t.start()
        for t in th:
            t.join()
        if self.failed:
            log(f"FAILED steps: {sorted(self.failed)} (logs in {self.logs}); blocked: "
                f"{sorted(s for s in STEPS if self.blocked(s))}")
        return not self.failed


def score(quick: bool = False):
    cwd, argv = SCORE
    e = env()
    t0 = time.time()
    with open(run_dir() / "logs" / "score.log", "a") as fh:
        rc = subprocess.run(argv + (["--quick"] if quick else []), cwd=str(cwd), env=e, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    (run_dir() / "steps" / "score.json").write_text(json.dumps({"step": "score", "exit": rc,
                                                                "seconds": round(time.time() - t0, 1)}, indent=1))
    log(f"score: exit {rc}, {(time.time() - t0) / 60:.1f} min")
    return rc == 0


def status():
    d = run_dir() / "steps"
    tot = {"cpu": 0.0, "gpu": 0.0}
    for s in list(STEPS) + ["score"]:
        f = d / f"{s}.json"
        if f.exists():
            r = json.loads(f.read_text())
            tot[r.get("lane", "cpu")] = tot.get(r.get("lane", "cpu"), 0.0) + r["seconds"]
            print(f"{s:22s} exit {r['exit']:>3}  {r['seconds'] / 60:7.1f} min")
        else:
            print(f"{s:22s} not run")
    print({k: f"{v / 3600:.2f} h" for k, v in tot.items()})


# ============================================================================ dry-run check
def heldout_rows() -> list[str]:
    rows = []
    for line in ROWS_LIST.read_text().splitlines():
        if line.startswith("| V") or line.startswith("| C_"):
            rows.append(line.split("|")[1].strip().split(" ")[0])
    return rows


def check():
    """Dry run: 2023 positions of every row against the build, per zone-hour; scorer 2023 net against the build."""
    import numpy as np
    import pandas as pd
    if R.MODE != "dryrun":
        raise SystemExit("check is the dry-run proof (window 2023)")
    lo, hi = (pd.Timestamp(x, tz=R.TZ) for x in (R.TEST_WINDOW[0], R.TEST_WINDOW[1] + dt.timedelta(days=1)))
    bres = {r["name"]: r for r in json.loads((R.BUILD_RESULTS / "v2_results.json").read_text())["rows"]}
    dres = {r["name"]: r for r in json.loads((R.RESULTS / "v2_results_dryrun.json").read_text())["rows"]}
    bd = pd.read_parquet(R.BUILD_RESULTS / "v2_daily.parquet")
    dd = pd.read_parquet(R.RESULTS / "v2_daily_dryrun.parquet")

    def pos(path):
        p = R.read_pre2024(path, "delivery_hour")
        p["delivery_hour"] = p["delivery_hour"].dt.tz_convert(R.TZ)
        p = p[(p["delivery_hour"] >= lo) & (p["delivery_hour"] < hi)]
        return p.groupby(["delivery_hour", "zone"])["mw"].sum()

    out = []
    for n in heldout_rows():
        r = {"row": n}
        fb, fr = R.BUILD_RESULTS / "pos" / f"{n}.parquet", R.POS / f"{n}.parquet"
        if not fr.exists():
            r["status"] = "no dry-run positions"
            out.append(r)
            continue
        a, b = pos(fb), pos(fr)
        j = pd.concat([a.rename("build"), b.rename("dry")], axis=1).fillna(0.0)
        d = (j["dry"] - j["build"]).abs()
        r.update({"zone_hours": int(len(j)), "exact": int((d == 0).sum()), "not_exact": int((d != 0).sum()),
                  "max_abs_diff_mw": float(d.max()) if len(d) else 0.0,
                  "build_mwh": float(j["build"].abs().sum()), "dry_mwh": float(j["dry"].abs().sum())})
        yb = (bres.get(n) or {}).get("years", {}).get("2023")
        yd = (dres.get(n) or {}).get("years", {}).get("2023")
        r.update({"net_2023_build": yb, "net_2023_dry": yd,
                  "net_2023_diff_usd": None if yb is None or yd is None else yd - yb})
        if n in bd.columns and n in dd.columns:
            x = bd.loc[(bd.index >= pd.Timestamp(R.TEST_WINDOW[0])), n]
            y = dd[n].reindex(x.index)
            r["daily_max_abs_diff_usd"] = float((x - y).abs().max())
        r["status"] = "EXACT" if r["not_exact"] == 0 and r.get("net_2023_diff_usd") == 0 else "DIFFERS"
        out.append(r)
    f = R.RESULTS / "v12" / "positions_recomputed_before_window.parquet"
    v12 = None
    if f.exists():
        q = pd.read_parquet(f)
        q["delivery_hour"] = q["delivery_hour"].dt.tz_convert(R.TZ)
        q = q.groupby(["delivery_hour", "zone"])["mw"].sum()
        bb = R.read_pre2024(R.BUILD_RESULTS / "pos" / "V12_alloc.parquet", "delivery_hour")
        bb["delivery_hour"] = bb["delivery_hour"].dt.tz_convert(R.TZ)
        bb = bb.groupby(["delivery_hour", "zone"])["mw"].sum().reindex(q.index).fillna(0.0)
        dv = (q - bb).abs()
        v12 = {"what": "V12 positions of the 4 refit quarters before the window (recomputed only to choose kappa)",
               "zone_hours": int(len(dv)), "exact": int((dv == 0).sum()), "max_abs_diff_mw": float(dv.max())}
    rec = {"written": time.strftime("%Y-%m-%dT%H:%M:%S"), "rows": out, "v12_before_window": v12}
    (R.RESULTS / "dryrun_check.json").write_text(json.dumps(rec, indent=1))
    L = ["# Dry run 2023 against the build", "", "| row | zone-hours | exact | not exact | max abs diff MW | "
         "2023 net build | 2023 net dry run | diff USD | status |", "|---|---|---|---|---|---|---|---|---|"]
    for r in out:
        L.append(f"| {r['row']} | {r.get('zone_hours', '')} | {r.get('exact', '')} | {r.get('not_exact', '')} | "
                 f"{r.get('max_abs_diff_mw', '')} | {r.get('net_2023_build', '')} | {r.get('net_2023_dry', '')} | "
                 f"{r.get('net_2023_diff_usd', '')} | {r['status']} |")
    if v12:
        L += ["", f"V12, 4 quarters before the window recomputed: {v12}"]
    (R.RESULTS / "dryrun_check.md").write_text("\n".join(L) + "\n")
    log("\n".join(L))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["all", "seed", "run", "score", "check", "status"])
    ap.add_argument("--only", default="", help="comma list of steps (run)")
    ap.add_argument("--quick", action="store_true", help="score without fragility draws")
    a = ap.parse_args()
    run_dir()
    log(f"mode {R.MODE}, window {R.TEST_WINDOW}, run dir {R.RUN_DIR}, tables {R.PARQUET}, panel {R.PANEL.name}, "
        f"day {R.DAYFEATS.name}")
    if a.cmd in ("seed", "all"):
        seed()
    if a.cmd in ("run", "all"):
        ok = Runner(set(a.only.split(",")) if a.only else None).run()
        if not ok and a.cmd == "all":
            raise SystemExit("steps failed: not scored")
    if a.cmd in ("score", "all"):
        if not score(a.quick):
            raise SystemExit("score failed")
    if a.cmd == "check" or (a.cmd == "all" and R.MODE == "dryrun"):
        check()
    if a.cmd == "status":
        status()


if __name__ == "__main__":
    main()
