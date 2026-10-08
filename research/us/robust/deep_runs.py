"""GPU runs for the robustness checks (OBJECTIVES.md addendum with idea 13, commit b31b516). Run on gene from
~/nyiso-us with the venv:

    .venv/bin/python robust/deep_runs.py seeds --kind base       C deep's model, 20 seeds in one process
    .venv/bin/python robust/deep_runs.py seeds --kind weather    B deep's model, 20 seeds in one process
    .venv/bin/python robust/deep_runs.py neighbours --kind base --variant dropout_0.2   (one step either side
                                                                 on dropout and weight decay, 3 seeds as registered)
    .venv/bin/python robust/deep_runs.py smoke                   the same code on small synthetic tables

What runs is the deep agent's own walk-forward, model/train_deep.py wf (monthly refits 2021-01 to 2023-12,
training up to two days before each month, configuration c2, idea B from 25 March 2021 and scored from 1 May
2021), called by import with its results and checkpoints pointed at results/robust/. The only addition is a
recorder on deep.DeepModel.predict that also keeps each seed's own prediction (the network outputs one seed at
a time, the same arithmetic as the ensemble); the ensemble that wf writes is unchanged and is checked against
the mean of the per-seed predictions at every refit.

Seeds share one process and one copy of the data on the GPU (deep_data keeps its arrays on the device); within
a refit they train one after another, as DeepModel.fit does for the registered three. Vectorising over seeds
would need a different network class (cuDNN GRUs have no batching rule under torch.func.vmap) and early
stopping differs per seed, so the loop is kept.

Holdout: train_deep.wf and every loader it uses call model/lock.py; nothing on or after 2024-01-01 is read.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rb  # noqa: E402  (puts pipeline/, model/, side/ on the path)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import deep  # noqa: E402
import train_deep as TD  # noqa: E402

CONFIG = "c2"                                    # the deep agent's configuration, fixed for every year
N_SEEDS = 20
VARIANTS = {"dropout_0.2": {"dropout": 0.2}, "dropout_0.4": {"dropout": 0.4},
            "weight_decay_1e-3": {"weight_decay": 1e-3}, "weight_decay_1e-1": {"weight_decay": 1e-1}}
REGISTERED_SEEDS = (0, 1, 2)
_ORIG_PREDICT = deep.DeepModel.predict


def install_recorder(sink: list, partial: Path | None):
    """deep.DeepModel.predict, unchanged in what it returns, also appends each seed's prediction to sink."""
    def predict(self, panel):
        nets = self.nets
        per = []
        try:
            for n in nets:
                self.nets = [n]
                per.append(_ORIG_PREDICT(self, panel).to_numpy(float))
        finally:
            self.nets = nets
        ens = _ORIG_PREDICT(self, panel)
        per = np.vstack(per).T
        gap = np.abs(per.mean(1) - ens.to_numpy(float))
        assert gap.max() <= 1e-3 + 1e-5 * np.abs(ens.to_numpy(float)).max(), f"seed mean differs by {gap.max()}"
        f = panel[["delivery_hour", "zone", "delivery_date"]].copy()
        for i, s in enumerate(self.seeds):
            f[f"seed_{s:02d}"] = per[:, i]
        f["ensemble"] = ens.to_numpy(float)
        sink.append(f)
        if partial is not None:
            pd.concat(sink).to_parquet(partial)
        return ens
    deep.DeepModel.predict = predict


def run_wf(kind: str, seeds, override: dict, tag: str, device: str, out: Path, span=None, start=None, root=None):
    """train_deep.wf for one kind and seed set, outputs in out/deep/<tag>/; returns its summary."""
    res = out / "deep" / tag
    res.mkdir(parents=True, exist_ok=True)
    done = res / "done.json"
    if done.exists():
        rb.log(f"{tag}: already done ({done}), skipped")
        return json.loads(done.read_text())
    args = argparse.Namespace(kind=kind, config=CONFIG, seeds=tuple(seeds), device=device, override=dict(override))
    log = TD.Log(res / "train.log")
    sink: list = []
    install_recorder(sink, res / "per_seed_partial.parquet")
    kw = {"results": res, "ckpt_dir": out / "ckpt" / tag}
    if span is not None:
        kw["span"] = span
    if start is not None:
        kw["start"] = start
    if root is not None:
        kw["root"] = root
    t0 = time.time()
    summ = TD.wf(args, log, **kw)
    deep.DeepModel.predict = _ORIG_PREDICT
    per = pd.concat(sink, ignore_index=True)
    assert len(per) == summ["rows"], (len(per), summ["rows"])
    assert per["delivery_hour"].max() < rb.END
    per.to_parquet(res / "per_seed.parquet")
    (res / "per_seed_partial.parquet").unlink(missing_ok=True)
    rec = {"tag": tag, "kind": kind, "config": CONFIG, "override": override, "seeds": list(seeds), "rows": summ["rows"],
           "refits": len(summ["refits"]), "seconds": round(time.time() - t0), "device": device,
           "ensemble_file": str(res / f"deep_wf_2021_2023_{TD.WF_NAME[kind]}.parquet"),
           "per_seed_file": str(res / "per_seed.parquet"), "metrics_by_year": summ.get("metrics_by_year"),
           "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    rb.write_json(done, rec)
    rb.log(f"{tag}: {summ['rows']} rows, {len(summ['refits'])} refits, {rec['seconds']} s")
    return rec


def seeds(kind: str, device: str, out: Path = rb.OUT, n: int = N_SEEDS):
    return run_wf(kind, range(n), {}, f"seeds_{kind}", device, out)


def neighbours(kind: str, variant: str, device: str, out: Path = rb.OUT):
    return run_wf(kind, REGISTERED_SEEDS, VARIANTS[variant], f"nb_{kind}_{variant}", device, out)


def smoke(device: str):
    """Synthetic tables (deep_data.write_synthetic_parquet, as train_deep.py smoke), a tiny network, two months."""
    import deep_data as DD
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        start = dt.date(2022, 9, 1)
        DD.write_synthetic_parquet(root, start, dt.date(2023, 2, 28), n_points=10, seed=3)
        tiny = dict(hidden=16, K=2, max_epochs=3, patience=2, min_epochs=1, val_days=10, batch=16, min_point_days=5)
        span = (dt.date(2023, 1, 1), dt.date(2023, 2, 28))
        out = root / "robust"
        r1 = run_wf("base", range(4), tiny, "smoke_seeds_base", device, out, span=span, start=start, root=root)
        r2 = run_wf("base", REGISTERED_SEEDS, {**tiny, **VARIANTS["weight_decay_1e-1"]}, "smoke_nb_base", device, out,
                    span=span, start=start, root=root)
        p = pd.read_parquet(r1["per_seed_file"])
        e = pd.read_parquet(r1["ensemble_file"])
        assert [c for c in p.columns if c.startswith("seed_")] == [f"seed_{i:02d}" for i in range(4)]
        m = e.merge(p, on=["delivery_hour", "zone"], suffixes=("", "_rec"))
        assert len(m) == len(e) and np.allclose(m["pred_gap"], m["ensemble"], atol=1e-4)
        assert np.allclose(m[[f"seed_{i:02d}" for i in range(4)]].mean(axis=1), m["pred_gap"], atol=1e-3)
        assert m[[f"seed_{i:02d}" for i in range(4)]].std(axis=1).max() > 0, "seeds identical"
        assert pd.read_parquet(r2["per_seed_file"]).shape[0] == r2["rows"]
        assert run_wf("base", range(4), tiny, "smoke_seeds_base", device, out)["rows"] == r1["rows"]   # skip when done
        rb.log(f"smoke ok on {device}: {r1['rows']} rows, {r1['refits']} refits, per-seed and ensemble agree")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["seeds", "neighbours", "smoke"])
    ap.add_argument("--kind", choices=["base", "weather"], default="base")
    ap.add_argument("--variant", choices=list(VARIANTS), default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("ROBUST_TORCH_THREADS", "1")))
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    rb.log(f"deep_runs {a.mode} kind {a.kind} variant {a.variant} device {a.device} pid {os.getpid()}")
    if a.mode == "smoke":
        smoke(a.device)
    elif a.mode == "seeds":
        seeds(a.kind, a.device, n=a.n_seeds)
    else:
        neighbours(a.kind, a.variant, a.device)


if __name__ == "__main__":
    main()
