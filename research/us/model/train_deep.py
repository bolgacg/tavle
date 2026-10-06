"""Training runs for the deep model (owner: deep agent). Run on gene from ~/nyiso-us/model:

    ../.venv/bin/python train_deep.py tune         choose the configuration on 2020-2022 (below)
    ../.venv/bin/python train_deep.py rehearsal    build on 2020-2022, monthly refits through 2023
    ../.venv/bin/python train_deep.py smoke        the same code on small synthetic tables, CPU

Tuning (CONTRACT.md: at most 8 configurations, chosen on 2020 to 2022, time-ordered inner split).
Data and labels are loaded through 2022-12-31 only. Two folds, each one refit whose training rows follow
walkforward.train_mask (delivery dates up to two days before the refit, labels public by its first bid):
    fold 1: refit 2022-01-01, predict delivery dates 2022-01-01 .. 2022-06-30
    fold 2: refit 2022-07-01, predict delivery dates 2022-07-01 .. 2022-12-31
Each configuration is the average of 3 seeds. Selection: the lowest mean squared error against the gap
clipped to +-250 USD/MWh (the training target), pooled over both folds. Squared error because the trading
rules use the expected gap (gbm.py uses it for the same reason); clipped because on 2022 the unclipped
error is decided by a handful of spike hours that no predictor foresees (a first pass, kept in the log,
used the unclipped error). Unclipped MSE, MAE, RMSE without the 1% largest |gap| hours and the idea-A
rule's profit per day at each year's fee are reported beside it and not used.

Rehearsal (CONTRACT.md "Rehearsal"): the selected configuration through walkforward.walk_forward, refit on
the 1st of every month of 2023. Writes ~/nyiso-us/results/deep_rehearsal_2023.parquet (panel index as
the index and as column panel_index, plus the panel keys and pred_gap), deep_rehearsal_2023.json (refit
records with seconds per refit) and checkpoints in ~/nyiso-us/checkpoints/deep/rehearsal_2023/.

Panel: the labels of panel.build_panel (panel._labels on a LockedStore), sorted and indexed exactly as
build_panel sorts them, so the index matches a panel built for the same dates. The deep model takes its
inputs from deep_data.py, not from the panel's feature columns.
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

import numpy as np
import pandas as pd
import torch

import deep
import deep_data as DD
import fees
import lock
import panel as P
import walkforward as W
from common import HOME, PARQUET, RESULTS, TZ

LOGS = HOME / "logs"
CKPT = HOME / "checkpoints" / "deep"
BUILD_START = dt.date(2020, 1, 1)
TUNE_END = dt.date(2022, 12, 31)
FOLDS = [(dt.date(2022, 1, 1), dt.date(2022, 6, 30)), (dt.date(2022, 7, 1), dt.date(2022, 12, 31))]
REHEARSAL = (dt.date(2023, 1, 1), dt.date(2023, 12, 31))


class Log:
    def __init__(self, path: Path | None):
        self.f = open(path, "a", buffering=1) if path else None

    def __call__(self, msg: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        if self.f:
            self.f.write(line + "\n")


def label_panel(start: dt.date, end: dt.date, root: Path = PARQUET) -> pd.DataFrame:
    """Key and label columns of panel.build_panel for delivery dates start..end, same order and index."""
    lock.assert_build_only([start, end])
    store = P.LockedStore(root=root)
    lab = P._labels(store, start, end)
    lab["bid_date"] = lab["delivery_date"] - pd.Timedelta(days=1)
    lab["hour"] = lab["delivery_hour"].dt.tz_convert(TZ).dt.hour
    cols = list(dict.fromkeys(P.KEY_COLS + P.LABEL_COLS))
    out = lab[cols].sort_values(["delivery_hour", "zone"], kind="stable").reset_index(drop=True)
    lock.assert_build_only(out["delivery_hour"])
    return out


def metrics(rows: pd.DataFrame, pred: pd.Series) -> dict:
    """Scores of predictions on rows with a gap (build or development years only)."""
    lock.assert_build_only(rows["delivery_hour"])
    ok = rows["gap"].notna()
    g, p = rows.loc[ok, "gap"].to_numpy(float), pred.reindex(rows.index)[ok].to_numpy(float)
    e = p - g
    cut = np.quantile(np.abs(g), 0.99)
    calm = np.abs(g) <= cut
    fee = fees.fee_for(rows.loc[ok, "delivery_date"])
    a = np.where(p <= -fee, -g - fee, 0.0)                      # idea A rule: supply when pred <= -fee
    days = rows.loc[ok, "delivery_date"].to_numpy()
    a_day = pd.Series(a).groupby(days).sum()
    zero = np.mean(g ** 2)
    gc = np.clip(g, -250, 250)
    return {"rows": int(ok.sum()), "mse_clip": float(np.mean((p - gc) ** 2)), "mse_clip_zero_pred": float(np.mean(gc ** 2)),
            "mse": float(np.mean(e ** 2)), "mse_zero_pred": float(zero),
            "mae": float(np.mean(np.abs(e))), "rmse_calm99": float(np.sqrt(np.mean(e[calm] ** 2))),
            "a_pnl_per_day": float(a_day.mean()), "a_share_supply": float(np.mean(p <= -fee)),
            "mean_pred": float(p.mean()), "mean_gap": float(g.mean())}


def _gpu(log):
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        log(f"GPU {torch.cuda.get_device_name(0)}: {free / 2**20:.0f} MiB free of {total / 2**20:.0f}")


def tune(args, log, root: Path = PARQUET, tune_end: dt.date = TUNE_END, folds=FOLDS, configs=None,
         out_name: str = "deep_tune_2022", start: dt.date = BUILD_START, results: Path = RESULTS) -> dict:
    lock.assert_build_only([tune_end])
    configs = configs or list(deep.CONFIGS)
    assert len(configs) <= 8
    t0 = time.time()
    pan = label_panel(start, tune_end, root)
    log(f"label panel {len(pan)} rows, {pan['delivery_date'].min().date()}..{pan['delivery_date'].max().date()}")
    dd = deep.preload(tune_end, root=root, log=log, first_day=start)
    log(f"deep_data through {tune_end}: {dd.T} hours, {len(dd.ptids)} points, {time.time() - t0:.0f} s")
    _gpu(log)
    res = {"protocol": __doc__.split("Tuning")[1].split("Rehearsal")[0].strip(), "folds": [list(map(str, f)) for f in folds],
           "seeds": list(args.seeds), "configs": {}}
    out = results / f"{out_name}.json"
    for name in configs:
        cfg = {**deep.resolve_config(name), **args.override}
        rec = {"config": cfg, "folds": []}
        preds = []
        for (a, b) in folds:
            tm = W.train_mask(pan, a)
            test = pan[(pan["delivery_date"] >= pd.Timestamp(a)) & (pan["delivery_date"] <= pd.Timestamp(b))]
            ts = time.time()
            m = deep.DeepModel(cfg, seeds=args.seeds, data=dd, device=args.device, log=log)
            m.fit(pan[tm])
            p = m.predict(test)
            preds.append(p)
            fr = {"refit": str(a), "train_last": m.info["train_last"], "train_days": m.info["n_days"],
                  "seconds": round(time.time() - ts, 1), "seeds": m.info["seeds"], **metrics(test, p)}
            rec["folds"].append(fr)
            log(f"tune {name} fold {a}: mse_clip {fr['mse_clip']:.2f} (zero {fr['mse_clip_zero_pred']:.2f}) "
                f"mse {fr['mse']:.1f} (zero {fr['mse_zero_pred']:.1f}) mae {fr['mae']:.2f} "
                f"A/day {fr['a_pnl_per_day']:.1f} {fr['seconds']} s")
        allp = pd.concat(preds)
        rec["pooled"] = metrics(pan.loc[allp.index], allp)
        res["configs"][name] = rec
        out.write_text(json.dumps(res, indent=1, default=str))
    best = min(res["configs"], key=lambda k: res["configs"][k]["pooled"]["mse_clip"])
    res["selected"] = best
    res["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(res, indent=1, default=str))
    log(f"selected {best}: " + json.dumps({k: round(v["pooled"]["mse_clip"], 3) for k, v in res["configs"].items()}))
    return res


def rehearsal(args, log, root: Path = PARQUET, span=REHEARSAL, out_name: str = "deep_rehearsal_2023",
              ckpt_dir: Path = CKPT / "rehearsal_2023", start: dt.date = BUILD_START,
              results: Path = RESULTS) -> dict:
    first, last = span
    lock.assert_build_only([first, last])
    name = args.config
    if name == "auto":
        tj = results / "deep_tune_2022.json"
        name = json.loads(tj.read_text())["selected"]
    cfg = {**deep.resolve_config(name), **args.override}
    t0 = time.time()
    pan = label_panel(start, last, root)
    dd = deep.preload(last, root=root, log=log, first_day=start)
    log(f"rehearsal {first}..{last} config {name} {deep.describe(cfg)}; data {time.time() - t0:.0f} s")
    _gpu(log)
    months = iter(W.month_starts(first, last))
    models = []

    def make():
        ms = next(months)
        m = deep.DeepModel(cfg, seeds=args.seeds, data=dd, device=args.device, log=log,
                           ckpt_path=ckpt_dir / f"{ms:%Y%m}.pt")
        models.append((ms, m))
        return m

    pred, fits = W.walk_forward(pan, make, first, last, log=log)
    for rec, (ms, m) in zip(fits, models):
        assert rec["month"] == ms.isoformat()
        rec.update({"fit_s": m.info["fit_s"], "train_days": m.info["n_days"], "n_points": m.info["n_points"],
                    "seeds": m.info["seeds"]})
    rows = pan.loc[pred.index]
    out = rows[P.KEY_COLS].copy()
    out["panel_index"] = out.index
    out["pred_gap"] = pred
    out["refit_month"] = rows["delivery_date"].dt.strftime("%Y-%m-01")
    out["model"] = f"deep_{name}"
    results.mkdir(parents=True, exist_ok=True)
    out.to_parquet(results / f"{out_name}.parquet")
    secs = [r["seconds"] for r in fits]
    summary = {"config_name": name, "config": cfg, "seeds": list(args.seeds), "span": [str(first), str(last)],
               "refits": fits, "seconds_per_refit_mean": float(np.mean(secs)), "seconds_per_refit_max": float(np.max(secs)),
               "total_seconds": round(time.time() - t0, 1), "rows": int(len(out)),
               "metrics_development_year": metrics(rows, pred),
               "gpu_max_mem_mib": (torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None)}
    (results / f"{out_name}.json").write_text(json.dumps(summary, indent=1, default=str))
    log(f"rehearsal done: {len(out)} rows, {np.mean(secs):.0f} s per refit, total {summary['total_seconds']:.0f} s")
    return summary


def smoke(args, log):
    """End to end on synthetic tables (no real data): tune two configs on one fold, rehearse two months."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        start = dt.date(2022, 9, 1)
        DD.write_synthetic_parquet(root, start, dt.date(2023, 2, 28), n_points=10, seed=3)
        res = root / "results"
        res.mkdir()
        args.override = {**dict(hidden=16, K=2, max_epochs=3, patience=2, min_epochs=1, val_days=10, batch=16,
                                min_point_days=5), **args.override}
        r = tune(args, log, root=root, tune_end=dt.date(2023, 1, 31), folds=[(dt.date(2023, 1, 1), dt.date(2023, 1, 31))],
                 configs=["c1", "c5"], start=start, results=res)
        args.config = r["selected"]
        s = rehearsal(args, log, root=root, span=(dt.date(2023, 1, 1), dt.date(2023, 2, 28)), ckpt_dir=root / "ckpt",
                      start=start, results=res)
        df = pd.read_parquet(res / "deep_rehearsal_2023.parquet")
        assert len(df) == s["rows"] and df["pred_gap"].notna().all()
        assert (df["panel_index"].to_numpy() == df.index.to_numpy()).all()
        assert len(list((root / "ckpt").glob("*.pt"))) == 2
        log(f"smoke ok: {len(df)} rows; seconds per refit {s['seconds_per_refit_mean']:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "rehearsal", "smoke"])
    ap.add_argument("--config", default="auto")
    ap.add_argument("--configs", default=None, help="comma list for tune (default all)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--override", default="{}", help="JSON dict merged into the config")
    args = ap.parse_args()
    args.seeds = tuple(int(x) for x in args.seeds.split(","))
    args.override = json.loads(args.override)
    LOGS.mkdir(parents=True, exist_ok=True) if args.mode != "smoke" else None
    log = Log(LOGS / f"deep_{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}.log" if args.mode != "smoke" else None)
    log(f"train_deep {args.mode} pid {os.getpid()} torch {torch.__version__} cuda {torch.cuda.is_available()}")
    if args.mode == "tune":
        tune(args, log, configs=args.configs.split(",") if args.configs else None)
    elif args.mode == "rehearsal":
        rehearsal(args, log)
    else:
        smoke(args, log)


if __name__ == "__main__":
    main()
