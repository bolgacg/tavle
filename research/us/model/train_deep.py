"""Training runs for the deep model (owner: deep agent). Run on gene from ~/nyiso-us/model:

    ../.venv/bin/python train_deep.py tune         choose the configuration on 2020-2022 (below)
    ../.venv/bin/python train_deep.py rehearsal    build on 2020-2022, monthly refits through 2023
    ../.venv/bin/python train_deep.py spike        idea A's spike model: 2022 predictions and 2023 walk-forward
    ../.venv/bin/python train_deep.py wf --kind base|weather|outages|spike   walk-forward 2021-01 .. 2023-12
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

Spike model (OBJECTIVES addendum, 6 Oct late evening; deep.DeepSpikeModel): the selected configuration's
backbone with a second output, P(spike = gap >= S), S from model/spike_config.json (50 if absent, logged).
  (a) one refit on 2022-01-01 (training rows by walkforward.train_mask: delivery dates to 2021-12-30),
      data loaded through 2022-12-31, P(spike) for every 2022 row -> deep_spike_2022.parquet (p* is
      chosen on these);
  (b) monthly refits through 2023 with walkforward.walk_forward -> deep_spike_rehearsal_2023.parquet.
Both files hold the panel index (as the index and as panel_index), the keys, p_spike and the gap output
pred_gap; a JSON beside each gives the threshold, refit records and the AUC of p_spike.

Walk-forward 2021 to 2023 (coordinator, 6 Oct night; evaluate.py rules): monthly refits from January 2021
to December 2023 through walkforward.walk_forward, training on delivery dates from January 2020 (idea B:
from 25 March 2021, scored from 1 May 2021) up to two days before each month. Configuration c2 is kept
fixed for every year (it was chosen on 2022 folds, so for 2021 and 2022 the configuration choice is not
out of sample; the 8 configurations differed by less than 1.5% there). Kinds:
  base     regression, no extra columns                     -> deep_wf_2021_2023_base.parquet
  weather  regression + idea B's x_ columns (rehearsal.load_all)            -> ..._B.parquet
  outages  regression + idea D's columns, the 20 outage sites for year Y counted on snapshots up to
           30 December of Y-1 (as evaluate.py), built in cache/deep/   -> ..._D.parquet
  spike    DeepSpikeModel, S from model/spike_config.json               -> ..._spike.parquet
Each file holds the panel index (index and panel_index), keys, pred_gap or p_spike, refit_month, model;
per-year copies results/deep_wf_<kind>_<year>.parquet are what evaluate.py reads.

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


def _spike_summary(rows: pd.DataFrame, both: pd.DataFrame, thr: float) -> dict:
    lock.assert_build_only(rows["delivery_hour"])
    ok = rows["gap"].notna()
    y = (rows.loc[ok, "gap"] >= thr).to_numpy(float)
    p = both.loc[ok.index[ok], "p_spike"].to_numpy(float)
    return {"rows": int(ok.sum()), "spike_rate": float(y.mean()), "auc": deep.auc(p, y),
            "mean_p": float(p.mean()), "p_quantiles": {q: float(np.quantile(p, q)) for q in (0.5, 0.9, 0.99)}}


def _spike_frame(rows, both, model_name, thr, refit) -> pd.DataFrame:
    """p_spike is calibrated (exact inverse of the refit's class weight pos_weight); p_spike_weighted is
    the weighted output it came from."""
    out = rows[P.KEY_COLS].copy()
    out["panel_index"] = out.index
    for c in ("p_spike", "p_spike_weighted", "pred_gap", "pos_weight"):
        out[c] = both[c]
    out["spike_threshold"] = thr
    out["refit_month"] = refit
    out["model"] = model_name
    return out


def spike(args, log, root: Path = PARQUET, start: dt.date = BUILD_START, results: Path = RESULTS,
          ckpt_dir: Path = CKPT / "spike", span22=(dt.date(2022, 1, 1), dt.date(2022, 12, 31)), span23=REHEARSAL,
          names=("deep_spike_2022", "deep_spike_rehearsal_2023"), threshold: float | None = None) -> dict:
    lock.assert_build_only([span22[1], span23[1]])
    name = args.config
    if name == "auto":
        name = json.loads((results / "deep_tune_2022.json").read_text())["selected"]
    cfg = {**deep.resolve_config(name), **args.override}
    thr, src = (threshold, "argument") if threshold is not None else deep.spike_threshold()
    log(f"spike model: config {name} {deep.describe(cfg)}; threshold S = {thr:g} ({src})")
    results.mkdir(parents=True, exist_ok=True)
    out = {}
    # (a) 2022, one refit on 2022-01-01, data through 2022-12-31 only
    t0 = time.time()
    pan = label_panel(start, span22[1], root)
    dd = deep.preload(span22[1], root=root, log=log, first_day=start)
    _gpu(log)
    ms = span22[0]
    tm = W.train_mask(pan, ms)
    test = pan[(pan["delivery_date"] >= pd.Timestamp(span22[0])) & (pan["delivery_date"] <= pd.Timestamp(span22[1]))]
    m = deep.DeepSpikeModel(cfg, threshold=thr, seeds=args.seeds, data=dd, device=args.device, log=log,
                            ckpt_path=ckpt_dir / f"{span22[0].year}.pt")
    ts = time.time()
    m.fit(pan[tm])
    both = m.predict_both(test).assign(pos_weight=m.info["pos_weight"])
    sec = round(time.time() - ts, 1)
    f22 = _spike_frame(test, both, m.name, thr, str(ms))
    f22.to_parquet(results / f"{names[0]}.parquet")
    s22 = {"config_name": name, "config": cfg, "threshold": thr, "threshold_source": src, "seeds": list(args.seeds),
           "refit": str(ms), "train_last": m.info["train_last"], "train_days": m.info["n_days"], "seconds": sec,
           "fit": {k: m.info[k] for k in ("train_spike_rate", "pos_weight", "seeds", "n_points")},
           "scores_2022": _spike_summary(test, both, thr), "total_seconds": round(time.time() - t0, 1)}
    (results / f"{names[0]}.json").write_text(json.dumps(s22, indent=1, default=str))
    log(f"spike 2022: {len(f22)} rows, AUC {s22['scores_2022']['auc']:.3f}, spike rate "
        f"{s22['scores_2022']['spike_rate']:.4f}, {sec} s")
    out["2022"] = s22
    # (b) 2023 walk-forward, monthly refits, data through 2023-12-31
    t0 = time.time()
    pan = label_panel(start, span23[1], root)
    dd = deep.preload(span23[1], root=root, log=log, first_day=start)
    months = iter(W.month_starts(*span23))
    models = []

    def make():
        mo = next(months)
        mm = deep.DeepSpikeModel(cfg, threshold=thr, seeds=args.seeds, data=dd, device=args.device, log=log,
                                 ckpt_path=ckpt_dir / f"{mo:%Y%m}.pt")
        models.append((mo, mm))
        return mm

    pred, fits = W.walk_forward(pan, make, *span23, log=log)
    lastb = pd.concat([mm.last_both for _, mm in models]).sort_index()
    for rec, (mo, mm) in zip(fits, models):
        assert rec["month"] == mo.isoformat()
        rec.update({"fit_s": mm.info["fit_s"], "train_days": mm.info["n_days"], "seeds": mm.info["seeds"],
                    "train_spike_rate": mm.info["train_spike_rate"], "pos_weight": mm.info["pos_weight"]})
    rows = pan.loc[pred.index]
    both = lastb.reindex(pred.index)
    assert np.allclose(both["p_spike"], pred)
    f23 = _spike_frame(rows, both, models[0][1].name, thr, rows["delivery_date"].dt.strftime("%Y-%m-01"))
    f23.to_parquet(results / f"{names[1]}.parquet")
    secs = [r["seconds"] for r in fits]
    s23 = {"config_name": name, "config": cfg, "threshold": thr, "threshold_source": src, "seeds": list(args.seeds),
           "refits": fits, "seconds_per_refit_mean": float(np.mean(secs)), "seconds_per_refit_max": float(np.max(secs)),
           "scores_2023": _spike_summary(rows, both, thr), "total_seconds": round(time.time() - t0, 1),
           "gpu_max_mem_mib": (torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None)}
    (results / f"{names[1]}.json").write_text(json.dumps(s23, indent=1, default=str))
    log(f"spike rehearsal done: {len(f23)} rows, AUC {s23['scores_2023']['auc']:.3f}, "
        f"{np.mean(secs):.0f} s per refit")
    out["2023"] = s23
    return out


WF_SPAN = (dt.date(2021, 1, 1), dt.date(2023, 12, 31))
WF_NAME = {"base": "base", "weather": "B", "outages": "D", "spike": "spike"}
B_TRAIN, B_FIRST = dt.date(2021, 3, 25), dt.date(2021, 5, 1)


def _lead_panel(log):
    """The lead modeller's panel with B's and D's columns (rehearsal.load_all, cached features only)."""
    import rehearsal as R
    store = P.LockedStore()
    panel, cols = R.load_all(store)
    return panel, cols, store, R


def p_star_grid() -> tuple:
    try:
        import gbm
        return tuple(gbm.P_STAR_GRID)
    except ImportError:                                   # no lightgbm here: the lead's declared grid
        return tuple(json.loads(deep.SPIKE_CONFIG.read_text())["p_star_grid"])


def choose_p_star(out: pd.DataFrame, panel: pd.DataFrame, results: Path, grid: tuple | None = None) -> dict:
    """p* for year Y (deep spike, idea A) = the cut in gbm.P_STAR_GRID with the highest mean daily net
    P&L of idea A on year Y-1's walk-forward predictions (strategies.idea_A, costs by fees.cost_for, the
    mean over Y-1's days with a settled gap, as evaluate.mean_net). Written to deep_wf_spike_choice.json
    as {"<Y>": p*}; the first year has no earlier walk-forward year and gets no entry."""
    import score as SC
    import strategies as S
    grid = grid or p_star_grid()
    years = sorted(out["delivery_date"].dt.year.unique())
    choice, scores = {}, {}
    for y0, y in zip(years[:-1], years[1:]):
        part = out[out["delivery_date"].dt.year == y0]
        rws = panel.loc[part.index]
        days = pd.DatetimeIndex(sorted(rws.loc[rws["gap"].notna(), "delivery_date"].unique()))
        sc = {}
        for ps in grid:
            led = S.idea_A(rws, part["p_spike"], ps)
            sc[str(ps)] = float(SC.daily(led, days).mean())
        best = max(sc, key=sc.get)
        choice[str(int(y))], scores[str(int(y))] = float(best), {"chosen_on": int(y0), "mean_daily_net": sc}
    res = {**choice, "how": "gbm.P_STAR_GRID cut with the best idea-A mean daily net P&L on the year before "
                            "(deep walk-forward predictions, calibrated p_spike)", "scores": scores,
           "S": float(out["spike_threshold"].iloc[0]), "grid": list(grid)}
    (results / "deep_wf_spike_choice.json").write_text(json.dumps(res, indent=1))
    return res


def wf(args, log, span=WF_SPAN, start: dt.date = BUILD_START, results: Path = RESULTS,
       ckpt_dir: Path = CKPT / "wf_2021_2023", root: Path = PARQUET, grid: tuple | None = None) -> dict:
    kind = args.kind
    lock.assert_build_only(list(span))
    name = args.config
    if name == "auto":
        name = json.loads((results / "deep_tune_2022.json").read_text())["selected"]
    cfg = {**deep.resolve_config(name), **args.override}
    t0 = time.time()
    lab = label_panel(start, span[1], root)
    extra_by_year, train_start, thr, src = {}, None, None, None
    if kind in ("base", "spike"):
        panel = lab
        for y in range(span[0].year, span[1].year + 1):
            extra_by_year[y] = []
    else:
        panel, cols, store, R = _lead_panel(log)
        panel = panel[panel["delivery_date"] <= pd.Timestamp(span[1])].reset_index(drop=True)
        assert len(panel) == len(lab) and (panel["delivery_hour"].to_numpy() == lab["delivery_hour"].to_numpy()).all() \
            and (panel["zone"].to_numpy() == lab["zone"].to_numpy()).all(), "lead panel order differs from build_panel order"
        if kind == "weather":
            ex = [c for c in cols["weather"] if c.startswith("x_")]
            for y in range(span[0].year, span[1].year + 1):
                extra_by_year[y] = ex
            train_start = B_TRAIN
        else:
            import strategies as S
            cdir = HOME / "cache" / "deep"
            cdir.mkdir(parents=True, exist_ok=True)
            for y in range(span[0].year, span[1].year + 1):
                cut = dt.date(y - 1, 12, 30)
                f = cdir / f"feat_outages_sites{cut:%Y%m%d}_prefixed.parquet"
                feat = P.read_locked(f, "delivery_hour", lock.read_end()) if f.exists() else None
                if feat is not None and not (len(feat) == len(panel)
                                             and (feat["delivery_hour"].to_numpy() == panel["delivery_hour"].to_numpy()).all()
                                             and (feat["zone"].to_numpy() == panel["zone"].to_numpy()).all()):
                    log(f"!! cached {f.name} does not cover this panel ({len(feat)} rows, panel {len(panel)}): rebuilt")
                    feat = None
                if feat is None:
                    ts = time.time()
                    feat = S.outage_features(panel, store, sites_through=cut).add_prefix(f"o{cut:%Y%m%d}_")
                    feat[["delivery_hour", "zone"]] = panel[["delivery_hour", "zone"]]
                    P.save_panel(feat, f)
                    log(f"outage features, sites through {cut}: {feat.shape[1] - 2} columns, {time.time() - ts:.0f} s")
                new = [c for c in feat.columns if c not in ("delivery_hour", "zone")]
                panel = panel.merge(feat, on=["delivery_hour", "zone"], how="left", validate="one_to_one")
                extra_by_year[y] = new
            panel.index = lab.index
    if kind == "spike":
        thr, src = deep.spike_threshold()
    dd = deep.preload(span[1], root=root, log=log, first_day=start)
    log(f"wf {kind} {span[0]}..{span[1]}: config {name} fixed {deep.describe(cfg)}"
        + (f"; S = {thr:g} ({src})" if kind == "spike" else "") + f"; setup {time.time() - t0:.0f} s")
    _gpu(log)
    preds, gaps, fits = [], [], []
    for y in range(span[0].year, span[1].year + 1):
        lo, hi = max(span[0], dt.date(y, 1, 1)), min(span[1], dt.date(y, 12, 31))
        if kind == "weather":
            lo = max(lo, B_FIRST)
        months = iter(W.month_starts(lo, hi))
        models = []
        ex = extra_by_year[y]

        def make(ex=ex, months=months, models=models):
            mo = next(months)
            path = ckpt_dir / kind / f"{mo:%Y%m}.pt"
            if kind == "spike":
                m = deep.DeepSpikeModel(cfg, threshold=thr, seeds=args.seeds, data=dd, device=args.device, log=log,
                                        ckpt_path=path)
            else:
                m = deep.DeepModel(cfg, seeds=args.seeds, data=dd, device=args.device, log=log, extra_cols=ex,
                                   ckpt_path=path)
            models.append((mo, m))
            return m

        p, f = W.walk_forward(panel, make, lo, hi, train_start=train_start, log=log)
        for rec, (mo, m) in zip(f, models):
            assert rec["month"] == mo.isoformat()
            rec.update({"fit_s": m.info["fit_s"], "train_days": m.info["n_days"], "n_extra": len(ex),
                        "best_epochs": [x["best_epoch"] for x in m.info["seeds"]]})
        preds.append(p)
        fits += f
        if kind == "spike":
            gaps.append(pd.concat([m.last_both for _, m in models]))
            for rec, (mo, m) in zip(f, models):
                rec.update({"pos_weight": m.info["pos_weight"], "train_spike_rate": m.info["train_spike_rate"]})
    pred = pd.concat(preds).sort_index()
    rows = panel.loc[pred.index]
    out = rows[P.KEY_COLS].copy()
    out["panel_index"] = out.index
    col = "p_spike" if kind == "spike" else "pred_gap"
    out[col] = pred
    if kind == "spike":
        lastb = pd.concat(gaps).reindex(pred.index)
        assert np.allclose(lastb["p_spike"], pred)
        for c in ("p_spike_weighted", "pred_gap", "pos_weight"):
            out[c] = lastb[c]
        out["spike_threshold"] = thr
    out["refit_month"] = rows["delivery_date"].dt.strftime("%Y-%m-01")
    out["model"] = f"deep_{'spike_' if kind == 'spike' else ''}{name}"
    results.mkdir(parents=True, exist_ok=True)
    out.to_parquet(results / f"deep_wf_2021_2023_{WF_NAME[kind]}.parquet")
    for y, part in out.groupby(out["delivery_date"].dt.year):
        part.to_parquet(results / f"deep_wf_{kind}_{y}.parquet")
    secs = [r["seconds"] for r in fits]
    summ = {"kind": kind, "config_name": name, "config": cfg, "config_note": "c2 kept fixed for every year",
            "seeds": list(args.seeds), "span": [str(span[0]), str(span[1])], "train_start": str(train_start or start),
            "threshold": thr, "threshold_source": src, "refits": fits, "rows": int(len(out)),
            "seconds_per_refit_mean": float(np.mean(secs)), "seconds_total": round(time.time() - t0, 1)}
    if kind == "spike":
        summ["auc_by_year"] = {}
        for y, part in out.groupby(out["delivery_date"].dt.year):
            g = panel.loc[part.index, "gap"]
            ok = g.notna().to_numpy()
            summ["auc_by_year"][int(y)] = deep.auc(part["p_spike"].to_numpy()[ok], (g[ok] >= thr).to_numpy(float))
    else:
        summ["metrics_by_year"] = {int(y): metrics(panel.loc[part.index], part["pred_gap"])
                                   for y, part in out.groupby(out["delivery_date"].dt.year)}
    if kind == "spike":
        summ["p_star_choice"] = choose_p_star(out, panel, results, grid)
    (results / f"deep_wf_2021_2023_{WF_NAME[kind]}.json").write_text(json.dumps(summ, indent=1, default=str))
    log(f"wf {kind} done: {len(out)} rows, {len(fits)} refits, {np.mean(secs):.0f} s per refit, "
        f"total {summ['seconds_total']:.0f} s")
    return summ


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
        sp = spike(args, log, root=root, start=start, results=res, ckpt_dir=root / "ckpt_spike",
                   span22=(dt.date(2022, 12, 1), dt.date(2022, 12, 31)), span23=(dt.date(2023, 1, 1), dt.date(2023, 2, 28)),
                   threshold=10.0)
        for nm in ("deep_spike_2022", "deep_spike_rehearsal_2023"):
            f = pd.read_parquet(res / f"{nm}.parquet")
            assert f["p_spike"].between(0, 1).all() and (f["panel_index"].to_numpy() == f.index.to_numpy()).all()
        for kind in ("base", "spike"):
            args.kind = kind
            w = wf(args, log, span=(dt.date(2022, 12, 1), dt.date(2023, 2, 28)), start=start, results=res,
                   ckpt_dir=root / "ckpt_wf", root=root, grid=(0.02, 0.05))
            f = pd.read_parquet(res / f"deep_wf_2021_2023_{WF_NAME[kind]}.parquet")
            assert len(f) == w["rows"] and (f["panel_index"].to_numpy() == f.index.to_numpy()).all()
            assert {p.name for p in res.glob(f"deep_wf_{kind}_20*.parquet")} == {f"deep_wf_{kind}_2022.parquet",
                                                                               f"deep_wf_{kind}_2023.parquet"}
        log(f"smoke ok: {len(df)} rows; seconds per refit {s['seconds_per_refit_mean']:.1f}; spike AUC 2023 "
            f"{sp['2023']['scores_2023']['auc']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tune", "rehearsal", "spike", "wf", "smoke"])
    ap.add_argument("--kind", default="base", choices=list(WF_NAME))
    ap.add_argument("--config", default="auto")
    ap.add_argument("--configs", default=None, help="comma list for tune (default all)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--override", default="{}", help="JSON dict merged into the config")
    args = ap.parse_args()
    args.seeds = tuple(int(x) for x in args.seeds.split(","))
    args.override = json.loads(args.override)
    LOGS.mkdir(parents=True, exist_ok=True) if args.mode != "smoke" else None
    tag = f"{args.mode}_{args.kind}" if args.mode == "wf" else args.mode
    log = Log(LOGS / f"deep_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.log" if args.mode != "smoke" else None)
    log(f"train_deep {args.mode} pid {os.getpid()} torch {torch.__version__} cuda {torch.cuda.is_available()}")
    if args.mode == "tune":
        tune(args, log, configs=args.configs.split(",") if args.configs else None)
    elif args.mode == "rehearsal":
        rehearsal(args, log)
    elif args.mode == "spike":
        spike(args, log)
    elif args.mode == "wf":
        wf(args, log)
    else:
        smoke(args, log)


if __name__ == "__main__":
    main()
