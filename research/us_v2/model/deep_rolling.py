"""GPU lane: the v1 deep model (deep.py, GRU over 168 hours of zone and point prices) on the rolling quarters.

    python deep_rolling.py c   [--device cuda]    V1 C deep: 5 seeds per refit, one process, the data tensors
                                                   (2010..2023) loaded once and kept on the GPU; also writes ens4,
                                                   the mean of the last 4 windows' models on each quarter (V11)
    python deep_rolling.py v10 [--device cuda]    V10: a model PRETRAINED on every delivery date from 2010-01-01 to
                                                   2 days before 1 January of each year (refit yearly), FINE-TUNED per
                                                   quarter on the 3-year window (lr / 10, early stopping inside the
                                                   window, then the same epochs on the whole window)

Both write results/v2/preds/deep_<mode>.parquet through rolling.run_rows (restartable per quarter; the
fitted models of each quarter are saved, so ens4 and the pretrained models survive a restart).
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import time

import numpy as np
import pandas as pd

import rolling as R
import deep as D1          # v1
import deep_data as DD     # v1

import torch

SEEDS = (0, 1, 2, 3, 4)
ENS = 4
FT = dict(lr_div=10.0, max_epochs=40, patience=8)


def load_data(log) -> DD.DeepData:
    d = DD.DeepData(R.LAST_SCORED, root=R.PARQUET, first_day=R.DATA_START, log=log)
    # Three 2010 load-forecast files (delivery 2010-04-27, 05-24, 05-29) carry a published_at before their own
    # issue date; the batch guard rejects them (issue date after the bid day). Conservative fix: hide those cells,
    # so the model sees no load forecast there (never more information than the rule allows).
    k = np.arange(d.n_days)[:, None, None]
    bad = d.LFm & (d.LFiss > k - 1)
    if bad.any():
        log(f"load forecast: {int(bad.sum())} cells with issue date after the bid day hidden "
            f"(days {sorted({str(d.first_day + dt.timedelta(days=int(i))) for i in np.argwhere(bad)[:, 0]})})")
        d.LFm = d.LFm & ~bad
        d.LF = np.where(bad, 0, d.LF).astype(np.float32)
    D1._CACHE["data"] = d
    return d


class FineTuned(D1.DeepModel):
    """DeepModel that starts from a pretrained model's networks (same points, scale and load mean)."""

    def __init__(self, pre: D1.DeepModel, **kw):
        super().__init__(pre.c, seeds=pre.seeds, data=pre.data, device=pre.device, log=pre.log)
        self.pre = pre
        self.c = {**pre.c, "lr": pre.c["lr"] / FT["lr_div"], "max_epochs": FT["max_epochs"],
                  "patience": FT["patience"], "name": pre.c["name"] + "_ft"}
        self.name = "deep_v10"
        self._seed_i = 0

    def _new_net(self, bias):
        return copy.deepcopy(self.pre.nets[self._seed_i]).train()

    def fit(self, panel_train: pd.DataFrame) -> None:
        t0 = time.time()
        pre, c = self.pre, self.c
        keys = D1.panel_keys(panel_train, need_gap=True)
        R.assert_pre2024(keys["delivery_date"], "v10 fine-tune")
        d = self.data
        Y, Ym, _ = self._dense(keys, keys["gap"].to_numpy(float), d)
        days = np.flatnonzero(Ym.any((1, 2)))
        if c["clip"]:
            Y = np.clip(Y, -c["clip"], c["clip"])
        self.s, self.cols, self.ptids_used, self.lf_mean = pre.s, pre.cols, pre.ptids_used, pre.lf_mean
        self.Yt = torch.as_tensor(np.nan_to_num(Y) / self.s, device=self.device)
        self.Ymt = torch.as_tensor(Ym, device=self.device)
        n_val = min(c["val_days"], max(5, len(days) // 10))
        tr, va = days[:-n_val], days[-n_val:]
        self.nets, info = [], []
        for i, seed in enumerate(self.seeds):
            self._seed_i = i
            net, best_ep, hist = self._train(seed, tr, va, c["max_epochs"], None, None, None)
            if best_ep >= 1:                               # 0 = no improvement on the pretrained model
                net, _, _ = self._train(seed, days, None, best_ep, None, None, None)
            else:
                net = copy.deepcopy(pre.nets[i])
            net.eval()
            self.nets.append(net)
            info.append({"seed": seed, "best_epoch": best_ep, "best_val": hist["best_val"]})
        self.info = {"config": c, "s": self.s, "n_days": int(len(days)), "seeds": info,
                     "pretrained_on": pre.info.get("train_first"), "pretrained_to": pre.info.get("train_last"),
                     "fit_s": round(time.time() - t0, 1)}
        del self.Yt, self.Ymt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["c", "v10"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config", default="c1")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    log = R.log
    panel = R.load_panel(columns=["delivery_hour", "zone", "delivery_date", "gap"])
    data = load_data(log)
    log(f"deep data {data.first_day}..{data.last_delivery}, {len(data.ptids)} points; panel {panel.shape}")
    name = f"deep_{a.mode}" + ("_smoke" if a.smoke else "")
    mdir = R.PREDS / f"{name}_models"
    mdir.mkdir(parents=True, exist_ok=True)
    cfg = D1.resolve_config(a.config)
    if a.smoke:
        cfg = {**cfg, "max_epochs": 2, "min_epochs": 1, "patience": 1}
    seeds = SEEDS[:2] if a.smoke else SEEDS

    def new_model():
        return D1.DeepModel(cfg, seeds=seeds, data=data, device=a.device, log=log)

    def loadm(path):
        return D1.DeepModel.load(path, data=data, device=a.device, log=log)

    if a.mode == "c":
        def fp(tr, te, q):
            m = new_model()
            m.fit(tr)
            m.save(mdir / f"{q[2]}.pt")
            out = pd.DataFrame(index=te.index)
            out["pred"] = m.predict(te)
            prev = [p for p in sorted(mdir.glob("*.pt")) if p.stem < q[2]][-(ENS - 1):]
            ens = [out["pred"].to_numpy()] + [loadm(p).predict(te).to_numpy() for p in prev]
            out["ens4"], out["ens_n"] = np.mean(ens, 0), len(ens)
            out["fit_s"] = m.info.get("fit_s")
            torch.cuda.empty_cache() if a.device == "cuda" else None
            return out
    else:
        pre_cache: dict = {}

        def pretrained(year: int) -> D1.DeepModel:
            if year in pre_cache:
                return pre_cache[year]
            f = mdir / f"pre_{year}.pt"
            if f.exists():
                m = loadm(f)
            else:
                cut = dt.date(year, 1, 1) - dt.timedelta(days=R.TRAIN_GAP_DAYS)
                d = pd.to_datetime(panel["delivery_date"])
                tr = panel[(d <= pd.Timestamp(cut)) & panel["gap"].notna()]
                R.assert_pre2024(tr["delivery_hour"], "v10 pretrain")
                m = new_model()
                m.fit(tr)
                m.save(f)
                log(f"v10 pretrained for {year}: {tr['delivery_date'].min().date()}..{tr['delivery_date'].max().date()}, "
                    f"{m.info.get('fit_s')} s")
            pre_cache.clear()
            pre_cache[year] = m
            return m

        def fp(tr, te, q):
            pre = pretrained(q[0].year)
            m = FineTuned(pre)
            m.fit(tr)
            out = pd.DataFrame(index=te.index)
            out["pred"] = m.predict(te)
            out["pre_pred"] = pre.predict(te)
            out["fit_s"] = m.info.get("fit_s")
            return out

    qs = R.quarters()[3:5] if a.smoke else None
    R.run_rows(name, fp, panel, qs=qs, log=log)
    if not a.smoke:
        R.done(f"v2_deep_{a.mode}")


if __name__ == "__main__":
    main()
