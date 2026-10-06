"""Deep model for the New York study (owner: deep agent; CONTRACT.md "Model interface").

    m = DeepModel(config)          # config: a key of CONFIGS or a dict; default "c1"
    m.fit(panel_train)             # rows (zone, delivery_hour, gap[, bid_date]) for delivery dates up to cutoff
    m.predict(panel) -> Series     # predicted gap (rt - da, USD/MWh), aligned to panel.index

At 05:00 New York time on bid day D it predicts the gap for every zone and every hour of D+1 in one
pass (24 x 11 outputs, one network shared across zones). Inputs (deep_data.py): 168 hours of zone
DA/RT prices with loss and congestion components, the gap, availability masks and calendar; a learned
linear projection of the generator and border points' DA and RT congestion to K channels each (only
points seen at least 30 days in the training window, and each only in the hours it published; masked
otherwise); the operator load forecast for D+1 and the day calendar. Encoder: GRU or causal temporal
convolution over the 168 hours. Head: MLP on [last state, mean of last 24 states, mean of all states,
future-block embedding] to 24 x 11. The output layer starts with zero weights and its bias at the
training mean per hour and zone, so an untrained network predicts that mean and training adds only
the deviations the data support (the gap is mostly noise: on 2022 no simple predictor beats zero).

Training: squared error on the gap clipped to +-250 USD/MWh, divided by s (the standard deviation of the
clipped training gaps). Squared error because the trading rules compare the EXPECTED gap with the fee
(as in gbm.py, whose grid also tries the same clip). AdamW, dropout, weight decay,
early stopping on the last `val_days` delivery days of the training window (time-ordered, inside the
window), then a refit on the whole window for exactly the best epoch count. Average of 3 seeds.
Optional `extra_cols`: numeric panel columns (e.g. weather or outage features for ideas B and D) that
enter through a small per zone-hour MLP added to the output.

Holdout: fit and predict call lock.assert_build_only on the panel's delivery dates, and the data
loader calls it again on what it reads.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import deep_data as DD
import lock
from deep_data import N_DCAL, N_LF, N_SEQ, N_Z, TZ, ZONES, WINDOW_H

BASE = dict(arch="gru", hidden=64, K=8, dropout=0.2, weight_decay=1e-3, lr=1e-3, loss="mse", clip=250.0,
            huber=1.0, batch=32, max_epochs=150, patience=15, min_epochs=5, val_days=90, refit_full=True,
            min_point_days=30)
# At most 8 configurations (CONTRACT.md); chosen on 2020-2022 by train_deep.py tune.
CONFIGS = {
    "c1": dict(),                                                         # GRU 64, K 8
    "c2": dict(dropout=0.3, weight_decay=1e-2),                           # stronger regularisation
    "c3": dict(hidden=32, K=4),                                           # smaller
    "c4": dict(hidden=128, dropout=0.3, weight_decay=1e-2, lr=5e-4),      # larger
    "c5": dict(arch="tcn"),                                               # temporal convolution
    "c6": dict(clip=None),                                                # no target clip
    "c7": dict(K=0),                                                      # no generator points (ablation)
    "c8": dict(K=16, dropout=0.3),                                        # more point channels
}
SEEDS = (0, 1, 2)


def resolve_config(config) -> dict:
    if config is None:
        config = "c1"
    if isinstance(config, str):
        c = {**BASE, **CONFIGS[config], "name": config}
    else:
        c = {**BASE, **config}
        c.setdefault("name", "custom")
    return c


# ----------------------------------------------------------------------------- data cache
_CACHE: dict = {}


def preload(last_delivery: dt.date, root=None, log=None, first_day: dt.date | None = None) -> DD.DeepData:
    """Load deep_data once through last_delivery and reuse it for every later fit/predict that needs
    no later date (walk-forward: preload the last month once instead of rebuilding per refit)."""
    kw = {} if root is None else {"root": root}
    if first_day is not None:
        kw["first_day"] = first_day
    d = DD.DeepData(last_delivery, log=log, **kw)
    _CACHE["data"] = d
    return d


def get_data(last_delivery: dt.date, log=None) -> DD.DeepData:
    d = _CACHE.get("data")
    if d is None or d.last_delivery < last_delivery:
        root = d.root if d is not None else None
        d = preload(last_delivery, root=root, log=log, first_day=d.first_day if d is not None else None)
    return d


# ----------------------------------------------------------------------------- panel keys
def panel_keys(panel: pd.DataFrame, need_gap: bool = False) -> pd.DataFrame:
    """zone, delivery_hour (tz-aware), delivery_date, bid_date, slot (local hour), zone code and gap,
    row for row with the panel (index = panel.index). Columns or index levels are accepted."""
    def get(name):
        if name in panel.columns:
            return pd.Series(panel[name].array)
        if name in (panel.index.names or []):
            return pd.Series(panel.index.get_level_values(name))
        return None
    zone, dh = get("zone"), get("delivery_hour")
    if zone is None or dh is None:
        raise KeyError("panel needs zone and delivery_hour (columns or index levels)")
    dh = pd.to_datetime(dh)
    if dh.dt.tz is None:
        raise ValueError("delivery_hour must be timezone-aware")
    k = pd.DataFrame({"zone": zone.astype(str), "delivery_hour": dh.dt.tz_convert(TZ)})
    k["delivery_date"] = k["delivery_hour"].dt.tz_localize(None).dt.normalize()
    k["slot"] = k["delivery_hour"].dt.hour.astype(int)
    k["bid_date"] = k["delivery_date"] - pd.Timedelta(days=1)
    bd = get("bid_date")
    if bd is not None:
        given = pd.to_datetime(bd)
        if given.dt.tz is not None:
            given = given.dt.tz_convert(TZ).dt.tz_localize(None)
        if (given.dt.normalize().to_numpy() != k["bid_date"].to_numpy()).any():
            raise ValueError("bid_date is not the day before the delivery date")
    zc = pd.Categorical(k["zone"], categories=ZONES).codes
    if (zc < 0).any():
        raise ValueError(f"unknown zones: {sorted(set(k['zone'][zc < 0]))[:5]}")
    k["zc"] = zc.astype(np.int64)
    if need_gap:
        g = get("gap")
        if g is None:
            raise KeyError("panel_train needs a gap column")
        k["gap"] = pd.to_numeric(g, errors="coerce").astype(float)
    k.index = panel.index
    return k


# ----------------------------------------------------------------------------- network
class TCN(nn.Module):
    def __init__(self, d, dropout, dilations=(1, 2, 4, 8, 16, 32, 64), ks=3):
        super().__init__()
        self.ks, self.dil = ks, dilations
        self.convs = nn.ModuleList([nn.Conv1d(d, d, ks, dilation=x) for x in dilations])
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                   # [B, T, d] -> [B, T, d], causal
        h = x.transpose(1, 2)
        for conv, d in zip(self.convs, self.dil):
            h = h + self.drop(F.gelu(conv(F.pad(h, ((self.ks - 1) * d, 0)))))
        return h.transpose(1, 2)


class Net(nn.Module):
    def __init__(self, c: dict, n_points: int, n_extra: int = 0):
        super().__init__()
        H, K, p = c["hidden"], c["K"], c["dropout"]
        self.K = K if n_points > 0 else 0
        if self.K:
            self.W = nn.Parameter(torch.randn(2, n_points, self.K) / math.sqrt(n_points))
        f_in = N_SEQ + (2 * self.K + 2 if self.K else 0)
        self.inp = nn.Sequential(nn.Linear(f_in, H), nn.LayerNorm(H), nn.GELU(), nn.Dropout(p))
        self.arch = c["arch"]
        self.enc = nn.GRU(H, H, batch_first=True) if self.arch == "gru" else TCN(H, p)
        f_fut = 24 * N_LF + 24 + 1 + N_DCAL
        self.fut = nn.Sequential(nn.Linear(f_fut, H), nn.GELU(), nn.Dropout(p))
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(4 * H, 2 * H), nn.GELU(), nn.Dropout(p),
                                  nn.Linear(2 * H, 24 * N_Z))
        self.n_extra = n_extra
        if n_extra:
            self.ctx = nn.Linear(4 * H, 16)
            self.row = nn.Sequential(nn.Linear(2 * n_extra + 16 + N_Z, 32), nn.GELU(), nn.Dropout(p),
                                     nn.Linear(32, 1))

    def forward(self, x: dict, lf_mean: torch.Tensor, xr=None, mr=None):
        seq = x["seq"]
        B = seq.shape[0]
        parts = [seq]
        if self.K:
            gx = x["gx"]                                                   # [B,T,P,2]
            parts += [torch.einsum("btp,pk->btk", gx[..., 0], self.W[0]),
                      torch.einsum("btp,pk->btk", gx[..., 1], self.W[1]),
                      x["gm"].mean(2)]
        h = self.inp(torch.cat(parts, -1))
        h = self.enc(h)[0] if self.arch == "gru" else self.enc(h)
        lf = torch.clamp(x["lf"] / lf_mean - 1.0, -1.0, 2.0) * x["lfm"]
        fut = torch.cat([lf.reshape(B, -1), x["lfm"].amax(-1), x["lf_age"].amax((1, 2))[:, None] / 5.0,
                         x["dcal"]], -1)
        z = torch.cat([h[:, -1], h[:, -24:].mean(1), h.mean(1), self.fut(fut)], -1)
        out = self.head(z).view(B, 24, N_Z)
        if self.n_extra:
            ctx = self.ctx(z)[:, None, None, :].expand(B, 24, N_Z, 16)
            zone = torch.eye(N_Z, device=z.device)[None, None].expand(B, 24, N_Z, N_Z)
            out = out + self.row(torch.cat([xr, mr, ctx, zone], -1)).squeeze(-1)
        return out


# ----------------------------------------------------------------------------- model
class DeepModel:
    name = "deep"

    def __init__(self, config=None, seeds=SEEDS, data: DD.DeepData | None = None, device=None, log=None,
                 extra_cols: list[str] | None = None, ckpt_path: Path | None = None):
        self.c = resolve_config(config)
        self.name = f"deep_{self.c['name']}"
        self.seeds = tuple(seeds)
        self.data = data
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.log = log or (lambda *_: None)
        self.extra_cols = list(extra_cols or [])
        self.ckpt_path = ckpt_path
        self.nets: list[Net] = []
        self.info: dict = {}

    # -- helpers ---------------------------------------------------------------------------
    def _data(self, last: dt.date) -> DD.DeepData:
        if self.data is not None and self.data.last_delivery >= last:
            return self.data
        new = get_data(last, log=self.log)
        if self.nets:                                   # fitted: map the trained points into the new data
            pos = np.searchsorted(new.ptids, self.ptids_used)
            ok = (pos < len(new.ptids)) & (new.ptids[np.minimum(pos, max(len(new.ptids) - 1, 0))] == self.ptids_used)
            if not ok.all() or new.first_day != self.data.first_day:
                raise ValueError("reloaded data does not cover the trained points")
            self.cols = torch.as_tensor(pos, dtype=torch.long, device=self.device)
        self.data = new
        return self.data

    def _dense(self, keys: pd.DataFrame, values: np.ndarray, d: DD.DeepData):
        """Average values per (delivery-day index, slot, zone) -> [n_days, 24, 11] and a mask."""
        k = (keys["delivery_date"].to_numpy().astype("datetime64[D]")
             - np.datetime64(d.first_day, "D")).astype(np.int64)
        shape = values.shape[1:] if values.ndim > 1 else ()
        tot = np.zeros((d.n_days, 24, N_Z) + shape)
        cnt = np.zeros((d.n_days, 24, N_Z) + shape)
        ok = np.isfinite(values)
        np.add.at(tot, (k, keys["slot"].to_numpy(), keys["zc"].to_numpy()), np.where(ok, values, 0))
        np.add.at(cnt, (k, keys["slot"].to_numpy(), keys["zc"].to_numpy()), ok)
        return (tot / np.maximum(cnt, 1)).astype(np.float32), cnt > 0, k

    def _extra(self, panel, keys, d):
        if not self.extra_cols:
            return None, None
        X = panel[self.extra_cols].to_numpy(float)
        X = (X - self.info["extra_mean"]) / self.info["extra_std"]
        v, m, _ = self._dense(keys, X, d)
        return torch.as_tensor(np.nan_to_num(v) * m, device=self.device), torch.as_tensor(m.astype(np.float32),
                                                                                          device=self.device)

    def _inputs(self, d, k_idx):
        return DD.batch(d, torch.as_tensor(k_idx, dtype=torch.long), self.cols, self.device, check=True)

    def _forward(self, net, d, k_idx, xr=None, mr=None):
        x = self._inputs(d, k_idx)
        sel = torch.as_tensor(k_idx, device=self.device)
        return net(x, self.lf_mean, None if xr is None else xr[sel], None if mr is None else mr[sel])

    # -- fit -------------------------------------------------------------------------------
    def fit(self, panel_train: pd.DataFrame) -> None:
        t0 = time.time()
        c = self.c
        keys = panel_keys(panel_train, need_gap=True)
        lock.assert_build_only(keys["delivery_date"])
        last = keys["delivery_date"].max().date()
        d = self._data(last)
        Y, Ym, _ = self._dense(keys, keys["gap"].to_numpy(float), d)
        days = np.flatnonzero(Ym.any((1, 2)))                       # delivery-day indices with a target
        if len(days) < 20:
            raise ValueError(f"only {len(days)} training days")
        gaps = keys["gap"].to_numpy(float)
        gaps = gaps[np.isfinite(gaps)]
        clip = c["clip"]
        if clip:
            gaps, Y = np.clip(gaps, -clip, clip), np.clip(Y, -clip, clip)
        self.s = float(max(np.std(gaps), 1.0))
        k_last = int(days[-1])
        enough = d.point_hours_before(k_last) >= 24 * c["min_point_days"]
        if c["K"] == 0:
            enough[:] = False
        self.cols = torch.as_tensor(np.flatnonzero(enough), dtype=torch.long, device=self.device)
        self.ptids_used = d.ptids[np.flatnonzero(enough)]
        lfv = np.where(d.LFm[days], d.LF[days], np.nan)
        self.lf_mean = torch.as_tensor(np.nanmean(lfv, axis=(0, 1)).astype(np.float32), device=self.device)
        bias = np.nanmean(np.where(Ym[days], Y[days] / self.s, np.nan), 0)
        bias = np.nan_to_num(bias).astype(np.float32)
        if self.extra_cols:
            X = panel_train[self.extra_cols].to_numpy(float)
            self.info["extra_mean"] = np.nanmean(X, 0)
            self.info["extra_std"] = np.where(np.nanstd(X, 0) > 0, np.nanstd(X, 0), 1.0)
        xr, mr = self._extra(panel_train, keys, d)
        self.Yt = torch.as_tensor(np.nan_to_num(Y) / self.s, device=self.device)
        self.Ymt = torch.as_tensor(Ym, device=self.device)
        n_val = min(c["val_days"], max(5, len(days) // 10))
        tr, va = days[:-n_val], days[-n_val:]
        self.nets, seeds_info = [], []
        for seed in self.seeds:
            ts = time.time()
            net, best_ep, hist = self._train(seed, tr, va, c["max_epochs"], bias, xr, mr)
            info = {"seed": seed, "best_epoch": best_ep, "best_val": hist["best_val"],
                    "epochs_run": hist["epochs"], "stage1_s": round(time.time() - ts, 1)}
            if c["refit_full"]:
                ts2 = time.time()
                net, _, _ = self._train(seed, days, None, max(best_ep, 1), bias, xr, mr)
                info["stage2_s"] = round(time.time() - ts2, 1)
            net.eval()
            self.nets.append(net)
            seeds_info.append(info)
            self.log(f"{self.name} seed {seed}: best epoch {best_ep}, val {hist['best_val']:.4f}, "
                     f"{info['stage1_s']}s + {info.get('stage2_s', 0)}s")
        self.info.update({"config": c, "s": self.s, "train_first": str(keys['delivery_date'].min().date()),
                          "train_last": str(last), "n_days": int(len(days)), "n_val": int(n_val),
                          "n_points": int(len(self.ptids_used)), "seeds": seeds_info,
                          "fit_s": round(time.time() - t0, 1)})
        del self.Yt, self.Ymt
        if self.ckpt_path:
            self.save(self.ckpt_path)

    def _loss(self, net, d, k_idx, xr, mr):
        out = self._forward(net, d, k_idx, xr, mr)
        sel = torch.as_tensor(k_idx, device=self.device)
        y, m = self.Yt[sel], self.Ymt[sel]
        return (self._crit(out, y) * m).sum() / m.sum().clamp(min=1)

    def _crit(self, out, y):
        if self.c["loss"] == "huber":
            return F.huber_loss(out, y, reduction="none", delta=self.c["huber"])
        return (out - y) ** 2

    def _train(self, seed, tr, va, max_epochs, bias, xr, mr):
        c, d = self.c, self.data
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = Net(c, len(self.cols), len(self.extra_cols)).to(self.device)
        with torch.no_grad():                    # start exactly at the training mean per hour and zone
            net.head[-1].weight.zero_()
            net.head[-1].bias.copy_(torch.as_tensor(bias.reshape(-1), device=self.device))
            if net.n_extra:
                net.row[-1].weight.zero_()
                net.row[-1].bias.zero_()
        opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        g = np.random.default_rng(seed)
        best, best_ep, best_state, bad, ep = math.inf, 0, None, 0, 0
        for ep in range(1, max_epochs + 1):
            net.train()
            perm = g.permutation(tr)
            for i in range(0, len(perm), c["batch"]):
                loss = self._loss(net, d, perm[i:i + c["batch"]], xr, mr)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            if va is None:
                continue
            v = self._eval(net, va, xr, mr)
            if v < best - 1e-5:
                best, best_ep, bad = v, ep, 0
                best_state = copy.deepcopy(net.state_dict())
            else:
                bad += 1
                if bad >= c["patience"] and ep >= c["min_epochs"]:
                    break
        if va is not None and best_state is not None:
            net.load_state_dict(best_state)
        return net, best_ep, {"best_val": best, "epochs": ep}

    @torch.no_grad()
    def _eval(self, net, va, xr, mr):
        net.eval()
        tot, n = 0.0, 0.0
        for i in range(0, len(va), 64):
            b = va[i:i + 64]
            out = self._forward(net, self.data, b, xr, mr)
            sel = torch.as_tensor(b, device=self.device)
            l = self._crit(out, self.Yt[sel])
            m = self.Ymt[sel]
            tot += float((l * m).sum())
            n += float(m.sum())
        return tot / max(n, 1)

    # -- predict ---------------------------------------------------------------------------
    @torch.no_grad()
    def predict_days(self, panel: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Ensemble output [n_days, 24, 11] in USD/MWh for the panel's delivery days."""
        keys = panel_keys(panel)
        lock.assert_build_only(keys["delivery_date"])
        d = self._data(keys["delivery_date"].max().date())
        k_all = (keys["delivery_date"].to_numpy().astype("datetime64[D]")
                 - np.datetime64(d.first_day, "D")).astype(np.int64)
        days = np.unique(k_all)
        xr, mr = self._extra(panel, keys, d)
        outs = []
        for net in self.nets:
            net.eval()
            o = [self._forward(net, d, days[i:i + 64], xr, mr).float().cpu().numpy() for i in range(0, len(days), 64)]
            outs.append(np.concatenate(o))
        return keys, days, np.mean(outs, 0) * self.s

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        keys, days, out = self.predict_days(panel)
        k_all = (keys["delivery_date"].to_numpy().astype("datetime64[D]")
                 - np.datetime64(self.data.first_day, "D")).astype(np.int64)
        pos = np.searchsorted(days, k_all)
        pred = out[pos, keys["slot"].to_numpy(), keys["zc"].to_numpy()]
        if not np.isfinite(pred).all():
            raise FloatingPointError("non-finite deep prediction")
        return pd.Series(pred, index=panel.index, name="pred_gap")

    # -- persistence -------------------------------------------------------------------------
    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.c, "seeds": self.seeds, "s": self.s, "lf_mean": self.lf_mean.cpu(),
                    "ptids_used": self.ptids_used, "extra_cols": self.extra_cols,
                    "extra_stats": {k: self.info[k] for k in ("extra_mean", "extra_std") if k in self.info},
                    "info": {k: v for k, v in self.info.items() if k not in ("extra_mean", "extra_std")},
                    "nets": [n.state_dict() for n in self.nets]}, path)

    @classmethod
    def load(cls, path: Path, data: DD.DeepData | None = None, device=None, log=None) -> "DeepModel":
        """Rebuild a fitted model; points are matched by ptid, so the data may hold a different set."""
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(ck["config"], seeds=ck["seeds"], data=data, device=device, log=log, extra_cols=ck["extra_cols"])
        m.s, m.ptids_used, m.info = ck["s"], ck["ptids_used"], {**ck["info"], **ck["extra_stats"]}
        m.lf_mean = ck["lf_mean"].to(m.device)
        if data is None:
            m.data = get_data(dt.date.fromisoformat(ck["info"]["train_last"]), log=log)
        pos = np.searchsorted(m.data.ptids, m.ptids_used)
        if not (pos < len(m.data.ptids)).all() or not (m.data.ptids[np.minimum(pos, len(m.data.ptids) - 1)]
                                                       == m.ptids_used).all():
            raise ValueError("the data lacks points the model was trained on")
        m.cols = torch.as_tensor(pos, dtype=torch.long, device=m.device)
        for sd in ck["nets"]:
            net = Net(m.c, len(m.cols), len(m.extra_cols)).to(m.device)
            net.load_state_dict(sd)
            net.eval()
            m.nets.append(net)
        return m


def describe(c: dict) -> str:
    return json.dumps({k: c[k] for k in ("arch", "hidden", "K", "dropout", "weight_decay", "lr", "loss", "clip")})
