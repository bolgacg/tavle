"""V13 network: one model over every input group, 5 seeds trained together on the GPU (OBJECTIVES.md, V13).

The seeds are batched: every layer holds one weight set per seed (leading dimension S) and all seeds see the same
batch of days in one forward pass (the GRU arch keeps one cuDNN GRU per seed). Each seed has its own initial
weights, input-dropout draws, early stopping and gradient clipping, so the result equals 5 independent fits.

Menu (at most three architectures, declared): "gru" (recurrent), "tcn" (causal temporal convolution, dilations
1..64), "attn" (2-layer, 4-head self-attention with learned positions). Shared parts: per-hour input layer over
[zone-price sequence, generator-point projection (K channels per field, masks)], encoder over 168 hours, pooling
[last, mean of last 24, mean], one small encoder per day-level group (load, border, outage, weather, calendar),
head to 24 hours x 11 zones x (gap, P(gap >= 25), P(gap >= 50), P(gap >= 100)).

Regularisation: dropout 0.3, AdamW weight decay 1e-2, input dropout per group (each removable group of each day is
zeroed with probability 0.15 per seed in training), early stopping on the last 90 days of the window (time-ordered).
Loss per seed: squared error of the clipped gap (+-250, divided by its training std) + mean class-weighted BCE of
the three spike outputs (positives weighted negatives / positives, capped at 1000; the weighted probability is
mapped back exactly, p = q / (q + w (1 - q)), as v1 DeepSpikeModel). Output layers start at zero weights with the
bias at the training mean (gap) and the weighted-loss optimum (spikes), per hour and zone.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import deep_data as DD
import rolling as R
import v13_data as VD

ARCHS = ("gru", "tcn", "attn")
SPIKES = (25, 50, 100)
CFG = dict(hidden=64, K=8, G=16, dropout=0.3, in_drop=0.15, weight_decay=1e-2, lr=1e-3, batch=32, max_epochs=60,
           patience=10, min_epochs=3, val_days=90, clip=250.0, min_point_days=30, heads=4, layers=2)
N_Z = DD.N_Z
N_OUT = 1 + len(SPIKES)


# ============================================================================ grouped layers (leading seed dim)
class GLinear(nn.Module):
    def __init__(self, S, i, o, bias=True):
        super().__init__()
        self.w = nn.Parameter(torch.randn(S, i, o) / math.sqrt(i))
        self.b = nn.Parameter(torch.zeros(S, o)) if bias else None

    def forward(self, x, shared=False):
        """x [S, ..., i] (or [..., i] shared by every seed when shared=True) -> [S, ..., o]."""
        S = self.w.shape[0]
        if shared:
            sh = x.shape[:-1]
            y = (x.reshape(-1, x.shape[-1]) @ self.w.permute(1, 0, 2).reshape(x.shape[-1], -1))
            y = y.reshape(*sh, S, -1).movedim(-2, 0)
        else:
            sh = x.shape[1:-1]
            y = torch.bmm(x.reshape(S, -1, x.shape[-1]), self.w).reshape(S, *sh, -1)
        if self.b is not None:
            y = y + self.b.view(S, *([1] * (y.dim() - 2)), -1)
        return y


class GLayerNorm(nn.Module):
    def __init__(self, S, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(S, d))
        self.b = nn.Parameter(torch.zeros(S, d))

    def forward(self, x):
        S = x.shape[0]
        y = F.layer_norm(x, x.shape[-1:])
        v = (S,) + (1,) * (x.dim() - 2) + (-1,)
        return y * self.g.view(v) + self.b.view(v)


class GTCN(nn.Module):
    def __init__(self, S, d, p, dil=(1, 2, 4, 8, 16, 32, 64), ks=3):
        super().__init__()
        self.S, self.d, self.ks, self.dil = S, d, ks, dil
        self.convs = nn.ModuleList([nn.Conv1d(S * d, S * d, ks, dilation=x, groups=S) for x in dil])
        self.drop = nn.Dropout(p)

    def forward(self, x):                               # [S,B,T,d]
        S, B, T, d = x.shape
        h = x.permute(1, 0, 3, 2).reshape(B, S * d, T)
        for conv, dl in zip(self.convs, self.dil):
            h = h + self.drop(F.gelu(conv(F.pad(h, ((self.ks - 1) * dl, 0)))))
        return h.reshape(B, S, d, T).permute(1, 0, 3, 2)


class GAttn(nn.Module):
    def __init__(self, S, d, p, heads, layers, T=DD.WINDOW_H):
        super().__init__()
        self.h = heads
        self.pos = nn.Parameter(torch.randn(S, T, d) * 0.02)
        self.ln1 = nn.ModuleList([GLayerNorm(S, d) for _ in range(layers)])
        self.qkv = nn.ModuleList([GLinear(S, d, 3 * d) for _ in range(layers)])
        self.o = nn.ModuleList([GLinear(S, d, d) for _ in range(layers)])
        self.ln2 = nn.ModuleList([GLayerNorm(S, d) for _ in range(layers)])
        self.f1 = nn.ModuleList([GLinear(S, d, 2 * d) for _ in range(layers)])
        self.f2 = nn.ModuleList([GLinear(S, 2 * d, d) for _ in range(layers)])
        self.drop = nn.Dropout(p)
        self.p = p

    def forward(self, x):                               # [S,B,T,d]
        S, B, T, d = x.shape
        x = x + self.pos[:, None]
        for i in range(len(self.qkv)):
            q, k, v = self.qkv[i](self.ln1[i](x)).chunk(3, -1)
            sh = lambda t: t.reshape(S * B, T, self.h, d // self.h).transpose(1, 2)  # noqa: E731
            a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), dropout_p=self.p if self.training else 0.0)
            a = a.transpose(1, 2).reshape(S, B, T, d)
            x = x + self.drop(self.o[i](a))
            x = x + self.drop(self.f2[i](F.gelu(self.f1[i](self.ln2[i](x)))))
        return x


class Net(nn.Module):
    def __init__(self, c: dict, S: int, arch: str, n_points: int, dims: dict):
        super().__init__()
        H, K, G, p = c["hidden"], c["K"], c["G"], c["dropout"]
        self.S, self.arch, self.K = S, arch, (K if n_points else 0)
        if self.K:
            self.W = nn.Parameter(torch.randn(S, 4, n_points, self.K) / math.sqrt(n_points))
        self.inp_seq = GLinear(S, VD.N_SEQ, H)
        self.inp_gen = GLinear(S, 4 * self.K + 2, H, bias=False) if self.K else None
        self.inp_ln = GLayerNorm(S, H)
        self.drop = nn.Dropout(p)
        if arch == "gru":
            self.enc = nn.ModuleList([nn.GRU(H, H, batch_first=True) for _ in range(S)])
        elif arch == "tcn":
            self.enc = GTCN(S, H, p)
        else:
            self.enc = GAttn(S, H, p, c["heads"], c["layers"])
        self.dims = dims
        self.day = nn.ModuleDict({g: GLinear(S, n, G) for g, n in dims.items()})
        self.fut = GLinear(S, G * len(dims), H)
        self.h1 = GLinear(S, 4 * H, 2 * H)
        self.h2 = GLinear(S, 2 * H, 24 * N_Z * N_OUT)

    def seed_of(self, name: str) -> int | None:
        """Seed index of a per-seed module parameter (GRU), None for grouped parameters."""
        if name.startswith("enc.") and self.arch == "gru":
            return int(name.split(".")[1])
        return None

    def forward(self, x: dict, day: dict, keep: dict):
        """x: hourly batch (shared); day: group -> [B, n] (shared); keep: group -> [S, B] 0/1 (input dropout and
        ablations). Returns [S, B, 24, 11, 4]."""
        S = self.S
        h = self.inp_seq(x["seq"], shared=True)                                   # [S,B,T,H]
        if self.K:
            gp = torch.einsum("btpf,sfpk->sbtfk", x["gx"], self.W).flatten(-2)       # [S,B,T,4K]
            gin = torch.cat([gp, x["gm"].unsqueeze(0).expand(S, -1, -1, -1)], -1)
            h = h + self.inp_gen(gin * keep["gen"][:, :, None, None])
        h = self.drop(F.gelu(self.inp_ln(h)))
        if self.arch == "gru":
            h = torch.stack([g(h[s])[0] for s, g in enumerate(self.enc)])
        else:
            h = self.enc(h)
        z = [h[:, :, -1], h[:, :, -24:].mean(2), h.mean(2)]
        e = []
        for g in self.dims:
            v = self.day[g](day[g], shared=True)                                  # [S,B,G]
            if g in keep:
                v = v * keep[g][..., None]
            e.append(F.gelu(v))
        z.append(self.drop(F.gelu(self.fut(torch.cat(e, -1)))))
        z = self.drop(torch.cat(z, -1))
        out = self.h2(self.drop(F.gelu(self.h1(z))))
        return out.view(S, -1, 24, N_Z, N_OUT)


# ============================================================================ model
def dense(keys: pd.DataFrame, values: np.ndarray, first_day, n_days):
    """Average per (delivery-day index, local hour, zone) -> [n_days, 24, 11] and a mask (v1 DeepModel._dense)."""
    k = (keys["delivery_date"].to_numpy().astype("datetime64[D]") - np.datetime64(first_day, "D")).astype(np.int64)
    tot = np.zeros((n_days, 24, N_Z))
    cnt = np.zeros((n_days, 24, N_Z))
    ok = np.isfinite(values)
    np.add.at(tot, (k, keys["slot"].to_numpy(), keys["zc"].to_numpy()), np.where(ok, values, 0))
    np.add.at(cnt, (k, keys["slot"].to_numpy(), keys["zc"].to_numpy()), ok)
    return (tot / np.maximum(cnt, 1)).astype(np.float32), cnt > 0


def keys_of(panel: pd.DataFrame) -> pd.DataFrame:
    import deep as D1                                         # v1 panel_keys
    return D1.panel_keys(panel, need_gap="gap" in panel.columns)


class V13:
    """fit(panel_train) / predict(panel_test) for one architecture, with the removed groups `drop`."""

    def __init__(self, data: VD.Data, blocks: VD.DayBlocks, arch: str, drop=(), seeds=(0, 1, 2, 3, 4),
                 device="cuda", cfg: dict | None = None, log=print):
        assert arch in ARCHS
        self.d, self.blocks, self.arch, self.drop = data, blocks, arch, tuple(drop)
        self.seeds, self.S = tuple(seeds), len(seeds)
        self.device = torch.device(device)
        self.c = {**CFG, **(cfg or {})}
        self.log = log
        self.info: dict = {}

    # -- inputs -----------------------------------------------------------------------------------
    def _day_inputs(self, k: torch.Tensor, x: dict) -> dict:
        D = self.D
        lf = torch.clamp(x["lf"] / self.lf_mean - 1.0, -1.0, 2.0) * x["lfm"]
        B = k.shape[0]
        return {"load": torch.cat([lf.reshape(B, -1), x["lfm"].amax(-1), x["lf_age"].amax((1, 2))[:, None] / 5.0,
                                   D["lfwow"][k], D["lfwow_miss"][k]], -1),
                "border": torch.cat([D["border"][k], D["border_miss"][k]], -1),
                "outage": torch.cat([D["out"][k], D["out_miss"][k]], -1),
                "weather": torch.cat([D["wx"][k].flatten(1), D["wxm"][k].amax(-1), D["wxsrc"][k]], -1),
                "calendar": x["dcal"]}

    def _dims(self) -> dict:
        b = self.blocks
        return {"load": 24 * DD.N_LF + 24 + 1 + b.lfwow.shape[1] + 1, "border": b.border.shape[1] + 1,
                "outage": b.out.shape[1] + 1, "weather": len(b.points) * 24 + len(b.points) + 2, "calendar": DD.N_DCAL}

    def _keep(self, B: int, train: bool, g: torch.Generator | None) -> dict:
        keep = {}
        for grp in VD.GROUPS:
            if grp in self.drop:
                keep[grp] = torch.zeros(self.S, B, device=self.device)
            elif train and self.c["in_drop"] > 0:
                keep[grp] = (torch.rand(self.S, B, device=self.device, generator=g) >= self.c["in_drop"]).float()
            else:
                keep[grp] = torch.ones(self.S, B, device=self.device)
        return keep

    def _forward(self, net, k_idx, train=False, g=None):
        k = torch.as_tensor(np.asarray(k_idx), dtype=torch.long, device=self.device)
        self.blocks.assert_aligned(np.asarray(k_idx))
        x = VD.batch(self.d, k, self.cols, self.device, check=True)
        day = self._day_inputs(k, x)
        keep = self._keep(len(k), train, g)
        return net(x, day, keep), k

    # -- loss -------------------------------------------------------------------------------------
    def _losses(self, out, k):
        """Per-seed loss [S]: gap MSE (scaled) + mean weighted BCE of the spike outputs."""
        y, m = self.Yt[k], self.Ymt[k]                       # [B,24,11]
        mse = ((out[..., 0] - y) ** 2 * m).sum((1, 2, 3)) / m.sum().clamp(min=1)
        b = 0.0
        for j in range(len(SPIKES)):
            lab = self.St[j][k]
            l = F.binary_cross_entropy_with_logits(out[..., 1 + j], lab.expand_as(out[..., 1 + j]),
                                                   pos_weight=self.pw[j], reduction="none")
            b = b + (l * m).sum((1, 2, 3)) / m.sum().clamp(min=1)
        return mse + b / len(SPIKES)

    # -- per-seed snapshots and clipping ------------------------------------------------------------
    def _snap(self, net, s):
        """Seed s's weights. Grouped parameters are seed-major (leading dim S, or S x d for grouped convs)."""
        return {n: (p.detach().reshape(self.S, -1)[s].clone() if net.seed_of(n) is None else p.detach().clone())
                for n, p in net.named_parameters() if net.seed_of(n) in (None, s)}

    def _restore(self, net, s, snap):
        with torch.no_grad():
            for n, p in net.named_parameters():
                if n in snap:
                    if net.seed_of(n) is None:
                        p.view(self.S, -1)[s].copy_(snap[n])
                    else:
                        p.copy_(snap[n])

    def _clip(self, net, max_norm=1.0):
        sq = torch.zeros(self.S, device=self.device)
        for n, p in net.named_parameters():
            if p.grad is None:
                continue
            s = net.seed_of(n)
            if s is None:
                sq = sq + p.grad.pow(2).reshape(self.S, -1).sum(1)
            else:
                sq[s] = sq[s] + p.grad.pow(2).sum()
        f = torch.clamp(max_norm / (sq.sqrt() + 1e-6), max=1.0)
        for n, p in net.named_parameters():
            if p.grad is None:
                continue
            s = net.seed_of(n)
            if s is None:
                p.grad.view(self.S, -1).mul_(f[:, None])
            else:
                p.grad.mul_(f[s])

    # -- fit --------------------------------------------------------------------------------------
    def fit(self, panel_train: pd.DataFrame):
        t0 = time.time()
        c, d = self.c, self.d
        keys = keys_of(panel_train)
        R.assert_pre2024(keys["delivery_date"], "v13 fit")
        g_raw = keys["gap"].to_numpy(float)
        Y, Ym = dense(keys, g_raw, d.first_day, d.n_days)
        days = np.flatnonzero(Ym.any((1, 2)))
        gaps = np.clip(g_raw[np.isfinite(g_raw)], -c["clip"], c["clip"])
        self.s = float(max(np.std(gaps), 1.0))
        Yc = np.clip(Y, -c["clip"], c["clip"]) / self.s
        enough = d.point_hours_before(int(days[-1])) >= 24 * c["min_point_days"]
        if "gen" in self.drop:
            enough[:] = False
        self.cols = torch.as_tensor(np.flatnonzero(enough), dtype=torch.long, device=self.device)
        self.ptids_used = d.ptids[np.flatnonzero(enough)]
        lfv = np.where(d.LFm[days], d.LF[days], np.nan)
        self.lf_mean = torch.as_tensor(np.nan_to_num(np.nanmean(lfv, axis=(0, 1)), nan=1.0).astype(np.float32),
                                       device=self.device).clamp(min=1.0)
        self.sc = VD.Scaler(self.blocks, days)
        self.D = VD.day_device(self.blocks, self.sc, self.device)
        # targets
        self.Yt = torch.as_tensor(Yc, device=self.device)
        self.Ymt = torch.as_tensor(Ym.astype(np.float32), device=self.device)
        gap_bias = np.nan_to_num(np.nanmean(np.where(Ym[days], Yc[days], np.nan), 0)).astype(np.float32)
        self.St, self.pw, self.wpos, sp_bias = [], [], [], []
        for S_ in SPIKES:
            lab = np.where(np.isfinite(g_raw), (g_raw >= S_).astype(float), np.nan)
            Sv, Sm = dense(keys, lab, d.first_day, d.n_days)
            Sv = (Sv > 0).astype(np.float32)                   # a fall-back double hour counts if either spiked
            pos, n = float((Sv[days] * Sm[days]).sum()), float(Sm[days].sum())
            rate = pos / max(n, 1.0)
            w = float(np.clip((n - pos) / max(pos, 1.0), 1.0, 1000.0))
            pz, nz = (Sv[days] * Sm[days]).sum(0), Sm[days].sum(0)
            pr = np.clip((pz + 10 * rate) / (nz + 10), 1e-4, 1 - 1e-4)
            sp_bias.append(np.log(w * pr / (1 - pr)).astype(np.float32))
            self.St.append(torch.as_tensor(Sv, device=self.device))
            self.pw.append(torch.tensor(w, device=self.device))
            self.wpos.append(w)
        bias = np.stack([gap_bias] + sp_bias, -1)              # [24,11,4]
        n_val = min(c["val_days"], max(5, len(days) // 10))
        tr, va = days[:-n_val], days[-n_val:]
        net = Net(c, self.S, self.arch, len(self.cols), self._dims()).to(self.device)
        with torch.no_grad():
            for s, seed in enumerate(self.seeds):                # per-seed initial weights from its own seed
                torch.manual_seed(seed)
                fresh = Net(c, self.S, self.arch, len(self.cols), self._dims()).to(self.device)
                self._restore(net, s, self._snap(fresh, s))
            net.h2.w.zero_()
            net.h2.b.copy_(torch.as_tensor(bias.reshape(-1), device=self.device)[None].expand(self.S, -1))
        opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        rng = np.random.default_rng(self.seeds[0])
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(self.seeds[0]) + 1000)
        best = np.full(self.S, np.inf)
        best_ep = np.zeros(self.S, int)
        bad = np.zeros(self.S, int)
        snaps = [self._snap(net, s) for s in range(self.S)]
        ep = 0
        for ep in range(1, c["max_epochs"] + 1):
            net.train()
            perm = rng.permutation(tr)
            for i in range(0, len(perm), c["batch"]):
                out, k = self._forward(net, perm[i:i + c["batch"]], train=True, g=gen)
                loss = self._losses(out, k).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                self._clip(net)
                opt.step()
            v = self._eval(net, va)
            for s in range(self.S):
                if v[s] < best[s] - 1e-5:
                    best[s], best_ep[s], bad[s] = v[s], ep, 0
                    snaps[s] = self._snap(net, s)
                else:
                    bad[s] += 1
            if ep >= c["min_epochs"] and (bad >= c["patience"]).all():
                break
        for s in range(self.S):
            self._restore(net, s, snaps[s])
        net.eval()
        self.net = net
        self.info = {"arch": self.arch, "drop": list(self.drop), "s": self.s, "n_days": int(len(days)),
                     "n_points": int(len(self.cols)), "best_epoch": best_ep.tolist(), "best_val": best.tolist(),
                     "epochs_run": ep, "pos_weight": self.wpos, "fit_s": round(time.time() - t0, 1)}
        self.log(f"v13 {self.arch} -{','.join(self.drop) or 'none'}: {len(days)} days, {len(self.cols)} points, "
                 f"best epochs {best_ep.tolist()}, val {np.round(best, 4).tolist()}, {self.info['fit_s']} s")
        del self.Yt, self.Ymt, self.St

    @torch.no_grad()
    def _eval(self, net, va):
        net.eval()
        tot = np.zeros(self.S)
        n = 0
        for i in range(0, len(va), 64):
            out, k = self._forward(net, va[i:i + 64])
            w = float(self.Ymt[k].sum())
            tot += self._losses(out, k).double().cpu().numpy() * w
            n += w
        return tot / max(n, 1.0)

    @torch.no_grad()
    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        keys = keys_of(panel)
        R.assert_pre2024(keys["delivery_date"], "v13 predict")
        k_all = (keys["delivery_date"].to_numpy().astype("datetime64[D]")
                 - np.datetime64(self.d.first_day, "D")).astype(np.int64)
        days = np.unique(k_all)
        self.net.eval()
        outs = []
        for i in range(0, len(days), 64):
            o, _ = self._forward(self.net, days[i:i + 64])
            outs.append(o.double().cpu().numpy())
        o = np.concatenate(outs, 1)                                     # [S, n_days, 24, 11, 4]
        pos = np.searchsorted(days, k_all)
        ix = (pos, keys["slot"].to_numpy(), keys["zc"].to_numpy())
        res = pd.DataFrame(index=panel.index)
        res["pred"] = (o[..., 0].mean(0) * self.s)[ix]
        for j, S_ in enumerate(SPIKES):
            q = 1.0 / (1.0 + np.exp(-o[..., 1 + j]))
            w = self.wpos[j]
            res[f"p_s{S_}"] = (q / (q + w * (1.0 - q))).mean(0)[ix]
        if not np.isfinite(res.to_numpy()).all():
            raise FloatingPointError("non-finite v13 prediction")
        return res
