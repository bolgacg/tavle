"""Idea 14, learned allocator over the four candidates (OBJECTIVES addendum 7 Oct 2026, commit d6571a3).

Per delivery day D+1 a small deep gate reads the day-level feature matrix (the same file and encoders as ideas 11
and 12, deep_policy.py) and splits one unit of risk between the four candidates C_deep, B_gbm, C_gbm and B_deep
(the lead modeller's hourly positions, side/lead_positions_2021_2023.parquet). Each candidate is pre-scaled to the
same worst-drawdown budget (100,000 USD, cap 5x) with a scale computed only from its realised daily profits on the
refit's training days, never from the full period. Position per zone-hour: sum_c w_c(D) * scale_c * mw_c.

Objective per day (scaled by s = 1.4826 * MAD of the equal-weight mix on the training days):
    J = M / s - LOSS_LAM * max(0, -M / s)^2 - kappa * KL(w || equal),   M = sum_c w_c * scale_c * P_c
where P_c is the candidate's own realised day profit after full cost (lab.py's cost rule). Training on the sum of
the candidates' own profits charges both sides when two candidates take opposite legs in one zone-hour; the lab
nets them, so the training objective is the conservative side. kappa in KAPPAS (3 values); the kappa used in year Y
is the one whose walk-forward weights earned the most in Y-1 on days public by 05:00 on 31 December of Y-1; 2021
has no earlier candidate year, so it uses the declared default (the middle value).

Declared before any run: LOSS_LAM 0.01 (the middle of ideas 11/12's grid); a candidate takes part in a refit only
with at least MIN_HIST (60) training days of its own profits, else its weight is 0 (C has profits from 2021-01,
B from 2021-05, so the first refits sit out or use C only); training days are those where every taking-part
candidate has a profit. Network: deep_policy's encoders and trunk (h_trunk 32), output layer zero-initialised (the
gate starts at equal weights), dropout 0.3, weight decay 0.1, early stopping as deep_policy, 5 seeds averaged.

Walk-forward: monthly refits 2021-01 to 2023-12; training days are delivery dates up to two days before the month
whose prices (hence the candidates' profits) were all public by 05:00 on the month's first bid day (deep_policy
train_rows). HOLDOUT: every loader filters delivery times before 2024-01-01 at load and asserts it.
"""
from __future__ import annotations

import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd

import deep_policy as DP
import lock
from common import ZONES, decision_time

torch, nn, F = DP.torch, DP.nn, DP.F

CANDS = ("C_deep", "B_gbm", "C_gbm", "B_deep")
NC = len(CANDS)
KAPPAS = (0.01, 0.1, 1.0)
LOSS_LAM = 0.01
DD_BUDGET = 100_000.0
SCALE_CAP = 5.0
MIN_HIST = 60
LEAD_POS = DP.NYISO_HOME / "side" / "lead_positions_2021_2023.parquet"
OUT = DP.RESULTS / "idea14_wf_2021_2023.parquet"
CONFIG = {**DP.CONFIG, "h_trunk": 32, "weight_decay": 0.1}


# ============================================================================ candidates
def load_candidates(path: Path = LEAD_POS) -> pd.DataFrame:
    import pyarrow.dataset as ds
    d = DP._read_build(Path(path), "delivery_hour", ["delivery_hour", "zone", "mw", "strategy"],
                       ds.field("strategy").isin(list(CANDS)))
    return d


class CandProfits:
    """P [n_days, 4]: each candidate's realised day profit at its exported size, after full cost by side per
    zone-hour (lab.py Frame.day_pnl); NaN on days the candidate has no positions. Rows follow tg.days."""

    def __init__(self, px: pd.DataFrame, cand: pd.DataFrame, tg: DP.Targets):
        h = DP.hourly_frame(px)
        DP.assert_build(cand["delivery_hour"], "candidates")
        lock.assert_build_only(cand["delivery_hour"])
        self.P = np.full((len(tg.days), NC), np.nan)
        for j, c in enumerate(CANDS):
            g = cand[cand["strategy"] == c][["delivery_hour", "zone", "mw"]]
            if not len(g):
                continue
            m = h.merge(g, on=["delivery_hour", "zone"], how="inner", validate="one_to_one")
            mw = m["mw"].to_numpy(float)
            cost = np.where(mw < 0, m["year"].map(DP.COST), m["year"].map(DP.LOADCOST))
            pnl = pd.Series(mw * m["gap"].to_numpy(float) - np.abs(mw) * cost).groupby(m["ddate"].to_numpy()).sum()
            r = tg.index_of(pnl.index)
            self.P[r[r >= 0], j] = pnl.to_numpy()[r >= 0]


def dd_scale(x: np.ndarray) -> float:
    """Size multiple at which the worst drawdown of the daily series x equals DD_BUDGET, capped at SCALE_CAP."""
    cum = np.concatenate([[0.0], np.cumsum(x)])
    mdd = float(-(cum - np.maximum.accumulate(cum)).min())
    return min(SCALE_CAP, DD_BUDGET / mdd) if mdd > 0 else SCALE_CAP


# ============================================================================ network
if nn is not None:
    class GateNet(nn.Module):
        def __init__(self, gens, dims: dict, c: dict):
            super().__init__()
            h, p = c["h_enc"], c["dropout"]
            self.wx, self.lf, self.px = (DP.BEnc(gens, 24, h, p), DP.BEnc(gens, dims["lf"], h, p),
                                         DP.BEnc(gens, dims["px"], h, p))
            self.gl = DP.BLinear(gens, dims["gl"], h)
            self.t1 = DP.BLinear(gens, 3 * (2 * h + 1) + h, c["h_trunk"])
            self.out = DP.BLinear(gens, c["h_trunk"], NC)
            self.drop = nn.Dropout(p)
            with torch.no_grad():
                self.out.w.zero_()
                self.out.b.zero_()

        def forward(self, xb):
            _, wxp = self.wx(xb["wx"], xb["wx_frac"])
            _, lfp = self.lf(xb["lf"], xb["lf_frac"])
            _, pxp = self.px(xb["px"], xb["px_frac"])
            g = F.gelu(self.gl(xb["gl"]))
            hh = self.drop(F.gelu(self.t1(self.drop(torch.cat([wxp, lfp, pxp, g], -1)))))
            return self.out(hh)                                     # logits [K,B,C]


def gate_weights(logits, avail):
    return torch.softmax(logits.masked_fill(~avail, -1e9), -1)


def gate_objective(logits, S, avail, kappa, lam: float, s: float):
    """J per model and day [K,B]. S: scaled candidate profits [K,B,C] (0 where not taking part)."""
    w = gate_weights(logits, avail)
    mix = (w * S).sum(-1) / s
    n = avail.sum().float()
    kl = (w * (torch.log(w.clamp_min(1e-12)) + torch.log(n)) * avail).sum(-1)
    return mix - lam * F.relu(-mix) ** 2 - kappa[:, None] * kl


class Alloc14:
    """Every kappa x every seed is one of K models trained side by side (as deep_policy.PolicyModel)."""

    def __init__(self, kappas=KAPPAS, seeds=DP.SEEDS, config=None, device=None):
        self.kappas, self.seeds = tuple(kappas), tuple(seeds)
        self.c = {**CONFIG, **(config or {})}
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.K = len(self.kappas) * len(self.seeds)
        self.kap_k = np.repeat(np.array(self.kappas, float), len(self.seeds))
        self.seed_k = np.tile(np.array(self.seeds, int), len(self.kappas))
        self.info: dict = {}

    def _t(self, a):
        return torch.as_tensor(np.ascontiguousarray(a), device=self.device)

    def fit(self, days: pd.DataFrame, S: np.ndarray, avail: np.ndarray):
        lock.assert_build_only(days["delivery_date"])
        assert days["delivery_date"].is_monotonic_increasing
        t0 = time.time()
        self.prep = DP.Prep(days, self.c["min_col_obs"])
        X = {k: self._t(v) for k, v in self.prep.arrays(days).items()}
        T = self._t(S.astype(np.float32))
        self.avail = self._t(avail.astype(bool))
        eq = S[:, avail].mean(1)
        self.s = float(max(1.4826 * np.median(np.abs(eq - np.median(eq))), 1.0))
        n = len(days)
        nv = min(int(min(self.c["val_max"], max(self.c["val_min"], round(self.c["val_frac"] * n)))), n // 2)
        tr, va = np.arange(n - nv), np.arange(n - nv, n)
        _, best_ep, best_val = self._train(X, T, tr, va, self.c["max_epochs"])
        self.net, _, _ = self._train(X, T, np.arange(n), None, int(best_ep.max()), snapshot_at=best_ep)
        self.net.eval()
        self.info = {"n_days": n, "n_val": nv, "s": self.s, "train_first": str(days["delivery_date"].min().date()),
                     "train_last": str(days["delivery_date"].max().date()),
                     "best_epochs": {str(k): best_ep[self.kap_k == k].astype(int).tolist() for k in self.kappas},
                     "fit_s": round(time.time() - t0, 1)}
        return self

    def _train(self, X, T, tr, va, max_epochs, snapshot_at=None):
        c, K = self.c, self.K
        gens = [torch.Generator().manual_seed(int(s)) for s in self.seed_k]
        net = GateNet(gens, self.prep.dims, c).to(self.device)
        torch.manual_seed(int(self.seed_k[0]) + 1000 * len(tr))
        opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        params = list(net.parameters())
        rngs = [np.random.default_rng(int(s)) for s in self.seed_k]
        kap = torch.as_tensor(self.kap_k, dtype=torch.float32, device=self.device)
        best, best_ep, bad = np.full(K, -np.inf), np.zeros(K, int), np.zeros(K, int)
        active = np.ones(K, bool)
        keep = {k: p.detach().clone() for k, p in net.named_parameters()}
        for ep in range(1, max_epochs + 1):
            net.train()
            perms = np.stack([r.permutation(tr) for r in rngs])
            for i in range(0, len(tr), c["batch"]):
                idx = torch.as_tensor(perms[:, i:i + c["batch"]], device=self.device)
                J = gate_objective(net({k: v[idx] for k, v in X.items()}), T[idx], self.avail, kap, LOSS_LAM, self.s)
                opt.zero_grad(set_to_none=True)
                (-J.mean(1).sum()).backward()
                DP._clip_per_model(params, K, c["clip"])
                opt.step()
            if snapshot_at is not None:
                m = torch.as_tensor(snapshot_at == ep, device=self.device)
                if m.any():
                    with torch.no_grad():
                        for k, p in net.named_parameters():
                            keep[k][m] = p.detach()[m]
            if va is None:
                continue
            net.eval()
            with torch.no_grad():
                idx = torch.as_tensor(np.tile(va, (K, 1)), device=self.device)
                v = gate_objective(net({k: x[idx] for k, x in X.items()}), T[idx], self.avail, kap, LOSS_LAM,
                                   self.s).mean(1).cpu().numpy()
            imp = active & (v > best + 1e-9)
            best[imp], best_ep[imp], bad[imp] = v[imp], ep, 0
            bad[active & ~imp] += 1
            if imp.any():
                m = torch.as_tensor(imp, device=self.device)
                with torch.no_grad():
                    for k, p in net.named_parameters():
                        keep[k][m] = p.detach()[m]
            active &= bad < c["patience"]
            if not active.any():
                break
        with torch.no_grad():
            for k, p in net.named_parameters():
                p.copy_(keep[k])
        return net, np.maximum(best_ep, 1), best

    def predict(self, days: pd.DataFrame) -> dict:
        """Per kappa: weights [n, 4], the seed average."""
        lock.assert_build_only(days["delivery_date"])
        X = {k: self._t(v) for k, v in self.prep.arrays(days).items()}
        with torch.no_grad():
            idx = torch.as_tensor(np.tile(np.arange(len(days)), (self.K, 1)), device=self.device)
            W = gate_weights(self.net({k: v[idx] for k, v in X.items()}), self.avail).cpu().numpy()
        return {k: W[self.kap_k == k].mean(0) for k in self.kappas}


# ============================================================================ walk-forward
def walk_forward(feats, tg, cp: CandProfits, first: dt.date, last: dt.date, kappas=KAPPAS, seeds=DP.SEEDS,
                 config=None, device=None, log=print, gap_days: int = DP.TRAIN_GAP_DAYS,
                 known_shift: pd.Timedelta | None = None):
    """Monthly refits over [first, last] -> (day rows for every kappa, refit records). gap_days and known_shift
    exist only so the lookahead test can open the filter."""
    lock.assert_build_only([first, last])
    feats = DP.mask_weather(feats.sort_values("delivery_date", kind="stable").reset_index(drop=True))
    rows, recs = [], []
    for ms in DP.month_starts(first, last):
        me = min(dt.date(ms.year + (ms.month == 12), ms.month % 12 + 1, 1) - dt.timedelta(days=1), last)
        known = decision_time(ms - dt.timedelta(days=1)) + (known_shift or pd.Timedelta(0))
        tr = feats.iloc[DP.train_rows(feats, tg, ms, gap_days, known)]
        te = feats[(feats["delivery_date"] >= pd.Timestamp(ms)) & (feats["delivery_date"] <= pd.Timestamp(me))]
        if gap_days >= DP.TRAIN_GAP_DAYS and known_shift is None and len(tr):
            assert tr["delivery_date"].max() <= pd.Timestamp(ms - dt.timedelta(days=DP.TRAIN_GAP_DAYS))
        P = cp.P[tg.index_of(tr["delivery_date"])]
        avail = np.isfinite(P).sum(0) >= MIN_HIST
        scale = np.zeros(NC)
        for j in np.flatnonzero(avail):
            scale[j] = dd_scale(P[np.isfinite(P[:, j]), j])
        ts = time.time()
        rec = {"month": str(ms), "taking_part": [c for c, a in zip(CANDS, avail) if a],
               "scale": dict(zip(CANDS, np.round(scale, 4).tolist()))}
        if avail.any():
            ok = np.isfinite(P[:, avail]).all(1)
            S = np.nan_to_num(P[ok] * scale[None, :])
            m = Alloc14(kappas, seeds, config, device).fit(tr[ok].reset_index(drop=True), S, avail)
            W = m.predict(te)
            rec.update(m.info)
        else:
            W = {k: np.zeros((len(te), NC)) for k in kappas}
            rec.update({"n_days": 0, "note": "no candidate with enough history: sit out"})
        rec["seconds"] = round(time.time() - ts, 1)
        for k, w in W.items():
            d = {"delivery_date": te["delivery_date"].to_numpy(), "kappa": k, "refit_month": str(ms),
                 "n_train": rec["n_days"]}
            for j, c in enumerate(CANDS):
                d[f"w_{c}"] = w[:, j]
                d[f"scale_{c}"] = scale[j]
            rows.append(pd.DataFrame(d))
        recs.append(rec)
        log(f"idea 14 refit {ms}: {rec['n_days']} days, part {rec['taking_part']}, scale {rec['scale']}, "
            f"{rec['seconds']} s")
    out = pd.concat(rows, ignore_index=True)
    out["delivery_date"] = pd.to_datetime(out["delivery_date"])
    DP.assert_build(out["delivery_date"], "idea14 output")
    lock.assert_build_only(out["delivery_date"])
    return out, recs


def day_net(g: pd.DataFrame, tg, cp: CandProfits) -> np.ndarray:
    """Training-objective-style day net of the mix (candidates' own profits, not netted across candidates)."""
    P = np.nan_to_num(cp.P[tg.index_of(g["delivery_date"])])
    A = np.stack([g[f"w_{c}"].to_numpy() * g[f"scale_{c}"].to_numpy() for c in CANDS], 1)
    return (A * P).sum(1)


def choose_kappas(allw: pd.DataFrame, tg, cp: CandProfits, years, kappas=KAPPAS) -> dict:
    out = {}
    for Y in years:
        last_bid = dt.date(Y - 1, 12, 31)
        known = DP._utc_naive(decision_time(last_bid))
        nets = {}
        for k in kappas:
            g = allw[(allw["kappa"] == k) & (allw["delivery_date"].dt.year == Y - 1)
                     & (allw["delivery_date"] <= pd.Timestamp(last_bid - dt.timedelta(days=1)))]
            if not len(g):
                continue
            okp = tg.pub_utc[tg.index_of(g["delivery_date"])] <= known
            nets[k] = float(day_net(g[okp], tg, cp).sum())
        if not nets:
            out[Y] = {"kappa": kappas[len(kappas) // 2], "how": "declared default (no walk-forward year before)"}
            continue
        best = max(kappas, key=lambda k: (nets.get(k, -np.inf), -kappas.index(k)))
        out[Y] = {"kappa": best, "how": f"best net on {Y - 1} walk-forward days", "net_by_kappa": nets}
    return out


def final_rows(allw: pd.DataFrame, choice: dict) -> pd.DataFrame:
    out = pd.concat([allw[(allw["delivery_date"].dt.year == Y) & (allw["kappa"] == ch["kappa"])]
                     for Y, ch in choice.items()], ignore_index=True).sort_values("delivery_date", kind="stable")
    DP.assert_build(out["delivery_date"], "idea14 final")
    lock.assert_build_only(out["delivery_date"])
    assert out["delivery_date"].is_unique
    return out.reset_index(drop=True)
