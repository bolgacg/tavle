"""V14e and V14f on the GPU: feed-forward networks, five seeds trained at once as one batched network.

    python v14_mlp.py e [--device cuda]   V14e Lago-style multi-output DNN: one network per window maps the day
                                          matrix of the bid day (every public block: load forecast, outages,
                                          prices, real-time-so-far, border, calendar; weather behind the flag) to
                                          all 24 hours x 11 zones of D+1. Menu (2): mlp_nowx, mlp_wx (no weather
                                          flag: mlp_nowx only)
    python v14_mlp.py f [--device cuda]   V14f one global row model for all zones with learned zone and hour
                                          embeddings on the panel features (v1 base, border; weather behind the
                                          flag). Menu (2): glob_mse (MSE) and glob_dfl (MSE plus a wrong-side
                                          penalty max(0, -pred * gap) on the scaled gap: a profit-shaped loss with
                                          a tail penalty, research rank 6)

Each menu entry is predicted on every quarter; v14_positions.py picks one per quarter on the trailing 4 quarters.
Training: 3-year window (rolling.window), the last 90 days held for early stopping (mean validation loss over
the seeds), then the seeds are refitted on the whole window for the best epoch count. Targets clipped at 250 and
scaled by their training std. Self-test first: results/v2/v14/selftest_mlp_<mode>.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

import rolling as R
import v14_core as C

SEEDS = 5
CFG_E = dict(hidden=(256, 128), dropout=0.2, in_drop=0.1, lr=1e-3, weight_decay=1e-3, batch=32, max_epochs=200,
             patience=20, min_epochs=10, val_days=90)
CFG_F = dict(hidden=(128, 64), dropout=0.1, in_drop=0.05, lr=1e-3, weight_decay=1e-4, batch=2048, max_epochs=40,
             patience=5, min_epochs=3, val_days=90)
DFL_LAMBDA = 1.0


# ============================================================================ V14e: day matrix -> 24 x 11
class DayModel:
    def __init__(self, keys: pd.DataFrame, X: np.ndarray, cols: list[str], panel: pd.DataFrame, device):
        C.assert_inputs_public(cols)
        self.keys, self.X, self.cols, self.device = keys, X, cols, device
        self.days = pd.DatetimeIndex(keys["delivery_date"])
        self.Y, self.M = C.dense_targets(panel, self.days)
        self.wx_cols = np.array([c.startswith(("wx__", "wxr__")) for c in cols])

    def fit_predict(self, q, te: pd.DataFrame, variant: str, cfg=CFG_E, seed=0, fixed_epochs=None) -> np.ndarray:
        lo, hi = R.window(q[0])
        dmask = (self.days >= pd.Timestamp(lo)) & (self.days <= pd.Timestamp(hi)) & self.M.any((1, 2))
        tr_days = np.flatnonzero(dmask)
        assert self.days[tr_days].max() <= C.label_cut(q)
        te_days = np.unique(self.days.get_indexer(pd.to_datetime(te["delivery_date"]).dt.normalize()))
        if (te_days < 0).any():
            raise KeyError("test day without a day-matrix row")
        use = ~self.wx_cols if variant == "mlp_nowx" else np.ones(len(self.cols), bool)
        Xs = self.X[:, use]
        sc = C.Scaler(Xs[tr_days])
        idx = np.concatenate([tr_days, te_days])
        Xg = torch.as_tensor(sc(Xs[idx]), device=self.device)
        Y = np.clip(self.Y[tr_days], -C.CLIP, C.CLIP)
        Mk = self.M[tr_days]
        s = float(max(np.std(Y[Mk]), 1.0))
        Yg = torch.as_tensor(np.nan_to_num(Y / s).reshape(len(tr_days), -1), device=self.device)
        Mg = torch.as_tensor(Mk.reshape(len(tr_days), -1).astype(np.float32), device=self.device)
        bias = torch.as_tensor(np.nan_to_num((Y * Mk).sum(0) / np.maximum(Mk.sum(0), 1) / s).reshape(-1),
                               device=self.device)

        def new():
            net = C.make_batched_mlp(SEEDS, Xg.shape[1], cfg["hidden"], 24 * C.NZ, cfg["dropout"], cfg["in_drop"])
            net = net.to(self.device)
            with torch.no_grad():
                net.out.w.mul_(0.1)
                net.out.b.copy_(bias.expand_as(net.out.b))
            return net

        def loss(net, b, train):
            o = net(Xg[b])                                           # [S, B, 264]
            m = Mg[b]
            return (((o - Yg[b]) ** 2) * m).sum((1, 2)) / m.sum().clamp(min=1)

        n = len(tr_days)
        if fixed_epochs is None:
            nv = min(cfg["val_days"], max(5, n // 10))
            torch.manual_seed(seed)
            _, best_ep, _ = C.train_batched(new(), n, loss, np.arange(n - nv), np.arange(n - nv, n), cfg, seed,
                                            self.device)
        else:
            best_ep = fixed_epochs
        torch.manual_seed(seed + 1)
        net, _, _ = C.train_batched(new(), n, loss, np.arange(n), None, cfg, seed + 1, self.device,
                                    fixed_epochs=best_ep)
        with torch.no_grad():
            o = net(Xg[n:]).mean(0).cpu().numpy().reshape(len(te_days), 24, C.NZ) * s
        full = np.full((len(self.days), 24, C.NZ), np.nan, np.float32)
        full[te_days] = o
        self.last_info = {"best_epoch": int(best_ep), "s": s, "n_days": int(n), "n_inputs": int(Xg.shape[1])}
        return C.gather_rows(full, self.days, te)


# ============================================================================ V14f: global row model
class GlobalModel:
    def __init__(self, panel: pd.DataFrame, feats: list[str], device):
        C.assert_inputs_public(feats)
        self.p, self.feats, self.device = panel, feats, device
        self.dd = pd.to_datetime(panel["delivery_date"]).dt.normalize()
        self.zc = pd.Categorical(panel["zone"], categories=C.ZONES).codes.astype(np.int64)
        self.hr = pd.to_datetime(panel["delivery_hour"]).dt.tz_convert(R.TZ).dt.hour.to_numpy().astype(np.int64)
        self.X = panel[feats].to_numpy(np.float32)

    def fit_predict(self, q, te: pd.DataFrame, variant: str, cfg=CFG_F, seed=0, fixed_epochs=None) -> np.ndarray:
        lo, hi = R.window(q[0])
        trm = ((self.dd >= pd.Timestamp(lo)) & (self.dd <= pd.Timestamp(hi)) & self.p["gap"].notna()).to_numpy()
        tr = np.flatnonzero(trm)
        assert self.dd.iloc[tr].max() <= C.label_cut(q)
        tei = self.p.index.get_indexer(te.index)
        if (tei < 0).any():
            raise KeyError("test rows not in the panel")
        sc = C.Scaler(self.X[tr])
        idx = np.concatenate([tr, tei])
        Xg = torch.as_tensor(sc(self.X[idx]), device=self.device)
        ids = {"zone": torch.as_tensor(self.zc[idx], device=self.device),
               "hour": torch.as_tensor(self.hr[idx], device=self.device)}
        y = np.clip(self.p["gap"].to_numpy(float)[tr], -C.CLIP, C.CLIP)
        s = float(max(np.std(y), 1.0))
        yg = torch.as_tensor((y / s).astype(np.float32), device=self.device)
        mu = float(np.mean(y / s))
        lam = DFL_LAMBDA if variant == "glob_dfl" else 0.0

        def new():
            net = C.make_batched_mlp(SEEDS, Xg.shape[1], cfg["hidden"], 1, cfg["dropout"], cfg["in_drop"],
                                     n_emb={"zone": (C.NZ, 8), "hour": (24, 4)}).to(self.device)
            with torch.no_grad():
                net.out.b.fill_(mu)
            return net

        def loss(net, b, train):
            o = net(Xg[b], {k: v[b] for k, v in ids.items()}).squeeze(-1)      # [S, B]
            yt = yg[b]
            l = (o - yt) ** 2
            if lam:
                l = l + lam * torch.relu(-o * yt)
            return l.mean(1)

        n = len(tr)
        order = np.argsort(self.dd.iloc[tr].to_numpy(), kind="stable")
        if fixed_epochs is None:
            cut = pd.Timestamp(hi) - pd.Timedelta(days=cfg["val_days"])
            va = order[self.dd.iloc[tr].to_numpy()[order] > cut]
            trn = order[self.dd.iloc[tr].to_numpy()[order] <= cut]
            torch.manual_seed(seed)
            _, best_ep, _ = C.train_batched(new(), n, loss, trn, va, cfg, seed, self.device)
        else:
            best_ep = fixed_epochs
        torch.manual_seed(seed + 1)
        net, _, _ = C.train_batched(new(), n, loss, np.arange(n), None, cfg, seed + 1, self.device,
                                    fixed_epochs=best_ep)
        with torch.no_grad():
            out = []
            for i in range(n, len(idx), 65536):
                b = torch.arange(i, min(i + 65536, len(idx)), device=self.device)
                out.append(net(Xg[b], {k: v[b] for k, v in ids.items()}).squeeze(-1).mean(0).cpu().numpy())
        self.last_info = {"best_epoch": int(best_ep), "s": s, "n_rows": int(n), "n_inputs": int(Xg.shape[1])}
        return np.concatenate(out) * s


# ============================================================================ self-test
def selftest(mode, build, panel, extra_noise, device, log) -> dict:
    """build(panel_like, noisy: bool) -> (model, variants). Prediction for one test day with 3 fixed epochs, then
    again after labels past the window cut and inputs of later bid days are replaced by noise."""
    q = R.quarters()[16]
    t = C.test_day(q)
    res = {"mode": mode, "written": C.stamp(), "tests": []}
    try:
        C.assert_inputs_public(["px__x", "y__book_supply_pnl"])
        raise C.LeakError("accepted")
    except C.LeakError as e:
        if str(e) == "accepted":
            raise
        res["tests"].append({"test": "label_input_refused", "ok": True})
    det = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    dd = pd.to_datetime(panel["delivery_date"])
    out = []
    for noisy in (False, True):
        p = C.noise_after(panel, C.label_cut(q), t - pd.Timedelta(days=1), extra_noise) if noisy else panel
        m, variants = build(p, noisy)
        te = p[dd == t]
        out.append({v: m.fit_predict(q, te, v, seed=0, fixed_epochs=3) for v in variants})
        del m
    for v in out[0]:
        res["tests"].append(C.check_same(out[0][v], out[1][v], f"mlp_{mode}_{v}_injected_lookahead {t.date()}"))
    torch.backends.cudnn.benchmark = det
    d = R.RESULTS / "v14"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"selftest_mlp_{mode}.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"self-test mlp {mode}: {len(res['tests'])} checks passed")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["e", "f"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest-only", action="store_true")
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    C.threads(2)
    torch.backends.cudnn.benchmark = True
    wx = C.weather_ok()
    log = R.log
    log(f"weather flag {'present' if wx else 'MISSING: weather inputs off'}")
    import pyarrow.parquet as pq
    if a.mode == "e":
        panel = R.load_panel(columns=["delivery_hour", "zone", "delivery_date", "gap"])
        keys, X, cols = C.load_day_matrix(wx)
        variants = ["mlp_nowx", "mlp_wx"] if wx else ["mlp_nowx"]
        log(f"day matrix {X.shape}, panel {panel.shape}")

        def build(p, noisy):
            Xn = X
            if noisy:                                                      # day rows after the test bid day
                t = C.test_day(R.quarters()[16])
                late = (keys["bid_date"] > t - pd.Timedelta(days=1)).to_numpy()
                Xn = X.copy()
                Xn[late] = np.random.default_rng(5).normal(0, 500, Xn[late].shape).astype(np.float32)
            return DayModel(keys, Xn, cols, p, a.device), variants
        extra = []
    else:
        names = pq.read_schema(str(R.PANEL)).names
        feats = C.panel_features(names, wx)
        panel = R.load_panel(columns=list(dict.fromkeys(["delivery_hour", "zone", "delivery_date", "gap"] + feats)))
        for c in feats:
            panel[c] = panel[c].astype(np.float32)
        variants = ["glob_mse", "glob_dfl"]
        log(f"panel {panel.shape}, {len(feats)} inputs")

        def build(p, noisy):
            return GlobalModel(p, feats, a.device), variants
        extra = feats
    selftest(a.mode, build, panel, extra, a.device, log)
    if a.selftest_only:
        return
    model, _ = build(panel, False)
    name = f"v14{a.mode}_mlp" + ("_smoke" if a.smoke else "")
    info = {}

    def fp(tr, te, q):
        out = pd.DataFrame(index=te.index)
        for v in variants:
            out[v] = model.fit_predict(q, te, v, fixed_epochs=2 if a.smoke else None)
            info[f"{q[2]}_{v}"] = model.last_info
        if a.device == "cuda":
            torch.cuda.empty_cache()
        return out

    qs = R.quarters()[3:5] if a.smoke else None
    R.run_rows(name, fp, panel, qs=qs, log=log)
    (R.RESULTS / "v14").mkdir(parents=True, exist_ok=True)
    (R.RESULTS / "v14" / f"{name}_info.json").write_text(json.dumps(info, indent=1))
    if not a.smoke:
        R.done(f"v14{a.mode}_mlp")


if __name__ == "__main__":
    main()
