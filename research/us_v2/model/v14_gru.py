"""V14 GPU runner for the v1 GRU (deep.py, config c1; data tensors loaded once through deep_rolling.load_data).

    python v14_gru.py a [--device cuda]   V14a window ensemble: one GRU per window (1, 2, 3, 5 years and expanding
                                          from 2010-01-01), each trained with recency weights (half-life 12 months,
                                          declared), seed 0; columns w1, w2, w3, w5, wexp and their mean gru_win
    python v14_gru.py b                   V14b quantile heads: the same GRU with a 24 x 11 x 5 pinball head
                                          (levels 0.10, 0.25, 0.50, 0.75, 0.90), seeds 0..2, 3-year window
    python v14_gru.py d                   V14d seeds 5..9 of the deep_c GRU (deep_c holds seeds 0..4), so the
                                          seed ensemble is 10 GRUs; column pred5 (mean of the 5 new seeds)
    python v14_gru.py <mode> --selftest-only   only the injected-lookahead self-test on the real tensors

Every mode first runs the self-test (results/v2/v14/selftest_gru_<mode>.json): (1) a tampered not-yet-public
cell must make the batch guard raise LookaheadError; (2) a tiny fit and prediction for one test day are repeated
after every price after 05:00 on its bid day and every label after the window cut are replaced by noise; the
prediction must not move. Predictions: results/v2/preds/v14<mode>_gru.parquet (rolling.run_rows, restartable).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import rolling as R
import v14_core as C
import deep as D1
import deep_data as DD
import deep_rolling as DR
from timing import LookaheadError

QUANTS = (0.10, 0.25, 0.50, 0.75, 0.90)
WINDOWS = (("w1", 1), ("w2", 2), ("w3", 3), ("w5", 5), ("wexp", None))
V14DIR = R.RESULTS / "v14"


# ============================================================================ models
class WeightedGRU(D1.DeepModel):
    """v1 DeepModel with per-day recency weights in the training loss (validation stays unweighted)."""

    half_life = C.HALF_LIFE_DAYS

    def _prepare_targets(self, keys, d, days):
        k_last = int(days[-1])
        age = np.maximum(k_last - np.arange(d.n_days), 0)
        w = (0.5 ** (age / self.half_life)).astype(np.float32)
        w = w / w[days].mean()
        self.dayw = torch.as_tensor(w, device=self.device)

    def _cleanup(self):
        self.dayw = None

    def _loss(self, net, d, k_idx, xr, mr):
        out = self._forward(net, d, k_idx, xr, mr)
        sel = torch.as_tensor(k_idx, device=self.device)
        y, m = self.Yt[sel], self.Ymt[sel]
        w = m * self.dayw[sel][:, None, None]
        return (self._crit(out, y) * w).sum() / w.sum().clamp(min=1e-6)


class QNet(nn.Module):
    def __init__(self, c, n_points):
        super().__init__()
        self.base = D1.Net(c, n_points)
        H = c["hidden"]
        self.base.head[-1] = nn.Linear(2 * H, 24 * DD.N_Z * len(QUANTS))

    def forward(self, x, lf_mean, xr=None, mr=None):
        z = self.base.encode(x, lf_mean)
        return self.base.head(z).view(z.shape[0], 24, DD.N_Z, len(QUANTS))


class QuantileGRU(D1.DeepModel):
    """The GRU with a pinball-loss head at QUANTS (mean pinball over levels, masked like v1)."""

    def _new_net(self, bias):
        net = QNet(self.c, len(self.cols)).to(self.device)
        with torch.no_grad():
            net.base.head[-1].weight.zero_()
            b = torch.as_tensor(bias, device=self.device)[..., None].expand(24, DD.N_Z, len(QUANTS))
            net.base.head[-1].bias.copy_(b.reshape(-1))
        return net

    def _crit(self, out, y):
        tau = torch.as_tensor(QUANTS, device=out.device)
        u = y[..., None] - out
        return torch.maximum(tau * u, (tau - 1) * u).mean(-1)

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        keys, days, out = self.predict_days(panel)
        k_all = (keys["delivery_date"].to_numpy().astype("datetime64[D]")
                 - np.datetime64(self.data.first_day, "D")).astype(np.int64)
        pos = np.searchsorted(days, k_all)
        q = np.sort(out[pos, keys["slot"].to_numpy(), keys["zc"].to_numpy()], axis=1)    # no crossing
        if not np.isfinite(q).all():
            raise FloatingPointError("non-finite quantile prediction")
        return pd.DataFrame(q, index=panel.index, columns=[f"gq{int(round(t * 100)):02d}" for t in QUANTS])


# ============================================================================ self-test
def selftest(mode: str, panel: pd.DataFrame, data, cfg: dict, device, log) -> dict:
    res = {"mode": mode, "written": C.stamp(), "tests": []}
    # (1) guard: a future RT cell stamped as public must raise
    q = R.quarters()[16]                                            # 2016Q1
    t = C.test_day(q)
    k = data.day_index(t.date())
    cols = torch.arange(min(30, len(data.ptids)), device=device)
    end = int(data.day_start[k])
    old = data.Zp[end - 3, 0, 1]
    data.Zp[end - 3, 0, 1] = int(data.dec_min[k]) - 1
    data._dev = {}
    try:
        DD.batch(data, torch.as_tensor([k]), cols, device, check=True)
        raise C.LeakError("batch guard did not raise on a tampered RT cell")
    except LookaheadError as e:
        res["tests"].append({"test": "guard_tampered_rt_cell", "ok": True, "raised": str(e)[:120]})
    finally:
        data.Zp[end - 3, 0, 1] = old
        data._dev = {}
        torch.cuda.empty_cache() if str(device).startswith("cuda") else None
    # (2) injected lookahead: tiny fit, one day, everything after 05:00 on its bid day replaced by noise
    tiny = {**cfg, "max_epochs": 2, "min_epochs": 1, "patience": 1, "val_days": 20}
    lo, hi = R.window(q[0], 1)
    dd = pd.to_datetime(panel["delivery_date"])
    tr = panel[(dd >= pd.Timestamp(lo)) & (dd <= pd.Timestamp(hi)) & panel["gap"].notna()]
    te = panel[dd == t]
    det = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

    def run(trn):
        m = model_for(mode, tiny, (0,), data, device, log)
        m.fit(trn)
        p = m.predict(te)
        return p.to_numpy() if hasattr(p, "to_numpy") else np.asarray(p)

    p0 = run(tr)
    h0 = int(data.day_start[k]) - DD.WINDOW_H                     # the whole input window of day t and day t
    h1 = int(data.day_start[min(k + 1, data.n_days)])
    dec = int(data.dec_min[k])
    g = np.random.default_rng(3)
    saved = (data.Zv[h0:h1].copy(), data.Gv[h0:h1].copy())
    zl, gl = data.Zp[h0:h1] > dec, data.Gp[h0:h1] > dec            # cells not public at 05:00 on the bid day
    zmask = np.concatenate([np.repeat(zl[..., :1], 3, -1), np.repeat(zl[..., 1:], 3, -1)], -1)
    Zn, Gn = data.Zv[h0:h1], data.Gv[h0:h1]
    Zn[zmask] = g.normal(0, 500, int(zmask.sum())).astype(np.float32)
    Gn[gl] = g.normal(0, 500, int(gl.sum())).astype(np.float32)
    res["perturbed_cells"] = {"zone": int(zmask.sum()), "gen": int(gl.sum())}
    data._dev = {}
    try:
        noisy = C.noise_after(tr, C.label_cut(q), t - pd.Timedelta(days=1), [])
        p1 = run(noisy)
        res["tests"].append(C.check_same(p0, p1, f"gru_{mode}_injected_lookahead {t.date()}"))
        Zn[~zmask] = Zn[~zmask] + 200.0                         # positive control: public cells DO matter
        data._dev = {}
        p2 = run(noisy)
        try:
            C.check_same(p0, p2, "positive control")
            raise C.LeakError("positive control: perturbing public inputs did not move the prediction")
        except C.LeakError as e:
            if str(e).startswith("positive control: perturbing"):
                raise
            res["tests"].append({"test": "positive_control_public_cells_move_prediction", "ok": True,
                                 "max_abs_diff": float(np.max(np.abs(p0 - p2)))})
    finally:
        data.Zv[h0:h1], data.Gv[h0:h1] = saved
        data._dev = {}
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = det
        torch.cuda.empty_cache() if str(device).startswith("cuda") else None
    V14DIR.mkdir(parents=True, exist_ok=True)
    (V14DIR / f"selftest_gru_{mode}.json").write_text(json.dumps(res, indent=1, default=str))
    log(f"self-test gru {mode}: {len(res['tests'])} checks passed")
    return res


def model_for(mode, cfg, seeds, data, device, log):
    cls = {"a": WeightedGRU, "b": QuantileGRU, "d": D1.DeepModel}[mode]
    return cls(cfg, seeds=seeds, data=data, device=device, log=log)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["a", "b", "d"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 warm-up quarters, 2 epochs")
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")
    C.threads(2)
    torch.backends.cudnn.benchmark = True
    log = R.log
    panel = R.load_panel(columns=["delivery_hour", "zone", "delivery_date", "gap"])
    data = DR.load_data(log)
    cfg = D1.resolve_config("c1")
    if a.smoke:
        cfg = {**cfg, "max_epochs": 2, "min_epochs": 1, "patience": 1}
    selftest(a.mode, panel, data, cfg, a.device, log)
    if a.selftest_only:
        return
    name = f"v14{a.mode}_gru" + ("_smoke" if a.smoke else "")
    dd_all = pd.to_datetime(panel["delivery_date"])

    def fp(tr, te, q):
        out = pd.DataFrame(index=te.index)
        if a.mode == "a":
            seen = {}
            for col, yrs in WINDOWS:
                lo, hi = R.window(q[0], yrs) if yrs else (R.DATA_START, R.window(q[0])[1])
                if lo in seen:                                   # same rows as a shorter window (early years)
                    out[col] = out[seen[lo]]
                    continue
                seen[lo] = col
                trw = panel[(dd_all >= pd.Timestamp(lo)) & (dd_all <= pd.Timestamp(hi)) & panel["gap"].notna()]
                R.assert_pre2024(trw["delivery_hour"], "v14a window")
                assert trw["delivery_date"].max() <= C.label_cut(q)
                m = model_for("a", cfg, (0,), data, a.device, log)
                m.fit(trw)
                out[col] = m.predict(te).to_numpy()
            out["gru_win"] = out[[c for c, _ in WINDOWS]].mean(axis=1)
        elif a.mode == "b":
            m = model_for("b", cfg, (0, 1, 2), data, a.device, log)
            m.fit(tr)
            out = out.join(m.predict(te))
        else:
            m = model_for("d", cfg, (5, 6, 7, 8, 9), data, a.device, log)
            m.fit(tr)
            out["pred5"] = m.predict(te).to_numpy()
        if a.device == "cuda":
            torch.cuda.empty_cache()
        return out

    qs = R.quarters()[3:5] if a.smoke else None
    R.run_rows(name, fp, panel, qs=qs, log=log)
    if not a.smoke:
        R.done(f"v14{a.mode}_gru")


if __name__ == "__main__":
    main()
