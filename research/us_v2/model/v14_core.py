"""V14 family, shared parts (OBJECTIVES.md, V14 addendum of 7 Oct; design in research/us_v2/dl_setups_research.md).

    Inputs      feature matrices loaded once as float32 (rolling.load_panel, the day matrix through read_pre2024);
                weather columns only when results/v2_weather_data.done exists; label columns can never be inputs
                (assert_inputs_public raises).
    Networks    BatchedMLP: S seeds trained at once as one batched network on the GPU (V14e, V14f).
    Positions   pair rule of strategies_v2 (load i + supply j when the predicted spread clears both costs), with
                a fixed menu per row chosen per quarter on the trailing 4 out-of-sample quarters
                (rolling.trailing_mask): a setting scores the sum of its 5 best positive pair nets, as in V13;
                the chosen setting's 5 best positive pairs trade the quarter. Ties go to the first setting.
    Lookahead   injected-lookahead harness: labels after the window cut and every input row after the bid day
                are replaced by noise; the prediction for the test day must not move.

HOLDOUT (absolute): nothing on or after 2024-01-01 is read (read_pre2024 / assert_pre2024 everywhere).
"""
from __future__ import annotations

import datetime as dt
import itertools
import os
import time

import numpy as np
import pandas as pd

import rolling as R

ZONES = R.ZONES
NZ = len(ZONES)
PAIRS = list(itertools.combinations(ZONES, 2))
PI = np.array([ZONES.index(i) for i, _ in PAIRS])
PJ = np.array([ZONES.index(j) for _, j in PAIRS])
MAX_PAIRS = 5
WEATHER_FLAG = R.RESULTS.parent / "v2_weather_data.done"
LABELS = {"gap", "y_da_lbmp", "y_rt_lbmp", "label_published_at"}
KEYS = {"bid_date", "delivery_date", "delivery_hour", "zone", "hour"}
HALF_LIFE_DAYS = 365.0                     # declared recency weight (V14a): weight halves every 12 months
CLIP = 250.0


class LeakError(AssertionError):
    pass


def weather_ok() -> bool:
    return WEATHER_FLAG.exists()


def threads(n: int = 2):
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    import torch
    torch.set_num_threads(n)


# ============================================================================ input guards
def assert_inputs_public(cols) -> None:
    """No label, no y__ day label and no published_at stamp may be a model input."""
    bad = [c for c in cols if c in LABELS or str(c).startswith("y__") or str(c).startswith("y_")
           or "published_at" in str(c)]
    if bad:
        raise LeakError(f"label columns offered as inputs: {bad[:5]}")


def panel_features(cols, with_weather: bool) -> list[str]:
    """Numeric per-row inputs of the panel: v1 base features (minus the zone code, which V14f embeds) and the
    border block; weather (wxr_) only behind the flag."""
    base = [c for c in R.base_features() if c != "zone_code"]
    out = [c for c in base if c in cols] + [c for c in cols if c.startswith("bord__")]
    if with_weather:
        out += [c for c in cols if c.startswith("wxr_")]
    out = list(dict.fromkeys(out))
    assert_inputs_public(out)
    return out


def day_feature_columns(cols, with_weather: bool) -> list[str]:
    keep = ("lf__", "lfwow__", "out__", "px__", "rtnow__", "cal__", "bord__") + (("wx__", "wxr__") if with_weather else ())
    out = [c for c in cols if c.startswith(keep)]
    assert_inputs_public(out)
    return out


# ============================================================================ data loaded once
def load_day_matrix(with_weather: bool) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Day matrix (one row per bid day): (keys frame with bid_date and delivery_date, X float32 [n, F], cols)."""
    import pyarrow.parquet as pq
    names = pq.read_schema(str(R.DAYFEATS)).names
    cols = day_feature_columns(names, with_weather)
    d = R.read_pre2024(R.DAYFEATS, "delivery_date", ["bid_date", "delivery_date"] + cols)
    d["delivery_date"] = pd.to_datetime(d["delivery_date"]).dt.normalize()
    d["bid_date"] = pd.to_datetime(d["bid_date"]).dt.normalize()
    if not (d["delivery_date"] - d["bid_date"] == pd.Timedelta(days=1)).all():
        raise LeakError("day matrix: delivery_date is not bid_date + 1")
    d = d.sort_values("delivery_date").reset_index(drop=True)
    X = d[cols].to_numpy(np.float32)
    return d[["bid_date", "delivery_date"]], X, cols


class Scaler:
    """Mean and std per column on training rows only; NaN -> 0 after scaling, clipped at +-8."""

    def __init__(self, X: np.ndarray):
        import warnings
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mu = np.nan_to_num(np.nanmean(X, 0)).astype(np.float32)
            sd = np.nan_to_num(np.nanstd(X, 0))
        self.sd = np.where(sd > 1e-6, sd, 1.0).astype(np.float32)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.mu) / self.sd
        return np.clip(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), -8, 8).astype(np.float32)


def recency_weights(dates: pd.Series | np.ndarray, end, half_life_days: float = HALF_LIFE_DAYS) -> np.ndarray:
    age = (pd.Timestamp(end) - pd.to_datetime(pd.Series(np.asarray(dates)))).dt.days.to_numpy(float)
    w = 0.5 ** (np.maximum(age, 0) / half_life_days)
    return (w / w.mean()).astype(np.float32)


def dense_targets(panel: pd.DataFrame, days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """gap per (delivery day in `days`, local hour slot, zone) -> Y [n, 24, 11] float32 and mask (DST 01:00
    twins averaged, as v1)."""
    dd = pd.to_datetime(panel["delivery_date"]).dt.normalize()
    k = days.get_indexer(dd)
    ok = (k >= 0) & panel["gap"].notna().to_numpy()
    slot = pd.to_datetime(panel["delivery_hour"]).dt.tz_convert(R.TZ).dt.hour.to_numpy()
    zc = pd.Categorical(panel["zone"], categories=ZONES).codes
    tot = np.zeros((len(days), 24, NZ))
    cnt = np.zeros((len(days), 24, NZ))
    g = panel["gap"].to_numpy(float)
    np.add.at(tot, (k[ok], slot[ok], zc[ok]), g[ok])
    np.add.at(cnt, (k[ok], slot[ok], zc[ok]), 1)
    return (tot / np.maximum(cnt, 1)).astype(np.float32), cnt > 0


def gather_rows(out: np.ndarray, days: pd.DatetimeIndex, rows: pd.DataFrame) -> np.ndarray:
    """[n_days, 24, 11, ...] -> per panel row."""
    k = days.get_indexer(pd.to_datetime(rows["delivery_date"]).dt.normalize())
    if (k < 0).any():
        raise KeyError("rows outside the predicted days")
    slot = pd.to_datetime(rows["delivery_hour"]).dt.tz_convert(R.TZ).dt.hour.to_numpy()
    zc = pd.Categorical(rows["zone"], categories=ZONES).codes
    return out[k, slot, zc]


# ============================================================================ batched-seed MLP
def _torch():
    import torch
    return torch


def make_batched_mlp(S: int, d_in: int, hidden: tuple, d_out: int, dropout: float, in_drop: float,
                     n_emb: dict | None = None):
    torch = _torch()
    nn = torch.nn

    class BatchedLinear(nn.Module):
        def __init__(self, i, o):
            super().__init__()
            self.w = nn.Parameter(torch.randn(S, i, o) * (2.0 / (i + o)) ** 0.5)
            self.b = nn.Parameter(torch.zeros(S, 1, o))

        def forward(self, x):                                   # [S, B, i] -> [S, B, o]
            return torch.baddbmm(self.b, x, self.w)

    class BatchedMLP(nn.Module):
        """S independent networks (one per seed) evaluated in one batched matmul chain. Optional learned
        embeddings per seed: n_emb = {name: (cardinality, dim)}; forward(x, ids={name: LongTensor [B]})."""

        def __init__(self):
            super().__init__()
            self.emb = nn.ParameterDict({k: nn.Parameter(torch.randn(S, n, e) * 0.1)
                                         for k, (n, e) in (n_emb or {}).items()})
            dims = [d_in + sum(e for _, e in (n_emb or {}).values())] + list(hidden)
            self.layers = nn.ModuleList([BatchedLinear(a, b) for a, b in zip(dims[:-1], dims[1:])])
            self.out = BatchedLinear(dims[-1], d_out)
            self.drop, self.in_drop = nn.Dropout(dropout), nn.Dropout(in_drop)

        def forward(self, x, ids=None):
            h = self.in_drop(x.expand(S, *x.shape) if x.dim() == 2 else x)
            if self.emb:
                parts = [h] + [self.emb[k][:, ids[k], :] for k in self.emb]
                h = torch.cat(parts, -1)
            for lay in self.layers:
                h = self.drop(torch.nn.functional.gelu(lay(h)))
            return self.out(h)

    return BatchedMLP()


def train_batched(net, n_rows: int, loss_fn, tr_idx: np.ndarray, va_idx: np.ndarray | None, cfg: dict,
                  seed: int, device, fixed_epochs: int | None = None) -> tuple:
    """Mini-batch AdamW on row indices; loss_fn(net, idx_tensor, train) -> per-seed loss [S]. With va_idx: early
    stopping on the mean validation loss over seeds, best state kept; returns (net, best_epoch, history)."""
    import copy
    torch = _torch()
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    max_ep = fixed_epochs or cfg["max_epochs"]
    best, best_ep, best_state, bad, hist = np.inf, 0, None, 0, []
    for ep in range(1, max_ep + 1):
        net.train()
        perm = g.permutation(tr_idx)
        for i in range(0, len(perm), cfg["batch"]):
            b = torch.as_tensor(perm[i:i + cfg["batch"]], device=device)
            loss = loss_fn(net, b, True).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        if va_idx is None:
            continue
        net.eval()
        with torch.no_grad():
            v = float(np.mean([float(loss_fn(net, torch.as_tensor(va_idx[i:i + 4096], device=device), False).mean())
                               for i in range(0, len(va_idx), 4096)]))
        hist.append(v)
        if v < best - 1e-5:
            best, best_ep, bad, best_state = v, ep, 0, copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad >= cfg["patience"] and ep >= cfg.get("min_epochs", 3):
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, best_ep or max_ep, hist


# ============================================================================ pairs and settings
def wide(rows: pd.DataFrame, cols: list[str]) -> dict:
    """Pivot long rows (delivery_hour, zone, ...) to [H, 11] arrays per column, plus gap, hours, dates, costs."""
    w = rows.pivot_table(index="delivery_hour", columns="zone", values=list(dict.fromkeys(cols + ["gap"])),
                         aggfunc="first", dropna=False)
    H = w.index
    R.assert_pre2024(H, "v14 wide")
    out = {c: w[c].reindex(columns=ZONES).to_numpy(float) for c in dict.fromkeys(cols + ["gap"])}
    dd = pd.DatetimeIndex(H.tz_convert(R.TZ).tz_localize(None).normalize())
    sup, lod = R.row_costs(dd.year)
    out.update(hours=H, dd=dd, pc=sup + lod)
    return out


def spread(A: np.ndarray) -> np.ndarray:
    return A[:, PI] - A[:, PJ]


def pair_pnl(W: dict, leg: np.ndarray) -> np.ndarray:
    G = spread(W["gap"])
    ok = np.isfinite(G)
    leg = np.where(ok, np.nan_to_num(leg), 0.0)
    return leg * np.nan_to_num(G) - np.abs(leg) * W["pc"][:, None], leg


def leg_point(W: dict, P: np.ndarray) -> np.ndarray:
    """C rule on a point forecast [H, 11]: +1 when pred_i - pred_j >= both costs, -1 when <= -costs."""
    d = spread(P)
    pc = W["pc"][:, None]
    return np.where(~np.isfinite(d), 0.0, np.where(d >= pc, 1.0, np.where(d <= -pc, -1.0, 0.0)))


def daily(W: dict, pnl: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(pnl, index=W["dd"]).groupby(level=0).sum()


def select_quarterly(W: dict, settings: dict) -> tuple[pd.DataFrame, dict]:
    """settings: name -> leg [H, 55] (a leg may be fractional or zero). Per quarter: the setting with the largest
    sum of its 5 best positive pair nets over the trailing 4 quarters (public by 05:00 on the first bid day),
    then that setting's 5 best positive pairs. Returns long positions (2012 on) and the per-quarter choices."""
    names = list(settings)
    legs, days = {}, {}
    for n in names:
        p, l = pair_pnl(W, settings[n])
        legs[n] = l.astype(np.float32)
        days[n] = daily(W, p)
    dser = pd.Series(W["dd"])
    mwz = np.zeros((len(W["hours"]), NZ))
    ch = {}
    for q in R.quarters():
        qs = ((dser >= pd.Timestamp(q[0])) & (dser <= pd.Timestamp(q[1]))).to_numpy()
        if not qs.any():
            continue
        best = (None, -np.inf, [])
        for n in names:
            D = days[n]
            m = R.trailing_mask(pd.DataFrame({"delivery_date": D.index}), q[0])
            if not m.any():
                continue
            tot = D.to_numpy()[m].sum(0)
            order = [k for k in np.argsort(-tot, kind="stable") if tot[k] > 0][:MAX_PAIRS]
            val = float(tot[order].sum()) if order else 0.0
            if val > best[1]:
                best = (n, val, order)
        n, val, chosen = best
        if n is None:
            ch[q[2]] = {"setting": None, "pairs": [], "how": "no earlier out-of-sample quarter: no pairs"}
            continue
        for k in chosen:
            mwz[qs, PI[k]] += legs[n][qs, k]
            mwz[qs, PJ[k]] -= legs[n][qs, k]
        ch[q[2]] = {"setting": n, "pairs": ["|".join(PAIRS[k]) for k in chosen],
                    "trailing_net": round(val, 1), "how": f"best of {len(names)} settings x 5 of 55 pairs, trailing 4Q"}
    long = pd.DataFrame(mwz, index=W["hours"], columns=ZONES).stack().rename("mw").reset_index()
    long.columns = ["delivery_hour", "zone", "mw"]
    return long, ch


# ============================================================================ injected-lookahead harness
def noise_after(panel: pd.DataFrame, label_cut: pd.Timestamp, feat_cut: pd.Timestamp, feats: list[str],
                seed: int = 7) -> pd.DataFrame:
    """Copy of the panel with every label of delivery dates after label_cut and every input of bid days after
    feat_cut replaced by large noise (the injected lookahead)."""
    g = np.random.default_rng(seed)
    p = panel.copy()
    dd = pd.to_datetime(p["delivery_date"]).dt.normalize()
    lm = (dd > label_cut).to_numpy()
    for c in [c for c in LABELS if c in p.columns and c != "label_published_at"]:
        p.loc[lm, c] = g.normal(0, 500, lm.sum())
    fm = ((dd - pd.Timedelta(days=1)) > feat_cut).to_numpy()
    for c in feats:
        if c in p.columns:
            if p[c].dtype.kind != "f":
                p[c] = p[c].astype(np.float32)
            p.loc[fm, c] = g.normal(0, 500, fm.sum()).astype(p[c].dtype)
    return p


def check_same(a: np.ndarray, b: np.ndarray, what: str, tol: float = 1e-3) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
    if not np.isfinite(diff) or diff > tol * max(1.0, float(np.nanmax(np.abs(a)))):
        raise LeakError(f"{what}: prediction moved by {diff:.4g} after the injected lookahead")
    return {"test": what, "max_abs_diff": diff, "ok": True}


def test_day(q) -> pd.Timestamp:
    return pd.Timestamp(q[0]) + pd.Timedelta(days=10)


def log(*a):
    R.log(*a)


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def label_cut(q) -> pd.Timestamp:
    return pd.Timestamp(q[0] - dt.timedelta(days=R.TRAIN_GAP_DAYS))
