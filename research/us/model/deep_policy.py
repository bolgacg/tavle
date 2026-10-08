"""Deep policies for ideas 11 and 12 (owner: policy agent; OBJECTIVES addendum 6 Oct 2026 23:20, commit a156551).

Both read the day-level feature matrix (results/day_features_2020_2023.parquet, deep_day.build_day_features:
one row per bid day D for delivery day D+1, every value public by 05:00 New York time on D) and decide one
position per zone for the whole of D+1 (held in every hour of that zone-day).

Idea 11, learned allocator. Four experts per zone-day: always supply (-1 MW every hour), sit out (0), virtual
load (+1 MW every hour), and the storm-day zone pair, a book-level expert: load in N.Y.C. and LONGIL, supply in
the two upstate zones (of A to F) whose supply legs did best on the training window's storm days (book P&L of
always-supply below its 10th percentile), chosen once a year from the first refit of that year. A gate network
puts weights on the experts: a softmax over (supply, out, load) per zone, and one weight on the pair per day
(the pair is a book, so the same weight in its four legs, sit-out elsewhere); each zone's four weights sum to 1.
It is trained to maximise the realised mixed profit of the next day, sum over zones of the weights times each
expert's own profit after its own costs, minus lambda times the squared shortfall below zero. The position is the
weighted mix: (1 - pi) * (w_load - w_supply) + pi * pair_leg. Training on the mixed expert profit charges the full
cost of both legs when supply and load are mixed in one zone; the scored net position pays less, so the training
objective is the conservative side.

Idea 12, end-to-end policy. The same encoder outputs a position a in [-1, 1] per zone (tanh): profit
a * sum_h(gap) - |a| * n_hours * cost(side), supply cost when a < 0 and load cost when a > 0, minus the same
shortfall penalty and MU * sum_z a^2 (a small L2 so it stays out unless confident).

Costs (storm_value.py's tables, exactly): supply legs Schedule 1 + FERC + uplift bound, load legs Schedule 1 + FERC.
Scale: day profits enter the loss divided by s = 1.4826 * MAD of the training window's daily always-supply book P&L.
Objective per day d: J_d = B_d / s - lambda * max(0, -B_d / s)^2 [- MU * sum_z a_z^2 for idea 12]; loss = -mean J.

Network (both ideas; declared before any result, not tuned): per-group encoders shared over their members, as in
deep_day: one MLP over each weather point's 24 hourly temperatures, one over each zone's load forecast and its
weekly change (48), one over each zone's price history and real-time hours already public (68); members pooled by
mean and max; outages and calendar enter a linear layer. Trunk 64 units. Zone head: trunk + that zone's price and
load encodings + a learned zone embedding (4) -> 32 -> 3 gate logits (idea 11) or 1 position (idea 12), output
layer zero-initialised (idea 12 starts flat; idea 11 starts uniform, pair weight 0.25). Pair weight from the trunk.
Dropout 0.3, AdamW weight decay 1e-2, lr 1e-3, batch 64 days, gradient norm clipped to 1 per model.

Walk-forward: a refit on the 1st of every month, trained on delivery dates from 2020-01-01 up to two days before
the month whose prices were all public by 05:00 on the month's first bid day (walkforward.py's rule, whole days).
Early stopping on the last 20 percent of the training days (60 to 180 days, time-ordered), patience 20, at most
200 epochs (cut from 30 and 300 before any walk-forward result, for CPU time: the GPU is shared), then a refit on
the whole window for each model's best epoch count. Average of 5 seeds. Trained on the CPU, one thread per idea.
lambda in LAMBDAS (3 values): every lambda is run; the lambda used in year Y is the one whose walk-forward
positions earned the most net profit (full cost) in year Y-1 on days whose prices were public by 05:00 on the
last bid day of Y-1 (ties to the first value). For 2021 that is a walk-forward over July to December 2020 (refits
from July 2020, trained on 2020 days only), run for this choice and not scored.
Weather inputs are masked before delivery 2021-03-25, and any feature column with fewer than 30 values in a
training window is masked for that refit (so a refit that never saw weather does not read it).

HOLDOUT: every loader filters delivery dates before 2024-01-01 at load and asserts it (here and lock.py).
"""
from __future__ import annotations

import datetime as dt
import itertools
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "pipeline"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lock  # noqa: E402
from common import TZ, ZONES, decision_time  # noqa: E402

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:            # the laptop has no torch; everything but the network still imports
    torch = nn = F = None

NYISO_HOME = Path.home() / "nyiso-us"
RESULTS = NYISO_HOME / "results"
PARQUET = NYISO_HOME / "parquet"
FEATS = RESULTS / "day_features_2020_2023.parquet"
END = pd.Timestamp("2024-01-01", tz=TZ)
END_NAIVE = pd.Timestamp("2024-01-01")
WX_START = pd.Timestamp("2021-03-25")          # first delivery date with GFS values (OBJECTIVES addendum)
TRAIN_START = pd.Timestamp("2020-01-01")
TRAIN_GAP_DAYS = 2

# storm_value.py, exactly (equal to fees.SUPPLY_RATES / fees.RATES; a test checks)
COST = {2020: 0.0862 + 0.010 + 0.007, 2021: 0.0757 + 0.015 + 0.004, 2022: 0.0853 + 0.016 + 0.003,
        2023: 0.1066 + 0.017 + 0.026}
LOADCOST = {y: c - u for (y, c), u in zip(COST.items(), [0.007, 0.004, 0.003, 0.026])}   # load legs pay no uplift
UPSTATE = ["WEST", "GENESE", "CENTRL", "NORTH", "MHK VL", "CAPITL"]                    # zones A to F
DOWNSTATE = ["N.Y.C.", "LONGIL"]
PAIRS = list(itertools.combinations(UPSTATE, 2))
LF_ZONES = list(ZONES) + ["NYISO"]
NZ = len(ZONES)
PRICE_SCALE = 25.0

LAMBDAS = (0.001, 0.01, 0.1)
SEEDS = (0, 1, 2, 3, 4)
CONFIG = dict(h_enc=16, h_trunk=64, h_head=32, zone_emb=4, dropout=0.3, weight_decay=1e-2, lr=1e-3, batch=64,
              max_epochs=200, patience=20, val_frac=0.2, val_min=60, val_max=180, clip=1.0, mu=0.01,
              storm_q=0.10, min_col_obs=30)


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", s)


# ============================================================================ holdout-safe loading
def assert_build(values, what: str = "") -> None:
    s = pd.Series(values).dropna()
    if not len(s):
        return
    mx = pd.Timestamp(s.max())
    ok = mx < (END if mx.tzinfo is not None else END_NAIVE)
    if not ok:
        raise AssertionError(f"HOLDOUT BREACH {what}: {mx}")


def _read_build(path: Path, timecol: str, columns=None, extra=None) -> pd.DataFrame:
    import pyarrow as pa
    import pyarrow.dataset as ds
    d = ds.dataset(str(path), format="parquet")
    typ = d.schema.field(timecol).type
    bound = pa.scalar(END if getattr(typ, "tz", None) else END_NAIVE, type=typ)
    flt = ds.field(timecol) < bound
    if extra is not None:
        flt = flt & extra
    df = d.to_table(filter=flt, columns=columns).to_pandas()
    assert_build(df[timecol], f"{path}:{timecol}")
    lock.assert_build_only(df[timecol])
    return df


def load_day_features(path: Path = FEATS) -> pd.DataFrame:
    """Day-feature matrix, delivery dates before 2024 only, weather masked before 2021-03-25."""
    df = _read_build(Path(path), "delivery_date")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"]).dt.normalize()
    df = df.sort_values("delivery_date", kind="stable").reset_index(drop=True)
    assert df["delivery_date"].is_unique
    return mask_weather(df)


def mask_weather(df: pd.DataFrame) -> pd.DataFrame:
    wx = [c for c in df.columns if c.startswith("wx__")]
    early = (pd.to_datetime(df["delivery_date"]) < WX_START).to_numpy()
    if wx and early.any():
        df = df.copy()
        df.loc[early, wx] = np.nan
    return df


def load_prices(path: Path = PARQUET / "prices_zone.parquet") -> pd.DataFrame:
    import pyarrow.dataset as ds
    px = _read_build(Path(path), "delivery_hour", ["delivery_hour", "zone", "da_lbmp", "rt_lbmp", "published_at"],
                     ds.field("zone").isin(ZONES))
    return px


# ============================================================================ targets
class Targets:
    """Per delivery date and zone: G = sum over hours of (RT - DA), N = hours with a settled gap, and the
    experts' realised profit after full costs. pub = when the day's last price was public."""

    def __init__(self, px: pd.DataFrame):
        x = px[px["zone"].isin(ZONES)]
        assert_build(x["delivery_hour"], "prices")
        lock.assert_build_only(x["delivery_hour"])
        dd = x["delivery_hour"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
        gap = x["rt_lbmp"] - x["da_lbmp"]
        ok = gap.notna()
        days = pd.DatetimeIndex(sorted(dd.unique()))
        G = gap[ok].groupby([dd[ok], x["zone"][ok]]).sum().unstack("zone")
        N = gap[ok].groupby([dd[ok], x["zone"][ok]]).size().unstack("zone")
        self.days = days
        self.G = G.reindex(index=days, columns=ZONES).fillna(0.0).to_numpy(float)
        self.N = N.reindex(index=days, columns=ZONES).fillna(0.0).to_numpy(float)
        self.pub = x["published_at"].groupby(dd).max().reindex(days)
        self.pub_utc = _utc_naive(self.pub)                      # datetime64, NaT never counts as public
        self.year = days.year.to_numpy()
        self.supc = pd.Series(self.year).map(COST).to_numpy(float)
        self.loadc = pd.Series(self.year).map(LOADCOST).to_numpy(float)
        self.Ps = -self.G - self.N * self.supc[:, None]          # always supply, per zone-day
        self.Pl = self.G - self.N * self.loadc[:, None]          # virtual load, per zone-day
        self.nrows = self.N.sum(1)

    def index_of(self, dates) -> np.ndarray:
        return self.days.get_indexer(pd.DatetimeIndex(pd.to_datetime(dates)).normalize())

    def net(self, rows: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Realised zone-day P&L of positions pos [n, 11] held every hour, full cost by side."""
        G, N = self.G[rows], self.N[rows]
        c = np.where(pos < 0, self.supc[rows, None], self.loadc[rows, None])
        return pos * G - np.abs(pos) * N * c


def pair_vector(pair) -> np.ndarray:
    v = np.zeros(NZ)
    for z in DOWNSTATE:
        v[ZONES.index(z)] = 1.0
    for z in pair:
        v[ZONES.index(z)] = -1.0
    return v


def choose_pair(tg: Targets, rows: np.ndarray, q: float = CONFIG["storm_q"]) -> tuple[tuple[str, str], dict]:
    """The two upstate zones whose supply legs did best on the storm days of the given (training) rows; storm day =
    always-supply book P&L at or below its q-quantile over those rows. Only the upstate legs differ between pairs."""
    book = tg.Ps[rows].sum(1)
    L = float(np.quantile(book, q))
    st = rows[book <= L]
    dn = tg.Pl[st][:, [ZONES.index(z) for z in DOWNSTATE]].sum()
    vals = [float(dn + tg.Ps[st][:, [ZONES.index(a), ZONES.index(b)]].sum()) for a, b in PAIRS]
    i = int(np.argmax(vals))
    return PAIRS[i], {"storm_L": L, "storm_days": int(len(st)), "pair_book_on_storm_days": vals[i]}


def pair_profit(tg: Targets, pair) -> np.ndarray:
    v = pair_vector(pair)
    return (np.where(v > 0, tg.Pl, 0.0) + np.where(v < 0, tg.Ps, 0.0)).sum(1)


# ============================================================================ inputs
class Prep:
    """Column groups (as deep_day.Layout) and per-column scaling fitted on the training rows only. Prices enter as
    asinh(x/25). A column with fewer than min_obs training values is masked in every row of this refit."""

    def __init__(self, train: pd.DataFrame, min_obs: int = CONFIG["min_col_obs"]):
        cols = feature_columns(train)
        pts = []
        for c in cols:
            if c.startswith("wx__"):
                p = c.split("__")[1]
                if p not in pts:
                    pts.append(p)
        self.wx = [[f"wx__{p}__h{h:02d}" for h in range(24)] for p in pts]
        self.lf = [[f"lf__{slug(z)}__h{h:02d}" for h in range(24)] + [f"lfwow__{slug(z)}__h{h:02d}" for h in range(24)]
                   for z in LF_ZONES]
        z0 = slug(ZONES[0])
        sfx_px = [c[len(f"px__{z0}__"):] for c in cols if c.startswith(f"px__{z0}__")]
        sfx_rt = [c[len(f"rtnow__{z0}__"):] for c in cols if c.startswith(f"rtnow__{z0}__")]
        self.px = [[f"px__{slug(z)}__{s}" for s in sfx_px] + [f"rtnow__{slug(z)}__{s}" for s in sfx_rt] for z in ZONES]
        used = {c for grp in self.wx + self.lf + self.px for c in grp}
        self.glob = [c for c in cols if c not in used]          # outages, calendar, rtnow__n_hours, anything new
        self.price_like = np.array([c.startswith(("px__", "rtnow__")) and c != "rtnow__n_hours" for c in self.all()])
        X = self._raw(train, mask=False)
        cnt = np.isfinite(X).sum(0)
        self.keep = cnt >= min_obs
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
        self.mu = np.where(np.isfinite(mu), mu, 0.0)
        self.sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
        self.dims = {"wx": 24, "lf": 48, "px": len(self.px[0]), "gl": max(len(self.glob), 1),
                     "n_wx": len(self.wx), "n_lf": len(self.lf)}

    def all(self) -> list[str]:
        return [c for g in self.wx for c in g] + [c for g in self.lf for c in g] + [c for g in self.px for c in g] + self.glob

    def _raw(self, df: pd.DataFrame, mask: bool = True) -> np.ndarray:
        X = np.array(df.reindex(columns=self.all()).to_numpy(float), dtype=float, copy=True)
        X[:, self.price_like] = np.arcsinh(X[:, self.price_like] / PRICE_SCALE)
        if mask:
            X[:, ~self.keep] = np.nan
        return X

    def arrays(self, df: pd.DataFrame) -> dict:
        X = self._raw(df)
        M = np.isfinite(X)
        Z = np.clip(np.where(M, (X - self.mu) / self.sd, 0.0), -10, 10).astype(np.float32)
        n = len(df)
        nw, nl, npx = len(self.wx) * 24, len(self.lf) * 48, len(self.px) * len(self.px[0])
        o = 0
        out = {}
        for key, k, width, ng in (("wx", nw, 24, len(self.wx)), ("lf", nl, 48, len(self.lf)),
                                  ("px", npx, len(self.px[0]), len(self.px))):
            out[key] = Z[:, o:o + k].reshape(n, ng, width)
            out[key + "_frac"] = M[:, o:o + k].reshape(n, ng, width).mean(2, dtype=np.float32)
            o += k
        gl = Z[:, o:]
        out["gl"] = gl if gl.shape[1] else np.zeros((n, 1), np.float32)
        return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if "__" in c and not c.startswith("y__")]
    assert not any(c.startswith("y__") for c in cols)
    return cols


# ============================================================================ batched network (K independent models)
def _uniform(gen, shape, bound):
    return (torch.rand(*shape, generator=gen) * 2 - 1) * bound


if nn is not None:
    class BLinear(nn.Module):
        """K independent linear layers; model k's initial weights come from its own generator only."""

        def __init__(self, gens, din, dout):
            super().__init__()
            b = 1.0 / math.sqrt(din)
            self.w = nn.Parameter(torch.stack([_uniform(g, (din, dout), b) for g in gens]))
            self.b = nn.Parameter(torch.stack([_uniform(g, (dout,), b) for g in gens]))

        def forward(self, x):                     # [K, ..., din]
            K, sh = x.shape[0], x.shape
            y = torch.baddbmm(self.b[:, None, :], x.reshape(K, -1, sh[-1]), self.w)
            return y.reshape(*sh[:-1], self.w.shape[-1])

    class BEnc(nn.Module):
        """One MLP shared over a group's members (points or zones), pooled by mean and max over members present."""

        def __init__(self, gens, d, h, p):
            super().__init__()
            self.l1, self.l2, self.drop = BLinear(gens, d + 1, h), BLinear(gens, h, h), nn.Dropout(p)

        def forward(self, x, frac):               # x [K,B,N,d], frac [K,B,N]
            avail = (frac > 0).float()
            e = F.gelu(self.l2(self.drop(F.gelu(self.l1(torch.cat([x, frac[..., None]], -1))))))
            e = e * avail[..., None]
            n = avail.sum(-1, keepdim=True)
            mean = e.sum(-2) / n.clamp(min=1)
            mx = torch.where(avail[..., None] > 0, e, torch.full_like(e, -1e4)).amax(-2)
            mx = torch.where(n > 0, mx, torch.zeros_like(mx))
            return e, torch.cat([mean, mx, (n > 0).float()], -1)

    class PolicyNet(nn.Module):
        def __init__(self, kind: int, gens, dims: dict, c: dict):
            super().__init__()
            h, p, e = c["h_enc"], c["dropout"], c["zone_emb"]
            self.kind = kind
            self.wx, self.lf, self.px = BEnc(gens, 24, h, p), BEnc(gens, dims["lf"], h, p), BEnc(gens, dims["px"], h, p)
            self.gl = BLinear(gens, dims["gl"], h)
            self.t1 = BLinear(gens, 3 * (2 * h + 1) + h, c["h_trunk"])
            self.zemb = nn.Parameter(torch.stack([torch.randn(NZ, e, generator=g) * 0.1 for g in gens]))
            self.z1 = BLinear(gens, c["h_trunk"] + 2 * h + e, c["h_head"])
            self.z2 = BLinear(gens, c["h_head"], 3 if kind == 11 else 1)
            self.drop = nn.Dropout(p)
            with torch.no_grad():
                self.z2.w.zero_()
                self.z2.b.zero_()
            if kind == 11:
                self.pair = BLinear(gens, c["h_trunk"], 1)
                with torch.no_grad():
                    self.pair.w.zero_()
                    self.pair.b.fill_(math.log(0.25 / 0.75))     # uniform over the four experts at the start

        def forward(self, xb):
            _, wxp = self.wx(xb["wx"], xb["wx_frac"])
            lfe, lfp = self.lf(xb["lf"], xb["lf_frac"])
            pxe, pxp = self.px(xb["px"], xb["px_frac"])
            g = F.gelu(self.gl(xb["gl"]))
            hh = self.drop(F.gelu(self.t1(self.drop(torch.cat([wxp, lfp, pxp, g], -1)))))
            K, B, H = hh.shape
            zin = torch.cat([hh[:, :, None].expand(K, B, NZ, H), pxe, lfe[:, :, :NZ],
                             self.zemb[:, None].expand(K, B, NZ, self.zemb.shape[-1])], -1)
            out = self.z2(self.drop(F.gelu(self.z1(zin))))
            if self.kind == 11:
                return out, self.pair(hh)[..., 0]                 # gate logits [K,B,Z,3], pair logit [K,B]
            return torch.tanh(out[..., 0]), None                  # position [K,B,Z]


def day_objective(kind: int, out, aux, T: dict, lam, mu: float, s: float):
    """J per model and day [K,B]: scaled profit minus lambda * squared shortfall (minus mu * sum a^2 for idea 12).
    T holds the batch's targets [K,B,...]: G, N, supc, loadc (idea 12) or Ps, Pl, Pp (idea 11)."""
    if kind == 12:
        a = out
        cost = T["N"] * (T["supc"][..., None] * F.relu(-a) + T["loadc"][..., None] * F.relu(a))
        prof = (a * T["G"] - cost).sum(-1) / s
        reg = mu * (a ** 2).sum(-1)
    else:
        w = torch.softmax(out, -1)
        pi = torch.sigmoid(aux)
        zone = (w[..., 0] * T["Ps"] + w[..., 2] * T["Pl"]).sum(-1)
        prof = ((1 - pi) * zone + pi * T["Pp"]) / s
        reg = 0.0
    return prof - lam[:, None] * F.relu(-prof) ** 2 - reg


def mixed_weights(out, aux):
    """Idea 11: effective weights over (supply, out, load, pair) per zone [K,B,Z,4]; they sum to 1."""
    w = torch.softmax(out, -1)
    pi = torch.sigmoid(aux)[..., None, None]
    return torch.cat([(1 - pi) * w, pi.expand(*w.shape[:-1], 1)], -1)


def _clip_per_model(params, K: int, max_norm: float):
    sq = None
    for p in params:
        if p.grad is not None:
            v = p.grad.reshape(K, -1).pow(2).sum(1)
            sq = v if sq is None else sq + v
    if sq is None:
        return
    scale = (max_norm / (sq.sqrt() + 1e-6)).clamp(max=1.0)
    for p in params:
        if p.grad is not None:
            p.grad.mul_(scale.view(K, *([1] * (p.grad.dim() - 1))))


# ============================================================================ the model (one refit)
class PolicyModel:
    """fit(train_days, targets) / predict(days) for idea 11 or 12; every lambda in `lambdas` x every seed is one of K
    models trained side by side (independent parameters, losses summed)."""

    def __init__(self, kind: int, lambdas=LAMBDAS, seeds=SEEDS, config: dict | None = None, device=None, log=None):
        assert kind in (11, 12)
        self.kind, self.lambdas, self.seeds = kind, tuple(lambdas), tuple(seeds)
        self.c = {**CONFIG, **(config or {})}
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.log = log or (lambda *_: None)
        self.K = len(self.lambdas) * len(self.seeds)
        self.lam_k = np.repeat(np.array(self.lambdas, float), len(self.seeds))
        self.seed_k = np.tile(np.array(self.seeds, int), len(self.lambdas))
        self.info: dict = {}

    def _tensors(self, arrs: dict) -> dict:
        return {k: torch.as_tensor(np.ascontiguousarray(v), device=self.device) for k, v in arrs.items()}

    def _targets(self, tg: Targets, rows: np.ndarray) -> dict:
        d = {"G": tg.G[rows], "N": tg.N[rows], "supc": tg.supc[rows], "loadc": tg.loadc[rows],
             "Ps": tg.Ps[rows], "Pl": tg.Pl[rows], "Pp": pair_profit(tg, self.pair)[rows] if self.kind == 11
             else np.zeros(len(rows))}
        return {k: torch.as_tensor(np.asarray(v, np.float32), device=self.device) for k, v in d.items()}

    def fit(self, days: pd.DataFrame, tg: Targets, pair=None) -> None:
        lock.assert_build_only(days["delivery_date"])
        t0 = time.time()
        days = days.sort_values("delivery_date", kind="stable").reset_index(drop=True)
        rows = tg.index_of(days["delivery_date"])
        assert (rows >= 0).all(), "training day without prices"
        self.pair = tuple(pair) if pair is not None else None
        if self.kind == 11:
            assert self.pair is not None
        self.prep = Prep(days, self.c["min_col_obs"])
        X = self._tensors(self.prep.arrays(days))
        T = self._targets(tg, rows)
        book = tg.Ps[rows].sum(1)
        self.s = float(max(1.4826 * np.median(np.abs(book - np.median(book))), 1.0))
        n = len(days)
        nv = int(min(self.c["val_max"], max(self.c["val_min"], round(self.c["val_frac"] * n))))
        nv = min(nv, n // 2)
        tr, va = np.arange(n - nv), np.arange(n - nv, n)
        _, best_ep, best_val = self._train(X, T, tr, va, self.c["max_epochs"])
        self.net, _, _ = self._train(X, T, np.arange(n), None, int(best_ep.max()), snapshot_at=best_ep)
        self.net.eval()
        self.info = {"kind": self.kind, "n_days": n, "n_val": nv, "s": self.s, "pair": self.pair,
                     "train_first": str(days["delivery_date"].min().date()),
                     "train_last": str(days["delivery_date"].max().date()),
                     "best_epochs": {str(l): best_ep[self.lam_k == l].astype(int).tolist() for l in self.lambdas},
                     "best_val_J": {str(l): np.round(best_val[self.lam_k == l], 4).tolist() for l in self.lambdas},
                     "masked_columns": int((~self.prep.keep).sum()), "fit_s": round(time.time() - t0, 1)}

    def _train(self, X, T, tr, va, max_epochs, snapshot_at=None):
        c, K = self.c, self.K
        gens = [torch.Generator().manual_seed(int(s)) for s in self.seed_k]
        net = PolicyNet(self.kind, gens, self.prep.dims, c).to(self.device)
        torch.manual_seed(int(self.seed_k[0]) + 1000 * len(tr))          # dropout masks
        opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
        params = list(net.parameters())
        rngs = [np.random.default_rng(int(s)) for s in self.seed_k]
        lam = torch.as_tensor(self.lam_k, dtype=torch.float32, device=self.device)
        best = np.full(K, -np.inf)
        best_ep = np.zeros(K, int)
        bad = np.zeros(K, int)
        active = np.ones(K, bool)
        keep = {k: p.detach().clone() for k, p in net.named_parameters()}
        for ep in range(1, max_epochs + 1):
            net.train()
            perms = np.stack([r.permutation(tr) for r in rngs])
            for i in range(0, len(tr), c["batch"]):
                idx = torch.as_tensor(perms[:, i:i + c["batch"]], device=self.device)
                xb = {k: v[idx] for k, v in X.items()}
                tb = {k: v[idx] for k, v in T.items()}
                out, aux = net(xb)
                J = day_objective(self.kind, out, aux, tb, lam, c["mu"], self.s)
                loss = -J.mean(1).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                _clip_per_model(params, K, c["clip"])
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
                out, aux = net({k: v[idx] for k, v in X.items()})
                v = day_objective(self.kind, out, aux, {k: t[idx] for k, t in T.items()}, lam, c["mu"],
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
        """Per lambda: position [n, 11] (seed average) and, for idea 11, weights [n, 11, 4] over
        (supply, out, load, pair)."""
        lock.assert_build_only(days["delivery_date"])
        X = self._tensors(self.prep.arrays(days))
        n = len(days)
        with torch.no_grad():
            idx = torch.as_tensor(np.tile(np.arange(n), (self.K, 1)), device=self.device)
            out, aux = self.net({k: v[idx] for k, v in X.items()})
            if self.kind == 11:
                W = mixed_weights(out, aux).cpu().numpy()                 # [K,n,Z,4]
            else:
                A = out.cpu().numpy()                                    # [K,n,Z]
        res = {}
        pv = pair_vector(self.pair) if self.kind == 11 else None
        for l in self.lambdas:
            m = self.lam_k == l
            if self.kind == 11:
                w = W[m].mean(0)
                res[l] = {"position": -w[..., 0] + w[..., 2] + w[..., 3] * pv[None, :], "weights": w}
            else:
                res[l] = {"position": A[m].mean(0), "weights": None}
        return res


# ============================================================================ walk-forward
def month_starts(first: dt.date, last: dt.date) -> list[dt.date]:
    out, m = [], dt.date(first.year, first.month, 1)
    while m <= last:
        out.append(m)
        m = dt.date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    return out


def _utc_naive(ts) -> np.ndarray:
    """tz-aware Series or Timestamp -> UTC datetime64[ns] (NaT kept), for safe array comparisons."""
    if isinstance(ts, pd.Timestamp):
        return np.datetime64(ts.tz_convert("UTC").tz_localize(None).as_unit("ns"))
    return pd.Series(ts).dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").to_numpy()


def train_rows(feats: pd.DataFrame, tg: Targets, ms: dt.date, gap_days: int = TRAIN_GAP_DAYS,
               known_by: pd.Timestamp | None = None) -> np.ndarray:
    """Positions in feats of the days a refit on the 1st of month ms may learn from: delivery date from 2020-01-01 to
    two days before ms, every price of the day public by 05:00 on the month's first bid day (ms - 1)."""
    cut = pd.Timestamp(ms - dt.timedelta(days=gap_days))
    known = _utc_naive(decision_time(ms - dt.timedelta(days=1)) if known_by is None else known_by)
    dd = pd.DatetimeIndex(feats["delivery_date"])
    r = tg.index_of(dd)
    ok = r >= 0
    rr = np.maximum(r, 0)
    pub_ok = np.where(ok, tg.pub_utc[rr] <= known, False)
    m = ok & (dd >= TRAIN_START) & (dd <= cut) & pub_ok & (np.where(ok, tg.nrows[rr], 0) > 0)
    return np.flatnonzero(np.asarray(m))


def walk_forward(feats: pd.DataFrame, tg: Targets, kind: int, first: dt.date, last: dt.date, lambdas=LAMBDAS,
                 seeds=SEEDS, config=None, device=None, log=print, gap_days: int = TRAIN_GAP_DAYS,
                 known_shift: pd.Timedelta | None = None):
    """Monthly refits over [first, last]: (long positions for every lambda, refit records). gap_days and known_shift
    exist only so the lookahead test can open the filter; the run uses the defaults."""
    lock.assert_build_only([first, last])
    feats = mask_weather(feats.sort_values("delivery_date", kind="stable").reset_index(drop=True))
    rows, recs, pairs = [], [], {}
    for ms in month_starts(first, last):
        me = min(dt.date(ms.year + (ms.month == 12), ms.month % 12 + 1, 1) - dt.timedelta(days=1), last)
        known = decision_time(ms - dt.timedelta(days=1)) + (known_shift or pd.Timedelta(0))
        ti = train_rows(feats, tg, ms, gap_days, known)
        tr = feats.iloc[ti]
        te = feats[(feats["delivery_date"] >= pd.Timestamp(ms)) & (feats["delivery_date"] <= pd.Timestamp(me))]
        if gap_days >= TRAIN_GAP_DAYS and known_shift is None:
            assert tr["delivery_date"].max() <= pd.Timestamp(ms - dt.timedelta(days=TRAIN_GAP_DAYS))
            assert tr["delivery_date"].max() < te["delivery_date"].min()
        pair, pinfo = None, None
        if kind == 11:
            if ms.year not in pairs:            # once a year, from the year's first refit window
                pairs[ms.year] = choose_pair(tg, tg.index_of(tr["delivery_date"]))
            pair, pinfo = pairs[ms.year]
        ts = time.time()
        m = PolicyModel(kind, lambdas, seeds, config, device)
        m.fit(tr, tg, pair)
        pr = m.predict(te)
        sec = round(time.time() - ts, 1)
        dd = te["delivery_date"].to_numpy()
        bd = te["bid_date"].to_numpy() if "bid_date" in te.columns else dd - np.timedelta64(1, "D")
        for l, r in pr.items():
            P = r["position"]
            W = r["weights"]
            for zi, z in enumerate(ZONES):
                d = {"delivery_date": dd, "bid_date": bd, "zone": z, "idea": kind, "lam": l,
                     "position": P[:, zi].astype(float), "refit_month": str(ms)}
                if W is not None:
                    d.update({"w_supply": W[:, zi, 0], "w_out": W[:, zi, 1], "w_load": W[:, zi, 2],
                              "w_pair": W[:, zi, 3], "pair_up1": pair[0], "pair_up2": pair[1]})
                rows.append(pd.DataFrame(d))
        rec = {"idea": kind, "month": str(ms), **m.info, "seconds": sec}
        if pinfo is not None:
            rec["pair_choice"] = pinfo
        recs.append(rec)
        log(f"idea {kind} refit {ms}: {m.info['n_days']} days to {m.info['train_last']} (val {m.info['n_val']}), "
            f"s {m.s:,.0f}, epochs {m.info['best_epochs']}, {sec} s" + (f", pair {pair}" if pair else ""))
    out = pd.concat(rows, ignore_index=True)
    out["delivery_date"] = pd.to_datetime(out["delivery_date"])
    assert_build(out["delivery_date"], "policy output")
    lock.assert_build_only(out["delivery_date"])
    return out, recs


def choose_lambdas(allpos: pd.DataFrame, tg: Targets, years, lambdas=LAMBDAS) -> dict:
    """Per scored year Y: the lambda whose walk-forward positions earned the most net profit at full cost in Y-1, on
    delivery dates up to 30 December of Y-1 whose prices were all public by 05:00 on 31 December (ties: first)."""
    out = {}
    for Y in years:
        last_bid = dt.date(Y - 1, 12, 31)
        known = decision_time(last_bid)
        cut = pd.Timestamp(last_bid - dt.timedelta(days=1))
        nets = {}
        for l in lambdas:
            g = allpos[(allpos["lam"] == l) & (allpos["delivery_date"].dt.year == Y - 1)
                       & (allpos["delivery_date"] <= cut)]
            if not len(g):
                continue
            P = g.pivot(index="delivery_date", columns="zone", values="position").reindex(columns=ZONES)
            r = tg.index_of(P.index)
            okp = tg.pub_utc[r] <= _utc_naive(known)
            nets[l] = float(tg.net(r[okp], P.to_numpy(float)[okp]).sum())
        if not nets:
            out[Y] = {"lam": lambdas[len(lambdas) // 2], "how": "declared default (no walk-forward year before)"}
            continue
        best = max(lambdas, key=lambda l: (nets.get(l, -np.inf), -lambdas.index(l)))
        out[Y] = {"lam": best, "how": f"best net on {Y - 1} walk-forward days", "net_by_lambda": nets}
    return out


# ============================================================================ storm_value.py's scoring, exactly
BANK = 500_000


def score(g, mw):
    """storm_value.score, copied verbatim. g: hourly rows with gap, year, ddate; mw: signed MW per row."""
    cost = np.where(mw < 0, g.year.map(COST), g.year.map(LOADCOST))
    pnl = mw * g.gap - np.abs(mw) * cost
    dd = pd.Series(pnl, index=g.index).groupby(g.ddate).sum()
    cum = dd.cumsum(); mdd = float((cum - cum.cummax()).min()); mwh = float(np.abs(mw).sum())  # noqa: E702
    return {'net_usd': round(float(dd.sum())), 'usd_per_mwh': round(float(dd.sum()) / max(mwh, 1), 2),
            'sharpe': round(float(dd.mean() / dd.std() * np.sqrt(365)), 2) if dd.std() > 0 else None,
            'max_drawdown_usd': round(mdd), 'worst_day_usd': round(float(dd.min())),
            'return_on_500k_pct': round(100 * float(dd.sum()) / BANK * 365 / dd.size, 1), 'mwh_per_day': round(mwh / dd.size),
            'net_without_best_3_days': round(float(dd.sort_values().iloc[:-3].sum())), 'best_day_usd': round(float(dd.max()))}


def hourly_frame(px: pd.DataFrame) -> pd.DataFrame:
    g = px[px["zone"].isin(ZONES)].copy()
    assert_build(g["delivery_hour"], "prices")
    g["gap"] = g["rt_lbmp"] - g["da_lbmp"]
    g["ddate"] = g["delivery_hour"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    g["year"] = g["ddate"].dt.year
    return g.reset_index(drop=True)


def hourly_mw(h: pd.DataFrame, pos: pd.DataFrame) -> np.ndarray:
    """Zone-day positions (delivery_date, zone, position) spread over every hour of the zone-day; else 0."""
    p = pos.set_index(["delivery_date", "zone"])["position"]
    mi = pd.MultiIndex.from_arrays([h["ddate"], h["zone"]])
    return np.nan_to_num(p.reindex(mi).to_numpy(float))
