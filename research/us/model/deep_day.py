"""Day-level deep model for ideas 3 to 6 (owner: deep agent; OBJECTIVES addendum 6 Oct 22:30).

One sample per bid day D (decision 05:00 New York time on D), predicting delivery day D+1:
  task "storm": P(the D+1 always-supply book loses more than L), L = the 10th percentile of daily
                always-supply P&L over the training window (ideas 3, 4, 6);
  task "zone":  each zone's D+1 supply P&L (11 outputs), Huber loss with a shrink-to-zero penalty (idea 5).

Day features (build_day_features), every one from timing.features_available_at(05:00 on D):
  wx__<point>__hHH     GFS 2 m temperature of every point for each hour of D+1 (timing.gfs_rule: run
                       issued two days ahead up to 21:00, three days ahead for 22:00 and 23:00); none
                       exist before 25 March 2021, so the weather block is masked there
  lf__<zone>__hHH      operator load forecast for D+1 (newest vintage public at 05:00), 11 zones + NYISO
  lfwow__<zone>__hHH   that forecast minus the forecast for D-6 that was public at 05:00 on D-7
  out__*               scheduled outages active on D+1 in the newest public snapshot, by voltage class
  px__<zone>__dJ__*    day-ahead and real-time prices and congestion for D-J (J = 0..6): daily mean and
                       max of DA, RT and gap, mean DA and RT congestion; RT only from hours already public
  rtnow__<zone>__hHH__*  real-time price and gap of D's hours 00..05 if published by 05:00
  cal__*               calendar of D+1
Labels (never inputs): y__book_supply_pnl, y__zone_supply_pnl__<zone> (1 MW supply in every hour after
the full supply cost, fees.supply_fee_for), y__label_published_at, y__n_rows.

Network: shared encoders with pooling: one MLP applied to every weather point's 24 temperatures (mean
and max pooled over the points present), one to every zone's load forecast and its weekly change, one to
every zone's price history; outages, calendar and the count of real-time hours known enter directly. A
64-unit trunk feeds the storm logit; the zone head is shared across zones (trunk + that zone's price and
load encodings + zone one-hot) with a zero-initialised output, so an untrained zone model predicts 0.
Dropout, weight decay, early stopping on the last 20% of training days (60 to 180), refit on the whole
window for the best epoch count, average of 5 seeds. No hyperparameter was tuned (values below are fixed
before any result).

Holdout: build_day_features calls lock.assert_build_only on the delivery dates before and after building;
the store is panel.LockedStore, which never holds a row on or after 2024-01-01.
"""
from __future__ import annotations

import copy
import datetime as dt
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
import timing as T  # noqa: E402
from common import TZ, ZONES, decision_time, local_midnight  # noqa: E402

LF_ZONES = list(ZONES) + ["NYISO"]
PX_STATS = ["da_mean", "da_max", "rt_mean", "rt_max", "gap_mean", "gap_max", "dacong_mean", "rtcong_mean"]
RTNOW_HOURS = range(6)
OUT_CLASSES = ["345kv_up", "230kv", "138kv", "other_kv"]
N_DAYS_HIST = 7
WX_START = dt.date(2021, 3, 25)
SCALE = 25.0
CONFIG = dict(h_enc=16, h_trunk=64, dropout=0.3, weight_decay=1e-2, lr=1e-3, batch=64, max_epochs=300,
              patience=30, val_frac=0.2, val_min=60, val_max=180, huber=1.0, shrink=0.1, storm_q=0.10)
SEEDS = (0, 1, 2, 3, 4)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", s)


def _points(store) -> list[str]:
    """Every GFS point in the weather table (12 on the real tables), sorted."""
    return sorted(store.table("weather_gfs")["point"].astype(str).unique())


# ----------------------------------------------------------------------------- features
def _day_features(D: dt.date, f: T.Frames, points: list[str], lf_hist: dict) -> dict:
    t = decision_time(D)
    d0 = pd.Timestamp(D)
    d1 = d0 + pd.Timedelta(days=1)
    row: dict = {"bid_date": d0, "delivery_date": d1}
    # weather: every point, every hour of D+1, three-day rule
    wx = T.gfs_rule(f["weather_gfs"], d1)
    if len(wx):
        h = wx["target_hour"].dt.tz_convert(TZ).dt.hour
        piv = wx.assign(h=h.to_numpy()).groupby(["point", "h"])["temperature_2m_c"].mean()
        for (p, hh), v in piv.items():
            row[f"wx__{_slug(p)}__h{hh:02d}"] = float(v)
    # load forecast for D+1 (bid_inputs selection) and its change against the D-6 forecast
    lf = f["load_forecast"]
    lf = lf[local_midnight(lf["target_hour"]) == d1]
    lf = lf.sort_values("published_at", kind="stable").drop_duplicates(["zone", "target_hour"], keep="last")
    cur = {}
    if len(lf):
        h = lf["target_hour"].dt.tz_convert(TZ).dt.hour
        piv = lf.assign(h=h.to_numpy()).groupby(["zone", "h"])["load_forecast_mw"].mean()
        cur = {(z, hh): float(v) for (z, hh), v in piv.items() if z in LF_ZONES}
    lf_hist[D] = cur
    prev = lf_hist.get(D - dt.timedelta(days=7), {})
    for (z, hh), v in cur.items():
        row[f"lf__{_slug(z)}__h{hh:02d}"] = v
        if (z, hh) in prev:
            row[f"lfwow__{_slug(z)}__h{hh:02d}"] = v - prev[(z, hh)]
    # outages active on D+1 in the newest public snapshot
    og = f["outages"]
    if len(og):
        import features_outages as FO
        snap = og[og["snapshot_date"] == og["snapshot_date"].max()]
        lo, hi = pd.Timestamp(d1, tz=TZ), pd.Timestamp(d1 + pd.Timedelta(days=1), tz=TZ)
        so, si = snap["sched_out"], snap["sched_in"]
        act = (so.isna() | (so < hi)) & (si.isna() | (si > lo))
        cls = np.array([FO.voltage_class(str(e)) for e in snap["equipment"]])
        row["out__n_active"] = int(act.sum())
        for c in OUT_CLASSES:
            row[f"out__n_{c}"] = int((act.to_numpy() & (cls == c)).sum())
        row["out__n_in_snapshot"] = len(snap)
        row["out__snapshot_age_days"] = (d0 - pd.Timestamp(snap["snapshot_date"].iloc[0])).days
    # prices: the last 7 local days (D-6 .. D), each component only where published
    da, rt = f["da_prices_zone"], f["rt_prices_zone"]
    da = da[da["zone"].isin(ZONES)]
    rt = rt[rt["zone"].isin(ZONES)]
    lo_day = d0 - pd.Timedelta(days=N_DAYS_HIST - 1)
    da = da.assign(ld=local_midnight(da["delivery_hour"]))
    rt = rt.assign(ld=local_midnight(rt["delivery_hour"]))
    da, rt = da[(da["ld"] >= lo_day) & (da["ld"] <= d0)], rt[(rt["ld"] >= lo_day) & (rt["ld"] <= d0)]
    g = da[["delivery_hour", "zone", "ld", "da_lbmp"]].merge(rt[["delivery_hour", "zone", "rt_lbmp"]],
                                                            on=["delivery_hour", "zone"], how="inner")
    g["gap"] = g["rt_lbmp"] - g["da_lbmp"]
    stats = {
        "da_mean": da.groupby(["ld", "zone"])["da_lbmp"].mean(), "da_max": da.groupby(["ld", "zone"])["da_lbmp"].max(),
        "rt_mean": rt.groupby(["ld", "zone"])["rt_lbmp"].mean(), "rt_max": rt.groupby(["ld", "zone"])["rt_lbmp"].max(),
        "gap_mean": g.groupby(["ld", "zone"])["gap"].mean(), "gap_max": g.groupby(["ld", "zone"])["gap"].max(),
        "dacong_mean": da.groupby(["ld", "zone"])["da_congestion"].mean(),
        "rtcong_mean": rt.groupby(["ld", "zone"])["rt_congestion"].mean()}
    for st, s in stats.items():
        for (ld, z), v in s.items():
            j = (d0 - ld).days
            row[f"px__{_slug(z)}__d{j}__{st}"] = float(v)
    # real-time hours of D already public
    now = g[g["ld"] == d0]
    hr = now["delivery_hour"].dt.tz_convert(TZ).dt.hour.to_numpy()
    for (z, hh, r, gp) in zip(now["zone"], hr, now["rt_lbmp"], now["gap"]):
        if hh in RTNOW_HOURS:
            row[f"rtnow__{_slug(z)}__h{hh:02d}__rt"] = float(r)
            row[f"rtnow__{_slug(z)}__h{hh:02d}__gap"] = float(gp)
    row["rtnow__n_hours"] = int(len(set(hr)))
    # calendar of D+1
    from deep_data import nerc_holidays
    dow, doy = d1.dayofweek, d1.dayofyear
    row.update({"cal__dow_sin": math.sin(2 * math.pi * dow / 7), "cal__dow_cos": math.cos(2 * math.pi * dow / 7),
                "cal__doy_sin": math.sin(2 * math.pi * doy / 365.25), "cal__doy_cos": math.cos(2 * math.pi * doy / 365.25),
                "cal__weekend": float(dow >= 5), "cal__holiday": float(d1.date() in nerc_holidays([d1.year]))})
    return row


def feature_columns(points: list[str]) -> list[str]:
    """Every feature column in a fixed order (absent values are NaN)."""
    cols = [f"wx__{_slug(p)}__h{h:02d}" for p in points for h in range(24)]
    cols += [f"lf__{_slug(z)}__h{h:02d}" for z in LF_ZONES for h in range(24)]
    cols += [f"lfwow__{_slug(z)}__h{h:02d}" for z in LF_ZONES for h in range(24)]
    cols += ["out__n_active"] + [f"out__n_{c}" for c in OUT_CLASSES] + ["out__n_in_snapshot", "out__snapshot_age_days"]
    cols += [f"px__{_slug(z)}__d{j}__{s}" for z in ZONES for j in range(N_DAYS_HIST) for s in PX_STATS]
    cols += [f"rtnow__{_slug(z)}__h{h:02d}__{k}" for z in ZONES for h in RTNOW_HOURS for k in ("rt", "gap")]
    cols += ["rtnow__n_hours", "cal__dow_sin", "cal__dow_cos", "cal__doy_sin", "cal__doy_cos", "cal__weekend",
             "cal__holiday"]
    return cols


def labels(store, d_first: pd.Timestamp, d_last: pd.Timestamp) -> pd.DataFrame:
    """Per delivery date: always-supply book P&L and per-zone supply P&L (1 MW every hour with a settled
    gap, full supply cost), and when the last of its prices was public."""
    import fees
    pz = store.table("prices_zone")
    lo, hi = pd.Timestamp(d_first, tz=TZ), pd.Timestamp(d_last + pd.Timedelta(days=1), tz=TZ)
    x = pz[(pz["delivery_hour"] >= lo) & (pz["delivery_hour"] < hi) & pz["zone"].isin(ZONES)].copy()
    lock.assert_build_only(x["delivery_hour"])
    x["dd"] = local_midnight(x["delivery_hour"])
    x["gap"] = x["rt_lbmp"] - x["da_lbmp"]
    ok = x["gap"].notna()
    x["pnl"] = np.where(ok, -x["gap"] - fees.supply_fee_for(x["dd"]), 0.0)
    z = x.groupby(["dd", "zone"])["pnl"].sum().unstack("zone").reindex(columns=ZONES)
    out = pd.DataFrame({"delivery_date": z.index})
    out["y__book_supply_pnl"] = z.sum(axis=1).to_numpy()
    for zn in ZONES:
        out[f"y__zone_supply_pnl__{_slug(zn)}"] = z[zn].to_numpy()
    out["y__label_published_at"] = x.groupby("dd")["published_at"].max().reindex(z.index).to_numpy()
    out["y__n_rows"] = x[ok].groupby("dd").size().reindex(z.index).fillna(0).astype(int).to_numpy()
    return out


def build_day_features(bid_days: list[dt.date], store, log=None, with_labels: bool = True) -> pd.DataFrame:
    """One row per bid day D (delivery D+1), every feature from features_available_at(05:00 on D)."""
    log = log or (lambda *_: None)
    lock.assert_build_only([d + dt.timedelta(days=1) for d in bid_days])
    points = _points(store)
    lf_hist: dict = {}
    days = sorted(bid_days)
    # the D-7 forecasts the weekly change needs, for the first week
    pre = [days[0] - dt.timedelta(days=k) for k in range(7, 0, -1)]
    rows, t0 = [], time.time()
    for i, D in enumerate(pre + days):
        t = decision_time(D)
        f = T.features_available_at(t, store, lookback_days=N_DAYS_HIST + 1, include_gen=False)
        T.assert_no_lookahead(f, t)                          # every row used was public by 05:00 on D
        r = _day_features(D, f, points, lf_hist)
        if D >= days[0]:
            rows.append(r)
        if i and i % 200 == 0:
            log(f"day features: {i}/{len(pre) + len(days)} bid days, {time.time() - t0:.0f} s")
    cols = feature_columns(points)
    extra = sorted(set().union(*[r.keys() for r in rows]) - set(cols) - {"bid_date", "delivery_date"})
    if extra:
        raise ValueError(f"unexpected feature columns: {extra[:5]}")
    df = pd.DataFrame(rows)
    df = df.reindex(columns=["bid_date", "delivery_date"] + cols).astype({c: float for c in cols})
    if with_labels:
        lab = labels(store, df["delivery_date"].min(), df["delivery_date"].max())
        df = df.merge(lab, on="delivery_date", how="left", validate="one_to_one")
    lock.assert_build_only(df["delivery_date"])
    return df


# ----------------------------------------------------------------------------- tensors
class Layout:
    """Column groups of the day-feature matrix, in network order."""

    def __init__(self, cols: list[str]):
        pts = sorted({c.split("__")[1] for c in cols if c.startswith("wx__")}, key=lambda p: cols.index(f"wx__{p}__h00"))
        self.wx = [[f"wx__{p}__h{h:02d}" for h in range(24)] for p in pts]
        self.lf = [[f"lf__{_slug(z)}__h{h:02d}" for h in range(24)] + [f"lfwow__{_slug(z)}__h{h:02d}" for h in range(24)]
                   for z in LF_ZONES]
        self.px = [[f"px__{_slug(z)}__d{j}__{s}" for j in range(N_DAYS_HIST) for s in PX_STATS]
                   + [f"rtnow__{_slug(z)}__h{h:02d}__{k}" for h in RTNOW_HOURS for k in ("rt", "gap")] for z in ZONES]
        self.glob = [c for c in cols if c.startswith(("out__", "cal__"))] + ["rtnow__n_hours"]
        self.price_like = {c for grp in self.px for c in grp}

    def all(self) -> list[str]:
        return [c for g in self.wx for c in g] + [c for g in self.lf for c in g] + [c for g in self.px for c in g] + self.glob


class Scaler:
    """Per-column centring and scaling fitted on training days only; prices enter as asinh(x/25)."""

    def __init__(self, df: pd.DataFrame, lay: Layout):
        self.lay = lay
        X = self._raw(df)
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)   # columns empty in this window
            self.mu = np.nanmean(X, 0)
            sd = np.nanstd(X, 0)
        self.sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
        self.mu = np.where(np.isfinite(self.mu), self.mu, 0.0)

    def _raw(self, df):
        cols = self.lay.all()
        X = np.array(df.reindex(columns=cols).to_numpy(float), dtype=float, copy=True)   # writable
        pl = np.array([c in self.lay.price_like for c in cols])
        X[:, pl] = np.arcsinh(X[:, pl] / SCALE)
        return X

    def transform(self, df) -> tuple[np.ndarray, np.ndarray]:
        X = self._raw(df)
        M = np.isfinite(X)
        Z = np.clip(np.where(M, (X - self.mu) / self.sd, 0.0), -10, 10)
        return Z.astype(np.float32), M


def tensors(df: pd.DataFrame, sc: Scaler, device):
    import torch
    lay = sc.lay
    Z, M = sc.transform(df)
    n = len(df)
    nw, nl, npx = len(lay.wx) * 24, len(lay.lf) * 48, len(lay.px) * len(lay.px[0])
    o = 0
    wx = Z[:, o:o + nw].reshape(n, len(lay.wx), 24); wxm = M[:, o:o + nw].reshape(n, len(lay.wx), 24); o += nw
    lf = Z[:, o:o + nl].reshape(n, len(lay.lf), 48); lfm = M[:, o:o + nl].reshape(n, len(lay.lf), 48); o += nl
    px = Z[:, o:o + npx].reshape(n, len(lay.px), -1); pxm = M[:, o:o + npx].reshape(n, len(lay.px), -1); o += npx
    gl = Z[:, o:]
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=device)  # noqa: E731
    return {"wx": t(wx), "wx_pt": t(wxm.any(2).astype(np.float32)), "wx_frac": t(wxm.mean(2, dtype=np.float32)),
            "lf": t(lf), "lf_pt": t(lfm.any(2).astype(np.float32)), "lf_frac": t(lfm.mean(2, dtype=np.float32)),
            "px": t(px), "px_frac": t(pxm.mean(2, dtype=np.float32)), "gl": t(gl)}


# ----------------------------------------------------------------------------- network
def _net(c: dict, lay: Layout):
    import torch
    import torch.nn as nn

    class Enc(nn.Module):
        def __init__(self, d_in, h, p):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(d_in + 1, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, h), nn.GELU())

        def forward(self, x, frac, avail):                     # x [B,N,d], frac/avail [B,N]
            e = self.f(torch.cat([x, frac[..., None]], -1)) * avail[..., None]
            n = avail.sum(1, keepdim=True).clamp(min=1)
            mean = e.sum(1) / n
            mx = torch.where(avail[..., None] > 0, e, torch.full_like(e, -1e4)).amax(1)
            mx = torch.where(avail.sum(1, keepdim=True) > 0, mx, torch.zeros_like(mx))
            return e, torch.cat([mean, mx, (avail.sum(1, keepdim=True) > 0).float()], -1)

    class DayNet(nn.Module):
        def __init__(self):
            super().__init__()
            h, p = c["h_enc"], c["dropout"]
            self.wx = Enc(24, h, p)
            self.lf = Enc(48, h, p)
            self.px = Enc(len(lay.px[0]), h, p)
            self.gl = nn.Sequential(nn.Linear(len(lay.glob), h), nn.GELU())
            d = 3 * (2 * h + 1) + h
            self.trunk = nn.Sequential(nn.Dropout(p), nn.Linear(d, c["h_trunk"]), nn.GELU(), nn.Dropout(p))
            self.storm = nn.Linear(c["h_trunk"], 1)
            nz = len(ZONES)
            self.zone = nn.Sequential(nn.Linear(c["h_trunk"] + 2 * h + nz, 32), nn.GELU(), nn.Dropout(p), nn.Linear(32, 1))
            self.register_buffer("eye", torch.eye(nz))

        def forward(self, x):
            _, wx = self.wx(x["wx"], x["wx_frac"], x["wx_pt"])
            lfe, lf = self.lf(x["lf"], x["lf_frac"], x["lf_pt"])
            pxe, px = self.px(x["px"], x["px_frac"], (x["px_frac"] > 0).float())
            h = self.trunk(torch.cat([wx, lf, px, self.gl(x["gl"])], -1))
            B, nz = h.shape[0], len(ZONES)
            zin = torch.cat([h[:, None].expand(B, nz, -1), pxe, lfe[:, :nz], self.eye[None].expand(B, nz, nz)], -1)
            return self.storm(h).squeeze(-1), self.zone(zin).squeeze(-1)

    return DayNet()


# ----------------------------------------------------------------------------- model
class DeepDayModel:
    """fit(days) / predict(days) on rows of the day-feature matrix (labels needed for fit only)."""

    def __init__(self, task: str, config: dict | None = None, seeds=SEEDS, device=None, log=None):
        import torch
        assert task in ("storm", "zone")
        self.task, self.c, self.seeds = task, {**CONFIG, **(config or {})}, tuple(seeds)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.log = log or (lambda *_: None)
        self.nets, self.info = [], {}

    def _targets(self, df):
        if self.task == "storm":
            return (df["y__book_supply_pnl"].to_numpy(float) < self.L).astype(np.float32)[:, None]
        return df[[f"y__zone_supply_pnl__{_slug(z)}" for z in ZONES]].to_numpy(float).astype(np.float32) / self.s

    def fit(self, days: pd.DataFrame) -> None:
        import torch
        lock.assert_build_only(days["delivery_date"])
        t0 = time.time()
        days = days[days["y__n_rows"] > 0].sort_values("delivery_date").reset_index(drop=True)
        cols = [c for c in days.columns if "__" in c and not c.startswith("y__")]
        self.lay = Layout(cols)
        self.sc = Scaler(days, self.lay)
        book = days["y__book_supply_pnl"].to_numpy(float)
        self.L = float(np.quantile(book, self.c["storm_q"]))
        zp = days[[f"y__zone_supply_pnl__{_slug(z)}" for z in ZONES]].to_numpy(float)
        self.s = float(max(1.4826 * np.median(np.abs(zp - np.median(zp))), 1.0))
        X = tensors(days, self.sc, self.device)
        Y = torch.as_tensor(self._targets(days), device=self.device)
        n = len(days)
        nv = int(min(self.c["val_max"], max(self.c["val_min"], round(self.c["val_frac"] * n))))
        tr, va = np.arange(n - nv), np.arange(n - nv, n)
        self.nets, seeds = [], []
        for sd in self.seeds:
            net, ep, best = self._train(sd, X, Y, tr, va, self.c["max_epochs"])
            net, _, _ = self._train(sd, X, Y, np.arange(n), None, max(ep, 1))
            net.eval()
            self.nets.append(net)
            seeds.append({"seed": sd, "best_epoch": ep, "best_val": best})
        self.info = {"task": self.task, "n_days": n, "n_val": nv, "L": self.L, "s": self.s,
                     "storm_rate_train": float((book < self.L).mean()), "seeds": seeds,
                     "train_first": str(days["delivery_date"].min().date()),
                     "train_last": str(days["delivery_date"].max().date()), "fit_s": round(time.time() - t0, 1)}

    def _loss(self, net, X, Y, idx, val=False):
        import torch
        import torch.nn.functional as F
        sel = torch.as_tensor(idx, device=self.device)
        xb = {k: v[sel] for k, v in X.items()}
        logit, zone = net(xb)
        if self.task == "storm":
            return F.binary_cross_entropy_with_logits(logit, Y[sel, 0])
        l = F.huber_loss(zone, Y[sel], delta=self.c["huber"])
        return l if val else l + self.c["shrink"] * (zone ** 2).mean()

    def _train(self, seed, X, Y, tr, va, max_epochs):
        import torch
        torch.manual_seed(seed)
        net = _net(self.c, self.lay).to(self.device)
        with torch.no_grad():
            net.zone[-1].weight.zero_()
            net.zone[-1].bias.zero_()
            r = float(np.clip(Y[:, 0].float().mean().item(), 1e-3, 1 - 1e-3)) if self.task == "storm" else 0.5
            net.storm.weight.zero_()
            net.storm.bias.fill_(math.log(r / (1 - r)))
        opt = torch.optim.AdamW(net.parameters(), lr=self.c["lr"], weight_decay=self.c["weight_decay"])
        g = np.random.default_rng(seed)
        best, best_ep, state, bad = math.inf, 0, None, 0
        for ep in range(1, max_epochs + 1):
            net.train()
            perm = g.permutation(tr)
            for i in range(0, len(perm), self.c["batch"]):
                loss = self._loss(net, X, Y, perm[i:i + self.c["batch"]])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            if va is None:
                continue
            net.eval()
            with torch.no_grad():
                v = float(self._loss(net, X, Y, va, val=True))
            if v < best - 1e-6:
                best, best_ep, bad, state = v, ep, 0, copy.deepcopy(net.state_dict())
            else:
                bad += 1
                if bad >= self.c["patience"]:
                    break
        if state is not None:
            net.load_state_dict(state)
        return net, best_ep, best

    def predict(self, days: pd.DataFrame):
        """storm: Series P(book loses more than L), index = days.index. zone: DataFrame of predicted
        supply P&L (USD) per zone."""
        import torch
        lock.assert_build_only(days["delivery_date"])
        X = tensors(days, self.sc, self.device)
        outs = []
        with torch.no_grad():
            for net in self.nets:
                net.eval()
                logit, zone = net(X)
                outs.append(torch.sigmoid(logit).cpu().numpy() if self.task == "storm" else zone.cpu().numpy() * self.s)
        m = np.mean(outs, 0)
        if self.task == "storm":
            return pd.Series(m, index=days.index, name="p_storm")
        return pd.DataFrame(m, index=days.index, columns=ZONES)
