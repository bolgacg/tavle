"""V13 inputs: every input public by 05:00 on bid day D, as dense tensors loaded once (OBJECTIVES.md, V13).

Groups (each can be removed for the ablations, and is dropped at random in training, "input dropout"):
  price    zone prices for the 168 hours before D+1: DA and RT LBMP, energy, loss, congestion, gap, masks
           (the v1 deep_data arrays and publication rule; energy = LBMP - loss - congestion). Never removed.
  gen      generator points (prices_gen, point_type 'gen'; the 4 border proxies are excluded here): DA and RT
           loss and congestion per point and hour, through a learned projection; a point is visible only in the
           hours it published and only once public (mask), so points that did not exist are masked.
  border   the bord__ block of the day matrix (4 borders x 7 days x 8 statistics), public by 05:00 on D.
  load     the operator load forecast for the 24 hours of D+1 (v1 deep_data: newest vintage public by 05:00)
           and the lfwow__ block (week-on-week change of the forecast) of the day matrix.
  outage   the out__ block of the day matrix (outage counts).
  weather  archived temperature forecasts for the 12 points x 24 hours of D+1: wxr__ (GEFS reforecast, to 2019)
           where present, else wx__ (GFS archive, from 25 March 2021); masked where neither exists; a source flag.
  calendar hour, weekday, day of year, weekend, holiday (deterministic). Never removed.

Timing: hourly cells use the v1 rule (published_at <= 05:00 on D) and the v1 rule-based guard runs on EVERY batch
(raises LookaheadError). Day-matrix blocks were filtered on their own published_at by 05:00 on D when the matrix
was built (pipeline/README.md, test_timing_v2.py); here each row must sit on its own delivery day (delivery_date
= bid_date + 1 = the tensor's day index), only the whitelisted prefixes are read, label columns (y__) are refused,
and the rows are aligned by delivery date, never by position.

Holdout: nothing on or after 2024-01-01 (rolling.assert_pre2024 on every read; the v1 lock in DeepData).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import torch

import rolling as R
import deep_data as DD                       # v1
from timing import LookaheadError            # v1 pipeline

GROUPS = ("gen", "border", "load", "outage", "weather")          # removable groups (ablations)
GEN_FIELDS4 = ("da_loss", "da_congestion", "rt_loss", "rt_congestion")
DAY_PREFIXES = {"border": ("bord__",), "load": ("lfwow__",), "outage": ("out__",), "weather": ("wxr__", "wx__")}
WX_SCALE = 10.0                              # degrees C


class Data(DD.DeepData):
    """v1 DeepData with 4 fields per generator point (asinh-scaled, float16) and without the border proxies."""

    def _load_gen(self, tables, lo, hi):
        files = self._gen_files(lo, hi)
        ext = set()
        for f in files:
            t = ds.dataset(str(f), format="parquet").to_table(columns=["ptid", "point_type"]).to_pandas()
            ext |= set(t.loc[t["point_type"] != "gen", "ptid"].unique())
        self.external_ptids = np.array(sorted(ext), np.int64)
        cols = ["delivery_hour", "ptid", *GEN_FIELDS4, "da_published_at", "rt_published_at"]
        ptids = set()
        for f in files:
            ptids |= set(ds.dataset(str(f), format="parquet").to_table(columns=["ptid"]).column(0).to_pylist())
        self.ptids = np.array(sorted(ptids - ext), np.int64)
        P = len(self.ptids)
        self.Gv = np.zeros((self.T, P, 4), np.float16)
        self.Gp = np.full((self.T, P, 2), DD.NEVER, np.int32)
        for f in files:
            p = DD._read(f, cols, "delivery_hour", lo, hi)
            p = p[~p["ptid"].isin(ext)]
            if not len(p):
                continue
            R.assert_pre2024(p["delivery_hour"], f"gen {f.name}")
            h = self._hidx(p["delivery_hour"])
            c = np.searchsorted(self.ptids, p["ptid"].to_numpy(np.int64))
            v = p[list(GEN_FIELDS4)].to_numpy(np.float32)
            dap, rtp = DD.minutes_ceil(p["da_published_at"]), DD.minutes_ceil(p["rt_published_at"])
            dap[~np.isfinite(v[:, :2]).all(1)] = DD.NEVER
            rtp[~np.isfinite(v[:, 2:]).all(1)] = DD.NEVER
            self.Gv[h, c] = np.arcsinh(np.nan_to_num(v) / DD.SCALE).astype(np.float16)
            self.Gp[h, c, 0], self.Gp[h, c, 1] = dap, rtp

    def point_hours_before(self, k_end: int) -> np.ndarray:
        return (self.Gp[: self.day_start[k_end], :, 0] < DD.NEVER).sum(0)


def load_data(log) -> Data:
    d = Data(R.LAST_SCORED, root=R.PARQUET, first_day=R.DATA_START, log=log)
    k = np.arange(d.n_days)[:, None, None]
    bad = d.LFm & (d.LFiss > k - 1)          # same conservative fix as deep_rolling.load_data
    if bad.any():
        log(f"load forecast: {int(bad.sum())} cells with issue date after the bid day hidden")
        d.LFm = d.LFm & ~bad
        d.LF = np.where(bad, 0, d.LF).astype(np.float32)
    return d


# ============================================================================ day-matrix blocks
def day_columns(cols: list[str], with_weather: bool) -> dict[str, list[str]]:
    out = {}
    for g, pre in DAY_PREFIXES.items():
        if g == "weather" and not with_weather:
            out[g] = []
            continue
        out[g] = [c for c in cols if c.startswith(pre)]
    for g, cs in out.items():
        if any(c.startswith("y__") for c in cs):
            raise LookaheadError(f"label column in group {g}")
    return out


def weather_points(cols: list[str]) -> list[str]:
    pts = sorted({c.split("__")[1] for c in cols if c.startswith(("wxr__", "wx__"))})
    return pts


class DayBlocks:
    """Day-level arrays aligned to the Data day index k (delivery day first_day + k; bid day k - 1).
    border [n, 224], lfwow [n, 288], out [n, 7] as raw floats (NaN = missing); weather [n, P, 24] (C), its mask,
    and source flags [n, 2] (reforecast, GFS)."""

    def __init__(self, data: Data, path: Path, with_weather: bool, log=print):
        sch = ds.dataset(str(path), format="parquet").schema.names
        groups = day_columns(sch, with_weather)
        if with_weather and not any(c.startswith("wxr__") for c in groups["weather"]):
            raise RuntimeError("weather requested but the day matrix has no wxr__ columns (weather stage not done)")
        cols = ["bid_date", "delivery_date"] + sorted({c for cs in groups.values() for c in cs})
        df = R.read_pre2024(path, "delivery_date", cols)
        dd = pd.to_datetime(df["delivery_date"]).dt.normalize()
        bd = pd.to_datetime(df["bid_date"]).dt.normalize()
        if not (dd == bd + pd.Timedelta(days=1)).all():
            raise LookaheadError("day matrix: delivery_date != bid_date + 1")
        if dd.duplicated().any():
            raise ValueError("day matrix: duplicate delivery dates")
        k = ((dd - pd.Timestamp(data.first_day)).dt.days).to_numpy()
        ok = (k >= 0) & (k < data.n_days)
        df, k = df[ok], k[ok]
        n = data.n_days
        self.k_present = np.zeros(n, bool)
        self.k_present[k] = True
        self.delivery = np.full(n, -1, np.int64)
        self.delivery[k] = (dd[ok] - pd.Timestamp(data.first_day)).dt.days.to_numpy()

        def block(cs):
            a = np.full((n, len(cs)), np.nan, np.float32)
            if cs:
                a[k] = df[cs].to_numpy(np.float32)
            return a
        self.border = block(groups["border"])
        self.lfwow = block(groups["load"])
        self.out = block(groups["outage"])
        self.cols = groups
        pts = weather_points(sch)
        self.points = pts
        W = np.full((n, len(pts), 24), np.nan, np.float32)
        src = np.zeros((n, 2), np.float32)
        if with_weather:
            for pi, p in enumerate(pts):
                r = [f"wxr__{p}__h{h:02d}" for h in range(24)]
                g = [f"wx__{p}__h{h:02d}" for h in range(24)]
                vr = df[r].to_numpy(np.float32) if all(c in df for c in r) else np.full((len(df), 24), np.nan, np.float32)
                vg = df[g].to_numpy(np.float32) if all(c in df for c in g) else np.full((len(df), 24), np.nan, np.float32)
                W[k, pi] = np.where(np.isfinite(vr), vr, vg)
                src[k, 0] = np.maximum(src[k, 0], np.isfinite(vr).any(1))
                src[k, 1] = np.maximum(src[k, 1], (~np.isfinite(vr) & np.isfinite(vg)).any(1))
        self.wx, self.wxm, self.wxsrc = W, np.isfinite(W), src
        log(f"day blocks: border {self.border.shape[1]}, lfwow {self.lfwow.shape[1]}, out {self.out.shape[1]}, "
            f"weather {len(pts)} points x 24 (present on {int(self.wxm.any((1, 2)).sum())} days)")

    def assert_aligned(self, k: np.ndarray) -> None:
        """Every row used for delivery day k is the matrix row of delivery day k (bid day k - 1)."""
        k = np.asarray(k)
        pres = self.k_present[k]
        if (self.delivery[k][pres] != k[pres]).any():
            raise LookaheadError("day block row not aligned to its delivery day")


class Scaler:
    """Per-column mean and std on the training days only (NaN ignored); applied as (x - m) / s, NaN -> 0."""

    def __init__(self, blocks: DayBlocks, days: np.ndarray):
        self.st = {}
        for name in ("border", "lfwow", "out"):
            a = getattr(blocks, name)[days]
            m = np.nanmean(a, 0) if a.shape[1] else np.zeros(0)
            s = np.nanstd(a, 0) if a.shape[1] else np.zeros(0)
            m = np.nan_to_num(m)
            s = np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
            self.st[name] = (m.astype(np.float32), s.astype(np.float32))
        w = blocks.wx[days]
        wm = float(np.nanmean(w)) if np.isfinite(w).any() else 0.0
        self.st["wx"] = (np.float32(wm), np.float32(WX_SCALE))


def day_device(blocks: DayBlocks, sc: Scaler, device) -> dict:
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a)).to(device)  # noqa: E731
    out = {}
    for name in ("border", "lfwow", "out"):
        a = getattr(blocks, name)
        m, s = sc.st[name]
        x = np.where(np.isfinite(a), (a - m) / s, 0.0).astype(np.float32)
        x = np.clip(x, -8, 8)
        out[name] = t(x)
        out[name + "_miss"] = t((~np.isfinite(a)).mean(1, keepdims=True).astype(np.float32) if a.shape[1]
                                else np.ones((len(a), 1), np.float32))
    wm, ws = sc.st["wx"]
    out["wx"] = t(np.where(blocks.wxm, (blocks.wx - wm) / ws, 0.0).astype(np.float32))
    out["wxm"] = t(blocks.wxm.astype(np.float32))
    out["wxsrc"] = t(blocks.wxsrc)
    return out


# ============================================================================ hourly tensors
def device_arrays(d: Data, device) -> dict:
    key = ("v13", str(device))
    if key not in d._dev:
        t = lambda a: torch.as_tensor(np.ascontiguousarray(a)).to(device)  # noqa: E731
        d._dev[key] = {"Zv": t(d.Zv), "Zp": t(d.Zp), "Gv": t(d.Gv), "Gp": t(d.Gp), "CAL": t(d.CAL),
                       "Hday": t(d.Hday), "Hmin": t(d.Hmin), "day_start": t(d.day_start), "dec_min": t(d.dec_min),
                       "LF": t(d.LF), "LFm": t(d.LFm), "LFp": t(d.LFp), "LFiss": t(d.LFiss), "DCAL": t(d.DCAL)}
    return d._dev[key]


N_SEQ = 4 * DD.N_Z + 4 * DD.N_Z + DD.N_Z + 2 * DD.N_Z + DD.N_CAL      # 129 sequence channels


def batch(d: Data, k, cols, device, check: bool = True) -> dict:
    """Hourly inputs for delivery-day indices k (bid day k - 1), with the v1 rule-based guard on every call."""
    A = device_arrays(d, device)
    k = torch.as_tensor(k, dtype=torch.long).to(device)
    ar = torch.arange(DD.WINDOW_H, device=device)
    idx = A["day_start"][k][:, None] - DD.WINDOW_H + ar[None]
    t = A["dec_min"][k]
    tt = t[:, None, None]
    zv, zp = A["Zv"][idx], A["Zp"][idx]
    m_da, m_rt = zp[..., 0] <= tt, zp[..., 1] <= tt
    g_idx = idx[:, :, None]
    gp = A["Gp"][g_idx, cols[None, None, :]]                          # [B,168,P,2]
    m_g = gp <= t[:, None, None, None]
    lfm = A["LFm"][k]
    if check:
        da_rule = (A["Hday"][idx] <= (k - 1)[:, None])[..., None]
        rt_rule = (A["Hmin"][idx] + DD.RT_RULE_MIN <= t[:, None])[..., None]
        lf_ok = (A["LFp"][k] <= tt) & (A["LFiss"][k] <= (k - 1)[:, None, None])
        viol = torch.stack([(m_da & ~da_rule).sum(), (m_rt & ~rt_rule).sum(),
                            (m_g[..., 0] & ~da_rule).sum(), (m_g[..., 1] & ~rt_rule).sum(), (lfm & ~lf_ok).sum()])
        if int(viol.sum()):
            names = ("zone_da", "zone_rt", "gen_da", "gen_rt", "load_forecast")
            raise LookaheadError(f"v13 batch guard: {dict(zip(names, viol.tolist()))}")
    zv0 = torch.nan_to_num(zv)
    da = torch.cat([zv0[..., :3], (zv0[..., 0] - zv0[..., 1] - zv0[..., 2])[..., None]], -1)   # lbmp, loss, cong, energy
    rt = torch.cat([zv0[..., 3:], (zv0[..., 3] - zv0[..., 4] - zv0[..., 5])[..., None]], -1)
    zda = torch.where(m_da[..., None], torch.asinh(da / DD.SCALE), 0.0)
    zrt = torch.where(m_rt[..., None], torch.asinh(rt / DD.SCALE), 0.0)
    both = m_da & m_rt
    gap = torch.where(both, torch.asinh((zv0[..., 3] - zv0[..., 0]) / DD.SCALE), 0.0)
    B = idx.shape[0]
    seq = torch.cat([zda.reshape(B, DD.WINDOW_H, -1), zrt.reshape(B, DD.WINDOW_H, -1), gap,
                     m_da.float(), m_rt.float(), A["CAL"][idx]], -1)
    gv = A["Gv"][g_idx, cols[None, None, :]].float()                  # [B,168,P,4] asinh-scaled
    mg = torch.cat([m_g[..., :1].expand(-1, -1, -1, 2), m_g[..., 1:].expand(-1, -1, -1, 2)], -1)
    gx = torch.where(mg, gv, 0.0)
    lf_age = torch.where(lfm, ((k - 1)[:, None, None] - A["LFiss"][k]).float(), 0.0)
    return {"seq": seq, "gx": gx, "gm": m_g.float().mean(2), "lf": A["LF"][k], "lfm": lfm.float(),
            "lf_age": lf_age, "dcal": A["DCAL"][k], "k": k}
