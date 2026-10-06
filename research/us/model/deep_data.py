"""Tensors for the deep model, built from the parquet tables (owner: deep agent; CONTRACT.md).

What the model may see for bid day D (decision at 05:00 New York time on D, delivery day D+1):
  * zone prices for the 168 hours that end at the local midnight starting D+1: day-ahead and
    real-time LBMP, loss and congestion (the sign-corrected *_congestion columns, positive = congestion
    raises the price; see ../pipeline/README.md "Congestion sign") and the gap rt - da. Each cell is
    visible only if its own published_at (da_published_at or rt_published_at) is at or before 05:00
    on D: the rule of timing.features_available_at, which also splits DA and RT by component time.
  * generator and border points (prices_gen): da_congestion and rt_congestion per point, same rule.
    A point that did not publish an hour is absent in that hour (masked), so every point is used
    only while it existed.
  * the operator load forecast for the hours of D+1: per target hour and zone the newest vintage
    published at or before 05:00 on D (the selection of timing.bid_inputs).
  * calendar (deterministic).

Proofs in test_deep_data.py: the visible cells equal what timing.features_available_at and
timing.bid_inputs return for the same day; a value published after the deadline never changes a
tensor; an injected lookahead cell (a mislabelled published_at, or a tampered mask) is caught by
the guard, which re-derives visibility from rules that do not trust published_at.

Holdout: every read calls lock.assert_build_only on the requested range before reading and again
on the delivery dates read. Parquet reads are filtered to the requested range.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent / "pipeline"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import lock  # noqa: E402
from common import DECISION_HOUR, FIRST_DAY, PARQUET, RT_LAG, TZ, ZONES  # noqa: E402
from timing import LookaheadError  # noqa: E402

WINDOW_H = 168                      # hours of history per bid day
PAD = WINDOW_H                      # empty hours before the first day, so every window is in range
SCALE = 25.0                        # price inputs enter as asinh(x / SCALE)
ZONE_FIELDS = ("da_lbmp", "da_loss", "da_congestion", "rt_lbmp", "rt_loss", "rt_congestion")
GEN_FIELDS = ("da_congestion", "rt_congestion")
LF_ZONES = list(ZONES) + ["NYISO"]
N_Z, N_LF = len(ZONES), len(LF_ZONES)
NEVER = np.iinfo(np.int32).max      # published_at of a missing cell
EPOCH_NS = pd.Timestamp("2019-01-01", tz="UTC").value
MIN_NS, HOUR_NS = 60 * 10**9, 3600 * 10**9
RT_RULE_MIN = 60 + int(RT_LAG / pd.Timedelta(minutes=1))   # an RT hour is public >= 75 min after it starts
N_CAL, N_DCAL = 8, 6


# ----------------------------------------------------------------------------- time helpers
def _ns(s) -> np.ndarray:
    """tz-aware Series/Index to int64 UTC nanoseconds (NaT -> int64 min)."""
    s = pd.Series(s)
    return s.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype(np.int64)


def minutes_ceil(s) -> np.ndarray:
    """published_at as whole minutes since 2019-01-01 UTC, rounded UP (never earlier than the truth).
    Because decision times are whole minutes, pub <= t  <=>  minutes_ceil(pub) <= minutes(t). NaT -> NEVER."""
    ns = _ns(s)
    nat = ns == np.iinfo(np.int64).min
    m = -((EPOCH_NS - ns) // MIN_NS)
    m[nat] = NEVER
    return m.astype(np.int32)


def minutes_exact(ts: pd.Timestamp) -> int:
    v = pd.Timestamp(ts).value - EPOCH_NS
    if v % MIN_NS:
        raise ValueError(f"{ts} is not a whole minute")
    return int(v // MIN_NS)


def local_midnight(d: dt.date) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(d, dt.time(0)), tz=TZ)


def decision_ts(bid_day: dt.date) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(bid_day, dt.time(DECISION_HOUR)), tz=TZ)


def local_dates(s) -> np.ndarray:
    """tz-aware timestamps to local New York dates (datetime64[D])."""
    return pd.Series(s).dt.tz_convert(TZ).dt.tz_localize(None).to_numpy().astype("datetime64[D]")


def nerc_holidays(years) -> set:
    """The six NERC off-peak holidays (observed dates)."""
    from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday, USLaborDay, USMemorialDay,
                                        USThanksgivingDay, sunday_to_monday)

    class NERC(AbstractHolidayCalendar):
        rules = [Holiday("NewYear", month=1, day=1, observance=sunday_to_monday), USMemorialDay,
                 Holiday("July4", month=7, day=4, observance=sunday_to_monday), USLaborDay,
                 USThanksgivingDay, Holiday("Christmas", month=12, day=25, observance=sunday_to_monday)]
    ys = sorted(years)
    h = NERC().holidays(pd.Timestamp(f"{ys[0]}-01-01"), pd.Timestamp(f"{ys[-1]}-12-31"))
    return {d.date() for d in h}


def _calendar(local: pd.DatetimeIndex, hol: set) -> np.ndarray:
    """Per-hour calendar: hour sin/cos, weekday sin/cos, day-of-year sin/cos, weekend, holiday."""
    hr = local.hour.to_numpy()
    dow = local.dayofweek.to_numpy()
    doy = local.dayofyear.to_numpy()
    days = local.tz_localize(None).normalize()
    holi = np.array([d.date() in hol for d in days]) if len(days) else np.zeros(0, bool)
    return np.stack([np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24),
                     np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                     np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
                     (dow >= 5).astype(float), holi.astype(float)], 1).astype(np.float32)


# ----------------------------------------------------------------------------- parquet reads
def _read(path: Path, columns: list[str], timecol: str, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    """Rows of one parquet file or a list of files with timecol in [lo, hi), filtered inside pyarrow."""
    d = ds.dataset([str(p) for p in path] if isinstance(path, list) else str(path), format="parquet")
    typ = d.schema.field(timecol).type
    flt = (ds.field(timecol) >= pa.scalar(lo, type=typ)) & (ds.field(timecol) < pa.scalar(hi, type=typ))
    return d.to_table(columns=columns, filter=flt).to_pandas()


def _from_tables(tables: dict, name: str, columns, timecol, lo, hi) -> pd.DataFrame:
    df = tables[name]
    df = df[(df[timecol] >= lo) & (df[timecol] < hi)]
    return df[list(columns)].reset_index(drop=True)


# ----------------------------------------------------------------------------- the data object
class DeepData:
    """Dense hourly arrays for delivery dates first_day .. last_delivery (inclusive).

    Grid: one row per absolute hour starting PAD hours before local midnight of first_day.
      Zv [T, 11, 6] zone fields, Zp [T, 11, 2] DA/RT published_at (minutes, ceil; NEVER if missing)
      Gv [T, P, 2] point da/rt congestion, Gp [T, P, 2] their published_at
      CAL [T, 8] calendar, Hday [T] local delivery-day index of the hour, Hmin [T] hour start (minutes)
    Per delivery-day index k (date first_day + k):
      day_start[k] grid index of its local midnight (k = 0 .. n_days), dec_min[k] = 05:00 on the bid
      day k-1 (minutes), LF/LFm/LFp/LFiss [n_days, 24, 12] load forecast for its hours (slot = local
      hour; the two 01:00 hours of a fall-back day are averaged), DCAL [n_days, 6] day calendar.
    """

    def __init__(self, last_delivery: dt.date, root: Path = PARQUET, tables: dict | None = None,
                 first_day: dt.date = FIRST_DAY, with_gen: bool = True, log=None):
        self.first_day, self.last_delivery = first_day, last_delivery
        self.root = Path(root)
        log = log or (lambda *_: None)
        lock.assert_build_only([last_delivery])                 # before any read
        self.n_days = (last_delivery - first_day).days + 1
        lo, hi = local_midnight(first_day), local_midnight(last_delivery + dt.timedelta(days=1))
        self.origin_ns = lo.value - PAD * HOUR_NS
        self.T = int((hi.value - self.origin_ns) // HOUR_NS)
        mids = pd.DatetimeIndex([local_midnight(first_day + dt.timedelta(days=k)) for k in range(self.n_days + 1)])
        self.day_start = ((_ns(mids) - self.origin_ns) // HOUR_NS).astype(np.int64)
        bid_days = [first_day + dt.timedelta(days=k - 1) for k in range(self.n_days)]
        self.dec_min = np.array([minutes_exact(decision_ts(d)) for d in bid_days], dtype=np.int64)

        grid = pd.DatetimeIndex(pd.to_datetime(self.origin_ns + np.arange(self.T, dtype=np.int64) * HOUR_NS,
                                               utc=True)).tz_convert(TZ)
        hol = nerc_holidays(range(first_day.year - 1, last_delivery.year + 2))
        self.CAL = _calendar(grid, hol)
        self.Hmin = ((self.origin_ns + np.arange(self.T, dtype=np.int64) * HOUR_NS - EPOCH_NS) // MIN_NS)
        gd = grid.tz_localize(None).normalize().to_numpy().astype("datetime64[D]")
        self.Hday = (gd - np.datetime64(first_day, "D")).astype(np.int64)
        dl = pd.DatetimeIndex(mids[:-1].tz_localize(None))
        dow, doy = dl.dayofweek.to_numpy(), dl.dayofyear.to_numpy()
        self.DCAL = np.stack([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                              np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
                              (dow >= 5).astype(float),
                              np.array([d.date() in hol for d in dl], float)], 1).astype(np.float32)

        self._load_zone(tables, lo, hi)
        log(f"deep_data: zone rows loaded, grid {self.T} hours")
        if with_gen:
            self._load_gen(tables, lo, hi)
            log(f"deep_data: {len(self.ptids)} points")
        else:
            self.ptids = np.zeros(0, np.int64)
            self.Gv = np.zeros((self.T, 0, 2), np.float32)
            self.Gp = np.zeros((self.T, 0, 2), np.int32)
        self._load_lf(tables, lo, hi)
        self._dev = {}

    # -- loaders --------------------------------------------------------------------------
    def _hidx(self, s) -> np.ndarray:
        ns = _ns(s)
        off = ns - self.origin_ns
        if (off % HOUR_NS).any():
            raise ValueError("delivery hours not on the hour grid")
        return (off // HOUR_NS).astype(np.int64)

    def _load_zone(self, tables, lo, hi):
        cols = ["delivery_hour", "zone", *ZONE_FIELDS, "da_published_at", "rt_published_at"]
        pz = (_from_tables(tables, "prices_zone", cols, "delivery_hour", lo, hi) if tables and "prices_zone" in tables
              else _read(self.root / "prices_zone.parquet", cols, "delivery_hour", lo, hi))
        lock.assert_build_only(pz["delivery_hour"])            # on what was read
        pz = pz[pz["zone"].isin(ZONES)]
        h = self._hidx(pz["delivery_hour"])
        z = pd.Categorical(pz["zone"], categories=ZONES).codes.astype(np.int64)
        if pd.Series(h * N_Z + z).duplicated().any():
            raise ValueError("duplicate zone-hours in prices_zone")
        vals = pz[list(ZONE_FIELDS)].to_numpy(np.float32)
        dap, rtp = minutes_ceil(pz["da_published_at"]), minutes_ceil(pz["rt_published_at"])
        dap[~np.isfinite(vals[:, :3]).all(1)] = NEVER
        rtp[~np.isfinite(vals[:, 3:]).all(1)] = NEVER
        self.Zv = np.full((self.T, N_Z, len(ZONE_FIELDS)), np.nan, np.float32)
        self.Zp = np.full((self.T, N_Z, 2), NEVER, np.int32)
        self.Zv[h, z] = vals
        self.Zp[h, z, 0], self.Zp[h, z, 1] = dap, rtp

    def _gen_files(self, lo, hi) -> list[Path]:
        months = pd.period_range(self.first_day, self.last_delivery, freq="M")
        return [p for p in (self.root / "prices_gen" / f"{m.strftime('%Y%m')}.parquet" for m in months) if p.exists()]

    def _load_gen(self, tables, lo, hi):
        cols = ["delivery_hour", "ptid", *GEN_FIELDS, "da_published_at", "rt_published_at"]
        if tables and "prices_gen" in tables:
            parts = [_from_tables(tables, "prices_gen", cols, "delivery_hour", lo, hi)]
        else:
            parts = [_read(f, cols, "delivery_hour", lo, hi) for f in self._gen_files(lo, hi)]
        parts = [p for p in parts if len(p)]
        for p in parts:
            lock.assert_build_only(p["delivery_hour"])
        self.ptids = np.array(sorted(set().union(*[set(p["ptid"].unique()) for p in parts])) if parts else [],
                              dtype=np.int64)
        P = len(self.ptids)
        self.Gv = np.full((self.T, P, 2), np.nan, np.float32)
        self.Gp = np.full((self.T, P, 2), NEVER, np.int32)
        for p in parts:
            h = self._hidx(p["delivery_hour"])
            c = np.searchsorted(self.ptids, p["ptid"].to_numpy(np.int64))
            v = p[list(GEN_FIELDS)].to_numpy(np.float32)
            dap, rtp = minutes_ceil(p["da_published_at"]), minutes_ceil(p["rt_published_at"])
            dap[~np.isfinite(v[:, 0])] = NEVER
            rtp[~np.isfinite(v[:, 1])] = NEVER
            self.Gv[h, c] = v
            self.Gp[h, c, 0], self.Gp[h, c, 1] = dap, rtp

    def _load_lf(self, tables, lo, hi):
        cols = ["issue_date", "published_at", "target_hour", "zone", "load_forecast_mw"]
        lf = (_from_tables(tables, "load_forecast", cols, "target_hour", lo, hi) if tables and "load_forecast" in tables
              else _read(self.root / "load_forecast.parquet", cols, "target_hour", lo, hi))
        lock.assert_build_only(lf["target_hour"])
        lf = lf[lf["zone"].isin(LF_ZONES)].copy()
        k = (local_dates(lf["target_hour"]) - np.datetime64(self.first_day, "D")).astype(np.int64)
        pubm = minutes_ceil(lf["published_at"])
        lf = lf[pubm <= self.dec_min[k]]                        # public at 05:00 on the bid day
        # timing.bid_inputs: newest published vintage per (zone, target hour)
        lf = lf.sort_values("published_at").drop_duplicates(["zone", "target_hour"], keep="last")
        self.lf_rows = lf.reset_index(drop=True)                # kept for the equivalence test
        k = (local_dates(lf["target_hour"]) - np.datetime64(self.first_day, "D")).astype(np.int64)
        slot = lf["target_hour"].dt.tz_convert(TZ).dt.hour.to_numpy()
        z = pd.Categorical(lf["zone"], categories=LF_ZONES).codes.astype(np.int64)
        iss = (pd.to_datetime(lf["issue_date"]).to_numpy().astype("datetime64[D]")
               - np.datetime64(self.first_day, "D")).astype(np.int64)
        shape = (self.n_days, 24, N_LF)
        tot, cnt = np.zeros(shape), np.zeros(shape)
        np.add.at(tot, (k, slot, z), lf["load_forecast_mw"].to_numpy(float))
        np.add.at(cnt, (k, slot, z), 1)
        self.LF = np.where(cnt > 0, tot / np.maximum(cnt, 1), 0).astype(np.float32)
        self.LFm = cnt > 0
        self.LFp = np.full(shape, -2**31, np.int64)
        np.maximum.at(self.LFp, (k, slot, z), minutes_ceil(lf["published_at"]).astype(np.int64))
        self.LFiss = np.full(shape, -10**6, np.int64)
        np.maximum.at(self.LFiss, (k, slot, z), iss)

    # -- indexing -------------------------------------------------------------------------
    def day_index(self, d) -> int:
        return (pd.Timestamp(d).date() - self.first_day).days

    def point_hours_before(self, k_end: int) -> np.ndarray:
        """Hours with a DA congestion value per point before the local midnight of delivery-day k_end."""
        return np.isfinite(self.Gv[: self.day_start[k_end], :, 0]).sum(0)

    # -- numpy reference sample (tests, guard) ----------------------------------------------
    def sample(self, bid_day: dt.date) -> dict:
        """Everything visible for bid day D, as numpy arrays with masks."""
        k = self.day_index(bid_day) + 1                          # delivery-day index of D+1
        end = int(self.day_start[k]); start = end - WINDOW_H
        t = int(self.dec_min[k])
        zp, gp = self.Zp[start:end], self.Gp[start:end]
        zv, gv = self.Zv[start:end], self.Gv[start:end]
        return {"k": k, "t": t, "start": start, "end": end,
                "zv": zv, "m_da": zp[..., 0] <= t, "m_rt": zp[..., 1] <= t,
                "gv": gv, "m_gda": gp[..., 0] <= t, "m_grt": gp[..., 1] <= t,
                "lf": self.LF[k], "m_lf": self.LFm[k]}

    def assert_sample_no_lookahead(self, bid_day: dt.date, s: dict):
        """Raise LookaheadError if any cell marked visible breaks a publication rule at 05:00 on D.
        Checked: stored published_at <= t; DA only for delivery dates <= D; RT only for hours that
        started at least 75 minutes before t (hour end + RT lag); load forecast only from files named
        on or before D and published <= t. The date and hour rules do not use published_at, so a
        mislabelled published_at is caught too."""
        k, t, start, end = s["k"], s["t"], s["start"], s["end"]
        bad = {}
        hday, hmin = self.Hday[start:end], self.Hmin[start:end]
        da_rule = (hday <= k - 1)[:, None]
        rt_rule = (hmin + RT_RULE_MIN <= t)[:, None]
        zp, gp = self.Zp[start:end], self.Gp[start:end]
        checks = {"zone_da": (s["m_da"], (zp[..., 0] <= t) & da_rule),
                  "zone_rt": (s["m_rt"], (zp[..., 1] <= t) & rt_rule),
                  "gen_da": (s["m_gda"], (gp[..., 0] <= t) & da_rule),
                  "gen_rt": (s["m_grt"], (gp[..., 1] <= t) & rt_rule),
                  "load_forecast": (s["m_lf"], (self.LFp[k] <= t) & (self.LFiss[k] <= k - 1))}
        for name, (vis, ok) in checks.items():
            n = int((vis & ~ok).sum())
            if n:
                bad[name] = n
        if bad:
            raise LookaheadError(f"bid day {bid_day}: cells visible after 05:00 rules: {bad}")

    # -- torch ----------------------------------------------------------------------------
    def device_arrays(self, device):
        import torch
        key = str(device)
        if key not in self._dev:
            t = lambda a, dtype=None: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype).to(device)  # noqa: E731
            self._dev[key] = {
                "Zv": t(self.Zv), "Zp": t(self.Zp), "Gv": t(self.Gv), "Gp": t(self.Gp), "CAL": t(self.CAL),
                "Hday": t(self.Hday), "Hmin": t(self.Hmin), "day_start": t(self.day_start),
                "dec_min": t(self.dec_min), "LF": t(self.LF), "LFm": t(self.LFm), "LFp": t(self.LFp),
                "LFiss": t(self.LFiss), "DCAL": t(self.DCAL)}
        return self._dev[key]


def batch(dd: DeepData, k, cols, device, check: bool = True) -> dict:
    """Model inputs for delivery-day indices k (LongTensor [B]; bid day = k - 1) on `device`.
    cols: LongTensor of point columns (indices into dd.ptids) the model was trained on.
    Visibility is published_at <= 05:00 on the bid day, cell by cell; with check=True the rule-based
    guard runs on the GPU and raises LookaheadError on any violation."""
    import torch
    A = dd.device_arrays(device)
    k = k.to(device)
    ar = torch.arange(WINDOW_H, device=device)
    idx = A["day_start"][k][:, None] - WINDOW_H + ar[None]          # [B, 168]
    t = A["dec_min"][k]                                              # [B]
    tt = t[:, None, None]
    zv, zp = A["Zv"][idx], A["Zp"][idx]                              # [B,168,11,6], [B,168,11,2]
    m_da, m_rt = zp[..., 0] <= tt, zp[..., 1] <= tt
    g_idx = idx[:, :, None]
    gv = A["Gv"][g_idx, cols[None, None, :]]                         # [B,168,P,2]
    gp = A["Gp"][g_idx, cols[None, None, :]]
    m_g = gp <= t[:, None, None, None]
    lfm = A["LFm"][k]
    if check:
        da_rule = (A["Hday"][idx] <= (k - 1)[:, None])[..., None]
        rt_rule = (A["Hmin"][idx] + RT_RULE_MIN <= t[:, None])[..., None]
        lf_ok = (A["LFp"][k] <= tt) & (A["LFiss"][k] <= (k - 1)[:, None, None])
        viol = torch.stack([(m_da & ~da_rule).sum(), (m_rt & ~rt_rule).sum(),
                            (m_g[..., 0] & ~da_rule).sum(), (m_g[..., 1] & ~rt_rule).sum(),
                            (lfm & ~lf_ok).sum()])
        if int(viol.sum()):                                          # one sync per batch
            names = ("zone_da", "zone_rt", "gen_da", "gen_rt", "load_forecast")
            raise LookaheadError(f"batch guard: {dict(zip(names, viol.tolist()))}")
    z = torch.asinh(torch.nan_to_num(zv) / SCALE)
    zda = torch.where(m_da[..., None], z[..., :3], 0.0)
    zrt = torch.where(m_rt[..., None], z[..., 3:], 0.0)
    both = m_da & m_rt
    gap = torch.where(both, torch.asinh(torch.nan_to_num(zv[..., 3] - zv[..., 0]) / SCALE), 0.0)
    B = idx.shape[0]
    seq = torch.cat([zda.reshape(B, WINDOW_H, -1), zrt.reshape(B, WINDOW_H, -1), gap,
                     m_da.float(), m_rt.float(), A["CAL"][idx]], -1)
    gx = torch.where(m_g, torch.asinh(torch.nan_to_num(gv) / SCALE), 0.0)
    lf_age = torch.where(lfm, ((k - 1)[:, None, None] - A["LFiss"][k]).float(), 0.0)
    return {"seq": seq, "gx": gx, "gm": m_g.float(), "lf": A["LF"][k], "lfm": lfm.float(),
            "lf_age": lf_age, "dcal": A["DCAL"][k], "k": k}


N_SEQ = 3 * N_Z + 3 * N_Z + N_Z + 2 * N_Z + N_CAL                    # 107 sequence channels


# ----------------------------------------------------------------------------- synthetic tables
def _at(d: dt.date, hh: int, mm: int) -> int:
    return pd.Timestamp(dt.datetime.combine(d, dt.time(hh, mm)), tz=TZ).value


def _ts(ns: np.ndarray, valid: np.ndarray | None = None) -> pd.Series:
    s = pd.Series(pd.to_datetime(np.asarray(ns, dtype=np.int64), utc=True)).dt.tz_convert(TZ)
    return s.where(valid) if valid is not None else s


def write_synthetic_parquet(root: Path, first: dt.date, last: dt.date, n_points: int = 24, seed: int = 0) -> dict:
    """Small tables with the real schema and publication rules (for tests and the CPU smoke run):
    DA public at 11:00 on D-1 (3% of days rewritten at 09:41 on D+1), RT at hour end + 15 min (8% of
    days revised at 14:02 on D+3), a few missing RT hours, points that appear and retire, two border
    proxies, isolf files named i holding targets i .. i+5 written 07:05 on i-1 (5% written 06:40 on i,
    after the deadline). Returns the frames."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "prices_gen").mkdir(parents=True, exist_ok=True)
    hours = pd.date_range(local_midnight(first), local_midnight(last + dt.timedelta(days=1)), freq="h",
                          inclusive="left")
    n = len(hours)
    hday = [h.date() for h in hours.tz_localize(None)]
    days = sorted(set(hday))
    late_da = {d: rng.random() < 0.03 for d in days}
    rev_rt = {d: rng.random() < 0.08 for d in days}
    da_pub_day = {d: _at(d + dt.timedelta(days=1), 9, 41) if late_da[d] else _at(d - dt.timedelta(days=1), 11, 0)
                  for d in days}
    da_pub = _ts(np.array([da_pub_day[d] for d in hday]))
    rt_ns = _ns(hours) + 75 * MIN_NS
    rev = np.array([rev_rt[d] for d in hday])
    rt_ns = np.where(rev, np.array([_at(d + dt.timedelta(days=3), 14, 2) for d in hday]), rt_ns)
    hod = hours.hour.to_numpy()
    base_da = 35 + 12 * np.sin(2 * np.pi * (hod - 9) / 24) + 8 * rng.standard_normal(n).cumsum() / np.sqrt(n)

    zrows = []
    for zi, zname in enumerate(ZONES):
        da = base_da * (1 + 0.05 * zi) + rng.normal(0, 2, n)
        spike = (rng.random(n) < 0.01) * rng.exponential(150, n)
        rt = da + rng.normal(-1.0, 6, n) + spike
        miss = rng.random(n) < 0.004
        rt[miss] = np.nan
        da_loss, rt_loss = rng.normal(1, 0.5, n), np.where(miss, np.nan, rng.normal(1, 0.5, n))
        da_cong = rng.normal(0, 3, n) * (zi > 6)
        rt_cong = np.where(miss, np.nan, rng.normal(0, 5, n) * (zi > 6))
        zrows.append(pd.DataFrame({
            "delivery_hour": hours, "zone": zname, "ptid": 61750 + zi,
            "da_lbmp": da, "da_loss": da_loss, "da_congestion_raw": -da_cong, "da_congestion": da_cong,
            "da_energy": da - da_loss - da_cong,
            "rt_lbmp": rt, "rt_loss": rt_loss, "rt_congestion_raw": -rt_cong, "rt_congestion": rt_cong,
            "rt_energy": rt - rt_loss - rt_cong,
            "da_published_at": da_pub.to_numpy(), "rt_published_at": _ts(rt_ns, ~miss).to_numpy(),
            "rt_revised": rev & ~miss}))
    pz = pd.concat(zrows, ignore_index=True).sort_values(["delivery_hour", "zone"], kind="stable")
    pz = pz.reset_index(drop=True)
    pz["published_at"] = pz[["da_published_at", "rt_published_at"]].max(axis=1)
    pz.to_parquet(root / "prices_zone.parquet", index=False)

    span = len(days)
    dayno = np.array([(d - days[0]).days for d in hday])
    rt_pub = _ts(rt_ns)
    grows = []
    for i in range(n_points + 2):
        ptid = 23500 + i if i < n_points else 24900 + i
        a = 0 if i % 3 else int(rng.integers(1, max(2, span // 2)))
        b = span if i % 5 else int(rng.integers(span // 2 + 1, span))
        keep = (dayno >= a) & (dayno < b)
        cong_da = rng.normal(0, 4, n) * (i % 4 == 0)
        cong_rt = cong_da + rng.normal(0, 3, n)
        grows.append(pd.DataFrame({
            "delivery_hour": hours[keep], "ptid": ptid, "name": f"P{i}",
            "point_type": "gen" if i < n_points else "external",
            "da_congestion": cong_da[keep], "rt_congestion": cong_rt[keep],
            "da_published_at": da_pub[keep].to_numpy(), "rt_published_at": rt_pub[keep].to_numpy()}))
    pg = pd.concat(grows, ignore_index=True)
    for c in ("da_lbmp", "da_loss", "rt_lbmp", "rt_loss", "da_energy", "rt_energy"):
        pg[c] = 30.0
    pg["da_congestion_raw"], pg["rt_congestion_raw"] = -pg["da_congestion"], -pg["rt_congestion"]
    pg["rt_revised"] = False
    pg["published_at"] = pg[["da_published_at", "rt_published_at"]].max(axis=1)
    ym = pg["delivery_hour"].dt.strftime("%Y%m")
    for m, part in pg.groupby(ym):
        part.sort_values(["delivery_hour", "ptid"]).to_parquet(root / "prices_gen" / f"{m}.parquet", index=False)

    lrows = []
    for i in [first + dt.timedelta(days=j) for j in range(-5, (last - first).days + 1)]:
        late = rng.random() < 0.05
        wr = _at(i, 6, 40) if late else _at(i - dt.timedelta(days=1), 7, 5)
        th = pd.date_range(local_midnight(i), local_midnight(i + dt.timedelta(days=6)), freq="h", inclusive="left")
        for zi, zname in enumerate(LF_ZONES):
            mw = (1500 + 300 * zi) * (1 + 0.25 * np.sin(2 * np.pi * (th.hour.to_numpy() - 8) / 24)) \
                * (1 + rng.normal(0, 0.02, len(th)))
            lrows.append(pd.DataFrame({"issue_date": i, "file_written_at": _ts(np.full(len(th), wr)).to_numpy(),
                                       "target_hour": th, "zone": zname, "load_forecast_mw": mw}))
    lf = pd.concat(lrows, ignore_index=True)
    lf["published_at"] = lf["file_written_at"]
    lf["lead_days"] = (lf["target_hour"].dt.tz_localize(None).dt.normalize()
                       - pd.to_datetime(lf["issue_date"])).dt.days
    lf = lf[(lf["target_hour"] >= local_midnight(first))
            & (lf["target_hour"] < local_midnight(last + dt.timedelta(days=1)))].reset_index(drop=True)
    lf.to_parquet(root / "load_forecast.parquet", index=False)
    return {"prices_zone": pz, "prices_gen": pg, "load_forecast": lf}
