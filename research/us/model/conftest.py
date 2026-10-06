"""Synthetic NYISO-shaped tables for unit tests (same columns and published_at rules as the real
parquet, README.md). They deliberately include January 2024 rows so tests can prove the lock keeps
them out. Nothing here touches real data."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pipeline"))

from common import TZ, ZONES, gfs_published_at  # noqa: E402  (valid - 40 h for day2, - 64 h for day3)

SYN_START = dt.date(2022, 10, 1)
SYN_END = dt.date(2024, 1, 10)          # inclusive; the January 2024 rows must never be loaded
REVISED_DAYS = {dt.date(2023, 3, 9), dt.date(2023, 11, 4), dt.date(2023, 7, 14)}


def _local_at(days, hh, mm=0):
    return pd.DatetimeIndex([pd.Timestamp(dt.datetime.combine(d, dt.time(hh, mm)), tz=TZ) for d in days])


def make_prices(rng, points, start=SYN_START, end=SYN_END, kind="zone"):
    hours = pd.date_range(pd.Timestamp(start, tz=TZ), pd.Timestamp(end + dt.timedelta(days=1), tz=TZ),
                          freq="h", inclusive="left")
    n = len(hours)
    rows = []
    loc = hours.tz_convert(TZ)
    ldate = loc.tz_localize(None).normalize()
    energy_da = 30 + 10 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.normal(0, 3, n)
    energy_rt = energy_da + rng.normal(-1, 8, n) + (rng.random(n) < 0.01) * rng.exponential(300, n)
    days = pd.Series(ldate.date)
    da_written = _local_at([d - dt.timedelta(days=1) for d in days], 9, 33)
    da_bound = _local_at([d - dt.timedelta(days=1) for d in days], 11, 0)
    rt_written = []
    for d in days:
        if d in REVISED_DAYS:
            rt_written.append(pd.Timestamp(dt.datetime.combine(d + dt.timedelta(days=6), dt.time(10, 0)), tz=TZ))
        else:
            rt_written.append(pd.Timestamp(dt.datetime.combine(d, dt.time(23, 57)), tz=TZ))
    rt_written = pd.DatetimeIndex(rt_written)
    revised = np.array([d in REVISED_DAYS for d in days])
    for i, (name, ptype) in enumerate(points):
        da_cong = rng.normal(0, 2, n) * (i % 3)
        rt_cong = da_cong + rng.normal(0, 3, n) * (i % 3)
        da_loss = rng.normal(0, 0.5, n)
        rt_loss = da_loss + rng.normal(0, 0.2, n)
        da = energy_da + da_loss + da_cong + i
        rt = energy_rt + rt_loss + rt_cong + i
        rt_pub = pd.Series(hours + pd.Timedelta(hours=1, minutes=15)).astype("datetime64[ns, America/New_York]")
        rw = pd.Series(rt_written).astype("datetime64[ns, America/New_York]")
        rt_pub = rt_pub.where(~(revised & (rw > rt_pub).to_numpy()), rw)
        dw = pd.Series(da_written).astype("datetime64[ns, America/New_York]")
        db = pd.Series(da_bound).astype("datetime64[ns, America/New_York]")
        da_pub = dw.where(dw >= db, db)
        df = pd.DataFrame({
            "delivery_hour": hours, "zone" if kind == "zone" else "name": name, "ptid": 61750 + i,
            "da_lbmp": da, "da_loss": da_loss, "da_congestion_raw": -da_cong, "da_congestion": da_cong,
            "da_energy": energy_da, "rt_lbmp": rt, "rt_loss": rt_loss, "rt_congestion_raw": -rt_cong,
            "rt_congestion": rt_cong, "rt_energy": energy_rt, "da_published_at": da_pub.to_numpy(),
            "rt_published_at": rt_pub.to_numpy(), "rt_revised": revised, "da_file_written_at": da_written,
            "rt_file_written_at": rt_written})
        if kind != "zone":
            df["point_type"] = ptype
        df["published_at"] = df[["da_published_at", "rt_published_at"]].max(axis=1)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def make_load_forecast(rng, start=SYN_START, end=SYN_END):
    out = []
    for i in range((end - start).days + 2):
        issue = start - dt.timedelta(days=1) + dt.timedelta(days=i)
        written = pd.Timestamp(dt.datetime.combine(issue - dt.timedelta(days=1), dt.time(7, 5)), tz=TZ)
        if issue == dt.date(2023, 6, 20):                     # a late file: written after 05:00 on its date
            written = pd.Timestamp(dt.datetime.combine(issue, dt.time(6, 30)), tz=TZ)
        th = pd.date_range(pd.Timestamp(issue, tz=TZ), pd.Timestamp(issue + dt.timedelta(days=6), tz=TZ),
                           freq="h", inclusive="left")
        lead = (th.tz_convert(TZ).tz_localize(None).normalize() - pd.Timestamp(issue)).days
        for z in ZONES + ["NYISO"]:
            base = 2000 if z != "NYISO" else 18000
            out.append(pd.DataFrame({"issue_date": issue, "file_written_at": written, "published_at": written,
                                     "target_hour": th, "lead_days": lead, "zone": z,
                                     "load_forecast_mw": base + rng.normal(0, 50, len(th)) + 10 * i}))
    return pd.concat(out, ignore_index=True)


def make_outages(start=SYN_START, end=SYN_END):
    out = []
    for i in range((end - start).days + 1):
        s = start + dt.timedelta(days=i)
        w = pd.Timestamp(dt.datetime.combine(s - dt.timedelta(days=1), dt.time(9, 40)), tz=TZ)
        out.append(pd.DataFrame({"snapshot_date": [s, s], "file_written_at": w, "published_at": w,
                                 "ptid": [1, 2], "equipment": ["LINE A", "LINE B"],
                                 "sched_out": pd.Timestamp(s, tz=TZ), "sched_in": pd.Timestamp(s, tz=TZ) + pd.Timedelta(days=3)}))
    return pd.concat(out, ignore_index=True)


def make_weather(rng, start=SYN_START, end=SYN_END):
    th = pd.date_range(pd.Timestamp(start, tz=TZ), pd.Timestamp(end + dt.timedelta(days=1), tz=TZ),
                       freq="h", inclusive="left")
    out = []
    tz_th = pd.Series(th)
    for z in ZONES:
        for lead in (48, 72):                     # previous_day2 and previous_day3 (three-day rule)
            out.append(pd.DataFrame({"point": z, "zone": z, "primary": True, "target_hour": th,
                                     "temperature_2m_c": rng.normal(10, 8, len(th)), "run_lead_hours": lead,
                                     "published_at": gfs_published_at(tz_th, lead).to_numpy()}))
    return pd.concat(out, ignore_index=True)


def write_synthetic(root: Path, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    make_prices(rng, [(z, "zone") for z in ZONES]).to_parquet(root / "prices_zone.parquet")
    make_load_forecast(rng).to_parquet(root / "load_forecast.parquet")
    make_outages().to_parquet(root / "outages.parquet")
    make_weather(rng).to_parquet(root / "weather_gfs.parquet")
    pts = [("GEN A", "gen"), ("GEN B", "gen"), ("GEN C", "gen"), ("H Q", "external"), ("NPX", "external"),
           ("O H", "external"), ("PJM", "external")]
    g = make_prices(rng, pts, kind="gen")
    (root / "prices_gen").mkdir(exist_ok=True)
    ym = g["delivery_hour"].dt.tz_convert(TZ).dt.strftime("%Y%m")
    for k, part in g.groupby(ym):
        part.reset_index(drop=True).to_parquet(root / "prices_gen" / f"{k}.parquet")
    return root


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory):
    return write_synthetic(tmp_path_factory.mktemp("parquet"))
