"""Shared paths, constants and time helpers for the New York (NYISO) virtual-trading study.

Canonical copy: ~/projects/tavle/research/us/pipeline on the laptop; rsync to gene ~/nyiso-us/pipeline.
HOLDOUT RULE: nothing in this package may compute a price statistic for dates on or after
HOLDOUT_START. Counting rows, files and missing hours is allowed for every year.
"""
from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path.home() / "nyiso-us"
RAW = HOME / "raw"
GFS_RAW = RAW / "gfs_global"
PARQUET = HOME / "parquet"
RESULTS = HOME / "results"

TZ = "America/New_York"
FIRST_DAY = dt.date(2020, 1, 1)
LAST_DAY = dt.date(2026, 9, 30)
HOLDOUT_START = pd.Timestamp("2024-01-01", tz=TZ)

ZONES = ["CAPITL", "CENTRL", "DUNWOD", "GENESE", "HUD VL", "LONGIL",
         "MHK VL", "MILLWD", "N.Y.C.", "NORTH", "WEST"]
EXTERNAL = ["H Q", "NPX", "O H", "PJM"]          # border proxies in the zone files
ISOLF_COLS = {"Capitl": "CAPITL", "Centrl": "CENTRL", "Dunwod": "DUNWOD", "Genese": "GENESE",
              "Hud Vl": "HUD VL", "Longil": "LONGIL", "Mhk Vl": "MHK VL", "Millwd": "MILLWD",
              "N.Y.C.": "N.Y.C.", "North": "NORTH", "West": "WEST", "NYISO": "NYISO"}

# Publication rules (see README.md, section "published_at").
DECISION_HOUR = 5                                 # 05:00 New York time on bid day D
DA_POST_BOUND = dt.time(11, 0)                    # DAM results for D are public by 11:00 on D-1
RT_LAG = pd.Timedelta(minutes=15)                 # hourly RT price public 15 min after its hour ends
RT_REVISION_GRACE = pd.Timedelta(hours=0)         # RT file written after 24:00 of its day = revised
# GFS (Open-Meteo previous-runs archive). previous_dayN = the value predicted N*24 h before valid
# time. Three-day weather rule (OBJECTIVES addendum, 6 Oct evening): day2 for delivery hours up to
# 21:00 New York time, day3 for 22:00 and 23:00.
GFS_LEAD_HOURS = {2: 48, 3: 72}                   # previous_dayN -> nominal hours before valid time
GFS_NEAREST_RUN_SLACK = pd.Timedelta(hours=3)     # Open-Meteo may take the nearest 6-hourly run: up to 3 h later
GFS_PROCESSING = pd.Timedelta(hours=5)            # run init to public, conservative
GFS_DAY2_LAST_LOCAL_HOUR = 21                     # day2 used for hours beginning <= 21:00, day3 after


def gfs_published_at(target_hour: pd.Series, run_lead_hours) -> pd.Series:
    """published_at = valid time - lead + 3 h + 5 h: day2 = valid - 40 h, day3 = valid - 64 h."""
    lead = pd.to_timedelta(pd.Series(run_lead_hours, index=target_hour.index).astype("int64"), unit="h")
    return target_hour - lead + GFS_NEAREST_RUN_SLACK + GFS_PROCESSING


def month_keys():
    """'YYYYMM01' for every month Jan 2020 .. Sep 2026."""
    y, m = FIRST_DAY.year, FIRST_DAY.month
    while (y, m) <= (LAST_DAY.year, LAST_DAY.month):
        yield f"{y}{m:02d}01"
        m += 1
        if m == 13:
            y, m = y + 1, 1


def zip_path(ym: str, series: str) -> Path:
    return RAW / f"{ym}{series}_csv.zip"


def entries(series: str):
    """Yield (file_date, written_at_naive_local, ZipFile, ZipInfo) for every daily CSV of a series.
    Zip entry modification times are NYISO server wall-clock time (Eastern, DST-aware): DAM
    files read 09:35 in both January and July."""
    for ym in month_keys():
        p = zip_path(ym, series)
        if not p.exists():
            continue
        z = zipfile.ZipFile(p)
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            d = dt.datetime.strptime(info.filename[:8], "%Y%m%d").date()
            yield d, dt.datetime(*info.date_time), z, info


def localize_wall(naive: pd.Series, first_occurrence: np.ndarray) -> pd.Series:
    """NYISO labels rows in local wall-clock time. On the fall-back day 01:00 appears twice: the
    first occurrence (by row order) is EDT, the second EST. On the spring-forward day 02:00 does
    not exist and must not appear (nonexistent='raise')."""
    return naive.dt.tz_localize(TZ, ambiguous=np.asarray(first_occurrence, dtype=bool),
                                nonexistent="raise")


def localize_written(ts: list[dt.datetime]) -> pd.Series:
    """Zip entry times (naive Eastern wall clock) to tz-aware. An ambiguous fall-back time is
    read as the later (EST) instant and a nonexistent spring-forward time is shifted forward,
    both conservative (later)."""
    s = pd.Series(pd.to_datetime(ts))
    return s.dt.tz_localize(TZ, ambiguous=np.zeros(len(s), dtype=bool), nonexistent="shift_forward")


def local_at(midnights: pd.Series, t: dt.time) -> pd.Series:
    """Local wall-clock instant at time t on each date. `midnights` are naive datetime64 values
    at 00:00 (US DST switches at 02:00, so 00:00 and 11:00 are never ambiguous or missing)."""
    s = pd.to_datetime(midnights) + pd.Timedelta(hours=t.hour, minutes=t.minute)
    return s.dt.tz_localize(TZ)


def local_midnight(ts: pd.Series) -> pd.Series:
    """Naive 00:00 of the local date of tz-aware instants."""
    return ts.dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()


def later(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise later of two tz-aware Series; NaT in one side yields the other."""
    return a.where(b.isna() | (a >= b), b)


def decision_time(bid_day: dt.date) -> pd.Timestamp:
    """05:00 New York time on bid day D; positions are for delivery day D+1."""
    return pd.Timestamp(dt.datetime.combine(bid_day, dt.time(DECISION_HOUR)), tz=TZ)


def expected_hours(year: int) -> pd.DatetimeIndex:
    """Every local delivery hour of a year (8760 or 8784; the spring gap is not an hour),
    clipped to LAST_DAY."""
    start = pd.Timestamp(f"{year}-01-01", tz=TZ)
    end = min(pd.Timestamp(f"{year + 1}-01-01", tz=TZ),
              pd.Timestamp(LAST_DAY + dt.timedelta(days=1), tz=TZ))
    return pd.date_range(start, end, freq="h", inclusive="left")
