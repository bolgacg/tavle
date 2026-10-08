"""Build the v2 parquet tables (delivery 2010-01-01 to 2023-12-31) in ~/nyiso-us/parquet_v2 with the v1
build code (research/us/pipeline/build_tables.py), generalised only by the year range in the v2
`common` shim. Same tables, columns and published_at rules as v1, plus weather_reforecast.

    cd ~/nyiso-us/pipeline_v2 && ../.venv/bin/python build_tables_v2.py all
        # or: prices | load | outages | weather_gfs | reforecast

v2 differences, all mechanical:
  * the real-time files carry a 7th column "Scarcity In Effect (Y/N)" in some years; only the first six
    columns are read (v1 read_price_month passes exactly six names, which would shift a 7-column file);
  * rows with delivery or target time on or after 2024-01-01 are dropped as they are read (holdout);
  * weather_gfs.parquet is v1's table filtered to target_hour < 2024-01-01 inside pyarrow (GFS archive
    starts 25 March 2021);
  * weather_reforecast.parquet: GEFS v12 reforecast 2 m temperature (fetch_reforecast.py), 2010-2019.
  * weather_gefs_joined.parquet (step gefs_joined, idea V16): the reforecast to 2019 plus the archived live
    GEFS v12 (fetch_gefs_live.py) from 2020-09-23, same rule; runs 2020-01-01..2020-09-22 do not exist.

Run mode (common.py, US_RUN_MODE): unset = the build above. dryrun / heldout write the same tables to
$US_RUN_DIR/parquet_v2 with the end date moved to the mode's last day (dryrun 2023-12-31, heldout
2026-09-30); every 2024-01-01 bound below is END = LAST_DAY + 1 day.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402  (v2 shim first: v1 modules then see the v2 range)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import build_tables as B  # noqa: E402  (v1 code)
from points import POINTS  # noqa: E402

assert B.PARQUET == C.PARQUET, "v1 build_tables did not pick up the v2 common shim"
END = pd.Timestamp(C.LAST_DAY + dt.timedelta(days=1), tz=C.TZ)          # 2024-01-01 00:00 New York (build)


def read_price_month(series: str, ym: str) -> pd.DataFrame:
    """v1 read_price_month with usecols=range(6) (older RT files have a 7th scarcity column)."""
    p = C.zip_path(ym, series)
    if not p.exists():
        return pd.DataFrame()
    z = zipfile.ZipFile(p)
    frames, written = [], {}
    for info in sorted(z.infolist(), key=lambda i: i.filename):
        d = dt.datetime.strptime(info.filename[:8], "%Y%m%d").date()
        df = pd.read_csv(z.open(info), header=0, names=B.PRICE_COLS, usecols=range(6),
                         dtype={"name": str, "ptid": "int64"})
        df["file_date"] = d
        written[d] = dt.datetime(*info.date_time)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    naive = pd.to_datetime(df["ts"], format="%m/%d/%Y %H:%M")
    first = df.assign(_n=naive).groupby(["file_date", "ptid", "_n"], sort=False).cumcount().to_numpy() == 0
    df["delivery_hour"] = C.localize_wall(naive, first)
    df["file_written_at"] = B.written_series(written, df["file_date"])
    df["file_date"] = pd.to_datetime(df["file_date"])
    B.LOG.setdefault("date_mismatch", {}).setdefault(series, 0)
    B.LOG["date_mismatch"][series] += int((C.local_midnight(df["delivery_hour"]) != df["file_date"]).sum())
    df = df[df["delivery_hour"] < END]
    return df.drop(columns=["ts"])


B.read_price_month = read_price_month


def _drop_after(path: Path, col: str):
    """Rewrite a parquet keeping rows with col < END (filter inside pyarrow; nothing computed)."""
    d = ds.dataset(str(path), format="parquet")
    typ = d.schema.field(col).type
    bound = B.pa.scalar(END, type=typ) if B.pa.types.is_timestamp(typ) else B.pa.scalar(END.date(), type=typ)
    t = d.to_table(filter=ds.field(col) < bound)
    B.LOG.setdefault("rows_dropped_holdout", {})[path.name] = d.count_rows() - t.num_rows
    pq.write_table(t, str(path) + ".tmp", compression="zstd")
    Path(str(path) + ".tmp").rename(path)


LF_EARLIEST = dt.time(6, 0)


def lf_published_at(issue_date: pd.Series, file_written_at: pd.Series) -> pd.Series:
    """v2 rule: later of the file's write time and 06:00 on the day before its issue date. NYISO writes isolf
    file D at about 07:05 on D-1, so this changes nothing for normal files. Three 2010 files (issue dates
    2010-04-27, 05-24, 05-29) carry write times of 00:00 to 01:02 on D-1; taken at face value, the file named
    D+1 would be usable at 05:00 on D, which v1's contract forbids (no vintage named after the bid day).
    The bound keeps that rule without editing the recorded write time."""
    bound = C.local_at(pd.to_datetime(issue_date) - pd.Timedelta(days=1), LF_EARLIEST)
    return C.later(file_written_at, bound)


def build_load():
    B.build_load()
    _drop_after(C.PARQUET / "load_forecast.parquet", "target_hour")
    f = C.PARQUET / "load_forecast.parquet"
    df = pd.read_parquet(f)
    new = lf_published_at(df["issue_date"], df["file_written_at"])
    B.LOG["load_published_at_raised_to_0600_dm1"] = {"rows": int((new != df["published_at"]).sum()),
                                                     "files": int(df.loc[new != df["published_at"], "issue_date"].nunique())}
    df["published_at"] = new
    df.to_parquet(str(f) + ".tmp", index=False, compression="zstd")
    os.replace(str(f) + ".tmp", f)


def build_weather_gfs():
    d = ds.dataset(str(C.V1_PARQUET / "weather_gfs.parquet"), format="parquet")
    typ = d.schema.field("target_hour").type
    df = d.to_table(filter=ds.field("target_hour") < B.pa.scalar(END, type=typ)).to_pandas()
    # Archived forecasts only (Bo's rule): every row names its archive, run time and lead. Open-Meteo's
    # previous_dayN is the run N*24 h before valid time, or the nearest 6-hourly run (up to 3 h later), so the
    # run time is recorded as its latest possible value; published_at = that + 5 h (v1 rule, unchanged).
    df["lead_hours"] = df["run_lead_hours"]
    df["issue_time"] = df["target_hour"] - pd.to_timedelta(df["run_lead_hours"], unit="h") + C.GFS_NEAREST_RUN_SLACK
    df["source"] = SOURCE_GFS
    df.to_parquet(C.PARQUET / "weather_gfs.parquet", index=False, compression="zstd")
    B.log(f"weather_gfs rows={len(df)} (v1 table, target_hour < {END.date()})")
    check_weather("weather_gfs")


def _gefs_frame(files, source: str, lag) -> pd.DataFrame:
    """weather_reforecast schema from the per-day JSONs (reforecast or archived live GEFS). Leads whose valid
    time is on or after 2024-01-01 are dropped as each file is read (holdout), before any row is made."""
    rows = []
    for f in files:
        r = json.loads(f.read_text())
        init = pd.Timestamp(r["init_utc"], tz="UTC")
        for lead, temps in r["leads"].items():
            if init + pd.Timedelta(hours=int(lead)) >= END:
                continue
            for p, v in temps.items():
                rows.append((p, init, int(lead), v))
    df = pd.DataFrame(rows, columns=["point", "init_utc", "lead_hours", "temperature_2m_c"])
    meta = pd.DataFrame([(p, z, lat, lon, prim) for p, (z, lat, lon, prim) in POINTS.items()],
                        columns=["point", "zone", "lat_req", "lon_req", "primary"])
    grid = {}
    if files:
        for p, (la, lo) in json.loads(files[0].read_text())["points"].items():
            grid[p] = (la, lo - 360 if lo > 180 else lo)
    meta["lat_grid"] = meta["point"].map(lambda p: grid.get(p, (np.nan, np.nan))[0])
    meta["lon_grid"] = meta["point"].map(lambda p: grid.get(p, (np.nan, np.nan))[1])
    df = df.merge(meta, on="point", how="left")
    df["target_hour"] = (df["init_utc"] + pd.to_timedelta(df["lead_hours"], unit="h")).dt.tz_convert(C.TZ)
    df["init_utc"] = df["init_utc"].dt.tz_convert(C.TZ)          # stored tz-aware NY like every table
    df["published_at"] = df["init_utc"] + lag
    df["member"] = "c00"
    df["issue_time"] = df["init_utc"]
    df["source"] = source
    df = df[df["target_hour"] < END]
    return df.sort_values(["point", "init_utc", "lead_hours"])[
        ["point", "zone", "primary", "lat_req", "lon_req", "lat_grid", "lon_grid", "init_utc", "lead_hours",
         "target_hour", "temperature_2m_c", "issue_time", "published_at", "member", "source"]]


def build_reforecast():
    """One row per point, 00 UTC init (day D) and lead (27..54 h, 3-hourly).
    published_at = init + 8 h (= 03:00 EST / 04:00 EDT on D), the same for every lead of a run."""
    files = sorted(C.REFORECAST_RAW.rglob("*.json"))
    df = _gefs_frame(files, SOURCE_REFORECAST, C.REFORECAST_PUBLISH_LAG)
    df.to_parquet(C.PARQUET / "weather_reforecast.parquet", index=False, compression="zstd")
    B.log(f"weather_reforecast rows={len(df)} runs={len(files)}")
    check_weather("weather_reforecast")


def gefs_live_files() -> list[Path]:
    """Archived live GEFS runs initialised on or before LAST_DAY (2023-12-31 in build and dry-run modes),
    chosen by FILE NAME: in those modes the held-out files (2024 onward) on disk are never opened."""
    out = []
    for f in sorted(C.GEFS_LIVE_RAW.glob("*/*.json")):
        d = dt.datetime.strptime(f.stem, "%Y%m%d").date()
        if C.GEFS_LIVE_FIRST <= d <= C.LAST_DAY:
            out.append(f)
    return out


def build_gefs_joined():
    """Idea V16: one consistent GEFS v12 source. The reforecast (runs 2010-01-01 to 2019-12-31) joined to the
    archived live GEFS v12 (runs 2020-09-23 to 2023-12-31), control member, 00 UTC, leads 27-54 h, same 12
    points, same rule published_at = init + 8 h. No run exists for 2020-01-01 to 2020-09-22 (not traded).
    The `source` column says which archive each row came from."""
    ref = sorted(f for f in C.REFORECAST_RAW.rglob("*.json") if f.stem < "20200101")
    live = gefs_live_files()
    df = pd.concat([_gefs_frame(ref, SOURCE_REFORECAST, C.REFORECAST_PUBLISH_LAG),
                    _gefs_frame(live, SOURCE_GEFS_LIVE, C.GEFS_LIVE_PUBLISH_LAG)], ignore_index=True)
    df = df.sort_values(["point", "init_utc", "lead_hours"]).reset_index(drop=True)
    df.to_parquet(C.PARQUET / "weather_gefs_joined.parquet", index=False, compression="zstd")
    B.LOG["gefs_joined_runs"] = {"reforecast": len(ref), "gefs_live": len(live)}
    B.log(f"weather_gefs_joined rows={len(df)} runs reforecast={len(ref)} live={len(live)}")
    check_weather("weather_gefs_joined")


SOURCE_GFS = "gfs_global previous runs (Open-Meteo previous-runs API, temperature_2m_previous_day2/3)"
SOURCE_REFORECAST = "GEFS reforecast v12 (AWS noaa-gefs-retrospective, control member c00, 00 UTC runs)"
SOURCE_GEFS_LIVE = "GEFS v12 live forecast archive (AWS noaa-gefs-pds pgrb2sp25, control member c00, 00 UTC runs)"
WEATHER_LAG = {"weather_gfs": C.GFS_PROCESSING, "weather_reforecast": C.REFORECAST_PUBLISH_LAG,
               "weather_gefs_joined": C.REFORECAST_PUBLISH_LAG}
WEATHER_SOURCE = {"weather_gfs": {SOURCE_GFS}, "weather_reforecast": {SOURCE_REFORECAST},
                  "weather_gefs_joined": {SOURCE_REFORECAST, SOURCE_GEFS_LIVE}}
GEFS_GAP = (pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp(C.GEFS_LIVE_FIRST, tz="UTC"))   # no run of either source


def weather_violations(name: str, df: pd.DataFrame) -> dict[str, int]:
    """Archived forecasts only: every row carries its archive, run time and lead; published_at = run time +
    the conservative publication delay; the run precedes the valid time by the lead; nothing is an analysis
    (lead > 0). Returns counts of violating rows per rule (empty = clean)."""
    v = {"source": int((~df["source"].isin(WEATHER_SOURCE[name])).sum()),
         "issue_time_missing": int(df["issue_time"].isna().sum()),
         "lead_not_positive": int((df["lead_hours"] <= 0).sum()),
         "published_rule": int((df["published_at"] != df["issue_time"] + WEATHER_LAG[name]).sum()),
         "published_not_after_issue": int((df["published_at"] <= df["issue_time"]).sum()),
         "issue_not_before_target": int((df["issue_time"] >= df["target_hour"]).sum())}
    if name == "weather_gefs_joined":
        init = df["issue_time"].dt.tz_convert("UTC")
        v["run_in_gap"] = int(((init >= GEFS_GAP[0]) & (init < GEFS_GAP[1])).sum())
        v["reforecast_after_2019"] = int(((df["source"] == SOURCE_REFORECAST) & (init >= GEFS_GAP[0])).sum())
        v["live_before_first"] = int(((df["source"] == SOURCE_GEFS_LIVE) & (init < GEFS_GAP[1])).sum())
        v["not_00utc"] = int((init.dt.hour != 0).sum())
    if name in ("weather_reforecast", "weather_gefs_joined"):
        v["lead_mismatch"] = int((df["target_hour"] != df["issue_time"] + pd.to_timedelta(df["lead_hours"], unit="h")).sum())
    else:
        v["lead_mismatch"] = int((df["target_hour"] - df["issue_time"] >
                                  pd.to_timedelta(df["lead_hours"], unit="h")).sum())
    return {k: n for k, n in v.items() if n}


def check_weather(name: str):
    df = pd.read_parquet(C.PARQUET / f"{name}.parquet")
    bad = weather_violations(name, df)
    if bad:
        raise SystemExit(f"{name}: weather rows that are not archived forecasts under the rule: {bad}")
    B.log(f"{name}: weather rule check clean ({len(df)} rows)")


def main(what: str):
    C.PARQUET.mkdir(parents=True, exist_ok=True)
    steps = {"prices": B.build_prices, "load": build_load, "outages": B.build_outages,
             "weather_gfs": build_weather_gfs, "reforecast": build_reforecast}
    if what == "gefs_joined":                       # V16 table, not part of "all"
        steps = {"gefs_joined": build_gefs_joined}
    t0 = time.time()
    for k, fn in steps.items():
        if what in (k, "all"):
            B.log(f"start {k}")
            fn()
    out = {k: (sorted(int(x) for x in v) if isinstance(v, set) else v) for k, v in B.LOG.items()}
    out["seconds"] = round(time.time() - t0)
    (C.PARQUET / f"build_log_{what}.json").write_text(json.dumps(out, indent=1, default=str))
    B.log(f"done {what} {json.dumps(out, default=str)[:2000]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
