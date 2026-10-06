"""Build the study's parquet tables on gene from ~/nyiso-us/raw into ~/nyiso-us/parquet.

    python build_tables.py all            # or: prices | load | outages | weather

Tables (every one carries published_at, America/New_York; rules in README.md):
  prices_zone.parquet      11 load zones, DA and hourly integrated RT, one row per zone-hour
  prices_gen/YYYYMM.parquet generator points (gen files) and the 4 border proxies (zone files)
  load_forecast.parquet    isolf: every vintage, issue date = file name date, per zone
  outages.parquet          outSched daily snapshots of scheduled transmission outages
  weather_gfs.parquet      Open-Meteo GFS temperature_2m_previous_day2, one point per zone

Congestion sign: NYISO publishes "Marginal Cost Congestion" with the opposite sign to the usual
convention, so LBMP = energy + losses - congestion_raw. Columns *_congestion_raw keep the file
value; *_congestion = -raw (positive = congestion raises the price); *_energy = LBMP - losses -
congestion. This script computes no price statistic; it only reshapes.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from common import (DA_POST_BOUND, EXTERNAL, GFS_PUB_OFFSET, GFS_RAW, ISOLF_COLS, PARQUET, RT_LAG,
                    RT_REVISION_GRACE, TZ, ZONES, entries, later, local_at, local_midnight,
                    localize_wall, localize_written, month_keys, zip_path)
from points import POINTS

PRICE_COLS = ["ts", "name", "ptid", "lbmp", "loss", "cong_raw"]
LOG: dict = {}


def log(msg: str):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------------------- prices
def read_price_month(series: str, ym: str) -> pd.DataFrame:
    """All rows of one monthly zip, with delivery_hour localized and the file's write time."""
    import zipfile
    p = zip_path(ym, series)
    if not p.exists():
        return pd.DataFrame()
    z = zipfile.ZipFile(p)
    frames, written = [], {}
    for info in sorted(z.infolist(), key=lambda i: i.filename):
        d = dt.datetime.strptime(info.filename[:8], "%Y%m%d").date()
        df = pd.read_csv(z.open(info), header=0, names=PRICE_COLS,
                         dtype={"name": str, "ptid": "int64"})
        df["file_date"] = d
        written[d] = dt.datetime(*info.date_time)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    naive = pd.to_datetime(df["ts"], format="%m/%d/%Y %H:%M")
    first = df.assign(_n=naive).groupby(["file_date", "ptid", "_n"], sort=False).cumcount().to_numpy() == 0
    df["delivery_hour"] = localize_wall(naive, first)
    df["file_written_at"] = written_series(written, df["file_date"])
    df["file_date"] = pd.to_datetime(df["file_date"])
    # Rows whose local delivery date differs from the file's date: counted, kept.
    LOG.setdefault("date_mismatch", {}).setdefault(series, 0)
    LOG["date_mismatch"][series] += int((local_midnight(df["delivery_hour"]) != df["file_date"]).sum())
    return df.drop(columns=["ts"])


def written_series(written: dict, dates: pd.Series) -> pd.Series:
    """Map file dates (python dates) to tz-aware zip entry write times, aligned to `dates`."""
    days = sorted(written)
    w = localize_written([written[d] for d in days])
    w.index = pd.to_datetime(pd.Series(days))
    return w.reindex(pd.to_datetime(dates)).set_axis(dates.index)


def dedupe(df: pd.DataFrame, series: str) -> pd.DataFrame:
    dup = df.duplicated(["delivery_hour", "ptid"], keep="last")
    LOG.setdefault("duplicate_rows_dropped", {}).setdefault(series, 0)
    LOG["duplicate_rows_dropped"][series] += int(dup.sum())
    return df[~dup]


def with_components(df: pd.DataFrame, pre: str) -> pd.DataFrame:
    out = df.rename(columns={"lbmp": f"{pre}_lbmp", "loss": f"{pre}_loss",
                             "cong_raw": f"{pre}_congestion_raw",
                             "file_written_at": f"{pre}_file_written_at"})
    out[f"{pre}_congestion"] = -out[f"{pre}_congestion_raw"]
    out[f"{pre}_energy"] = out[f"{pre}_lbmp"] - out[f"{pre}_loss"] - out[f"{pre}_congestion"]
    return out


def pair(da: pd.DataFrame, rt: pd.DataFrame) -> pd.DataFrame:
    """Outer-join DA and RT on (delivery_hour, ptid) and stamp publication times."""
    da = with_components(da, "da").drop(columns=["file_date"])
    rt = with_components(rt, "rt").drop(columns=["file_date"])
    m = da.merge(rt, on=["delivery_hour", "ptid"], how="outer", suffixes=("_da", "_rt"))
    name_da, name_rt = m.pop("name_da"), m.pop("name_rt")
    m["name"] = name_da.fillna(name_rt)
    dd = local_midnight(m["delivery_hour"])
    # Day-ahead: public by 11:00 on D-1 (posting is typically ~09:35 per the zip entry times),
    # or the file's write time if the file was rewritten later (a correction).
    bound = local_at(dd - pd.Timedelta(days=1), DA_POST_BOUND)
    m["da_published_at"] = later(bound, m["da_file_written_at"]).where(m["da_file_written_at"].notna())
    # Real-time hourly: public at end of hour + RT_LAG. If the daily file was rewritten after
    # midnight ending its day (a price correction), the archived value became public only at
    # that rewrite, so published_at = max(hour end + lag, rewrite time).
    hour_pub = m["delivery_hour"] + pd.Timedelta(hours=1) + RT_LAG
    day_end = local_at(dd + pd.Timedelta(days=1), dt.time(0, 0))
    m["rt_revised"] = (m["rt_file_written_at"] > day_end + RT_REVISION_GRACE).fillna(False).astype(bool)
    rt_pub = hour_pub.where(~m["rt_revised"], later(hour_pub, m["rt_file_written_at"]))
    m["rt_published_at"] = rt_pub.where(m["rt_file_written_at"].notna())
    m["published_at"] = later(m["da_published_at"], m["rt_published_at"])
    return m


ZONE_ORDER = ["delivery_hour", "zone", "ptid",
              "da_lbmp", "da_loss", "da_congestion_raw", "da_congestion", "da_energy",
              "rt_lbmp", "rt_loss", "rt_congestion_raw", "rt_congestion", "rt_energy",
              "da_published_at", "rt_published_at", "published_at", "rt_revised",
              "da_file_written_at", "rt_file_written_at"]
GEN_ORDER = ["delivery_hour", "ptid", "name", "point_type"] + ZONE_ORDER[3:]


def build_prices():
    (PARQUET / "prices_gen").mkdir(parents=True, exist_ok=True)
    zone_parts = []
    for ym in month_keys():
        t0 = time.time()
        daz, rtz = read_price_month("damlbmp_zone", ym), read_price_month("rtlbmp_zone", ym)
        dag, rtg = read_price_month("damlbmp_gen", ym), read_price_month("rtlbmp_gen", ym)
        # zones
        z = pair(dedupe(daz[daz.name.isin(ZONES)], "damlbmp_zone"),
                 dedupe(rtz[rtz.name.isin(ZONES)], "rtlbmp_zone") if len(rtz) else daz.iloc[0:0])
        z = z.rename(columns={"name": "zone"})
        zone_parts.append(z[ZONE_ORDER])
        # generators + border proxies
        g = pair(dedupe(dag, "damlbmp_gen"), dedupe(rtg, "rtlbmp_gen"))
        g["point_type"] = "gen"
        ext_da = daz[daz.name.isin(EXTERNAL)]
        ext_rt = rtz[rtz.name.isin(EXTERNAL)] if len(rtz) else daz.iloc[0:0]
        e = pair(dedupe(ext_da, "damlbmp_ext"), dedupe(ext_rt, "rtlbmp_ext"))
        e["point_type"] = "external"
        overlap = set(e.ptid) & set(g.ptid)
        LOG.setdefault("external_ptids_also_in_gen_files", set()).update(overlap)
        e = e[~e.ptid.isin(overlap)]
        gen = pd.concat([g, e], ignore_index=True)[GEN_ORDER].sort_values(["delivery_hour", "ptid"])
        pq.write_table(pa.Table.from_pandas(gen, preserve_index=False),
                       PARQUET / "prices_gen" / f"{ym[:6]}.parquet", row_group_size=200_000,
                       compression="zstd")
        log(f"prices {ym[:6]} zone={len(z)} gen={len(gen)} {time.time() - t0:.0f}s")
    zone = pd.concat(zone_parts, ignore_index=True).sort_values(["delivery_hour", "zone"])
    zone.to_parquet(PARQUET / "prices_zone.parquet", index=False, compression="zstd")
    log(f"prices_zone rows={len(zone)}")


# ------------------------------------------------------------------------------ load forecast
def build_load():
    frames, written = [], {}
    for d, w, z, info in entries("isolf"):
        df = pd.read_csv(z.open(info))
        df["issue_date"] = d
        written[d] = w
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    naive = pd.to_datetime(df.pop("Time Stamp"), format="%m/%d/%Y %H:%M")
    first = df.assign(_n=naive).groupby(["issue_date", "_n"], sort=False).cumcount().to_numpy() == 0
    df["target_hour"] = localize_wall(naive, first)
    df["file_written_at"] = written_series(written, df["issue_date"])
    long = df.rename(columns=ISOLF_COLS).melt(
        id_vars=["issue_date", "file_written_at", "target_hour"], value_vars=list(ISOLF_COLS.values()),
        var_name="zone", value_name="load_forecast_mw")
    long["published_at"] = long["file_written_at"]
    long["lead_days"] = (local_midnight(long["target_hour"]) - pd.to_datetime(long["issue_date"])).dt.days
    long = long.sort_values(["issue_date", "zone", "target_hour"])[
        ["issue_date", "file_written_at", "published_at", "target_hour", "lead_days", "zone", "load_forecast_mw"]]
    long.to_parquet(PARQUET / "load_forecast.parquet", index=False, compression="zstd")
    log(f"load_forecast rows={len(long)} files={len(written)}")


# ----------------------------------------------------------------------------------- outages
def parse_local(s: pd.Series) -> pd.Series:
    """Naive Eastern wall-clock strings to tz-aware; ambiguous = later (EST), gap = shifted."""
    t = pd.to_datetime(s, format="%m/%d/%Y %H:%M:%S", errors="coerce")
    return t.dt.tz_localize(TZ, ambiguous=np.zeros(len(t), dtype=bool), nonexistent="shift_forward")


def build_outages():
    frames, written = [], {}
    for d, w, z, info in entries("outSched"):
        df = pd.read_csv(z.open(info), dtype=str)
        df.columns = ["snapshot_ts_raw", "ptid", "equipment", "sched_out_raw", "sched_in_raw"]
        df["snapshot_date"] = d
        written[d] = w
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["file_written_at"] = written_series(written, df["snapshot_date"])
    df["published_at"] = df["file_written_at"]
    df["ptid"] = pd.to_numeric(df["ptid"], errors="coerce").astype("Int64")
    df["sched_out"] = parse_local(df["sched_out_raw"])
    df["sched_in"] = parse_local(df["sched_in_raw"])
    LOG["outage_unparsed_times"] = int(df["sched_out"].isna().sum() + df["sched_in"].isna().sum())
    df = df[["snapshot_date", "file_written_at", "published_at", "ptid", "equipment",
             "sched_out", "sched_in", "sched_out_raw", "sched_in_raw", "snapshot_ts_raw"]]
    df.to_parquet(PARQUET / "outages.parquet", index=False, compression="zstd")
    log(f"outages rows={len(df)} snapshots={len(written)}")


# ----------------------------------------------------------------------------------- weather
def build_weather():
    frames = []
    for name, (zone, lat, lon, primary) in POINTS.items():
        for f in sorted(GFS_RAW.glob(f"{name}_*.json")):
            d = json.loads(f.read_text())
            h = d["hourly"]
            t = pd.to_datetime(pd.Series(h["time"]), format="%Y-%m-%dT%H:%M").dt.tz_localize("UTC")
            df = pd.DataFrame({"point": name, "zone": zone, "primary": primary,
                               "lat_req": lat, "lon_req": lon,
                               "lat_grid": d.get("latitude"), "lon_grid": d.get("longitude"),
                               "target_hour": t.dt.tz_convert(TZ),
                               "temperature_2m_c": pd.to_numeric(pd.Series(h["temperature_2m_previous_day2"]),
                                                                 errors="coerce"),
                               "run_lead_hours": 48})
            frames.append(df)
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["point", "target_hour"], keep="last")
    # Open-Meteo previous_day2 = value predicted >= 48 h before the target hour (run init <=
    # target - 48 h); GFS output is public about 3.5-5 h after init, so 6 h is a safe bound.
    df["published_at"] = df["target_hour"] - GFS_PUB_OFFSET
    df = df.sort_values(["point", "target_hour"])
    df.to_parquet(PARQUET / "weather_gfs.parquet", index=False, compression="zstd")
    log(f"weather_gfs rows={len(df)} points={df.point.nunique()}")


def main(what: str):
    PARQUET.mkdir(parents=True, exist_ok=True)
    steps = {"prices": build_prices, "load": build_load, "outages": build_outages, "weather": build_weather}
    for k, fn in steps.items():
        if what in (k, "all"):
            log(f"start {k}")
            fn()
    out = {k: (sorted(int(x) for x in v) if isinstance(v, set) else v) for k, v in LOG.items()}
    (PARQUET / f"build_log_{what}.json").write_text(json.dumps(out, indent=1, default=str))
    log(f"done {what} {json.dumps(out, default=str)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
