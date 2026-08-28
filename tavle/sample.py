"""A committed slice of raw data so CI can build and test the marts with no
network at all.

The slice is not the most recent 45 days; it is the seam. Landing
2025-09-15 to 2025-10-31 means the committed sample contains the last
hourly prices, the first quarter-hourly ones, and the October DST night,
so every test that matters (gaps, seam continuity, quarter-hours in fours)
is exercised on every push rather than only against live data."""
import pathlib
import shutil
import sys

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SAMPLE = ROOT / "data" / "sample"
WINDOW = ("2025-09-15", "2025-11-01")
TS = {"Elspotprices": "HourUTC", "DayAheadPrices": "TimeUTC",
      "ProductionConsumptionSettlement": "HourUTC", "ecb_fx": "date"}


def snapshot(window=WINDOW):
    SAMPLE.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    lo, hi = window
    for ds, col in TS.items():
        src = RAW / ds
        if not src.exists() or not any(src.glob("*.parquet")):
            print(f"skip {ds}: nothing landed")
            continue
        out = SAMPLE / f"{ds}.parquet"
        con.execute(f"""
            copy (select * from read_parquet('{src}/*.parquet', union_by_name=true)
                  where cast({col} as timestamp) >= timestamp '{lo}'
                    and cast({col} as timestamp) <  timestamp '{hi}')
            to '{out}' (format parquet)""")
        n = con.execute(f"select count(*) from read_parquet('{out}')").fetchone()[0]
        print(f"snapshot {ds}: {n} rows -> {out}")


def restore():
    RAW.mkdir(parents=True, exist_ok=True)
    for ds in TS:
        src = SAMPLE / f"{ds}.parquet"
        dst = RAW / ds
        dst.mkdir(parents=True, exist_ok=True)
        if src.exists() and not any(dst.glob("*.parquet")):
            shutil.copy(src, dst / "part-sample.parquet")
            print(f"restored {ds}")


if __name__ == "__main__":
    {"snapshot": snapshot, "restore": restore}[sys.argv[1] if len(sys.argv) > 1 else "restore"]()
