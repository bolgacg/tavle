"""A committed slice of raw data so CI can build and test the marts without
network, and a snapshot command the nightly lane uses to refresh it.

`snapshot` keeps the last 45 days of every raw dataset as one Parquet
file each under data/sample/. `restore` copies those into data/raw/ when
data/raw/ is empty, which is exactly the state of a fresh checkout."""
import pathlib
import shutil
import sys

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SAMPLE = ROOT / "data" / "sample"
TS = {"Elspotprices": "HourUTC", "DayAheadPrices": "TimeUTC",
      "ProductionConsumptionSettlement": "HourUTC", "ecb_fx": "date"}


def snapshot(days=45):
    SAMPLE.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    for ds, col in TS.items():
        src = RAW / ds
        if not src.exists() or not any(src.glob("*.parquet")):
            continue
        out = SAMPLE / f"{ds}.parquet"
        con.execute(f"""
            copy (select * from read_parquet('{src}/*.parquet', union_by_name=true)
                  where cast({col} as timestamp) >= (select max(cast({col} as timestamp)) from read_parquet('{src}/*.parquet', union_by_name=true)) - interval {days} day)
            to '{out}' (format parquet)""")
        print(f"snapshot {ds}: {out}")


def restore():
    for ds in TS:
        src = SAMPLE / f"{ds}.parquet"
        dst = RAW / ds
        if src.exists() and not (dst.exists() and any(dst.glob("*.parquet"))):
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst / "part-sample.parquet")
            print(f"restored {ds}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "restore"
    {"snapshot": snapshot, "restore": restore}[cmd]()
