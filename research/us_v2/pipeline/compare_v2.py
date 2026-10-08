"""Compare two v2 data trees file by file: every table, prices_gen month, feature part and feature matrix.

    ../.venv/bin/python -I compare_v2.py ~/nyiso-us/parquet_v2 $US_RUN_DIR/parquet_v2
    ../.venv/bin/python -I compare_v2.py ~/nyiso-us/parquet_v2 $US_RUN_DIR/parquet_v2 --before 2024-01-01

Equal = same columns and arrow types, same row count, and the same multiset of row hashes
(pandas.util.hash_pandas_object, index ignored), so rows are equal after sorting. "order" says whether
the rows are also in the same order. With --before, both sides are filtered inside pyarrow to rows whose
time column is before that date (bid_date: before the day before it), and B may hold extra files (later
months and years) that are listed by name only. The feature matrices are matched by prefix
(panel_2010_2023 with panel_2010_2026). Exit status 1 if anything differs. Hashes and counts only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

TIME_COL = {"prices_zone": "delivery_hour", "load_forecast": "target_hour", "outages": "snapshot_date",
            "weather_gfs": "target_hour", "weather_reforecast": "target_hour",
            "weather_gefs_joined": "target_hour", "border_hourly": "bid_date", "border_day": "bid_date",
            "wxr_hourly": "delivery_hour", "wxr_joined_hourly": "delivery_hour"}


def key(rel: str) -> str:
    """Name used to pair files: the feature matrices drop their last year."""
    return re.sub(r"^features/(panel|day)_(\d{4})_\d{4}\.parquet$", r"features/\1_\2", rel)


def time_col(rel: str) -> str:
    name = Path(rel).stem
    if rel.startswith("prices_gen/") or name.startswith("panel"):
        return "delivery_hour"
    if name.startswith("day"):
        return "delivery_date"
    return TIME_COL[name]


def bound(typ: pa.DataType, d: dt.date):
    if pa.types.is_timestamp(typ):
        return pa.scalar(pd.Timestamp(d, tz=typ.tz) if typ.tz else pd.Timestamp(d), type=typ)
    if pa.types.is_date(typ):
        return pa.scalar(d, type=typ)
    raise TypeError(typ)


def read(path: Path, col: str | None, before: dt.date | None) -> tuple[pa.Schema, pd.DataFrame]:
    d = ds.dataset(str(path), format="parquet")
    flt = None
    if before is not None:
        b = before - dt.timedelta(days=1) if col == "bid_date" else before
        flt = ds.field(col) < bound(d.schema.field(col).type, b)
    t = d.to_table(filter=flt)
    return t.schema.remove_metadata(), t.to_pandas()


def files(root: Path) -> dict[str, str]:
    out = {}
    for f in sorted(root.rglob("*.parquet")):
        rel = f.relative_to(root).as_posix()
        out[key(rel)] = rel
    return out


def compare(a: Path, b: Path, before: dt.date | None) -> dict:
    fa, fb = files(a), files(b)
    res = {"only_in_a": sorted(set(fa) - set(fb)), "only_in_b": sorted(set(fb) - set(fa)), "files": {}}
    for k in sorted(set(fa) & set(fb)):
        col = time_col(fa[k]) if before is not None else None
        sa, da = read(a / fa[k], col, before)
        sb, db = read(b / fb[k], col, before)
        r = {"rows_a": len(da), "rows_b": len(db), "schema": sa.equals(sb)}
        if r["schema"] and len(da) == len(db):
            ha = pd.util.hash_pandas_object(da, index=False).to_numpy()
            hb = pd.util.hash_pandas_object(db, index=False).to_numpy()
            r["order"] = bool((ha == hb).all())
            r["equal"] = bool((np.sort(ha) == np.sort(hb)).all())
        else:
            r["equal"] = False
        res["files"][f"{fa[k]} | {fb[k]}" if fa[k] != fb[k] else fa[k]] = r
        print(("OK  " if r["equal"] else "DIFF"), fa[k], fb[k] if fa[k] != fb[k] else "", r, flush=True)
    res["all_equal"] = (all(r["equal"] for r in res["files"].values()) and not res["only_in_a"]
                        and (before is not None or not res["only_in_b"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--before", type=dt.date.fromisoformat)
    ap.add_argument("--out", type=Path)
    x = ap.parse_args()
    res = compare(x.a.expanduser(), x.b.expanduser(), x.before)
    print(f"{len(res['files'])} files compared, only in A: {res['only_in_a']}, only in B: {len(res['only_in_b'])} "
          f"files, all equal: {res['all_equal']}", flush=True)
    if x.out:
        x.out.write_text(json.dumps(res, indent=1))
    sys.exit(0 if res["all_equal"] else 1)


if __name__ == "__main__":
    main()
