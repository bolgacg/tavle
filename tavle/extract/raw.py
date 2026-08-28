"""Raw landing zone: whatever the source said, as Parquet, with three
bookkeeping columns so any number on the desk page can be walked back to
the request that produced it."""
import datetime as dt
import pathlib

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

RAW = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"


def land(records, dataset, request_url, raw_dir=RAW):
    """Write one batch of records for `dataset` and return the file path.
    Empty batches write nothing and return None."""
    if not records:
        return None
    fetched_at = dt.datetime.now(dt.timezone.utc)
    table = pa.Table.from_pylist(records)
    n = len(table)
    table = table.append_column("_source", pa.array([dataset] * n))
    table = table.append_column("_fetched_at", pa.array([fetched_at] * n, pa.timestamp("us", tz="UTC")))
    table = table.append_column("_request", pa.array([request_url] * n))
    out_dir = pathlib.Path(raw_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"part-{fetched_at.strftime('%Y%m%dT%H%M%S%f')}.parquet"
    pq.write_table(table, path)
    return path


def watermark(dataset, ts_column, raw_dir=RAW):
    """Latest timestamp already landed for `dataset`, or None."""
    files = sorted((pathlib.Path(raw_dir) / dataset).glob("*.parquet"))
    if not files:
        return None
    con = duckdb.connect()
    glob = str(pathlib.Path(raw_dir) / dataset / "*.parquet")
    row = con.execute(
        f"select max(cast({ts_column} as timestamp)) from read_parquet('{glob}', union_by_name=true)"
    ).fetchone()
    return row[0]
