"""Rebuild yesterday: the nightly self-diff of the six versions.

Every night the platform re-fetches the recent past (three days for most
sources, a hundred for the settlement, which Energinet revises for months)
and rebuilds the marts, and the latest fetch wins. That is the right rule,
and it is also how a number changes under the desk without anyone
noticing. So before overwriting last night's rows this task compares them
with tonight's, version by version: which versions of which hours appeared,
which vanished, which changed by more than half a megawatt-hour, and the
largest changes by name. The log is a committed JSON file the page reads;
the state is a small Parquet file of the recent window, committed too, so
the comparison survives a fresh checkout in CI."""
import datetime as dt
import json
import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "tavle.duckdb"
STATE = ROOT / "data" / "state" / "wind_versions_recent.parquet"
LOG = ROOT / "docs" / "versions" / "revisions.json"
WINDOW_DAYS = 120       # wider than the hundred-day settlement re-fetch
EPS_MWH = 0.5           # below this a change is rounding, not a revision
VERSIONS = ["day-ahead", "intraday", "five-hour", "one-hour", "real-time", "settled"]


def current_rows(con, window_days=WINDOW_DAYS):
    """The recent window of wind_versions_long as (area, hour, version) -> value."""
    rows = con.execute(f"""
        select area, hour_utc, version, value_mwh from wind_versions_long
        where hour_utc >= (select max(hour_utc) from wind_versions_long) - interval {window_days} day
        order by 1, 2, 3""").fetchall()
    return {(a, h.isoformat(), v): float(x) for a, h, v, x in rows}


def previous_rows(state=STATE):
    if not pathlib.Path(state).exists():
        return None
    con = duckdb.connect()
    rows = con.execute(f"select area, hour_utc, version, value_mwh from read_parquet('{state}')").fetchall()
    con.close()
    return {(a, h.isoformat(), v): float(x) for a, h, v, x in rows}


def diff(prev, cur, eps=EPS_MWH, top=5):
    """Compare two {(area, hour, version): value} maps. Pure; the unit tests live here."""
    out = {"appeared": {v: 0 for v in VERSIONS}, "vanished": {v: 0 for v in VERSIONS},
           "revised": {v: {"n": 0, "max_abs_mwh": 0.0, "sum_abs_mwh": 0.0} for v in VERSIONS},
           "unchanged": 0, "largest": []}
    changes = []
    for k, x in cur.items():
        v = k[2]
        if k not in prev:
            out["appeared"][v] += 1
            continue
        d = abs(x - prev[k])
        if d > eps:
            r = out["revised"][v]
            r["n"] += 1
            r["max_abs_mwh"] = max(r["max_abs_mwh"], d)
            r["sum_abs_mwh"] += d
            changes.append((d, k, prev[k], x))
        else:
            out["unchanged"] += 1
    for k in prev:
        if k not in cur:
            out["vanished"][k[2]] += 1
    changes.sort(reverse=True)
    out["largest"] = [{"area": k[0], "hour_utc": k[1], "version": k[2], "before_mwh": round(a, 1),
                       "after_mwh": round(b, 1), "change_mwh": round(b - a, 1)} for _, k, a, b in changes[:top]]
    for v in VERSIONS:
        r = out["revised"][v]
        r["mean_abs_mwh"] = round(r["sum_abs_mwh"] / r["n"], 2) if r["n"] else None
        r["max_abs_mwh"] = round(r["max_abs_mwh"], 2)
        del r["sum_abs_mwh"]
    return out


def write_state(con, state=STATE, window_days=WINDOW_DAYS):
    pathlib.Path(state).parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        copy (select area, hour_utc, version, value_mwh from wind_versions_long
              where hour_utc >= (select max(hour_utc) from wind_versions_long) - interval {window_days} day)
        to '{state}' (format parquet)""")


def spans(con):
    row = con.execute("""
        select max(hour_utc) filter (where version = 'settled'),
               max(hour_utc) filter (where version = 'real-time'),
               max(hour_utc) filter (where version = 'day-ahead'),
               count(*)
        from wind_versions_long""").fetchone()
    settled_hi, realtime_hi, da_hi, n = row
    return {"settled_hi": settled_hi.isoformat() if settled_hi else None,
            "realtime_hi": realtime_hi.isoformat() if realtime_hi else None,
            "day_ahead_hi": da_hi.isoformat() if da_hi else None,
            "settlement_lag_days": (realtime_hi - settled_hi).days if settled_hi and realtime_hi else None,
            "rows": n}


def run(db=DB, state=STATE, log=LOG, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    con = duckdb.connect(str(db), read_only=True)
    try:
        cur = current_rows(con)
        entry = {"night": now.strftime("%Y-%m-%d"), "at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), **spans(con)}
        prev = previous_rows(state)
        if prev is None:
            entry["baseline"] = True
            entry["window_rows"] = len(cur)
        else:
            entry["baseline"] = False
            entry["window_rows"] = len(cur)
            entry.update(diff(prev, cur))
    finally:
        con.close()
    # the state is written by a fresh, writable, in-memory connection reading the file
    con = duckdb.connect(str(db), read_only=True)
    try:
        write_state(con, state)
    finally:
        con.close()
    pathlib.Path(log).parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(pathlib.Path(log).read_text()) if pathlib.Path(log).exists() else []
    # a re-run on the same night replaces that night's diff; a baseline is never replaced,
    # because the next run's diff is against it and the log should show where it started
    if entries and entries[-1].get("night") == entry["night"] and not entries[-1].get("baseline"):
        entries[-1] = entry
    else:
        entries.append(entry)
    pathlib.Path(log).write_text(json.dumps(entries, indent=1))
    if entry["baseline"]:
        return f"baseline written: {entry['window_rows']} rows in the window"
    rev = sum(r["n"] for r in entry["revised"].values())
    app = sum(entry["appeared"].values())
    return f"{app} rows appeared, {rev} revised, {entry['unchanged']} unchanged"


if __name__ == "__main__":
    print(run())
