"""Build docs/versions/index.html: six versions of one hour of wind.

Two kinds of number meet on this page and the file keeps them apart. The
study numbers come from research/results/versions.json, computed once by
research/versions_study.py on the full history and dated; the feed's gap
episodes and bias anatomy are part of that. The live numbers come from the
marts at build time: the last fortnight of versions, the newest hour that
has all six, the settlement lag, and the revision log the nightly
self-diff appends to. Nothing is typed in."""
import datetime as dt
import json
import pathlib
import re

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "tavle.duckdb"
RES = ROOT / "research" / "results" / "versions.json"
LOG = ROOT / "docs" / "versions" / "revisions.json"
RUN_RESULTS = ROOT / "dbt" / "target" / "run_results.json"
PROJECT = ROOT / "dbt" / "dbt_project.yml"
OUT = ROOT / "docs" / "versions" / "index.html"
TEMPLATE = pathlib.Path(__file__).with_name("versions_template.html")
TESTS = ["realtime_agrees_with_settlement", "realtime_feed_is_mostly_complete",
         "settlement_arrives_within_the_promised_lag", "realtime_bias_is_within_band"]
COLS = "v_day_ahead, v_intraday, v_5h, v_1h, v_realtime, realtime_readings, v_settled, versions_present"


def q(con, sql):
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def dbt_vars():
    s = PROJECT.read_text()
    return {k: float(v) for k, v in re.findall(r"^\s*(versions_\w+):\s*([-\d.]+)", s, flags=re.M)}


def test_status():
    if not RUN_RESULTS.exists():
        return None
    rr = json.loads(RUN_RESULTS.read_text())
    out = {}
    for r in rr.get("results", []):
        name = r.get("unique_id", "").split(".")[-1]
        for t in TESTS:
            if name.startswith(t):
                out[t] = {"status": r.get("status"), "failures": r.get("failures")}
    return {"at": rr.get("metadata", {}).get("generated_at"), "tests": out} if out else None


def collect():
    con = duckdb.connect(str(DB), read_only=True)
    d = {"study": json.loads(RES.read_text()), "vars": dbt_vars(), "tests": test_status(),
         "revisions": json.loads(LOG.read_text()) if LOG.exists() else [],
         "built_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    today = dt.datetime.now(dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    d["recent"] = q(con, f"""
        select area, hour_utc, versions_present
        from wind_versions where hour_utc >= timestamp '{today}' - interval 15 day and hour_utc < timestamp '{today}' order by 1, 2""")
    d["live_presets"] = []
    row = q(con, f"select area, hour_utc, {COLS} from wind_versions where versions_present = 6 and realtime_readings = 12 and area = 'DK1' order by hour_utc desc limit 1")
    if row:
        d["live_presets"].append({"why": "the newest DK1 hour that has all six versions, as of this build", **row[0]})
    y = today - dt.timedelta(hours=12)
    row = q(con, f"select area, hour_utc, {COLS} from wind_versions where area = 'DK1' and hour_utc = timestamp '{y}'")
    if row:
        d["live_presets"].append({"why": "yesterday at 12:00 UTC in DK1: the sixth version has not been published yet", **row[0]})
    d["spans"] = q(con, """
        select area,
            max(hour_utc) filter (where v_settled is not null)  as settled_hi,
            max(hour_utc) filter (where v_realtime is not null) as realtime_hi,
            max(hour_utc) filter (where v_day_ahead is not null) as day_ahead_hi,
            min(hour_utc) as first_hour, count(*) as hours,
            count(*) filter (where versions_present = 6) as hours_six
        from wind_versions group by 1 order by 1""")
    # Historical numbers (the feed's gap episodes, short hours, the bias
    # anatomy) come frozen from the study: this build's warehouse is the
    # committed sample plus recent increments, and a historical claim read
    # from it would describe the sample, not the record.
    d["gaps"] = d["study"]["gaps"]
    d["short_hours"] = d["study"]["short_hours"]
    d["anatomy"] = d["study"]["anatomy"]
    d["runs"] = q(con, """
        select run_id, task, status, started from ops.runs
        where task in ('extract_realtime', 'revisions', 'versions_page', 'dbt_build') order by started desc limit 8""")
    con.close()
    return d


def build():
    data = collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(TEMPLATE.read_text().replace("__DATA__", json.dumps(data, default=str)))
    return OUT, data


if __name__ == "__main__":
    p, d = build()
    print(f"wrote {p}: {len(d['recent'])} recent rows, {len(d['revisions'])} nights in the revision log, {len(d['gaps'])} feed gaps")
