"""Build docs/index.html: the work-sample page.

Written to the same doctrine as the other demos in this family. House
editorial style, a research question rather than a product name, the
problem before the machinery, one manipulable thing per idea, every
finding shipped with its remedy, and an honest failure pane. The data on
the page is written by the pipeline, never by hand.

The query console runs DuckDB compiled to WebAssembly against the
published Parquet marts, so a visitor's question is answered by the same
engine the platform uses, in their own browser, with the SQL shown."""
import datetime as dt
import json
import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "tavle.duckdb"
DOCS = ROOT / "docs"
OUT = DOCS / "index.html"
RUN_RESULTS = ROOT / "dbt" / "target" / "run_results.json"

# Every test, with what a desk loses if it is not there. A test list without
# this column is a list of file names.
TEST_MEANING = {
    "no_gaps_on_hourly_grid": "An hour missing anywhere between 2013 and tomorrow. A failed fetch looks exactly like a quiet market.",
    "seam_is_continuous": "The last hourly price and the first quarter-hourly one must be one hour apart. If a side of the seam is missing, every average that spans October 2025 is wrong.",
    "quarter_hours_come_in_fours": "Four prices per hour after the switch. Three or five means a DST night leaked through or a fetch was partial.",
    "fx_only_on_business_days": "The ECB does not publish on weekends. A Sunday rate means the parser invented one.",
    "interior_days_are_complete": "Every day except the first and last must have a full set of hours. The edges are partial by construction and are flagged, not averaged.",
    "wind_share_is_plausible": "Wind above three times consumption is a unit or join mistake, not a windy day.",
}


# ---------------------------------------------------------------------------
# The question catalogue. One definition, used three ways: the menus on the
# page, the SQL shown to the reader, and the answer computed here at build
# time. The console therefore answers instantly and correctly even if the
# in-browser engine is blocked, and when the engine is up it re-runs the same
# SQL live: the two agreeing is itself worth showing.
PERIODS = {
    "30":   ("last 30 days", "hour_utc >= now() - interval 30 day", "day_dk >= current_date - interval 30 day"),
    "365":  ("last 12 months", "hour_utc >= now() - interval 365 day", "day_dk >= current_date - interval 365 day"),
    "2025": ("2025", "hour_utc >= timestamp '2025-01-01' and hour_utc < timestamp '2026-01-01'",
             "day_dk >= date '2025-01-01' and day_dk < date '2026-01-01'"),
    "2022": ("2022, the price spike", "hour_utc >= timestamp '2022-01-01' and hour_utc < timestamp '2023-01-01'",
             "day_dk >= date '2022-01-01' and day_dk < date '2023-01-01'"),
    "all":  ("the whole series", "true", "true"),
}
METRICS = {
    "avg":    "Average price",
    "max":    "Highest price",
    "min":    "Lowest price",
    "neg":    "Negative-price hours",
    "spread": "DK1 minus DK2 spread",
    "wind":   "Average price by wind share",
}


def catalogue_sql(metric, area, period):
    _, wh, whd = PERIODS[period]
    if metric == "avg":
        return (f"select round(avg(price_eur), 2) as avg_eur_per_mwh, count(*) as hours\n"
                f"from power_hourly\nwhere area = '{area}' and {wh}")
    if metric == "max":
        return (f"select hour_utc, round(price_eur, 2) as eur_per_mwh\nfrom power_hourly\n"
                f"where area = '{area}' and {wh}\norder by price_eur desc limit 5")
    if metric == "min":
        return (f"select hour_utc, round(price_eur, 2) as eur_per_mwh\nfrom power_hourly\n"
                f"where area = '{area}' and {wh}\norder by price_eur asc limit 5")
    if metric == "neg":
        return (f"select count(*) as negative_hours,\n"
                f"       round(100.0 * count(*) / (select count(*) from power_hourly\n"
                f"                                 where area = '{area}' and {wh}), 2) as pct_of_hours\n"
                f"from power_hourly\nwhere area = '{area}' and price_eur < 0 and {wh}")
    if metric == "spread":
        return (f"select round(avg(spread_eur), 2) as avg_dk1_minus_dk2,\n"
                f"       round(min(spread_eur), 2) as widest_negative,\n"
                f"       round(max(spread_eur), 2) as widest_positive\n"
                f"from desk_daily\nwhere area = 'DK1' and spread_eur is not null and {whd}")
    return (f"select round(wind_share * 10) / 10 as wind_share_band, count(*) as hours,\n"
            f"       round(median(price_eur), 1) as median_eur\nfrom power_context\n"
            f"where area = '{area}' and wind_share is not null and {wh}\ngroup by 1 order by 1")


def build_catalogue(con):
    out = {}
    for metric in METRICS:
        for area in ("DK1", "DK2"):
            for period in PERIODS:
                sql = catalogue_sql(metric, area, period)
                cur = con.execute(sql)
                cols = [d[0] for d in cur.description]
                rows = [[jsonable(v) for v in r] for r in cur.fetchall()]
                out[f"{metric}|{area}|{period}"] = {"sql": sql, "cols": cols, "rows": rows}
    return out


def q(con, sql, params=None):
    cur = con.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def jsonable(v):
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    return v


def collect(db=DB):
    con = duckdb.connect(str(db), read_only=True)
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    d = {"built_at": now.isoformat(timespec="minutes") + "Z"}
    d["daily"] = q(con, """
        select area, day_dk, round(avg_eur,2) as avg_eur, negative_hours, is_complete,
               round(spread_eur,2) as spread
        from desk_daily where day_dk >= current_date - interval 400 day order by day_dk, area""")
    d["recent"] = q(con, """
        select instrument, interval_start_utc, round(value,2) as value
        from prices where unit = 'EUR/MWh' and interval_start_utc >= now()::timestamp - interval 10 day
        order by interval_start_utc""")
    d["freshness"] = q(con, """
        select instrument, max(interval_start_utc) as last_interval, count(*) as n_rows
        from prices group by 1 order by 1""")
    d["wind"] = q(con, """
        select round(wind_share,3) as wind_share, round(price_eur,1) as price_eur, area
        from power_context where hour_utc >= now()::timestamp - interval 365 day and wind_share is not null""")
    d["catalogue"] = build_catalogue(con)
    d["metrics"] = METRICS
    d["periods"] = {k: v[0] for k, v in PERIODS.items()}
    d["runs"] = q(con, """
        select task, attempt, status, started, epoch(finished - started) as seconds
        from ops.runs where run_id = (select max(run_id) from ops.runs) order by started""")
    con.close()

    tests = []
    if RUN_RESULTS.exists():
        for r in json.loads(RUN_RESULTS.read_text()).get("results", []):
            uid = r.get("unique_id", "")
            if uid.startswith("test."):
                name = uid.split(".")[2]
                tests.append({"name": name, "status": r.get("status"),
                              "means": TEST_MEANING.get(name.split("_")[0] if False else name, "")})
    d["tests"] = tests
    seam = DOCS / "seam.json"
    d["seam"] = json.loads(seam.read_text()) if seam.exists() else None
    ask = DOCS / "ask-results.json"
    d["ask"] = json.loads(ask.read_text()) if ask.exists() else None
    return json.loads(json.dumps(d, default=jsonable))


PAGE = pathlib.Path(__file__).with_name("page_template.html")


def build(db=DB, out=OUT):
    data = collect(db)
    html = PAGE.read_text().replace("__DATA__", json.dumps(data))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out, data


if __name__ == "__main__":
    path, data = build()
    print(f"wrote {path}: {len(data['daily'])} daily rows, {len(data['tests'])} tests, "
          f"seam {'yes' if data['seam'] else 'no'}, ask {'yes' if data['ask'] else 'no'}")
