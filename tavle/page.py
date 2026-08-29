"""Build docs/index.html, the work-sample page.

One story in three acts. A desk asks a plain question and gets a wrong
answer with no error attached; the reader sees why, then breaks the data
themselves and watches the tests catch it, then puts their own question
to the warehouse in their browser. Everything on the page is computed
here at build time; the browser engine only re-confirms it live."""
import datetime as dt
import json
import pathlib

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "tavle.duckdb"
DOCS = ROOT / "docs"
DATA = DOCS / "data"
OUT = DOCS / "index.html"
RUN_RESULTS = ROOT / "dbt" / "target" / "run_results.json"
TEMPLATE = pathlib.Path(__file__).with_name("page_template.html")


def jsonable(v):
    if isinstance(v, (dt.datetime, dt.date)):
        return v.isoformat()
    return v


# ---------------------------------------------------------------------------
# The published marts. The browser queries these same files.
def export_marts(con):
    DATA.mkdir(parents=True, exist_ok=True)
    con.execute(f"""copy (select area, hour_utc, round(price_eur,2) as price_eur, source
        from power_hourly order by area, hour_utc) to '{DATA}/power_hourly.parquet' (format parquet, compression zstd)""")
    con.execute(f"""copy (select area, day_dk, round(avg_eur,2) as avg_eur, round(low_eur,2) as low_eur,
        round(high_eur,2) as high_eur, negative_hours, hours, is_complete, round(spread_eur,2) as spread_eur,
        round(eurdkk,4) as eurdkk, round(avg_dkk,2) as avg_dkk
        from desk_daily order by day_dk, area) to '{DATA}/desk_daily.parquet' (format parquet, compression zstd)""")
    con.execute(f"""copy (select area, hour_utc, round(price_eur,2) as price_eur, round(wind_mwh,1) as wind_mwh,
        round(solar_mwh,1) as solar_mwh, round(consumption_mwh,1) as consumption_mwh, round(wind_share,4) as wind_share
        from power_context order by area, hour_utc) to '{DATA}/power_context.parquet' (format parquet, compression zstd)""")
    con.execute(f"""copy (select rate_date, currency, rate from stg_fx order by rate_date, currency)
        to '{DATA}/fx.parquet' (format parquet, compression zstd)""")


# ---------------------------------------------------------------------------
# Act 2: the tests, written once as SQL the browser can also run. Each returns
# a single count; anything above zero is a failure. {ph} {fx} {pc} are the
# table names, which the fault injector swaps for altered views.
TESTS = [
    ("no gaps on the hourly grid",
     "An hour missing anywhere between 2013 and tomorrow. A failed download looks exactly like a quiet market.",
     "select count(*) from (select area, hour_utc, lead(hour_utc) over (partition by area order by hour_utc) as nxt from {ph}) where nxt - hour_utc <> interval 1 hour"),
    ("one price per hour per zone",
     "The same hour landing twice, which happens on the night the clocks go back and whenever a window is fetched twice.",
     "select count(*) from (select area, hour_utc, count(*) as c from {ph} group by 1, 2 having c > 1)"),
    ("the two datasets meet exactly",
     "The last hourly price and the first quarter-hourly one must be one hour apart. If a side is missing, every average across October 2025 is wrong.",
     "with o as (select area, max(hour_utc) as t from {ph} where source = 'Elspotprices' group by 1), n as (select area, min(hour_utc) as t from {ph} where source = 'DayAheadPrices' group by 1) select count(*) from o join n using (area) where n.t - o.t <> interval 1 hour"),
    ("every interior day is complete",
     "A day with fewer than 23 hours or more than 25 is a partial download, not a short day. The first and last days are partial by construction and are exempt.",
     "with d as (select area, cast(hour_utc as date) as day, count(*) as h from {ph} group by 1, 2), b as (select area, min(day) as lo, max(day) as hi from d group by 1) select count(*) from d join b using (area) where day > lo and day < hi and h not in (23, 24, 25)"),
    ("exchange rates only on business days",
     "The European Central Bank does not publish on weekends. A Sunday rate means the parser invented one.",
     "select count(*) from {fx} where dayofweek(rate_date) in (0, 6)"),
    ("wind share is physically possible",
     "Wind above three times consumption is a units or join mistake, not a windy day.",
     "select count(*) from {pc} where wind_share < 0 or wind_share > 3"),
]

# The faults a reader can inject. Each is a view definition over the base
# tables; the tests then run against the altered view.
FAULTS = [
    ("none", "Leave the data alone", "The platform as published.",
     {}),
    ("drop_hours", "Lose six hours of a day", "Remove 06:00 to 11:00 on 14 March 2026 in DK1, as a download that timed out halfway would.",
     {"ph": "select * from power_hourly where not (area = 'DK1' and hour_utc >= timestamp '2026-03-14 06:00:00' and hour_utc < timestamp '2026-03-14 12:00:00')"}),
    ("dup_hour", "Duplicate the clock-change hour", "Land the repeated hour of 26 October 2025 twice in DK2, as a naive loader does on the night the clocks go back.",
     {"ph": "select * from power_hourly union all select * from power_hourly where area = 'DK2' and hour_utc = timestamp '2025-10-26 00:00:00'"}),
    ("seam", "Lose the first quarter-hourly day", "Drop 1 October 2025 from the new dataset in DK1, so the old series and the new one no longer touch.",
     {"ph": "select * from power_hourly where not (area = 'DK1' and source = 'DayAheadPrices' and hour_utc < timestamp '2025-10-01 22:00:00')"}),
    ("sunday_fx", "Invent a Sunday exchange rate", "Add a EUR/DKK rate for Sunday 23 August 2026, as a parser that fills gaps forward would.",
     {"fx": "select * from fx union all select date '2026-08-23', 'DKK', 7.4753"}),
    ("wind_units", "Confuse kilowatt-hours with megawatt-hours", "Multiply the wind column by a thousand, the classic units mistake.",
     {"pc": "select area, hour_utc, price_eur, wind_mwh * 1000 as wind_mwh, solar_mwh, consumption_mwh, wind_share * 1000 as wind_share from power_context"}),
]


def run_faults(con):
    """Precompute, for every fault, how many rows each test returns."""
    con.execute(f"create or replace view power_hourly_x as select * from read_parquet('{DATA}/power_hourly.parquet')")
    con.execute(f"create or replace view fx_x as select * from read_parquet('{DATA}/fx.parquet')")
    con.execute(f"create or replace view power_context_x as select * from read_parquet('{DATA}/power_context.parquet')")
    out = []
    for key, label, what, views in FAULTS:
        names = {"ph": "power_hourly_x", "fx": "fx_x", "pc": "power_context_x"}
        for k, sql in views.items():
            v = f"fault_{key}_{k}"
            sql_x = sql.replace("power_hourly", "power_hourly_x").replace("from fx", "from fx_x").replace("power_context", "power_context_x")
            con.execute(f"create or replace view {v} as {sql_x}")
            names[k] = v
        results = []
        for name, meaning, sql in TESTS:
            n = con.execute(sql.format(**names)).fetchone()[0]
            results.append(int(n))
        out.append({"key": key, "label": label, "what": what, "views": views, "results": results})
    return out


# ---------------------------------------------------------------------------
# Act 3: the question catalogue, answered at build time.
PERIODS = {
    "30":   ("the last 30 days", "hour_utc >= now() - interval 30 day", "day_dk >= current_date - interval 30 day"),
    "365":  ("the last 12 months", "hour_utc >= now() - interval 365 day", "day_dk >= current_date - interval 365 day"),
    "2025": ("2025", "hour_utc >= timestamp '2025-01-01' and hour_utc < timestamp '2026-01-01'",
             "day_dk >= date '2025-01-01' and day_dk < date '2026-01-01'"),
    "2022": ("2022, the energy crisis", "hour_utc >= timestamp '2022-01-01' and hour_utc < timestamp '2023-01-01'",
             "day_dk >= date '2022-01-01' and day_dk < date '2023-01-01'"),
    "all":  ("the whole series since 2013", "true", "true"),
}
METRICS = {
    "avg": "the average price", "neg": "how many hours went below zero", "max": "the five most expensive hours",
    "min": "the five cheapest hours", "spread": "how far apart the two zones were", "wind": "price by how much wind there was",
}


def catalogue_sql(metric, area, period):
    _, wh, whd = PERIODS[period]
    if metric == "avg":
        return f"select round(avg(price_eur), 2) as avg_eur_per_mwh, count(*) as hours\nfrom power_hourly\nwhere area = '{area}' and {wh}"
    if metric == "max":
        return f"select hour_utc, round(price_eur, 2) as eur_per_mwh\nfrom power_hourly\nwhere area = '{area}' and {wh}\norder by price_eur desc limit 5"
    if metric == "min":
        return f"select hour_utc, round(price_eur, 2) as eur_per_mwh\nfrom power_hourly\nwhere area = '{area}' and {wh}\norder by price_eur asc limit 5"
    if metric == "neg":
        return (f"select count(*) as negative_hours,\n       round(100.0 * count(*) / (select count(*) from power_hourly\n"
                f"                                 where area = '{area}' and {wh}), 2) as pct_of_hours\n"
                f"from power_hourly\nwhere area = '{area}' and price_eur < 0 and {wh}")
    if metric == "spread":
        return (f"select round(avg(spread_eur), 2) as avg_dk1_minus_dk2,\n       round(min(spread_eur), 2) as widest_negative,\n"
                f"       round(max(spread_eur), 2) as widest_positive\nfrom desk_daily\nwhere area = 'DK1' and spread_eur is not null and {whd}")
    return (f"select round(wind_share * 10) / 10 as wind_share_band, count(*) as hours,\n       round(median(price_eur), 1) as median_eur\n"
            f"from power_context\nwhere area = '{area}' and wind_share is not null and {wh}\ngroup by 1 order by 1")


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


# ---------------------------------------------------------------------------
BUGS = [
    ["Daily bars at the edges were partial", "The first and last day of any window have fewer hours, so their averages were quietly wrong.", "Completeness is flagged per day and the test applies to interior days rather than being weakened."],
    ["The scheduler locked out its own build step", "The runner held the warehouse's write lock for a whole run, so dbt could not open the file it was about to build. The run ledger recorded the failure, which is the argument for having one.", "The ledger opens per write; a regression test guards it."],
    ["Views carried the directory they were built in", "A view over Parquet keeps a relative path in its definition, so the warehouse only opened from the folder dbt ran in.", "Staging is materialised as tables."],
    ["The SQL model could not see the wind table", "The read-only guard listed four tables, not five, so every question about wind was refused and scored against the model. My bug, marked as its miss.", "Fixed, and the evaluation rerun with the miss recorded."],
]


def collect(db=DB):
    con = duckdb.connect(str(db), read_only=True)
    export_marts(con)
    d = {"built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes").replace("+00:00", "Z")}
    d["stats"] = {
        "rows": con.execute("select count(*) from prices").fetchone()[0],
        "hours": con.execute("select count(*) from power_hourly").fetchone()[0],
        "first": str(con.execute("select min(hour_utc) from power_hourly").fetchone()[0])[:10],
        "last": str(con.execute("select max(interval_start_utc) from prices").fetchone()[0])[:16].replace("T", " "),
        "negative": con.execute("select count(*) from power_hourly where price_eur < 0").fetchone()[0],
    }
    d["catalogue"] = build_catalogue(con)
    d["metrics"] = METRICS
    d["periods"] = {k: v[0] for k, v in PERIODS.items()}
    d["run"] = [dict(zip(["task", "status", "seconds"], r)) for r in con.execute("""
        select task, status, round(epoch(finished - started), 1) from ops.runs
        where run_id = (select max(run_id) from ops.runs) order by started""").fetchall()]
    con.close()

    # the fault injector runs against the published parquet, the same files the browser gets
    mem = duckdb.connect()
    d["tests"] = [{"name": n, "means": m, "sql": s} for n, m, s in TESTS]
    d["faults"] = run_faults(mem)
    mem.close()

    if RUN_RESULTS.exists():
        rr = json.loads(RUN_RESULTS.read_text())["results"]
        t = [r for r in rr if r["unique_id"].startswith("test.")]
        d["dbt_tests"] = {"n": len(t), "passing": sum(1 for r in t if r["status"] == "pass")}
    else:
        d["dbt_tests"] = {"n": 0, "passing": 0}
    seam = DOCS / "seam.json"
    d["seam"] = json.loads(seam.read_text()) if seam.exists() else None
    ask = DOCS / "ask-results.json"
    d["ask"] = json.loads(ask.read_text()) if ask.exists() else None
    d["bugs"] = BUGS
    return json.loads(json.dumps(d, default=jsonable))


def build(db=DB, out=OUT):
    data = collect(db)
    html = TEMPLATE.read_text().replace("__DATA__", json.dumps(data))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out, data


if __name__ == "__main__":
    path, data = build()
    print(f"wrote {path}: {len(data['catalogue'])} answers, {len(data['faults'])} faults, "
          f"{data['dbt_tests']['passing']}/{data['dbt_tests']['n']} dbt tests, ask {'yes' if data['ask'] else 'no'}")
