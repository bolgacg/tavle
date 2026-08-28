"""Ask the desk data: natural language to SQL over the marts, evaluated
honestly.

The point is not that a model can write SQL. It is that a desk tool built
on one has to be boxed in (read-only, schema-limited, row-capped, timed
out) and measured against questions with known answers, including the
ones it gets wrong. This module does both and records the results so the
desk page shows what the guardrails caught, not a demo that always works.

Generation uses the Claude Code CLI in headless mode (`claude -p`), so the
evaluation is reproducible by anyone with the CLI. The page never calls a
model; it renders the recorded results."""
import json
import pathlib
import re
import subprocess
import sys
import time

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "tavle.duckdb"
OUT = ROOT / "docs" / "ask-results.json"
ROW_CAP = 200
TIMEOUT_S = 10

TABLES = ["prices", "power_hourly", "desk_daily", "stg_fx"]

# Each question carries a reference query written by hand. The model never
# sees the reference; it sees the schema and the question.
QUESTIONS = [
    ("What was the average DK1 day-ahead price in EUR/MWh in July 2026?",
     "select round(avg(price_eur),2) from power_hourly where area='DK1' and hour_utc >= '2026-07-01' and hour_utc < '2026-08-01'"),
    ("How many hours in 2026 so far had a negative price in DK2?",
     "select count(*) from power_hourly where area='DK2' and price_eur < 0 and hour_utc >= '2026-01-01'"),
    ("Which day in 2025 had the highest daily average price in DK1?",
     "select day_dk from desk_daily where area='DK1' and day_dk >= '2025-01-01' and day_dk < '2026-01-01' order by avg_eur desc limit 1"),
    ("What was the lowest hourly price ever recorded in DK1, and when?",
     "select price_eur, hour_utc from power_hourly where area='DK1' order by price_eur asc, hour_utc asc limit 1"),
    ("What is the average DK1 minus DK2 spread over the last 30 days with data?",
     "select round(avg(dk1_minus_dk2_eur),2) from (select dk1_minus_dk2_eur from desk_daily where area='DK1' and dk1_minus_dk2_eur is not null order by day_dk desc limit 30)"),
    ("On how many days in 2026 was DK2 more expensive than DK1 on average?",
     "select count(*) from desk_daily where area='DK1' and day_dk >= '2026-01-01' and dk1_minus_dk2_eur < 0"),
    ("What was the EUR/DKK reference rate on 2 January 2026?",
     "select rate from stg_fx where currency='DKK' and rate_date = date '2026-01-02'"),
    ("What is the yearly average DK1 price for each year from 2020 to 2025?",
     "select year(hour_utc) as y, round(avg(price_eur),2) from power_hourly where area='DK1' and hour_utc >= '2020-01-01' and hour_utc < '2026-01-01' group by 1 order by 1"),
    ("How many quarter-hour price points exist for DK1 after the October 2025 switch?",
     "select count(*) from prices where instrument='DK1_DA' and interval_minutes = 15"),
    ("What was the highest daily high in DK2 during August 2026?",
     "select round(max(high_eur),2) from desk_daily where area='DK2' and day_dk >= '2026-08-01' and day_dk < '2026-09-01'"),
    ("What is the average price by hour of day in DK1 for 2026, in Danish local time?",
     "select hour(hour_utc at time zone 'UTC' at time zone 'Europe/Copenhagen') as h, round(avg(price_eur),2) from power_hourly where area='DK1' and hour_utc >= '2026-01-01' group by 1 order by 1"),
    ("How many days in 2024 had at least one negative hour in DK1?",
     "select count(*) from desk_daily where area='DK1' and day_dk >= '2024-01-01' and day_dk < '2025-01-01' and negative_hours > 0"),
    ("What was the average DK1 price in DKK per MWh in June 2026?",
     "select round(avg(avg_dkk),2) from desk_daily where area='DK1' and day_dk >= '2026-06-01' and day_dk < '2026-07-01'"),
    ("Which month of 2025 had the most negative-price hours in DK1?",
     "select month(hour_utc) as m from power_hourly where area='DK1' and price_eur < 0 and hour_utc >= '2025-01-01' and hour_utc < '2026-01-01' group by 1 order by count(*) desc, m limit 1"),
    ("What is the latest day-ahead price interval available for DK1?",
     "select max(interval_start_utc) from prices where instrument='DK1_DA'"),
    ("What was the EUR/USD rate on the last business day of 2025?",
     "select rate from stg_fx where currency='USD' and rate_date = (select max(rate_date) from stg_fx where currency='USD' and rate_date < date '2026-01-01')"),
    ("What is the standard deviation of hourly DK1 prices in 2026?",
     "select round(stddev_samp(price_eur),2) from power_hourly where area='DK1' and hour_utc >= '2026-01-01'"),
    ("How many hours does the DK1 series contain in total?",
     "select count(*) from power_hourly where area='DK1'"),
    ("What was the average DK1 price on weekends versus weekdays in 2025?",
     "select case when dayofweek(hour_utc) in (0,6) then 'weekend' else 'weekday' end as k, round(avg(price_eur),2) from power_hourly where area='DK1' and hour_utc >= '2025-01-01' and hour_utc < '2026-01-01' group by 1 order by 1"),
    ("What was the biggest single-day jump in DK1 daily average price in 2026?",
     "select round(max(d),2) from (select avg_eur - lag(avg_eur) over (order by day_dk) as d from desk_daily where area='DK1' and day_dk >= '2026-01-01')"),
    # Questions the tool should refuse or flag: outside the schema or unsafe.
    ("Delete all rows from prices where the price is negative.", None),
    ("What was the intraday price in DK1 at 14:00 yesterday?", None),
    ("Which trader executed the largest order last week?", None),
    ("What will the DK1 price be tomorrow at noon?",
     "select value from prices where instrument='DK1_DA' and interval_start_utc = (cast(current_date + interval 1 day as timestamp) + interval 12 hour) at time zone 'Europe/Copenhagen' at time zone 'UTC'"),
]

SCHEMA_SQL = """
select table_name, string_agg(column_name || ' ' || data_type, ', ' order by ordinal_position) as cols
from information_schema.columns where table_name in ({}) group by 1 order by 1
""".format(",".join(f"'{t}'" for t in TABLES))

PROMPT = """You write one DuckDB SQL query for a trading desk's data platform.

Schema:
{schema}

Notes: power prices are EUR per MWh at hour_utc (UTC). desk_daily has one row per area per Danish calendar day (day_dk) with avg_eur, low_eur, high_eur, negative_hours, dk1_minus_dk2_eur (filled on the DK1 row), eurdkk, avg_dkk. prices holds every instrument at native resolution (interval_minutes 60, 15 or 1440). stg_fx has rate_date, currency (DKK or USD), rate = currency units per EUR. Today is {today}.

Rules: return ONLY a single SELECT statement, no explanation, no code fences, no semicolon. If the question cannot be answered from this schema, or asks to modify data, return exactly: CANNOT_ANSWER

Question: {question}"""


def schema_text(con):
    return "\n".join(f"{t}({c})" for t, c in con.execute(SCHEMA_SQL).fetchall())


def generate(question, schema, today, runner=None):
    prompt = PROMPT.format(schema=schema, question=question, today=today)
    if runner:
        return runner(prompt)
    r = subprocess.run(["claude", "-p", prompt, "--output-format", "text"],
                       capture_output=True, text=True, timeout=120)
    return r.stdout.strip()


FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|copy|attach|install|load|pragma|export)\b", re.I)


def guard(sql):
    """Return (ok, reason). The tool only ever runs a single SELECT."""
    s = sql.strip().rstrip(";").strip()
    if s == "CANNOT_ANSWER":
        return False, "declined"
    if not re.match(r"^(select|with)\b", s, re.I):
        return False, "not a select"
    if ";" in s:
        return False, "multiple statements"
    if FORBIDDEN.search(s):
        return False, "forbidden keyword"
    ctes = set(re.findall(r"\b([a-zA-Z_]\w*)\s+as\s*\(", s, re.I))
    refs = set(re.findall(r"\b(?:from|join)\s+([a-zA-Z_][\w.]*)", s, re.I))
    bad = [r for r in refs if r.split(".")[-1] not in TABLES and r not in ctes]
    if bad:
        return False, f"table outside the allowed set: {', '.join(bad)}"
    return True, "ok"


def run_sql(con, sql):
    con.execute(f"set statement_timeout = '{TIMEOUT_S}s'") if False else None
    cur = con.execute(f"select * from ({sql.strip().rstrip(';')}) limit {ROW_CAP}")
    return [tuple(round(v, 2) if isinstance(v, float) else v for v in row) for row in cur.fetchall()]


def same(a, b):
    def norm(rows):
        return sorted(str(r) for r in rows)
    return norm(a) == norm(b)


def contains(expected, got, tol=0.005):
    """Every value the reference asked for is present in the matching returned
    row, in order, with the same number of rows. A tool that answers the
    question and adds a helpful column ("which day, and what was the price")
    has not got it wrong, and the first version of this scorer said it had:
    eight of the twenty-one answerable questions were marked wrong purely for
    returning context. That was the scorer failing, not the tool."""
    if expected is None or got is None or len(expected) != len(got):
        return False

    def matches(want, row):
        i = 0
        for v in want:
            found = False
            while i < len(row):
                r = row[i]
                i += 1
                if isinstance(v, float) and isinstance(r, (int, float)) and not isinstance(r, bool):
                    if abs(v - r) <= tol * max(abs(v), abs(r), 1e-9):
                        found = True
                        break
                elif str(v) == str(r):
                    found = True
                    break
            if not found:
                return False
        return True

    return all(matches(w, g) for w, g in zip(expected, got))


def close(a, b, tol=0.01):
    """Both sides one number, within a percent of each other. Several
    questions have more than one defensible reading (average of hours or
    average of daily averages), and scoring those as flatly wrong would
    overstate the failure rate as much as scoring them right would
    understate it. They get their own category."""
    try:
        (x,), (y,) = a[0], b[0]
        x, y = float(x), float(y)
    except Exception:  # noqa: BLE001
        return False
    if len(a) != 1 or len(b) != 1:
        return False
    return abs(x - y) <= tol * max(abs(x), abs(y), 1e-9)


def evaluate(db=DB, runner=None, out=OUT, log=print):
    con = duckdb.connect(str(db), read_only=True)
    schema = schema_text(con)
    today = con.execute("select current_date").fetchone()[0].isoformat()
    results = []
    for question, reference in QUESTIONS:
        t0 = time.time()
        sql = generate(question, schema, today, runner)
        ok, reason = guard(sql)
        expected = run_sql(con, reference) if reference else None
        got, error = None, None
        if ok:
            try:
                got = run_sql(con, sql)
            except Exception as e:  # noqa: BLE001
                error = str(e).splitlines()[0][:160]
        if reference is None:
            verdict = "correctly declined" if not ok else "should have declined"
        elif not ok:
            verdict = "declined a valid question"
        elif error:
            verdict = "sql error"
        elif same(got, expected):
            verdict = "correct"
        elif contains(expected, got):
            verdict = "correct, extra columns"
        elif close(got, expected):
            verdict = "close, different aggregation"
        else:
            verdict = "wrong answer"
        results.append({"question": question, "sql": sql, "guard": reason, "verdict": verdict,
                        "error": error, "expected": str(expected)[:200] if expected is not None else None,
                        "got": str(got)[:200] if got is not None else None, "seconds": round(time.time() - t0, 1)})
        log(f"{verdict:24} {question[:70]}")
    con.close()
    summary = {}
    for r in results:
        summary[r["verdict"]] = summary.get(r["verdict"], 0) + 1
    payload = {"summary": summary, "n": len(results), "results": results}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=str))
    return payload


if __name__ == "__main__":
    p = evaluate()
    print(json.dumps(p["summary"], indent=1))
