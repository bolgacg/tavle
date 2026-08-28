"""A DAG runner that fits in one screen.

Airflow-shaped, not Airflow: tasks declare dependencies, run in topological
order, retry with backoff, and every attempt lands in a run ledger inside
the same DuckDB file the marts live in, so "what ran, when, and did it
work" is a SQL question. The task table below would translate to an
Airflow DAG file in an afternoon; the point here is that the platform is
inspectable without a scheduler UI."""
import datetime as dt
import pathlib
import subprocess
import sys
import time
import traceback
import uuid

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "tavle.duckdb"
DBT = ROOT / "dbt"


def _sh(*cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r.stdout


def extract_eds(dataset):
    def run():
        return _sh(sys.executable, "-m", "tavle.extract", "eds", dataset)
    return run


def extract_ecb():
    return _sh(sys.executable, "-m", "tavle.extract", "ecb")


def dbt_build():
    return _sh(sys.executable, "-m", "dbt.cli.main", "build", "--profiles-dir", ".", cwd=DBT)


def build_page():
    return _sh(sys.executable, "-m", "tavle.page")


TASKS = {
    "extract_elspot":     {"deps": [], "fn": extract_eds("Elspotprices")},
    "extract_dayahead":   {"deps": [], "fn": extract_eds("DayAheadPrices")},
    "extract_production": {"deps": [], "fn": extract_eds("ProductionConsumptionSettlement")},
    "extract_fx":         {"deps": [], "fn": extract_ecb},
    "dbt_build":          {"deps": ["extract_elspot", "extract_dayahead", "extract_production", "extract_fx"], "fn": dbt_build},
    "page":               {"deps": ["dbt_build"], "fn": build_page},
}


def topological(tasks):
    seen, order = set(), []

    def visit(name, path=()):
        if name in seen:
            return
        if name in path:
            raise ValueError(f"cycle through {name}")
        for d in tasks[name]["deps"]:
            visit(d, path + (name,))
        seen.add(name)
        order.append(name)

    for n in tasks:
        visit(n)
    return order


def ledger(db=DB):
    """Create the ledger and return nothing: the connection is deliberately
    not held open. DuckDB allows a single writer per file, and dbt runs in
    its own process against the same file, so a runner that kept its
    connection open would lock out the very task it is running. That is
    exactly what the first version did, and the run ledger recorded the
    failure, which is the argument for having one."""
    pathlib.Path(db).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db))
    try:
        con.execute("create schema if not exists ops")
        con.execute("""create table if not exists ops.runs (
            run_id varchar, task varchar, attempt integer, started timestamp,
            finished timestamp, status varchar, message varchar)""")
    finally:
        con.close()


def record(db, row, attempts=30, sleep=time.sleep):
    """Append one row, opening and closing the connection each time. Retries
    while another process holds the write lock rather than losing the record."""
    for _ in range(attempts):
        try:
            con = duckdb.connect(str(db))
            try:
                con.execute("insert into ops.runs values (?,?,?,?,?,?,?)", row)
            finally:
                con.close()
            return True
        except duckdb.IOException:
            sleep(1.0)
    return False


def run(tasks=TASKS, only=None, retries=2, backoff=10.0, db=DB, sleep=time.sleep, log=print):
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    order = topological(tasks)
    if only:
        order = [t for t in order if t in only]
    failed = set()
    ledger(db)
    for name in order:
        if any(d in failed for d in tasks[name]["deps"]):
            record(db, [run_id, name, 0, None, None, "skipped", "upstream failed"], sleep=sleep)
            log(f"{name}: skipped (upstream failed)")
            failed.add(name)
            continue
        for attempt in range(1, retries + 2):
            started = dt.datetime.now(dt.timezone.utc)
            try:
                out = tasks[name]["fn"]() or ""
                record(db, [run_id, name, attempt, started, dt.datetime.now(dt.timezone.utc), "ok", out[-500:]], sleep=sleep)
                log(f"{name}: ok (attempt {attempt})")
                break
            except Exception as e:  # noqa: BLE001, the ledger is the point
                msg = "".join(traceback.format_exception_only(type(e), e))[-500:]
                record(db, [run_id, name, attempt, started, dt.datetime.now(dt.timezone.utc), "failed", msg], sleep=sleep)
                log(f"{name}: failed attempt {attempt}: {msg.strip()}")
                if attempt <= retries:
                    sleep(backoff * attempt)
                else:
                    failed.add(name)
    return run_id, failed


if __name__ == "__main__":
    only = sys.argv[1:] or None
    rid, bad = run(only=only)
    print(f"run {rid}: {'FAILED ' + ','.join(sorted(bad)) if bad else 'all ok'}")
    sys.exit(1 if bad else 0)
