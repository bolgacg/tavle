import duckdb

from tavle import dag


def test_topological_order_respects_dependencies():
    tasks = {"c": {"deps": ["a", "b"], "fn": None}, "a": {"deps": [], "fn": None}, "b": {"deps": ["a"], "fn": None}}
    assert dag.topological(tasks) == ["a", "b", "c"]


def test_failed_task_is_retried_then_downstream_skipped(tmp_path):
    db = tmp_path / "t.duckdb"
    calls = {"a": 0}

    def flaky():
        calls["a"] += 1
        raise RuntimeError("boom")

    tasks = {"a": {"deps": [], "fn": flaky}, "b": {"deps": ["a"], "fn": lambda: "never"}}
    run_id, failed = dag.run(tasks, retries=1, db=db, sleep=lambda s: None, log=lambda m: None)
    assert failed == {"a", "b"}
    assert calls["a"] == 2
    rows = duckdb.connect(str(db)).execute(
        "select task, attempt, status from ops.runs where run_id = ? order by task, attempt", [run_id]).fetchall()
    assert rows == [("a", 1, "failed"), ("a", 2, "failed"), ("b", 0, "skipped")]


def test_success_is_recorded_once(tmp_path):
    db = tmp_path / "t.duckdb"
    tasks = {"a": {"deps": [], "fn": lambda: "done"}}
    run_id, failed = dag.run(tasks, db=db, sleep=lambda s: None, log=lambda m: None)
    assert not failed
    rows = duckdb.connect(str(db)).execute("select status, message from ops.runs where run_id = ?", [run_id]).fetchall()
    assert rows == [("ok", "done")]


def test_ledger_does_not_hold_the_write_lock(tmp_path):
    """The first version kept one connection open for the whole run, which
    locked dbt out of the same DuckDB file. This is that regression."""
    import duckdb as ddb
    db = tmp_path / "t.duckdb"

    def writes_from_another_connection():
        con = ddb.connect(str(db))          # a second writer, like dbt
        con.execute("create table if not exists probe (x integer)")
        con.execute("insert into probe values (1)")
        con.close()
        return "wrote"

    run_id, failed = dag.run({"a": {"deps": [], "fn": writes_from_another_connection}},
                             db=db, sleep=lambda s: None, log=lambda m: None)
    assert not failed
    con = ddb.connect(str(db))
    assert con.execute("select count(*) from probe").fetchone()[0] == 1
    assert con.execute("select status from ops.runs where run_id = ?", [run_id]).fetchone()[0] == "ok"
