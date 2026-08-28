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
