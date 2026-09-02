from tavle import revisions as R


def test_diff_counts_appeared_revised_vanished_per_version():
    prev = {("DK1", "2026-01-01T00:00:00", "settled"): 100.0,
            ("DK1", "2026-01-01T00:00:00", "real-time"): 101.0,
            ("DK1", "2026-01-01T01:00:00", "settled"): 50.0}
    cur = {("DK1", "2026-01-01T00:00:00", "settled"): 103.0,     # revised by 3
           ("DK1", "2026-01-01T00:00:00", "real-time"): 101.2,   # rounding, unchanged
           ("DK1", "2026-01-01T02:00:00", "settled"): 70.0}      # appeared; 01:00 vanished
    d = R.diff(prev, cur)
    assert d["revised"]["settled"] == {"n": 1, "max_abs_mwh": 3.0, "mean_abs_mwh": 3.0}
    assert d["revised"]["real-time"]["n"] == 0
    assert d["appeared"]["settled"] == 1 and d["vanished"]["settled"] == 1
    assert d["unchanged"] == 1
    assert d["largest"][0]["change_mwh"] == 3.0 and d["largest"][0]["version"] == "settled"


def test_largest_is_sorted_and_capped():
    prev = {("DK1", f"2026-01-01T{h:02d}:00:00", "settled"): 0.0 for h in range(10)}
    cur = {k: float(i + 1) for i, k in enumerate(prev)}
    d = R.diff(prev, cur, top=3)
    assert [x["change_mwh"] for x in d["largest"]] == [10.0, 9.0, 8.0]


def test_run_writes_a_baseline_then_a_diff(tmp_path):
    import duckdb
    db = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db))
    con.execute("""create table wind_versions_long as
        select 'DK1' as area, timestamp '2026-01-01 00:00' as hour_utc, 'settled' as version, 100.0 as value_mwh
        union all select 'DK1', timestamp '2026-01-01 00:00', 'real-time', 99.0""")
    con.close()
    state, log = tmp_path / "state.parquet", tmp_path / "log.json"
    msg = R.run(db=db, state=state, log=log)
    assert msg.startswith("baseline")
    con = duckdb.connect(str(db))
    con.execute("update wind_versions_long set value_mwh = 110 where version = 'settled'")
    con.close()
    import datetime as dt
    msg = R.run(db=db, state=state, log=log, now=dt.datetime(2026, 1, 2, 5, 40, tzinfo=dt.timezone.utc))
    assert "1 revised" in msg
    import json
    entries = json.loads(log.read_text())
    assert len(entries) == 2 and entries[1]["revised"]["settled"]["n"] == 1


def test_same_night_rerun_replaces_the_diff_but_never_the_baseline(tmp_path):
    import datetime as dt
    import json
    import duckdb
    db = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db))
    con.execute("create table wind_versions_long as select 'DK1' as area, timestamp '2026-01-01 00:00' as hour_utc, 'settled' as version, 100.0 as value_mwh")
    con.close()
    state, log = tmp_path / "s.parquet", tmp_path / "l.json"
    night = dt.datetime(2026, 1, 2, 5, 40, tzinfo=dt.timezone.utc)
    R.run(db=db, state=state, log=log, now=night)                 # baseline
    R.run(db=db, state=state, log=log, now=night)                 # same night: a diff, appended
    R.run(db=db, state=state, log=log, now=night)                 # same night again: replaces the diff
    entries = json.loads(log.read_text())
    assert [e["baseline"] for e in entries] == [True, False]
