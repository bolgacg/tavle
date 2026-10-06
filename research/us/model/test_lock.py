"""The holdout lock raises (CONTRACT.md). Runs anywhere: python -m pytest -q test_lock.py"""
from __future__ import annotations

import datetime as dt
import subprocess

import pandas as pd
import pytest

import lock

LOCKED_ENV = {}                                   # no US_HOLDOUT_RUN


def test_build_dates_pass():
    lock.assert_build_only([dt.date(2020, 1, 1), dt.date(2023, 12, 31)], env=LOCKED_ENV)
    lock.assert_build_only(pd.Series(pd.date_range("2020-01-01", "2023-12-31 23:00", freq="h", tz=lock.TZ)),
                           env=LOCKED_ENV)
    lock.assert_build_only([], env=LOCKED_ENV)


@pytest.mark.parametrize("value", [
    dt.date(2024, 1, 1),
    "2024-01-01",
    pd.Timestamp("2024-01-01 00:00", tz=lock.TZ),
    pd.Timestamp("2024-01-01 05:00", tz="UTC"),          # 00:00 in New York
    pd.Series(pd.to_datetime(["2023-06-01", "2026-09-30"])),
    pd.Index(pd.date_range("2023-12-31", periods=2, freq="D", tz=lock.TZ)),
])
def test_holdout_dates_raise(value):
    with pytest.raises(lock.HoldoutLocked):
        lock.assert_build_only(value, env=LOCKED_ENV)


def test_local_date_not_utc_date():
    # 23:00 on 31 Dec 2023 in New York is 04:00 UTC on 1 Jan 2024: a build hour, must pass.
    lock.assert_build_only(pd.Timestamp("2024-01-01 04:00", tz="UTC"), env=LOCKED_ENV)


def test_env_alone_does_not_unlock(tmp_path):
    with pytest.raises(lock.HoldoutLocked):
        lock.assert_build_only(dt.date(2024, 1, 1), study_dir=tmp_path, env={lock.ENV_FLAG: "1"})


def test_real_checkout_is_locked_without_env():
    with pytest.raises(lock.HoldoutLocked):
        lock.assert_build_only(dt.date(2025, 6, 1), env=LOCKED_ENV)
    with pytest.raises(lock.HoldoutLocked):
        lock.read_end(dt.date(2024, 2, 1), env=LOCKED_ENV)
    assert lock.read_end(env=LOCKED_ENV) == lock.HOLDOUT_START


def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=lock-test", "-c", "user.email=t@t",
                           *a], check=True, capture_output=True, text=True).stdout.strip()


def test_full_mechanism_in_a_temporary_repository(tmp_path):
    """FREEZE + matching code + env unlocks; any one missing, or changed code, keeps it locked."""
    study = tmp_path / "research" / "us"
    (study / "model").mkdir(parents=True)
    (study / "pipeline").mkdir()
    (study / "model" / "m.py").write_text("x = 1\n")
    (study / "pipeline" / "p.py").write_text("y = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "freeze")
    h = _git(tmp_path, "rev-parse", "HEAD")
    env = {lock.ENV_FLAG: "1"}
    late = dt.date(2024, 3, 1)

    with pytest.raises(lock.HoldoutLocked, match="FREEZE"):               # no FREEZE file
        lock.assert_build_only(late, study_dir=study, env=env)
    (study / "FREEZE").write_text("commit deadbeefdeadbeef\n")
    with pytest.raises(lock.HoldoutLocked, match="not a commit"):         # bogus hash
        lock.assert_build_only(late, study_dir=study, env=env)
    (study / "FREEZE").write_text(f"frozen at {h}\n")
    with pytest.raises(lock.HoldoutLocked):                               # env missing
        lock.assert_build_only(late, study_dir=study, env={})
    lock.assert_build_only(late, study_dir=study, env=env)                # all three: unlocked
    (study / "model" / "m.py").write_text("x = 2\n")                       # code changed
    with pytest.raises(lock.HoldoutLocked, match="differs"):
        lock.assert_build_only(late, study_dir=study, env=env)
    (study / "model" / "m.py").write_text("x = 1\n")
    (study / "model" / "new.py").write_text("z = 1\n")                     # untracked code
    with pytest.raises(lock.HoldoutLocked, match="untracked"):
        lock.assert_build_only(late, study_dir=study, env=env)

