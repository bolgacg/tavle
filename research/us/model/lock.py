"""Holdout lock for the New York study (CONTRACT.md, "The holdout lock").

    assert_build_only(delivery_dates)   raises HoldoutLocked if any delivery date is on or after
                                        2024-01-01, unless the study is unlocked (all three below)
    read_end()                          first delivery date a loader may read (2024-01-01 while locked)

Unlocked only when ALL hold:
  1. env US_HOLDOUT_RUN=1
  2. <study>/FREEZE exists and names a commit hash (research/us/FREEZE on the laptop)
  3. that hash is a commit of the enclosing git repository, `git diff --quiet <hash> -- <study>/model
     <study>/pipeline` passes, and no untracked .py file sits in those folders (stricter than the
     contract: an untracked file would otherwise escape the diff).
A copy outside a git repository (gene ~/nyiso-us) can therefore never unlock.

Delivery dates are local New York dates: a tz-aware timestamp is converted to America/New_York first,
a naive one is read as a local date. Every loader calls assert_build_only before it reads and again on
what it read.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "America/New_York"
HOLDOUT_START = dt.date(2024, 1, 1)
MODEL_DIR = Path(__file__).resolve().parent
STUDY_DIR = MODEL_DIR.parent
ENV_FLAG = "US_HOLDOUT_RUN"
_HASH = re.compile(r"\b[0-9a-f]{7,40}\b")


class HoldoutLocked(RuntimeError):
    pass


def _local_dates(delivery_dates) -> np.ndarray:
    """Any date-like input (scalar, list, Series, Index, array) as numpy datetime64[D] local dates."""
    if isinstance(delivery_dates, (str, dt.date, dt.datetime, pd.Timestamp, np.datetime64)):
        delivery_dates = [delivery_dates]
    if isinstance(delivery_dates, pd.DataFrame):
        raise TypeError("pass a column, not a DataFrame")
    s = pd.Series(delivery_dates) if not isinstance(delivery_dates, pd.Series) else delivery_dates
    if len(s) == 0:
        return np.array([], dtype="datetime64[D]")
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert(TZ).dt.tz_localize(None)
    elif s.dtype == object:
        vals = []
        for v in s:
            if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
                vals.append(pd.NaT)
                continue
            ts = pd.Timestamp(v)
            vals.append(ts.tz_convert(TZ).tz_localize(None) if ts.tzinfo is not None else ts)
        s = pd.Series(pd.to_datetime(vals))
    else:
        s = pd.to_datetime(s)
    s = s.dropna()
    return s.to_numpy().astype("datetime64[D]")


def freeze_hash(study_dir: Path = STUDY_DIR) -> str | None:
    f = Path(study_dir) / "FREEZE"
    if not f.exists():
        return None
    m = _HASH.search(f.read_text())
    return m.group(0) if m else None


def _git(study_dir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(study_dir), *args], capture_output=True, text=True)


def unlock_status(study_dir: Path = STUDY_DIR, env: dict | None = None) -> tuple[bool, str]:
    """(unlocked, reason). Checks the environment first, so a normal run never calls git."""
    env = os.environ if env is None else env
    study_dir = Path(study_dir)
    if env.get(ENV_FLAG) != "1":
        return False, f"{ENV_FLAG} is not 1"
    h = freeze_hash(study_dir)
    if h is None:
        return False, f"{study_dir / 'FREEZE'} is missing or names no commit hash"
    top = _git(study_dir, "rev-parse", "--show-toplevel")
    if top.returncode != 0:
        return False, f"{study_dir} is not inside a git repository"
    if _git(study_dir, "cat-file", "-e", f"{h}^{{commit}}").returncode != 0:
        return False, f"FREEZE names {h}, which is not a commit here"
    paths = [str(study_dir / "model"), str(study_dir / "pipeline")]
    if _git(study_dir, "diff", "--quiet", h, "--", *paths).returncode != 0:
        return False, f"model or pipeline differs from the frozen commit {h}"
    untracked = _git(study_dir, "ls-files", "--others", "--exclude-standard", "--", *paths).stdout.split()
    untracked_py = [u for u in untracked if u.endswith(".py")]
    if untracked_py:
        return False, f"untracked code in model or pipeline: {untracked_py[:5]}"
    return True, f"unlocked: frozen at {h}, code unchanged, {ENV_FLAG}=1"


def assert_build_only(delivery_dates, study_dir: Path = STUDY_DIR, env: dict | None = None) -> None:
    """Raise HoldoutLocked if any delivery date is on or after 2024-01-01 and the study is locked."""
    d = _local_dates(delivery_dates)
    if len(d) == 0:
        return
    late = d >= np.datetime64(HOLDOUT_START, "D")
    if not late.any():
        return
    ok, why = unlock_status(study_dir, env)
    if not ok:
        raise HoldoutLocked(f"{int(late.sum())} delivery date(s) on or after {HOLDOUT_START} "
                            f"(latest requested {d.max()}); holdout is locked: {why}")


def read_end(requested_end: dt.date | None = None, study_dir: Path = STUDY_DIR, env: dict | None = None) -> dt.date:
    """The exclusive end date for a loader. Locked: never later than 2024-01-01. A later
    requested_end must pass assert_build_only first."""
    if requested_end is None:
        requested_end = HOLDOUT_START
    assert_build_only([requested_end - dt.timedelta(days=1)], study_dir, env)
    return requested_end
