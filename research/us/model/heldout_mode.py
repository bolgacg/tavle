"""Run mode for the single held-out run of the New York study (versions 1 and 2), written 8 Oct 2026.

Three modes, chosen by the environment variable US_RUN_MODE:

    (unset)    build    the build-year code paths, unchanged: nothing on or after 2024-01-01 is read.
    dryrun     dryrun   the held-out code paths with a test window that ends on 2023-12-31: they must
                        reproduce the build-year positions for 2023 exactly. Never reads 2024 or later.
    heldout    heldout  the held-out window 2024-01-01 to 2026-09-30. Opens only when the v1 lock is open
                        (research/us/FREEZE names a commit of this repository, US_HOLDOUT_RUN=1, research/us
                        model and pipeline unchanged since it) AND every other code folder of both versions is
                        unchanged since that commit with no untracked .py file in it.

The held-out and dry-run modes write only under US_RUN_DIR (required, must not be the build results tree).
They extend the walk-forward (v1: month by month, each year's choices made on the year before) and the
rolling windows (v2: quarter by quarter, settings on the trailing four out-of-sample quarters) from the
build years' own outputs, which are copied into US_RUN_DIR first, never modified in place.
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
from pathlib import Path

import lock

BUILD_END = dt.date(2023, 12, 31)
HELDOUT = (dt.date(2024, 1, 1), dt.date(2026, 9, 30))
DRYRUN = (dt.date(2023, 1, 1), dt.date(2023, 12, 31))
REPO_DIR = lock.STUDY_DIR.parents[1]
CODE_DIRS = ["research/us/model", "research/us/pipeline", "research/us/side", "research/us/robust",
             "research/us/chartdata", "research/us_v2/model", "research/us_v2/ideas", "research/us_v2/pipeline"]


class ModeError(RuntimeError):
    pass


def mode(env: dict | None = None) -> str:
    env = os.environ if env is None else env
    m = env.get("US_RUN_MODE", "") or "build"
    if m not in ("build", "dryrun", "heldout"):
        raise ModeError(f"US_RUN_MODE={m!r}: must be unset, dryrun or heldout")
    if m == "heldout":
        ok, why = unlock_status(env)
        if not ok:
            raise ModeError(f"held-out mode refused: {why}")
    return m


def unlock_status(env: dict | None = None) -> tuple[bool, str]:
    """The v1 lock, plus: every code folder of both versions unchanged since the frozen commit."""
    ok, why = lock.unlock_status(env=env)
    if not ok:
        return False, why
    h = lock.freeze_hash()
    g = lambda *a: subprocess.run(["git", "-C", str(REPO_DIR), *a], capture_output=True, text=True)
    paths = [str(REPO_DIR / p) for p in CODE_DIRS]
    if g("diff", "--quiet", h, "--", *paths).returncode != 0:
        return False, f"a code folder differs from the frozen commit {h}"
    untracked = [u for u in g("ls-files", "--others", "--exclude-standard", "--", *paths).stdout.split()
                 if u.endswith(".py")]
    if untracked:
        return False, f"untracked code: {untracked[:5]}"
    return True, f"{why}; every code folder of both versions unchanged"


def window(env: dict | None = None) -> tuple[dt.date, dt.date] | None:
    """(first, last) delivery date of the test window this mode scores; None in build mode."""
    m = mode(env)
    return {"build": None, "dryrun": DRYRUN, "heldout": HELDOUT}[m]


def last_day(env: dict | None = None) -> dt.date:
    """Last delivery date any loader may read in this mode."""
    w = window(env)
    return BUILD_END if w is None else w[1]


def read_end(env: dict | None = None) -> dt.date:
    """Exclusive end date for loaders in this mode (2024-01-01 in build and dry-run modes)."""
    return last_day(env) + dt.timedelta(days=1)


def run_dir(env: dict | None = None) -> Path | None:
    """US_RUN_DIR in dryrun and heldout modes (required); None in build mode."""
    env = os.environ if env is None else env
    if mode(env) == "build":
        return None
    d = env.get("US_RUN_DIR")
    if not d:
        raise ModeError("US_RUN_DIR must be set in dryrun and heldout modes")
    p = Path(d).expanduser().resolve()
    home = Path.home() / "nyiso-us"
    for build_tree in (home / "results", home / "cache", home / "side", home / "parquet", home / "parquet_v2"):
        if p == build_tree.resolve() or build_tree.resolve() in p.parents:
            raise ModeError(f"US_RUN_DIR {p} is inside the build tree {build_tree}")
    p.mkdir(parents=True, exist_ok=True)
    return p
