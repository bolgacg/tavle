"""v2 shim over the v1 `common` module: same constants, rules and helpers, a different year range and
output folder. Put this folder BEFORE the v1 pipeline folder on sys.path; then every v1 module that does
`from common import ...` (build_tables, timing) gets the v2 values, and v1 itself is untouched.

v1 source is executed into this module's namespace, so its functions (month_keys, entries,
expected_hours) read the v2 FIRST_DAY / LAST_DAY / PARQUET below.
HOLDOUT RULE: v2 tables stop at 2023-12-31; nothing with delivery or target time on or after
2024-01-01 is written or computed on.

Run mode (research/us/model/heldout_mode.py, env US_RUN_MODE): unset = build, exactly the values below.
dryrun and heldout move PARQUET and RESULTS under US_RUN_DIR and set LAST_DAY to the mode's last day
(dryrun 2023-12-31, heldout 2026-09-30, which opens only through the lock on the frozen clone).
"""
from __future__ import annotations

import datetime as _dt
import os as _os
import sys as _sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent
_V1 = next(p for p in (_HERE.parent.parent / "us" / "pipeline", _Path.home() / "nyiso-us" / "pipeline")
           if (p / "common.py").exists())
exec(compile((_V1 / "common.py").read_text(), str(_V1 / "common.py"), "exec"), globals())
if str(_V1) not in _sys.path:
    _sys.path.append(str(_V1))                      # timing, points, build_tables resolve from v1

V1_PIPELINE = _V1
V1_PARQUET = HOME / "parquet"
PARQUET = HOME / "parquet_v2"
RESULTS = HOME / "results" / "v2"
FIRST_DAY = _dt.date(2010, 1, 1)
LAST_DAY = _dt.date(2023, 12, 31)                   # last delivery day in v2 (holdout starts 2024-01-01)
REFORECAST_RAW = RAW / "gefs_reforecast"
# GEFS v12 reforecast: 00 UTC run of day D. Operational GEFS 00 UTC output through day 3 is on NOMADS
# about 4.5 to 5.5 h after initialisation; init + 8 h is the conservative bound used here
# (= 03:00 EST / 04:00 EDT on D, before the 05:00 decision).
REFORECAST_PUBLISH_LAG = pd.Timedelta(hours=8)
# Archived live GEFS v12 (AWS noaa-gefs-pds, fetch_gefs_live.py), idea V16: the same model, member (c00),
# run (00 UTC), leads (27-54 h) and points as the reforecast, from the first v12 cycle in the archive.
# Same publication rule as the reforecast (init + 8 h). 2020-01-01 to 2020-09-22 has no run of either source.
# Files dated on or after 2024-01-01 exist on disk for the held-out run and are never opened by v2 code.
GEFS_LIVE_RAW = RAW / "gefs_live"
GEFS_LIVE_FIRST = _dt.date(2020, 9, 23)
GEFS_LIVE_PUBLISH_LAG = REFORECAST_PUBLISH_LAG

RUN_MODE = "build"
RUN_DIR = None
if _os.environ.get("US_RUN_MODE"):     # build mode never imports the mode module
    _MODEL = next(p for p in (_HERE.parent.parent / "us" / "model", HOME / "model")
                  if (p / "heldout_mode.py").exists())
    if str(_MODEL) not in _sys.path:
        _sys.path.append(str(_MODEL))
    import heldout_mode as _HM
    RUN_MODE = _HM.mode()                           # raises if held-out mode is refused by the lock
    if RUN_MODE != "build":
        RUN_DIR = _HM.run_dir()
        PARQUET = RUN_DIR / "parquet_v2"
        RESULTS = RUN_DIR / "results" / "v2"
        LAST_DAY = _HM.last_day()
