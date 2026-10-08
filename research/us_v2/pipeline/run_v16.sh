#!/bin/bash
# Idea V16 end to end on gene (OBJECTIVES.md, V16 addendum). Waits for the laptop's rsync of
# raw/gefs_live (flag raw/gefs_live/.sync_complete), builds weather_gefs_joined (reforecast runs to 2019 +
# live GEFS v12 from 2020-09-23; held-out files 2024+ are never opened), the joined wxr block, runs the full
# timing tests and the injected-lookahead tests, then V4's model and rule on the joined source
# (results/v2/pos/V16_B_gefs_joined.parquet) and touches results/v16.done. Any failure stops before the
# flag and writes results/v16.failed, so the held final scoring does not start without V16.
# cd ~/nyiso-us && setsid nohup bash pipeline_v2/run_v16.sh > logs/v16.log 2>&1 < /dev/null &
set -uo pipefail
cd ~/nyiso-us
PY=~/nyiso-us/.venv/bin/python
fail() { echo "== $(date +%T) FAILED: $1"; echo "$1 $(date)" > results/v16.failed; exit 1; }
until [ -f raw/gefs_live/.sync_complete ]; do sleep 120; done
rm -f results/v16.failed
cd pipeline_v2
echo "== $(date +%T) build weather_gefs_joined"; $PY build_tables_v2.py gefs_joined 2>&1 | tail -3
[ "${PIPESTATUS[0]}" = 0 ] || fail "build gefs_joined"
echo "== $(date +%T) wxr joined block";      $PY features_v2.py wxrj || fail "features wxrj"
echo "== $(date +%T) timing tests (full)"
$PY -m pytest -q -p no:cacheprovider test_timing_v2.py 2>&1 | tail -6
[ "${PIPESTATUS[0]}" = 0 ] || fail "timing tests"
cd ../v2/ideas
echo "== $(date +%T) injected-lookahead tests"
$PY -m pytest -q -p no:cacheprovider test_lookahead.py 2>&1 | tail -4
[ "${PIPESTATUS[0]}" = 0 ] || fail "lookahead tests"
echo "== $(date +%T) V16 rolling 2013-2023"
OMP_NUM_THREADS=2 US_LGBM_THREADS=2 $PY -I run_ideas.py V16_B_gefs_joined || fail "run_ideas V16"
[ -f ../../results/v2/pos/V16_B_gefs_joined.parquet ] || fail "positions missing"
touch ../../results/v16.done
echo "== $(date +%T) V16 DONE"
