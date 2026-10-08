#!/bin/bash
# v2 data for the held-out run (or its dry run): every v2 table, the GEFS joined table, the feature parts
# and both matrices, rebuilt from raw into $US_RUN_DIR/parquet_v2 by the build code in this folder, then
# the timing tests over the whole range and every bid day of the window. Any failure, failed test or
# skipped test stops the run before $US_RUN_DIR/parquet_v2/DATA_OK is written.
#   dry run (window 2023, must equal the build tree file for file):
#     US_RUN_MODE=dryrun US_RUN_DIR=~/nyiso-us_dryrun/data nohup bash run_heldout_data.sh > dry.log 2>&1 &
#   held-out (delivery to 2026-09-30; from the frozen clone, opens only through the lock):
#     US_RUN_MODE=heldout US_HOLDOUT_RUN=1 US_RUN_DIR=~/nyiso-us_heldout/data nohup bash run_heldout_data.sh > ho.log 2>&1 &
# In held-out mode the rows before 2024-01-01 must also equal the build tree (compare_v2.py --before).
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
PY=${PY:-$HOME/nyiso-us/.venv/bin/python}
BUILD_TREE=${BUILD_TREE:-$HOME/nyiso-us/parquet_v2}
t0=$(date +%s); st() { echo "== $(date +%T) $1 ($(( $(date +%s) - t0 )) s since start)"; }
fail() { st "FAILED: $1"; exit 1; }
case "${US_RUN_MODE:-}" in dryrun|heldout) ;; *) fail "US_RUN_MODE must be dryrun or heldout";; esac
OUT=$($PY -c 'import common as C; print(C.RUN_MODE, C.LAST_DAY, C.PARQUET)') || fail "mode refused"
read -r MODE LAST PQ <<< "$OUT"
st "mode $MODE, last delivery day $LAST, output $PQ"
[ "$MODE" = "$US_RUN_MODE" ] || fail "mode mismatch"
if [ -e "$PQ" ] && [ "${RESUME:-0}" != 1 ]; then fail "$PQ exists (RESUME=1 to reuse its parts)"; fi
mkdir -p "$PQ"
rm -f "$PQ/DATA_OK"

for s in prices load outages weather_gfs reforecast gefs_joined; do
  st "build $s"; $PY build_tables_v2.py $s 2>&1 | tail -3; [ "${PIPESTATUS[0]}" = 0 ] || fail "build $s"
done
for s in panel border wxr wxrj day assemble; do
  st "features $s"; $PY features_v2.py $s || fail "features $s"
done

st "timing tests (whole range) and window tests (every bid day)"
$PY -m pytest -q -p no:cacheprovider -rs --junitxml="$PQ/timing_tests.xml" test_timing_v2.py test_window_v2.py 2>&1 | tail -15
[ "${PIPESTATUS[0]}" = 0 ] || fail "timing tests"
$PY -c 'import sys, xml.etree.ElementTree as E
r = E.parse(sys.argv[1]).getroot(); s = r if r.tag == "testsuite" else r.find("testsuite")
n = {k: int(s.get(k)) for k in ("tests", "failures", "errors", "skipped")}; print("junit", n)
sys.exit(1 if n["failures"] or n["errors"] or n["skipped"] or n["tests"] == 0 else 0)' "$PQ/timing_tests.xml" \
  || fail "a timing test failed or was skipped"

if [ "$MODE" = dryrun ]; then
  st "compare with the build tree (every file, every row)"
  $PY -I compare_v2.py "$BUILD_TREE" "$PQ" --out "$PQ/compare_build.json" | tail -3
  [ "${PIPESTATUS[0]}" = 0 ] || fail "dry run differs from the build tree"
else
  st "compare rows before 2024-01-01 with the build tree"
  $PY -I compare_v2.py "$BUILD_TREE" "$PQ" --before 2024-01-01 --out "$PQ/compare_build.json" | tail -3
  [ "${PIPESTATUS[0]}" = 0 ] || fail "build-year rows differ from the build tree"
fi
date > "$PQ/DATA_OK"
st "DATA DONE: $PQ/features/panel_2010_${LAST:0:4}.parquet and day_2010_${LAST:0:4}.parquet"
