#!/bin/bash
# v2 stage 1 (no reforecast): tables, timing test, feature matrices, then results/v2_price_data.done.
# cd ~/nyiso-us && nohup bash pipeline_v2/run_price_stage.sh > logs/v2_price_stage.log 2>&1 < /dev/null &
set -euo pipefail
cd ~/nyiso-us/pipeline_v2
PY=../.venv/bin/python
t0=$(date +%s); st() { echo "== $(date +%T) $1 ($(( $(date +%s) - t0 )) s since start)"; }
for s in prices load outages weather_gfs; do st "build $s"; $PY build_tables_v2.py $s 2>&1 | tail -3; done
st "timing test"
$PY -m pytest -q -p no:cacheprovider test_timing_v2.py -k "not reforecast and not wxr" 2>&1 | tail -15
st "features panel";  $PY features_v2.py panel
st "features border"; $PY features_v2.py border
st "features day";    $PY features_v2.py day
st "assemble";        $PY features_v2.py assemble
touch ../results/v2_price_data.done
st "PRICE STAGE DONE"
