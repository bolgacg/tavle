#!/bin/bash
# v2 stage 2: waits for the reforecast download and the price stage, builds weather_reforecast, checks the
# archived-forecast rule and the timing of every weather row, joins the wxr blocks into both matrices,
# then results/v2_weather_data.done and results/v2_data.done.
# cd ~/nyiso-us && setsid nohup bash pipeline_v2/run_weather_stage.sh > logs/v2_weather_stage.log 2>&1 < /dev/null &
set -euo pipefail
cd ~/nyiso-us/pipeline_v2
PY=../.venv/bin/python
until grep -q "^DONE" ../logs/fetch_reforecast.log; do sleep 120; done
# one retry pass for any failed day (skip-if-present)
../.venv_grib/bin/python -I fetch_reforecast.py 2010 2019 > ../logs/fetch_reforecast_retry.log 2>&1 || true
until [ -f ../results/v2_price_data.done ]; do sleep 60; done
echo "== $(date +%T) build reforecast"; $PY build_tables_v2.py reforecast 2>&1 | tail -3
echo "== $(date +%T) weather tests"
$PY -m pytest -q -p no:cacheprovider test_timing_v2.py -k "reforecast or weather" 2>&1 | tail -5
echo "== $(date +%T) wxr";      $PY features_v2.py wxr
echo "== $(date +%T) wxr test"; $PY -m pytest -q -p no:cacheprovider test_timing_v2.py -k "wxr" 2>&1 | tail -3
echo "== $(date +%T) assemble"; $PY features_v2.py assemble
touch ../results/v2_weather_data.done ../results/v2_data.done
echo "== $(date +%T) WEATHER STAGE DONE"
