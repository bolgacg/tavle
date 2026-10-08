#!/bin/bash
# 7 Oct: load_forecast published_at rule (06:00 on D-1 bound), then rebuild the 2010 feature parts and assemble.
set -euo pipefail
cd ~/nyiso-us/pipeline_v2
PY=../.venv/bin/python
$PY build_tables_v2.py load 2>&1 | tail -2
rm -f ../parquet_v2/features/parts/panel_base_2010.parquet ../parquet_v2/features/parts/day_base_2010.parquet
$PY features_v2.py panel
$PY features_v2.py day
$PY features_v2.py assemble
$PY -m pytest -q -p no:cacheprovider test_timing_v2.py -k "not reforecast and not wxr" 2>&1 | tail -4
echo FIX DONE
