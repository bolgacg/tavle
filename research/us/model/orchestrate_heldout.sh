#!/bin/bash
# The single held-out run of versions 1 and 2, start to finish, on the training machine without a session.
# Written 8 Oct 2026. Run from a clean clone of the frozen commit, with research/us/FREEZE naming that commit:
#
#   setsid nohup bash research/us/model/orchestrate_heldout.sh <run dir> > <run dir>/orchestrate.log 2>&1 < /dev/null &
#
# Stages, in order; each leaves <run dir>/markers/<stage>.done and is skipped on a restart:
#   inputs     raw inputs for 2024-01 to 2026-09 present, counted by FILE NAME only (no value is opened)
#   dry_data   v2 tables and features rebuilt for the dry-run window (2023); must equal the build tree, tests pass
#   dry_v2     every v2 row's 2023 positions recomputed; every row must equal the build exactly
#   dry_v1     every v1 row's 2023 positions, predictions and daily net recomputed; must equal the build exactly
#   lock       the held-out lock opens for this checkout (FREEZE, US_HOLDOUT_RUN=1, code unchanged)
#   ho_data    v2 tables and features to 2026-09-30 with the publication-time tests; build-year rows must equal
#   ho_reh     the registered verdict run (rehearsal.py heldout, score.py)
#   ho_v1      every v1 row over 2024-01-01 to 2026-09-30, scored by the lab's scoring
#   ho_v2      every v2 row, rolling quarters 2024Q1 to 2026Q3, scored by score_v2.py
#   chartdata  research/us/chartdata/build_chartdata.py --include-heldout into <run dir>/chartdata (not the page)
#   summary    DONE.md: the held-out table and the too-good audit of every row that clears the bar
# Any failed stage or failed proof writes STOPPED.md with the reason and exits; nothing after it runs.
set -uo pipefail
RUN="$(readlink -f "${1:?usage: orchestrate_heldout.sh <run dir>}")"
MODEL="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
REPO="$(git -C "$MODEL" rev-parse --show-toplevel)"
PY="${PY:-$HOME/nyiso-us/.venv/bin/python}"
V2M="$REPO/research/us_v2/model"
V2P="$REPO/research/us_v2/pipeline"
mkdir -p "$RUN/markers" "$RUN/logs"
cd "$RUN"
t0=$(date +%s)
st() { echo "== $(date '+%F %T') $*"; }
stop() {
  { echo "# Held-out run STOPPED"; echo; echo "Stage: $1"; echo; echo "Reason: $2"; echo;
    echo "Time: $(date '+%F %T')"; echo "Log: $RUN/orchestrate.log and $RUN/logs/"; } > "$RUN/STOPPED.md"
  st "STOPPED at $1: $2"; exit 1; }
done_() { date '+%F %T' > "$RUN/markers/$1.done"; st "stage $1 done ($(( $(date +%s) - t0 )) s since start)"; }
is_done() { [ -f "$RUN/markers/$1.done" ]; }
[ -f "$RUN/STOPPED.md" ] && { st "STOPPED.md exists; remove it only after reading it"; exit 1; }
H="$(grep -oE '\b[0-9a-f]{7,40}\b' "$REPO/research/us/FREEZE" | head -1)" || true
[ -n "$H" ] || stop start "research/us/FREEZE missing or names no commit"
st "repo $REPO at $(git -C "$REPO" rev-parse HEAD), frozen code commit $H, run dir $RUN"
unset US_RUN_MODE US_RUN_DIR US_HOLDOUT_RUN US_CACHE_DIR V2_PARQUET V2_PANEL V2_DAY V2_RESULTS

# ---------------------------------------------------------------- inputs (names only)
if ! is_done inputs; then
  st "inputs: counting held-out files by name"
  miss=""
  for s in damlbmp_zone rtlbmp_zone damlbmp_gen rtlbmp_gen isolf outSched; do
    for y in 2024 2025; do n=$(ls "$HOME/nyiso-us/raw/${y}"*"${s}_csv.zip" 2>/dev/null | wc -l); [ "$n" = 12 ] || miss="$miss $s/$y:$n"; done
    n=$(ls "$HOME/nyiso-us/raw/2026"0[1-9]*"${s}_csv.zip" 2>/dev/null | wc -l); [ "$n" = 9 ] || miss="$miss $s/2026:$n"
  done
  for y in 2024 2025 2026; do
    n=$(find "$HOME/nyiso-us/raw/gefs_live/$y" -type f 2>/dev/null | wc -l)
    want=$([ $y = 2024 ] && echo 366 || ([ $y = 2025 ] && echo 365 || echo 273))
    [ "$n" -ge "$want" ] || miss="$miss gefs_live/$y:$n<$want"
    ls "$HOME/nyiso-us/raw/gfs_global/"*"_${y}.json" >/dev/null 2>&1 || miss="$miss gfs_global/$y"
  done
  [ -z "$miss" ] || stop inputs "missing held-out inputs (by file name):$miss"
  done_ inputs
fi

# ---------------------------------------------------------------- dry runs (window 2023)
if ! is_done dry_data; then
  st "dry_data"
  [ -e "$RUN/dry_data/parquet_v2" ] && export RESUME=1
  ( cd "$V2P" && US_RUN_MODE=dryrun US_RUN_DIR="$RUN/dry_data" PY="$PY" bash run_heldout_data.sh ) > "$RUN/logs/dry_data.log" 2>&1 \
    || stop dry_data "v2 data dry run failed or differs from the build tree (logs/dry_data.log)"
  unset RESUME
  [ -f "$RUN/dry_data/parquet_v2/DATA_OK" ] || stop dry_data "no DATA_OK after the v2 data dry run"
  done_ dry_data
fi
if ! is_done dry_v2; then
  st "dry_v2"
  ( cd "$V2M" && US_RUN_MODE=dryrun US_RUN_DIR="$RUN/dry_v2" "$PY" heldout_v2.py all ) > "$RUN/logs/dry_v2.log" 2>&1 \
    || stop dry_v2 "v2 dry run failed (logs/dry_v2.log)"
  "$PY" "$MODEL/heldout_gate.py" v2 "$RUN/dry_v2" > "$RUN/logs/gate_v2.log" 2>&1 \
    || stop dry_v2 "a v2 row does not reproduce 2023 exactly: $(tail -5 "$RUN/logs/gate_v2.log" | tr '\n' ' ')"
  done_ dry_v2
fi
if ! is_done dry_v1; then
  st "dry_v1"
  ( cd "$MODEL" && US_RUN_MODE=dryrun US_RUN_DIR="$RUN/dry_v1" US_CACHE_DIR="$RUN/dry_v1/cache" "$PY" heldout_v1.py all ) \
    > "$RUN/logs/dry_v1.log" 2>&1 || stop dry_v1 "v1 dry run failed (logs/dry_v1.log)"
  "$PY" "$MODEL/heldout_gate.py" v1 "$RUN/dry_v1" > "$RUN/logs/gate_v1.log" 2>&1 \
    || stop dry_v1 "a v1 row does not reproduce 2023 exactly: $(tail -5 "$RUN/logs/gate_v1.log" | tr '\n' ' ')"
  done_ dry_v1
fi

# ---------------------------------------------------------------- held-out
export US_HOLDOUT_RUN=1
if ! is_done lock; then
  ( cd "$MODEL" && US_RUN_MODE=heldout "$PY" -c 'import heldout_mode as H, sys; ok, why = H.unlock_status(); print("lock:", why); sys.exit(0 if ok else 1)' ) \
    > "$RUN/logs/lock.log" 2>&1 || stop lock "the held-out lock does not open: $(cat "$RUN/logs/lock.log")"
  done_ lock
fi
if ! is_done ho_data; then
  st "ho_data"
  [ -e "$RUN/ho_data/parquet_v2" ] && export RESUME=1
  ( cd "$V2P" && US_RUN_MODE=heldout US_RUN_DIR="$RUN/ho_data" PY="$PY" bash run_heldout_data.sh ) > "$RUN/logs/ho_data.log" 2>&1 \
    || stop ho_data "v2 held-out data rebuild failed, a timing test failed, or build-year rows differ (logs/ho_data.log)"
  unset RESUME
  [ -f "$RUN/ho_data/parquet_v2/DATA_OK" ] || stop ho_data "no DATA_OK after the held-out data rebuild"
  done_ ho_data
fi
if ! is_done ho_reh; then
  st "ho_reh (registered verdict run)"
  ( cd "$MODEL" && US_CACHE_DIR="$RUN/ho_reh/cache" "$PY" rehearsal.py heldout ) > "$RUN/logs/ho_reh.log" 2>&1 \
    || stop ho_reh "rehearsal.py heldout failed (logs/ho_reh.log)"
  mkdir -p "$RUN/ho_reh"; cp "$REPO"/research/us/results/heldout_* "$RUN/ho_reh/" 2>/dev/null
  done_ ho_reh
fi
if ! is_done ho_v1; then
  st "ho_v1"
  ( cd "$MODEL" && US_RUN_MODE=heldout US_RUN_DIR="$RUN/ho_v1" US_CACHE_DIR="$RUN/ho_v1/cache" "$PY" heldout_v1.py all ) \
    > "$RUN/logs/ho_v1.log" 2>&1 || stop ho_v1 "v1 held-out run failed (logs/ho_v1.log)"
  done_ ho_v1
fi
if ! is_done ho_v2; then
  st "ho_v2"
  ( cd "$V2M" && US_RUN_MODE=heldout US_RUN_DIR="$RUN/ho_v2" V2_PARQUET="$RUN/ho_data/parquet_v2" "$PY" heldout_v2.py all ) \
    > "$RUN/logs/ho_v2.log" 2>&1 || stop ho_v2 "v2 held-out run failed (logs/ho_v2.log)"
  done_ ho_v2
fi

# ---------------------------------------------------------------- chart data and summary
if ! is_done chartdata; then
  st "chartdata"
  V1R=$(ls "$RUN"/ho_v1/side/lab_results_2021_*.json 2>/dev/null | head -1)
  V1D=$(ls "$RUN"/ho_v1/side/lab_daily_2021_*.parquet 2>/dev/null | head -1)
  V2R=$(ls "$RUN"/ho_v2/results/v2/v2_results_heldout.json 2>/dev/null | head -1)
  V2D=$(ls "$RUN"/ho_v2/results/v2/v2_daily_all_heldout.parquet 2>/dev/null | head -1)
  if [ -n "$V1R" ] && [ -n "$V1D" ] && [ -n "$V2R" ] && [ -n "$V2D" ]; then
    ( cd "$REPO/research/us/chartdata" && "$PY" -I build_chartdata.py --only data --include-heldout --end 2026-10-01 \
        --v1-results "$V1R" --v1-daily "$V1D" --v2-results "$V2R" --v2-daily "$V2D" --out "$RUN/chartdata" ) \
      > "$RUN/logs/chartdata.log" 2>&1 || echo "chartdata failed (logs/chartdata.log); results stand, rerun by hand" > "$RUN/CHARTDATA_FAILED"
  else
    echo "chartdata inputs not found: v1 $V1R $V1D v2 $V2R $V2D" > "$RUN/CHARTDATA_FAILED"
  fi
  done_ chartdata
fi
if ! is_done summary; then
  "$PY" "$MODEL/heldout_summary.py" "$RUN" > "$RUN/logs/summary.log" 2>&1 \
    || { echo "summary failed (logs/summary.log); results are in ho_v1, ho_v2, ho_reh" > "$RUN/DONE.md"; }
  done_ summary
fi
st "ALL DONE: $RUN/DONE.md"
