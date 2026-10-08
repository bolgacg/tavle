#!/bin/bash
# GPU-lane job of the robustness checks (queue_gpu.txt runs it from ~/nyiso-us after results/selection_fix.done).
# Two worker processes share the GPU, longest task first; each task is one process that loads the data once
# and keeps it on the GPU. Every task skips itself if its done.json exists, so a rerun resumes. Afterwards the
# summary is rewritten with everything that is complete. `bash robust/run_gpu.sh smoke` runs the synthetic
# smoke test in both workers instead and leaves results/robust/ untouched.
cd ~/nyiso-us || exit 1
L=results/robust/logs
mkdir -p "$L"
export OMP_NUM_THREADS=1
if [ "$1" = "smoke" ]; then
  tasks=("smoke" "smoke"); L=$(mktemp -d)
else
  tasks=("seeds --kind base" "seeds --kind weather"
         "neighbours --kind base --variant dropout_0.2" "neighbours --kind base --variant dropout_0.4"
         "neighbours --kind base --variant weight_decay_1e-3" "neighbours --kind base --variant weight_decay_1e-1"
         "neighbours --kind weather --variant dropout_0.2" "neighbours --kind weather --variant dropout_0.4"
         "neighbours --kind weather --variant weight_decay_1e-3" "neighbours --kind weather --variant weight_decay_1e-1")
fi
export L
printf '%s\n' "${tasks[@]}" | xargs -P "${ROBUST_GPU_WORKERS:-2}" -I{} bash -c '
  a="{}"; n=${a//[^A-Za-z0-9_.]/_}
  echo "$(date "+%F %T") start $a" >> "$L/gpu_tasks.txt"
  .venv/bin/python robust/deep_runs.py $a > "$L/deep_${n}_$$.log" 2>&1; rc=$?
  echo "$(date "+%F %T") end $a exit $rc" >> "$L/gpu_tasks.txt"; exit $rc'
rc=$?
cat "$L/gpu_tasks.txt"
if [ "$1" != "smoke" ]; then
  .venv/bin/python robust/summary.py > "$L/summary_after_gpu.log" 2>&1
  echo "summary exit $?"
fi
echo "gpu tasks exit $rc"
exit $rc
