#!/usr/bin/env bash
# The single held-out run (OBJECTIVES item 4), from a clean checkout of the frozen commit on gene.
# PREPARED, NOT RUN. Run it only on the orchestrator's instruction, after research/us/FREEZE exists and
# names the frozen commit, and that commit is on origin/master.
#
#   research/us/model/heldout.sh            verify, copy, clone, check the lock opens, start the run
#   research/us/model/heldout.sh --check    the same, but stop before the run starts
#   research/us/model/heldout.sh --with-v1  also, after rehearsal.py heldout, the version 1 runner over every
#                                           HELDOUT-LIST row (heldout_v1.py all, US_RUN_MODE=heldout, its own
#                                           run directory <remote>/v1run); --check --with-v1 also prints its plan
#
# Laptop side: bundle origin/master, copy it and FREEZE to gene.
# Gene side: fetch the bundle into an empty repository, check out the FREEZE commit (detached), put
# FREEZE beside it (FREEZE cannot be inside the commit it names), confirm model/ and pipeline/ are
# clean, confirm lock.unlock_status() is true for THIS checkout, then run
# `rehearsal.py heldout` under nohup with US_HOLDOUT_RUN=1 and a cache directory of its own.
# Results land in <checkout>/research/us/results/heldout_*.json; copy them back with the command
# printed at the end. A second run never overwrites the first (time-stamped names).
set -euo pipefail

CHECK_ONLY=0
WITH_V1=0
for a in "$@"; do
  case "$a" in
    --check) CHECK_ONLY=1 ;;
    --with-v1) WITH_V1=1 ;;
    *) echo "unknown option $a"; exit 1 ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
FREEZE="$REPO/research/us/FREEZE"
GENE="${GENE_HOST:-gene}"
VENV='~/nyiso-us/.venv/bin/python'

[[ -f "$FREEZE" ]] || { echo "STOP: $FREEZE does not exist (no freeze yet)"; exit 1; }
SHORT="$(grep -oE '\b[0-9a-f]{7,40}\b' "$FREEZE" | head -1 || true)"
[[ -n "$SHORT" ]] || { echo "STOP: FREEZE names no commit hash"; exit 1; }

git -C "$REPO" fetch --quiet origin master
H="$(git -C "$REPO" rev-parse --verify "${SHORT}^{commit}")"
git -C "$REPO" merge-base --is-ancestor "$H" origin/master \
  || { echo "STOP: $H is not on origin/master"; exit 1; }
for f in research/us/model/lock.py research/us/model/rehearsal.py research/us/model/frozen_choices.json \
         research/us/model/spike_config.json; do
  git -C "$REPO" cat-file -e "$H:$f" || { echo "STOP: $f is not in $H"; exit 1; }
done
echo "frozen commit $H"

WORK="$(mktemp -d)"
BUNDLE="$WORK/tavle-$H.bundle"
git -C "$REPO" bundle create "$BUNDLE" origin/master
git -C "$REPO" bundle verify "$BUNDLE" >/dev/null
REMOTE="nyiso-us/heldout/$H"
ssh "$GENE" "mkdir -p ~/$REMOTE"
scp -q "$BUNDLE" "$GENE:$REMOTE/tavle.bundle"
scp -q "$FREEZE" "$GENE:$REMOTE/FREEZE"
rm -rf "$WORK"

ssh "$GENE" bash -s -- "$H" "$REMOTE" "$CHECK_ONLY" "$VENV" "$WITH_V1" <<'GENE_SIDE'
set -euo pipefail
H="$1"; REMOTE="$HOME/$2"; CHECK_ONLY="$3"; PY="${4/#\~/$HOME}"; WITH_V1="$5"
cd "$REMOTE"
if [[ -d tavle ]]; then echo "STOP: $REMOTE/tavle exists already (an earlier run?); move it aside first"; exit 1; fi
git init -q tavle
git -C tavle fetch -q ../tavle.bundle "refs/remotes/origin/master:refs/heads/frozen-source"
git -C tavle -c advice.detachedHead=false checkout -q --detach "$H"
[[ "$(git -C tavle rev-parse HEAD)" == "$H" ]]
cp FREEZE tavle/research/us/FREEZE
[[ -z "$(git -C tavle status --porcelain -- research/us/model research/us/pipeline)" ]] \
  || { echo "STOP: checkout not clean"; exit 1; }
cd tavle/research/us/model
export US_HOLDOUT_RUN=1 US_CACHE_DIR="$REMOTE/cache"
"$PY" -c 'import lock, sys; ok, why = lock.unlock_status(); print("lock:", why); sys.exit(0 if ok else 1)'
if [[ "$WITH_V1" == "1" ]]; then
  US_RUN_MODE=heldout US_RUN_DIR="$REMOTE/v1run" US_CACHE_DIR="$REMOTE/v1run/cache" "$PY" heldout_v1.py plan
fi
if [[ "$CHECK_ONLY" == "1" ]]; then echo "check only: lock opens for $H; not running"; exit 0; fi
if [[ "$WITH_V1" == "1" ]]; then
  nohup bash -c "\"$PY\" rehearsal.py heldout > \"$REMOTE/heldout.log\" 2>&1; \
    US_RUN_MODE=heldout US_RUN_DIR=\"$REMOTE/v1run\" US_CACHE_DIR=\"$REMOTE/v1run/cache\" \"$PY\" heldout_v1.py all > \"$REMOTE/heldout_v1.log\" 2>&1" \
    < /dev/null > /dev/null 2>&1 &
  echo "started pid $! ; logs $REMOTE/heldout.log then $REMOTE/heldout_v1.log"
else
  nohup "$PY" rehearsal.py heldout > "$REMOTE/heldout.log" 2>&1 < /dev/null &
  echo "started pid $! ; log $REMOTE/heldout.log"
fi
GENE_SIDE

echo "when the log ends with 'held-out run written', copy the results back:"
echo "  scp '$GENE:$REMOTE/tavle/research/us/results/heldout_*' '$REPO/research/us/results/'"
if [[ "$WITH_V1" == "1" ]]; then
  echo "version 1 rows (when heldout_v1.log ends with 'run written'): $GENE:$REMOTE/v1run/side/"
fi
