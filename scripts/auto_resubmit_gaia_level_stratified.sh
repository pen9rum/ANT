#!/bin/bash
# Watches one GAIA level-stratified job (one or more space-separated methods,
# e.g. "ant_gaia ant_gaia_h" run together in one sbatch submission) and,
# every time it leaves the queue (whether by normal completion or hitting
# its --time limit), checks EVERY method's output jsonl row count. If any
# method is still short of the full 103, resubmits the identical sbatch
# command with the SAME method set (resume=True means already-completed
# rows, for methods that already finished, are never redone -- run_suite
# just walks straight through them) and keeps watching the new job.
# Repeats until every method reaches 103/103 or a real error is detected.
#
# Usage:
#   ./scripts/auto_resubmit_gaia_level_stratified.sh "<methods>" <first_job_id> <worker_base_url>
#   ./scripts/auto_resubmit_gaia_level_stratified.sh "ant_gaia ant_gaia_h" 3201252 http://host:8000/v1
set -uo pipefail

METHODS="$1"
JOB_ID="$2"
WORKER_BASE_URL="$3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TARGET=103

_all_done() {
    for m in $METHODS; do
        f="output/runs/gaia-level-stratified-r10/${m}/${m}.jsonl"
        n=0
        [ -f "$f" ] && n=$(wc -l < "$f")
        echo "[$(date)] $m: $n/$TARGET rows."
        [ "$n" -lt "$TARGET" ] && return 1
    done
    return 0
}

while true; do
    echo "[$(date)] Watching job $JOB_ID for methods: $METHODS ..."
    while squeue -j "$JOB_ID" -h 2>/dev/null | grep -q .; do
        sleep 30
    done
    echo "[$(date)] Job $JOB_ID left the queue."

    if _all_done; then
        echo "[$(date)] All methods DONE."
        break
    fi

    echo "[$(date)] Not all methods done yet, resubmitting: $METHODS ..."
    NEW_JOB_ID=$(METHODS="$METHODS" PER_LEVEL=999 MAX_ROUNDS=10 WORKER_BASE_URL="$WORKER_BASE_URL" \
        sbatch --partition=ghx4 --time=2:00:00 scripts/deltaai_gaia_level_stratified.sbatch \
        | awk '{print $4}')
    if [ -z "$NEW_JOB_ID" ]; then
        echo "[$(date)] sbatch resubmission FAILED. Stopping auto-resubmit loop."
        exit 1
    fi
    echo "[$(date)] Resubmitted as job $NEW_JOB_ID."
    JOB_ID="$NEW_JOB_ID"
done

echo "AUTO_RESUBMIT_COMPLETE"
