#!/bin/bash
# Watches one GAIA level-stratified job (ant_gaia or ant_gaia_h) and, every
# time it leaves the queue (whether by normal completion or hitting its
# --time limit), checks the output jsonl's row count. If not yet at the
# full 103, resubmits the identical sbatch command (resume=True means the
# already-completed rows are never redone) and keeps watching the new job.
# Repeats until 103/103 or a real error is detected. Meant to be run once
# in the background per method.
#
# Usage:
#   ./scripts/auto_resubmit_gaia_level_stratified.sh <method> <first_job_id> <worker_base_url>
set -uo pipefail

METHOD="$1"
JOB_ID="$2"
WORKER_BASE_URL="$3"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUT_FILE="output/runs/gaia-level-stratified-r10/${METHOD}/${METHOD}.jsonl"
TARGET=103

while true; do
    echo "[$(date)] Watching job $JOB_ID for $METHOD..."
    while squeue -j "$JOB_ID" -h 2>/dev/null | grep -q .; do
        sleep 30
    done
    echo "[$(date)] Job $JOB_ID for $METHOD left the queue."

    N=0
    if [ -f "$OUT_FILE" ]; then
        N=$(wc -l < "$OUT_FILE")
    fi
    echo "[$(date)] $METHOD: $N/$TARGET rows so far."

    if [ "$N" -ge "$TARGET" ]; then
        echo "[$(date)] $METHOD: DONE ($N/$TARGET)."
        break
    fi

    echo "[$(date)] $METHOD: not done yet, resubmitting..."
    NEW_JOB_ID=$(METHODS="$METHOD" PER_LEVEL=999 MAX_ROUNDS=10 WORKER_BASE_URL="$WORKER_BASE_URL" \
        sbatch --partition=ghx4 --time=2:00:00 scripts/deltaai_gaia_level_stratified.sbatch \
        | awk '{print $4}')
    if [ -z "$NEW_JOB_ID" ]; then
        echo "[$(date)] $METHOD: sbatch resubmission FAILED. Stopping auto-resubmit loop."
        exit 1
    fi
    echo "[$(date)] $METHOD: resubmitted as job $NEW_JOB_ID."
    JOB_ID="$NEW_JOB_ID"
done

echo "AUTO_RESUBMIT_${METHOD}_COMPLETE"
