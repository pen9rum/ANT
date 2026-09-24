#!/bin/bash
# Blocks until BOTH ant_gaia and ant_gaia_h reach 103/103 rows (checking
# every 60s), then submits the best-of-2 retry job on ghx4. Meant to be
# launched once in the background and left alone.
set -uo pipefail

WORKER_BASE_URL="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ANT_GAIA_FILE="output/runs/gaia-level-stratified-r10/ant_gaia/ant_gaia.jsonl"
ANT_GAIA_H_FILE="output/runs/gaia-level-stratified-r10/ant_gaia_h/ant_gaia_h.jsonl"

_row_count() {
    local f="$1"
    [ -f "$f" ] && wc -l < "$f" || echo 0
}

echo "[$(date)] Waiting for ant_gaia AND ant_gaia_h to both reach 103/103..."
while true; do
    n_ant_gaia=$(_row_count "$ANT_GAIA_FILE")
    n_ant_gaia_h=$(_row_count "$ANT_GAIA_H_FILE")
    echo "[$(date)] ant_gaia=$n_ant_gaia/103, ant_gaia_h=$n_ant_gaia_h/103"
    if [ "$n_ant_gaia" -ge 103 ] && [ "$n_ant_gaia_h" -ge 103 ]; then
        break
    fi
    sleep 60
done

echo "[$(date)] Both at 103/103. Submitting best-of-2 retry job on ghx4..."
JOB_ID=$(WORKER_BASE_URL="$WORKER_BASE_URL" sbatch scripts/deltaai_gaia_best_of_2_retry.sbatch | awk '{print $4}')
echo "[$(date)] Submitted job $JOB_ID"

while squeue -j "$JOB_ID" -h 2>/dev/null | grep -q .; do
    sleep 60
done
echo "BEST_OF_2_RETRY_DONE (job $JOB_ID)"
