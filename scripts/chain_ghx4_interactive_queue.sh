#!/bin/bash
# Runs ant_gaia_h -> s2g_rag_gaia -> dense_retrieval_gaia in STRICT sequence
# on ghx4-interactive (the fast-turnaround, 1-job-per-user QOS partition).
# Each stage is watched to full completion (103/103 rows, resubmitting on
# every timeout) BEFORE the next stage's job is even submitted -- so a
# later stage can never jump ahead of an earlier one still in progress,
# unlike pre-submitting all three up front (where a resubmitted earlier
# stage would get a fresh, later submission timestamp than an
# already-queued later stage).
#
# Usage:
#   ./scripts/chain_ghx4_interactive_queue.sh <worker_base_url> [first_ant_gaia_h_job_id]
set -uo pipefail

WORKER_BASE_URL="$1"
FIRST_JOB_ID="${2:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

_wait_for_job() {
    local job_id="$1"
    while squeue -j "$job_id" -h 2>/dev/null | grep -q .; do
        sleep 30
    done
}

_row_count() {
    local f="$1"
    [ -f "$f" ] && wc -l < "$f" || echo 0
}

echo "[$(date)] === Stage 1: ant_gaia_h ==="
JOB_ID="$FIRST_JOB_ID"
if [ -z "$JOB_ID" ]; then
    JOB_ID=$(METHODS="ant_gaia_h" PER_LEVEL=999 MAX_ROUNDS=10 WORKER_BASE_URL="$WORKER_BASE_URL" \
        sbatch scripts/deltaai_gaia_level_stratified.sbatch | awk '{print $4}')
    echo "[$(date)] Submitted ant_gaia_h as job $JOB_ID"
fi
while true; do
    _wait_for_job "$JOB_ID"
    N=$(_row_count "output/runs/gaia-level-stratified-r10/ant_gaia_h/ant_gaia_h.jsonl")
    echo "[$(date)] ant_gaia_h: $N/103 rows."
    [ "$N" -ge 103 ] && break
    JOB_ID=$(METHODS="ant_gaia_h" PER_LEVEL=999 MAX_ROUNDS=10 WORKER_BASE_URL="$WORKER_BASE_URL" \
        sbatch scripts/deltaai_gaia_level_stratified.sbatch | awk '{print $4}')
    echo "[$(date)] ant_gaia_h: resubmitted as job $JOB_ID"
done
echo "[$(date)] === Stage 1 DONE: ant_gaia_h 103/103 ==="

echo "[$(date)] === Stage 2: s2g_rag_gaia ==="
JOB_ID=$(METHODS="s2g_rag_gaia" PER_LEVEL=999 SKIP_PER_LEVEL=0 \
    sbatch scripts/deltaai_gaia_level_stratified_baselines.sbatch | awk '{print $4}')
echo "[$(date)] Submitted s2g_rag_gaia as job $JOB_ID"
while true; do
    _wait_for_job "$JOB_ID"
    N=$(_row_count "output/runs/gaia-level-stratified-r10/s2g_rag_gaia/s2g_rag_gaia.jsonl")
    echo "[$(date)] s2g_rag_gaia: $N/103 rows."
    [ "$N" -ge 103 ] && break
    JOB_ID=$(METHODS="s2g_rag_gaia" PER_LEVEL=999 SKIP_PER_LEVEL=0 \
        sbatch scripts/deltaai_gaia_level_stratified_baselines.sbatch | awk '{print $4}')
    echo "[$(date)] s2g_rag_gaia: resubmitted as job $JOB_ID"
done
echo "[$(date)] === Stage 2 DONE: s2g_rag_gaia 103/103 ==="

echo "[$(date)] === Stage 3: dense_retrieval_gaia ==="
JOB_ID=$(METHODS="dense_retrieval_gaia" sbatch scripts/deltaai_gaia_bakeoff.sbatch | awk '{print $4}')
echo "[$(date)] Submitted dense_retrieval_gaia as job $JOB_ID"
while true; do
    _wait_for_job "$JOB_ID"
    N=$(_row_count "output/runs/gaia-text-103/dense_retrieval_gaia/dense_retrieval_gaia.jsonl")
    echo "[$(date)] dense_retrieval_gaia: $N/103 rows."
    [ "$N" -ge 103 ] && break
    JOB_ID=$(METHODS="dense_retrieval_gaia" sbatch scripts/deltaai_gaia_bakeoff.sbatch | awk '{print $4}')
    echo "[$(date)] dense_retrieval_gaia: resubmitted as job $JOB_ID"
done
echo "[$(date)] === Stage 3 DONE: dense_retrieval_gaia 103/103 ==="

echo "CHAIN_ALL_DONE"
