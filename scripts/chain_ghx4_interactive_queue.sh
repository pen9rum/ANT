#!/bin/bash
# Runs ant_gaia_h -> s2g_rag_gaia -> ant_gaia -> best-of-2 retry (ant_gaia +
# ant_gaia_h) -> dense_retrieval_gaia -> owl_gaia (36-question stratified
# sample) in STRICT sequence on ghx4-interactive (the fast-turnaround,
# 1-job-per-user QOS partition). Each stage is watched to full completion
# before the next stage's job is even submitted -- see this script's
# earlier revisions in git history for why (a resubmitted earlier stage
# getting a fresh, later submission timestamp than an already-queued
# later stage would let the later one jump ahead).
#
# Stages 1/2 (ant_gaia_h, s2g_rag_gaia) are idempotent guards here: if
# they already reached 103/103 in an earlier invocation of this script,
# their while-loop exits immediately without submitting anything.
#
# Usage:
#   ./scripts/chain_ghx4_interactive_queue.sh <worker_base_url> [first_job_id_for_current_stage]
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
N=$(_row_count "output/runs/gaia-level-stratified-r10/ant_gaia_h/ant_gaia_h.jsonl")
if [ "$N" -lt 103 ] && [ -z "$JOB_ID" ]; then
    JOB_ID=$(METHODS="ant_gaia_h" PER_LEVEL=999 MAX_ROUNDS=10 WORKER_BASE_URL="$WORKER_BASE_URL" \
        sbatch scripts/deltaai_gaia_level_stratified.sbatch | awk '{print $4}')
    echo "[$(date)] Submitted ant_gaia_h as job $JOB_ID"
fi
while true; do
    N=$(_row_count "output/runs/gaia-level-stratified-r10/ant_gaia_h/ant_gaia_h.jsonl")
    [ "$N" -ge 103 ] && break
    [ -n "$JOB_ID" ] && _wait_for_job "$JOB_ID"
    N=$(_row_count "output/runs/gaia-level-stratified-r10/ant_gaia_h/ant_gaia_h.jsonl")
    echo "[$(date)] ant_gaia_h: $N/103 rows."
    [ "$N" -ge 103 ] && break
    JOB_ID=$(METHODS="ant_gaia_h" PER_LEVEL=999 MAX_ROUNDS=10 WORKER_BASE_URL="$WORKER_BASE_URL" \
        sbatch scripts/deltaai_gaia_level_stratified.sbatch | awk '{print $4}')
    echo "[$(date)] ant_gaia_h: resubmitted as job $JOB_ID"
done
echo "[$(date)] === Stage 1 DONE: ant_gaia_h 103/103 ==="
JOB_ID=""

echo "[$(date)] === Stage 2: s2g_rag_gaia ==="
N=$(_row_count "output/runs/gaia-level-stratified-r10/s2g_rag_gaia/s2g_rag_gaia.jsonl")
if [ "$N" -lt 103 ]; then
    JOB_ID=$(METHODS="s2g_rag_gaia" PER_LEVEL=999 SKIP_PER_LEVEL=0 \
        sbatch scripts/deltaai_gaia_level_stratified_baselines.sbatch | awk '{print $4}')
    echo "[$(date)] Submitted s2g_rag_gaia as job $JOB_ID"
fi
while true; do
    N=$(_row_count "output/runs/gaia-level-stratified-r10/s2g_rag_gaia/s2g_rag_gaia.jsonl")
    [ "$N" -ge 103 ] && break
    [ -n "$JOB_ID" ] && _wait_for_job "$JOB_ID"
    N=$(_row_count "output/runs/gaia-level-stratified-r10/s2g_rag_gaia/s2g_rag_gaia.jsonl")
    echo "[$(date)] s2g_rag_gaia: $N/103 rows."
    [ "$N" -ge 103 ] && break
    JOB_ID=$(METHODS="s2g_rag_gaia" PER_LEVEL=999 SKIP_PER_LEVEL=0 \
        sbatch scripts/deltaai_gaia_level_stratified_baselines.sbatch | awk '{print $4}')
    echo "[$(date)] s2g_rag_gaia: resubmitted as job $JOB_ID"
done
echo "[$(date)] === Stage 2 DONE: s2g_rag_gaia 103/103 ==="

echo "[$(date)] === Stage 3: ant_gaia ==="
JOB_ID=$(METHODS="ant_gaia" PER_LEVEL=999 MAX_ROUNDS=10 WORKER_BASE_URL="$WORKER_BASE_URL" \
    sbatch scripts/deltaai_gaia_level_stratified.sbatch | awk '{print $4}')
echo "[$(date)] Submitted ant_gaia as job $JOB_ID"
while true; do
    _wait_for_job "$JOB_ID"
    N=$(_row_count "output/runs/gaia-level-stratified-r10/ant_gaia/ant_gaia.jsonl")
    echo "[$(date)] ant_gaia: $N/103 rows."
    [ "$N" -ge 103 ] && break
    JOB_ID=$(METHODS="ant_gaia" PER_LEVEL=999 MAX_ROUNDS=10 WORKER_BASE_URL="$WORKER_BASE_URL" \
        sbatch scripts/deltaai_gaia_level_stratified.sbatch | awk '{print $4}')
    echo "[$(date)] ant_gaia: resubmitted as job $JOB_ID"
done
echo "[$(date)] === Stage 3 DONE: ant_gaia 103/103 ==="

echo "[$(date)] === Stage 4: best-of-2 retry (ant_gaia + ant_gaia_h) ==="
JOB_ID=$(WORKER_BASE_URL="$WORKER_BASE_URL" \
    sbatch scripts/deltaai_gaia_best_of_2_retry.sbatch | awk '{print $4}')
echo "[$(date)] Submitted best-of-2 retry as job $JOB_ID"
while true; do
    _wait_for_job "$JOB_ID"
    N1=$(_row_count "output/runs/gaia-level-stratified-r10/ant_gaia/ant_gaia.best_of_2.jsonl")
    N2=$(_row_count "output/runs/gaia-level-stratified-r10/ant_gaia_h/ant_gaia_h.best_of_2.jsonl")
    echo "[$(date)] best-of-2: ant_gaia=$N1/103, ant_gaia_h=$N2/103."
    [ "$N1" -ge 103 ] && [ "$N2" -ge 103 ] && break
    JOB_ID=$(WORKER_BASE_URL="$WORKER_BASE_URL" \
        sbatch scripts/deltaai_gaia_best_of_2_retry.sbatch | awk '{print $4}')
    echo "[$(date)] best-of-2: resubmitted as job $JOB_ID"
done
echo "[$(date)] === Stage 4 DONE: best-of-2 retry complete for both methods ==="

echo "[$(date)] === Stage 5: dense_retrieval_gaia ==="
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
echo "[$(date)] === Stage 5 DONE: dense_retrieval_gaia 103/103 ==="

echo "[$(date)] === Stage 6: owl_gaia (36-question stratified sample, NOT full 103) ==="
JOB_ID=$(METHODS="owl_gaia" PER_LEVEL=12 SKIP_PER_LEVEL=0 \
    sbatch scripts/deltaai_gaia_level_stratified_baselines.sbatch | awk '{print $4}')
echo "[$(date)] Submitted owl_gaia as job $JOB_ID"
while true; do
    _wait_for_job "$JOB_ID"
    N=$(_row_count "output/runs/gaia-level-stratified-r10/owl_gaia/owl_gaia.jsonl")
    echo "[$(date)] owl_gaia: $N/36 rows."
    [ "$N" -ge 36 ] && break
    JOB_ID=$(METHODS="owl_gaia" PER_LEVEL=12 SKIP_PER_LEVEL=0 \
        sbatch scripts/deltaai_gaia_level_stratified_baselines.sbatch | awk '{print $4}')
    echo "[$(date)] owl_gaia: resubmitted as job $JOB_ID"
done
echo "[$(date)] === Stage 6 DONE: owl_gaia 36/36 ==="

echo "CHAIN_ALL_DONE"
