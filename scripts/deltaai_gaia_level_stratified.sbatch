#!/bin/bash
# Runs scripts/run_gaia_level_stratified.py as a proper Slurm job instead of
# directly on the login node -- same reasoning as deltaai_worker_model_bakeoff.sbatch:
# this driver does no local GPU/heavy-CPU work (GPT-4.1 + Tavily/DuckDuckGo API
# calls, plus HTTP calls to an already-running vLLM worker server), but a
# ~1.5h wall-clock run of this shape has been confirmed to get killed when run
# directly on the login node.
#
# run_suite(..., resume=True) means a resubmit after a kill/preemption picks
# up exactly where the incremental per-question flush left off.
#
# Usage:
#   WORKER_BASE_URL=http://<vllm-host>:8000/v1 PER_LEVEL=12 MAX_ROUNDS=10 \
#     sbatch scripts/deltaai_gaia_level_stratified.sbatch
#SBATCH --job-name=gaia-level-strat
#SBATCH --account=bchx-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

WORKER_BASE_URL="${WORKER_BASE_URL:?set WORKER_BASE_URL, e.g. http://gh123.hsn.cm.delta.internal.ncsa.edu:8000/v1}"
WORKER_MODEL="${WORKER_MODEL:-qwen3-8b}"
PER_LEVEL="${PER_LEVEL:-12}"
MAX_ROUNDS="${MAX_ROUNDS:-10}"
METHODS="${METHODS:-ant_gaia ant_gaia_h}"

source "$REPO_ROOT/.venv/bin/activate"
set -a
source "$REPO_ROOT/.env"
set +a

python scripts/run_gaia_level_stratified.py \
    --per-level "$PER_LEVEL" \
    --max-rounds "$MAX_ROUNDS" \
    --worker-base-url "$WORKER_BASE_URL" \
    --worker-model "$WORKER_MODEL" \
    --methods $METHODS
