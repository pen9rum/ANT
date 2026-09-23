#!/bin/bash
# Runs scripts/run_gaia_bakeoff.py as a proper Slurm job -- same
# login-node-avoidance reasoning as the other GAIA sbatch scripts here.
#
# Usage:
#   METHODS="dense_retrieval_gaia" sbatch scripts/deltaai_gaia_bakeoff.sbatch
#   METHODS="ant_gaia_h" WORKER_BASE_URL=http://host:8000/v1 \
#     sbatch scripts/deltaai_gaia_bakeoff.sbatch
#SBATCH --job-name=gaia-bakeoff
#SBATCH --account=bchx-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

METHODS="${METHODS:?set METHODS, e.g. \"dense_retrieval_gaia\"}"
WORKER_BASE_URL="${WORKER_BASE_URL:-}"
LIMIT="${LIMIT:-}"

source "$REPO_ROOT/.venv/bin/activate"
set -a
source "$REPO_ROOT/.env"
set +a

ARGS=(--methods $METHODS)
if [ -n "$WORKER_BASE_URL" ]; then
    ARGS+=(--worker-base-url "$WORKER_BASE_URL")
fi
if [ -n "$LIMIT" ]; then
    ARGS+=(--limit "$LIMIT")
fi

python scripts/run_gaia_bakeoff.py "${ARGS[@]}"
