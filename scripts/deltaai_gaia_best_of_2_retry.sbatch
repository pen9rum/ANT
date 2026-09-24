#!/bin/bash
# Runs scripts/run_gaia_best_of_2_retry.py as a proper Slurm job on ghx4.
# Usage:
#   WORKER_BASE_URL=http://host:8000/v1 sbatch scripts/deltaai_gaia_best_of_2_retry.sbatch
#SBATCH --job-name=gaia-best-of-2
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

WORKER_BASE_URL="${WORKER_BASE_URL:?set WORKER_BASE_URL}"

source "$REPO_ROOT/.venv/bin/activate"
set -a
source "$REPO_ROOT/.env"
set +a

python scripts/run_gaia_best_of_2_retry.py \
    --methods ant_gaia ant_gaia_h \
    --worker-base-url "$WORKER_BASE_URL"
