#!/bin/bash
# Runs scripts/run_gaia_level_stratified_baselines.py (chosen GAIA baseline
# agents on the same 36-question level-stratified GAIA-Text-103 subset) as a
# proper Slurm job -- same login-node-avoidance reasoning as the other GAIA
# sbatch scripts in this directory.
#
# Usage:
#   METHODS="direct_gaia sparse_retrieval_gaia" \
#     sbatch scripts/deltaai_gaia_level_stratified_baselines.sbatch
#SBATCH --job-name=gaia-baselines-strat
#SBATCH --account=bchx-dtai-gh
#SBATCH --partition=ghx4
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

METHODS="${METHODS:?set METHODS, e.g. \"direct_gaia sparse_retrieval_gaia\"}"

source "$REPO_ROOT/.venv/bin/activate"
set -a
source "$REPO_ROOT/.env"
set +a

python scripts/run_gaia_level_stratified_baselines.py --methods $METHODS
