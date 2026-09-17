#!/bin/bash
# Real ANT experiment entrypoint on DeltaAI: `ant index` then `ant eval`
# (src/ant/cli.py's own two-step CLI flow), on the ghx4 partition.
#
# Defaults to the tiny 2-question self-referential fixture with
# --synthesize none/--judge heuristic (no OPENAI_API_KEY required) so an
# unparametrized `sbatch` submission can never accidentally become a large,
# expensive run. Override DATASET/REPO/LIMIT/SYNTHESIZE/JUDGE via
# --export=ALL,VAR=value for a real dataset -- and get the resulting task
# count/resource estimate reviewed before submitting one that isn't this
# tiny default.
#
# Usage:
#   sbatch scripts/deltaai_eval.sbatch                     # tiny smoke default
#   sbatch --export=ALL,DATASET=output/real.jsonl,LIMIT=20,SYNTHESIZE=openai \
#     scripts/deltaai_eval.sbatch                           # real run, reviewed first
#SBATCH --job-name=ant-eval
#SBATCH --account=bchx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# sbatch copies the submitted script to /var/spool/slurmd/job<id>/slurm_script
# and runs it from there, so deriving REPO_ROOT from BASH_SOURCE resolves to
# that spool dir (no write access) instead of the repo -- confirmed live.
# SLURM_SUBMIT_DIR (the directory `sbatch` was invoked from) is the correct
# anchor.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs .cache/fastembed_tmp

export TMPDIR="$REPO_ROOT/.cache/fastembed_tmp"
export ANT_EMBEDDING_THREADS="${SLURM_CPUS_PER_TASK:-16}"

source "$REPO_ROOT/.venv/bin/activate"

# --- parameters (override via sbatch --export=ALL,VAR=value) ---
DATASET="${DATASET:-scripts/fixtures/tiny_self_eval.jsonl}"
REPO="${REPO:-src/ant/domain}"
INDEX_PATH="${INDEX_PATH:-.ant/deltaai-run}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID}}"
LIMIT="${LIMIT:-}"
MAX_ROUNDS="${MAX_ROUNDS:-2}"
SYNTHESIZE="${SYNTHESIZE:-none}"
JUDGE="${JUDGE:-heuristic}"

echo "== ant index =="
ant index "$REPO" --out "$INDEX_PATH"

echo "== ant eval =="
LIMIT_ARGS=()
if [[ -n "$LIMIT" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi
ant eval "$DATASET" \
  --repo "$REPO" \
  --index "$INDEX_PATH" \
  --run-id "$RUN_ID" \
  --max-rounds "$MAX_ROUNDS" \
  --synthesize "$SYNTHESIZE" \
  --judge "$JUDGE" \
  "${LIMIT_ARGS[@]}"

echo "== done: output/runs/${RUN_ID}/results.jsonl + summary.json =="
