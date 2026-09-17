#!/bin/bash
# Job C on DeltaAI: SWE-QA-Pro Dense Retrieval baseline, N=80 (all 8
# repos/80 questions -- this manifest has no small "smoke" subset, unlike
# deltaai_eval.sbatch's tiny default, since deltaai_sweqa_dense.py always
# validates the full frozen manifest before launching).
#
# Real paid inference (OPENAI_API_KEY required, .env already confirmed
# working via `ant openai-smoke`) -- review the estimated question count
# (80, fixed) and repo sizes before submitting if cost is a concern; this
# reuses the exact same DenseRetrievalRepoAgent/synthesize() contract as
# every other Dense Retrieval baseline in this project, so cost/question
# should track what Job B already showed on this machine's local runs.
#
# Time limit generous on purpose: embedding real multi-thousand-chunk repo
# corpora is slow even off a laptop, and DenseRetrievalRepoAgent's own
# incremental-flush cache means a run that hits the time limit or gets
# preempted can simply be resubmitted -- already-flushed files are
# skipped, not re-embedded.
#
# Usage:
#   sbatch scripts/deltaai_sweqa_dense.sbatch
#SBATCH --job-name=sweqa-dense
#SBATCH --account=bchx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs .cache/fastembed_tmp

export TMPDIR="$REPO_ROOT/.cache/fastembed_tmp"
# Same thread-pool caps introduced for job_b's OOM fix -- onnxruntime's CPU
# arena and the tokenizer's own rayon pool both oversubscribe every core by
# default; on a shared/allocation-limited node that's wasteful even when
# --cpus-per-task itself is generous.
export ANT_EMBEDDING_THREADS="${ANT_EMBEDDING_THREADS:-8}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
# Per-inference-call batch size for large-repo embedding -- the arena
# strategy alone did not bound peak memory on the smaller local machine
# this was diagnosed on (see commit 3366cd3); 32 plateaued under 1GB
# there. DeltaAI's --mem=64G has far more headroom, so this can likely go
# higher than the local default without OOM -- 64 is a reasonable first
# try, raise it if flush logs show comfortable memory margin.
export ANT_REPO_EMBED_BATCH_SIZE="${ANT_REPO_EMBED_BATCH_SIZE:-64}"

source "$REPO_ROOT/.venv/bin/activate"

echo "== Job C: SWE-QA-Pro Dense Retrieval, N=80 =="
python scripts/deltaai_sweqa_dense.py

echo "== done: output/runs/sweqa-pro-dense/dense_retrieval.jsonl =="
