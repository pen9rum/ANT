#!/bin/bash
# RepoDistill (No-Training) on RepoProbe-Python (108) + SWE-QA-Pro (80),
# GPU-accelerated on a DeltaAI GH200 node. See deltaai_repodistill.py's own
# docstring for why this exists as a separate DeltaAI entrypoint: CABA's
# local Qwen2.5-Coder-0.5B perplexity/MMR scoring measured 11-121 minutes
# PER QUESTION on the local CPU-only machine this was developed on (188
# questions total would take days) -- CABA now auto-detects and uses CUDA
# when available (ant.external_wrappers.repodistill_caba's
# _select_device_and_dtype, ANT_REPODISTILL_DEVICE=auto by default), which
# should cut that local-compute cost by a large factor on this node's GPU.
#
# Real paid inference (OPENAI_API_KEY required for the GPT-4.1
# orchestrator/CAPO/answer calls -- CABA itself is local and free either
# way). Review the local machine's own cost-per-question figures (roughly
# $0.01-0.10/question observed there) before submitting if cost is a
# concern; GPU vs CPU changes wall-clock, not $ cost, since CABA was
# never a paid call.
#
# Time limit generous on purpose, same rationale as deltaai_sweqa_dense.sbatch:
# run_suite(..., resume=True) means a run that hits the time limit or gets
# preempted can simply be resubmitted -- already-completed questions
# (one row per question already in the output .jsonl) are skipped, not
# re-answered or re-billed.
#
# Usage:
#   sbatch scripts/deltaai_repodistill.sbatch
#SBATCH --job-name=repodistill
#SBATCH --account=bchx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

# CABA's own perplexity/embed forward passes now run on CUDA by default
# when available (auto-detected) -- no env var needed to opt in. Force CPU
# instead (e.g. for a controlled before/after comparison) with:
#   export ANT_REPODISTILL_DEVICE=cpu
# Force a specific GPU if this node ever has more than one:
#   export ANT_REPODISTILL_DEVICE=cuda:0

# This job has the GPU to itself (no concurrent CPU-bound job on this
# node the way the local machine had job_b's dense embedding fighting
# CABA for cores) -- safe to raise this well past the local default of 4.
# --cpus-per-task=16 above; leave a couple threads of headroom for
# tokenization/data-loading overhead rather than requesting exactly 16.
export ANT_REPODISTILL_TORCH_THREADS="${ANT_REPODISTILL_TORCH_THREADS:-12}"

source "$REPO_ROOT/.venv/bin/activate"

echo "== RepoDistill: RepoProbe-Python (108) + SWE-QA-Pro (80) =="
python scripts/deltaai_repodistill.py

echo "== done: output/runs/repoprobe-python-full/repodistill.jsonl, output/runs/sweqa-pro-80/repodistill.jsonl =="
