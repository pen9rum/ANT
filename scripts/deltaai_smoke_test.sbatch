#!/bin/bash
# Repeatable smoke test: environment + GPU visibility + ANT core import +
# dense retrieval + one no-API-key coordinator pass. Not a real experiment --
# runs on ghx4-interactive, 1 GPU, a couple of minutes.
#
# Usage: sbatch scripts/deltaai_smoke_test.sbatch
#SBATCH --job-name=ant-smoke
#SBATCH --account=bchx-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00
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

# fastembed's default cache dir is <tempfile.gettempdir()>/fastembed_cache,
# which resolves to node-local /tmp by default -- not visible across login
# vs. compute nodes. Redirecting TMPDIR onto the repo's own (shared
# filesystem) .cache/ keeps one persistent, reusable model cache.
export TMPDIR="$REPO_ROOT/.cache/fastembed_tmp"

# Without this, onnxruntime sizes its thread pool off the WHOLE node's
# logical CPU count (288 on a DeltaAI GH200 node) instead of this job's
# actual --cpus-per-task allocation, then fails pthread_setaffinity_np for
# every thread outside the job's cgroup cpuset -- harmless noise, but a lot
# of it. Confirmed live on gh152.
export ANT_EMBEDDING_THREADS="${SLURM_CPUS_PER_TASK:-16}"

source "$REPO_ROOT/.venv/bin/activate"
python "$REPO_ROOT/scripts/deltaai_smoke_test.py"
