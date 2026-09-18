#!/bin/bash
# Worker-model bake-off: launches ONE vLLM OpenAI-compatible server on a
# DeltaAI GH200 node, serving either Qwen3-8B or Qwen3.5-9B as ANTMAN's
# WORKER-scoped model (select_lookups/plan_worker_actions only -- the
# orchestrator, Need Graph/routing/stuck-detection/reframe/synthesis,
# stays on GPT-4.1 the whole time; see ant.evaluation_suite.vllm_provider
# and ant.coordinator.local.LocalCoordinator's worker_reasoner seam for
# the code side of this split).
#
# BF16, unquantized, on purpose (see this pilot's own design notes): a
# paper experiment comparing worker-model *architectures* should not
# confound the result with a quantization-induced accuracy drop.
#   Qwen3-8B      ~16.4GB BF16 weights
#   Qwen3.5-9B    ~19.3GB BF16 weights
# Both fit comfortably in a single GH200's HBM alongside KV cache at the
# --max-model-len below.
#
# CONTEXT CAP: --max-model-len below is the AUTHORITATIVE enforcement of
# this pilot's "worker sees only what its own tools found, never the
# model's full native context window" design constraint -- not a
# client-side truncation. A request that would exceed it fails loudly at
# the server rather than being silently cut by a second, disagreeing
# heuristic on the ANT side. Do not raise this past 32768 without an
# explicit, disclosed decision to do so -- see the pilot's own design
# notes for why (Qwen3.5's native 262K context existing is not a license
# to hand the worker the whole repo).
#
# BEFORE SUBMITTING, CHECK:
#   1. Does this node have outbound internet to huggingface.co? Many
#      HPC compute partitions do not (only login/data-transfer nodes do).
#      If not: pre-download the checkpoint on a login node first via
#      `huggingface-cli download <MODEL_REPO_ID> --local-dir <path>`
#      and point MODEL_PATH below at that local directory instead of the
#      bare HF repo id -- vLLM's --model accepts either.
#   2. HF_TOKEN env var set if the model repo requires auth (Qwen's own
#      checkpoints are typically ungated, but confirm on the actual repo
#      page before assuming).
#   3. After the job starts, its stdout log (logs/%x-%j.out) prints the
#      node hostname vLLM bound to and the port -- that combination is
#      the worker_base_url the pilot harness needs, e.g.
#      http://<hostname>:8000/v1. This job does not know its own
#      hostname until SLURM assigns the node, so this must be read from
#      the log after submission, not guessed in advance.
#
# Usage:
#   MODEL_REPO_ID="Qwen/Qwen3-8B" MODEL_NAME=qwen3-8b sbatch scripts/deltaai_vllm_worker_server.sbatch
#   MODEL_REPO_ID="Qwen/Qwen3.5-9B" MODEL_NAME=qwen3.5-9b sbatch scripts/deltaai_vllm_worker_server.sbatch
#SBATCH --job-name=vllm-worker
#SBATCH --account=bchx-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

MODEL_REPO_ID="${MODEL_REPO_ID:?set MODEL_REPO_ID, e.g. Qwen/Qwen3-8B}"
MODEL_NAME="${MODEL_NAME:?set MODEL_NAME, the short id the pilot harness will pass as worker_model, e.g. qwen3-8b}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

echo "== vLLM worker server =="
echo "model repo/path : $MODEL_REPO_ID"
echo "served-model-name: $MODEL_NAME"
echo "max-model-len    : $MAX_MODEL_LEN"
echo "node hostname    : $(hostname)"
echo "port             : $PORT"
echo "=> worker_base_url for the pilot harness: http://$(hostname):$PORT/v1"

# A dedicated vLLM venv is expected at .venv-vllm (separate from ANT's own
# .venv -- vLLM pins its own torch/transformers/CUDA stack that need not
# match ANT's dependency set). Create it once with:
#   python -m venv .venv-vllm && .venv-vllm/bin/pip install vllm
source "$REPO_ROOT/.venv-vllm/bin/activate"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_REPO_ID" \
    --served-model-name "$MODEL_NAME" \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --host 0.0.0.0 \
    --port "$PORT"
