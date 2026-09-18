"""Worker-model bake-off pilot: 30 fixed tasks (5 HotpotQA + 5 MuSiQue +
10 RepoProbe-Python + 10 SWE-QA-Pro) x 2 worker-model checkpoints
(Qwen3-8B, Qwen3.5-9B), orchestrator FIXED to gpt-4.1 the whole time.

Tests whether ANTMAN's Need Graph / routing / stuck-detection / reframe /
synthesis (all GPT-4.1) can effectively delegate LOCAL, territory-scoped
tool-call reasoning (select_lookups/plan_worker_actions -- see
ant.coordinator.local.LocalCoordinator's worker_reasoner seam and
ant.evaluation_suite.vllm_provider's own docstring) to a much smaller
model served locally via vLLM, without materially hurting end-task score.

REQUIRES a live vLLM OpenAI-compatible server already running (see
scripts/deltaai_vllm_worker_server.sbatch) -- this script does not start
one itself and refuses to run against a default/placeholder URL.

Cost note: real paid inference (GPT-4.1, orchestrator side only -- worker
calls are free/local). Review the printed cost estimate this script
prints for its FIRST few completed rows before letting the rest run; this
is deliberately not auto-launched.

Usage:
    # --worker-model defaults to "qwen3-8b" -- omit it for the default run:
    python scripts/run_worker_model_bakeoff.py --worker-base-url http://gh-node-042:8000/v1
    # backup checkpoint, explicit:
    python scripts/run_worker_model_bakeoff.py \\
        --worker-model qwen3.5-9b --worker-base-url http://gh-node-042:8001/v1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ant.agents.ant_adapter import AntAgent  # noqa: E402
from ant.agents.ant_document_adapter import AntDocumentAgent  # noqa: E402
from ant.benchmarks.hotpotqa import HotpotQaAdapter  # noqa: E402
from ant.benchmarks.musique import MuSiQueAdapter  # noqa: E402
from ant.benchmarks.repoprobe import RepoProbeAdapter  # noqa: E402
from ant.benchmarks.sweqa_pro import SweQaProAdapter  # noqa: E402
from ant.evaluation_suite.runner import run_suite  # noqa: E402

ORCHESTRATOR_MODEL = "gpt-4.1"

# Default worker checkpoint: Qwen3-8B (Apache 2.0, HF repo "Qwen/Qwen3-8B"),
# satisfies a strict "<=8B parameters" reading. Qwen3.5-9B (Apache 2.0, HF
# repo "Qwen/Qwen3.5-9B") is the documented backup/preferred alternative if
# an "8B-class" (not exact-8B) reading is acceptable -- see this pilot's own
# design notes for the tradeoff (stronger agentic/tool-use benchmarks vs.
# being technically 9B). Both names here are the `served-model-name` the
# vLLM server is launched with (see scripts/deltaai_vllm_worker_server.sbatch's
# MODEL_NAME), not the HF repo id itself -- --worker-model must match
# whichever name that server was actually started with.
DEFAULT_WORKER_MODEL = "qwen3-8b"
BACKUP_WORKER_MODEL = "qwen3.5-9b"

_MANIFEST_DIR = REPO_ROOT / "third_party" / "manifests" / "long_context"
_DOCUMENT_MANIFESTS = [
    _MANIFEST_DIR / "natural_pilot_manifest_15task.json",
    _MANIFEST_DIR / "natural_pilot_manifest_10new_per_benchmark.json",
    _MANIFEST_DIR / "natural_pilot_manifest_15task_extension_16to30.json",
]
_REPOPROBE_MANIFEST = (
    REPO_ROOT / "third_party" / "manifests" / "repoprobe" / "sample_manifest_repoprobe_python_full.json"
)
_SWEQA_MANIFEST = (
    REPO_ROOT / "third_party" / "manifests" / "sweqa_pro" / "sample_manifest_sweqa_pro_80.json"
)

# Fixed sample sizes (30 total): 5 HotpotQA + 5 MuSiQue + 10 RepoProbe +
# 10 SWE-QA-Pro. First N task_ids of each frozen manifest, sorted -- a
# deterministic, reproducible sample, not cherry-picked by outcome (chosen
# BEFORE this pilot's own results exist).
_DOCUMENT_SAMPLE_N = 5
_REPO_SAMPLE_N = 10


def _document_task_ids(manifest_paths: list[Path]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for benchmark, rows in manifest["benchmarks"].items():
            seen = merged.setdefault(benchmark, [])
            for row in rows:
                if row["task_id"] not in seen:
                    seen.append(row["task_id"])
    return merged


def _build_task_set() -> dict[str, list]:
    """Returns {benchmark_name: [TaskExample, ...]} for the fixed 30-task
    sample, one adapter load per benchmark."""
    doc_ids = _document_task_ids(_DOCUMENT_MANIFESTS)
    tasks: dict[str, list] = {}

    hotpot_adapter = HotpotQaAdapter()
    hotpot_by_id = {e.task_id: e for e in hotpot_adapter.load_examples(limit=40)}
    hotpot_sample_ids = sorted(doc_ids["hotpotqa"])[:_DOCUMENT_SAMPLE_N]
    missing = [tid for tid in hotpot_sample_ids if tid not in hotpot_by_id]
    assert not missing, f"STOP: hotpotqa sample task_ids not found: {missing}"
    tasks["hotpotqa"] = [hotpot_by_id[tid] for tid in hotpot_sample_ids]

    musique_adapter = MuSiQueAdapter()
    musique_by_id = {e.task_id: e for e in musique_adapter.load_examples(limit=40)}
    musique_sample_ids = sorted(doc_ids["musique"])[:_DOCUMENT_SAMPLE_N]
    missing = [tid for tid in musique_sample_ids if tid not in musique_by_id]
    assert not missing, f"STOP: musique sample task_ids not found: {missing}"
    tasks["musique"] = [musique_by_id[tid] for tid in musique_sample_ids]

    repoprobe_manifest = json.loads(_REPOPROBE_MANIFEST.read_text(encoding="utf-8"))
    assert repoprobe_manifest["frozen_final"] is True
    repoprobe_adapter = RepoProbeAdapter()
    repoprobe_all = []
    for short in ["FieldStation42", "adk-python", "agent-framework", "browser-use",
                  "crawl4ai", "docling", "sglang", "yasb"]:
        repoprobe_all.extend(repoprobe_adapter.load_examples(repo_filter=short))
    repoprobe_by_id = {e.task_id: e for e in repoprobe_all}
    repoprobe_sample_ids = sorted(repoprobe_manifest["task_ids"])[:_REPO_SAMPLE_N]
    missing = [tid for tid in repoprobe_sample_ids if tid not in repoprobe_by_id]
    assert not missing, f"STOP: repoprobe sample task_ids not found: {missing}"
    tasks["repoprobe"] = [repoprobe_by_id[tid] for tid in repoprobe_sample_ids]

    sweqa_manifest = json.loads(_SWEQA_MANIFEST.read_text(encoding="utf-8"))
    assert sweqa_manifest["frozen_final"] is True
    sweqa_adapter = SweQaProAdapter()
    sweqa_all = []
    for _short, full in sweqa_manifest["included_repositories"].items():
        sweqa_all.extend(sweqa_adapter.load_examples(repo_filter=full))
    sweqa_by_id = {e.task_id: e for e in sweqa_all}
    sweqa_sample_ids = sorted(sweqa_manifest["task_ids"])[:_REPO_SAMPLE_N]
    missing = [tid for tid in sweqa_sample_ids if tid not in sweqa_by_id]
    assert not missing, f"STOP: sweqa_pro sample task_ids not found: {missing}"
    tasks["sweqa_pro"] = [sweqa_by_id[tid] for tid in sweqa_sample_ids]

    total = sum(len(v) for v in tasks.values())
    assert total == 30, f"STOP: expected exactly 30 tasks, got {total}: {{k: len(v) for k, v in tasks.items()}}"
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker-model",
        default=DEFAULT_WORKER_MODEL,
        help=f"served-model-name on the vLLM server (default: {DEFAULT_WORKER_MODEL}; "
        f"pass {BACKUP_WORKER_MODEL!r} for the backup checkpoint)",
    )
    parser.add_argument(
        "--worker-base-url",
        required=True,
        help="e.g. http://gh-node-042:8000/v1 -- no default, must be a real, live vLLM server",
    )
    parser.add_argument("--worker-max-context-tokens", type=int, default=16384)
    parser.add_argument("--max-rounds", type=int, default=6)
    args = parser.parse_args()

    assert args.worker_base_url.startswith("http://") or args.worker_base_url.startswith("https://"), (
        "STOP: --worker-base-url must be a real, live vLLM server URL -- "
        "this script refuses to run against a placeholder/unset value"
    )

    tasks = _build_task_set()
    print(f"Task set validated: { {k: len(v) for k, v in tasks.items()} } (30 total)", flush=True)

    out_dir = REPO_ROOT / "output" / "runs" / "worker-model-bakeoff" / args.worker_model
    doc_agent = AntDocumentAgent(
        model=ORCHESTRATOR_MODEL,
        max_rounds=args.max_rounds,
        worker_model=args.worker_model,
        worker_base_url=args.worker_base_url,
        worker_max_context_tokens=args.worker_max_context_tokens,
    )
    repo_agent = AntAgent(
        model=ORCHESTRATOR_MODEL,
        max_rounds=args.max_rounds,
        worker_model=args.worker_model,
        worker_base_url=args.worker_base_url,
        worker_max_context_tokens=args.worker_max_context_tokens,
    )

    document_adapters = {"hotpotqa": HotpotQaAdapter(), "musique": MuSiQueAdapter()}
    repo_adapters = {"repoprobe": RepoProbeAdapter(), "sweqa_pro": SweQaProAdapter()}

    all_results = []
    for bench_name, examples in tasks.items():
        is_document = bench_name in document_adapters
        adapter = document_adapters[bench_name] if is_document else repo_adapters[bench_name]
        agent = doc_agent if is_document else repo_agent
        out_path = out_dir / bench_name / f"ant_worker_{args.worker_model}.jsonl"

        print(f"\n=== {bench_name} ({'document' if is_document else 'repo'}-track): "
              f"{len(examples)} questions, worker_model={args.worker_model} ===", flush=True)
        started = time.time()
        results = run_suite(
            benchmark=adapter,
            agent=agent,
            examples=examples,
            out_path=out_path,
            trajectory_dump_dir=out_dir / bench_name / "trajectories",
            resume=True,
        )
        all_results.extend(results)
        cost = sum(r.usage.estimated_cost_usd for r in results)
        errors = sum(1 for r in results if r.status != "completed")
        print(f"=== {bench_name} done: {len(results)} rows, {errors} errors, "
              f"${cost:.4f} orchestrator cost, {round(time.time() - started, 1)}s ===", flush=True)

    total_cost = sum(r.usage.estimated_cost_usd for r in all_results)
    n_errors = sum(1 for r in all_results if r.status != "completed")
    print(f"\nDone. worker_model={args.worker_model} N={len(all_results)} errors={n_errors} "
          f"total_orchestrator_cost_this_run=${total_cost:.4f}", flush=True)


if __name__ == "__main__":
    main()
