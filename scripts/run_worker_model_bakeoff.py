"""Worker-model ablation: the FULL frozen evaluation set (30 HotpotQA + 30
2WikiMultiHopQA + 30 MuSiQue + 108 RepoProbe-Python + 80 SWE-QA-Pro = 278
questions) with ANTMAN's worker execution model swapped from GPT-4.1 to a
locally-served checkpoint (Qwen3-8B by default, Qwen3.5-9B as a documented
backup), orchestrator FIXED to gpt-4.1 the whole time.

Uses the EXACT SAME frozen manifests/task_id sets every other baseline and
every other ANTMAN variant in this project (ChainRAG/LongAgent/CoA/Dense/
Sparse Retrieval/S2G-RAG/RepoDistill/the original all-GPT-4.1 ANTMAN run)
was scored against -- not a reduced sample -- so this run's scores are
directly comparable to those frozen numbers, not just a standalone sanity
check.

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
calls are free/local, served by the vLLM box). 278 questions is the same
order of magnitude as the original all-GPT-4.1 ANTMAN document-track +
Track A runs already paid for in this project -- confirm the cost
estimate for this specific run (orchestrator-only, so likely cheaper per
question than the original all-GPT-4.1 run, since worker-side calls are
no longer billed) before letting the full 278 run unattended; this is
deliberately not auto-launched.

Usage:
    # --worker-model defaults to "qwen3-8b" -- omit it for the default run:
    python scripts/run_worker_model_bakeoff.py --worker-base-url http://gh-node-042:8000/v1
    # backup checkpoint, explicit:
    python scripts/run_worker_model_bakeoff.py \\
        --worker-model qwen3.5-9b --worker-base-url http://gh-node-042:8001/v1
    # one benchmark at a time (e.g. to smoke-test before the full 278):
    python scripts/run_worker_model_bakeoff.py --worker-base-url ... --benchmarks hotpotqa
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
from ant.benchmarks.twowikimultihopqa import TwoWikiMultihopQaAdapter  # noqa: E402
from ant.evaluation_suite.runner import run_suite  # noqa: E402

ORCHESTRATOR_MODEL = "gpt-4.1"

# Default worker checkpoint: Qwen3-8B (Apache 2.0, HF repo "Qwen/Qwen3-8B"),
# satisfies a strict "<=8B parameters" reading. Qwen3.5-9B (Apache 2.0, HF
# repo "Qwen/Qwen3.5-9B") is the documented backup/preferred alternative if
# an "8B-class" (not exact-8B) reading is acceptable -- see this ablation's
# own design notes for the tradeoff (stronger agentic/tool-use benchmarks vs.
# being technically 9B). Both names here are the `served-model-name` the
# vLLM server is launched with (see scripts/deltaai_vllm_worker_server.sbatch's
# MODEL_NAME), not the HF repo id itself -- --worker-model must match
# whichever name that server was actually started with.
DEFAULT_WORKER_MODEL = "qwen3-8b"
BACKUP_WORKER_MODEL = "qwen3.5-9b"

# Frozen N=30-per-benchmark document-track set: the UNION of these three
# manifests (5 original + 10 extension + 15 extension) -- same pattern and
# same independent verification as run_document_track_adaptive.py and
# run_s2g_rag_natural_pilot_n30.py, which established this yields exactly
# 30 distinct task_ids per benchmark. A single-manifest default would
# silently run only that one file's own (smaller) rows.
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

DOCUMENT_ADAPTERS = {
    "hotpotqa": HotpotQaAdapter,
    "2wikimultihopqa": TwoWikiMultihopQaAdapter,
    "musique": MuSiQueAdapter,
}
REPO_ADAPTERS = {"repoprobe": RepoProbeAdapter, "sweqa_pro": SweQaProAdapter}
ALL_BENCHMARKS = sorted({*DOCUMENT_ADAPTERS, *REPO_ADAPTERS})


def _document_task_ids(manifest_paths: list[Path]) -> dict[str, list[str]]:
    """Union of task_ids across all given manifest files, per benchmark."""
    merged: dict[str, list[str]] = {}
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for benchmark, rows in manifest["benchmarks"].items():
            seen = merged.setdefault(benchmark, [])
            for row in rows:
                if row["task_id"] not in seen:
                    seen.append(row["task_id"])
    return merged


def _build_task_set(wanted_benchmarks: list[str]) -> dict[str, list]:
    """Returns {benchmark_name: [TaskExample, ...]} for the FULL frozen set
    of each requested benchmark -- 30/30/30 document-track, 108/80 repo-QA,
    never a subsample."""
    doc_ids = _document_task_ids(_DOCUMENT_MANIFESTS)
    tasks: dict[str, list] = {}

    for bench_name in wanted_benchmarks:
        if bench_name in DOCUMENT_ADAPTERS:
            adapter = DOCUMENT_ADAPTERS[bench_name]()
            by_id = {e.task_id: e for e in adapter.load_examples(limit=40)}
            wanted_ids = sorted(doc_ids[bench_name])
            missing = [tid for tid in wanted_ids if tid not in by_id]
            assert not missing, f"STOP: {bench_name} manifest task_ids not found: {missing}"
            examples = [by_id[tid] for tid in wanted_ids]
            assert len(examples) == 30, f"STOP: {bench_name} expected 30 tasks, got {len(examples)}"
            tasks[bench_name] = examples

        elif bench_name == "repoprobe":
            manifest = json.loads(_REPOPROBE_MANIFEST.read_text(encoding="utf-8"))
            assert manifest["frozen_final"] is True
            assert manifest["total_questions"] == 108
            adapter = RepoProbeAdapter()
            all_examples = []
            for short in ["FieldStation42", "adk-python", "agent-framework", "browser-use",
                          "crawl4ai", "docling", "sglang", "yasb"]:
                all_examples.extend(adapter.load_examples(repo_filter=short))
            by_id = {e.task_id: e for e in all_examples}
            wanted_ids = sorted(manifest["task_ids"])
            missing = [tid for tid in wanted_ids if tid not in by_id]
            assert not missing, f"STOP: repoprobe manifest task_ids not found: {missing[:10]}"
            examples = [by_id[tid] for tid in wanted_ids]
            assert len(examples) == 108, f"STOP: repoprobe expected 108 tasks, got {len(examples)}"
            tasks["repoprobe"] = examples

        elif bench_name == "sweqa_pro":
            manifest = json.loads(_SWEQA_MANIFEST.read_text(encoding="utf-8"))
            assert manifest["frozen_final"] is True
            assert manifest["total_task_ids"] == 80
            adapter = SweQaProAdapter()
            all_examples = []
            for _short, full in manifest["included_repositories"].items():
                all_examples.extend(adapter.load_examples(repo_filter=full))
            by_id = {e.task_id: e for e in all_examples}
            wanted_ids = sorted(manifest["task_ids"])
            missing = [tid for tid in wanted_ids if tid not in by_id]
            assert not missing, f"STOP: sweqa_pro manifest task_ids not found: {missing[:10]}"
            examples = [by_id[tid] for tid in wanted_ids]
            assert len(examples) == 80, f"STOP: sweqa_pro expected 80 tasks, got {len(examples)}"
            tasks["sweqa_pro"] = examples

        else:
            raise AssertionError(f"unknown benchmark: {bench_name}")

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
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=ALL_BENCHMARKS,
        choices=ALL_BENCHMARKS,
        help="subset to run (default: all 5, 278 questions total)",
    )
    args = parser.parse_args()

    assert args.worker_base_url.startswith("http://") or args.worker_base_url.startswith("https://"), (
        "STOP: --worker-base-url must be a real, live vLLM server URL -- "
        "this script refuses to run against a placeholder/unset value"
    )

    tasks = _build_task_set(args.benchmarks)
    total = sum(len(v) for v in tasks.values())
    print(f"Task set validated: { {k: len(v) for k, v in tasks.items()} } ({total} total)", flush=True)

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

    document_adapters = {name: cls() for name, cls in DOCUMENT_ADAPTERS.items()}
    repo_adapters = {name: cls() for name, cls in REPO_ADAPTERS.items()}

    all_results = []
    for bench_name, examples in tasks.items():
        is_document = bench_name in DOCUMENT_ADAPTERS
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
