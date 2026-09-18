"""RepoDistill (No-Training) on RepoProbe-Python (108) + SWE-QA-Pro (80),
ported for DeltaAI GPU nodes.

Identical algorithm/contract to the local scratchpad version this was
ported from: same frozen manifests, same RepoDistillAdapter (GraphRAG
retrieval -> CAPO turns -> CABA compression -> answer generation), same
resume-by-file-existence pattern via run_suite(..., resume=True).

THE ONLY REASON THIS SCRIPT EXISTS SEPARATELY FROM THE LOCAL ONE: CABA's
local perplexity/MMR scoring (ant.external_wrappers.repodistill_caba.
QwenPerplexityScorer) now auto-detects and uses CUDA when available (see
that module's `_select_device_and_dtype` -- ANT_REPODISTILL_DEVICE=auto
by default), which on this machine's local CPU-only hardware measured
11-121 minutes per question even after fixing a separate CPU-thread-
oversubscription bug (torch's default thread pool starving a concurrent
CPU-bound job). On a GH200's GPU that local compute should drop by a
large factor -- CABA's own docstring explains why CPU uses float32 and
CUDA uses bfloat16, and this is exactly the small-model-forward-pass
workload GPUs are built for. GPT-4.1 orchestrator/CAPO calls are
unaffected either way (network-bound, not local compute).

ANT_REPODISTILL_TORCH_THREADS still applies (bounds the CPU-side
intra-op pool -- tokenization, data prep -- never GPU kernel scheduling);
raise it from the local default of 4 once this is confirmed running
alone on its own node (see deltaai_repodistill.sbatch).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ant.benchmarks.repoprobe import RepoProbeAdapter  # noqa: E402
from ant.benchmarks.sweqa_pro import SweQaProAdapter  # noqa: E402
from ant.evaluation_suite.runner import run_suite  # noqa: E402
from ant.external_wrappers.repodistill import RepoDistillAdapter  # noqa: E402

OUT_DIR = REPO_ROOT / "output" / "runs"
agent = RepoDistillAdapter()
assert agent.name == "repodistill"
assert agent.model == "gpt-4.1"


def run_repoprobe() -> None:
    manifest_path = REPO_ROOT / "third_party" / "manifests" / "repoprobe" / "sample_manifest_repoprobe_python_full.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_questions"] == 108
    assert manifest["frozen_final"] is True
    manifest_ids = set(manifest["task_ids"])
    assert len(manifest_ids) == 108

    adapter = RepoProbeAdapter()
    all_examples = []
    for short in ["FieldStation42", "adk-python", "agent-framework", "browser-use",
                  "crawl4ai", "docling", "sglang", "yasb"]:
        all_examples.extend(adapter.load_examples(repo_filter=short))
    by_id = {e.task_id: e for e in all_examples}
    missing = manifest_ids - set(by_id)
    assert not missing, f"STOP: RepoProbe {len(missing)} manifest task_ids not found: {sorted(missing)[:10]}"
    examples = [by_id[tid] for tid in sorted(manifest_ids)]
    assert len(examples) == 108
    print("[RepoProbe-Python] VALIDATED: 108/108 frozen task_ids resolved.", flush=True)

    out_path = OUT_DIR / "repoprobe-python-full" / "repodistill.jsonl"
    by_repo: dict[str, list] = {}
    for e in examples:
        by_repo.setdefault(e.metadata.get("repo_short_name", e.metadata.get("repo", "?")), []).append(e)

    all_results = []
    for repo, repo_examples in by_repo.items():
        print(f"\n=== RepoProbe {repo}: {len(repo_examples)} questions ===", flush=True)
        started = time.time()
        results = run_suite(
            benchmark=adapter,
            agent=agent,
            examples=repo_examples,
            out_path=out_path,
            trajectory_dump_dir=OUT_DIR / "repoprobe-python-full" / "trajectories",
            resume=True,
        )
        all_results.extend(results)
        cost = sum(r.usage.estimated_cost_usd + r.metric.metadata.get("judge_cost_usd", 0.0) for r in results)
        errors = sum(1 for r in results if r.status != "completed")
        print(f"=== RepoProbe {repo} done: {len(results)} new rows, {errors} errors, "
              f"${cost:.4f} this repo, {round(time.time() - started, 1)}s elapsed ===", flush=True)

    total_cost = sum(r.usage.estimated_cost_usd + r.metric.metadata.get("judge_cost_usd", 0.0) for r in all_results)
    n_errors = sum(1 for r in all_results if r.status != "completed")
    print(f"\n[RepoProbe-Python] DONE. N_new_this_run={len(all_results)} errors={n_errors} "
          f"total_cost_this_run=${total_cost:.4f}", flush=True)


def run_sweqa_pro() -> None:
    manifest_path = REPO_ROOT / "third_party" / "manifests" / "sweqa_pro" / "sample_manifest_sweqa_pro_80.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["frozen_final"] is True
    assert manifest["total_task_ids"] == 80
    manifest_ids = set(manifest["task_ids"])
    assert len(manifest_ids) == 80
    repos = manifest["included_repositories"]

    adapter = SweQaProAdapter()
    all_examples = []
    for _short, full in repos.items():
        all_examples.extend(adapter.load_examples(repo_filter=full))
    by_id = {e.task_id: e for e in all_examples}
    missing = manifest_ids - set(by_id)
    assert not missing, f"STOP: SWE-QA-Pro {len(missing)} manifest task_ids not found: {sorted(missing)[:10]}"
    examples = [by_id[tid] for tid in sorted(manifest_ids)]
    assert len(examples) == 80
    print("[SWE-QA-Pro] VALIDATED: 80/80 frozen task_ids resolved.", flush=True)

    out_path = OUT_DIR / "sweqa-pro-80" / "repodistill.jsonl"
    by_repo: dict[str, list] = {}
    for e in examples:
        by_repo.setdefault(e.metadata.get("repo", "?"), []).append(e)

    all_results = []
    for repo, repo_examples in by_repo.items():
        print(f"\n=== SWE-QA-Pro {repo}: {len(repo_examples)} questions ===", flush=True)
        started = time.time()
        results = run_suite(
            benchmark=adapter,
            agent=agent,
            examples=repo_examples,
            out_path=out_path,
            trajectory_dump_dir=OUT_DIR / "sweqa-pro-80" / "trajectories",
            resume=True,
        )
        all_results.extend(results)
        cost = sum(r.usage.estimated_cost_usd + r.metric.metadata.get("judge_cost_usd", 0.0) for r in results)
        errors = sum(1 for r in results if r.status != "completed")
        print(f"=== SWE-QA-Pro {repo} done: {len(results)} new rows, {errors} errors, "
              f"${cost:.4f} this repo, {round(time.time() - started, 1)}s elapsed ===", flush=True)

    total_cost = sum(r.usage.estimated_cost_usd + r.metric.metadata.get("judge_cost_usd", 0.0) for r in all_results)
    n_errors = sum(1 for r in all_results if r.status != "completed")
    print(f"\n[SWE-QA-Pro] DONE. N_new_this_run={len(all_results)} errors={n_errors} "
          f"total_cost_this_run=${total_cost:.4f}", flush=True)


if __name__ == "__main__":
    run_repoprobe()
    run_sweqa_pro()
    print("\nALL DONE", flush=True)
