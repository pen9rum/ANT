"""Job C -- SWE-QA-Pro Dense Retrieval, N=80, ported for DeltaAI.

Identical logic/contract to the local scratchpad version this was ported
from: same frozen 80-task manifest, same DenseRetrievalRepoAgent (dense
embedding + one synthesize() call per question, no ReAct/condensation),
same resume-by-file-existence pattern via run_suite(..., resume=True).

REPO_ROOT is derived from this file's own location rather than
hardcoded, so it works regardless of the machine/OS it runs on --
deltaai_sweqa_dense.sbatch invokes this from the repo root (same
SLURM_SUBMIT_DIR anchoring deltaai_eval.sbatch already uses), but this
script itself makes no assumption about cwd.

DenseRetrievalRepoAgent's own embedding path (ant.agents.dense_retrieval_repo)
already builds/extends its per-repo index incrementally, flushing to disk
every ANT_REPO_EMBED_BATCH_SIZE*~60 chunks (see that module's
_FLUSH_CHUNK_BUDGET) rather than all-or-nothing -- so a SLURM time limit
or preemption mid-run loses at most the last unflushed group, not
whatever repo was in progress. Set --time generously (embedding 8 real
repos' full text corpora is not fast even on more CPU/RAM than a laptop)
and just resubmit if it runs out -- already-covered files are skipped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ant.agents.dense_retrieval_repo import DenseRetrievalRepoAgent  # noqa: E402
from ant.benchmarks.sweqa_pro import SweQaProAdapter  # noqa: E402
from ant.evaluation_suite.runner import run_suite  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "third_party" / "manifests" / "sweqa_pro" / "sample_manifest_sweqa_pro_80.json"
OUT_DIR = REPO_ROOT / "output" / "runs" / "sweqa-pro-dense"
OUT_PATH = OUT_DIR / "dense_retrieval.jsonl"
TRAJ_DIR = OUT_DIR / "trajectories"
INDEX_ROOT = REPO_ROOT / ".ant" / "eval-suite-dense"

# Smallest-chunk-count-first, same rationale as Job B's own script: surface
# progress and catch problems on a cheap repo before the expensive ones.
REPO_ORDER = ["qibo", "seaborn", "sanic", "yt-dlp", "Pillow", "sqlfluff", "pennylane", "sphinx"]


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["frozen_final"] is True
    assert manifest["total_task_ids"] == 80
    manifest_ids = set(manifest["task_ids"])
    assert len(manifest_ids) == 80
    repos = manifest["included_repositories"]  # short_name -> full_name
    assert set(REPO_ORDER) == set(repos), "REPO_ORDER must match manifest's included_repositories"

    adapter = SweQaProAdapter()
    all_examples = []
    for _short, full in repos.items():
        all_examples.extend(adapter.load_examples(repo_filter=full))
    by_id = {e.task_id: e for e in all_examples}

    missing = manifest_ids - set(by_id)
    assert not missing, f"STOP: {len(missing)} manifest task_ids not found: {sorted(missing)[:10]}"

    examples = [
        by_id[tid]
        for short in REPO_ORDER
        for tid in sorted(manifest_ids)
        if by_id[tid].metadata["repo"] == repos[short]
    ]
    assert len(examples) == 80
    assert len({e.task_id for e in examples}) == 80

    from collections import Counter

    counts = Counter(e.metadata["repo"] for e in examples)
    for short, full in repos.items():
        assert counts.get(full, 0) == 10, f"STOP: {short} expected 10 questions, got {counts.get(full, 0)}"

    for e in examples:
        assert e.reference, f"STOP: {e.task_id} has empty reference"

    checked = set()
    for e in examples:
        r = e.metadata["repo"]
        if r in checked:
            continue
        checked.add(r)
        env_root = adapter.prepare_environment(e)
        assert env_root.exists(), f"STOP: repo checkout missing for {r}"

    print("VALIDATION PASSED: 80/80 task_ids resolved, 8/8 repos checked out (10 questions each), "
          "all references present.")
    print("Proceeding to paid inference.")

    agent = DenseRetrievalRepoAgent(index_root=INDEX_ROOT)
    all_results = []
    for short in REPO_ORDER:
        full = repos[short]
        repo_examples = [e for e in examples if e.metadata["repo"] == full]
        print(f"\n=== {short}: {len(repo_examples)} questions -- building/reusing dense index, then answering ===")
        import time as _time

        started = _time.time()
        results = run_suite(
            benchmark=adapter,
            agent=agent,
            examples=repo_examples,
            out_path=OUT_PATH,
            trajectory_dump_dir=TRAJ_DIR,
            resume=True,
        )
        all_results.extend(results)
        repo_cost = sum(
            r.usage.estimated_cost_usd + r.metric.metadata.get("judge_cost_usd", 0.0) for r in results
        )
        repo_errors = sum(1 for r in results if r.status != "completed")
        print(
            f"=== {short} done: {len(results)} new rows, {repo_errors} errors, "
            f"${repo_cost:.4f} this repo, {round(_time.time() - started, 1)}s elapsed ==="
        )

    total_cost = sum(
        r.usage.estimated_cost_usd + r.metric.metadata.get("judge_cost_usd", 0.0) for r in all_results
    )
    n_errors = sum(1 for r in all_results if r.status != "completed")
    print(f"\nJob C done. N_new_this_run={len(all_results)} errors={n_errors} total_cost_this_run=${total_cost:.4f}")


if __name__ == "__main__":
    main()
