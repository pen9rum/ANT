"""One-off: run a chosen set of GAIA baseline agents on the SAME 36-question
level-stratified GAIA-Text-103 subset used by run_gaia_level_stratified.py
(first 12 sorted task_ids per level 1/2/3) -- same selection function, same
determinism, so every script's output is directly comparable question-for
-question with ant_gaia/ant_gaia_h's results.

Usage:
    python scripts/run_gaia_level_stratified_baselines.py \\
        --methods direct_gaia sparse_retrieval_gaia
    python scripts/run_gaia_level_stratified_baselines.py \\
        --methods matched_react_gaia s2g_rag_gaia
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import ant.agents.dense_retrieval_gaia  # noqa: E402, F401
import ant.agents.direct_gaia  # noqa: E402, F401
import ant.agents.matched_react_gaia  # noqa: E402, F401
import ant.agents.owl_gaia  # noqa: E402, F401
import ant.agents.retrieval_gaia  # noqa: E402, F401
import ant.external_wrappers.s2g_rag_gaia  # noqa: E402, F401
from ant.benchmarks.gaia import GaiaAdapter  # noqa: E402
from ant.evaluation_suite.registry import get_agent  # noqa: E402
from ant.evaluation_suite.runner import run_suite  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "third_party" / "manifests" / "gaia" / "gaia_text_103_manifest.json"
OUT_DIR = REPO_ROOT / "output" / "runs" / "gaia-level-stratified-r10"
ALL_METHODS = [
    "direct_gaia",
    "sparse_retrieval_gaia",
    "dense_retrieval_gaia",
    "matched_react_gaia",
    "s2g_rag_gaia",
    "owl_gaia",
]


def _select_examples(
    skip_per_level: int, per_level: int, by_id: dict, wanted_ids: set[str]
) -> list:
    by_level: dict[str, list[str]] = {"1": [], "2": [], "3": []}
    for task_id in wanted_ids:
        by_level[by_id[task_id].metadata["level"]].append(task_id)
    for level in by_level:
        by_level[level].sort()
    selected_ids: list[str] = []
    for level in ("1", "2", "3"):
        take = by_level[level][skip_per_level : skip_per_level + per_level]
        print(
            f"  Level {level}: {len(take)}/{len(by_level[level])} selected "
            f"(skip={skip_per_level})"
        )
        selected_ids.extend(take)
    return [by_id[task_id] for task_id in selected_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", required=True, choices=ALL_METHODS)
    parser.add_argument("--per-level", type=int, default=12)
    parser.add_argument("--skip-per-level", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen"
    wanted_ids = set(manifest["task_ids"])

    benchmark = GaiaAdapter(source="live")
    all_examples = benchmark.load_examples()
    if benchmark.resolved_source() != "live":
        raise SystemExit(
            f"resolved_source()={benchmark.resolved_source()!r}, not 'live' -- set HF_TOKEN."
        )
    by_id = {e.task_id: e for e in all_examples}
    missing = wanted_ids - set(by_id)
    assert not missing, f"{len(missing)} frozen task_ids not found live: {sorted(missing)[:10]}"

    print(f"Selecting {args.per_level} question(s) per level (skip={args.skip_per_level}):")
    examples = _select_examples(args.skip_per_level, args.per_level, by_id, wanted_ids)
    print(f"Total: {len(examples)} questions")

    for method_name in args.methods:
        agent = get_agent(method_name)
        out_path = OUT_DIR / method_name / f"{method_name}.jsonl"
        print(f"\n=== {method_name}: {len(examples)} questions ===", flush=True)
        started = time.time()
        results = run_suite(
            benchmark=benchmark,
            agent=agent,
            examples=examples,
            out_path=out_path,
            trajectory_dump_dir=OUT_DIR / method_name / "trajectories",
            resume=True,
        )
        cost = sum(r.usage.estimated_cost_usd for r in results)
        errors = sum(1 for r in results if r.status != "completed")
        n_correct = sum(1 for r in results if r.metric.native_score > 0)
        print(
            f"=== {method_name} done: {len(results)} new rows, {errors} errors, "
            f"{n_correct}/{len(results)} correct this run, ${cost:.4f}, "
            f"{round(time.time() - started, 1)}s ===",
            flush=True,
        )


if __name__ == "__main__":
    main()
