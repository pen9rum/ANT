"""Runs ant_gaia (full ANTMAN) and/or ant_gaia_h (ANTMAN-H) on the first N
GAIA-Text-103 questions per level (1/2/3), at a caller-chosen max_rounds.
Per the user's own decision, max_rounds=10 (not ant_gaia.AntGaiaAgent's
DEFAULT_MAX_ROUNDS=6) is THE standard setting for these two methods going
forward -- output lands in output/runs/gaia-level-stratified-r{max_rounds}/,
which is therefore the canonical directory for them, not a side experiment.

Selection is deterministic: sorted task_ids within each level, first
`--per-level` taken -- not a random sample, so re-running with a larger
--per-level is a strict superset of a smaller run (safe to resume/extend;
resume=True means already-completed task_ids are skipped, not re-billed).
Pass --per-level 999 (or any number >= the largest level's count) to select
every one of the 103 questions.

Usage:
    # 12/level (36 total) smoke-scale probe:
    python scripts/run_gaia_level_stratified.py --per-level 12 --max-rounds 10 \\
        --worker-base-url http://host:8000/v1
    # all 103, ANTMAN-H only (extends/resumes any smaller prior run):
    python scripts/run_gaia_level_stratified.py --per-level 999 --max-rounds 10 \\
        --methods ant_gaia_h --worker-base-url http://host:8000/v1
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

from ant.agents.ant_gaia import AntGaiaAgent  # noqa: E402
from ant.benchmarks.gaia import GaiaAdapter  # noqa: E402
from ant.evaluation_suite.runner import run_suite  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "third_party" / "manifests" / "gaia" / "gaia_text_103_manifest.json"


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
    parser.add_argument("--per-level", type=int, default=12)
    parser.add_argument("--skip-per-level", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument(
        "--worker-base-url", default=None, help="required only if --methods includes ant_gaia_h"
    )
    parser.add_argument("--worker-model", default="qwen3-8b")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["ant_gaia", "ant_gaia_h"],
        choices=["ant_gaia", "ant_gaia_h"],
    )
    args = parser.parse_args()
    if "ant_gaia_h" in args.methods and not args.worker_base_url:
        raise SystemExit("--worker-base-url is required when --methods includes ant_gaia_h")

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
    print(f"Total: {len(examples)} questions, max_rounds={args.max_rounds}")

    out_dir = REPO_ROOT / "output" / "runs" / f"gaia-level-stratified-r{args.max_rounds}"

    agents_by_name = {
        "ant_gaia": lambda: AntGaiaAgent(max_rounds=args.max_rounds),
        "ant_gaia_h": lambda: AntGaiaAgent(
            max_rounds=args.max_rounds,
            worker_model=args.worker_model,
            worker_base_url=args.worker_base_url,
        ),
    }
    for method_name in args.methods:
        agent = agents_by_name[method_name]()
        agent.name = method_name
        out_path = out_dir / method_name / f"{method_name}.jsonl"
        print(f"\n=== {method_name}: {len(examples)} questions ===", flush=True)
        started = time.time()
        results = run_suite(
            benchmark=benchmark,
            agent=agent,
            examples=examples,
            out_path=out_path,
            trajectory_dump_dir=out_dir / method_name / "trajectories",
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
