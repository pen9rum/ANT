"""Best-of-2 rescue pass for ant_gaia / ant_gaia_h: re-runs ONLY the
task_ids each method currently scored 0 on (from the full 103-question
gaia-level-stratified-r10 run), then writes a merged "best of the two
attempts" file per task_id -- keeping the retry's row when it scores
higher than the original, keeping the original otherwise. Exploits the
same run-to-run stochasticity already observed live in this project (the
same task_id, same method, same max_rounds has produced different scores
across separate runs) as a cheap way to recover some of it, rather than
claiming either single run is the "true" score.

This does NOT overwrite the original per-question files (resume=True
would incorrectly skip already-"completed" rows if it did) -- retries
land in a separate `-retry` output directory, and the merge is a THIRD,
clearly-labeled file, so the original single-attempt numbers stay
recoverable for comparison.

Usage:
    python scripts/run_gaia_best_of_2_retry.py --methods ant_gaia ant_gaia_h \\
        --worker-base-url http://host:8000/v1
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

OUT_DIR = REPO_ROOT / "output" / "runs" / "gaia-level-stratified-r10"
RETRY_DIR = REPO_ROOT / "output" / "runs" / "gaia-level-stratified-r10-retry"
MAX_ROUNDS = 10


def _load_rows(path: Path) -> dict[str, dict]:
    return {
        json.loads(line)["task_id"]: json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _build_agent(method: str, worker_base_url: str):
    if method == "ant_gaia":
        agent = AntGaiaAgent(max_rounds=MAX_ROUNDS)
    elif method == "ant_gaia_h":
        agent = AntGaiaAgent(
            max_rounds=MAX_ROUNDS, worker_model="qwen3-8b", worker_base_url=worker_base_url
        )
    else:
        raise SystemExit(f"unsupported method {method!r}")
    agent.name = method
    return agent


def run_one_method(method: str, worker_base_url: str) -> None:
    original_path = OUT_DIR / method / f"{method}.jsonl"
    assert original_path.exists(), f"{original_path} does not exist -- run the main pass first"
    original_rows = _load_rows(original_path)
    assert len(original_rows) == 103, f"{method}: expected 103 rows, found {len(original_rows)}"

    wrong_ids = sorted(
        tid for tid, row in original_rows.items() if row["metric"]["native_score"] <= 0
    )
    print(f"{method}: {len(wrong_ids)}/103 currently wrong, retrying those.")
    if not wrong_ids:
        print(f"{method}: nothing to retry, all 103 already correct.")
        return

    benchmark = GaiaAdapter(source="live")
    all_examples = benchmark.load_examples()
    assert benchmark.resolved_source() == "live"
    by_id = {e.task_id: e for e in all_examples}
    retry_examples = [by_id[tid] for tid in wrong_ids]

    agent = _build_agent(method, worker_base_url)
    retry_out_path = RETRY_DIR / method / f"{method}.jsonl"
    started = time.time()
    results = run_suite(
        benchmark=benchmark,
        agent=agent,
        examples=retry_examples,
        out_path=retry_out_path,
        trajectory_dump_dir=RETRY_DIR / method / "trajectories",
        resume=True,
    )
    cost = sum(r.usage.estimated_cost_usd for r in results)
    n_recovered = sum(1 for r in results if r.metric.native_score > 0)
    print(
        f"{method}: retry done, {n_recovered}/{len(results)} newly correct this pass, "
        f"${cost:.4f}, {round(time.time() - started, 1)}s"
    )

    # Merge: best of (original, retry) per task_id.
    retry_rows = _load_rows(retry_out_path)
    merged = []
    for tid, orig_row in original_rows.items():
        retry_row = retry_rows.get(tid)
        retry_better = (
            retry_row is not None
            and retry_row["metric"]["native_score"] > orig_row["metric"]["native_score"]
        )
        if retry_better:
            merged.append({**retry_row, "_source": "retry"})
        else:
            merged.append({**orig_row, "_source": "original"})

    merged_path = OUT_DIR / method / f"{method}.best_of_2.jsonl"
    with merged_path.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row) + "\n")

    n_correct_best = sum(1 for r in merged if r["metric"]["native_score"] > 0)
    n_flipped = sum(1 for r in merged if r["_source"] == "retry")
    print(
        f"{method}: best-of-2 merged -> {merged_path.relative_to(REPO_ROOT)} "
        f"({n_correct_best}/103 correct, {n_flipped} question(s) improved by the retry)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["ant_gaia", "ant_gaia_h"])
    parser.add_argument("--worker-base-url", required=True)
    args = parser.parse_args()
    for method in args.methods:
        print(f"\n=== {method} ===")
        run_one_method(method, args.worker_base_url)


if __name__ == "__main__":
    main()
