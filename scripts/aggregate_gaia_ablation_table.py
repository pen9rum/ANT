"""Aggregates the GAIA-Ablation40 6-row table (Full ANTMAN + the 5
adaptive-coordination ablations) from `run_gaia_bakeoff.py`'s output --
one row per profile, columns aligned to slot into the paper's Table 5
alongside the 512K-controlled and RepoProbe-Python/SWE-QA-Pro ablation
rows.

READ THIS BEFORE TRUSTING THE "active coordination" NUMBER FOR GAIA:
`ant_gaia.py`'s own docstring explains why this substrate uses exactly
ONE WorkerCard/territory ("worker-gaia") -- GAIA's search/open_url/
inspect_file/inspect_table/run_python surface has no natural territory
partition the way a multi-file repo or a multi-page site does. The
paper's own "active coordination" metric (Appendix A.2's |A(q)|) counts
DISTINCT territory-backed workers activated -- with M=1 available, that
count is trivially <= 1 for every GAIA row, regardless of profile. This
is a structural fact about how GAIA was wired, not a computation bug, and
it carries NO ablation signal here the way it does for RepoProbe/
SWE-QA-Pro/512K (which have many territories). This script reports it
anyway (for schema alignment with the other two benchmarks' Table 5 rows)
AND reports `coordination_rounds` (mean rounds actually used before
termination) as the metric that DOES vary meaningfully for a
single-territory substrate -- read both, but treat rounds as the
informative one for GAIA specifically.

Usage:
    python scripts/aggregate_gaia_ablation_table.py
    python scripts/aggregate_gaia_ablation_table.py --out-dir output/runs/gaia-ablation40
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PROFILE_ORDER = [
    "ant_gaia_full",
    "ant_gaia_static",
    "ant_gaia_graph_free_adaptive",
    "ant_gaia_no_need_revision",
    "ant_gaia_no_adaptive_rerouting",
    "ant_gaia_no_recovery",
]
PROFILE_LABELS = {
    "ant_gaia_full": "Full ANTMAN",
    "ant_gaia_static": "Static ANTMAN",
    "ant_gaia_graph_free_adaptive": "Graph-free Adaptive",
    "ant_gaia_no_need_revision": "w/o Need Revision",
    "ant_gaia_no_adaptive_rerouting": "w/o Adaptive Rerouting",
    "ant_gaia_no_recovery": "w/o Recovery",
}


def _distinct_workers(trajectory: list[dict]) -> int:
    workers: set[str] = set()
    for round_ in trajectory:
        for node in round_.get("node_executions", []):
            workers.update(node.get("worker_ids") or [])
    return len(workers)


def _has_global_fallback(trajectory: list[dict]) -> bool:
    for round_ in trajectory:
        for node in round_.get("node_executions", []):
            if node.get("special_tactic") == "global_fallback":
                return True
    return False


def _load_profile(out_dir: Path, profile: str) -> dict:
    results_path = out_dir / profile / f"{profile}.jsonl"
    traj_dir = out_dir / profile / "trajectories"
    if not results_path.exists():
        return {"profile": profile, "n_completed": 0, "n_error": 0, "missing": True}

    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    completed = [r for r in rows if r["status"] == "completed"]
    errored = [r for r in rows if r["status"] != "completed"]

    native_scores = [r["metric"]["native_score"] for r in completed]
    normalized_scores = [r["metric"]["normalized_score"] for r in completed]
    costs = [r["usage"]["estimated_cost_usd"] for r in completed]
    llm_calls = [r["usage"]["llm_calls"] for r in completed]

    distinct_worker_counts = []
    fallback_flags = []
    for r in completed:
        traj_path = traj_dir / f"{profile}-{r['task_id']}.json"
        if not traj_path.exists():
            continue
        traj = json.loads(traj_path.read_text(encoding="utf-8")).get("trajectory", [])
        distinct_worker_counts.append(_distinct_workers(traj))
        fallback_flags.append(_has_global_fallback(traj))
        rounds_used = len(traj)
        r["_rounds_used"] = rounds_used
    rounds_used_list = [r["_rounds_used"] for r in completed if "_rounds_used" in r]

    def _mean(xs: list[float]) -> float | None:
        return round(statistics.mean(xs), 4) if xs else None

    return {
        "profile": profile,
        "label": PROFILE_LABELS.get(profile, profile),
        "n_completed": len(completed),
        "n_error": len(errored),
        "error_task_ids": [r["task_id"] for r in errored],
        "mean_native_score": _mean(native_scores),
        "mean_normalized_score": _mean(normalized_scores),
        "mean_cost_usd": _mean(costs),
        "mean_llm_calls": _mean(llm_calls),
        "mean_distinct_workers": _mean(distinct_worker_counts),
        "mean_coordination_rounds": _mean(rounds_used_list),
        "global_fallback_rate_pct": (
            round(100 * sum(fallback_flags) / len(fallback_flags), 1) if fallback_flags else None
        ),
        "missing": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "output" / "runs" / "gaia-ablation40"
    )
    args = parser.parse_args()

    rows = [_load_profile(args.out_dir, profile) for profile in PROFILE_ORDER]

    header = (
        f"{'Profile':<24} {'n_ok':>5} {'n_err':>5} {'native':>8} {'norm':>8} "
        f"{'$/q':>8} {'calls/q':>8} {'workers':>8} {'rounds':>7} {'fallback%':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        if row.get("missing"):
            print(f"{row['profile']:<24} (no output yet)")
            continue
        print(
            f"{row['label']:<24} {row['n_completed']:>5} {row['n_error']:>5} "
            f"{row['mean_native_score']:>8} {row['mean_normalized_score']:>8} "
            f"{row['mean_cost_usd']:>8} {row['mean_llm_calls']:>8} "
            f"{row['mean_distinct_workers']:>8} {row['mean_coordination_rounds']:>7} "
            f"{row['global_fallback_rate_pct']:>10}"
        )
        if row["n_error"]:
            print(f"    ERROR task_ids (not counted in the above): {row['error_task_ids']}")

    print()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
