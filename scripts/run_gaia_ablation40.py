"""Runs the adaptive-coordination ablation agents (Full ANTMAN + the 5
ablations) on the frozen 40-question GAIA-Text-103 ablation subset
(third_party/manifests/gaia/gaia_ablation40_manifest.json), at a
caller-chosen max_rounds. Per the user's own decision, max_rounds=10 is
the standard going forward (not AntGaiaAgent's own DEFAULT_MAX_ROUNDS=6),
so agents here are constructed fresh with that value rather than reusing
the ant.agents.ablation_gaia_agents module-level instances (which are
registered at import time with the class default, max_rounds=6).

Designed to run standalone on any machine with this repo + a configured
.env (OPENAI_API_KEY, TAVILY_API_KEY, HF_TOKEN) -- no vLLM server needed,
since none of these six methods use a separate worker model.

Output: output/runs/gaia-ablation40-r{max_rounds}/<method>/<method>.jsonl
        + trajectories/*.json (same shape as every other GAIA runner in
        this repo, so aggregate_gaia_ablation_table.py can read it
        directly once you point --out-dir at it).

Usage:
    # one method at a time (recommended first, to check cost/behavior):
    python scripts/run_gaia_ablation40.py --methods ant_gaia_static

    # several:
    python scripts/run_gaia_ablation40.py --methods ant_gaia_static \\
        ant_gaia_no_need_revision ant_gaia_no_adaptive_rerouting ant_gaia_no_recovery

    # all six (Full ANTMAN + 5 ablations):
    python scripts/run_gaia_ablation40.py
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
from ant.coordinator.ablations import PROFILES  # noqa: E402
from ant.evaluation_suite.runner import run_suite  # noqa: E402

MANIFEST_PATH = (
    REPO_ROOT / "third_party" / "manifests" / "gaia" / "gaia_ablation40_manifest.json"
)
ABLATION_PROFILE_KEYS = [
    "static",
    "graph_free_adaptive",
    "no_need_revision",
    "no_adaptive_rerouting",
    "no_recovery",
]
ALL_METHODS = ["ant_gaia_full"] + [f"ant_gaia_{key}" for key in ABLATION_PROFILE_KEYS]


def _build_agent(method: str, max_rounds: int):
    if method == "ant_gaia_full":
        agent = AntGaiaAgent(max_rounds=max_rounds)
        agent.name = "ant_gaia_full"
        return agent
    profile_key = method[len("ant_gaia_") :]
    if profile_key not in PROFILES or PROFILES[profile_key].is_full:
        raise SystemExit(f"Unknown ablation method {method!r}; expected one of {ALL_METHODS}")
    from ant.agents.ablation_gaia_agents import AblationAntGaiaAgent

    return AblationAntGaiaAgent(profile_key, max_rounds=max_rounds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument("--max-rounds", type=int, default=10)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen"
    assert manifest["count"] == 40
    wanted_ids = set(manifest["task_ids"])
    assert len(wanted_ids) == 40

    benchmark = GaiaAdapter(source="live")
    all_examples = benchmark.load_examples()
    if benchmark.resolved_source() != "live":
        raise SystemExit(
            f"resolved_source()={benchmark.resolved_source()!r}, not 'live' -- set HF_TOKEN."
        )
    by_id = {e.task_id: e for e in all_examples}
    missing = wanted_ids - set(by_id)
    assert not missing, f"{len(missing)} frozen task_ids not found live: {sorted(missing)[:10]}"
    examples = [by_id[tid] for tid in sorted(wanted_ids)]
    print("Validated GAIA-Ablation40: 40/40 frozen task_ids resolved (source=live).")
    print(f"Running {len(examples)} question(s) x {len(args.methods)} method(s), "
          f"max_rounds={args.max_rounds}: {args.methods}")

    out_dir = REPO_ROOT / "output" / "runs" / f"gaia-ablation40-r{args.max_rounds}"

    for method_name in args.methods:
        agent = _build_agent(method_name, args.max_rounds)
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
