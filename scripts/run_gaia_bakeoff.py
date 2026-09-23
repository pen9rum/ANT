"""GAIA bake-off driver for the paper's Table 4 GAIA column (7 methods:
Direct, Sparse Retrieval, Dense Retrieval, Matched ReAct, S2G-RAG, ANTMAN,
ANTMAN-H) PLUS the 5 adaptive-coordination ablations from Table 5/10
(Static ANTMAN, Graph-free Adaptive, w/o Need Revision, w/o Adaptive
Rerouting, w/o Recovery) -- 12 methods total.

TWO DIFFERENT FROZEN TASK SETS, NEVER MIXED:
  * The 7 main methods run on the full GAIA-Text-103 set
    (third_party/manifests/gaia/gaia_text_103_manifest.json) -- the
    community-standard text-only GAIA validation subset (WebThinker/
    RUC-NLPIR, adopted by MiroThinker/MiroFlow), NOT the broader 152-task
    capability-covered manifest.json in the same directory. See that
    manifest's own provenance block and
    `third_party/manifests/gaia/PROVENANCE.md` for why.
  * The 5 ablations run ONLY on the frozen 40-question stratified subset
    (third_party/manifests/gaia/gaia_ablation40_manifest.json, seed=42,
    proportional-by-level) -- ablations are diagnostic (Table 5's own
    "matched execution budgets" framing), not a leaderboard number, so
    they run at a fraction of the cost rather than the full 103. Output
    lands in a SEPARATE directory tree (gaia-ablation40/) so its
    resume-by-task_id bookkeeping never collides with the 103-set's.

REQUIRES a Hugging Face account that has accepted `gaia-benchmark/GAIA`'s
own gate, with that account's token visible to this process (HF_TOKEN env
var or `huggingface-cli login`) -- without one, `GaiaAdapter(source="auto")`
silently falls back to 10 synthetic fixture examples (see that adapter's
own module docstring), which is fine for a wiring smoke test but is NOT
the real benchmark. This script asserts `resolved_source() == "live"`
after loading unless `--allow-synthetic` is passed, so a real run can
never be silently substituted with fixtures.

Cost note: real paid inference (GPT-4.1 for every method's own answer
call, plus Matched ReAct/S2G-RAG/ANTMAN/ablation's own tool-call loops)
AND real Tavily/DuckDuckGo search-API calls for every method except
Direct. This is deliberately not auto-launched at full scale -- confirm
the printed task count/method list before letting a run go unattended,
same policy as `run_worker_model_bakeoff.py`.

Usage:
    # wiring smoke test on synthetic fixtures, no gated access needed:
    python scripts/run_gaia_bakeoff.py --allow-synthetic --limit 2 --methods direct_gaia

    # real run (needs HF_TOKEN), one method at a time recommended first:
    python scripts/run_gaia_bakeoff.py --methods direct_gaia --limit 5

    # one ablation on its frozen 40-question subset:
    python scripts/run_gaia_bakeoff.py --methods ant_gaia_static

    # full 103-question run, all 7 main methods, only after reviewing cost:
    python scripts/run_gaia_bakeoff.py --methods direct_gaia sparse_retrieval_gaia \\
        dense_retrieval_gaia matched_react_gaia s2g_rag_gaia ant_gaia ant_gaia_h

    # all 12 (7 main on the 103-set + 5 ablations on the 40-set):
    python scripts/run_gaia_bakeoff.py --worker-base-url http://host:8000/v1
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

# Importing these registers each agent with ant.evaluation_suite.registry.
import ant.agents.ablation_gaia_agents  # noqa: E402, F401
import ant.agents.ant_gaia  # noqa: E402, F401
import ant.agents.dense_retrieval_gaia  # noqa: E402, F401
import ant.agents.direct_gaia  # noqa: E402, F401
import ant.agents.matched_react_gaia  # noqa: E402, F401
import ant.agents.retrieval_gaia  # noqa: E402, F401
import ant.external_wrappers.s2g_rag_gaia  # noqa: E402, F401
from ant.benchmarks.gaia import GaiaAdapter  # noqa: E402
from ant.evaluation_suite.registry import get_agent  # noqa: E402
from ant.evaluation_suite.runner import run_suite  # noqa: E402

MANIFEST_PATH = REPO_ROOT / "third_party" / "manifests" / "gaia" / "gaia_text_103_manifest.json"
ABLATION_MANIFEST_PATH = (
    REPO_ROOT / "third_party" / "manifests" / "gaia" / "gaia_ablation40_manifest.json"
)
OUT_DIR = REPO_ROOT / "output" / "runs" / "gaia-text-103"
ABLATION_OUT_DIR = REPO_ROOT / "output" / "runs" / "gaia-ablation40"

MAIN_METHODS = [
    "direct_gaia",
    "sparse_retrieval_gaia",
    "dense_retrieval_gaia",
    "matched_react_gaia",
    "s2g_rag_gaia",
    "ant_gaia",  # ANTMAN
    "ant_gaia_h",  # ANTMAN-H, same agent class, worker_model set below
]
ABLATION_METHODS = [
    # "ant_gaia_full" is Full ANTMAN (all 5 mechanisms on) run on the SAME
    # 40-question ablation subset, in the SAME invocation as the 5
    # ablations below -- the reference row an ablation table needs, never
    # substituted with the (differently-scoped) 103-question ant_gaia run.
    "ant_gaia_full",
    "ant_gaia_static",
    "ant_gaia_graph_free_adaptive",
    "ant_gaia_no_need_revision",
    "ant_gaia_no_adaptive_rerouting",
    "ant_gaia_no_recovery",
]
ALL_METHODS = MAIN_METHODS + ABLATION_METHODS

ANTMAN_H_WORKER_MODEL = "qwen3-8b"


def _build_agent(method: str, worker_base_url: str | None):
    if method == "ant_gaia_h":
        if not worker_base_url:
            raise SystemExit("ant_gaia_h (ANTMAN-H) requires --worker-base-url")
        from ant.agents.ant_gaia import AntGaiaAgent

        agent = AntGaiaAgent(worker_model=ANTMAN_H_WORKER_MODEL, worker_base_url=worker_base_url)
        agent.name = "ant_gaia_h"  # distinct output file from plain ant_gaia
        return agent
    if method == "ant_gaia_full":
        from ant.agents.ant_gaia import AntGaiaAgent

        agent = AntGaiaAgent()
        agent.name = "ant_gaia_full"  # distinct output dir from the 103-set's ant_gaia
        return agent
    return get_agent(method)


def _load_live_or_synthetic(allow_synthetic: bool):
    benchmark = GaiaAdapter()
    all_examples = benchmark.load_examples()
    source = benchmark.resolved_source()
    if source != "live" and not allow_synthetic:
        raise SystemExit(
            f"GaiaAdapter resolved to source={source!r}, not 'live' -- no HF_TOKEN visible. "
            "Pass --allow-synthetic for a wiring smoke test on the 10 fixture examples, or "
            "set HF_TOKEN / run `huggingface-cli login` for a real GAIA run."
        )
    return benchmark, all_examples, source


def _select_frozen_subset(
    all_examples, source, manifest_path: Path, expected_count: int, label: str
):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen"
    assert manifest["count"] == expected_count
    wanted_ids = set(manifest["task_ids"])
    assert len(wanted_ids) == expected_count

    if source == "live":
        by_id = {e.task_id: e for e in all_examples}
        missing = wanted_ids - set(by_id)
        assert not missing, (
            f"STOP: {len(missing)} {label} task_ids not found: {sorted(missing)[:10]}"
        )
        examples = [by_id[tid] for tid in sorted(wanted_ids)]
        assert len(examples) == expected_count
        print(
            f"[{label}] VALIDATED: {expected_count}/{expected_count} "
            "frozen task_ids resolved (source=live)."
        )
    else:
        examples = all_examples
        print(
            f"[{label}] source={source!r} (synthetic fixtures, NOT {label}) -- "
            f"{len(examples)} examples"
        )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="*", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument(
        "--limit", type=int, default=None, help="slices whichever example set(s) are requested"
    )
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument(
        "--worker-base-url",
        default=None,
        help="required only for ant_gaia_h (ANTMAN-H), e.g. http://host:8000/v1",
    )
    args = parser.parse_args()

    main_methods = [m for m in args.methods if m in MAIN_METHODS]
    ablation_methods = [m for m in args.methods if m in ABLATION_METHODS]
    print(f"Requested {len(args.methods)} method(s): {args.methods}")

    benchmark, all_examples, source = _load_live_or_synthetic(args.allow_synthetic)

    if main_methods:
        main_examples = _select_frozen_subset(
            all_examples, source, MANIFEST_PATH, 103, "GAIA-Text-103"
        )
        if args.limit is not None:
            main_examples = main_examples[: args.limit]
        _run_methods(main_methods, benchmark, main_examples, OUT_DIR, args.worker_base_url)

    if ablation_methods:
        ablation_examples = _select_frozen_subset(
            all_examples, source, ABLATION_MANIFEST_PATH, 40, "GAIA-Ablation40"
        )
        if args.limit is not None:
            ablation_examples = ablation_examples[: args.limit]
        _run_methods(
            ablation_methods, benchmark, ablation_examples, ABLATION_OUT_DIR, args.worker_base_url
        )


def _run_methods(methods, benchmark, examples, out_dir: Path, worker_base_url: str | None) -> None:
    for method in methods:
        agent = _build_agent(method, worker_base_url)
        out_path = out_dir / method / f"{method}.jsonl"
        print(f"\n=== {agent.name}: {len(examples)} questions ===", flush=True)
        started = time.time()
        results = run_suite(
            benchmark=benchmark,
            agent=agent,
            examples=examples,
            out_path=out_path,
            trajectory_dump_dir=out_dir / method / "trajectories",
            resume=True,
        )
        cost = sum(r.usage.estimated_cost_usd for r in results)
        errors = sum(1 for r in results if r.status != "completed")
        n_correct = sum(1 for r in results if r.metric.native_score > 0)
        print(
            f"=== {agent.name} done: {len(results)} new rows, {errors} errors, "
            f"{n_correct}/{len(results)} correct this run, ${cost:.4f}, "
            f"{round(time.time() - started, 1)}s ===",
            flush=True,
        )


if __name__ == "__main__":
    main()
