"""GAIA-Text-103 bake-off: all 7 methods from the paper's Table 4 GAIA
column (Direct, Sparse Retrieval, Dense Retrieval, Matched ReAct,
S2G-RAG, ANTMAN, ANTMAN-H).

USES THE FROZEN GAIA-TEXT-103 SUBSET (third_party/manifests/gaia/
gaia_text_103_manifest.json) -- the community-standard text-only GAIA
validation subset (WebThinker/RUC-NLPIR, adopted by MiroThinker/MiroFlow),
NOT the broader 152-task capability-covered manifest.json in the same
directory. See that manifest's own provenance block and
`third_party/manifests/gaia/PROVENANCE.md` for why.

REQUIRES a Hugging Face account that has accepted `gaia-benchmark/GAIA`'s
own gate, with that account's token visible to this process (HF_TOKEN env
var or `huggingface-cli login`) -- without one, `GaiaAdapter(source="auto")`
silently falls back to 10 synthetic fixture examples (see that adapter's
own module docstring), which is fine for a wiring smoke test but is NOT
the real benchmark. This script asserts `resolved_source() == "live"`
after loading unless `--allow-synthetic` is passed, so a real run can
never be silently substituted with fixtures.

Cost note: real paid inference (GPT-4.1 for every method's own answer
call, plus Matched ReAct/S2G-RAG/ANTMAN's own tool-call loops) AND real
Tavily/DuckDuckGo search-API calls for every method except Direct. This
is deliberately not auto-launched at the full 103-question scale --
confirm the printed task count/method list before letting a full run go
unattended, same policy as `run_worker_model_bakeoff.py`.

Usage:
    # wiring smoke test on synthetic fixtures, no gated access needed:
    python scripts/run_gaia_bakeoff.py --allow-synthetic --limit 2 --methods direct_gaia

    # real run (needs HF_TOKEN), one method at a time recommended first:
    python scripts/run_gaia_bakeoff.py --methods direct_gaia --limit 5

    # full 103-question run, all 7 methods, only after reviewing cost:
    python scripts/run_gaia_bakeoff.py
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
OUT_DIR = REPO_ROOT / "output" / "runs" / "gaia-text-103"

ALL_METHODS = [
    "direct_gaia",
    "sparse_retrieval_gaia",
    "dense_retrieval_gaia",
    "matched_react_gaia",
    "s2g_rag_gaia",
    "ant_gaia",  # ANTMAN
    "ant_gaia_h",  # ANTMAN-H, same agent class, worker_model set below
]

ANTMAN_H_WORKER_MODEL = "qwen3-8b"


def _build_agent(method: str, worker_base_url: str | None):
    if method == "ant_gaia_h":
        if not worker_base_url:
            raise SystemExit("ant_gaia_h (ANTMAN-H) requires --worker-base-url")
        from ant.agents.ant_gaia import AntGaiaAgent

        agent = AntGaiaAgent(worker_model=ANTMAN_H_WORKER_MODEL, worker_base_url=worker_base_url)
        agent.name = "ant_gaia_h"  # distinct output file from plain ant_gaia
        return agent
    return get_agent(method)


def _validated_examples(limit: int | None, allow_synthetic: bool):
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen"
    assert manifest["count"] == 103
    wanted_ids = set(manifest["task_ids"])
    assert len(wanted_ids) == 103

    benchmark = GaiaAdapter()
    all_examples = benchmark.load_examples()
    source = benchmark.resolved_source()
    if source != "live" and not allow_synthetic:
        raise SystemExit(
            f"GaiaAdapter resolved to source={source!r}, not 'live' -- no HF_TOKEN visible. "
            "Pass --allow-synthetic for a wiring smoke test on the 10 fixture examples, or "
            "set HF_TOKEN / run `huggingface-cli login` for a real GAIA-Text-103 run."
        )

    if source == "live":
        by_id = {e.task_id: e for e in all_examples}
        missing = wanted_ids - set(by_id)
        assert not missing, f"STOP: {len(missing)} task_ids not found: {sorted(missing)[:10]}"
        examples = [by_id[tid] for tid in sorted(wanted_ids)]
        assert len(examples) == 103
        print("[GAIA-Text-103] VALIDATED: 103/103 frozen task_ids resolved (source=live).")
    else:
        examples = all_examples
        print(
            f"[GAIA] source={source!r} (synthetic fixtures, NOT GAIA-Text-103) -- "
            f"{len(examples)} examples"
        )

    if limit is not None:
        examples = examples[:limit]
    return benchmark, examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="*", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument(
        "--worker-base-url",
        default=None,
        help="required only for ant_gaia_h (ANTMAN-H), e.g. http://host:8000/v1",
    )
    args = parser.parse_args()

    benchmark, examples = _validated_examples(args.limit, args.allow_synthetic)
    print(f"Running {len(examples)} example(s) x {len(args.methods)} method(s): {args.methods}")

    for method in args.methods:
        agent = _build_agent(method, args.worker_base_url)
        out_path = OUT_DIR / method / f"{method}.jsonl"
        print(f"\n=== {agent.name}: {len(examples)} questions ===", flush=True)
        started = time.time()
        results = run_suite(
            benchmark=benchmark,
            agent=agent,
            examples=examples,
            out_path=out_path,
            trajectory_dump_dir=OUT_DIR / method / "trajectories",
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
