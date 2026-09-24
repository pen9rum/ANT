"""OWL GAIA baseline smoke tests. Run with the MAIN venv's python (this
script only imports `ant.*`; `OwlGaiaAgent.run()` itself shells out to
`.venv-owl` for the parts that need camel-ai) -- except test 1, which
must run under `.venv-owl` since it imports camel-ai directly to verify
the production tool-building code.

Usage:
    # Test 1 (fairness: dynamic workers get exactly our 5 tools, nothing
    # else) -- needs camel-ai, so run it with the OTHER venv:
    .venv-owl/bin/python scripts/smoke_test_owl_gaia.py --test fairness

    # Test 2/3 (end-to-end OwlGaiaAgent.run() on synthetic GAIA fixtures)
    # -- needs ant.*, so run it with the MAIN venv:
    .venv/bin/python scripts/smoke_test_owl_gaia.py --test e2e

    # Both, if you have a way to run each half with the right venv:
    .venv-owl/bin/python scripts/smoke_test_owl_gaia.py --test fairness
    .venv/bin/python scripts/smoke_test_owl_gaia.py --test e2e
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _load_env() -> None:
    env_path = REPO_ROOT / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def test_fairness() -> None:
    """Confirms, against the REAL production tool-building code in
    owl_worker_subprocess.py (not a re-declared copy), that: (1) a
    dynamically-created worker's tools exactly match our 5 bridged tools
    when new_worker_agent is set the way build_workforce() sets it, and
    (2) the initial explicitly-registered worker carries the identical
    set. Both assertions run against a LIVE Workforce object -- no
    mocking of camel-ai's own worker-creation code path."""
    _load_env()
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType

    import owl_worker_subprocess as sub

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        model_type=os.environ.get("ANT_MODEL", "openai/gpt-4.1"),
        url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )
    expected_names = sorted(t.get_function_name() for t in sub.build_bridged_tools())
    print(f"Expected bridged tool names: {expected_names}")

    tracker = sub.UsageTracker()
    workforce, initial_worker = sub.build_workforce(model, tracker, has_attachment=False)

    initial_tool_names = sorted(initial_worker.tool_dict.keys())
    assert initial_tool_names == expected_names, (
        f"initial worker tools {initial_tool_names} != expected {expected_names}"
    )
    print(f"PASS: initial worker tools == {initial_tool_names}")

    dynamically_created = asyncio.run(
        workforce._create_new_agent(role="Dynamic Test Worker", sys_msg="probe")
    )
    dynamic_tool_names = sorted(dynamically_created.tool_dict.keys())
    assert dynamic_tool_names == expected_names, (
        f"dynamically-created worker tools {dynamic_tool_names} != expected {expected_names}"
    )
    print(f"PASS: dynamically-created worker tools == {dynamic_tool_names}")
    print("PASS: no OWL-native toolkit (SearchToolkit/CodeExecutionToolkit/ThinkingToolkit) leaked in")


def test_e2e() -> None:
    """Runs OwlGaiaAgent.run() end-to-end on the synthetic GAIA fixtures
    (no gated HF access needed, no real GAIA data used) -- real Tavily +
    real GPT-4.1 (via OpenRouter) + real OWL subprocess + real bridge
    server, exactly the production code path run_gaia_bakeoff.py drives."""
    _load_env()
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import ant.agents.owl_gaia  # noqa: F401 - registers the agent
    from ant.benchmarks.gaia import GaiaAdapter
    from ant.evaluation_suite.registry import get_agent

    benchmark = GaiaAdapter(source="synthetic")
    examples = benchmark.load_examples(limit=1)
    assert examples, "no synthetic GAIA fixtures found"
    example = examples[0]
    print(f"Running task_id={example.task_id!r}: {example.question[:100]!r}")

    environment_root = benchmark.prepare_environment(example)
    agent = get_agent("owl_gaia")
    result = agent.run(example, environment_root)

    print(f"final_answer: {result.final_answer[:300]!r}")
    print(f"usage: {result.usage.model_dump()}")
    print(f"tool_calls logged: {result.usage.tool_calls}")
    print(f"gaia_call_log tools used: {[c['tool'] for c in result.metadata['gaia_call_log']]}")

    assert "FINAL ANSWER:" in result.final_answer, "GAIA answer template missing"
    assert result.usage.llm_calls > 0, "no LLM calls recorded -- usage tracking is broken"
    assert result.usage.tool_calls >= 0

    metric = benchmark.score(example, result)
    print(f"native_score: {metric.native_score}, normalized_score: {metric.normalized_score}")
    print("PASS: end-to-end OwlGaiaAgent run produced a scoreable, correctly-templated answer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", choices=["fairness", "e2e"], required=True)
    args = parser.parse_args()
    if args.test == "fairness":
        test_fairness()
    else:
        test_e2e()


if __name__ == "__main__":
    main()
