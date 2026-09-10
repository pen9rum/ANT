"""RepoGraph baseline-fidelity tests -- Matched ReAct + RepoGraph audit.

Two things this file proves, as required by that audit:
  1. The `search_repograph` tool is genuinely visible in the model-facing
     schema (the exact system prompt string sent to the LLM) whenever a
     MatchedReActAgent is constructed with RepoGraphTool wired in as an
     extra_tools entry -- not just present in Python-side bookkeeping.
  2. A manually constructed `{"tool": "search_repograph", ...}` decision
     dispatches through the REAL RepoGraphTool (not a mock) and returns
     real graph evidence -- proving the whole extra_tools -> dispatch ->
     RepoGraph wiring works end to end, not just each piece in isolation.

The LLM call itself is still mocked (per this suite's own no-real-API-calls
test policy) -- only the RepoGraph side (a local subprocess to a pinned
external checkout, no network/API call) is real. Tests that need the
checkout/venv to be present skip cleanly if it is not (e.g. a clean
checkout that hasn't stood up third_party/checkouts/repograph yet).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ant.agents.matched_react import MatchedReActAgent
from ant.benchmarks.base import TaskExample
from ant.domain import TokenUsage
from ant.external_wrappers.repograph import RepoGraphNotCheckedOut, RepoGraphTool

QIBO_REPO = Path("repos/qibo")
# Confirmed live (RepoGraph fidelity audit): "main" resolves to a real node
# (examples/vqregressor/main.py) in qibo's own constructed graph.
KNOWN_GOOD_SYMBOL = "main"


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


def test_repograph_tool_description_is_visible_in_the_model_facing_system_prompt(
    tmp_path: Path, monkeypatch
) -> None:
    """The exact string sent to the model must contain both the tool name
    and the faithfully-ported usage instruction -- not just an internal
    Python-side available_tools tuple. Fully mocked LLM call."""
    from ant.agents import matched_react as react_module

    captured_prompts: list[str] = []

    def fake_responses_json(self, prompt, max_output_tokens=512):
        captured_prompts.append(prompt)
        return _FakeResponse(json.dumps({"thought": "done", "finish": "answer"}))

    monkeypatch.setattr(react_module.CountingOpenAIProvider, "responses_json", fake_responses_json)
    monkeypatch.setattr(
        react_module.CountingOpenAIProvider, "drain_usage", lambda self: TokenUsage()
    )

    agent = MatchedReActAgent(
        extra_tools={"search_repograph": lambda query, example: []},
        extra_tool_descriptions={"search_repograph": RepoGraphTool.TOOL_DESCRIPTION},
    )
    example = TaskExample(benchmark="sweqa_pro", task_id="q1", question="Where is X?", reference="")
    agent.run(example, tmp_path)

    assert captured_prompts, "the LLM must have been called at least once"
    first_prompt = captured_prompts[0]
    assert "search_repograph" in first_prompt
    assert "Before you finish and answer, always look up related context" in first_prompt
    assert "key functions or classes as search terms" in first_prompt


def test_repograph_tool_description_matches_the_faithfully_ported_official_instruction() -> None:
    """Regression guard: the ported text must still contain the two
    method-intrinsic instructions found in the official SWE-agent
    integration (instance_template tips #6/#7), not just a bare tool
    docstring -- see RepoGraphTool's own FIDELITY AUDIT docstring."""
    desc = RepoGraphTool.TOOL_DESCRIPTION
    assert "always look up related context using this tool" in desc
    assert "most related and key functions or classes as search terms" in desc


@pytest.mark.skipif(
    not (Path("third_party/checkouts/repograph/repograph/_ant_query.py").exists()),
    reason="RepoGraph checkout not present in this environment",
)
@pytest.mark.skipif(not QIBO_REPO.exists(), reason="qibo repo checkout not present")
def test_manually_constructed_search_repograph_decision_dispatches_and_returns_real_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    """End-to-end, real RepoGraphTool (no mock): a manually scripted
    decision sequence forces the agent to call search_repograph on its
    first step, then finish -- proving the tool is reachable through the
    real MatchedReActAgent decision loop and returns real, non-empty
    graph evidence from a real repository."""
    from ant.agents import matched_react as react_module

    tool = RepoGraphTool(timeout_seconds=150)
    try:
        tool._require_checkout()
    except RepoGraphNotCheckedOut:
        pytest.skip("RepoGraph checkout/venv not fully set up in this environment")

    repo_dir = QIBO_REPO.resolve()
    extra_tool = tool.as_extra_tool(repo_dir)

    step = {"n": 0}

    def fake_responses_json(self, prompt, max_output_tokens=512):
        step["n"] += 1
        if step["n"] == 1:
            decision = {
                "thought": "look up the graph",
                "tool": "search_repograph",
                "query": KNOWN_GOOD_SYMBOL,
            }
        else:
            decision = {"thought": "done", "finish": "answer"}
        return _FakeResponse(json.dumps(decision))

    monkeypatch.setattr(react_module.CountingOpenAIProvider, "responses_json", fake_responses_json)
    monkeypatch.setattr(
        react_module.CountingOpenAIProvider, "drain_usage", lambda self: TokenUsage()
    )

    agent = MatchedReActAgent(
        extra_tools={"search_repograph": extra_tool},
        extra_tool_descriptions={"search_repograph": RepoGraphTool.TOOL_DESCRIPTION},
    )
    example = TaskExample(
        benchmark="sweqa_pro",
        task_id="q1",
        question=f"Where is {KNOWN_GOOD_SYMBOL} used?",
        reference="",
    )
    result = agent.run(example, repo_dir)

    assert result.trajectory[0]["tool"] == "search_repograph"
    real_evidence = result.trajectory[0]["results"]
    assert real_evidence, "RepoGraph must return real, non-empty evidence for a known-good symbol"
    # RepoGraphTool.query() always embeds the queried symbol in its own
    # `reason` field (f"RepoGraph {relation} of {symbol!r} ...") -- a
    # reliable signal this evidence genuinely came from a real graph
    # lookup, not a stub.
    assert all(KNOWN_GOOD_SYMBOL in item["reason"] for item in real_evidence)
    assert all(item["path"] for item in real_evidence)
