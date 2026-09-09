"""Phase K baseline-invariant tests -- every provider/tool call is mocked,
no real API calls anywhere in this file. Each test asserts the specific
structural invariant Phase F requires of that tier, not just "it runs".
"""

from __future__ import annotations

import json
from pathlib import Path

from ant.agents.ant_adapter import AntAgent
from ant.agents.direct import DirectAgent
from ant.agents.matched_react import MatchedReActAgent
from ant.agents.retrieval import RetrievalAgent
from ant.benchmarks.base import TaskExample
from ant.domain import TokenUsage


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _NeverCalledSearchTool:
    """Standing in for LocalSearchTool in the Direct-tier test -- any
    method call at all fails the test immediately, proving Direct never
    touches the repository."""

    def __getattr__(self, name: str):
        raise AssertionError(f"Direct tier must never call any repo tool, tried: {name}")


def test_direct_agent_never_accesses_the_repository(tmp_path: Path, monkeypatch) -> None:
    from ant.agents import direct as direct_module

    monkeypatch.setattr(
        direct_module.OpenAIProvider,
        "responses_text",
        lambda self, prompt, max_output_tokens=512: _FakeResponse("a closed-book answer"),
    )
    monkeypatch.setattr(
        direct_module.OpenAIProvider, "drain_usage", lambda self: TokenUsage()
    )
    # A repo root that would crash immediately if anything tried to read it.
    poisoned_root = tmp_path / "does-not-exist-and-must-never-be-touched"

    agent = DirectAgent()
    example = TaskExample(benchmark="sweqa_pro", task_id="q1", question="Where is X?", reference="")
    result = agent.run(example, poisoned_root)

    assert result.final_answer == "a closed-book answer"
    assert result.usage.tool_calls == 0
    assert not poisoned_root.exists()  # never created, never read


def test_retrieval_agent_stops_within_its_round_budget_even_if_llm_never_says_enough(
    tmp_path: Path, monkeypatch
) -> None:
    from ant.agents import retrieval as retrieval_module

    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")

    search_calls = {"count": 0}
    monkeypatch.setattr(
        retrieval_module.LocalSearchTool,
        "search",
        lambda self, query, files, limit=8: (
            search_calls.__setitem__("count", search_calls["count"] + 1) or []
        ),
    )
    # The decision LLM NEVER says "enough" and NEVER stops on its own --
    # only the hard round budget (_TIER2_MAX_ROUNDS) may end the loop.
    monkeypatch.setattr(
        retrieval_module.OpenAIProvider,
        "responses_json",
        lambda self, prompt, max_output_tokens=256: _FakeResponse(
            json.dumps({"enough": False, "next_query": "keep searching"})
        ),
    )
    monkeypatch.setattr(
        retrieval_module.OpenAIProvider,
        "synthesize",
        lambda self, question, evidence: "best-effort answer",
    )
    monkeypatch.setattr(
        retrieval_module.OpenAIProvider, "drain_usage", lambda self: TokenUsage()
    )

    agent = RetrievalAgent()
    example = TaskExample(benchmark="sweqa_pro", task_id="q1", question="Where is X?", reference="")
    result = agent.run(example, tmp_path)

    assert search_calls["count"] == retrieval_module._TIER2_MAX_ROUNDS
    assert result.usage.tool_calls == retrieval_module._TIER2_MAX_ROUNDS
    assert result.final_answer == "best-effort answer"


def test_matched_react_respects_its_tool_call_budget_and_uses_no_need_graph_concepts(
    tmp_path: Path, monkeypatch
) -> None:
    from ant.agents import matched_react as react_module

    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")

    llm_calls = {"count": 0}

    def fake_responses_json(self, prompt, max_output_tokens=512):
        llm_calls["count"] += 1
        # Never declares "finish" -- only the tool_call_budget may end this.
        decision = {"thought": "keep going", "tool": "search", "query": "foo"}
        return _FakeResponse(json.dumps(decision))

    monkeypatch.setattr(react_module.OpenAIProvider, "responses_json", fake_responses_json)
    monkeypatch.setattr(react_module.LocalSearchTool, "search", lambda self, q, f, limit=6: [])
    monkeypatch.setattr(
        react_module.OpenAIProvider, "synthesize", lambda self, question, evidence: "forced answer"
    )
    monkeypatch.setattr(react_module.OpenAIProvider, "drain_usage", lambda self: TokenUsage())

    budget = 5
    agent = MatchedReActAgent(tool_call_budget=budget)
    example = TaskExample(benchmark="sweqa_pro", task_id="q1", question="Where is X?", reference="")
    result = agent.run(example, tmp_path)

    assert llm_calls["count"] == budget
    assert result.termination_reason == "budget_exhausted"
    assert result.final_answer == "forced answer"
    assert result.metadata["tool_call_budget"] == budget
    # Structural negative-space check: nothing in this module IMPORTS any
    # Need-Graph/WorkerCard/coordinator concept -- checked against the
    # actual import statements only (not comments/docstrings, which
    # legitimately name these concepts when explaining what's excluded).
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(react_module))
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported_names.update(alias.name for alias in node.names)
            if node.module:
                imported_names.add(node.module)
        elif isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
    for forbidden in ("WorkerCard", "NeedGraph", "LocalCoordinator", "ant.coordinator"):
        assert not any(forbidden in name for name in imported_names), (
            f"matched_react.py must never import anything referencing {forbidden!r}, "
            f"found in imports: {imported_names}"
        )


def test_matched_react_budget_is_fixed_at_construction_and_ignores_example_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    """Primary protocol: one budget, pre-specified at construction time,
    applied uniformly to every example -- never a per-example override read
    from the example being run. An earlier version of this agent read
    `example.metadata["matched_react_tool_call_budget"]` as a retrospective
    per-question override (each question's budget set to that question's
    own already-measured real ANT tool-call count); that is a fairness
    defect (it conditions the baseline's compute on the very ANT run it is
    compared against) and has been removed. This test locks in that a
    metadata field of that name is now inert."""
    from ant.agents import matched_react as react_module

    llm_calls = {"count": 0}

    def fake_responses_json(self, prompt, max_output_tokens=512):
        llm_calls["count"] += 1
        return _FakeResponse(json.dumps({"thought": "x", "tool": "search", "query": "foo"}))

    monkeypatch.setattr(react_module.OpenAIProvider, "responses_json", fake_responses_json)
    monkeypatch.setattr(react_module.LocalSearchTool, "search", lambda self, q, f, limit=6: [])
    monkeypatch.setattr(
        react_module.OpenAIProvider, "synthesize", lambda self, question, evidence: "answer"
    )
    monkeypatch.setattr(react_module.OpenAIProvider, "drain_usage", lambda self: TokenUsage())

    agent = MatchedReActAgent(tool_call_budget=5)
    example = TaskExample(
        benchmark="sweqa_pro",
        task_id="q1",
        question="Where is X?",
        reference="",
        # An old-shaped metadata override, if it were still honored, would
        # push the budget to 7 -- it must be ignored, leaving the
        # constructor's own budget (5) in force.
        metadata={"matched_react_tool_call_budget": 7},
    )
    agent.run(example, tmp_path)
    assert llm_calls["count"] == 5


def test_matched_react_defaults_to_ant_worker_parity_tool_limits(
    tmp_path: Path, monkeypatch
) -> None:
    """An earlier version of this agent called every tool with a uniform
    limit=6, rationalized after the fact as "generous compensation" for
    being a single agent -- a justification that was never actually
    specified anywhere before being written down. The fairness-closure
    audit's own principle is: expose comparable PRIMITIVE capabilities:
    ANT's advantage should come only from its coordination mechanisms, not
    a bigger single-call limit. This locks in that the DEFAULT
    construction now calls each tool with exactly ANT_PARITY_TOOL_LIMITS'
    own per-tool value (copied directly from AutonomousWorker's real,
    operative per-call limits), not the old uniform 6."""
    from ant.agents import matched_react as react_module

    captured_limits: dict[str, int] = {}

    def fake_responses_json(self, prompt, max_output_tokens=512):
        if "search" not in captured_limits:
            return _FakeResponse(json.dumps({"thought": "x", "tool": "search", "query": "foo"}))
        if "subclasses" not in captured_limits:
            return _FakeResponse(
                json.dumps({"thought": "x", "tool": "subclasses", "query": "Foo"})
            )
        if "navigate" not in captured_limits:
            return _FakeResponse(json.dumps({"thought": "x", "tool": "navigate", "query": "Foo"}))
        return _FakeResponse(json.dumps({"thought": "done", "finish": "answer"}))

    def fake_search(self, q, f, limit=8):
        captured_limits["search"] = limit
        return []

    def fake_subclasses(self, symbol, f, limit=8):
        captured_limits["subclasses"] = limit
        return []

    def fake_resolve_symbol(self, symbol, f, limit=6, need=""):
        captured_limits["navigate"] = limit
        return []

    def fake_navigate(self, symbol, f, limit=6):
        return []

    monkeypatch.setattr(react_module.OpenAIProvider, "responses_json", fake_responses_json)
    monkeypatch.setattr(react_module.LocalSearchTool, "search", fake_search)
    monkeypatch.setattr(react_module.LocalSearchTool, "subclasses", fake_subclasses)
    monkeypatch.setattr(react_module.LocalSearchTool, "resolve_symbol", fake_resolve_symbol)
    monkeypatch.setattr(react_module.LocalSearchTool, "navigate", fake_navigate)
    monkeypatch.setattr(
        react_module.OpenAIProvider, "synthesize", lambda self, question, evidence: "answer"
    )
    monkeypatch.setattr(react_module.OpenAIProvider, "drain_usage", lambda self: TokenUsage())

    agent = MatchedReActAgent()
    assert agent.tool_result_limits == react_module.ANT_PARITY_TOOL_LIMITS
    example = TaskExample(benchmark="sweqa_pro", task_id="q1", question="Where is X?", reference="")
    agent.run(example, tmp_path)

    assert captured_limits["search"] == 4
    assert captured_limits["subclasses"] == 4
    assert captured_limits["navigate"] == 2


def test_matched_react_sensitivity_tool_limits_are_uniform_and_not_the_default(
    tmp_path: Path,
) -> None:
    """GENEROUS_SENSITIVITY_TOOL_LIMITS exists only as an explicit, opt-in
    alternate configuration for a later sensitivity experiment -- never
    selected unless a caller passes it in by name."""
    from ant.agents import matched_react as react_module

    assert set(react_module.GENEROUS_SENSITIVITY_TOOL_LIMITS.values()) == {6}
    assert react_module.GENEROUS_SENSITIVITY_TOOL_LIMITS != react_module.ANT_PARITY_TOOL_LIMITS
    default_agent = MatchedReActAgent()
    assert default_agent.tool_result_limits != react_module.GENEROUS_SENSITIVITY_TOOL_LIMITS
    sensitivity_agent = MatchedReActAgent(
        tool_result_limits=react_module.GENEROUS_SENSITIVITY_TOOL_LIMITS
    )
    assert sensitivity_agent.tool_result_limits == react_module.GENEROUS_SENSITIVITY_TOOL_LIMITS


def test_ant_agent_never_passes_cross_task_memory(tmp_path: Path, monkeypatch) -> None:
    from ant.agents import ant_adapter as ant_adapter_module
    from ant.domain import EvidenceState

    captured_kwargs: dict = {}
    real_init = ant_adapter_module.LocalCoordinator.__init__

    def spy_init(self, repo_root, workers, **kwargs):
        captured_kwargs.update(kwargs)
        real_init(self, repo_root, workers, **kwargs)

    monkeypatch.setattr(ant_adapter_module.LocalCoordinator, "__init__", spy_init)
    monkeypatch.setattr(
        ant_adapter_module.LocalCoordinator,
        "ask",
        lambda self, question, max_rounds=6: EvidenceState(question=question, answer="stub answer"),
    )
    (tmp_path / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")

    agent = AntAgent(index_root=tmp_path / "index")
    example = TaskExample(benchmark="sweqa_pro", task_id="q1", question="Where is X?", reference="")
    result = agent.run(example, tmp_path)

    assert result.final_answer == "stub answer"
    # The two explicit no-cross-task-memory params either aren't passed at
    # all (falling through to LocalCoordinator's own [] default) or are
    # passed as empty -- never populated.
    assert not captured_kwargs.get("memory_routes")
    assert not captured_kwargs.get("cross_repo_experience")
