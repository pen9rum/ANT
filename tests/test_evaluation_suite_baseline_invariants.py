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


def test_matched_react_per_question_budget_override_via_metadata(
    tmp_path: Path, monkeypatch
) -> None:
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

    # Constructor default is DEFAULT_TOOL_CALL_BUDGET (50), but this
    # question's own metadata (e.g. from a real ANT trace) overrides it --
    # this is the mechanism the SWE-QA-Pro smoke test uses for per-question
    # real-opportunity-matched budgets.
    agent = MatchedReActAgent()
    example = TaskExample(
        benchmark="sweqa_pro",
        task_id="q1",
        question="Where is X?",
        reference="",
        metadata={"matched_react_tool_call_budget": 7},
    )
    agent.run(example, tmp_path)
    assert llm_calls["count"] == 7


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
