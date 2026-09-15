"""Tests for ant.agents.matched_react_web.MatchedReActWebAgent. The LLM
call site is scripted/mocked (no real API call); page fetching uses a
real PageCache over a static, in-memory mock fetcher (no real network
call) -- zero paid inference, per the governing spec.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ant.agents import matched_react_web as react_module
from ant.agents.matched_react_web import DEFAULT_MAX_STEPS, MatchedReActWebAgent
from ant.benchmarks.base import TaskExample
from ant.domain import TokenUsage
from ant.evaluation_suite.web_fetch import _RawFetch
from ant.providers.openai_provider import ResponseResult

SECRET_ANSWER = "SECRET_GOLD_ANSWER_MUST_NEVER_LEAK"
SECRET_SOURCE = "SECRET_SOURCE_WEBSITE_MUST_NEVER_LEAK"


class _ScriptedProvider:
    """Returns pre-scripted responses in order -- `json_responses` feeds
    every `responses_json()` call in sequence, `text_responses` feeds
    `responses_text()` (used only by the forced-finish path)."""

    def __init__(self, json_responses: list[str], text_responses: list[str] | None = None) -> None:
        self._json_responses = list(json_responses)
        self._text_responses = list(text_responses or ["forced final answer"])
        self.prompts_sent: list[str] = []
        self._calls = 0

    def responses_json(self, prompt: str, max_output_tokens: int = 512) -> ResponseResult:
        self.prompts_sent.append(prompt)
        self._calls += 1
        text = self._json_responses.pop(0)
        return ResponseResult(
            text=text, usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15), raw={}
        )

    def responses_text(self, prompt: str, max_output_tokens: int = 512) -> ResponseResult:
        self.prompts_sent.append(prompt)
        self._calls += 1
        text = self._text_responses.pop(0) if self._text_responses else "forced final answer"
        return ResponseResult(
            text=text, usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15), raw={}
        )

    def drain_call_count(self) -> int:
        count = self._calls
        self._calls = 0
        return count

    def drain_usage(self) -> TokenUsage:
        return TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2, estimated_cost_usd=0.001)

    def drain_retry_log(self) -> list[dict]:
        return []


def _mock_fetcher(pages: dict[str, str]):
    def _fetch(url: str) -> _RawFetch:
        if url in pages:
            return _RawFetch(http_status=200, redirect_target=None, html=pages[url], error=None)
        return _RawFetch(
            http_status=404, redirect_target=None, html="", error="HTTPError: 404 Not Found"
        )

    return _fetch


def _example(root_url: str = "http://conf.example.com/") -> TaskExample:
    return TaskExample(
        benchmark="webwalkerqa",
        task_id="webwalkerqa-abc123",
        question="Who is the keynote speaker?",
        reference=SECRET_ANSWER,
        metadata={
            "root_url": root_url,
            "_audit_only": {"source_websites": [SECRET_SOURCE], "golden_path": [SECRET_SOURCE]},
        },
    )


def _run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pages: dict[str, str], provider, agent=None
):
    monkeypatch.setattr(
        react_module, "urllib_fetcher", lambda timeout_seconds=10.0: _mock_fetcher(pages)
    )
    monkeypatch.setattr(react_module, "CountingOpenAIProvider", lambda model: provider)
    agent = agent or MatchedReActWebAgent()
    result = agent.run(_example(), tmp_path)
    return result, provider


# --- basic finish / navigate behavior ---


def test_agent_finishes_immediately_from_root_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://conf.example.com/": "<p>Keynote: Dr. Jane Smith</p><a href='/speakers'>Speakers</a>"
    }
    provider = _ScriptedProvider(
        json_responses=[
            json.dumps(
                {"thought": "I see the answer", "action": "finish", "answer": "Dr. Jane Smith"}
            )
        ]
    )
    result, provider = _run(monkeypatch, tmp_path, pages, provider)

    assert result.final_answer == "Dr. Jane Smith"
    assert result.termination_reason == "agent_declared_finish"
    assert result.metadata["navigation_steps"] == 0
    assert result.metadata["pages_visited"] == 1


def test_agent_navigates_then_finishes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pages = {
        "http://conf.example.com/": "<a href='/speakers'>Speakers</a>",
        "http://conf.example.com/speakers": "<p>Keynote: Dr. Jane Smith</p>",
    }
    provider = _ScriptedProvider(
        json_responses=[
            json.dumps({"thought": "click speakers", "action": "navigate", "link_index": 0}),
            json.dumps({"thought": "found it", "action": "finish", "answer": "Dr. Jane Smith"}),
        ]
    )
    result, provider = _run(monkeypatch, tmp_path, pages, provider)

    assert result.final_answer == "Dr. Jane Smith"
    assert result.metadata["navigation_steps"] == 1
    assert result.metadata["pages_visited"] == 2
    assert result.trajectory[0]["action"] == "navigate"
    assert result.trajectory[0]["to_url"] == "http://conf.example.com/speakers"


# --- malformed decisions do not crash ---


def test_invalid_link_index_is_recorded_and_does_not_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {"http://conf.example.com/": "<a href='/a'>A</a>"}
    provider = _ScriptedProvider(
        json_responses=[
            json.dumps({"thought": "bad index", "action": "navigate", "link_index": 99}),
            json.dumps({"thought": "give up", "action": "finish", "answer": "unknown"}),
        ]
    )
    result, provider = _run(monkeypatch, tmp_path, pages, provider)

    assert result.final_answer == "unknown"
    assert "error" in result.trajectory[0]
    assert result.metadata["navigation_steps"] == 0  # the invalid attempt never actually navigated


def test_unrecognized_action_is_recorded_and_does_not_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {"http://conf.example.com/": "<p>hi</p>"}
    provider = _ScriptedProvider(
        json_responses=[
            json.dumps({"thought": "confused", "action": "search", "query": "not allowed"}),
            json.dumps({"thought": "ok", "action": "finish", "answer": "final"}),
        ]
    )
    result, provider = _run(monkeypatch, tmp_path, pages, provider)
    assert result.final_answer == "final"
    assert result.trajectory[0]["error"] == "unrecognized action"


# --- step budget exhaustion forces a finish ---


def test_step_budget_exhaustion_forces_a_finish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://conf.example.com/": "<a href='/a'>A</a>",
        "http://conf.example.com/a": "<a href='/'>Home</a>",
    }
    agent = MatchedReActWebAgent(max_steps=2)
    # Always navigate, never finish -- must exhaust the 2-step budget.
    provider = _ScriptedProvider(
        json_responses=[
            json.dumps({"thought": "go", "action": "navigate", "link_index": 0}),
            json.dumps({"thought": "go back", "action": "navigate", "link_index": 0}),
        ],
        text_responses=["Dr. Jane Smith (forced)"],
    )
    result, provider = _run(monkeypatch, tmp_path, pages, provider, agent=agent)

    assert result.final_answer == "Dr. Jane Smith (forced)"
    assert result.termination_reason == "step_budget_exhausted_forced_finish"
    assert result.metadata["step_budget_exhausted"] is True
    assert result.metadata["navigation_steps"] == 2


# --- inaccessible page during trajectory ---


def test_inaccessible_page_reached_via_navigate_is_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://conf.example.com/": "<a href='/dead'>Dead link</a>"
    }  # /dead not in pages -> 404
    provider = _ScriptedProvider(
        json_responses=[
            json.dumps({"thought": "click it", "action": "navigate", "link_index": 0}),
            json.dumps({"thought": "give up", "action": "finish", "answer": "unknown"}),
        ]
    )
    result, provider = _run(monkeypatch, tmp_path, pages, provider)

    assert result.metadata["n_inaccessible_pages_in_trajectory"] == 1
    assert result.trajectory[0]["status"] == "error"


# --- gold leakage prevention ---


def test_no_gold_answer_or_audit_metadata_ever_enters_a_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {"http://conf.example.com/": "<p>Some content</p>"}
    provider = _ScriptedProvider(
        json_responses=[json.dumps({"thought": "done", "action": "finish", "answer": "final"})]
    )
    result, provider = _run(monkeypatch, tmp_path, pages, provider)

    for prompt in provider.prompts_sent:
        assert SECRET_ANSWER not in prompt
        assert SECRET_SOURCE not in prompt


# --- same-site / discovered-links-only navigation (end-to-end, via the real EvalWebEnvironment) ---


def test_navigation_is_restricted_to_links_actually_on_the_current_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Only one link is ever offered; an off-site link embedded in the HTML
    # must never appear as a selectable option at all (filtered by
    # PageCache's own same-site restriction before the agent ever sees it).
    pages = {"http://conf.example.com/": "<a href='/a'>A</a><a href='http://evil.com/x'>Evil</a>"}
    provider = _ScriptedProvider(
        json_responses=[json.dumps({"thought": "look", "action": "finish", "answer": "n/a"})]
    )
    monkeypatch.setattr(
        react_module, "urllib_fetcher", lambda timeout_seconds=10.0: _mock_fetcher(pages)
    )
    monkeypatch.setattr(react_module, "CountingOpenAIProvider", lambda model: provider)
    agent = MatchedReActWebAgent()
    agent.run(_example(), tmp_path)

    assert "http://evil.com/x" not in provider.prompts_sent[0]
    assert "http://conf.example.com/a" in provider.prompts_sent[0]


# --- cost/usage accounting ---


def test_usage_and_cost_are_drained_from_the_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {"http://conf.example.com/": "<p>hi</p>"}
    provider = _ScriptedProvider(
        json_responses=[json.dumps({"thought": "done", "action": "finish", "answer": "final"})]
    )
    result, provider = _run(monkeypatch, tmp_path, pages, provider)

    assert result.usage.llm_calls == 1
    assert result.usage.estimated_cost_usd == 0.001


def test_default_max_steps_matches_paper_cap() -> None:
    assert DEFAULT_MAX_STEPS == 15
    assert MatchedReActWebAgent().max_steps == 15


def test_agent_is_registered() -> None:
    from ant.evaluation_suite.registry import get_agent

    agent = get_agent("matched_react_web")
    assert isinstance(agent, MatchedReActWebAgent)
