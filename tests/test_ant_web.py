"""Tests for ant.agents.ant_web.AntWebAgent. Page fetching uses a real
PageCache over a static, in-memory mock fetcher (no real network call);
LLM coordination uses ant.providers.mock.MockLLMProvider (deterministic,
no API key, the same default ant.coordinator.local.LocalCoordinator
itself falls back to when reasoner/synthesizer are both omitted) --
zero paid inference, per the governing spec.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ant.agents import ant_web as ant_web_module
from ant.agents.ant_web import (
    DEFAULT_MAX_ROUNDS,
    DEFAULT_NAV_BUDGET,
    AntWebAgent,
    _bootstrap_territories,
)
from ant.benchmarks.base import TaskExample
from ant.domain import TokenUsage
from ant.evaluation_suite.web_fetch import _RawFetch
from ant.providers.mock import MockLLMProvider

SECRET_ANSWER = "SECRET_GOLD_ANSWER_MUST_NEVER_LEAK"
SECRET_SOURCE = "SECRET_SOURCE_WEBSITE_MUST_NEVER_LEAK"


class _CountingMockProvider(MockLLMProvider):
    """MockLLMProvider (the WorkerReasoner protocol, the same deterministic
    fallback LocalCoordinator itself defaults to when no reasoner is
    given) plus a minimal deterministic `synthesize`/`synthesize_coalition`
    (MockLLMProvider itself only implements the reasoner side -- confirmed
    live: LocalCoordinator's default-reasoner fallback leaves `synthesizer`
    None, so its own synthesis branch is simply skipped; AntWebAgent.run()
    passes the same object as both, so this test double must cover both)
    and the drain_call_count()/drain_usage() surface CountingOpenAIProvider
    normally provides -- so AntWebAgent.run() can complete without a real
    API key or network call."""

    def synthesize(self, *, question, evidence, **kwargs):
        return f"mock answer for: {question}" if evidence else ""

    def synthesize_coalition(self, *, question, worker_ids, evidence, **kwargs):
        return f"mock coalition answer for: {question}" if evidence else ""

    def drain_call_count(self) -> int:
        return 0

    def drain_usage(self) -> TokenUsage:
        return TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2, estimated_cost_usd=0.0)

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


# --- bootstrap territory-building (no LLM involved at all) ---


def test_bootstrap_creates_one_worker_per_root_level_link_plus_root(tmp_path: Path) -> None:
    pages = {
        "http://conf.example.com/": (
            "<a href='/speakers'>Speakers</a><a href='/schedule'>Schedule</a>"
        ),
        "http://conf.example.com/speakers": "<p>Dr. Jane Smith</p>",
        "http://conf.example.com/schedule": "<p>9am keynote</p>",
    }
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        ant_web_module, "urllib_fetcher", lambda timeout_seconds=10.0: _mock_fetcher(pages)
    )
    workers, materialized_dir, n_nav, n_inaccessible, exhausted = _bootstrap_territories(
        _example(), tmp_path, nav_budget=15, timeout_seconds=10.0
    )
    monkeypatch.undo()

    assert len(workers) == 3  # root + speakers + schedule
    assert {w.id for w in workers} == {"worker-root", "worker-0", "worker-1"}
    assert n_nav == 2
    assert n_inaccessible == 0
    assert exhausted is False
    assert (materialized_dir / "page_root.txt").exists()


def test_bootstrap_skips_inaccessible_links_without_creating_a_worker(tmp_path: Path) -> None:
    pages = {
        "http://conf.example.com/": "<a href='/dead'>Dead</a><a href='/ok'>OK</a>",
        "http://conf.example.com/ok": "<p>hi</p>",
    }
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        ant_web_module, "urllib_fetcher", lambda timeout_seconds=10.0: _mock_fetcher(pages)
    )
    workers, materialized_dir, n_nav, n_inaccessible, exhausted = _bootstrap_territories(
        _example(), tmp_path, nav_budget=15, timeout_seconds=10.0
    )
    monkeypatch.undo()

    assert len(workers) == 2  # root + OK (dead link skipped, no worker)
    assert n_inaccessible == 1


def test_bootstrap_respects_the_nav_budget(tmp_path: Path) -> None:
    pages = {
        "http://conf.example.com/": "".join(f"<a href='/p{i}'>P{i}</a>" for i in range(10)),
        **{f"http://conf.example.com/p{i}": f"<p>page {i}</p>" for i in range(10)},
    }
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        ant_web_module, "urllib_fetcher", lambda timeout_seconds=10.0: _mock_fetcher(pages)
    )
    workers, materialized_dir, n_nav, n_inaccessible, exhausted = _bootstrap_territories(
        _example(), tmp_path, nav_budget=3, timeout_seconds=10.0
    )
    monkeypatch.undo()

    assert n_nav == 3
    assert exhausted is True
    assert len(workers) == 4  # root + 3 navigated pages


def test_bootstrap_never_touches_gold_metadata(tmp_path: Path) -> None:
    pages = {
        "http://conf.example.com/": "<a href='/a'>A</a>",
        "http://conf.example.com/a": "<p>hi</p>",
    }
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        ant_web_module, "urllib_fetcher", lambda timeout_seconds=10.0: _mock_fetcher(pages)
    )
    workers, materialized_dir, *_ = _bootstrap_territories(
        _example(), tmp_path, nav_budget=15, timeout_seconds=10.0
    )
    monkeypatch.undo()

    for w in workers:
        assert SECRET_ANSWER not in " ".join(w.responsibilities)
        assert SECRET_SOURCE not in " ".join(w.responsibilities)
    for f in materialized_dir.glob("*.txt"):
        assert SECRET_ANSWER not in f.read_text(encoding="utf-8")


# --- full run() wiring, via the deterministic MockLLMProvider ---


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pages: dict[str, str], agent=None):
    monkeypatch.setattr(
        ant_web_module, "urllib_fetcher", lambda timeout_seconds=10.0: _mock_fetcher(pages)
    )
    monkeypatch.setattr(
        ant_web_module, "CountingOpenAIProvider", lambda model: _CountingMockProvider()
    )
    agent = agent or AntWebAgent()
    return agent.run(_example(), tmp_path)


def test_run_completes_and_produces_an_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://conf.example.com/": "<a href='/speakers'>Speakers</a>",
        "http://conf.example.com/speakers": "<p>Dr. Jane Smith is the keynote speaker.</p>",
    }
    result = _run(monkeypatch, tmp_path, pages)

    assert isinstance(result.final_answer, str)
    assert result.metadata["territories_discovered"] == 2
    assert result.metadata["nav_budget"] == DEFAULT_NAV_BUDGET
    assert result.metadata["max_rounds"] == DEFAULT_MAX_ROUNDS


def test_run_reports_active_workers_and_rounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://conf.example.com/": "<a href='/a'>A</a>",
        "http://conf.example.com/a": "<p>authenticate_user is defined here.</p>",
    }
    result = _run(monkeypatch, tmp_path, pages)

    assert result.metadata["active_workers"] >= 0
    assert result.metadata["n_rounds"] >= 0
    assert "n_reroutes" in result.metadata
    assert "n_need_revisions" in result.metadata


def test_run_never_leaks_gold_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pages = {"http://conf.example.com/": "<p>hi</p>"}
    result = _run(monkeypatch, tmp_path, pages)
    assert SECRET_ANSWER not in result.final_answer
    assert SECRET_SOURCE not in result.final_answer


def test_default_nav_budget_and_max_rounds_are_both_15() -> None:
    # This task's own explicit override: nav_budget matches ReAct's own
    # step budget (fairness); max_rounds=15 is a deliberate task-specific
    # override of ANTMAN's canonical max_rounds=6 default (per the user's
    # own explicit instruction), not a core-code change.
    agent = AntWebAgent()
    assert agent.nav_budget == 15
    assert agent.max_rounds == 15
    assert DEFAULT_NAV_BUDGET == 15
    assert DEFAULT_MAX_ROUNDS == 15


def test_agent_is_registered() -> None:
    from ant.evaluation_suite.registry import get_agent

    agent = get_agent("ant_web")
    assert isinstance(agent, AntWebAgent)
