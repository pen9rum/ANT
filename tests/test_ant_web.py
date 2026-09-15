"""Tests for ant.agents.ant_web.AntWebAgent and
ant.agents.web_navigation_tool.WebSearchTool. Page fetching uses a real
PageCache over a static, in-memory mock fetcher (no real network call).
Link-choice/synthesis LLM calls use small deterministic test doubles
(no API key, zero paid inference), per the governing spec's "no-cost
structural smoke test before paid inference" requirement.
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
from ant.agents.web_navigation_tool import WebSearchTool, WorkerFrontier, worker_key_from_files
from ant.benchmarks.base import TaskExample
from ant.domain import TokenUsage
from ant.evaluation_suite.web_fetch import PageCache, _RawFetch
from ant.evaluation_suite.web_scope import EvalWebEnvironment
from ant.providers.mock import MockLLMProvider

SECRET_ANSWER = "SECRET_GOLD_ANSWER_MUST_NEVER_LEAK"
SECRET_SOURCE = "SECRET_SOURCE_WEBSITE_MUST_NEVER_LEAK"


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


def _env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pages: dict[str, str],
    root_url: str,
    max_steps: int = 15,
):
    monkeypatch.setattr(
        ant_web_module, "urllib_fetcher", lambda timeout_seconds=10.0: _mock_fetcher(pages)
    )
    # A fresh, test-unique cache dir every call -- PageCache is disk-backed,
    # so a shared literal path would let one test's cached pages leak into
    # another test reusing the same root_url with different mock content.
    cache = PageCache(cache_dir=tmp_path / "pages", fetcher=_mock_fetcher(pages), root_url=root_url)
    return EvalWebEnvironment(root_url, cache, max_steps=max_steps)


# --- worker_key_from_files ---


def test_worker_key_from_files_parses_the_double_underscore_prefix() -> None:
    assert worker_key_from_files(["worker-0__root.txt"]) == "worker-0"
    assert worker_key_from_files(["worker-0__root.txt", "worker-0__hop0.txt"]) == "worker-0"
    assert worker_key_from_files([]) is None
    assert worker_key_from_files(["no_prefix.txt"]) is None


# --- bootstrap: root-only fetch, no eager link-following ---


def test_bootstrap_creates_one_worker_per_link_plus_root_without_fetching_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://conf.example.com/": (
            "<a href='/speakers'>Speakers</a><a href='/schedule'>Schedule</a>"
        ),
        "http://conf.example.com/speakers": "<p>Dr. Jane Smith</p>",
        "http://conf.example.com/schedule": "<p>9am keynote</p>",
    }
    monkeypatch.setattr(
        ant_web_module, "urllib_fetcher", lambda timeout_seconds=10.0: _mock_fetcher(pages)
    )
    workers, frontiers, materialized_dir, env, n_inaccessible = _bootstrap_territories(
        _example(), tmp_path, nav_budget=15, nav_link_cap=15, timeout_seconds=10.0
    )

    assert len(workers) == 3  # root + speakers + schedule
    assert {w.id for w in workers} == {"worker-root", "worker-0", "worker-1"}
    assert n_inaccessible == 0
    # The whole point of the fix: bootstrap must NOT have fetched the
    # linked pages yet -- only the root page has actually been visited.
    assert env.discovered_pages() == ["http://conf.example.com/"]
    assert env.step_count() == 0
    assert (materialized_dir / "worker-root__root.txt").exists()
    assert frontiers["worker-0"].assigned_link is not None
    assert frontiers["worker-0"].taken_first_hop is False
    assert frontiers["worker-root"].taken_first_hop is True  # no assigned link to chase


def test_bootstrap_respects_nav_link_cap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pages = {"http://conf.example.com/": "".join(f"<a href='/p{i}'>P{i}</a>" for i in range(10))}
    monkeypatch.setattr(
        ant_web_module, "urllib_fetcher", lambda timeout_seconds=10.0: _mock_fetcher(pages)
    )
    workers, frontiers, materialized_dir, env, n_inaccessible = _bootstrap_territories(
        _example(), tmp_path, nav_budget=15, nav_link_cap=3, timeout_seconds=10.0
    )
    assert len(workers) == 4  # root + 3 (capped)


def test_bootstrap_handles_inaccessible_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ant_web_module, "urllib_fetcher", lambda timeout_seconds=10.0: _mock_fetcher({})
    )
    workers, frontiers, materialized_dir, env, n_inaccessible = _bootstrap_territories(
        _example(), tmp_path, nav_budget=15, nav_link_cap=15, timeout_seconds=10.0
    )
    assert workers == []
    assert n_inaccessible == 1


def test_bootstrap_never_touches_gold_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://conf.example.com/": "<a href='/a'>A</a>",
        "http://conf.example.com/a": "<p>hi</p>",
    }
    monkeypatch.setattr(
        ant_web_module, "urllib_fetcher", lambda timeout_seconds=10.0: _mock_fetcher(pages)
    )
    workers, frontiers, materialized_dir, env, _ = _bootstrap_territories(
        _example(), tmp_path, nav_budget=15, nav_link_cap=15, timeout_seconds=10.0
    )
    for w in workers:
        assert SECRET_ANSWER not in " ".join(w.responsibilities)
        assert SECRET_SOURCE not in " ".join(w.responsibilities)
    for f in materialized_dir.glob("*.txt"):
        assert SECRET_ANSWER not in f.read_text(encoding="utf-8")


# --- WebSearchTool: the actual multi-hop navigation mechanism ---


class _FixedLinkChoiceProvider:
    """Deterministic test double for the raw responses_text() call
    WebSearchTool makes directly for link-choice -- returns a fixed index
    (or "NONE") regardless of prompt content."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def responses_text(self, prompt: str, max_output_tokens: int = 16):
        from ant.providers.openai_provider import ResponseResult

        self.prompts.append(prompt)
        return ResponseResult(text=self.reply, usage=TokenUsage(), raw={})


def test_search_takes_the_mandatory_first_hop_even_when_root_already_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Regression test for a real bug found on a live paid run: gating even
    # the first hop behind a term-overlap check let root's own generic
    # navigation text satisfy the check for most decomposed sub-needs, so
    # workers never reached their own assigned territory at all (9 active
    # workers, 10 rounds, 0 navigation steps in that run). The first hop
    # must be unconditional, regardless of whether root-only content
    # happens to already overlap with the query.
    pages = {
        "http://x.test/": "<a href='/listing'>Listing</a>",
        "http://x.test/listing": "more about the keynote speaker",
    }
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text(
        "the keynote speaker is Dr. Jane Smith", encoding="utf-8"
    )
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    tool = WebSearchTool(materialized, env, _FixedLinkChoiceProvider("NONE"), frontiers)

    results = tool.search("keynote speaker", ["worker-0__root.txt"])

    assert results
    assert env.step_count() == 1  # the mandatory first hop still happened
    assert frontiers["worker-0"].taken_first_hop is True


def test_search_does_not_repeat_the_first_hop_on_a_later_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {"http://x.test/": "<a href='/listing'>Listing</a>"}
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text(
        "the keynote speaker is Dr. Jane Smith", encoding="utf-8"
    )
    frontier = WorkerFrontier(current_page=root, assigned_link=root.links[0], taken_first_hop=True)
    frontiers = {"worker-0": frontier}
    tool = WebSearchTool(materialized, env, _FixedLinkChoiceProvider("NONE"), frontiers)

    results = tool.search("keynote speaker", ["worker-0__root.txt"])

    assert results
    assert env.step_count() == 0  # already taken in an earlier call -- not repeated
    assert tool.nav_log == []


def test_search_takes_deterministic_first_hop_when_local_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://x.test/": "<a href='/listing'>Listing</a>",
        "http://x.test/listing": "the keynote speaker is Dr. Jane Smith",
    }
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing relevant here", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _FixedLinkChoiceProvider("NONE")
    tool = WebSearchTool(materialized, env, provider, frontiers)
    files = ["worker-0__root.txt"]

    results = tool.search("keynote speaker", files)

    assert results
    assert env.step_count() == 1  # exactly one deterministic hop, no LLM call needed for it
    assert provider.prompts == []  # first hop is deterministic, never asks the LLM
    assert "worker-0__hop0.txt" in files
    assert tool.nav_log[0]["worker"] == "worker-0"


def test_search_follows_llm_chosen_second_hop_when_first_hop_insufficient(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://x.test/": "<a href='/listing'>Listing</a>",
        "http://x.test/listing": "<a href='/article'>Update 19</a><a href='/other'>Other</a>",
        "http://x.test/article": "the keynote speaker is Dr. Jane Smith",
        "http://x.test/other": "irrelevant content",
    }
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing relevant", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _FixedLinkChoiceProvider("0")  # always picks the first offered link
    tool = WebSearchTool(materialized, env, provider, frontiers)
    files = ["worker-0__root.txt"]

    results = tool.search("keynote speaker", files)

    assert results
    assert env.step_count() == 2  # deterministic hop 1 (listing) + LLM-chosen hop 2 (article)
    assert len(provider.prompts) == 1  # only the second hop needed an LLM call
    assert "article" in provider.prompts[0] or "Update 19" in provider.prompts[0]


def test_search_respects_no_arbitrary_jump_on_out_of_range_choice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://x.test/": "<a href='/listing'>Listing</a>",
        "http://x.test/listing": "<a href='/article'>Article</a>",
    }
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _FixedLinkChoiceProvider("99")  # out of range -- must not navigate anywhere
    tool = WebSearchTool(materialized, env, provider, frontiers)
    files = ["worker-0__root.txt"]

    tool.search("keynote speaker", files)

    # The out-of-range index must not cause any navigation beyond the
    # one deterministic first hop -- _choose_link_via_llm rejects it and
    # returns None, so the loop stops rather than jumping anywhere.
    assert env.step_count() == 1
    assert len(tool.nav_log) == 1
    assert tool.nav_log[0]["url"] == "http://x.test/listing"


def test_shared_nav_budget_stops_a_second_worker_once_exhausted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://x.test/": "<a href='/a'>A</a><a href='/b'>B</a>",
        "http://x.test/a": "content a",
        "http://x.test/b": "content b",
    }
    env = _env(
        tmp_path, monkeypatch, pages, "http://x.test/", max_steps=1
    )  # shared budget: 1 total
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    (materialized / "worker-1__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {
        "worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0]),
        "worker-1": WorkerFrontier(current_page=root, assigned_link=root.links[1]),
    }
    provider = _FixedLinkChoiceProvider("NONE")
    tool = WebSearchTool(materialized, env, provider, frontiers)

    tool.search("keynote speaker", ["worker-0__root.txt"])
    tool.search("keynote speaker", ["worker-1__root.txt"])

    assert env.step_count() == 1  # NOT 2 -- the shared budget is global, not per-worker
    assert len(tool.nav_log) == 1


def test_dense_search_never_triggers_navigation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {"http://x.test/": "<a href='/a'>A</a>", "http://x.test/a": "content"}
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    tool = WebSearchTool(materialized, env, _FixedLinkChoiceProvider("0"), frontiers)

    tool.dense_search("q", ["worker-0__root.txt"])

    assert env.step_count() == 0
    assert tool.nav_log == []


# --- full run() wiring, deterministic end-to-end (no API key) ---


class _StubProvider(MockLLMProvider):
    """MockLLMProvider (LocalCoordinator's own default reasoner fallback)
    plus the small extra surface AntWebAgent.run() and WebSearchTool need:
    synthesize/synthesize_coalition (MockLLMProvider itself only
    implements the reasoner side), drain_call_count/drain_usage/
    drain_retry_log (the CountingOpenAIProvider surface), and
    responses_text (WebSearchTool's own direct link-choice call)."""

    def __init__(self, link_choice_reply: str = "NONE") -> None:
        self.link_choice_reply = link_choice_reply
        self.responses_text_calls = 0

    def synthesize(self, *, question, evidence, **kwargs):
        return f"mock answer for: {question}" if evidence else ""

    def synthesize_coalition(self, *, question, worker_ids, evidence, **kwargs):
        return f"mock coalition answer for: {question}" if evidence else ""

    def responses_text(self, prompt: str, max_output_tokens: int = 16):
        from ant.providers.openai_provider import ResponseResult

        self.responses_text_calls += 1
        return ResponseResult(text=self.link_choice_reply, usage=TokenUsage(), raw={})

    def drain_call_count(self) -> int:
        return self.responses_text_calls

    def drain_usage(self) -> TokenUsage:
        return TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2, estimated_cost_usd=0.0)

    def drain_retry_log(self) -> list[dict]:
        return []


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, pages: dict[str, str], agent=None):
    monkeypatch.setattr(
        ant_web_module, "urllib_fetcher", lambda timeout_seconds=10.0: _mock_fetcher(pages)
    )
    monkeypatch.setattr(ant_web_module, "CountingOpenAIProvider", lambda model: _StubProvider())
    agent = agent or AntWebAgent()
    return agent.run(_example(), tmp_path)


def test_run_completes_and_produces_an_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://conf.example.com/": "<a href='/speakers'>Speakers</a>",
        "http://conf.example.com/speakers": "Dr. Jane Smith is the keynote speaker.",
    }
    result = _run(monkeypatch, tmp_path, pages)

    assert isinstance(result.final_answer, str)
    assert result.metadata["nav_budget"] == DEFAULT_NAV_BUDGET
    assert result.metadata["max_rounds"] == DEFAULT_MAX_ROUNDS


def test_run_reports_navigation_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pages = {
        "http://conf.example.com/": "<a href='/a'>A</a>",
        "http://conf.example.com/a": "authenticate_user is defined here.",
    }
    result = _run(monkeypatch, tmp_path, pages)

    for key in (
        "navigation_steps",
        "navigation_steps_by_worker",
        "nav_log",
        "territories_discovered",
        "pages_fetched",
        "active_workers",
        "n_rounds",
        "n_reroutes",
        "n_need_revisions",
        "nav_budget_exhausted",
        "final_need_graph_size",
    ):
        assert key in result.metadata


def test_run_never_leaks_gold_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pages = {"http://conf.example.com/": "hi"}
    result = _run(monkeypatch, tmp_path, pages)
    assert SECRET_ANSWER not in result.final_answer
    assert SECRET_SOURCE not in result.final_answer


def test_shared_navigation_budget_is_never_exceeded_in_a_full_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A root with many links and a provider that always tries to keep
    # navigating (never says NONE) -- the assertion inside
    # AntWebAgent.run() itself (env.step_count() <= nav_budget) is the
    # real guard; this test just exercises a path likely to stress it.
    links = "".join(f"<a href='/p{i}'>P{i}</a>" for i in range(20))
    pages = {"http://conf.example.com/": links}
    pages.update({f"http://conf.example.com/p{i}": f"<a href='/p{i}b'>more</a>" for i in range(20)})
    agent = AntWebAgent(nav_budget=5, max_rounds=3, nav_link_cap=20)
    result = _run(monkeypatch, tmp_path, pages, agent=agent)
    assert result.metadata["navigation_steps"] <= 5


def test_default_nav_budget_and_max_rounds() -> None:
    # nav_budget matches ReAct's own step budget (fairness); max_rounds=10
    # is a deliberate coordination-budget choice, independent of
    # nav_budget -- see this module's own docstring. NOT max_rounds=15
    # (an earlier, corrected design decision).
    agent = AntWebAgent()
    assert agent.nav_budget == 15
    assert agent.max_rounds == 10
    assert DEFAULT_NAV_BUDGET == 15
    assert DEFAULT_MAX_ROUNDS == 10


def test_agent_is_registered() -> None:
    from ant.evaluation_suite.registry import get_agent

    agent = get_agent("ant_web")
    assert isinstance(agent, AntWebAgent)


# --- the governing spec's explicit "multi-hop reachability" regression:
# root -> listing -> article -> evidence, where article is NOT present in
# the initial bootstrap roster at all. ---


def test_multi_hop_answer_unreachable_at_bootstrap_is_reachable_during_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://conf.example.com/": "<a href='/release-notes'>Release Notes</a>",
        "http://conf.example.com/release-notes": ("<a href='/update-19'>Update 19.18309</a>"),
        "http://conf.example.com/update-19": (
            "Lipizzaner Cavalry increases attack and hitpoints of Uhlans by 20%."
        ),
    }
    monkeypatch.setattr(
        ant_web_module, "urllib_fetcher", lambda timeout_seconds=10.0: _mock_fetcher(pages)
    )
    monkeypatch.setattr(
        ant_web_module, "CountingOpenAIProvider", lambda model: _StubProvider(link_choice_reply="0")
    )
    agent = AntWebAgent()
    result = agent.run(_example(), tmp_path)

    # The bootstrap roster itself must NOT contain the answer page --
    # that's the whole point (matching the real bug: it's 2 hops deep).
    assert result.metadata["territories_discovered"] == 2  # root + release-notes only
    # But the live run must have actually reached it via real navigation.
    assert result.metadata["navigation_steps"] >= 2
    assert any("update-19" in entry["url"] for entry in result.metadata["nav_log"])
