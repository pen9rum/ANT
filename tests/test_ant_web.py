"""Tests for ant.agents.ant_web.AntWebAgent and
ant.agents.web_navigation_tool.WebSearchTool. Page fetching uses a real
PageCache over a static, in-memory mock fetcher (no real network call).
Link-choice/synthesis LLM calls use small deterministic test doubles
(no API key, zero paid inference), per the governing spec's "no-cost
structural smoke test before paid inference" requirement.
"""

from __future__ import annotations

import json
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
        _example(), tmp_path, nav_budget=15, max_candidate_workers=15, timeout_seconds=10.0
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


def test_bootstrap_respects_max_candidate_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {"http://conf.example.com/": "".join(f"<a href='/p{i}'>P{i}</a>" for i in range(10))}
    monkeypatch.setattr(
        ant_web_module, "urllib_fetcher", lambda timeout_seconds=10.0: _mock_fetcher(pages)
    )
    workers, frontiers, materialized_dir, env, n_inaccessible = _bootstrap_territories(
        _example(), tmp_path, nav_budget=15, max_candidate_workers=3, timeout_seconds=10.0
    )
    assert len(workers) == 4  # root + 3 (capped)


def test_bootstrap_handles_inaccessible_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        ant_web_module, "urllib_fetcher", lambda timeout_seconds=10.0: _mock_fetcher({})
    )
    workers, frontiers, materialized_dir, env, n_inaccessible = _bootstrap_territories(
        _example(), tmp_path, nav_budget=15, max_candidate_workers=15, timeout_seconds=10.0
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
        _example(), tmp_path, nav_budget=15, max_candidate_workers=15, timeout_seconds=10.0
    )
    for w in workers:
        assert SECRET_ANSWER not in " ".join(w.responsibilities)
        assert SECRET_SOURCE not in " ".join(w.responsibilities)
    for f in materialized_dir.glob("*.txt"):
        assert SECRET_ANSWER not in f.read_text(encoding="utf-8")


# --- WebSearchTool: the actual multi-hop, grounded-decision navigation
# mechanism. Decisions carry two ORTHOGONAL fields -- progress ("none" |
# "partial" | "resolved") and action ("continue" | "return" | "dead_end")
# -- see web_navigation_tool.py's own module docstring for the full
# rationale (a single conflated status field was confirmed live, via a
# 5-task forensic ablation, to force a worker with genuinely useful but
# incomplete evidence into either overclaiming "resolved" or silently
# discarding what it found). Grounding verification governs ONLY which
# evidence gets accepted, never the model's own chosen action. There is
# no ungrounded fallback: search() returns exactly the grounded evidence
# accumulated this dispatch, possibly empty. ---


def _none_continue(index: int) -> dict:
    return {
        "progress": "none",
        "action": "continue",
        "evidence": [],
        "next_link_index": index,
        "reason": "",
    }


def _dead_end() -> dict:
    return {
        "progress": "none",
        "action": "dead_end",
        "evidence": [],
        "next_link_index": None,
        "reason": "",
    }


def _partial_continue(index: int, claim: str, supporting_text: str) -> dict:
    return {
        "progress": "partial",
        "action": "continue",
        "evidence": [{"claim": claim, "supporting_text": supporting_text}],
        "next_link_index": index,
        "reason": "",
    }


def _partial_return(claim: str, supporting_text: str) -> dict:
    return {
        "progress": "partial",
        "action": "return",
        "evidence": [{"claim": claim, "supporting_text": supporting_text}],
        "next_link_index": None,
        "reason": "",
    }


def _resolved(claim: str, supporting_text: str) -> dict:
    return {
        "progress": "resolved",
        "action": "return",
        "evidence": [{"claim": claim, "supporting_text": supporting_text}],
        "next_link_index": None,
        "reason": "",
    }


class _ScriptedDecisionProvider:
    """Deterministic test double for WebSearchTool's own direct
    responses_text() calls (the merged link-choice + grounded-progress
    decision call). `decisions` is consumed in order, one entry per call;
    each entry is either a dict (auto-serialized to the JSON schema
    _decide() expects) or a raw string (used as-is, e.g. to simulate a
    malformed/unparseable model reply). The last entry repeats if more
    calls happen than were scripted."""

    def __init__(self, decisions: list[dict | str]) -> None:
        self._decisions = [d if isinstance(d, str) else json.dumps(d) for d in decisions]
        self.prompts: list[str] = []
        self._index = 0

    def responses_text(self, prompt: str, max_output_tokens: int = 16):
        from ant.providers.openai_provider import ResponseResult

        self.prompts.append(prompt)
        text = self._decisions[min(self._index, len(self._decisions) - 1)]
        self._index += 1
        return ResponseResult(text=text, usage=TokenUsage(), raw={})


def test_search_takes_the_mandatory_first_hop_even_when_root_already_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Regression test for a real bug found on a live paid run: gating even
    # the first hop behind a stopping heuristic left workers stuck on
    # root (9 active workers, 10 rounds, 0 navigation steps in that run).
    # The first hop must be unconditional.
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
    provider = _ScriptedDecisionProvider([_dead_end()])
    tool = WebSearchTool(materialized, env, provider, frontiers)

    results = tool.search("keynote speaker", ["worker-0__root.txt"])

    # No ungrounded fallback exists anymore: none+dead_end with no
    # grounded evidence returns nothing, even though root.txt's own text
    # would have matched a plain keyword search.
    assert results == []
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
    provider = _ScriptedDecisionProvider([_dead_end()])
    tool = WebSearchTool(materialized, env, provider, frontiers)

    results = tool.search("keynote speaker", ["worker-0__root.txt"])

    assert results == []
    assert env.step_count() == 0  # already taken in an earlier call -- not repeated
    assert tool.nav_log == []


def test_search_takes_deterministic_first_hop_then_makes_one_decision_call(
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
    provider = _ScriptedDecisionProvider(
        [_resolved("keynote speaker", "the keynote speaker is Dr. Jane Smith")]
    )
    tool = WebSearchTool(materialized, env, provider, frontiers)
    files = ["worker-0__root.txt"]

    results = tool.search("keynote speaker", files)

    assert results
    assert env.step_count() == 1  # the hop itself is deterministic, no LLM call needed for it
    assert len(provider.prompts) == 1  # exactly one decision call, evaluating the new page
    assert "worker-0__hop0.txt" in files
    assert tool.nav_log[0]["worker"] == "worker-0"
    assert results[0].quote == "the keynote speaker is Dr. Jane Smith"


def test_search_continues_to_an_llm_chosen_second_hop_when_first_page_unresolved(
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
    provider = _ScriptedDecisionProvider(
        [_none_continue(0), _resolved("keynote speaker", "the keynote speaker is Dr. Jane Smith")]
    )
    tool = WebSearchTool(materialized, env, provider, frontiers)
    files = ["worker-0__root.txt"]

    results = tool.search("keynote speaker", files)

    assert results
    assert env.step_count() == 2  # deterministic hop 1 (listing) + decision-chosen hop 2 (article)
    assert len(provider.prompts) == 2  # one decision call per page visited
    assert results[0].quote == "the keynote speaker is Dr. Jane Smith"


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
    provider = _ScriptedDecisionProvider([_none_continue(99)])  # out of range
    tool = WebSearchTool(materialized, env, provider, frontiers)
    files = ["worker-0__root.txt"]

    results = tool.search("keynote speaker", files)

    # The out-of-range index must not cause any navigation beyond the one
    # deterministic first hop -- _resolve_link_index rejects it and
    # returns None, so the loop stops rather than jumping anywhere.
    assert results == []
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
    provider = _ScriptedDecisionProvider([_dead_end()])
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
    provider = _ScriptedDecisionProvider([_none_continue(0)])
    tool = WebSearchTool(materialized, env, provider, frontiers)

    tool.dense_search("q", ["worker-0__root.txt"])

    assert env.step_count() == 0
    assert tool.nav_log == []


# --- Web ANTMAN Fix v3's own required structural tests: term-overlap is
# no longer a stopping signal at all; resolution is grounded/verified. ---


def test_v3_a_generic_entity_repetition_alone_does_not_stop_the_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A page that merely repeats the question's own entity name (the kind
    # of text that trivially "term-overlapped" under an earlier, removed
    # heuristic) must NOT be treated as sufficient just because the
    # decision call could easily rationalize it -- only an explicit,
    # grounded claim (verified against real page text) can add evidence.
    # Here the model is scripted to correctly recognize the generic page
    # is not enough and continue.
    pages = {
        "http://x.test/": "<a href='/landing'>Landing</a>",
        "http://x.test/landing": (
            "Age of Empires Age of Empires Age of Empires -- welcome to the site! "
            "<a href='/patchnotes'>Patch Notes</a>"
        ),
        "http://x.test/patchnotes": "Lipizzaner Cavalry increases attack and hitpoints by 20%.",
    }
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _ScriptedDecisionProvider(
        [
            _none_continue(0),  # correctly judges the generic landing page insufficient
            _resolved(
                "Lipizzaner Cavalry effect",
                "Lipizzaner Cavalry increases attack and hitpoints by 20%.",
            ),
        ]
    )
    tool = WebSearchTool(materialized, env, provider, frontiers)

    results = tool.search("Age of Empires Lipizzaner Cavalry percentage", ["worker-0__root.txt"])

    assert env.step_count() == 2  # continued past the generic page to patchnotes
    assert results
    assert "20%" in results[0].quote


def test_v3_c_a_genuinely_sufficient_page_stops_and_returns_grounded_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://x.test/": "<a href='/answer'>Answer</a>",
        "http://x.test/answer": ("Lipizzaner Cavalry increases attack and hitpoints by 20%."),
    }
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _ScriptedDecisionProvider(
        [_resolved("percentage", "Lipizzaner Cavalry increases attack and hitpoints by 20%.")]
    )
    tool = WebSearchTool(materialized, env, provider, frontiers)

    results = tool.search("percentage increase", ["worker-0__root.txt"])

    assert env.step_count() == 1  # stopped right after the mandatory hop -- no further navigation
    assert len(provider.prompts) == 1
    assert len(results) == 1
    assert results[0].quote == "Lipizzaner Cavalry increases attack and hitpoints by 20%."
    assert results[0].claim == "percentage"


def test_v3_d_a_hallucinated_supporting_span_cannot_resolve_the_need(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://x.test/": "<a href='/page'>Page</a>",
        "http://x.test/page": "this page talks about something else entirely",
    }
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text(
        "percentage increase mentioned here", encoding="utf-8"
    )
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    # The model CLAIMS resolution with a quote that does not actually
    # appear anywhere on the page -- a hallucination.
    provider = _ScriptedDecisionProvider(
        [
            _resolved(
                "percentage", "Lipizzaner Cavalry increases attack by 99% (never actually said)"
            )
        ]
    )
    tool = WebSearchTool(materialized, env, provider, frontiers)

    results = tool.search("percentage increase", ["worker-0__root.txt"])

    # Grounding verification must reject the hallucinated span, and there
    # is no ungrounded fallback anymore -- the Need must never be marked
    # resolved from it, and the dispatch returns nothing at all (action
    # was "return", so the loop stops immediately with zero accumulated
    # grounded evidence).
    assert results == []


def test_v3_e_a_dead_end_page_returns_control_without_further_navigation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://x.test/": "<a href='/deadend'>Dead End</a>",
        "http://x.test/deadend": "<a href='/more'>More</a>",
        "http://x.test/more": "the real answer is here",
    }
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _ScriptedDecisionProvider([_dead_end()])  # judges "deadend" unproductive
    tool = WebSearchTool(materialized, env, provider, frontiers)

    tool.search("the real answer", ["worker-0__root.txt"])

    assert env.step_count() == 1  # only the mandatory hop -- dead_end stopped further navigation
    assert len(provider.prompts) == 1


def test_v4_identical_need_and_page_state_hits_the_decision_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Regression test for the profiling-driven optimization: a worker
    # re-dispatched with the SAME need against the SAME (unchanged) page
    # must not re-ask the model the identical question.
    pages = {"http://x.test/": "<a href='/a'>A</a>", "http://x.test/a": "nothing relevant"}
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _ScriptedDecisionProvider([_dead_end()])
    tool = WebSearchTool(materialized, env, provider, frontiers)
    files = ["worker-0__root.txt"]

    tool.search("same need", files)  # first dispatch: real decision call, then dead_end
    first_prompt_count = len(provider.prompts)
    tool.search("same need", files)  # second dispatch: identical (need, page) -- must be cached
    tool.search("same need", files)  # third dispatch: still cached

    assert first_prompt_count == 1
    assert len(provider.prompts) == 1  # no new LLM call on the second or third dispatch
    assert env.step_count() == 1  # and definitely no re-navigation


def test_v4_a_different_need_on_the_same_page_is_not_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {"http://x.test/": "<a href='/a'>A</a>", "http://x.test/a": "nothing relevant"}
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _ScriptedDecisionProvider([_dead_end()])
    tool = WebSearchTool(materialized, env, provider, frontiers)
    files = ["worker-0__root.txt"]

    tool.search("need one", files)
    tool.search("need two", files)  # different need, same page -- must NOT be a cache hit

    assert len(provider.prompts) == 2


def test_v4_b_cache_enabled_false_disables_the_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Forensic/ablation knob: cache_enabled=False must reproduce exactly
    # what happened before the cache existed -- an identical (need, page)
    # triple asks the model again every time, never reusing a stored
    # decision.
    pages = {"http://x.test/": "<a href='/a'>A</a>", "http://x.test/a": "nothing relevant"}
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _ScriptedDecisionProvider([_dead_end()])
    tool = WebSearchTool(materialized, env, provider, frontiers, cache_enabled=False)
    files = ["worker-0__root.txt"]

    tool.search("same need", files)
    tool.search("same need", files)
    tool.search("same need", files)

    assert len(provider.prompts) == 3  # every dispatch re-asked the model
    assert tool._decision_cache == {}  # nothing was ever stored
    assert all(not entry["cache_hit"] for entry in tool.decision_log)


def test_decision_log_records_progress_action_and_grounding_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://x.test/": "<a href='/a'>A</a>",
        "http://x.test/a": "the exact answer is here",
    }
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _ScriptedDecisionProvider([_resolved("claim", "the exact answer is here")])
    tool = WebSearchTool(materialized, env, provider, frontiers)

    tool.search("find the answer", ["worker-0__root.txt"])

    assert len(tool.decision_log) == 1
    entry = tool.decision_log[0]
    assert entry["progress"] == "resolved"
    assert entry["action"] == "return"
    assert entry["cache_hit"] is False
    assert entry["n_evidence_proposed"] == 1
    assert entry["n_evidence_grounded"] == 1  # the quote is real, grounding succeeds


# --- Web ANTMAN Fix v2's own required structural regression tests ---
# (A) no breadth-first bootstrap spending, (B) deep single-dispatch
# execution, (C) persistent frontier across dispatches, (D) selective
# activation, (E) global budget -- see this file's own module docstring.


def test_v2_a_bootstrap_creates_many_candidates_with_zero_navigation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {"http://conf.example.com/": "".join(f"<a href='/p{i}'>P{i}</a>" for i in range(20))}
    monkeypatch.setattr(
        ant_web_module, "urllib_fetcher", lambda timeout_seconds=10.0: _mock_fetcher(pages)
    )
    workers, frontiers, materialized_dir, env, _ = _bootstrap_territories(
        _example(), tmp_path, nav_budget=15, max_candidate_workers=40, timeout_seconds=10.0
    )
    assert len(workers) == 21  # root + all 20 candidates -- creation is free
    assert env.step_count() == 0
    assert all(not f.taken_first_hop for wid, f in frontiers.items() if wid != "worker-root")


def test_v2_b_one_dispatch_can_traverse_a_full_deep_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # root -> section -> listing -> article -- the answer ("evidence") only
    # lives on article, 3 hops from root. A single search() call for one
    # worker must be able to reach it without any artificial hop cap.
    pages = {
        "http://x.test/": "<a href='/section'>Section</a>",
        "http://x.test/section": "<a href='/listing'>Listing</a>",
        "http://x.test/listing": "<a href='/article'>Article</a>",
        "http://x.test/article": "the answer contains rare_evidence_token here",
    }
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _ScriptedDecisionProvider(
        [
            _none_continue(0),  # evaluating "section": not resolved, follow "listing"
            _none_continue(0),  # evaluating "listing": not resolved, follow "article"
            _resolved("evidence", "the answer contains rare_evidence_token here"),
        ]
    )
    tool = WebSearchTool(materialized, env, provider, frontiers)

    results = tool.search("rare_evidence_token", ["worker-0__root.txt"])

    assert results
    assert env.step_count() == 3  # section, listing, article -- all in ONE dispatch
    assert [e["url"] for e in tool.nav_log] == [
        "http://x.test/section",
        "http://x.test/listing",
        "http://x.test/article",
    ]
    assert results[0].quote == "the answer contains rare_evidence_token here"


def test_v2_c_a_later_dispatch_resumes_from_the_saved_frontier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://x.test/": "<a href='/section'>Section</a>",
        "http://x.test/section": "<a href='/listing'>Listing</a>",
        "http://x.test/listing": "<a href='/article'>Article</a>",
        "http://x.test/article": "the real evidence is here",
    }
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    # First dispatch: continues past "section" to "listing", then the
    # decision call evaluating "listing" itself comes back malformed
    # (simulating a transient failure) -- the dispatch ends there, NOT
    # because of any progress/action judgment, with the frontier saved
    # at "listing".
    provider = _ScriptedDecisionProvider(
        [
            _none_continue(0),  # evaluating "section": follow "listing"
            "not valid json -- simulated decision-call failure",
            _none_continue(0),  # second dispatch, evaluating "listing" again: follow "article"
            _resolved("evidence", "the real evidence is here"),
        ]
    )
    tool = WebSearchTool(materialized, env, provider, frontiers)
    files = ["worker-0__root.txt"]

    tool.search("q", files)
    assert env.step_count() == 2
    assert frontiers["worker-0"].current_page.url == "http://x.test/listing"

    # Second dispatch (a later round, same worker, different need): must
    # continue from "listing", NOT restart at root -- it should never
    # re-navigate to section or listing again, and must reach article.
    provider.prompts.clear()
    results = tool.search("q", files)

    assert env.step_count() == 3  # exactly one more hop, not a restart
    urls = [e["url"] for e in tool.nav_log]
    assert urls == ["http://x.test/section", "http://x.test/listing", "http://x.test/article"]
    assert provider.prompts  # the second dispatch's own decision call(s)
    assert "listing" in provider.prompts[0]  # offered from the SAVED current page
    assert results[0].quote == "the real evidence is here"


def test_v2_d_only_activated_workers_ever_consume_navigation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {"http://x.test/": "".join(f"<a href='/p{i}'>P{i}</a>" for i in range(20))}
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    frontiers = {}
    for i, link in enumerate(root.links):
        worker_id = f"worker-{i}"
        (materialized / f"{worker_id}__root.txt").write_text("nothing", encoding="utf-8")
        frontiers[worker_id] = WorkerFrontier(current_page=root, assigned_link=link)
    tool = WebSearchTool(materialized, env, _ScriptedDecisionProvider([_dead_end()]), frontiers)

    # Coordinator "activates" only worker-0 and worker-5.
    tool.search("q", ["worker-0__root.txt"])
    tool.search("q", ["worker-5__root.txt"])

    assert env.step_count() == 2
    activated = {"worker-0", "worker-5"}
    for worker_id, frontier in frontiers.items():
        if worker_id in activated:
            assert frontier.taken_first_hop is True
        else:
            assert frontier.taken_first_hop is False


def test_v2_e_global_budget_never_exceeded_across_many_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {"http://x.test/": "".join(f"<a href='/p{i}'>P{i}</a>" for i in range(20))}
    pages.update({f"http://x.test/p{i}": f"<a href='/p{i}b'>more</a>" for i in range(20)})
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/", max_steps=15)
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    frontiers = {}
    for i, link in enumerate(root.links):
        worker_id = f"worker-{i}"
        (materialized / f"{worker_id}__root.txt").write_text("nothing", encoding="utf-8")
        frontiers[worker_id] = WorkerFrontier(current_page=root, assigned_link=link)
    provider = _ScriptedDecisionProvider([_none_continue(0)])  # always tries to keep navigating
    tool = WebSearchTool(materialized, env, provider, frontiers)

    for worker_id in list(frontiers):
        tool.search("q", [f"{worker_id}__root.txt"])
        assert env.step_count() <= 15

    assert env.step_count() == 15
    assert len(tool.nav_log) == 15


# --- Web ANTMAN Fix v5's own required structural tests: the two-field
# progress/action schema (see web_navigation_tool.py's own module
# docstring). ---


def test_v5_a_no_useful_evidence_yields_none_continue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://x.test/": "<a href='/a'>A</a>",
        "http://x.test/a": "nothing relevant here at all",
    }
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _ScriptedDecisionProvider([_dead_end()])  # ends the dispatch so we can inspect
    tool = WebSearchTool(materialized, env, provider, frontiers)

    tool.search("irrelevant need", ["worker-0__root.txt"])

    # The mandatory hop reaches "/a" with nothing useful -- the decision
    # for evaluating "/a" (the only real LLM call here) must have been
    # none+continue/dead_end, never claiming any evidence.
    assert tool.decision_log[0]["progress"] == "none"
    assert tool.decision_log[0]["n_evidence_proposed"] == 0


def test_v5_b_one_useful_but_insufficient_fact_yields_partial_continue(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://x.test/": "<a href='/a'>A</a>",
        "http://x.test/a": "<a href='/b'>B</a>the update happened in October 2024",
        "http://x.test/b": "the exact percentage is 20%",
    }
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _ScriptedDecisionProvider(
        [
            _partial_continue(0, "update timing", "the update happened in October 2024"),
            _resolved("percentage", "the exact percentage is 20%"),
        ]
    )
    tool = WebSearchTool(materialized, env, provider, frontiers)

    results = tool.search("what percentage and when", ["worker-0__root.txt"])

    assert tool.decision_log[0]["progress"] == "partial"
    assert tool.decision_log[0]["action"] == "continue"
    assert tool.decision_log[0]["n_evidence_grounded"] == 1
    # The partial fact from hop 1 must survive into the final result
    # alongside hop 2's resolved fact -- "preserved across later hops".
    quotes = {r.quote for r in results}
    assert "the update happened in October 2024" in quotes
    assert "the exact percentage is 20%" in quotes


def test_v5_c_partial_evidence_preserved_across_later_hops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Three hops, each contributing one grounded partial fact, none of
    # them individually resolving the Need -- the worker eventually
    # judges the local path exhausted and returns everything it found.
    pages = {
        "http://x.test/": "<a href='/a'>A</a>",
        "http://x.test/a": "<a href='/b'>B</a>fact one is here",
        "http://x.test/b": "<a href='/c'>C</a>fact two is here",
        "http://x.test/c": "fact three is here",
    }
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _ScriptedDecisionProvider(
        [
            _partial_continue(0, "fact one", "fact one is here"),
            _partial_continue(0, "fact two", "fact two is here"),
            _partial_return("fact three", "fact three is here"),
        ]
    )
    tool = WebSearchTool(materialized, env, provider, frontiers)

    results = tool.search("gather all three facts", ["worker-0__root.txt"])

    assert env.step_count() == 3
    quotes = {r.quote for r in results}
    assert quotes == {"fact one is here", "fact two is here", "fact three is here"}


def test_v5_d_partial_return_stops_without_claiming_full_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {"http://x.test/": "<a href='/a'>A</a>", "http://x.test/a": "the venue is Boston"}
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _ScriptedDecisionProvider([_partial_return("venue", "the venue is Boston")])
    tool = WebSearchTool(materialized, env, provider, frontiers)

    results = tool.search("what venue and what date", ["worker-0__root.txt"])

    assert env.step_count() == 1  # returned immediately, did not keep navigating
    assert len(results) == 1
    assert results[0].quote == "the venue is Boston"
    assert results[0].claim == "venue"
    # The tool itself never claims full resolution here -- that judgment
    # belongs to ANTMAN's own existing check_need_resolution, operating
    # on this grounded partial evidence plus whatever else accumulates
    # for this Need across later rounds/dispatches (untouched core
    # machinery, not re-implemented here).
    assert tool.decision_log[-1]["progress"] == "partial"


def test_v5_e_multiple_grounded_partial_pieces_accumulate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://x.test/": "<a href='/a'>A</a>",
        "http://x.test/a": "<a href='/b'>B</a>speaker is Dr. Jane Smith",
        "http://x.test/b": "the session starts at 9am",
    }
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _ScriptedDecisionProvider(
        [
            _partial_continue(0, "speaker", "speaker is Dr. Jane Smith"),
            _partial_return("time", "the session starts at 9am"),
        ]
    )
    tool = WebSearchTool(materialized, env, provider, frontiers)

    results = tool.search("who is speaking and when", ["worker-0__root.txt"])

    assert len(results) == 2
    claims = {r.claim for r in results}
    assert claims == {"speaker", "time"}


def test_v5_f_fully_sufficient_evidence_yields_resolved_return(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://x.test/": "<a href='/a'>A</a>",
        "http://x.test/a": "the full answer: percentage is 20% for attack and hitpoints",
    }
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _ScriptedDecisionProvider(
        [_resolved("full answer", "the full answer: percentage is 20% for attack and hitpoints")]
    )
    tool = WebSearchTool(materialized, env, provider, frontiers)

    tool.search("percentage for attack and hitpoints", ["worker-0__root.txt"])

    assert tool.decision_log[0]["progress"] == "resolved"
    assert tool.decision_log[0]["action"] == "return"
    assert tool.decision_log[0]["n_evidence_grounded"] == 1


def test_v5_g_hallucinated_partial_evidence_fails_verification_but_navigation_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pages = {
        "http://x.test/": "<a href='/a'>A</a>",
        "http://x.test/a": "<a href='/b'>B</a>this page never mentions any percentage",
        "http://x.test/b": "the real percentage is 20%",
    }
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text("nothing", encoding="utf-8")
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _ScriptedDecisionProvider(
        [
            # Claims partial progress with a HALLUCINATED quote (not
            # actually on the page) but still chooses to continue.
            _partial_continue(0, "percentage", "the percentage is definitely 99% (invented)"),
            _resolved("percentage", "the real percentage is 20%"),
        ]
    )
    tool = WebSearchTool(materialized, env, provider, frontiers)

    results = tool.search("what percentage", ["worker-0__root.txt"])

    # The hallucinated span must never appear in the final evidence...
    assert not any("99%" in r.quote or "invented" in r.quote for r in results)
    # ...but the model's own chosen action ("continue") is still honored
    # independently of whether its evidence grounded -- grounding governs
    # evidence inclusion only, never navigation control flow -- so hop 2
    # still happens and its real, grounded fact is captured.
    assert env.step_count() == 2
    assert any(r.quote == "the real percentage is 20%" for r in results)
    assert tool.decision_log[0]["n_evidence_proposed"] == 1
    assert tool.decision_log[0]["n_evidence_grounded"] == 0


def test_v5_h_no_evidence_bypasses_grounding_via_local_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The old ungrounded local-BM25 fallback is gone entirely: even when
    # the materialized files contain text that would trivially match the
    # query via plain keyword search, a dead_end/no-evidence decision
    # must return nothing -- local search may still exist as a tool
    # (used by rank_symbols/resolve_symbol/etc., unrelated pass-throughs)
    # but nothing it finds may enter search()'s own returned evidence
    # pool without passing through the grounded decision + verification
    # path.
    pages = {"http://x.test/": "<a href='/a'>A</a>", "http://x.test/a": "irrelevant"}
    env = _env(tmp_path, monkeypatch, pages, "http://x.test/")
    root = env.root_page()
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    (materialized / "worker-0__root.txt").write_text(
        "the exact keyword the query is looking for", encoding="utf-8"
    )
    frontiers = {"worker-0": WorkerFrontier(current_page=root, assigned_link=root.links[0])}
    provider = _ScriptedDecisionProvider([_dead_end()])
    tool = WebSearchTool(materialized, env, provider, frontiers)

    results = tool.search("the exact keyword the query is looking for", ["worker-0__root.txt"])

    assert results == []
    # Confirm the local tool itself, if called directly, WOULD have found
    # a match -- proving this is a genuine provenance guarantee, not a
    # coincidence of the fixture having no matching text at all.
    direct_local_hits = tool._local.search(
        "the exact keyword the query is looking for", ["worker-0__root.txt"]
    )
    assert direct_local_hits  # the old code path would have returned this


# --- full run() wiring, deterministic end-to-end (no API key) ---


class _StubProvider(MockLLMProvider):
    """MockLLMProvider (LocalCoordinator's own default reasoner fallback)
    plus the small extra surface AntWebAgent.run() and WebSearchTool need:
    synthesize/synthesize_coalition (MockLLMProvider itself only
    implements the reasoner side), drain_call_count/drain_usage/
    drain_retry_log (the CountingOpenAIProvider surface), and
    responses_text (WebSearchTool's own direct grounded-decision call).

    A full agent.run() dispatches MULTIPLE workers, each independently
    entering the decision loop -- which worker's search() gets called
    first, second, etc. is Need-Graph/routing-determined, not something a
    test controls. A shared position-based script (like
    _ScriptedDecisionProvider, fine for single-worker WebSearchTool-level
    tests) is therefore the wrong tool here: entry N could get consumed
    by evaluating the WRONG worker's page. `decide_fn` instead is a pure
    function of the PROMPT TEXT itself (which embeds the current page's
    own content), so the right decision is returned no matter which
    worker or what order asks for it. Defaults to always dead_end -- the
    safest minimal-navigation default for tests that don't care about the
    exact decision.
    """

    def __init__(self, decide_fn=None) -> None:
        self._decide_fn = decide_fn or (lambda prompt: _dead_end())
        self.responses_text_calls = 0

    def synthesize(self, *, question, evidence, **kwargs):
        return f"mock answer for: {question}" if evidence else ""

    def synthesize_coalition(self, *, question, worker_ids, evidence, **kwargs):
        return f"mock coalition answer for: {question}" if evidence else ""

    def responses_text(self, prompt: str, max_output_tokens: int = 16):
        from ant.providers.openai_provider import ResponseResult

        self.responses_text_calls += 1
        decision = self._decide_fn(prompt)
        text = decision if isinstance(decision, str) else json.dumps(decision)
        return ResponseResult(text=text, usage=TokenUsage(), raw={})

    def drain_call_count(self) -> int:
        return self.responses_text_calls

    def drain_usage(self) -> TokenUsage:
        return TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2, estimated_cost_usd=0.0)

    def drain_retry_log(self) -> list[dict]:
        return []


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    pages: dict[str, str],
    agent=None,
    decide_fn=None,
):
    monkeypatch.setattr(
        ant_web_module, "urllib_fetcher", lambda timeout_seconds=10.0: _mock_fetcher(pages)
    )
    monkeypatch.setattr(
        ant_web_module, "CountingOpenAIProvider", lambda model: _StubProvider(decide_fn)
    )
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
    # navigating (never says dead_end) -- the assertion inside
    # AntWebAgent.run() itself (env.step_count() <= nav_budget) is the
    # real guard; this test just exercises a path likely to stress it.
    links = "".join(f"<a href='/p{i}'>P{i}</a>" for i in range(20))
    pages = {"http://conf.example.com/": links}
    pages.update({f"http://conf.example.com/p{i}": f"<a href='/p{i}b'>more</a>" for i in range(20)})
    agent = AntWebAgent(nav_budget=5, max_rounds=3, max_candidate_workers=20)
    result = _run(
        monkeypatch, tmp_path, pages, agent=agent, decide_fn=lambda prompt: _none_continue(0)
    )
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
    answer_text = "Lipizzaner Cavalry increases attack and hitpoints of Uhlans by 20%."

    def _decide_fn(prompt: str) -> dict:
        # Content-aware, not order-based: multiple workers (worker-root,
        # worker-0) independently enter the decision loop in an order
        # this test does not control, so the right decision must be a
        # pure function of which page's content the prompt actually
        # shows, never a shared queue position.
        if answer_text in prompt:
            return _resolved("percentage", answer_text)
        return _none_continue(0)

    monkeypatch.setattr(
        ant_web_module, "CountingOpenAIProvider", lambda model: _StubProvider(_decide_fn)
    )
    agent = AntWebAgent()
    result = agent.run(_example(), tmp_path)

    # The bootstrap roster itself must NOT contain the answer page --
    # that's the whole point (matching the real bug: it's 2 hops deep).
    assert result.metadata["territories_discovered"] == 2  # root + release-notes only
    # But the live run must have actually reached it via real navigation.
    assert result.metadata["navigation_steps"] >= 2
    assert any("update-19" in entry["url"] for entry in result.metadata["nav_log"])
