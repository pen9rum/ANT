"""Tests for ant.evaluation_suite.web_scope -- the EvalWebEnvironment/
WebTerritory substrate abstraction. Every fetch is a static mock; zero
real network calls, zero LLM calls.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from ant.evaluation_suite.web_fetch import PageCache, _RawFetch
from ant.evaluation_suite.web_scope import (
    EvalWebEnvironment,
    WebTerritory,
    build_territory_from_discovered,
)


def _mock_fetcher(pages: dict[str, str]):
    def _fetch(url: str) -> _RawFetch:
        if url in pages:
            return _RawFetch(http_status=200, redirect_target=None, html=pages[url], error=None)
        return _RawFetch(
            http_status=404, redirect_target=None, html="", error="HTTPError: 404 Not Found"
        )

    return _fetch


def _env(
    tmp_path: Path, pages: dict[str, str], root: str, *, max_steps: int | None = None
) -> EvalWebEnvironment:
    cache = PageCache(cache_dir=tmp_path, fetcher=_mock_fetcher(pages), root_url=root)
    return EvalWebEnvironment(root, cache, max_steps=max_steps)


# --- root page / navigation ---


def test_root_page_is_available_without_navigation(tmp_path: Path) -> None:
    pages = {"http://example.com/": "<p>Root</p><a href='/a'>A</a>"}
    env = _env(tmp_path, pages, "http://example.com/")

    root = env.root_page()

    assert root.status == "ok"
    assert env.step_count() == 0  # root_page() does not consume a navigation step


def test_navigate_follows_a_real_discovered_link(tmp_path: Path) -> None:
    pages = {
        "http://example.com/": "<p>Root</p><a href='/a'>A</a>",
        "http://example.com/a": "<p>Page A</p>",
    }
    env = _env(tmp_path, pages, "http://example.com/")
    root = env.root_page()

    a = env.navigate(root, "http://example.com/a")

    assert a.status == "ok"
    assert "Page A" in a.text
    assert env.step_count() == 1


def test_navigate_rejects_a_url_not_discovered_on_the_from_page(tmp_path: Path) -> None:
    pages = {
        "http://example.com/": "<p>Root</p>",  # no links at all
        "http://example.com/secret": "<p>Should never be reachable</p>",
    }
    env = _env(tmp_path, pages, "http://example.com/")
    root = env.root_page()

    with pytest.raises(ValueError, match="was not among the links discovered"):
        env.navigate(root, "http://example.com/secret")


def test_navigate_enforces_max_steps_budget(tmp_path: Path) -> None:
    pages = {
        "http://example.com/": "<a href='/a'>A</a>",
        "http://example.com/a": "<a href='/b'>B</a>",
        "http://example.com/b": "<p>B</p>",
    }
    env = _env(tmp_path, pages, "http://example.com/", max_steps=1)
    root = env.root_page()
    a = env.navigate(root, "http://example.com/a")  # step 1, allowed

    with pytest.raises(RuntimeError, match="navigation step budget exhausted"):
        env.navigate(a, "http://example.com/b")  # step 2, budget exhausted


def test_navigate_respects_same_site_restriction_via_the_underlying_cache(tmp_path: Path) -> None:
    pages = {"http://example.com/": "<a href='http://evil.com/x'>Evil</a>"}
    cache = PageCache(
        cache_dir=tmp_path, fetcher=_mock_fetcher(pages), root_url="http://example.com/"
    )
    env = EvalWebEnvironment("http://example.com/", cache)
    root = env.root_page()

    # The link was extracted (present in root.links) but same_site filtering
    # inside PageCache.get() already dropped it before it reached root.links.
    assert "http://evil.com/x" not in root.links


# --- inspect: re-read only, not a second discovery channel ---


def test_inspect_returns_already_visited_page_content(tmp_path: Path) -> None:
    pages = {"http://example.com/": "<p>Root</p>"}
    env = _env(tmp_path, pages, "http://example.com/")
    env.root_page()

    page = env.inspect("http://example.com/")

    assert page.status == "ok"


def test_inspect_of_an_unvisited_url_does_not_fetch(tmp_path: Path) -> None:
    calls: list[str] = []

    def _fetcher(url: str) -> _RawFetch:
        calls.append(url)
        return _RawFetch(http_status=200, redirect_target=None, html="<p>Hi</p>", error=None)

    cache = PageCache(cache_dir=tmp_path, fetcher=_fetcher, root_url="http://example.com/")
    env = EvalWebEnvironment("http://example.com/", cache)

    page = env.inspect("http://example.com/never-visited")

    assert page.status == "not_fetched"
    assert calls == []  # inspect() never triggers a fetch


def test_discovered_pages_lists_only_visited_urls(tmp_path: Path) -> None:
    pages = {
        "http://example.com/": "<a href='/a'>A</a>",
        "http://example.com/a": "<p>A</p>",
    }
    env = _env(tmp_path, pages, "http://example.com/")
    root = env.root_page()
    env.navigate(root, "http://example.com/a")

    assert env.discovered_pages() == ["http://example.com/", "http://example.com/a"]


# --- territory construction: only from already-discovered pages, no gold input possible ---


def test_build_territory_from_discovered_pages(tmp_path: Path) -> None:
    pages = {
        "http://example.com/": "<a href='/a'>A</a>",
        "http://example.com/a": "<p>A</p>",
    }
    env = _env(tmp_path, pages, "http://example.com/")
    root = env.root_page()
    env.navigate(root, "http://example.com/a")

    territory = build_territory_from_discovered(
        "territory-1", anchor_url="http://example.com/a", pages=env.discovered_pages()
    )

    assert isinstance(territory, WebTerritory)
    assert territory.pages == ["http://example.com/", "http://example.com/a"]


def test_build_territory_from_discovered_has_no_gold_field_parameter() -> None:
    # Structural leakage guard: assert the function signature itself has no
    # slot through which golden_path/source_websites/gold_answer could be
    # threaded in -- not just "nothing currently passes it in."
    params = set(inspect.signature(build_territory_from_discovered).parameters)
    for forbidden in ("golden_path", "source_websites", "gold_answer", "answer"):
        assert forbidden not in params


def test_web_territory_dataclass_has_no_gold_field() -> None:
    fields = {f for f in WebTerritory.__dataclass_fields__}
    for forbidden in ("golden_path", "source_websites", "gold_answer", "answer"):
        assert forbidden not in fields
