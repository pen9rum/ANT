"""Tests for ant.evaluation_suite.web_fetch -- URL normalization,
same-site restriction, deterministic HTML extraction, and the disk-backed
page cache. Every fetch here is a static, in-memory mock -- zero real
network calls, zero LLM calls, per the governing spec's "zero paid
inference, use static/mock webpages for tests" requirement.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from ant.evaluation_suite.web_fetch import (
    FetchedPage,
    PageCache,
    _ensure_ascii_url,
    _RawFetch,
    extract_text_and_links,
    normalize_url,
    same_site,
    urllib_fetcher,
)

# --- URL normalization ---


def test_normalize_url_lowercases_scheme_and_host() -> None:
    assert normalize_url("HTTP://Example.COM/Path") == "http://example.com/Path"


def test_normalize_url_strips_default_ports() -> None:
    assert normalize_url("http://example.com:80/a") == normalize_url("http://example.com/a")
    assert normalize_url("https://example.com:443/a") == normalize_url("https://example.com/a")


def test_normalize_url_keeps_non_default_port() -> None:
    assert normalize_url("http://example.com:8080/a") == "http://example.com:8080/a"


def test_normalize_url_strips_fragment() -> None:
    assert normalize_url("http://example.com/a#section") == normalize_url("http://example.com/a")


def test_normalize_url_strips_trailing_slash_except_root() -> None:
    assert normalize_url("http://example.com/a/") == normalize_url("http://example.com/a")
    assert normalize_url("http://example.com/") == "http://example.com/"
    assert normalize_url("http://example.com") == "http://example.com/"


def test_normalize_url_sorts_query_params() -> None:
    assert normalize_url("http://example.com/a?b=2&a=1") == normalize_url(
        "http://example.com/a?a=1&b=2"
    )


def test_normalize_url_resolves_relative_against_base() -> None:
    assert (
        normalize_url("/sub/page", base_url="http://example.com/x") == "http://example.com/sub/page"
    )
    assert (
        normalize_url("page2", base_url="http://example.com/sub/page1")
        == "http://example.com/sub/page2"
    )


# --- same-site restriction ---


def test_same_site_matches_identical_host() -> None:
    assert same_site("http://example.com/a", "http://example.com/")


def test_same_site_matches_www_variant() -> None:
    assert same_site("http://www.example.com/a", "http://example.com/")
    assert same_site("http://example.com/a", "http://www.example.com/")


def test_same_site_rejects_different_host() -> None:
    assert not same_site("http://evil.com/a", "http://example.com/")


def test_same_site_rejects_unrelated_subdomain() -> None:
    assert not same_site("http://evil.example.com/a", "http://example.com/")


# --- deterministic HTML extraction ---


def test_extract_text_and_links_pulls_visible_text_only() -> None:
    html = """
    <html><head><style>.x{color:red}</style><script>var x=1;</script></head>
    <body><h1>Title</h1><p>Some visible text.</p></body></html>
    """
    text, links = extract_text_and_links(html, base_url="http://example.com/")
    assert "Title" in text
    assert "Some visible text." in text
    assert "color:red" not in text
    assert "var x=1" not in text


def test_extract_text_and_links_resolves_and_dedupes_links() -> None:
    html = """
    <a href="/a">A</a>
    <a href="/a">A again</a>
    <a href="b">B relative</a>
    <a href="javascript:void(0)">JS</a>
    <a href="mailto:x@example.com">Mail</a>
    """
    text, links = extract_text_and_links(html, base_url="http://example.com/dir/")
    assert links == ["http://example.com/a", "http://example.com/dir/b"]


def test_extract_text_and_links_is_deterministic() -> None:
    html = "<p>Hello</p><a href='/x'>x</a><a href='/y'>y</a>"
    r1 = extract_text_and_links(html, base_url="http://example.com/")
    r2 = extract_text_and_links(html, base_url="http://example.com/")
    assert r1 == r2


# --- ASCII/IRI-to-URI conversion (regression: real Chinese-hosted gold
# source pages in WebWalkerQA embed raw non-ASCII characters directly in
# the URL path -- confirmed live, this used to crash the fetcher with an
# uncaught UnicodeEncodeError deep inside http.client) ---


def test_ensure_ascii_url_percent_encodes_non_ascii_path() -> None:
    url = "https://example.com/新闻;jsessionid=abc"
    ascii_url = _ensure_ascii_url(url)
    ascii_url.encode("ascii")  # must not raise
    assert "%" in ascii_url


def test_ensure_ascii_url_leaves_already_ascii_url_unchanged() -> None:
    url = "https://example.com/a/b?x=1&y=2"
    assert _ensure_ascii_url(url) == url


def test_ensure_ascii_url_idna_encodes_non_ascii_host() -> None:
    ascii_url = _ensure_ascii_url("https://例え.jp/path")
    ascii_url.encode("ascii")  # must not raise
    assert "xn--" in ascii_url


def test_urllib_fetcher_sends_only_ascii_urls_to_urlopen() -> None:
    # End-to-end regression check for the real bug (not just the helper in
    # isolation): urlopen itself is mocked out, so this exercises
    # urllib_fetcher()'s own wiring of _ensure_ascii_url without any real
    # network call.
    mock_response = MagicMock()
    mock_response.geturl.return_value = "https://example.com/%E6%96%B0%E9%97%BB"
    mock_response.status = 200
    mock_response.read.return_value = b"<p>hi</p>"
    mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        fetcher = urllib_fetcher()
        result = fetcher("https://example.com/新闻")

    assert result.error is None
    sent_request = mock_urlopen.call_args[0][0]
    sent_request.full_url.encode("ascii")  # must not raise -- this is the regression


# --- PageCache ---


def _mock_fetcher(pages: dict[str, str]):
    """Static mock fetcher over a dict of {normalized_url: html}. Any URL
    not in `pages` reports an error (simulating a dead/inaccessible page)."""

    def _fetch(url: str) -> _RawFetch:
        if url in pages:
            return _RawFetch(http_status=200, redirect_target=None, html=pages[url], error=None)
        return _RawFetch(
            http_status=404, redirect_target=None, html="", error="HTTPError: 404 Not Found"
        )

    return _fetch


def test_page_cache_fetches_and_caches_on_disk(tmp_path: Path) -> None:
    pages = {"http://example.com/": "<p>Root</p><a href='/a'>A</a>"}
    cache = PageCache(
        cache_dir=tmp_path, fetcher=_mock_fetcher(pages), root_url="http://example.com/"
    )

    page = cache.get("http://example.com/")

    assert page.status == "ok"
    assert "Root" in page.text
    assert page.links == ["http://example.com/a"]
    assert not page.from_cache
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_page_cache_reuses_cache_without_refetching(tmp_path: Path) -> None:
    calls: list[str] = []

    def _counting_fetcher(url: str) -> _RawFetch:
        calls.append(url)
        return _RawFetch(http_status=200, redirect_target=None, html="<p>Hi</p>", error=None)

    cache = PageCache(cache_dir=tmp_path, fetcher=_counting_fetcher, root_url="http://example.com/")
    cache.get("http://example.com/")
    cache.get("http://example.com/")  # memory-hit
    second_cache = PageCache(
        cache_dir=tmp_path, fetcher=_counting_fetcher, root_url="http://example.com/"
    )
    page = second_cache.get("http://example.com/")  # disk-hit (fresh instance)

    assert len(calls) == 1
    assert page.from_cache


def test_page_cache_deduplicates_urls_differing_only_by_fragment(tmp_path: Path) -> None:
    calls: list[str] = []

    def _counting_fetcher(url: str) -> _RawFetch:
        calls.append(url)
        return _RawFetch(http_status=200, redirect_target=None, html="<p>Hi</p>", error=None)

    cache = PageCache(cache_dir=tmp_path, fetcher=_counting_fetcher, root_url="http://example.com/")
    cache.get("http://example.com/page")
    cache.get("http://example.com/page#section2")

    assert len(calls) == 1


def test_page_cache_records_inaccessible_page_distinctly(tmp_path: Path) -> None:
    cache = PageCache(cache_dir=tmp_path, fetcher=_mock_fetcher({}), root_url="http://example.com/")

    page = cache.get("http://example.com/dead")

    assert page.status == "error"
    assert page.http_status == 404
    assert page.error is not None
    assert page.text == ""


def test_page_cache_distinguishes_error_from_genuinely_empty_page(tmp_path: Path) -> None:
    pages = {"http://example.com/empty": ""}
    cache = PageCache(
        cache_dir=tmp_path, fetcher=_mock_fetcher(pages), root_url="http://example.com/"
    )

    page = cache.get("http://example.com/empty")

    assert page.status == "ok"  # genuinely fetched, just empty -- NOT "error"
    assert page.text == ""


def test_page_cache_without_fetcher_reports_not_fetched(tmp_path: Path) -> None:
    cache = PageCache(cache_dir=tmp_path, fetcher=None)

    page = cache.get("http://example.com/")

    assert page.status == "not_fetched"


def test_page_cache_rejects_urls_outside_the_root_site(tmp_path: Path) -> None:
    calls: list[str] = []

    def _counting_fetcher(url: str) -> _RawFetch:
        calls.append(url)
        return _RawFetch(http_status=200, redirect_target=None, html="<p>Hi</p>", error=None)

    cache = PageCache(cache_dir=tmp_path, fetcher=_counting_fetcher, root_url="http://example.com/")

    page = cache.get("http://other-site.com/page")

    assert page.status == "error"
    assert "outside the permitted root-site domain" in page.error
    assert calls == []  # never even attempted to fetch


def test_page_cache_records_redirect_target(tmp_path: Path) -> None:
    def _fetcher(url: str) -> _RawFetch:
        return _RawFetch(
            http_status=200,
            redirect_target="http://example.com/final",
            html="<p>Hi</p>",
            error=None,
        )

    cache = PageCache(cache_dir=tmp_path, fetcher=_fetcher, root_url="http://example.com/")

    page = cache.get("http://example.com/old")

    assert page.redirect_target == "http://example.com/final"


def test_fetched_page_json_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    page = FetchedPage(
        url="http://example.com/",
        requested_url="http://example.com",
        status="ok",
        http_status=200,
        redirect_target=None,
        text="hello",
        links=["http://example.com/a"],
        error=None,
        fetched_at="2026-01-01T00:00:00+00:00",
    )
    restored = FetchedPage.from_json(page.to_json())
    assert restored.url == page.url
    assert restored.text == page.text
    assert restored.links == page.links
