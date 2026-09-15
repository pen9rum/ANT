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
    PageLink,
    _ensure_ascii_url,
    _extract_meta_refresh_target,
    _looks_like_text_content,
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
    assert [(link.text, link.url) for link in links] == [
        ("A", "http://example.com/a"),  # first-seen anchor text kept ("A", not "A again")
        ("B relative", "http://example.com/dir/b"),
    ]


def test_extract_text_and_links_captures_anchor_text() -> None:
    html = '<a href="/speakers">Meet the Speakers</a>'
    _text, links = extract_text_and_links(html, base_url="http://example.com/")
    assert len(links) == 1
    assert links[0].text == "Meet the Speakers"
    assert links[0].url == "http://example.com/speakers"


def test_extract_text_and_links_handles_link_with_no_text() -> None:
    html = '<a href="/icon"><img src="x.png"/></a>'
    _text, links = extract_text_and_links(html, base_url="http://example.com/")
    assert len(links) == 1
    assert links[0].text == ""
    assert links[0].url == "http://example.com/icon"


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


# --- binary/non-text content detection (regression: a real WebWalkerQA
# gold source page served binary content that crashed html.parser with an
# uncaught AssertionError instead of a clean, catchable error) ---


def test_looks_like_text_content_accepts_plain_html() -> None:
    assert _looks_like_text_content("text/html; charset=utf-8", b"<p>hello</p>")


def test_looks_like_text_content_rejects_image_content_type() -> None:
    assert not _looks_like_text_content("image/png", b"\x89PNG\r\n\x1a\n")


def test_looks_like_text_content_rejects_pdf_content_type() -> None:
    assert not _looks_like_text_content("application/pdf", b"%PDF-1.4")


def test_looks_like_text_content_rejects_nul_byte_even_with_no_content_type() -> None:
    assert not _looks_like_text_content("", b"garbled\x00binary\x00data")


def test_looks_like_text_content_accepts_empty_content_type_with_no_nul_byte() -> None:
    assert _looks_like_text_content("", b"plain text, no content-type header")


def test_urllib_fetcher_rejects_non_text_response_instead_of_crashing() -> None:
    mock_response = MagicMock()
    mock_response.geturl.return_value = "https://example.com/file.pdf"
    mock_response.status = 200
    mock_response.read.return_value = b"%PDF-1.4 binary garbage"
    mock_response.headers = {"Content-Type": "application/pdf"}
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False

    with patch("urllib.request.urlopen", return_value=mock_response):
        fetcher = urllib_fetcher()
        result = fetcher("https://example.com/file.pdf")

    assert result.error is not None
    assert "non-text content" in result.error
    assert result.html == ""


# --- meta-refresh client-side redirect (regression: a real WebWalkerQA
# root URL, ciie.org, serves nothing but a meta-refresh stub -- every
# major browser follows this automatically, confirmed via caniuse.com;
# the official WebWalker environment's own Playwright-rendered pipeline
# would too, so a plain HTTP client not following it is a genuine
# environment fidelity gap, not a benchmark-specific heuristic) ---


def test_extract_meta_refresh_target_finds_url() -> None:
    html = '<html><head><meta http-equiv="refresh" content="0;url=zbh/index.html"></head></html>'
    target = _extract_meta_refresh_target(html, base_url="https://www.ciie.org/")
    assert target == "https://www.ciie.org/zbh/index.html"


def test_extract_meta_refresh_target_handles_quoted_url_and_spacing() -> None:
    html = '<meta http-equiv="refresh" content="5; url=\'/next-page\'" />'
    target = _extract_meta_refresh_target(html, base_url="https://example.com/")
    assert target == "https://example.com/next-page"


def test_extract_meta_refresh_target_ignores_non_refresh_meta_tags() -> None:
    html = '<meta charset="utf-8"><meta name="description" content="0;url=/should-not-match">'
    assert _extract_meta_refresh_target(html, base_url="https://example.com/") is None


def test_extract_meta_refresh_target_ignores_refresh_with_no_url() -> None:
    html = '<meta http-equiv="refresh" content="30">'  # pure timed reload, no redirect
    assert _extract_meta_refresh_target(html, base_url="https://example.com/") is None


def test_urllib_fetcher_follows_meta_refresh_end_to_end() -> None:
    stub_response = MagicMock()
    stub_response.geturl.return_value = "https://www.ciie.org/"
    stub_response.status = 200
    stub_response.read.return_value = b'<meta http-equiv="refresh" content="0;url=zbh/index.html">'
    stub_response.headers = {"Content-Type": "text/html; charset=utf-8"}
    stub_response.__enter__.return_value = stub_response
    stub_response.__exit__.return_value = False

    final_response = MagicMock()
    final_response.geturl.return_value = "https://www.ciie.org/zbh/index.html"
    final_response.status = 200
    final_response.read.return_value = b"<p>Real page content</p>"
    final_response.headers = {"Content-Type": "text/html; charset=utf-8"}
    final_response.__enter__.return_value = final_response
    final_response.__exit__.return_value = False

    with patch("urllib.request.urlopen", side_effect=[stub_response, final_response]):
        fetcher = urllib_fetcher()
        result = fetcher("https://www.ciie.org/")

    assert result.error is None
    assert "Real page content" in result.html
    assert result.redirect_target == "https://www.ciie.org/zbh/index.html"


def test_urllib_fetcher_meta_refresh_chain_has_a_hop_limit() -> None:
    # A genuinely alternating A<->B cycle -- a literal self-redirect (A
    # refreshing to itself) is intentionally treated as "arrived, no
    # further hop needed" rather than looping, so this uses two distinct
    # URLs to actually exercise the hop-count limit.
    def _response_for(url: str):
        target = "/b" if url.endswith("/a") else "/a"
        response = MagicMock()
        response.geturl.return_value = f"https://example.com{'/a' if target == '/b' else '/b'}"
        response.status = 200
        response.read.return_value = (
            f'<meta http-equiv="refresh" content="0;url={target}">'.encode()
        )
        response.headers = {"Content-Type": "text/html; charset=utf-8"}
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        return response

    with patch("urllib.request.urlopen", side_effect=lambda req, **k: _response_for(req.full_url)):
        fetcher = urllib_fetcher()
        result = fetcher("https://example.com/a")

    assert result.error is not None
    assert "meta-refresh chain exceeded" in result.error


def test_page_cache_degrades_to_error_when_html_parsing_itself_raises(tmp_path: Path) -> None:
    # Simulates the exact live failure: content that passes the
    # content-type/NUL-byte screen (urllib_fetcher's own check) but still
    # trips html.parser internally -- PageCache's own defense-in-depth
    # must catch it, never let it propagate.
    def _fetcher(url: str) -> _RawFetch:
        return _RawFetch(
            http_status=200, redirect_target=None, html="<p>looks fine</p>", error=None
        )

    cache = PageCache(cache_dir=tmp_path, fetcher=_fetcher, root_url="http://example.com/")
    with patch(
        "ant.evaluation_suite.web_fetch.extract_text_and_links",
        side_effect=AssertionError("expected name token at '<![garbled'"),
    ):
        page = cache.get("http://example.com/")

    assert page.status == "error"
    assert "HTML parse failure" in page.error


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
    assert [(link.text, link.url) for link in page.links] == [("A", "http://example.com/a")]
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
        links=[PageLink(text="A", url="http://example.com/a")],
        error=None,
        fetched_at="2026-01-01T00:00:00+00:00",
    )
    restored = FetchedPage.from_json(page.to_json())
    assert restored.url == page.url
    assert restored.text == page.text
    assert restored.links == page.links


def test_fetched_page_from_json_accepts_legacy_bare_url_links() -> None:
    # A cache entry written before PageLink existed (e.g. the original
    # 11.7% run's own on-disk cache) stored links as bare URL strings --
    # loading it must not crash.
    data = {
        "url": "http://example.com/",
        "requested_url": "http://example.com/",
        "status": "ok",
        "http_status": 200,
        "redirect_target": None,
        "text": "hello",
        "links": ["http://example.com/a"],
        "error": None,
        "fetched_at": "2026-01-01T00:00:00+00:00",
    }
    restored = FetchedPage.from_json(data)
    assert restored.links == [PageLink(text="", url="http://example.com/a")]
