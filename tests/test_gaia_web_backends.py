"""Tests for GAIA's real-network search/fetch backends. No real network
call in any test here -- `urllib.request.urlopen` is monkeypatched with a
static, in-memory response, matching this suite's existing convention
for `web_fetch.py` (see `tests/test_web_fetch.py`)."""

from __future__ import annotations

import io
import json
from urllib import error as urllib_error

import pytest

from ant.evaluation_suite.gaia_web_backends import (
    DuckDuckGoSearchBackend,
    FallbackSearchBackend,
    GaiaWebBackendError,
    TavilySearchBackend,
    _clean_duckduckgo_redirect_url,
    build_default_search_backend,
)


class _FakeHTTPResponse(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200):
        super().__init__(body)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def geturl(self) -> str:
        return "https://example.invalid/"


def _patch_urlopen(monkeypatch, response_body: bytes, *, raise_http_error: int | None = None):
    def fake_urlopen(request, timeout=None):
        if raise_http_error is not None:
            raise urllib_error.HTTPError(
                request.full_url, raise_http_error, "boom", {}, io.BytesIO(b"error detail")
            )
        return _FakeHTTPResponse(response_body)

    monkeypatch.setattr(
        "ant.evaluation_suite.gaia_web_backends.urllib_request.urlopen", fake_urlopen
    )


class TestTavilySearchBackend:
    def test_requires_a_non_empty_key(self):
        with pytest.raises(ValueError):
            TavilySearchBackend(api_key="")

    def test_maps_results_to_search_hits(self, monkeypatch):
        body = json.dumps(
            {
                "results": [
                    {"title": "A", "url": "https://a.example", "content": "snippet a"},
                    {"title": "B", "url": "https://b.example", "content": "snippet b"},
                ]
            }
        ).encode("utf-8")
        _patch_urlopen(monkeypatch, body)
        backend = TavilySearchBackend(api_key="tvly-test")
        hits = backend.search("query", 5)
        assert [h.title for h in hits] == ["A", "B"]
        assert [h.url for h in hits] == ["https://a.example", "https://b.example"]
        assert [h.snippet for h in hits] == ["snippet a", "snippet b"]

    def test_http_error_raises_gaia_web_backend_error(self, monkeypatch):
        _patch_urlopen(monkeypatch, b"", raise_http_error=401)
        backend = TavilySearchBackend(api_key="tvly-test")
        with pytest.raises(GaiaWebBackendError):
            backend.search("query", 5)


class TestDuckDuckGoResultParsing:
    def test_parses_real_shaped_result_markup(self, monkeypatch):
        href = (
            "//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.example.org%2Fpage"
            "&amp;rut=x"
        )
        html = f"""
        <div class="result">
          <a class="result__a" href="{href}">
            Example Title
          </a>
          <a class="result__snippet">This is the snippet text.</a>
        </div>
        """.encode()
        _patch_urlopen(monkeypatch, html)
        backend = DuckDuckGoSearchBackend()
        hits = backend.search("query", 5)
        assert len(hits) == 1
        assert hits[0].title.strip() == "Example Title"
        assert hits[0].url == "https://en.example.org/page"
        assert hits[0].snippet.strip() == "This is the snippet text."

    def test_respects_limit(self, monkeypatch):
        one_result = (
            '<a class="result__a" href="https://x{i}.example">T{i}</a>'
            '<a class="result__snippet">S{i}</a>'
        )
        html = "".join(one_result.format(i=i) for i in range(10)).encode("utf-8")
        _patch_urlopen(monkeypatch, html)
        backend = DuckDuckGoSearchBackend()
        hits = backend.search("query", 3)
        assert len(hits) == 3

    def test_no_matches_returns_empty_not_an_error(self, monkeypatch):
        _patch_urlopen(monkeypatch, b"<html><body>no results here</body></html>")
        backend = DuckDuckGoSearchBackend()
        assert backend.search("query", 5) == []


class TestRedirectUnwrap:
    def test_unwraps_uddg_redirect(self):
        wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa%3Fb%3Dc&rut=abc"
        assert _clean_duckduckgo_redirect_url(wrapped) == "https://example.org/a?b=c"

    def test_passes_through_a_plain_url_unchanged(self):
        assert _clean_duckduckgo_redirect_url("https://example.org/plain") == (
            "https://example.org/plain"
        )


class TestFallbackSearchBackend:
    class _StubBackend:
        def __init__(self, hits=None, exc=None):
            self._hits = hits or []
            self._exc = exc
            self.calls = 0

        def search(self, query, limit):
            self.calls += 1
            if self._exc:
                raise self._exc
            return self._hits

    def test_uses_primary_when_it_succeeds(self):
        primary = self._StubBackend(hits=["hit-from-primary"])
        secondary = self._StubBackend(hits=["hit-from-secondary"])
        backend = FallbackSearchBackend(primary=primary, secondary=secondary)
        assert backend.search("q", 5) == ["hit-from-primary"]
        assert backend.last_used == "primary"
        assert backend.fallback_count == 0
        assert secondary.calls == 0

    def test_falls_back_when_primary_raises(self):
        primary = self._StubBackend(exc=RuntimeError("primary down"))
        secondary = self._StubBackend(hits=["hit-from-secondary"])
        backend = FallbackSearchBackend(primary=primary, secondary=secondary)
        assert backend.search("q", 5) == ["hit-from-secondary"]
        assert backend.last_used == "secondary"
        assert backend.fallback_count == 1

    def test_raises_gaia_web_backend_error_when_both_fail(self):
        primary = self._StubBackend(exc=RuntimeError("primary down"))
        secondary = self._StubBackend(exc=RuntimeError("secondary down too"))
        backend = FallbackSearchBackend(primary=primary, secondary=secondary)
        with pytest.raises(GaiaWebBackendError):
            backend.search("q", 5)


class TestBuildDefaultSearchBackend:
    def test_no_key_gives_duckduckgo_only(self):
        backend = build_default_search_backend(tavily_api_key="")
        assert isinstance(backend, DuckDuckGoSearchBackend)

    def test_key_present_gives_tavily_primary_fallback_backend(self):
        backend = build_default_search_backend(tavily_api_key="tvly-test")
        assert isinstance(backend, FallbackSearchBackend)
        assert isinstance(backend.primary, TavilySearchBackend)
        assert isinstance(backend.secondary, DuckDuckGoSearchBackend)

    def test_reads_env_var_when_key_not_passed_explicitly(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-from-env")
        backend = build_default_search_backend()
        assert isinstance(backend, FallbackSearchBackend)
        assert backend.primary.api_key == "tvly-from-env"

    def test_env_var_absent_gives_duckduckgo_only(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        backend = build_default_search_backend()
        assert isinstance(backend, DuckDuckGoSearchBackend)
