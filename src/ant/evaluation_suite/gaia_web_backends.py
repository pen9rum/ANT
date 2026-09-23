"""Real network `SearchBackend`/`FetchBackend` implementations for
`ant.agents.gaia_tools.GaiaToolRegistry` -- GAIA's `search`/`open_url`
tools need genuine open-web access (there is no local corpus the way
repo-QA has a git checkout), so unlike every other substrate in this
project this one talks to the real internet.

Tavily is the primary search backend (a real, ranked search API --
GAIA's own difficulty comes from needing to find specific, often
obscure facts, and a weak/unofficial search source would bottleneck
every method equally, making a method comparison measure search luck
instead of reasoning quality). DuckDuckGo's `html.duckduckgo.com/html/`
endpoint is a keyless fallback ONLY for when Tavily is unavailable
(missing key, quota exhausted, transient outage) -- it is explicitly a
degraded mode, not a peer option: it is HTML-scraped (not a documented
API), can break on a markup change with no warning, and carries real
rate-limit/block risk from a shared HPC egress IP. `FallbackSearchBackend`
logs which backend actually answered each query so a run's own trace
shows whenever it was running in degraded mode.

Fetch is `UrllibFetchBackend`, a thin adapter over
`ant.evaluation_suite.web_fetch`'s existing real-network fetcher (already
used by the WebWalkerQA/Track B substrate) -- reused rather than
reimplemented, per this project's own "one real fetcher" convention.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib import error as urllib_error
from urllib import request as urllib_request

from ant.agents.gaia_tools import SearchBackend, SearchHit
from ant.evaluation_suite.web_fetch import extract_text_and_links, urllib_fetcher

DEFAULT_TIMEOUT_SECONDS = 15.0
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
DUCKDUCKGO_HTML_URL = "https://html.duckduckgo.com/html/"
_USER_AGENT = "ANTMAN-GAIA-eval/0.1"


class GaiaWebBackendError(RuntimeError):
    """A search/fetch backend failed. Distinct from `ToolUnavailableError`
    (that's "no backend wired at all") -- this is "a wired backend tried
    and failed", which `GaiaToolRegistry.search`/`open_url` already
    re-raises and logs verbatim."""


@dataclass(frozen=True)
class TavilySearchBackend:
    """Primary search backend: the real Tavily search API.

    `api_key` is required at construction (no silent no-op mode) so a
    missing key fails loudly at wiring time, not as a mysterious empty
    result three rounds into a run.
    """

    api_key: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    search_depth: str = "basic"

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("TavilySearchBackend requires a non-empty api_key")

    def search(self, query: str, limit: int) -> Sequence[SearchHit]:
        payload = json.dumps(
            {
                "query": query,
                "max_results": max(1, min(int(limit), 20)),
                "search_depth": self.search_depth,
            }
        ).encode("utf-8")
        req = urllib_request.Request(
            TAVILY_SEARCH_URL,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": _USER_AGENT,
            },
        )
        try:
            with urllib_request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise GaiaWebBackendError(f"Tavily HTTP {exc.code}: {detail}") from exc
        except urllib_error.URLError as exc:
            raise GaiaWebBackendError(f"Tavily request failed: {exc.reason}") from exc

        results = body.get("results", [])
        return [
            SearchHit(
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                snippet=str(item.get("content", "")),
            )
            for item in results
        ]


class _DuckDuckGoResultParser(HTMLParser):
    """Extracts (title, url, snippet) triples from
    `html.duckduckgo.com/html/`'s response markup. Deliberately tolerant:
    this is scraping an undocumented page, not a stable API, so a
    partial/best-effort parse is the right failure mode -- an unmatched
    tag should never raise.

    Structure targeted (as served, subject to change without notice):
    `<a class="result__a" href="...">TITLE</a>` for the link/title, and
    `<a class="result__snippet" ...>SNIPPET</a>` for the snippet
    immediately following each result.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hits: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capturing: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = dict(attrs)
        classes = (attr_map.get("class") or "").split()
        if "result__a" in classes:
            self._current = {"title": "", "url": attr_map.get("href") or "", "snippet": ""}
            self.hits.append(self._current)
            self._capturing = "title"
        elif "result__snippet" in classes and self._current is not None:
            self._capturing = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._capturing = None

    def handle_data(self, data: str) -> None:
        if self._capturing and self._current is not None:
            self._current[self._capturing] += data


def _clean_duckduckgo_redirect_url(href: str) -> str:
    """DuckDuckGo's HTML results wrap the real URL in a redirect
    (`//duckduckgo.com/l/?uddg=<url-encoded-real-url>&...`) -- unwrap it
    so callers get the actual destination, matching what a real API's
    `url` field would give."""
    match = re.search(r"[?&]uddg=([^&]+)", href)
    if not match:
        return href
    from urllib.parse import unquote

    return unquote(match.group(1))


@dataclass(frozen=True)
class DuckDuckGoSearchBackend:
    """Keyless fallback search backend. See module docstring: degraded
    mode only, not a peer of `TavilySearchBackend` -- HTML-scraped, no
    uptime/format guarantee, real block risk from a shared egress IP."""

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def search(self, query: str, limit: int) -> Sequence[SearchHit]:
        from urllib.parse import urlencode

        payload = urlencode({"q": query}).encode("utf-8")
        req = urllib_request.Request(
            DUCKDUCKGO_HTML_URL,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": _USER_AGENT,
            },
        )
        try:
            with urllib_request.urlopen(req, timeout=self.timeout_seconds) as response:
                html = response.read().decode("utf-8", errors="replace")
        except urllib_error.HTTPError as exc:
            raise GaiaWebBackendError(f"DuckDuckGo HTTP {exc.code}") from exc
        except urllib_error.URLError as exc:
            raise GaiaWebBackendError(f"DuckDuckGo request failed: {exc.reason}") from exc

        parser = _DuckDuckGoResultParser()
        parser.feed(html)
        hits = [
            SearchHit(
                title=item["title"].strip(),
                url=_clean_duckduckgo_redirect_url(item["url"]),
                snippet=item["snippet"].strip(),
            )
            for item in parser.hits
            if item["url"]
        ]
        return hits[: max(1, int(limit))]


@dataclass
class FallbackSearchBackend:
    """Tries `primary` first; on ANY exception, falls back to
    `secondary` and records which one actually answered (`last_used`) so
    a run's own trace/logs make degraded-mode operation visible instead
    of silently blending two backends of very different reliability.
    """

    primary: SearchBackend
    secondary: SearchBackend
    last_used: str = field(default="", init=False)
    fallback_count: int = field(default=0, init=False)

    def search(self, query: str, limit: int) -> Sequence[SearchHit]:
        try:
            hits = self.primary.search(query, limit)
            self.last_used = "primary"
            return hits
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: ANY primary
            # failure (auth, quota, network, malformed response) should
            # degrade to the fallback rather than sink the whole task.
            self.fallback_count += 1
            self.last_used = "secondary"
            try:
                return self.secondary.search(query, limit)
            except Exception as fallback_exc:  # noqa: BLE001
                raise GaiaWebBackendError(
                    f"both search backends failed: primary={exc!r} secondary={fallback_exc!r}"
                ) from fallback_exc


@dataclass(frozen=True)
class UrllibFetchBackend:
    """Adapts the existing WebWalkerQA/Track B real-network fetcher
    (`ant.evaluation_suite.web_fetch.urllib_fetcher`) to GAIA's
    `FetchBackend` protocol (`fetch(url) -> str`, plain extracted text --
    no same-site restriction, unlike Track B's navigation-scoped fetcher,
    since GAIA's `open_url` may legitimately need to leave whatever site
    a search result happened to point at)."""

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def fetch(self, url: str) -> str:
        raw_fetch = urllib_fetcher(timeout_seconds=self.timeout_seconds)(url)
        if raw_fetch.error is not None:
            raise GaiaWebBackendError(f"fetch failed for {url!r}: {raw_fetch.error}")
        text, _links = extract_text_and_links(raw_fetch.html, url)
        return text


def build_default_search_backend(
    *, tavily_api_key: str | None = None
) -> FallbackSearchBackend | DuckDuckGoSearchBackend:
    """Tavily-primary/DuckDuckGo-fallback if a Tavily key is available
    (`tavily_api_key`, defaulting to the `TAVILY_API_KEY` env var);
    DuckDuckGo-only otherwise -- explicit about which mode it built so a
    caller can log/assert on it rather than discover degraded-only mode
    by surprise mid-run."""
    key = tavily_api_key if tavily_api_key is not None else os.getenv("TAVILY_API_KEY", "")
    ddg = DuckDuckGoSearchBackend()
    if not key:
        return ddg
    return FallbackSearchBackend(primary=TavilySearchBackend(api_key=key), secondary=ddg)


def build_default_fetch_backend() -> UrllibFetchBackend:
    return UrllibFetchBackend()
