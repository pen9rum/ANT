"""Deterministic web page fetch/cache layer for the Track B (web-navigation)
evaluation substrate -- URL normalization, same-site restriction,
HTML-to-text/link extraction, and a disk-backed page cache.

Stdlib-only (`urllib.request` for fetching, `html.parser.HTMLParser` for
extraction) -- no `requests`/`httpx`/`BeautifulSoup`/`crawl4ai` dependency
is declared anywhere in pyproject.toml, and this module deliberately does
not add one. A heavy third-party crawl stack (e.g. the OFFICIAL WebWalker
method's own Crawl4AI dependency) belongs in an isolated
`external_wrappers/`-style venv when that baseline is actually implemented
-- not in ant's own core/eval-suite dependency surface, matching this
project's existing "minimal core deps, heavy third-party stacks isolated"
convention (see e.g. `external_wrappers/sweqa_pro_native_agent.py`'s own
isolated-venv docstring).

The real network fetcher (`urllib_fetcher`) is injected via a callable
(`Fetcher` protocol), never hardcoded into `PageCache` -- every test in
`tests/test_web_fetch.py` runs against a static, in-memory mock fetcher,
never a real network call, per the governing spec's "zero paid inference,
use static/mock webpages for tests" requirement (this whole module makes
zero LLM calls in any case; the concern here is a real HTTP call, not
cost).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qsl, quote, urldefrag, urlencode, urljoin, urlsplit, urlunsplit

DEFAULT_TIMEOUT_SECONDS = 10.0
# 2 MiB -- reject huge/generated pages, same spirit as MAX_TEXT_FILE_BYTES
# elsewhere in this suite.
MAX_PAGE_BYTES = 2_000_000

# Content-Type prefixes/values that are text-ish enough to attempt HTML
# parsing on. Not an exhaustive MIME registry -- just enough to catch the
# common non-text cases (images, PDFs, video/audio, generic binary
# downloads) a real website can serve for a URL that looks like a normal
# page link.
_TEXTLIKE_CONTENT_TYPES = ("text/", "application/xhtml+xml", "application/xml", "application/json")


def _looks_like_text_content(content_type: str, raw: bytes) -> bool:
    """Regression guard: confirmed live on a real WebWalkerQA gold source
    page that a server can respond to a normal-looking link with binary
    content whose garbled bytes, once decoded with errors="replace" and
    fed to html.parser, can trip an internal, uncaught AssertionError
    ("expected name token") deep in _markupbase -- not a clean, catchable
    parse failure. Two independent signals, either one sufficient to
    reject: (1) the Content-Type header, when present and unambiguous;
    (2) a NUL byte in the first 8 KiB (the same heuristic
    EvalRepoEnvironment._looks_like_text uses for repo files) -- text
    content essentially never contains a NUL byte, binary formats
    frequently do near their start.
    """
    header = content_type.split(";", 1)[0].strip().lower()
    if header and not any(header.startswith(prefix) for prefix in _TEXTLIKE_CONTENT_TYPES):
        return False
    if b"\x00" in raw[:8192]:
        return False
    return True


def normalize_url(url: str, *, base_url: str | None = None) -> str:
    """Canonical URL form used as BOTH the cache key and the identity a
    territory/navigation primitive compares against. Two URLs that
    normalize to the same string are treated as the SAME page -- this is
    what "deduplicate URLs/fragments" means structurally, not a separate
    dedup pass bolted on afterward.

    - Resolved against `base_url` first (when given) -- a relative link
      extracted from a page's own HTML is meaningless without this.
    - Scheme and host lowercased (case-insensitive per RFC 3986); path
      left as-is (path segments ARE case-sensitive on most servers).
    - Default port stripped (`:80` on http, `:443` on https).
    - Fragment stripped -- two URLs differing only by `#section` are
      treated as the same page for navigation/dedup purposes (a disclosed
      simplification: this can under-distinguish single-page-app anchor
      routing, but WebWalkerQA's own benchmark targets institutional/
      conference/education/game sites, not SPA-heavy anchor navigation).
    - Query parameters sorted (so `?a=1&b=2` and `?b=2&a=1` normalize
      identically) -- order carries no meaning for cache-key purposes.
    - Trailing slash stripped from the path, EXCEPT when the path is empty
      or exactly `/` (kept as `/` so the root is never normalized to an
      empty string).
    """
    resolved = urljoin(base_url, url) if base_url else url
    resolved, _fragment = urldefrag(resolved)
    parts = urlsplit(resolved)
    scheme = parts.scheme.lower()
    host = parts.hostname.lower() if parts.hostname else ""
    port = parts.port
    default_port = {"http": 80, "https": 443}.get(scheme)
    netloc = host if port is None or port == default_port else f"{host}:{port}"
    path = parts.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    if not path:
        path = "/"
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


def _registrable_host(url: str) -> str:
    """Host with a single leading "www." stripped -- NOT a real public-
    suffix-list-based registrable-domain computation (no such library is
    a project dependency). A disclosed simplification: covers the common
    `www.example.com` <-> `example.com` case exactly, and is otherwise a
    strict exact-host match -- it will correctly REJECT an unrelated
    subdomain (e.g. `evil.example.com` when root is `example.com`) rather
    than risk under-restricting, since erring toward "same-site" being too
    narrow (occasionally blocking a legitimate same-organization subdomain)
    is the safer failure mode for a benchmark that requires root-site-only
    traversal, versus too broad (silently letting navigation leave the
    site).
    """
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def same_site(url: str, root_url: str) -> bool:
    """True iff `url` belongs to the same site as `root_url`, per
    `_registrable_host`'s documented (simplified, disclosed) policy."""
    return bool(_registrable_host(url)) and _registrable_host(url) == _registrable_host(root_url)


@dataclass(frozen=True)
class PageLink:
    """One navigable link: its visible anchor text alongside its
    destination URL -- matching the official WebWalker environment's own
    `(button_text, url)` representation (`app.py::extract_links_with_text`,
    confirmed by direct code reading), not a bare URL. A prior version of
    this module showed only the destination URL with no label at all;
    confirmed live that this made navigation decisions largely blind
    (sample page: 66.9 links/page on average, shown as opaque URLs like
    `/animated-icon-bundles` with no indication what a link actually
    leads to) and was a primary, evidenced driver of the repeated-
    navigation pathology seen in every step-budget-exhaustion trajectory
    in the original N=60 Matched ReAct Web run.
    """

    text: str
    url: str


class _TextLinkExtractor(HTMLParser):
    """Stdlib-only HTML -> (visible text, outgoing (anchor_text, href)
    list) extractor. Deterministic: same HTML input always produces the
    same output, no external model/heuristic-library involved.
    `script`/`style`/`noscript` content is excluded from the extracted
    text (not genuinely visible page content); every other tag's text
    content is kept, whitespace-collapsed at the end.

    Anchor text is accumulated separately, between an `<a href=...>` start
    tag and its matching `</a>`, so each link carries its own visible
    label -- not just interleaved into the page's general text flow with
    no link back to its URL.
    """

    _SKIP_TAGS = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._text_parts: list[str] = []
        self._links: list[tuple[str, str]] = []
        self._skip_depth = 0
        self._current_anchor_href: str | None = None
        self._current_anchor_text_parts: list[str] = []

    def _finalize_anchor(self) -> None:
        if self._current_anchor_href is not None:
            text = " ".join(self._current_anchor_text_parts).strip()
            self._links.append((text, self._current_anchor_href))
            self._current_anchor_href = None
            self._current_anchor_text_parts = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "a":
            # A new <a> before the previous one's </a> arrived (real HTML
            # is not always well-nested) -- finalize whatever text the
            # prior anchor had accumulated rather than silently merging
            # two links' text together.
            self._finalize_anchor()
            href = next((value for name, value in attrs if name == "href" and value), None)
            if href:
                self._current_anchor_href = href

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag == "a":
            self._finalize_anchor()

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "a":
            self._finalize_anchor()

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self._text_parts.append(data.strip())
        if self._current_anchor_href is not None and data.strip():
            self._current_anchor_text_parts.append(data.strip())

    def result(self) -> tuple[str, list[tuple[str, str]]]:
        self._finalize_anchor()  # an unclosed trailing <a> at EOF
        text = "\n".join(self._text_parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text, self._links


def extract_text_and_links(html: str, base_url: str) -> tuple[str, list[PageLink]]:
    """Deterministic HTML -> (extracted_text, normalized_deduplicated
    PageLinks). Links are resolved against `base_url` and normalized via
    `normalize_url` -- callers never see a raw/relative href. Deduplicated
    by URL (keeping the first-seen anchor text for a URL that appears more
    than once on the same page, e.g. a logo link and a nav-menu link to
    the same home page).
    """
    parser = _TextLinkExtractor()
    parser.feed(html)
    text, raw_links = parser.result()
    seen: dict[str, str] = {}
    for anchor_text, href in raw_links:
        if href.strip().lower().startswith(("javascript:", "mailto:", "tel:")):
            continue
        normalized = normalize_url(href, base_url=base_url)
        if normalized not in seen:
            seen[normalized] = anchor_text
    return text, [PageLink(text=anchor_text, url=url) for url, anchor_text in seen.items()]


@dataclass(frozen=True)
class FetchedPage:
    """One fetch attempt's complete, cacheable outcome -- distinguishes
    "genuinely fetched, empty content" from "inaccessible" from "not yet
    fetched" via `status`, per the governing spec's explicit requirement.
    """

    url: str
    requested_url: str
    status: str  # "ok" | "error" | "not_fetched"
    http_status: int | None
    redirect_target: str | None
    text: str
    links: list[PageLink]
    error: str | None
    fetched_at: str
    from_cache: bool = False

    def to_json(self) -> dict:
        data = {
            "url": self.url,
            "requested_url": self.requested_url,
            "status": self.status,
            "http_status": self.http_status,
            "redirect_target": self.redirect_target,
            "text": self.text,
            "links": [{"text": link.text, "url": link.url} for link in self.links],
            "error": self.error,
            "fetched_at": self.fetched_at,
        }
        return data

    @classmethod
    def from_json(cls, data: dict) -> FetchedPage:
        raw_links = data.get("links", [])
        # A cache entry written before PageLink existed stored links as
        # bare URL strings -- load it as (text="", url) so an old on-disk
        # cache (e.g. the original 11.7% run's, kept untouched in its own
        # namespace) can still be inspected, rather than crashing.
        links = [
            PageLink(text="", url=item) if isinstance(item, str) else PageLink(**item)
            for item in raw_links
        ]
        return cls(
            url=data["url"],
            requested_url=data["requested_url"],
            status=data["status"],
            http_status=data.get("http_status"),
            redirect_target=data.get("redirect_target"),
            text=data.get("text", ""),
            links=links,
            error=data.get("error"),
            fetched_at=data["fetched_at"],
            from_cache=True,
        )


@dataclass(frozen=True)
class _RawFetch:
    http_status: int | None
    redirect_target: str | None
    html: str
    error: str | None


class Fetcher(Protocol):
    def __call__(self, url: str) -> _RawFetch: ...


def _ensure_ascii_url(url: str) -> str:
    """IRI -> URI conversion: percent-encodes any raw non-ASCII characters
    in the path/query/fragment, and IDNA-encodes a non-ASCII host, so the
    URL can be sent as a valid HTTP request line.

    Regression note: confirmed live on WebWalkerQA's own gold source pages
    (several Chinese-hosted sites embed raw CJK characters directly in the
    URL path/query, e.g. a full-width semicolon) -- `http.client` requires
    the request line to be pure ASCII, and `urllib.request` does NOT
    perform this conversion for the caller automatically; passing such a
    URL straight to `Request()` raises `UnicodeEncodeError` deep inside
    `http.client.HTTPConnection.putrequest`, not a clean, catchable
    `URLError`/`HTTPError` the rest of this module already handles.
    """
    parts = urlsplit(url)
    host = parts.hostname or ""
    try:
        host.encode("ascii")
        netloc_host = host
    except UnicodeEncodeError:
        netloc_host = host.encode("idna").decode("ascii")
    netloc = f"{netloc_host}:{parts.port}" if parts.port else netloc_host
    if parts.username:
        credentials = parts.username + (f":{parts.password}" if parts.password else "")
        netloc = f"{credentials}@{netloc}"
    path = quote(parts.path, safe="/%")
    query = quote(parts.query, safe="=&%")
    fragment = quote(parts.fragment, safe="%")
    return urlunsplit((parts.scheme, netloc, path, query, fragment))


_META_TAG_RE = re.compile(r"<meta\b([^>]*)>", re.IGNORECASE)
_ATTR_RE = re.compile(
    r"""(\w[\w-]*)\s*=\s*"([^"]*)"|(\w[\w-]*)\s*=\s*'([^']*)'|(\w[\w-]*)\s*=\s*([^\s"'>]+)"""
)
MAX_META_REFRESH_HOPS = 5


def _extract_meta_refresh_target(html: str, base_url: str) -> str | None:
    """Finds `<meta http-equiv="refresh" content="N;url=TARGET">` and
    returns TARGET resolved against `base_url`, or None if no such tag
    exists. This is standard, universally-supported browser behavior
    (equivalent to the HTTP `Refresh` header; supported by every major
    browser, confirmed via caniuse.com) -- any real browser, and therefore
    the official WebWalker environment's own Playwright-rendered Crawl4AI
    pipeline, follows this automatically as part of normal page load. A
    plain HTTP client does not, by construction: `urllib.request` only
    follows HTTP-level (3xx header) redirects, never a client-side meta
    tag embedded in the response body. Confirmed live: a real WebWalkerQA
    root URL (ciie.org) serves exactly this pattern and nothing else,
    leaving every question rooted there with a completely empty page
    (zero text, zero links) under the old behavior -- a genuine
    environment fidelity gap, not a benchmark-specific heuristic.
    """
    for match in _META_TAG_RE.finditer(html):
        attrs: dict[str, str] = {}
        for attr_match in _ATTR_RE.finditer(match.group(1)):
            groups = attr_match.groups()
            name, value = next(
                (groups[i], groups[i + 1]) for i in (0, 2, 4) if groups[i] is not None
            )
            attrs[name.lower()] = value
        if attrs.get("http-equiv", "").strip().lower() != "refresh":
            continue
        content = attrs.get("content", "")
        _delay, _sep, rest = content.partition(";")
        if not _sep:
            continue
        _key, _eq, target = rest.strip().partition("=")
        target = target.strip().strip("'\"")
        if not target:
            continue
        return urljoin(base_url, target)
    return None


def urllib_fetcher(timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> Fetcher:
    """Builds a real-network `Fetcher` backed by `urllib.request`. Kept as
    a factory (not a bare module-level function) so `timeout_seconds` is
    explicit and immutable per fetcher instance, and so tests never need
    to construct one at all (they inject their own static mock instead).
    """

    def _fetch_once(url: str) -> _RawFetch:
        import urllib.error
        import urllib.request

        ascii_url = _ensure_ascii_url(url)
        request = urllib.request.Request(
            ascii_url, headers={"User-Agent": "ANTMAN-WebWalkerQA-eval/0.1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                final_url = response.geturl()
                redirect_target = (
                    final_url if normalize_url(final_url) != normalize_url(url) else None
                )
                raw = response.read(MAX_PAGE_BYTES + 1)
                if len(raw) > MAX_PAGE_BYTES:
                    return _RawFetch(
                        http_status=response.status,
                        redirect_target=redirect_target,
                        html="",
                        error=f"page exceeds MAX_PAGE_BYTES ({MAX_PAGE_BYTES})",
                    )
                content_type = response.headers.get("Content-Type", "")
                if not _looks_like_text_content(content_type, raw):
                    return _RawFetch(
                        http_status=response.status,
                        redirect_target=redirect_target,
                        html="",
                        error=f"non-text content (Content-Type={content_type!r}), not parsed",
                    )
                charset = "utf-8"
                if "charset=" in content_type:
                    charset = (
                        content_type.split("charset=", 1)[1].split(";", 1)[0].strip() or "utf-8"
                    )
                try:
                    html = raw.decode(charset, errors="replace")
                except LookupError:
                    html = raw.decode("utf-8", errors="replace")
                return _RawFetch(
                    http_status=response.status,
                    redirect_target=redirect_target,
                    html=html,
                    error=None,
                )
        except urllib.error.HTTPError as exc:
            return _RawFetch(
                http_status=exc.code, redirect_target=None, html="", error=f"HTTPError: {exc}"
            )
        except urllib.error.URLError as exc:
            return _RawFetch(
                http_status=None, redirect_target=None, html="", error=f"URLError: {exc.reason}"
            )
        except UnicodeError as exc:
            # Defense in depth for whatever _ensure_ascii_url doesn't fully
            # cover (e.g. an IDNA-encoding failure on a malformed host) --
            # one bad URL must report as a normal fetch error, never crash
            # a whole batch (confirmed live: this exact failure mode took
            # down a 338-URL eligibility-check run before this was added).
            return _RawFetch(
                http_status=None, redirect_target=None, html="", error=f"UnicodeError: {exc}"
            )
        except (TimeoutError, OSError) as exc:
            return _RawFetch(
                http_status=None,
                redirect_target=None,
                html="",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _fetch(url: str) -> _RawFetch:
        current_url = url
        overall_redirect: str | None = None
        for _hop in range(MAX_META_REFRESH_HOPS):
            result = _fetch_once(current_url)
            if result.error is not None:
                return result
            hop_redirect = result.redirect_target or current_url
            if result.redirect_target is not None:
                overall_redirect = result.redirect_target
            target = _extract_meta_refresh_target(result.html, base_url=hop_redirect)
            if target is None or normalize_url(target) == normalize_url(hop_redirect):
                return _RawFetch(
                    http_status=result.http_status, redirect_target=overall_redirect,
                    html=result.html, error=None,
                )
            current_url = target
            overall_redirect = target
        return _RawFetch(
            http_status=None, redirect_target=None, html="",
            error=f"meta-refresh chain exceeded {MAX_META_REFRESH_HOPS} hops",
        )

    return _fetch


@dataclass
class PageCache:
    """Disk-backed, deduplicating page cache scoped to one root site.

    `restrict_to_root`, when `root_url` is given: `get()` refuses to fetch
    (returns a `status="error"` `FetchedPage`, never raises) any URL that
    is not `same_site(url, root_url)` -- the one enforcement point for
    "same-domain/root-site restriction unless the official benchmark
    clearly requires an external link" (no such requirement was found in
    the audited benchmark structure -- see `docs/webwalkerqa_ant_mapping.md`
    section 1: "explore only within that website").
    """

    cache_dir: Path
    fetcher: Fetcher | None = None
    root_url: str | None = None
    restrict_to_root: bool = True
    _memory: dict[str, FetchedPage] = field(default_factory=dict, init=False, repr=False)

    def _cache_path(self, normalized_url: str) -> Path:
        import hashlib

        digest = hashlib.sha256(normalized_url.encode()).hexdigest()[:32]
        return self.cache_dir / f"{digest}.json"

    def get(self, url: str, *, base_url: str | None = None) -> FetchedPage:
        """Normalize `url`, return a cached result if one exists (memory,
        then disk), otherwise fetch via `self.fetcher` (never fetches if
        `self.fetcher` is None -- returns `status="not_fetched"` instead,
        so a caller with no live-network fetcher configured degrades to a
        no-op rather than crashing)."""
        normalized = normalize_url(url, base_url=base_url)

        if (
            self.restrict_to_root
            and self.root_url is not None
            and not same_site(normalized, self.root_url)
        ):
            return FetchedPage(
                url=normalized,
                requested_url=url,
                status="error",
                http_status=None,
                redirect_target=None,
                text="",
                links=[],
                error="rejected: outside the permitted root-site domain",
                fetched_at=_now(),
            )

        if normalized in self._memory:
            return replace(self._memory[normalized], from_cache=True)

        disk_path = self._cache_path(normalized)
        if disk_path.exists():
            page = FetchedPage.from_json(json.loads(disk_path.read_text(encoding="utf-8")))
            self._memory[normalized] = page
            return page

        if self.fetcher is None:
            page = FetchedPage(
                url=normalized,
                requested_url=url,
                status="not_fetched",
                http_status=None,
                redirect_target=None,
                text="",
                links=[],
                error=None,
                fetched_at=_now(),
            )
            return page

        raw = self.fetcher(normalized)
        if raw.error is not None:
            page = FetchedPage(
                url=normalized,
                requested_url=url,
                status="error",
                http_status=raw.http_status,
                redirect_target=raw.redirect_target,
                text="",
                links=[],
                error=raw.error,
                fetched_at=_now(),
            )
        else:
            effective_url = raw.redirect_target or normalized
            try:
                # Defense in depth on top of urllib_fetcher's own content-
                # type/NUL-byte screening: html.parser can still raise an
                # uncaught AssertionError on sufficiently malformed input
                # (confirmed live -- see _looks_like_text_content's own
                # docstring). One bad page must degrade to a normal
                # "error" FetchedPage, never crash the caller.
                text, links = extract_text_and_links(raw.html, base_url=effective_url)
                parse_error: str | None = None
            except Exception as exc:  # noqa: BLE001
                text, links = "", []
                parse_error = f"HTML parse failure: {type(exc).__name__}: {exc}"

            if parse_error is not None:
                page = FetchedPage(
                    url=normalized,
                    requested_url=url,
                    status="error",
                    http_status=raw.http_status,
                    redirect_target=raw.redirect_target,
                    text="",
                    links=[],
                    error=parse_error,
                    fetched_at=_now(),
                )
            else:
                if self.restrict_to_root and self.root_url is not None:
                    links = [link for link in links if same_site(link.url, self.root_url)]
                page = FetchedPage(
                    url=normalized,
                    requested_url=url,
                    status="ok",
                    http_status=raw.http_status,
                    redirect_target=raw.redirect_target,
                    text=text,
                    links=links,
                    error=None,
                    fetched_at=_now(),
                )

        self._memory[normalized] = page
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = disk_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(page.to_json(), ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(disk_path)
        return page

    def known_pages(self) -> list[FetchedPage]:
        return list(self._memory.values())


def _now() -> str:
    return datetime.now(UTC).isoformat()
