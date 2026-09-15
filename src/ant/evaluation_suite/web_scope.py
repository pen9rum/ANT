"""ANTMAN's Track B (web-navigation) substrate abstraction: territories,
navigation, and page inspection over a root-site-scoped web environment.

PROPOSED TERRITORY SEMANTICS (per the governing spec's request to report
these clearly before freezing anything -- this is the design this module
implements, not a separate document nobody reads):

- **Territory = a runtime-discovered, provisional set of pages**, anchored
  to one page reached from an already-visited page's own link list. This
  is the central departure from Track A: `EvalRepoEnvironment.iter_files()`
  and `discover_territories()` enumerate a repository's COMPLETE file
  universe before any agent action happens; a website has no equivalent
  pre-existing structural listing (`docs/webwalkerqa_ant_mapping.md`
  section 4, audited against the primary paper -- no sitemap/tree API is
  exposed to the agent). A `WebTerritory` therefore starts empty/minimal
  and GROWS as `EvalWebEnvironment.navigate()` discovers more pages, never
  pre-populated from anything the agent hasn't actually reached.
- **Worker = specialized over one such territory** (one first-level
  section/subtree reachable from the root, mirroring the paper's own
  domain examples -- e.g. a conference site's "Schedule" vs. "Speakers"
  vs. "Venue" sections) -- but WHICH pages belong to a worker's territory
  is discovered through the worker's own navigation, not assigned upfront
  by a routing/indexing pass the way Track A's `build_worker_cards` reads
  a repo's real directory structure.
- **Navigation primitive = `EvalWebEnvironment.navigate(from_page, to_url)`**
  -- follows a link that was ACTUALLY present in `from_page.links`
  (enforced, not just documented: raises if `to_url` was never discovered
  there). This is the one, sole way new pages become knowable to the
  environment -- there is no "jump to any URL" primitive, and no upfront
  full-graph listing a worker could route against instead of navigating.
- **Page inspection primitive = `EvalWebEnvironment.inspect(url)`** --
  re-reads a page's already-fetched content (from cache/memory). It is
  NOT a second way to discover new pages; inspecting a URL the
  environment has never visited returns a `status="not_fetched"`
  placeholder rather than silently fetching it, keeping `navigate()` the
  single discovery channel.
- **What this module explicitly does NOT do** (per the governing spec):
  no parameter anywhere in this module accepts a full/gold website graph;
  `build_territory_from_discovered()` only ever accepts pages an
  `EvalWebEnvironment` has ALREADY visited through real navigation, so
  `Golden_Path` cannot be pre-materialized into a `WebTerritory` even by
  a careless caller (there is no argument slot for it); nothing in this
  module reads `source_websites` at all, so it cannot seed routing;
  `same_site`-restricted, discovered-link-only navigation is exactly what
  keeps this a root-site-traversal task rather than open-web search.

This module intentionally stops at the substrate/primitive layer -- it
does NOT wire a `LocalCoordinator`, Need Graph, or WorkerCard-equivalent
for the web substrate (that is explicitly out of scope for this
environment-validation pass; see the governing spec's "STOP after
environment validation" instruction). `WebTerritory` is a plain data
container a future Track B coordinator layer would consume, analogous in
ROLE to Track A's `WorkerCard`, not a rename of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from ant.evaluation_suite.web_fetch import FetchedPage, PageCache, normalize_url


@dataclass
class WebTerritory:
    """A runtime-discovered, provisional page set -- see this module's own
    docstring for the full semantics. `pages` only ever grows by a caller
    passing in URLs that `EvalWebEnvironment` has already visited; nothing
    in this class fetches or discovers on its own.
    """

    id: str
    anchor_url: str
    pages: list[str] = field(default_factory=list)
    discovered_from: str | None = None


class EvalWebEnvironment:
    """Root-site-scoped web substrate. See this module's own docstring for
    the full territory/navigation/inspection design. `max_steps`, when
    given, enforces WebWalkerQA's own 15-step exploration ceiling
    (`docs/webwalkerqa_ant_mapping.md` section 2) at the navigation
    primitive itself -- not left to a caller to remember to check.
    """

    def __init__(self, root_url: str, cache: PageCache, *, max_steps: int | None = None) -> None:
        self.root_url = normalize_url(root_url)
        self.cache = cache
        self.max_steps = max_steps
        self._visited: dict[str, FetchedPage] = {}
        self._step_count = 0

    def root_page(self) -> FetchedPage:
        """The one page available without any prior navigation. Does not
        consume a navigation step -- the root is the environment's own
        starting point, not a discovered link."""
        page = self.cache.get(self.root_url)
        self._visited[page.url] = page
        return page

    def navigate(self, from_page: FetchedPage, to_url: str) -> FetchedPage:
        """Follow a link that was ACTUALLY present in `from_page.links`.

        Raises `ValueError` if `to_url` was never discovered on
        `from_page` -- the one hard enforcement point preventing a
        teleport to an arbitrary URL that was never actually reached
        through real link-following, and `RuntimeError` once `max_steps`
        navigations have already happened.
        """
        normalized_target = normalize_url(to_url, base_url=from_page.url)
        if normalized_target not in {link.url for link in from_page.links}:
            raise ValueError(
                f"{normalized_target!r} was not among the links discovered on "
                f"{from_page.url!r} -- navigate() only follows links actually "
                "present on an already-visited page."
            )
        if self.max_steps is not None and self._step_count >= self.max_steps:
            raise RuntimeError(f"navigation step budget exhausted (max_steps={self.max_steps})")
        self._step_count += 1
        page = self.cache.get(normalized_target)
        self._visited[page.url] = page
        return page

    def inspect(self, url: str) -> FetchedPage:
        """Re-read an already-visited page's content. NOT a second
        discovery channel: a URL this environment has never visited
        returns a `status="not_fetched"` placeholder rather than being
        silently fetched -- `navigate()` is the sole way new pages become
        knowable.
        """
        normalized = normalize_url(url)
        if normalized in self._visited:
            return self._visited[normalized]
        return FetchedPage(
            url=normalized,
            requested_url=url,
            status="not_fetched",
            http_status=None,
            redirect_target=None,
            text="",
            links=[],
            error=None,
            fetched_at=datetime.now(UTC).isoformat(),
        )

    def discovered_pages(self) -> list[str]:
        return sorted(self._visited)

    def step_count(self) -> int:
        return self._step_count


def build_territory_from_discovered(
    territory_id: str,
    anchor_url: str,
    pages: list[str],
    *,
    discovered_from: str | None = None,
) -> WebTerritory:
    """Turns pages an `EvalWebEnvironment` has ALREADY visited (e.g. from
    `discovered_pages()`) into a `WebTerritory`. Deliberately takes only
    already-discovered page URLs as input -- there is no parameter through
    which `golden_path`/`source_websites`/gold_answer could be threaded
    in, by construction (see `tests/test_web_scope.py`'s leakage-prevention
    tests, which assert this function's own signature has no such
    parameter).
    """
    return WebTerritory(
        id=territory_id,
        anchor_url=normalize_url(anchor_url),
        pages=list(pages),
        discovered_from=discovered_from,
    )
