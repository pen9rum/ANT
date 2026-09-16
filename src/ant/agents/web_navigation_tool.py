"""Web-specific substitute for `ant.tools.local.LocalSearchTool`, injected
into an unmodified `LocalCoordinator` via its `search_tool_factory` seam
(see `LocalCoordinator.__init__`'s own docstring on that parameter).

WHY THIS EXISTS: the first version of `AntWebAgent` built a frozen,
root+one-hop worker roster at bootstrap time and handed `LocalCoordinator`
a static file set -- confirmed live (a 24-task paid run, 0/24 correct)
that this makes WebWalkerQA answer pages structurally unreachable,
because those answers are essentially always 2+ hops deep (confirmed:
100% of a 24-task sample of the corrected ReAct baseline's own
trajectories needed >=2 navigate() calls, several needed 10-14). A
worker therefore needs to be able to click into NEW pages, live, during
its own reasoning -- not be frozen to whatever bootstrap pre-fetched.

HOW: `AutonomousWorker.run()` (src/ant/workers/autonomous.py) calls
`self.tools.search(...)` and `.dense_search(...)` UNCONDITIONALLY at the
start of every worker execution, before any reasoner-driven tool
planning happens. This class's own `search()` exploits exactly that
call: it first tries a plain local BM25 search over whatever pages are
already materialized for that worker; only when that comes back empty
does it spend real navigation budget -- one deterministic hop into the
worker's own assigned first-level link (its territory specialization,
established at bootstrap, never LLM-chosen, so a worker's identity stays
tied to the section it was created for), then, if still unresolved,
further LLM-mediated hops chosen from whatever page it's currently on,
by asking a plain (anchor_text, url) "which of these looks most likely
to help" question -- never seeing the question's reference answer,
golden_path, or source_websites (those never reach this class at all).

Every real navigation goes through the SAME `EvalWebEnvironment`
instance for the whole query (shared across every worker), so its
`max_steps` budget is enforced globally, not per-worker: once any
worker's navigation has used up the shared ceiling,
`EvalWebEnvironment.navigate()` itself raises and every subsequent
worker's own navigation attempts simply stop trying, whatever their own
individual round/need history looks like. `EvalWebEnvironment.navigate()`
also independently enforces "only a link actually present on the current
page" (raises ValueError otherwise) -- this class relies on that
enforcement rather than re-implementing it.

Everything else (rank_symbols/resolve_symbol/navigate[symbol]/
references/indexed_callers/callers/callees/assignments/imports/
subclasses/read_region) is delegated unchanged to a wrapped
LocalSearchTool over the materialized-pages directory: those pages are
plain local .txt files once fetched, so the existing code-oriented tool
implementation already reads them correctly. Real web pages essentially
never contain literal `class `/`def ` tokens, so `AutonomousWorker`'s own
mechanical candidate-symbol extraction typically yields nothing for
these calls to act on -- expected and harmless; all real work for this
substrate happens inside search()/dense_search() above.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ant.domain import Evidence
from ant.evaluation_suite.web_fetch import FetchedPage, PageLink
from ant.evaluation_suite.web_scope import EvalWebEnvironment
from ant.retrieval.relevance import extract_terms
from ant.tools.local import LocalSearchTool

# Deliberately NO cap on how many hops one search() call may chase --
# a worker must be able to go anchor -> child -> grandchild -> ... within
# a single dispatch, stopping only on sufficient evidence, no remaining
# relevant link (_choose_link_via_llm returns None), or the shared
# nav_budget running out (all three checked every iteration below). No
# infinite-loop risk: frontier.visited_urls/exhausted_links only grow, so
# a page's own (finite) link list monotonically shrinks as candidates,
# and EvalWebEnvironment's own step budget is a hard, finite ceiling
# regardless. Confirmed live (10-task validation, see this module's own
# git history): an earlier, arbitrary per-call hop cap of 3 caused the
# shared budget to be spent mostly as many workers' single mandatory hops
# rather than a few promising workers going deep -- concretely, task
# webwalkerqa-37023bb2's worker-0 reached the exact parent page of the
# answer and then stopped there, while the remaining budget went to
# unrelated workers' own first hops instead of letting worker-0 continue.
# How many of a page's own links get shown to the link-choice LLM call --
# a prompt-size guard, not a relevance filter (candidates beyond this cap
# are simply never offered, in extraction order, the same "never sort by
# relevance" discipline the bootstrap link roster itself uses).
MAX_LINK_CANDIDATES_SHOWN = 40


@dataclass
class WorkerFrontier:
    """One worker's own live navigation state -- current page, what it's
    already tried, and whether it has taken its mandatory first hop into
    its own assigned link yet. Mutated in place by WebSearchTool.search();
    NOT stored on WorkerCard itself (keeps this a purely additive,
    web-specific side-channel, no domain-model schema change).
    """

    current_page: FetchedPage
    assigned_link: PageLink | None
    visited_urls: set[str] = field(default_factory=set)
    exhausted_links: set[str] = field(default_factory=set)
    taken_first_hop: bool = False
    next_file_index: int = 0


def worker_key_from_files(files: list[str]) -> str | None:
    """Every file this substrate ever materializes is named
    f"{worker_id}__{tag}.txt" (see AntWebAgent's own bootstrap) -- so any
    filename in a worker's own `files` list (its position never matters,
    only that at least one exists) recovers which worker owns this
    search() call, purely by string convention, no separate identity
    channel needed."""
    for name in files:
        if "__" in name:
            return name.split("__", 1)[0]
    return None


class WebSearchTool:
    def __init__(
        self,
        materialized_dir: Path,
        env: EvalWebEnvironment,
        provider,
        frontiers: dict[str, WorkerFrontier],
    ) -> None:
        self._local = LocalSearchTool(materialized_dir, index_path=None)
        self._materialized_dir = materialized_dir
        self._env = env
        self._provider = provider
        self._frontiers = frontiers
        self.nav_log: list[dict] = []

    # --- the two methods AutonomousWorker.run() calls unconditionally ---

    def search(
        self, query: str, files: list[str], limit: int = 8, context_lines: int = 6
    ) -> list[Evidence]:
        worker_key = worker_key_from_files(files)
        if worker_key is None:
            return self._local.search(query, files, limit=limit, context_lines=context_lines)
        frontier = self._frontiers.get(worker_key)

        # The worker's own assigned first-level link is its territory
        # specialization (established at bootstrap) -- it must actually be
        # visited at least once, UNCONDITIONALLY, the first time this
        # worker's search() is ever called, regardless of whether root-only
        # content happens to loosely overlap with whatever need text
        # triggered this call. Confirmed live (a real paid task: 9 active
        # workers, 10 rounds, 0 navigation steps) that gating even this
        # first hop behind a term-overlap check was wrong: root's own
        # generic navigation text (dates, update names) satisfied that
        # check for essentially every decomposed sub-need, so every worker
        # stayed planted on root and its own actual territory was never
        # reached -- reproducing a milder version of the original one-hop
        # bootstrap bug this class exists to fix. Only hop 2+ (below) is
        # gated on whether what's materialized so far actually helps.
        if frontier is not None and not frontier.taken_first_hop:
            frontier.taken_first_hop = True  # at most one attempt, ever, regardless of outcome
            link = frontier.assigned_link
            if link is not None and self._budget_remaining():
                page = self._follow(worker_key, frontier, link, query)
                if page is not None:
                    self._materialize(worker_key, frontier, page, files)

        results = self._local.search(query, files, limit=limit, context_lines=context_lines)
        if frontier is None:
            return results
        # LocalSearchTool.search() ranks and returns its top-k over
        # whatever text is indexed even when nothing actually matches the
        # query (it has no relevance THRESHOLD, only a ranking) -- so
        # `results` being non-empty is not itself proof the need is
        # answered by what's currently materialized. A cheap, honest
        # term-overlap check against the raw materialized text is what
        # actually decides whether to spend further navigation budget
        # looking deeper, not the mere presence of low-relevance search
        # hits.
        if self._term_overlap_with_materialized(query, files):
            return results
        # Persistent deep navigation, no per-call hop cap (see this
        # module's own top-of-file note): keep following LLM-chosen links
        # from wherever this worker currently sits until evidence is
        # sufficient, no relevant link remains, or the shared budget runs
        # out -- any one of the three can fire on any iteration.
        sufficient = False
        while not sufficient:
            if not self._budget_remaining():
                break
            link = self._choose_link_via_llm(frontier, query)
            if link is None:
                break
            page = self._follow(worker_key, frontier, link, query)
            if page is None:
                continue
            self._materialize(worker_key, frontier, page, files)
            results = self._local.search(query, files, limit=limit, context_lines=context_lines)
            sufficient = self._term_overlap_with_materialized(query, files)
        return results

    def _term_overlap_with_materialized(self, query: str, files: list[str]) -> bool:
        terms = extract_terms(query)
        if not terms:
            return True  # nothing to match against -- don't navigate on an empty need
        combined = []
        for name in files:
            try:
                combined.append(
                    (self._materialized_dir / name).read_text(encoding="utf-8", errors="replace")
                )
            except OSError:
                continue
        haystack = " ".join(combined).lower()
        return any(term.lower() in haystack for term in terms)

    def dense_search(self, query: str, files: list[str], limit: int = 4) -> list[Evidence]:
        # No live-navigation side effect here: search() (called first,
        # unconditionally, every round -- autonomous.py:83 before :100)
        # already had the chance to navigate for this exact need. A
        # second independent navigation attempt from dense_search would
        # double-spend the shared budget for no new decision basis.
        return self._local.dense_search(query, files, limit=limit)

    # --- navigation internals ---

    def _budget_remaining(self) -> bool:
        return self._env.max_steps is None or self._env.step_count() < self._env.max_steps

    def _follow(
        self, worker_key: str, frontier: WorkerFrontier, link: PageLink, query: str
    ) -> FetchedPage | None:
        try:
            page = self._env.navigate(frontier.current_page, link.url)
        except (ValueError, RuntimeError):
            frontier.exhausted_links.add(link.url)
            return None
        frontier.taken_first_hop = True
        frontier.current_page = page
        frontier.visited_urls.add(page.url)
        self.nav_log.append(
            {"worker": worker_key, "anchor_text": link.text, "url": page.url, "via_need": query}
        )
        return page

    def _materialize(
        self, worker_key: str, frontier: WorkerFrontier, page: FetchedPage, files: list[str]
    ) -> None:
        filename = f"{worker_key}__hop{frontier.next_file_index}.txt"
        frontier.next_file_index += 1
        (self._materialized_dir / filename).write_text(page.text, encoding="utf-8")
        if filename not in files:
            files.append(filename)

    def _choose_link_via_llm(self, frontier: WorkerFrontier, query: str) -> PageLink | None:
        candidates = [
            link
            for link in frontier.current_page.links
            if link.url not in frontier.visited_urls and link.url not in frontier.exhausted_links
        ][:MAX_LINK_CANDIDATES_SHOWN]
        if not candidates:
            return None
        listing = "\n".join(
            f"{i}. {link.text!r} -> {link.url}" for i, link in enumerate(candidates)
        )
        prompt = (
            "You are navigating a website's own real pages to gather evidence for a "
            "research need. You may only follow a link that is actually listed below.\n\n"
            f"Need: {query}\n"
            f"Current page: {frontier.current_page.url}\n"
            f"Available links on the current page (index. anchor text -> URL):\n{listing}\n\n"
            "Reply with ONLY the index number of the single most promising link to follow "
            "next, or reply NONE if nothing listed looks relevant."
        )
        try:
            result = self._provider.responses_text(prompt, max_output_tokens=16)
        except Exception:  # noqa: BLE001 -- a link-choice failure must not crash the worker
            return None
        text = result.text.strip().upper()
        if "NONE" in text:
            return None
        match = re.search(r"\d+", text)
        if not match:
            return None
        index = int(match.group())
        return candidates[index] if 0 <= index < len(candidates) else None

    # --- pure passthrough: already-materialized pages are plain local
    # files, so the existing code-oriented tool implementation is correct
    # as-is for these. Real web text essentially never matches the
    # class/def-oriented candidate extraction that feeds these calls, so
    # they are expected to be called rarely, with small/empty inputs. ---

    def rank_symbols(
        self, symbols: list[str], files: list[str], limit: int = 8, need: str = ""
    ) -> list[str]:
        return self._local.rank_symbols(symbols, files, limit=limit, need=need)

    def resolve_symbol(
        self, symbol: str, files: list[str], limit: int = 6, need: str = ""
    ) -> list[Evidence]:
        return self._local.resolve_symbol(symbol, files, limit=limit, need=need)

    def navigate(self, symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        return self._local.navigate(symbol, files, limit=limit)

    def references(self, symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        return self._local.references(symbol, files, limit=limit)

    def indexed_callers(self, symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        return self._local.indexed_callers(symbol, files, limit=limit)

    def callers(self, symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        return self._local.callers(symbol, files, limit=limit)

    def callees(self, symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        return self._local.callees(symbol, files, limit=limit)

    def assignments(self, symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        return self._local.assignments(symbol, files, limit=limit)

    def imports(self, module_or_symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        return self._local.imports(module_or_symbol, files, limit=limit)

    def subclasses(self, symbol: str, files: list[str], limit: int = 8) -> list[Evidence]:
        return self._local.subclasses(symbol, files, limit=limit)

    def read_region(self, path: str, line: int, context_lines: int = 12) -> Evidence:
        return self._local.read_region(path, line, context_lines=context_lines)
