"""Web-specific substitute for `ant.tools.local.LocalSearchTool`, injected
into an unmodified `LocalCoordinator` via its `search_tool_factory` seam
(see `LocalCoordinator.__init__`'s own docstring on that parameter).

WHY THIS EXISTS / HISTORY: the first version of `AntWebAgent` built a
frozen, root+one-hop worker roster at bootstrap time -- confirmed live
(24-task run, 0/24) that answers are essentially always 2+ hops deep. The
next version added live navigation gated behind a per-call hop cap plus a
crude term-overlap sufficiency check -- confirmed live (10-task
validation, 1/10) both were wrong: the cap capped depth even when budget
remained, and the crude check called nearly anything "sufficient"
immediately (a generic entity name substring-matches almost every page).
The version after that replaced term-overlap with a single grounded
resolved/continue/dead_end decision per hop -- confirmed live via a
5-task forensic ablation (decision_log audit, see this module's own git
history) that THIS was also wrong in a different way: the schema
conflated "how much progress was made" with "what to do next" into one
status field, so a worker that found a genuinely useful but incomplete
fact had only two bad options -- overclaim "resolved" (rejected by
grounding more than half the time it was tried) or say "continue" (79%
of all decisions across the suite), with no way to report "I found
something real, it's not enough, here it is anyway." The forensic audit
also found an evidence-provenance leak: when the decision loop ended
without a grounded resolve, search() fell back to plain ungrounded local
BM25 results, which is how tasks with ZERO grounded spans still had
dozens of items reach the coordinator and several reach final synthesis.

THE FIX (this version): two orthogonal fields replace the single status.
`progress` ("none" | "partial" | "resolved") is how much the CURRENT
PAGE's own content contributes toward the Need, always verified by
`_verify_and_build_evidence` against real page text before being
trusted -- never taken on the model's word. `action` ("continue" |
"return" | "dead_end") is what happens next, independent of how much
progress was made. Valid combinations: (none, continue), (partial,
continue), (partial, return), (resolved, return), (none, dead_end) --
"resolved" is never paired with "continue". The ungrounded local-search
fallback is gone entirely: search() now returns exactly the grounded
evidence accumulated this dispatch (across every hop it took, chained
within one call -- "preserved across later hops"), nothing else, even if
that is an empty list. This costs exactly the same one call per hop the
earlier single-status design already made -- a richer response schema on
the same call, not an extra one.

Once grounded evidence flows cleanly (no ungrounded noise, correctly
need-scoped via LocalCoordinator's own need_ids stamping), ANTMAN's
existing, UNTOUCHED core machinery does the rest: `check_need_resolution`
already has a resolved/partial/unresolved judgment
(src/ant/providers/base.py) that operates on the CUMULATIVE evidence
pool for a need across every round that has touched it -- runtime Need
revision/rerouting was already built to consume exactly this. This file
supplies clean grounded evidence; it does not need to (and does not)
reimplement any of that.

HOW: `AutonomousWorker.run()` (src/ant/workers/autonomous.py) calls
`self.tools.search(...)` and `.dense_search(...)` UNCONDITIONALLY at the
start of every worker execution. A worker's first hop into its own
assigned first-level link (its territory specialization, established at
bootstrap) is unconditional and deterministic, never LLM-chosen --
confirmed live that gating even this first hop behind any "is it needed"
heuristic left workers stuck on root. From there, the structured
decision loop chains as many further hops as the shared budget and
available links allow -- no per-call depth cap -- until the model
chooses to stop (`return`/`dead_end`) or the budget runs out.

Every real navigation goes through the SAME `EvalWebEnvironment`
instance for the whole query (shared across every worker), so its
`max_steps` budget is enforced globally, not per-worker. It also
independently enforces "only a link actually present on the current
page" (raises ValueError otherwise) -- this class relies on that
enforcement rather than re-implementing it. Nothing here ever reads a
question's reference answer, golden_path, or source_websites.

Everything else (rank_symbols/resolve_symbol/navigate[symbol]/
references/indexed_callers/callers/callees/assignments/imports/
subclasses/read_region) is delegated unchanged to a wrapped
LocalSearchTool over the materialized-pages directory: those pages are
plain local .txt files once fetched, so the existing code-oriented tool
implementation already reads them correctly. `dense_search()` also
delegates, but is a genuine no-op for this substrate: AntWebAgent never
supplies an `index_path`, and `LocalSearchTool.dense_search` returns []
immediately whenever `self.index_path` is falsy -- confirmed by direct
reading of that method, not assumed, so this delegation is NOT a second
evidence-provenance leak.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from ant.domain import Evidence
from ant.evaluation_suite.web_fetch import FetchedPage, PageLink
from ant.evaluation_suite.web_scope import EvalWebEnvironment
from ant.tools.local import LocalSearchTool

# Deliberately NO cap on how many hops one search() call may chase -- a
# worker must be able to go anchor -> child -> grandchild -> ... within a
# single dispatch, stopping only when the model itself chooses to (action
# "return"/"dead_end") or the shared nav_budget runs out. No infinite-loop
# risk: frontier.visited_urls/exhausted_links only grow, so a page's own
# (finite) link list monotonically shrinks as candidates, and
# EvalWebEnvironment's own step budget is a hard, finite ceiling anyway.
# How many of a page's own links get shown to the decision call -- a
# prompt-size guard, not a relevance filter (candidates beyond this cap
# are simply never offered, in extraction order, the same "never sort by
# relevance" discipline the bootstrap link roster itself uses).
MAX_LINK_CANDIDATES_SHOWN = 40
# How much of the current page's own text gets shown to the decision
# call -- a prompt-size guard only, consistent with this codebase's own
# existing convention of bounding an individual evidence quote/region
# (e.g. LocalSearchTool.read_region's own [:2400] cap), not a fidelity
# regression on top of it: the full, untruncated page is still what gets
# materialized and BM25-searched elsewhere.
MAX_PAGE_TEXT_SHOWN = 6000

# The only (progress, action) pairs a decision may legally report.
# "resolved" is never paired with "continue" -- once the model claims
# full resolution it must hand back to the coordinator, not keep
# wandering with an unresolved Need it just claimed to have closed.
VALID_PROGRESS_ACTION_PAIRS = {
    ("none", "continue"),
    ("partial", "continue"),
    ("partial", "return"),
    ("resolved", "return"),
    ("none", "dead_end"),
}


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
    current_filename: str | None = None  # materialized file backing current_page, for Evidence.path
    visited_urls: set[str] = field(default_factory=set)
    exhausted_links: set[str] = field(default_factory=set)
    taken_first_hop: bool = False
    next_file_index: int = 0


@dataclass
class WorkerDecision:
    progress: str  # "none" | "partial" | "resolved"
    action: str  # "continue" | "return" | "dead_end"
    evidence: list[dict]
    next_link_index: int | None
    reason: str


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


def _normalize_for_grounding(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _find_grounded_span(page_text: str, supporting_text: str) -> tuple[int, int] | None:
    """Deterministically checks that `supporting_text` is actually present
    in (or a whitespace/case-normalized match of) `page_text`, and if so
    returns its (1-indexed) line span. A small growing-window line search
    rather than a single exact substring check: tolerates a claim quoted
    with different whitespace/line-wrapping than the source without
    requiring a full normalized-offset remapping of the whole page.
    Returns None if the claim cannot be located -- the caller must then
    treat it as unverified/hallucinated, never as grounded.
    """
    normalized_target = _normalize_for_grounding(supporting_text)
    if not normalized_target:
        return None
    lines = page_text.splitlines()
    for start in range(len(lines)):
        for end in range(start, min(start + 8, len(lines))):
            window = " ".join(lines[start : end + 1])
            if normalized_target in _normalize_for_grounding(window):
                return start + 1, end + 1
    return None


def _parse_json_object(text: str) -> dict | None:
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


class WebSearchTool:
    def __init__(
        self,
        materialized_dir: Path,
        env: EvalWebEnvironment,
        provider,
        frontiers: dict[str, WorkerFrontier],
        cache_enabled: bool = True,
    ) -> None:
        self._local = LocalSearchTool(materialized_dir, index_path=None)
        self._materialized_dir = materialized_dir
        self._env = env
        self._provider = provider
        self._frontiers = frontiers
        self.nav_log: list[dict] = []
        # cache_enabled=False is a forensic/ablation knob only (see
        # _decide's own docstring on why the cache is semantically sound
        # by construction) -- prompts, grounding, and navigation
        # semantics are completely unaffected either way; this only
        # controls whether an identical (need, page_state) triple skips
        # the LLM call or asks it again.
        self._cache_enabled = cache_enabled
        self._decision_cache: dict[tuple[str, str, tuple[str, ...]], WorkerDecision] = {}
        # Forensic instrumentation: one entry per _decide() call (cache
        # hit or fresh), independent of nav_log (which only records
        # actual navigations -- a dead_end/invalid/resolved-without-a-hop
        # decision never appears there at all). n_evidence_grounded is
        # filled in by search() after _verify_and_build_evidence runs,
        # since grounding happens outside _decide() itself.
        self.decision_log: list[dict] = []

    # --- the two methods AutonomousWorker.run() calls unconditionally ---

    def search(
        self, query: str, files: list[str], limit: int = 8, context_lines: int = 6
    ) -> list[Evidence]:
        worker_key = worker_key_from_files(files)
        if worker_key is None:
            return []
        frontier = self._frontiers.get(worker_key)
        if frontier is None:
            return []

        # The worker's own assigned first-level link is its territory
        # specialization (established at bootstrap) -- it must actually be
        # visited at least once, unconditionally, the first time this
        # worker's search() is ever called.
        if not frontier.taken_first_hop:
            frontier.taken_first_hop = True  # at most one attempt, ever, regardless of outcome
            link = frontier.assigned_link
            if link is not None and self._budget_remaining():
                page = self._follow(worker_key, frontier, link, query)
                if page is not None:
                    self._materialize(worker_key, frontier, page, files)

        # Grounded, LLM-mediated resolution loop: each iteration makes ONE
        # structured (progress, action) decision call about whatever page
        # the worker is currently on (see this module's own top docstring
        # for the full design rationale). Grounding verification governs
        # ONLY whether an iteration's claimed evidence is accepted into
        # `accumulated` -- it never overrides the model's own `action`,
        # so a worker that (rightly or wrongly) decides to keep going
        # still gets to try the next page even if this page's claim
        # didn't pan out. Accumulates across every hop this dispatch
        # takes ("partial evidence preserved across later hops"); no
        # ungrounded fallback -- an empty return is a legitimate, honest
        # outcome, not something to paper over with noisy BM25 hits.
        accumulated: list[Evidence] = []
        while True:
            candidates = self._candidates_for(frontier)
            decision = self._decide(frontier, query, candidates, worker_key)
            if decision is None:
                break
            if decision.evidence:
                grounded = self._verify_and_build_evidence(frontier, decision, worker_key) or []
                self.decision_log[-1]["n_evidence_grounded"] = len(grounded)
                accumulated.extend(grounded)
            if decision.action == "return":
                return accumulated
            if decision.action == "dead_end":
                break
            # action == "continue"
            if not self._budget_remaining():
                break
            link = self._resolve_link_index(candidates, decision.next_link_index)
            if link is None:
                break
            page = self._follow(worker_key, frontier, link, query)
            if page is None:
                continue
            self._materialize(worker_key, frontier, page, files)

        return accumulated

    def dense_search(self, query: str, files: list[str], limit: int = 4) -> list[Evidence]:
        # No live-navigation side effect here: search() (called first,
        # unconditionally, every round -- autonomous.py:83 before :100)
        # already had the chance to navigate for this exact need. A
        # second independent navigation attempt from dense_search would
        # double-spend the shared budget for no new decision basis. Also
        # a genuine no-op for this substrate (see this module's own top
        # docstring): index_path=None means LocalSearchTool.dense_search
        # returns [] immediately, so this delegation is not a second
        # evidence-provenance leak.
        return self._local.dense_search(query, files, limit=limit)

    # --- navigation + grounded-decision internals ---

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
        frontier.current_filename = filename
        if filename not in files:
            files.append(filename)

    def _candidates_for(self, frontier: WorkerFrontier) -> list[PageLink]:
        return [
            link
            for link in frontier.current_page.links
            if link.url not in frontier.visited_urls and link.url not in frontier.exhausted_links
        ][:MAX_LINK_CANDIDATES_SHOWN]

    def _resolve_link_index(self, candidates: list[PageLink], index: int | None) -> PageLink | None:
        if index is None or not (0 <= index < len(candidates)):
            return None
        return candidates[index]

    def _decide(
        self, frontier: WorkerFrontier, query: str, candidates: list[PageLink], worker_key: str
    ) -> WorkerDecision | None:
        # Memoized by (need, page_url, exact candidate set): this triple
        # fully determines the prompt's content, so an identical triple
        # can only recur when NOTHING about the worker's situation
        # actually changed since the last time -- current_page only ever
        # advances via a real _follow() hop (never reverts), and
        # visited_urls/exhausted_links (which the candidate set is
        # filtered through) only change as a SIDE EFFECT of that same
        # hop. So if current_page is unchanged, the candidate set is
        # PROVABLY unchanged too, and the cached decision is not a stale
        # approximation -- it is the same input the model would see
        # again. _verify_and_build_evidence still re-runs its own cheap,
        # deterministic, local grounding check on every use regardless of
        # cache hit or miss (never trusts the cache for correctness, only
        # for skipping the call).
        cache_key = (query, frontier.current_page.url, tuple(link.url for link in candidates))

        def _log(
            *, progress: str, action: str, cache_hit: bool, decision: WorkerDecision | None
        ) -> None:
            self.decision_log.append(
                {
                    "worker": worker_key,
                    "need": query,
                    "page_url": frontier.current_page.url,
                    "cache_hit": cache_hit,
                    "progress": progress,
                    "action": action,
                    "n_evidence_proposed": len(decision.evidence) if decision else 0,
                    "n_evidence_grounded": None,  # filled in by search() after verification
                    "next_link_index": decision.next_link_index if decision else None,
                }
            )

        if self._cache_enabled:
            cached = self._decision_cache.get(cache_key)
            if cached is not None:
                _log(
                    progress=cached.progress, action=cached.action, cache_hit=True, decision=cached
                )
                return cached

        page_text = frontier.current_page.text[:MAX_PAGE_TEXT_SHOWN]
        listing = (
            "\n".join(f"{i}. {link.text!r} -> {link.url}" for i, link in enumerate(candidates))
            if candidates
            else "(no further links available on this page)"
        )
        prompt = (
            "You are gathering evidence from a real website to resolve a research need.\n\n"
            f"Need: {query}\n\n"
            f"Current page ({frontier.current_page.url}) content:\n---\n{page_text}\n---\n\n"
            "Available links on this page you may follow next (index. anchor text -> URL):\n"
            f"{listing}\n\n"
            "Decide TWO separate things:\n\n"
            "1. PROGRESS -- how much does the CURRENT PAGE's own content above contribute "
            "toward resolving the Need?\n"
            '   - "none": nothing on this page usefully supports the Need.\n'
            '   - "partial": this page contains a genuine, exactly-quotable fact that supports '
            "the Need, but does not by itself fully answer it.\n"
            '   - "resolved": this page\'s own content, on its own, is sufficient to fully '
            "answer the Need.\n"
            '   Only claim "partial" or "resolved" if you can quote EXACT text copied from the '
            "page above as support -- never invent, paraphrase, or infer evidence that is not "
            "literally present in the page content shown.\n\n"
            "2. ACTION -- what should happen next?\n"
            '   - "continue": follow one more link from this page (pick the single most '
            "promising).\n"
            '   - "return": stop searching from here and report back now.\n'
            '   - "dead_end": no evidence here and no further link looks worth trying.\n\n'
            "Valid (progress, action) combinations ONLY:\n"
            '  ("none", "continue") -- nothing found yet, this path still looks promising.\n'
            '  ("partial", "continue") -- found a useful fact, but still want to explore '
            "further from here for more.\n"
            '  ("partial", "return") -- found a useful fact, but this local path is not worth '
            "deepening further -- hand back what you have.\n"
            '  ("resolved", "return") -- found everything needed to fully answer the Need.\n'
            '  ("none", "dead_end") -- nothing found and no further link here looks relevant.\n'
            'Never combine "resolved" with "continue".\n\n'
            'Reply with ONLY a single JSON object, no other text: {"progress": "none" | '
            '"partial" | "resolved", "action": "continue" | "return" | "dead_end", "evidence": '
            '[{"claim": "...", "supporting_text": "...exact text copied from the page above..."}], '
            '"next_link_index": <integer or null>, "reason": "..."}\n'
            '"evidence" must be an empty list when progress is "none". "next_link_index" must be '
            'an integer index from the links list above when action is "continue", and null '
            "otherwise."
        )
        try:
            result = self._provider.responses_text(prompt, max_output_tokens=500)
        except Exception:  # noqa: BLE001 -- a decision-call failure must not crash the worker
            _log(progress="invalid", action="invalid", cache_hit=False, decision=None)
            return None
        data = _parse_json_object(result.text)
        if data is None:
            _log(progress="invalid", action="invalid", cache_hit=False, decision=None)
            return None
        raw_progress = data.get("progress")
        raw_action = data.get("action")
        if (raw_progress, raw_action) not in VALID_PROGRESS_ACTION_PAIRS:
            _log(progress="invalid", action="invalid", cache_hit=False, decision=None)
            return None
        progress = cast(str, raw_progress)
        action = cast(str, raw_action)
        evidence = data.get("evidence")
        if not isinstance(evidence, list) or progress == "none":
            evidence = []
        next_link_index = data.get("next_link_index")
        if not isinstance(next_link_index, int) or action != "continue":
            next_link_index = None
        decision = WorkerDecision(
            progress=progress,
            action=action,
            evidence=[item for item in evidence if isinstance(item, dict)],
            next_link_index=next_link_index,
            reason=str(data.get("reason", "")),
        )
        _log(progress=progress, action=action, cache_hit=False, decision=decision)
        # Only a SUCCESSFUL decision is cached -- a transient call/parse
        # failure (the `return None` paths above) must stay retryable on
        # the next identical (need, page_state), not get permanently
        # poisoned into "no decision ever" for that input.
        if self._cache_enabled:
            self._decision_cache[cache_key] = decision
        return decision

    def _verify_and_build_evidence(
        self, frontier: WorkerFrontier, decision: WorkerDecision, worker_key: str
    ) -> list[Evidence] | None:
        """Deterministically re-checks every claimed span against the real
        page text -- called regardless of whether `decision` came from a
        fresh call or a cache hit, so a cache hit is never trusted for
        correctness, only for skipping the LLM call. A span that fails
        this check is silently dropped (not inserted into worker
        evidence, never propagated, never reaches synthesis) -- this IS
        the "downgrade progress rather than pretending progress occurred"
        behavior: a claimed partial/resolved decision whose evidence
        fails grounding simply contributes nothing, regardless of what
        the model claimed.
        """
        page_text = frontier.current_page.text
        path = frontier.current_filename or ""
        built: list[Evidence] = []
        for item in decision.evidence:
            supporting_text = str(item.get("supporting_text", "")).strip()
            claim = str(item.get("claim", "")).strip()
            if not supporting_text or not path:
                continue
            span = _find_grounded_span(page_text, supporting_text)
            if span is None:
                continue  # hallucinated / not actually present on the page -- reject
            line_start, line_end = span
            built.append(
                Evidence(
                    path=path,
                    line_start=line_start,
                    line_end=line_end,
                    quote=supporting_text[:2400],
                    reason=claim or "Grounded worker decision.",
                    claim=claim,
                    worker_id=worker_key,
                )
            )
        return built or None

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
