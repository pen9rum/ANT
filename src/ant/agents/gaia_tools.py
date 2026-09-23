"""The GAIA substrate's SHARED tool registry -- one object, one capability
set, consumed identically by every method this suite compares.

THE FAIRNESS INVARIANT THIS MODULE EXISTS TO ENFORCE:

    ANTMAN must never hold a capability a Matched-ReAct-shaped baseline
    over the same substrate would not also hold.

That is enforced structurally rather than by review discipline.
`GaiaToolRegistry` is the ONLY object in this substrate that can reach a
search backend, a fetch backend, an attachment, or the compute
primitive. A ReAct-shaped agent consumes it by naming tools from
`available_tools()` and calling `invoke()`; ANTMAN consumes it by being
handed `GaiaCoordinatorSearchTool`, which is a thin duck-typed *adapter
over the very same registry instance* -- it owns no backend of its own
and adds no method that reaches around the registry. There is therefore
no capability either shape can express that the other cannot, and
`tests/test_gaia_tools.py` asserts it on the live objects rather than
trusting this paragraph.

WHY A REGISTRY RATHER THAN TWO PARALLEL TOOLBOXES: the web substrate
learned this the hard way. `matched_react_web.py` and
`web_navigation_tool.py` each construct their own access path over
`EvalWebEnvironment`, which is fine but means "do both methods really
see the same web?" is answerable only by reading both files carefully.
Here the question is answerable by identity: `registry is registry`.

TOOL SURFACE (the five primitives the GAIA capability audit calls for):
  * `search(query, limit)`      -- open-web search.
  * `open_url(url)`             -- fetch one page as text.
  * `inspect_file(offset, limit)` -- read the task's attachment as text.
  * `inspect_table(...)`        -- read the task's attachment as rows.
  * `run_python(code)`          -- execute Python in an isolated sandbox.

THE SANDBOX IS THE SECURITY-SENSITIVE PART OF THIS FILE. `run_python`
replaced the restricted-AST `compute()` this substrate originally
shipped, which means the registry now holds a genuine
arbitrary-code-execution primitive driven by model output.
`ant.evaluation_suite.gaia_sandbox`'s module docstring is the mechanism
writeup AND the explicit list of what the isolation does not cover; it
should be read before this tool is pointed at a live run. Note that the
execution limits live on the REGISTRY (`python_timeout_seconds`,
`python_memory_bytes`), not on the caller, so the fairness invariant
above extends to compute budget and not just to capability names.

The sandbox is also deliberately unable to reach the task's attachment:
its scratch directory starts empty. If `run_python` could open the
attachment path it would become a way around the modality policy -- a
method could read an image's bytes and claim a perception capability the
substrate declares UNSUPPORTED for everyone.

BACKENDS ARE INJECTED, NEVER DEFAULTED TO SOMETHING LIVE. A registry
built without a search backend does not silently return `[]` -- it
raises `ToolUnavailableError`. An empty result list and "there is no
search engine wired up" are completely different facts, and conflating
them is how a substrate misconfiguration turns into a fake accuracy
number. This is the same reasoning that removed the ungrounded local-BM25
fallback from `web_navigation_tool.py` (see that module's own history).

ZERO-INFERENCE NOTE: nothing in this module calls an LLM. `run_python`
executes model-CHOSEN code, but it executes it in a subprocess with no
network and no model access -- there is no API key in that child's
environment by construction (`gaia_sandbox._build_child_environment`
builds an allowlist, never a copy of `os.environ`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ant.domain.models import Evidence
from ant.evaluation_suite.gaia_sandbox import (
    DEFAULT_MEMORY_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    SandboxResult,
    run_python,
)
from ant.evaluation_suite.gaia_scope import (
    GaiaEnvironment,
    GaiaSubstrateError,
    TableView,
)
from ant.providers.openai_provider import _loads_json_object

TOOL_SEARCH = "search"
TOOL_OPEN_URL = "open_url"
TOOL_INSPECT_FILE = "inspect_file"
TOOL_INSPECT_TABLE = "inspect_table"
#: Renamed from `compute` when the restricted-AST arithmetic evaluator was
#: replaced by a real sandboxed interpreter. The name change is
#: deliberate and not cosmetic: `compute(expression)` promised an
#: arithmetic expression, `run_python(code)` accepts a program, and a
#: tool whose name understates what it does is a documentation bug in a
#: security-relevant place.
TOOL_RUN_PYTHON = "run_python"

#: Canonical, ordered tool surface. Ordered (not a set) so that any
#: prompt rendering built from it is byte-stable across runs -- a set's
#: iteration order would make otherwise-identical runs diff.
GAIA_TOOL_NAMES: tuple[str, ...] = (
    TOOL_SEARCH,
    TOOL_OPEN_URL,
    TOOL_INSPECT_FILE,
    TOOL_INSPECT_TABLE,
    TOOL_RUN_PYTHON,
)


class ToolUnavailableError(GaiaSubstrateError):
    """A tool was called whose backend was never wired up. Deliberately an
    error rather than an empty result -- see the module docstring."""


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str


@runtime_checkable
class SearchBackend(Protocol):
    """Anything that can answer an open-web query. A `Protocol` rather
    than a base class so a test double, a cached offline corpus, and a
    live search API are all equally first-class -- the registry never
    needs to know which it holds."""

    def search(self, query: str, limit: int) -> Sequence[SearchHit]: ...


@runtime_checkable
class FetchBackend(Protocol):
    """Anything that can turn a URL into page text."""

    def fetch(self, url: str) -> str: ...


@dataclass(frozen=True)
class ToolCall:
    """One immutable record of one tool invocation.

    DETERMINISM IS THE POINT. There is deliberately no timestamp, no
    duration, and no object `repr()` in this record: two runs that make
    the same calls against the same backends produce byte-identical
    logs, so a log diff is evidence of a real behavioural change rather
    than noise. Wall-clock accounting already lives in `UsageStats`,
    which is where a caller should look for it.

    Failures are recorded, not swallowed: `ok=False` plus `error` is
    written BEFORE the exception is re-raised, so a crashed run's log
    still shows what was attempted.
    """

    seq: int
    tool: str
    arguments: dict[str, Any]
    ok: bool
    summary: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "tool": self.tool,
            # Sorted so the serialized form is stable regardless of the
            # order a caller happened to pass keyword arguments in.
            "arguments": dict(sorted(self.arguments.items())),
            "ok": self.ok,
            "summary": self.summary,
            "error": self.error,
        }


@dataclass
class GaiaToolRegistry:
    """The single capability surface for one GAIA task.

    Constructed once per task and shared by whichever agent runs it. Holds
    the task's `GaiaEnvironment` (attachment access) plus optional search
    and fetch backends. Holds NO question text and NO reference answer --
    there is no constructor slot for either, so no tool can condition on
    the gold answer even by accident.
    """

    environment: GaiaEnvironment
    search_backend: SearchBackend | None = None
    fetch_backend: FetchBackend | None = None
    max_search_results: int = 10
    #: Sandbox limits live on the REGISTRY, not on the caller, so every
    #: method that drives this registry gets the same wall clock and the
    #: same memory ceiling. If a caller could pass its own timeout,
    #: "ANTMAN was allowed 60s of compute and the baseline 10s" would be
    #: a silent fairness break rather than a visible configuration.
    python_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    python_memory_bytes: int = DEFAULT_MEMORY_BYTES
    _calls: list[ToolCall] = field(default_factory=list, repr=False)

    # ---- introspection -------------------------------------------------

    def available_tools(self) -> tuple[str, ...]:
        """The full declared surface, independent of whether a backend is
        wired. A tool whose backend is missing still EXISTS and still
        fails loudly when called; hiding it here would let a
        misconfiguration read as "this method chose not to search"."""
        return GAIA_TOOL_NAMES

    def tool_specs(self) -> list[dict[str, Any]]:
        """Machine-readable descriptions, for building a ReAct prompt or a
        worker card. Both agent shapes render from THIS, so neither can
        advertise a tool the other lacks."""
        has_attachment = self.environment.has_attachment()
        return [
            {
                "name": TOOL_SEARCH,
                "description": "Search the open web. Args: query (str), limit (int).",
                "available": self.search_backend is not None,
            },
            {
                "name": TOOL_OPEN_URL,
                "description": "Fetch one URL and return its text. Args: url (str).",
                "available": self.fetch_backend is not None,
            },
            {
                "name": TOOL_INSPECT_FILE,
                "description": "Read this task's attachment as text. Args: offset, limit (int).",
                "available": has_attachment,
            },
            {
                "name": TOOL_INSPECT_TABLE,
                "description": "Read this task's attachment as rows. Args: max_rows (int).",
                "available": has_attachment,
            },
            {
                "name": TOOL_RUN_PYTHON,
                "description": (
                    "Run a Python program in an isolated, network-free, stdlib-only "
                    "sandbox and return its output. The value of a trailing bare "
                    "expression is reported, so a final `print()` is optional. "
                    f"Limits: {self.python_timeout_seconds:g}s wall clock, "
                    f"{self.python_memory_bytes // (1024 * 1024)} MiB memory, no "
                    "filesystem access outside an empty scratch directory (the task's "
                    "attachment is NOT reachable from here -- use inspect_file / "
                    "inspect_table). Args: code (str)."
                ),
                "available": True,
            },
        ]

    # ---- call log ------------------------------------------------------

    def call_log(self) -> list[ToolCall]:
        return list(self._calls)

    def log_as_dicts(self) -> list[dict[str, Any]]:
        return [call.to_dict() for call in self._calls]

    def tool_call_count(self) -> int:
        return len(self._calls)

    def _record(
        self,
        tool: str,
        arguments: dict[str, Any],
        ok: bool,
        summary: str,
        error: str | None = None,
    ) -> None:
        self._calls.append(
            ToolCall(
                seq=len(self._calls),
                tool=tool,
                arguments=arguments,
                ok=ok,
                summary=summary,
                error=error,
            )
        )

    # ---- the five primitives -------------------------------------------

    def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        arguments = {"query": query, "limit": limit}
        if self.search_backend is None:
            message = (
                "No search backend is wired into this GaiaToolRegistry. This is a "
                "substrate configuration error, not an empty result set."
            )
            self._record(TOOL_SEARCH, arguments, ok=False, summary="", error=message)
            raise ToolUnavailableError(message)
        capped = max(1, min(int(limit), self.max_search_results))
        try:
            hits = list(self.search_backend.search(query, capped))
        except Exception as exc:
            self._record(TOOL_SEARCH, arguments, ok=False, summary="", error=repr(exc))
            raise
        self._record(TOOL_SEARCH, arguments, ok=True, summary=f"{len(hits)} hits")
        return hits

    def open_url(self, url: str) -> str:
        arguments = {"url": url}
        if self.fetch_backend is None:
            message = (
                "No fetch backend is wired into this GaiaToolRegistry. This is a "
                "substrate configuration error, not an empty page."
            )
            self._record(TOOL_OPEN_URL, arguments, ok=False, summary="", error=message)
            raise ToolUnavailableError(message)
        try:
            text = self.fetch_backend.fetch(url)
        except Exception as exc:
            self._record(TOOL_OPEN_URL, arguments, ok=False, summary="", error=repr(exc))
            raise
        self._record(TOOL_OPEN_URL, arguments, ok=True, summary=f"{len(text)} chars")
        return text

    def inspect_file(self, offset: int = 0, limit: int = 20_000) -> str:
        """Read the task's attachment as text. Propagates
        `UnsupportedModalityError` unchanged -- an image or audio
        attachment fails here explicitly, by design."""
        arguments = {"offset": offset, "limit": limit}
        try:
            text = self.environment.read_text()
        except GaiaSubstrateError as exc:
            self._record(TOOL_INSPECT_FILE, arguments, ok=False, summary="", error=str(exc))
            raise
        window = text[offset : offset + limit]
        self._record(
            TOOL_INSPECT_FILE,
            arguments,
            ok=True,
            summary=f"{len(window)} chars of {len(text)}",
        )
        return window

    def inspect_table(self, max_rows: int = 200) -> TableView:
        arguments = {"max_rows": max_rows}
        try:
            view = self.environment.read_table(max_rows=max_rows)
        except GaiaSubstrateError as exc:
            self._record(TOOL_INSPECT_TABLE, arguments, ok=False, summary="", error=str(exc))
            raise
        self._record(
            TOOL_INSPECT_TABLE,
            arguments,
            ok=True,
            summary=f"{len(view.rows)} rows from {view.sheet_name}",
        )
        return view

    def run_python(self, code: str) -> str:
        """Execute a Python program in the isolated sandbox and return
        its rendered output.

        **Read `ant.evaluation_suite.gaia_sandbox`'s module docstring
        before relying on this.** It is a real arbitrary-code-execution
        primitive driven by model output, and that docstring is the
        security writeup -- including an explicit list of what the
        isolation does NOT protect against.

        DOES NOT RAISE on user-code failure. A syntax error, an
        exception, a timeout and a denied operation all come back as a
        rendered string, because an agent has to be able to *read* its
        own failure to correct itself; raising would turn an ordinary
        debugging loop into a dead end. The call is still logged with
        `ok=False`, so a failed program is never mistaken for a
        successful one when the log is read.

        The limits are the registry's, not the caller's -- see
        `python_timeout_seconds` / `python_memory_bytes`. A caller-chosen
        budget would be a fairness hole.
        """
        arguments = {"code": code}
        result: SandboxResult = run_python(
            code,
            timeout_seconds=self.python_timeout_seconds,
            memory_bytes=self.python_memory_bytes,
        )
        rendered = result.render()
        self._record(
            TOOL_RUN_PYTHON,
            arguments,
            ok=result.ok,
            # A BOUNDED, non-varying summary: the rendered output can be
            # tens of kilobytes and would swamp a log diff, and anything
            # derived from wall-clock time would break the byte-identical
            # log guarantee `ToolCall` exists to provide.
            summary=f"{len(rendered)} chars" if result.ok else "",
            error=None if result.ok else (result.error or "sandboxed execution failed"),
        )
        return rendered

    # ---- uniform dispatch (what a ReAct-shaped agent drives) -----------

    def invoke(self, tool: str, **kwargs: Any) -> Any:
        """Name-based dispatch, so a ReAct loop can act on a model-chosen
        tool name without a hand-written if/elif chain that could drift
        out of sync with `available_tools()`."""
        if tool not in GAIA_TOOL_NAMES:
            raise ToolUnavailableError(
                f"Unknown GAIA tool {tool!r}. Available: {list(GAIA_TOOL_NAMES)}"
            )
        return getattr(self, tool)(**kwargs)


class GaiaCoordinatorSearchTool:
    """Duck-typed stand-in for `ant.tools.local.LocalSearchTool`, injected
    into an UNMODIFIED `LocalCoordinator` through its `search_tool_factory`
    constructor parameter (read that parameter's own comment in
    `coordinator/local.py` for the seam's contract).

    It wraps a `GaiaToolRegistry` INSTANCE -- it does not build one, does
    not hold a backend, and has no path to a capability the registry
    lacks. That is what makes "ANTMAN and Matched ReAct share one
    substrate" a structural fact rather than a convention.

    On the symbol-oriented half of `LocalSearchTool`'s surface
    (`rank_symbols`/`callers`/`callees`/`subclasses`/...): these are
    code-navigation primitives with no GAIA analogue -- there is no call
    graph in a spreadsheet or a web page. They return empty results
    HONESTLY rather than being faked into keyword matches that would look
    like structural evidence to the coordinator. This is the same
    judgement `web_navigation_tool.py` made, reached independently for the
    same reason; the difference is that it could delegate to a real
    `LocalSearchTool` over materialized page files, whereas GAIA has no
    single materialized corpus to delegate to.

    THE `reasoner` PARAMETER -- WHY IT EXISTS: `rank_symbols()` always
    returning `[]` (above) has a second consequence beyond "no structural
    evidence": `AutonomousWorker.run()` only routes to `worker_reasoner`
    (`select_lookups`/`plan_worker_actions`) when `candidate_symbols` is
    non-empty, so on this substrate those calls NEVER fire and a
    worker-model swap (ANTMAN-H's Qwen3-8B) is otherwise architecturally
    inert here -- `AutonomousWorker.run()`'s own `self.tools.search(...)`
    call is unconditional and un-reasoned for every substrate, repo-QA
    included; repo-QA just also has a second, symbol-gated call site that
    happens to carry the swap. GAIA has no symbol-gated call site to piggy
    back on, so the swap has to live in the one call site GAIA does have:
    this method. When `reasoner` is set, `search()` uses it to (1) turn
    the coordinator's need text into a search query instead of forwarding
    it verbatim, and (2) decide which, if any, hits are worth fetching in
    full via `open_url` -- both real local-reasoning decisions, not a new
    capability: a ReAct-shaped agent over the same registry already makes
    both decisions itself, per tool-call, with its own model. `reasoner`
    defaults to `None` and every call below degrades to the prior
    deterministic behaviour on `None` or on any reasoning-call failure, so
    this is additive and never changes the no-`reasoner` runtime.
    """

    def __init__(self, registry: GaiaToolRegistry, reasoner: Any | None = None) -> None:
        self.registry = registry
        self.reasoner = reasoner

    def search(
        self, query: str, files: list[str], limit: int = 8, context_lines: int = 6
    ) -> list[Evidence]:
        """Maps the coordinator's file-oriented search onto GAIA's
        heterogeneous sources, returning one `Evidence` per hit. `files`
        is accepted for interface compatibility and ignored: GAIA has no
        repository file universe to scope a query to."""
        del files, context_lines
        evidence: list[Evidence] = []
        if self.registry.search_backend is not None:
            search_query = query
            if self.reasoner is not None:
                search_query = self._reasoned_query(query)
            hits = self.registry.search(search_query, limit=limit)
            for hit in hits:
                evidence.append(
                    Evidence(
                        path=hit.url,
                        line_start=0,
                        line_end=0,
                        quote=hit.snippet,
                        reason=f"web search hit: {hit.title}",
                    )
                )
            if self.reasoner is not None and hits and self.registry.fetch_backend is not None:
                evidence.extend(self._reasoned_follow_up(query, hits, limit))
        if self.registry.environment.has_attachment():
            evidence.extend(self._attachment_evidence(query, limit))
        return evidence[:limit]

    def _reasoned_query(self, need: str) -> str:
        """Worker-local reasoning step 1 (see `reasoner` note on the class
        docstring): let the worker model rewrite `need` into a search
        query. Falls back to `need` verbatim on any failure -- a bad
        reasoning call must degrade to the old deterministic behaviour,
        never abort the search."""
        prompt = (
            "You are a search assistant. Rewrite the following information "
            "need as a single, effective web search query. Reply with ONLY "
            "the query text -- no quotes, no explanation.\n\n"
            f"Information need: {need}"
        )
        try:
            refined = self.reasoner.responses_text(prompt, max_output_tokens=64).text.strip()
        except Exception:
            return need
        return refined.strip('"') or need

    def _reasoned_follow_up(
        self, need: str, hits: list[SearchHit], limit: int
    ) -> list[Evidence]:
        """Worker-local reasoning step 2 (see `reasoner` note on the class
        docstring): decide which (0-2) search hits are worth fetching in
        full via `open_url`, instead of only ever returning search-engine
        snippets. Bounded to 2 fetches per call regardless of `limit`, so
        the added cost/latency stays comparable across worker models."""
        listing = "\n".join(
            f"{i}. {hit.title} -- {hit.url}\n   {hit.snippet[:200]}"
            for i, hit in enumerate(hits)
        )
        prompt = (
            "Given this information need and these web search results, decide "
            "which results (0 to 2 of them) are worth opening in full to answer "
            'the need. Reply with ONLY a JSON object: {"open": [<indices>]}.\n\n'
            f"Information need: {need}\n\nSearch results:\n{listing}"
        )
        try:
            decision = _loads_json_object(
                self.reasoner.responses_json(prompt, max_output_tokens=64).text
            )
            indices = decision.get("open", [])
        except Exception:
            return []
        out: list[Evidence] = []
        for index in indices:
            if len(out) >= 2 or len(out) >= limit:
                break
            if not isinstance(index, int) or not (0 <= index < len(hits)):
                continue
            hit = hits[index]
            try:
                text = self.registry.open_url(hit.url)
            except Exception:
                continue
            out.append(
                Evidence(
                    path=hit.url,
                    line_start=0,
                    line_end=0,
                    quote=text[:2000],
                    reason=f"worker-reasoned follow-up fetch: {hit.title}",
                )
            )
        return out

    def _attachment_evidence(self, query: str, limit: int) -> list[Evidence]:
        """Keyword-locates query terms inside the attachment. Failures are
        swallowed HERE and only here, deliberately: an unsupported-modality
        attachment must not abort a query whose answer may live entirely on
        the web. The failure is still visible -- `inspect_file` already
        wrote an `ok=False` entry to the registry's call log before
        raising, so the capability gap is recorded, not erased."""
        try:
            text = self.registry.inspect_file()
        except GaiaSubstrateError:
            return []
        name = self.registry.environment.file_name or "attachment"
        terms = [term for term in query.lower().split() if len(term) > 2]
        out: list[Evidence] = []
        for index, line in enumerate(text.splitlines()):
            if any(term in line.lower() for term in terms):
                out.append(
                    Evidence(
                        path=name,
                        line_start=index + 1,
                        line_end=index + 1,
                        quote=line.strip()[:500],
                        reason="attachment keyword match",
                    )
                )
                if len(out) >= limit:
                    break
        return out

    def dense_search(self, query: str, files: list[str], limit: int = 4) -> list[Evidence]:
        """No embedding index exists for this substrate. Returns [] --
        the same honest no-op `web_navigation_tool.dense_search` resolves
        to, and for the same reason: a fabricated dense result would be
        ungrounded evidence entering the coordinator."""
        del query, files, limit
        return []

    # --- code-navigation surface with no GAIA analogue (see class docstring) ---

    def rank_symbols(
        self, symbols: list[str], files: list[str], limit: int = 8, need: str = ""
    ) -> list[str]:
        del symbols, files, limit, need
        return []

    def resolve_symbol(
        self, symbol: str, files: list[str], limit: int = 6, need: str = ""
    ) -> list[Evidence]:
        del symbol, files, limit, need
        return []

    def navigate(self, symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        del symbol, files, limit
        return []

    def references(self, symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        del symbol, files, limit
        return []

    def indexed_callers(self, symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        del symbol, files, limit
        return []

    def callers(self, symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        del symbol, files, limit
        return []

    def callees(self, symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        del symbol, files, limit
        return []

    def assignments(self, symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        del symbol, files, limit
        return []

    def imports(self, module_or_symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        del module_or_symbol, files, limit
        return []

    def subclasses(self, symbol: str, files: list[str], limit: int = 8) -> list[Evidence]:
        del symbol, files, limit
        return []

    def read_region(self, path: str, line: int, context_lines: int = 12) -> Evidence:
        """Re-reads a window of the attachment around `line`. The one
        code-navigation primitive that DOES have a GAIA analogue -- "show
        me more around this hit" is meaningful for a CSV or a document."""
        try:
            text = self.registry.inspect_file()
        except GaiaSubstrateError as exc:
            return Evidence(
                path=path, line_start=line, line_end=line, quote="", reason=f"unavailable: {exc}"
            )
        lines = text.splitlines()
        start = max(0, line - 1 - context_lines)
        end = min(len(lines), line + context_lines)
        return Evidence(
            path=path,
            line_start=start + 1,
            line_end=end,
            quote="\n".join(lines[start:end]),
            reason="attachment region",
        )


def build_gaia_search_tool_factory(registry: GaiaToolRegistry):
    """Returns a `search_tool_factory`-shaped callable bound to THIS
    registry, for handing to `LocalCoordinator(search_tool_factory=...)`.

    The `(repo_root, index_path)` arguments the coordinator passes are
    accepted and ignored -- GAIA has neither -- which is exactly why the
    seam is a factory callable rather than a subclass: the coordinator's
    own core logic stays untouched and unaware that this substrate has no
    repository.
    """

    def factory(repo_root, index_path):  # noqa: ANN001 - shape fixed by the coordinator seam
        del repo_root, index_path
        return GaiaCoordinatorSearchTool(registry)

    return factory


def serialize_call_log(registry: GaiaToolRegistry) -> str:
    """Canonical JSON of the call log -- sorted keys, no whitespace
    jitter -- so two runs can be compared with a byte diff."""
    return json.dumps(registry.log_as_dicts(), sort_keys=True, indent=2)
