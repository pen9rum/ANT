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
  * `compute(expression)`       -- deterministic arithmetic.

BACKENDS ARE INJECTED, NEVER DEFAULTED TO SOMETHING LIVE. A registry
built without a search backend does not silently return `[]` -- it
raises `ToolUnavailableError`. An empty result list and "there is no
search engine wired up" are completely different facts, and conflating
them is how a substrate misconfiguration turns into a fake accuracy
number. This is the same reasoning that removed the ungrounded local-BM25
fallback from `web_navigation_tool.py` (see that module's own history).

ZERO-INFERENCE NOTE: nothing in this module calls an LLM. The compute
primitive is a restricted AST evaluator, not `exec`; see `compute()`.
"""

from __future__ import annotations

import ast
import json
import operator
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ant.domain.models import Evidence
from ant.evaluation_suite.gaia_scope import (
    GaiaEnvironment,
    GaiaSubstrateError,
    TableView,
)

TOOL_SEARCH = "search"
TOOL_OPEN_URL = "open_url"
TOOL_INSPECT_FILE = "inspect_file"
TOOL_INSPECT_TABLE = "inspect_table"
TOOL_COMPUTE = "compute"

#: Canonical, ordered tool surface. Ordered (not a set) so that any
#: prompt rendering built from it is byte-stable across runs -- a set's
#: iteration order would make otherwise-identical runs diff.
GAIA_TOOL_NAMES: tuple[str, ...] = (
    TOOL_SEARCH,
    TOOL_OPEN_URL,
    TOOL_INSPECT_FILE,
    TOOL_INSPECT_TABLE,
    TOOL_COMPUTE,
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
                "name": TOOL_COMPUTE,
                "description": "Evaluate an arithmetic expression. Args: expression (str).",
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

    def compute(self, expression: str) -> str:
        """Deterministic arithmetic over a RESTRICTED expression grammar.

        This is NOT a Python sandbox and deliberately does not try to be
        one. GAIA does include tasks that would benefit from running
        arbitrary code, but standing up a genuinely safe execution
        sandbox is its own reviewed piece of work -- and an unsafe one
        wired into an agent loop is an arbitrary-code-execution primitive
        driven by model output. So this pass ships the safe subset and
        declares the gap rather than shipping `exec` and hoping.

        Supported: numeric literals, `+ - * / // % **`, unary `+`/`-`,
        comparisons, and parentheses. Everything else -- names, calls,
        attributes, subscripts, imports, comprehensions -- raises
        `ValueError`, which is recorded as a failed call rather than
        silently returning something wrong.
        """
        arguments = {"expression": expression}
        try:
            value = _safe_eval(expression)
        except Exception as exc:
            self._record(TOOL_COMPUTE, arguments, ok=False, summary="", error=str(exc))
            raise
        rendered = repr(value)
        self._record(TOOL_COMPUTE, arguments, ok=True, summary=rendered)
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


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
# Caps an expression like `9**9**9`, which is syntactically tiny but
# would otherwise hang the process building an enormous integer.
_MAX_POW_EXPONENT = 1_000


def _safe_eval(expression: str) -> Any:
    tree = ast.parse(expression, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Only numeric literals are allowed, got {node.value!r}.")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _eval_node(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp):
        handler = _BIN_OPS.get(type(node.op))
        if handler is None:
            raise ValueError(f"Operator {type(node.op).__name__} is not allowed.")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
            raise ValueError(f"Exponent {right} exceeds the allowed ceiling.")
        return handler(left, right)
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            handler = _CMP_OPS.get(type(op))
            if handler is None:
                raise ValueError(f"Comparison {type(op).__name__} is not allowed.")
            right = _eval_node(comparator)
            if not handler(left, right):
                return False
            left = right
        return True
    raise ValueError(
        f"Expression node {type(node).__name__} is not allowed -- compute() evaluates "
        "arithmetic only, never names, calls, or attribute access."
    )


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
    """

    def __init__(self, registry: GaiaToolRegistry) -> None:
        self.registry = registry

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
            for hit in self.registry.search(query, limit=limit):
                evidence.append(
                    Evidence(
                        path=hit.url,
                        line_start=0,
                        line_end=0,
                        quote=hit.snippet,
                        reason=f"web search hit: {hit.title}",
                    )
                )
        if self.registry.environment.has_attachment():
            evidence.extend(self._attachment_evidence(query, limit))
        return evidence[:limit]

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
