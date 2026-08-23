from __future__ import annotations

import re
from dataclasses import dataclass

from ant.domain import (
    Evidence,
    ExecutionDiagnostic,
    UnresolvedNeed,
    WorkerAction,
    WorkerCard,
    WorkerObservation,
)
from ant.retrieval import STOP_WORDS, extract_terms, score_evidence
from ant.tools.local import LocalSearchTool

SYMBOL_RE = re.compile(r"\b(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)")
BASE_RE = re.compile(r"\bclass\s+[A-Za-z_][A-Za-z0-9_]*\(([^)]*)\)")
DOC_LABEL_RE = re.compile(r"^(Args|Example|Examples|Note|Notes|Return|Returns|Raises)$")


@dataclass(frozen=True)
class WorkerRunConfig:
    max_tool_calls: int = 11
    evidence_limit: int = 8


class AutonomousWorker:
    def __init__(self, repo_root, card: WorkerCard, tools: LocalSearchTool) -> None:
        self.repo_root = repo_root
        self.card = card
        self.tools = tools

    def run(self, need: str, config: WorkerRunConfig | None = None) -> WorkerObservation:
        config = config or WorkerRunConfig()
        actions: list[WorkerAction] = []
        evidence: list[Evidence] = []
        tool_calls = 0

        search_results = self.tools.search(need, self.card.files, limit=4)
        tool_calls += 1
        actions.append(
            WorkerAction(
                tool="search",
                query=need,
                result_count=len(search_results),
                rationale="Initial local BM25/symbol search for current need.",
            )
        )
        evidence.extend(search_results)

        # Runs every round alongside the lexical search above, not as a
        # worker-chosen option: a candidate whose wording shares nothing
        # lexically with `need` is invisible to search() no matter how the
        # need happened to be phrased. No-ops (returns []) when no embedding
        # index has been built for this repo.
        dense_results = self.tools.dense_search(need, self.card.files, limit=4)
        tool_calls += 1
        actions.append(
            WorkerAction(
                tool="dense_search",
                query=need,
                result_count=len(dense_results),
                rationale="Embedding-similarity search for paraphrase-only matches.",
            )
        )
        evidence.extend(dense_results)

        candidate_symbols = self.tools.rank_symbols(
            _candidate_symbols(need, evidence),
            self.card.files,
            limit=8,
            need=need,
        )
        flow_or_call_path = _asks_for_flow_or_call_path(need)
        if _asks_for_inheritance(need):
            for symbol in candidate_symbols:
                if tool_calls >= config.max_tool_calls:
                    break
                subclass_results = self.tools.subclasses(symbol, self.card.files, limit=4)
                tool_calls += 1
                actions.append(
                    WorkerAction(
                        tool="subclasses",
                        query=symbol,
                        result_count=len(subclass_results),
                        rationale=(
                            "Find class definitions that inherit from the surfaced base symbol."
                        ),
                    )
                )
                evidence.extend(subclass_results)

        if flow_or_call_path:
            for symbol in candidate_symbols[:4]:
                if tool_calls >= config.max_tool_calls:
                    break
                import_results = self.tools.imports(symbol, self.card.files, limit=2)
                tool_calls += 1
                actions.append(
                    WorkerAction(
                        tool="imports",
                        query=symbol,
                        result_count=len(import_results),
                        rationale=(
                            "Resolve imports before following call/data-flow across implementation."
                        ),
                    )
                )
                evidence.extend(import_results)
                if tool_calls >= config.max_tool_calls:
                    break
                caller_results = self.tools.indexed_callers(symbol, self.card.files, limit=2)
                if not caller_results:
                    caller_results = self.tools.callers(symbol, self.card.files, limit=2)
                tool_calls += 1
                actions.append(
                    WorkerAction(
                        tool="callers",
                        query=symbol,
                        result_count=len(caller_results),
                        rationale="Follow call path into blocks that invoke the current symbol.",
                    )
                )
                evidence.extend(caller_results)
                if tool_calls >= config.max_tool_calls:
                    break
                assignment_results = self.tools.assignments(symbol, self.card.files, limit=2)
                tool_calls += 1
                actions.append(
                    WorkerAction(
                        tool="assignments",
                        query=symbol,
                        result_count=len(assignment_results),
                        rationale=(
                            "Trace local data-flow uses and assignments for the current symbol."
                        ),
                    )
                )
                evidence.extend(assignment_results)

        for symbol in candidate_symbols:
            if tool_calls >= config.max_tool_calls:
                break
            nav_results = self.tools.resolve_symbol(symbol, self.card.files, limit=2, need=need)
            if not nav_results:
                nav_results = self.tools.navigate(symbol, self.card.files, limit=2)
            tool_calls += 1
            actions.append(
                WorkerAction(
                    tool="navigate",
                    query=symbol,
                    result_count=len(nav_results),
                    rationale=(
                        "Navigate to definitions or implementation blocks for surfaced symbol."
                    ),
                )
            )
            evidence.extend(nav_results)
            if tool_calls >= config.max_tool_calls:
                break
            ref_results = self.tools.references(symbol, self.card.files, limit=2)
            tool_calls += 1
            actions.append(
                WorkerAction(
                    tool="references",
                    query=symbol,
                    result_count=len(ref_results),
                    rationale="Find local references/call sites for navigated symbol.",
                )
            )
            evidence.extend(ref_results)
            if tool_calls >= config.max_tool_calls:
                break
            caller_results = self.tools.indexed_callers(symbol, self.card.files, limit=1)
            if not caller_results:
                caller_results = self.tools.callers(symbol, self.card.files, limit=1)
            tool_calls += 1
            actions.append(
                WorkerAction(
                    tool="callers",
                    query=symbol,
                    result_count=len(caller_results),
                    rationale="Find caller blocks that invoke the current symbol.",
                )
            )
            evidence.extend(caller_results)
            if tool_calls >= config.max_tool_calls:
                break
            callee_results = self.tools.callees(symbol, self.card.files, limit=1)
            tool_calls += 1
            actions.append(
                WorkerAction(
                    tool="callees",
                    query=symbol,
                    result_count=len(callee_results),
                    rationale="Navigate from the current implementation to called helpers.",
                )
            )
            evidence.extend(callee_results)
            if tool_calls >= config.max_tool_calls:
                break
            assignment_results = self.tools.assignments(symbol, self.card.files, limit=1)
            tool_calls += 1
            actions.append(
                WorkerAction(
                    tool="assignments",
                    query=symbol,
                    result_count=len(assignment_results),
                    rationale="Inspect local assignments related to the current symbol.",
                )
            )
            evidence.extend(assignment_results)

        evidence = _rank_evidence(_dedupe(evidence), need)[: config.evidence_limit]
        needs, diagnostics = _detect_unresolved_needs(
            card=self.card,
            need=need,
            evidence=evidence,
            actions=actions,
            budget_exhausted=tool_calls >= config.max_tool_calls,
        )
        return WorkerObservation(
            worker_id=self.card.id,
            territory_id=self.card.territory_id,
            evidence=evidence,
            unresolved_needs=needs,
            diagnostics=diagnostics,
            actions=actions,
            stop_reason="budget_exhausted" if tool_calls >= config.max_tool_calls else "complete",
        )


def _candidate_symbols(query: str, evidence: list[Evidence]) -> list[str]:
    ordered = []
    seen = set()

    def add(symbol: str) -> None:
        cleaned = "".join(ch for ch in symbol.strip() if ch.isalnum() or ch == "_")
        if len(cleaned) <= 3 or cleaned in seen or _is_non_navigation_token(cleaned):
            return
        if "_" in cleaned or cleaned[:1].isupper():
            ordered.append(cleaned)
            seen.add(cleaned)

    for item in evidence:
        for symbol in SYMBOL_RE.findall(item.quote):
            add(symbol)
        for bases in BASE_RE.findall(item.quote):
            for symbol in bases.split(","):
                add(symbol)

    for text in [query, *(item.quote for item in evidence)]:
        for token in text.replace("(", " ").replace(")", " ").replace(".", " ").split():
            add(token)
    return ordered[:8]


def _asks_for_inheritance(need: str) -> bool:
    lowered = need.lower()
    indicators = [
        "subclass",
        "subclasses",
        "inherit",
        "inherits",
        "inherited",
        "base class",
        "derived",
    ]
    return any(indicator in lowered for indicator in indicators)


def _asks_for_flow_or_call_path(need: str) -> bool:
    lowered = need.lower()
    indicators = [
        "call path",
        "code path",
        "data flow",
        "data-flow",
        "flow",
        "pipeline",
        "through",
        "from",
        "into",
        "sample",
        "probabilities",
        "frequencies",
        "measurement",
        "handled",
    ]
    return any(indicator in lowered for indicator in indicators)


def _detect_unresolved_needs(
    *,
    card: WorkerCard,
    need: str,
    evidence: list[Evidence],
    actions: list[WorkerAction],
    budget_exhausted: bool,
) -> tuple[list[UnresolvedNeed], list[ExecutionDiagnostic]]:
    needs: list[UnresolvedNeed] = []
    diagnostics: list[ExecutionDiagnostic] = []
    suggested_terms = _suggested_need_terms(need)

    if budget_exhausted:
        diagnostics.append(
            ExecutionDiagnostic(
                kind="budget_exhausted",
                message=f"{card.id} exhausted its tool budget before coverage was confirmed.",
                tool="worker_loop",
                suggested_terms=suggested_terms[:6],
            )
        )
    if evidence and not any("class " in item.quote or "def " in item.quote for item in evidence):
        diagnostics.append(
            ExecutionDiagnostic(
                kind="missing_implementation_block",
                message=f"{card.id} found references but no implementation block.",
                tool="navigation",
                suggested_terms=suggested_terms[:6],
            )
        )
    if actions and all(action.result_count == 0 for action in actions if action.tool != "search"):
        diagnostics.append(
            ExecutionDiagnostic(
                kind="empty_navigation",
                message=f"{card.id} follow-up navigation produced no supporting results.",
                tool="navigation",
                suggested_terms=suggested_terms[:6],
            )
        )
    return needs, diagnostics


def _suggested_need_terms(need: str) -> list[str]:
    terms = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", need):
        lowered = token.lower()
        if _is_non_navigation_token(token) or lowered in STOP_WORDS:
            continue
        terms.append(token)
    return sorted(set(terms))


def _is_non_navigation_token(token: str) -> bool:
    return bool(DOC_LABEL_RE.match(token)) or token.lower() in STOP_WORDS


def _dedupe(evidence: list[Evidence]) -> list[Evidence]:
    # A chunk can legitimately be surfaced by both the lexical search() and
    # dense_search() in the same round -- e.g. a paraphrase that also
    # happens to share a term with the query. Keeping only the first-seen
    # copy would silently drop that item's dense_score whenever the lexical
    # copy (added first) is what's kept, undercutting the fused reranker's
    # signal for exactly the items where both channels agree. Merge the
    # dense_score forward onto the kept copy instead of just discarding it.
    result: list[Evidence] = []
    index_by_key: dict[tuple[str, int, int], int] = {}
    for item in evidence:
        key = (item.path, item.line_start, item.line_end)
        if key in index_by_key:
            existing = result[index_by_key[key]]
            if item.dense_score > existing.dense_score:
                result[index_by_key[key]] = existing.model_copy(
                    update={"dense_score": item.dense_score}
                )
            continue
        index_by_key[key] = len(result)
        result.append(item)
    return result


def _rank_evidence(evidence: list[Evidence], need: str) -> list[Evidence]:
    terms = extract_terms(need)

    def score(item: Evidence) -> int:
        return score_evidence(
            quote=item.quote,
            path=item.path,
            reason=item.reason,
            terms=terms,
            dense_score=item.dense_score,
        )

    return sorted(evidence, key=score, reverse=True)
