from __future__ import annotations

import re
from dataclasses import dataclass

from ant.domain import Evidence, UnresolvedNeed, WorkerAction, WorkerCard, WorkerObservation
from ant.tools.local import (
    STOP_WORDS,
    LocalSearchTool,
)
from ant.tools.path_prior import has_low_value_part, has_source_part, is_low_value_path

SYMBOL_RE = re.compile(r"\b(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)")
BASE_RE = re.compile(r"\bclass\s+[A-Za-z_][A-Za-z0-9_]*\(([^)]*)\)")
DOC_LABEL_RE = re.compile(r"^(Args|Example|Examples|Note|Notes|Return|Returns|Raises)$")


@dataclass(frozen=True)
class WorkerRunConfig:
    max_tool_calls: int = 10
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

        candidate_symbols = self.tools.rank_symbols(
            _candidate_symbols(need, evidence),
            self.card.files,
            limit=8,
        )
        for symbol in candidate_symbols:
            if tool_calls >= config.max_tool_calls:
                break
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
        needs = _detect_unresolved_needs(
            card=self.card,
            need=need,
            evidence=evidence,
            actions=actions,
            budget_exhausted=tool_calls >= config.max_tool_calls,
        )
        if not evidence:
            needs.append(
                UnresolvedNeed(
                    description=f"{self.card.id} exhausted local tools without evidence.",
                    suggested_terms=need.split()[:6],
                )
            )
        return WorkerObservation(
            worker_id=self.card.id,
            territory_id=self.card.territory_id,
            evidence=evidence,
            unresolved_needs=needs,
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


def _detect_unresolved_needs(
    *,
    card: WorkerCard,
    need: str,
    evidence: list[Evidence],
    actions: list[WorkerAction],
    budget_exhausted: bool,
) -> list[UnresolvedNeed]:
    needs: list[UnresolvedNeed] = []
    suggested_terms = _suggested_need_terms(need)

    if budget_exhausted:
        needs.append(
            UnresolvedNeed(
                description=f"{card.id} exhausted tool budget before coverage was confirmed.",
                suggested_terms=suggested_terms[:6],
                suggested_territories=[card.territory_id],
            )
        )
    if evidence and not any("class " in item.quote or "def " in item.quote for item in evidence):
        needs.append(
            UnresolvedNeed(
                description=f"{card.id} found references but no implementation block.",
                suggested_terms=suggested_terms[:6],
                suggested_territories=[card.territory_id],
            )
        )
    if actions and all(action.result_count == 0 for action in actions if action.tool != "search"):
        needs.append(
            UnresolvedNeed(
                description=f"{card.id} follow-up navigation produced no supporting results.",
                suggested_terms=suggested_terms[:6],
                suggested_territories=[card.territory_id],
            )
        )
    return needs


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
    result = []
    seen = set()
    for item in evidence:
        key = (item.path, item.line_start, item.line_end)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _rank_evidence(evidence: list[Evidence], need: str) -> list[Evidence]:
    terms = {term.lower() for term in need.replace("_", " ").split() if len(term) > 2}

    def score(item: Evidence) -> int:
        quote = item.quote.lower()
        path = item.path.replace("\\", "/")
        reason = item.reason.lower()
        value = 0
        if "class " in quote or "def " in quote:
            value += 8
        if "definition" in reason or "implementation" in reason:
            value += 6
        if path.endswith(".py"):
            value += 3
        if has_source_part(path):
            value += 2
        if is_low_value_path(path) or has_low_value_part(path):
            value -= 8
        value += sum(1 for term in terms if term in quote)
        return value

    return sorted(evidence, key=score, reverse=True)
