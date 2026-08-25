from __future__ import annotations

import re

from ant.domain import Evidence, UnresolvedNeed, WorkerCard, WorkerObservation

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


class MockLLMProvider:
    """Deterministic reasoner used to develop orchestration without an API key."""

    def observe(
        self,
        *,
        question: str,
        worker_id: str,
        territory_id: str,
        evidence: list[Evidence],
    ) -> WorkerObservation:
        needs = []
        if not evidence:
            needs.append(
                UnresolvedNeed(
                    description=f"Need grounded evidence for: {question}",
                    kind="missing_evidence",
                    missing=f"Grounded evidence for {question}",
                    scope="unknown",
                    source_worker_id=worker_id,
                    suggested_terms=[term for term in TOKEN_RE.findall(question) if len(term) > 2][
                        :6
                    ],
                    suggested_territories=[],
                )
            )
        return WorkerObservation(
            worker_id=worker_id,
            territory_id=territory_id,
            unresolved_needs=needs,
        )

    def select_lookups(
        self,
        *,
        need: str,
        evidence: list[Evidence],
        candidates: list[str],
    ) -> list[str]:
        # Deterministic pass-through: this mock exists to develop/test
        # orchestration without an API key, not to exercise the actual
        # candidate-filtering judgment (that's OpenAIProvider's job, tested
        # against a real model). Returning candidates unchanged keeps every
        # existing test's assumptions about which lookups happen intact.
        return candidates

    def select_workers(
        self,
        *,
        query: str,
        need: UnresolvedNeed | None,
        candidates: list[WorkerCard],
        limit: int,
        memory_hints: dict[str, str],
    ) -> list[str]:
        # Identity pass-through, same rationale as select_lookups above:
        # this mock exists so orchestration/routing tests can run without an
        # API key, not to exercise LLM worker-selection judgment. Returning
        # candidates in their given (already lexically/dense ranked) order
        # keeps every existing routing test's assumptions intact.
        return [worker.id for worker in candidates]

    def select_evidence(
        self,
        *,
        question: str,
        evidence: list[Evidence],
        limit: int,
    ) -> tuple[list[str], list[str]]:
        # Identity pass-through (same rationale as select_lookups/
        # select_workers above): keep every index in the given (already
        # score-ranked) order, up to limit, and never request an expanded
        # region, so existing tests built around the old fixed score-based
        # cut keep working unchanged.
        return [str(index) for index in range(len(evidence))][:limit], []

    def plan_worker_actions(
        self,
        *,
        need: str,
        evidence: list[Evidence],
        candidate_symbols: list[str],
        available_tools: list[str],
        hints: list[str],
        max_actions: int,
    ) -> list[tuple[str, str]]:
        # Deterministic stand-in that reproduces the pre-existing fixed tool
        # sequence (inheritance/flow hints first if present, then
        # navigate/references/callers/callees/assignments per symbol) as a
        # plan instead of inline execution -- keeps every test/coordinator
        # scenario that runs without a real OpenAIProvider behaving the same
        # as before this became a reasoner-driven plan. Real "should I keep
        # going" judgment is OpenAIProvider's job, tested against a model.
        plan: list[tuple[str, str]] = []
        available = set(available_tools)
        is_inheritance = any("inheritance" in hint.lower() for hint in hints)
        is_flow = any("flow" in hint.lower() or "call path" in hint.lower() for hint in hints)
        if is_inheritance and "subclasses" in available:
            plan.extend(("subclasses", symbol) for symbol in candidate_symbols)
        if is_flow:
            for symbol in candidate_symbols[:4]:
                plan.extend(
                    (tool, symbol)
                    for tool in ("imports", "callers", "assignments")
                    if tool in available
                )
        for symbol in candidate_symbols:
            plan.extend(
                (tool, symbol)
                for tool in ("navigate", "references", "callers", "callees", "assignments")
                if tool in available
            )
        return plan[:max_actions]

    def should_continue_recruiting(
        self,
        *,
        question: str,
        need: UnresolvedNeed,
        evidence: list[Evidence],
        rounds_completed: int,
    ) -> bool:
        # Always continue: this mock exists so orchestration tests can run
        # without an API key, not to exercise the "are we genuinely stuck"
        # judgment (that's OpenAIProvider's job). Old behavior had no such
        # check at all -- rounds ran until max_rounds or no pending need --
        # so always-continue-until-the-caller's-max_rounds reproduces that
        # exactly for every test that doesn't supply a real reasoner.
        return True
