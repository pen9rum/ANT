from __future__ import annotations

import re

from ant.domain import Evidence, UnresolvedNeed, WorkerObservation

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
