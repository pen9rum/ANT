from __future__ import annotations

import re

from ant.domain import UnresolvedNeed, WorkerObservation

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


class MockLLMProvider:
    """Deterministic reasoner used to develop orchestration without an API key."""

    def observe(
        self,
        *,
        question: str,
        worker_id: str,
        territory_id: str,
        evidence_count: int,
    ) -> WorkerObservation:
        needs = []
        if evidence_count == 0:
            needs.append(
                UnresolvedNeed(
                    description=f"{worker_id} found no evidence for: {question}",
                    suggested_terms=[
                        term for term in TOKEN_RE.findall(question) if len(term) > 2
                    ][:6],
                    suggested_territories=[],
                )
            )
        return WorkerObservation(
            worker_id=worker_id,
            territory_id=territory_id,
            unresolved_needs=needs,
        )
