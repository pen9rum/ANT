from __future__ import annotations

from typing import Protocol

from ant.domain import WorkerObservation


class WorkerReasoner(Protocol):
    def observe(
        self,
        *,
        question: str,
        worker_id: str,
        territory_id: str,
        evidence_count: int,
    ) -> WorkerObservation: ...
