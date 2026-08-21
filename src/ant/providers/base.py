from __future__ import annotations

from typing import Protocol, runtime_checkable

from ant.domain import Evidence, WorkerCard, WorkerObservation


class WorkerReasoner(Protocol):
    def observe(
        self,
        *,
        question: str,
        worker_id: str,
        territory_id: str,
        evidence_count: int,
    ) -> WorkerObservation: ...


class CardGenerator(Protocol):
    def generate_card(
        self,
        *,
        repo_root: str,
        territory_root: str,
        files: list[str],
    ) -> WorkerCard: ...


class AnswerSynthesizer(Protocol):
    def synthesize(self, *, question: str, evidence: list[Evidence]) -> str: ...

    def synthesize_coalition(
        self,
        *,
        question: str,
        worker_ids: list[str],
        evidence: list[Evidence],
    ) -> str: ...


@runtime_checkable
class UsageReporter(Protocol):
    def drain_usage(self) -> object: ...
