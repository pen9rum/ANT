from __future__ import annotations

from typing import Protocol, runtime_checkable

from ant.domain import AbsenceProof, Evidence, UnresolvedNeed, WorkerCard, WorkerObservation


class WorkerReasoner(Protocol):
    def observe(
        self,
        *,
        question: str,
        worker_id: str,
        territory_id: str,
        evidence: list[Evidence],
    ) -> WorkerObservation: ...

    def select_lookups(
        self,
        *,
        need: str,
        evidence: list[Evidence],
        candidates: list[str],
    ) -> list[str]: ...

    def select_workers(
        self,
        *,
        query: str,
        need: UnresolvedNeed | None,
        candidates: list[WorkerCard],
        limit: int,
        memory_hints: dict[str, str],
    ) -> list[str]: ...

    def select_evidence(
        self,
        *,
        question: str,
        evidence: list[Evidence],
        limit: int,
    ) -> tuple[list[str], list[str]]:
        """Returns (keep_indices, expand_indices): which pooled evidence
        indices to keep for synthesis, and which of those (a subset) have a
        quote too narrow to answer from and should have their source region
        reopened first -- see LocalCoordinator._select_evidence and Shared
        Evidence State's "evidence compression is reversible" principle.
        Both are index strings into the `evidence` argument.
        """
        ...

    def plan_worker_actions(
        self,
        *,
        need: str,
        evidence: list[Evidence],
        candidate_symbols: list[str],
        available_tools: list[str],
        hints: list[str],
        max_actions: int,
    ) -> list[tuple[str, str]]: ...

    def should_continue_recruiting(
        self,
        *,
        question: str,
        need: UnresolvedNeed,
        evidence: list[Evidence],
        rounds_completed: int,
    ) -> bool: ...


class EvolutionReasoner(Protocol):
    """Judges colony-reorganization decisions (evolve_workers) that were
    previously decided by numeric thresholds alone (route counts, file
    overlap ratios). Structural signals still decide *which* worker/pair is
    even a candidate -- this is the judgment call on top of that candidate,
    same pool-then-LLM-decides shape as WorkerReasoner's select_* methods.
    """

    def should_specialize(
        self,
        *,
        worker_id: str,
        worker_summary: str,
        candidate_groups: dict[str, list[str]],
        route_summaries: list[str],
    ) -> bool: ...

    def should_merge(
        self,
        *,
        worker_a_id: str,
        worker_a_summary: str,
        worker_b_id: str,
        worker_b_summary: str,
    ) -> bool: ...


class CardGenerator(Protocol):
    def generate_card(
        self,
        *,
        repo_root: str,
        territory_root: str,
        files: list[str],
    ) -> WorkerCard: ...


class AnswerSynthesizer(Protocol):
    def synthesize(
        self,
        *,
        question: str,
        evidence: list[Evidence],
        absence_proofs: list[AbsenceProof] | None = None,
    ) -> str: ...

    def synthesize_coalition(
        self,
        *,
        question: str,
        worker_ids: list[str],
        evidence: list[Evidence],
        absence_proofs: list[AbsenceProof] | None = None,
    ) -> str: ...


@runtime_checkable
class UsageReporter(Protocol):
    def drain_usage(self) -> object: ...
