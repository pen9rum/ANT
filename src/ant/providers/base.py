from __future__ import annotations

from typing import Protocol, runtime_checkable

from ant.domain import (
    AbsenceProof,
    Evidence,
    NeedResolution,
    UnresolvedNeed,
    WorkerCard,
    WorkerObservation,
)


class WorkerReasoner(Protocol):
    def observe(
        self,
        *,
        question: str,
        worker_id: str,
        territory_id: str,
        evidence: list[Evidence],
    ) -> WorkerObservation: ...

    def check_need_resolution(
        self,
        *,
        need: UnresolvedNeed,
        new_evidence: list[Evidence],
        question: str,
    ) -> NeedResolution:
        """Generalizes the old 3-need_type heuristic closure check to every
        need_type via real judgment: does the evidence gathered *since* this
        need was raised actually satisfy it (resolved), make real headway
        without fully closing it (partial -- refine the need instead of
        re-raising it verbatim), or leave it no better off (unresolved)?
        `new_evidence` is this round's own additions, not the full
        accumulated pool, so the verdict reflects what just happened, not
        what an earlier round already established.
        """
        ...

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

    def decide_local_action(
        self,
        *,
        need: UnresolvedNeed,
        evidence: list[Evidence],
        worker_progress: str,
        worker: WorkerCard,
    ) -> str:
        """Only called for a local-scope need's *second and later* attempt
        (the first attempt has no prior progress yet to judge). Replaces
        letting the routing score alone decide -- purely by coincidence --
        whether the same worker keeps going: makes "continue with the
        current worker" a deliberate action instead of an emergent side
        effect of scoring. Returns one of:
        - "continue": the current worker is still the right one; give it
          another round in its own territory rather than re-routing.
        - "handoff": a different worker should take this need instead.
        - "coalition": pull in a second worker to reason jointly with the
          current one, rather than replacing it.
        - "escalate": normal recruitment is not going to resolve this;
          jump straight to the escalation ladder (see
          LocalCoordinator._escalate_stuck_need) instead of spending
          another normal round first.
        """
        ...


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

    def decide_episode_action(
        self,
        *,
        strategy: str,
        need_terms: list[str],
        occurrences: int,
        successes: int,
        total_evidence_gain: int,
        workers: list[str],
    ) -> str:
        """Judges a recurring collaboration-episode pattern aggregated
        across multiple *tasks* (see ColonyMemoryStore.aggregate_episodes),
        e.g. "temporary_bridge resolved a proxy-validation-shaped need in
        3/3 tasks with real evidence gain". Structural evolve_workers
        signals (route counts, coalition recurrence) only see raw
        co-occurrence; this is the richer signal -- which *specific*
        temporary adaptation actually worked, and how often -- evolve is
        meant to learn from. Returns one of "no_change", "strengthen_route",
        "birth_bridge", "merge".
        """
        ...


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
