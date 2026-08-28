from __future__ import annotations

from typing import Protocol, runtime_checkable

from ant.domain import (
    AbsenceProof,
    Evidence,
    FrontierResult,
    NeedGraph,
    NeedResolution,
    PlanningRound,
    RepairPlan,
    RoundPlan,
    TaskTrajectoryPackage,
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

    def plan_round(
        self,
        *,
        question: str,
        graph: NeedGraph,
        resolution_results: dict[str, NeedResolution],
        evidence: list[Evidence],
        workers: list[WorkerCard],
        memory_hints: dict[str, str],
        frontier: FrontierResult,
        observed_needs: list[UnresolvedNeed],
        incomplete_parents: list[str],
        cross_repo_experience: list[str],
        validation_feedback: str = "",
        repair_guidance: str = "",
    ) -> RoundPlan:
        """The single per-round Orchestrator planning call: replaces
        select_workers/decide_local_action and the hand-coded escalation
        ladder together. Reads the whole current picture -- the Need
        Graph (all three status dimensions, but writes none of them; see
        NeedNode's docstring), this round's freshly-computed
        check_need_resolution results, the full zero-truncation evidence
        pool, every worker's routing_summary (never a relevance-filtered
        subset -- see WorkerCard.routing_summary), repo-local memory
        hints, and the Dependency Graph Analyzer's frontier -- and decides
        three things at once: how to decompose/create/edit need nodes,
        which worker(s) to assign to each ready-frontier node (a
        single-worker entry is a plain follow-up/handoff, multiple is a
        coalition -- both are the same kind of entry, no separate
        escalation-tactic vocabulary), and, only for a stuck subgraph
        handed to it via `frontier.stuck_subgraphs`, whether its recovery
        plan needs one of the two special, executor-mediated tactics
        ("temporary_bridge" | "global_fallback" -- everything else a
        recovery plan might do, like reassigning or redecomposing, is
        expressed as ordinary graph_updates/assignments, no special flag).

        `observed_needs` is the coordinator's persistent buffer of gaps
        raised by workers' own observe() calls that have not yet been
        acted on -- shown explicitly (not folded into `evidence`) so a
        real gap can't quietly get lost in evidence-context noise; this
        call decides per item whether to create a node from it, merge it
        into an existing node's edit, or leave/discard it, and reports
        which indices it handled via RoundPlan.resolved_observed_need_indices.

        `incomplete_parents` lists parent need_ids whose children just all
        resolved but whose own closure check (check_need_resolution on the
        parent itself) came back "unresolved" -- the decomposition under
        them didn't actually cover their original scope. These need_ids
        are not otherwise assignable (a node with children is never a
        direct execution target), so this is the one explicit channel that
        forces the Orchestrator to add more children (or otherwise address
        them) rather than the graph silently having no path forward.

        `cross_repo_experience` is a handful of repo-agnostic verbal case
        studies retrieved (once per task, by semantic similarity to the
        question -- see GlobalMemoryStore.retrieve_similar) from *other*
        repos' finished tasks -- reference text only, never a rule this
        call is required to follow; whether a past pattern actually
        applies here is this call's own judgment to make.

        `validation_feedback` is "" on the normal call; the coordinator
        passes a non-empty description of a detected depends_on cycle on
        the one retry it makes when this call's own graph_updates would
        produce one (depends_on is entirely LLM-drawn, so a cycle here is
        presumptively a planning mistake -- see
        ant.coordinator.graph_analyzer.find_cycles -- not evidence the
        underlying needs are genuinely circular).

        `repair_guidance` is "" on every ordinary ask() call; only
        LocalCoordinator.retry_from_trajectory (task-conditioned/"fast"
        evolution) sets it, to the advisory portion of a RepairPlan
        rendered as text (see ant.coordinator.repair). It is a suggestion
        from a repair analysis of this exact question's own prior failed
        attempt, not an instruction this call is required to follow --
        same status as cross_repo_experience, just task-scoped instead of
        cross-repo.
        """
        ...

    def summarize_task_experience(
        self,
        *,
        question: str,
        rounds: list[PlanningRound],
        unresolved_needs: list[UnresolvedNeed],
        evidence_count: int,
    ) -> str:
        """Called once, after a task finishes: a repo-agnostic verbal case
        study of how it went -- what kind of need showed up, where it got
        stuck, what recovery actually worked and why -- for
        GlobalMemoryStore.record_experience. Must abstract away anything
        repo-specific (worker ids, exact file/symbol names, this repo's own
        vocabulary): the whole point is a pattern transferable to a
        completely different codebase, not a summary of this one task.
        Returns "" for a task with nothing collaboration-shaped worth
        remembering (e.g. a single ready-node, single-round lookup with no
        stuck/recovery/coalition shape to it) -- not every task is worth
        recording.
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

    def summarize_routing(self, *, card: WorkerCard) -> str:
        """Same generation as CardGenerator.summarize_routing, declared
        here too because evolve_workers' specialize/merge/bridge-birth
        sites build new WorkerCards directly (not via
        CardGenerator.generate_card) and only ever receive an
        EvolutionReasoner, not a separate CardGenerator -- one concrete
        provider implements both protocols; this method is just declared
        on whichever protocol each call site already has in hand.
        """
        ...


class FastEvolutionReasoner(Protocol):
    """Task-conditioned ("fast") evolution: judges a single finished
    task's own trajectory (see ant.coordinator.repair.
    assemble_trajectory_package) and proposes an *ephemeral*,
    task-local repair -- the opposite timescale from EvolutionReasoner
    above, which only ever acts on patterns aggregated across many
    finished tasks and mutates the persistent colony. Nothing this
    protocol's caller does writes to IndexStore/ColonyMemoryStore/
    GlobalMemoryStore; a RepairPlan only ever seeds one more
    LocalCoordinator.ask() call for the *same* question.
    """

    def propose_repair(self, *, package: TaskTrajectoryPackage) -> RepairPlan:
        """Reasons only from `package` -- the prior attempt's own
        trajectory (stuck nodes, what was tried and failed, evidence
        gathered vs. missing, how the graph was decomposed). Never given a
        reference answer or judge score; `package.prior_answer` is context
        only, not a correctness signal. Returns a RepairPlan whose actions
        are one of: reuse_assignment, replace_assignment, merge_needs,
        redecompose, change_dependency, form_local_bridge,
        force_global_search (see RepairAction's docstring and
        LocalCoordinator.retry_from_trajectory for how each is applied).
        An empty RepairPlan (no actions) is a valid "just retry with
        carried-forward state, no extra guidance" verdict, not an error.
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

    def summarize_routing(self, *, card: WorkerCard) -> str:
        """Fixed-format, short routing text for `card` -- territory + core
        capability + typical needs handled. Generated once whenever a
        WorkerCard is created or its responsibilities change (birth,
        specialize, merge, bridge-birth) and persisted on
        WorkerCard.routing_summary. This, not the full card, is what the
        Orchestrator planning call reads for every worker every round: the
        colony is shown to it with no relevance-based prefiltering (see
        WorkerCard.routing_summary's docstring), so this exists to keep
        that affordable by compressing each worker's *representation*, not
        by excluding any worker from consideration.
        """
        ...


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
