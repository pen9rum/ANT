from __future__ import annotations

from typing import Protocol, runtime_checkable

from ant.domain import (
    AbsenceProof,
    Evidence,
    EvidenceUpgradeVerdict,
    FrontierResult,
    GraphConsolidationPlan,
    GroundedUpdate,
    NeedAlignmentPlan,
    NeedGraph,
    NeedNode,
    NeedResolution,
    PlanningRound,
    ProposedNode,
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

    def verify_evidence_upgrade(
        self,
        *,
        need: UnresolvedNeed,
        epistemic_state: str,
        new_evidence: list[Evidence],
        question: str,
    ) -> EvidenceUpgradeVerdict:
        """Grounded Fast Repair's Evidence Upgrade Gate: called only during
        a fast-repair `ask()` call (`enforce_alignment=True`), for every
        leaf need whose `check_need_resolution` verdict this round is
        "resolved"/"partial" -- gen0-carried-over or born during this same
        retry alike, no distinction (see LocalCoordinator.
        _apply_evidence_upgrade_gate). `epistemic_state` is
        `RecoveryState.epistemic_states.get(need_id, "open")` -- the
        need's grounded standing immediately before this round (an "open"
        need born this retry with no prior proof either way, a gen0 need
        that survived alignment as-is, or "open" again if it was just
        reframed -- see NeedAlignmentVerdict).

        The one question this answers: does `new_evidence` DIRECTLY
        establish the entity/relation the need actually asks about -- not
        an adjacent subsystem, a similarly-named symbol, or a different
        mechanism that merely shares vocabulary. Confirmed live this is
        the dominant fast-repair failure mode: a correct, honest gen0
        hedge ("insufficient evidence" / "not in this repo") gets
        replaced by a confident wrong claim once the retry finds
        something adjacent-but-irrelevant and `check_need_resolution`
        (which has no concept of "adjacent, therefore reject") accepts
        it.

        A False `approved` (including on a malformed/unparseable
        response -- never let a parse failure look like a confident
        upgrade) makes the coordinator revert this round's resolution
        back to unresolved, so the need keeps whatever epistemic
        commitment it already had rather than accepting an unsupported
        upgrade. A True `approved` must include `supported_claim` and
        `evidence_ids` (string indices into the evidence pool, same
        convention as UnresolvedNeed.evidence_ids) -- these become a
        GroundedUpdate, the only channel through which
        AnswerSynthesizer.synthesize's patch mode may strengthen
        `prior_answer`'s wording for this need.
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
        incomplete_parents: list[str],
        cross_repo_experience: list[str],
        validation_feedback: str = "",
        repair_guidance: str = "",
        stuck_tried_workers: dict[str, list[str]] | None = None,
        candidate_probes: dict[str, dict[str, list[Evidence]]] | None = None,
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

        This call does NOT see worker-observed gaps (`observed_needs`) --
        that responsibility, and the decision of whether a *new*
        `graph_updates` proposal actually becomes a permanent node at all,
        belongs entirely to WorkerReasoner.consolidate_graph now (see its
        own docstring and GraphConsolidationPlan). This call's own
        `graph_updates` may still propose brand-new nodes (a new need_id
        entry) -- those are provisional until consolidate_graph decides on
        them, same as an observed need; an *existing* need_id in
        `graph_updates` is still applied directly and immediately, as
        always (revising wording, editing `depends_on`/`related_to`/
        `children` between already-real nodes is this call's own free
        judgment, not routed through consolidation).

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

        `stuck_tried_workers` maps a need_id inside one of
        `frontier.stuck_subgraphs` to the worker ids RecoveryState already
        recorded as tried-with-no-progress on it (empty/None for a need
        with no stuck history yet, which is every need until it has spent
        _STUCK_THRESHOLD rounds without progress). This is advisory too --
        the coordinator does not require assignments to avoid these ids --
        but repeating one anyway is expected to be a deliberate, informed
        choice (e.g. reusing a worker with a different, more specific need
        this round) rather than the planner simply having no memory of
        what already failed: LocalCoordinator.ask() enforces this
        mechanically after the call returns, overriding an assignment that
        names *only* already-tried workers for a still-stuck need with a
        forced global_fallback rather than executing the repeat.

        `workers` here is already narrowed to a per-need retrieval-ranked
        candidate set (see LocalCoordinator._candidate_workers_for_round)
        -- structural, not advisory: an assignment naming a worker id
        outside this list is dropped when the response is parsed (see
        _parse_round_plan's valid_worker_ids check), so an id must come
        from here to take effect at all.

        `candidate_probes` maps a ready need_id to {worker_id: anchors} --
        each candidate's own cheap, local pre-commit search()/dense_search()
        result (see LocalCoordinator._probe_need_candidates), at most a
        handful of Evidence items each, empty for a candidate that found
        nothing. This is the primary signal for *which* candidate to
        actually commit to: prefer a candidate whose probe turned up
        something concretely relevant over one that merely sounds relevant
        by name/routing_summary -- confirmed live that the latter alone is
        not reliable (a "gates" question pulling assignment toward a
        gates-named worker over a better-ranked one with no actual gates-
        drawing content). Still not a hard rule: a candidate with no probe
        anchors can still be the right call (a need's answer may not be
        lexically/semantically close to it at all), and this call keeps
        free choice within `workers`.
        """
        ...

    def consolidate_graph(
        self,
        *,
        question: str,
        active_nodes: dict[str, NeedNode],
        proposals: list[ProposedNode],
        candidate_hints: dict[str, list[str]],
        enforce_alignment: bool = False,
    ) -> GraphConsolidationPlan:
        """The Graph Organizer: the one place a *new* need node actually
        comes into existence. Runs once per round, after this round's
        assignments have executed (so it can also see what worker
        execution surfaced), on the Potential Needs Buffer -- this round's
        new-id `graph_updates` proposals from plan_round PLUS the
        coordinator's persistent worker-observed-needs buffer, unified
        (see ProposedNode.source). Owns exactly one concern: keep the
        problem representation from duplicating or exploding, never
        worker routing or resolution status.

        `active_nodes` is every existing node that is not yet resolved and
        not abandoned -- keyed by their real, permanent need_id. `proposals`
        is this round's buffer. `candidate_hints` maps each proposal's
        `proposal_id` to a short list of nearby existing node ids (dense-
        embedding nearest neighbors over need text, computed by the
        coordinator before this call -- see
        LocalCoordinator._candidate_hints_for_proposals) -- exactly the
        same "retrieval narrows, judgment decides" split that fixed worker
        routing tonight: embedding similarity is never itself the merge
        decision, a fixed cosine threshold cannot tell "same gap reworded"
        from "a more specific child" from "genuinely related but distinct"
        apart, only a real judgment call can. A proposal's hints existing
        is not a suggestion to merge -- plenty of genuinely-new proposals
        will have nearby-but-distinct existing nodes as hints; this call
        is free to still say "create".

        Return exactly one GraphConsolidationDecision per proposal (a
        proposal with no decision is treated as "create" by the
        coordinator, the same safe default MockLLMProvider always
        returns). See GraphConsolidationDecision's own docstring for what
        each action does structurally.

        `enforce_alignment` is True only inside a Grounded Fast Repair
        retry (LocalCoordinator.ask(enforce_alignment=True), set by
        retry_from_trajectory) -- when True, additionally judge, for each
        proposal, whether resolving it would directly help answer
        `question` (not merely "is this a reasonable code question on its
        own"); if not, decide "drop" regardless of novelty/dedup
        considerations, exactly the same test NeedAlignmentVerdict applies
        to a fast retry's carried-over stuck nodes before round 0 -- this
        is what keeps a node born mid-retry from drifting onto the wrong
        sub-question the same way an original one could. False (the
        default, used by every ordinary gen0/slow-gen1 round) leaves
        `consolidate_graph`'s behavior exactly as it was before this
        parameter existed -- dedup/decomposition judgment only, no
        alignment-to-original-question check.
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

        Called with a `package` that has already been through
        assess_need_alignment/apply_alignment_verdicts -- any stuck node
        this call sees has already been kept as-is or reframed onto the
        original question; repair targeting never has to work around a
        drifted framing itself.
        """
        ...

    def assess_need_alignment(
        self, *, question: str, package: TaskTrajectoryPackage
    ) -> NeedAlignmentPlan:
        """Grounded Fast Repair's Need Alignment Gate -- the first thing
        that runs in retry_from_trajectory, before propose_repair. For
        every `package.stuck_nodes` entry, judges only one thing: **would
        fully answering this need, as currently framed, directly help
        answer `question`** -- not "is this a reasonable code question on
        its own". Confirmed live this is a real, distinct failure mode
        from evidence adequacy: the two most catastrophic fast-repair
        score drops observed this session both had the Orchestrator
        decompose a need onto a plausible-sounding but wrong sub-question
        (TLS/auth inheritance -> role-resolution inheritance; package
        release version -> build-environment env_version) -- evidence was
        genuinely adequate *for the wrong sub-question*, and the actually-
        correct evidence was sitting in the same pool the whole time,
        simply never the target of that node's own resolution check.

        Each `StuckNodeSummary.epistemic_state` is shown as **grounded,
        read-only context** (computed deterministically from real
        AbsenceProof records by assemble_trajectory_package, never by a
        reasoner) -- this call may use it to inform a verdict but must
        never itself decide or report an epistemic_state; NeedAlignment
        Verdict has no such field.

        Returns one verdict per stuck need: "keep" (framing is fine, no
        change), "reframe" (framing has drifted -- `reframed_need`
        replaces it, and the need is treated as a fresh investigation:
        epistemic state and everything accumulated under the old framing
        is reset, never carried forward under the new one -- see
        apply_alignment_verdicts), or "drop" (resolving this, even
        perfectly, would not help answer `question` -- discard it; this
        is NOT the same as "tried and failed", so it must never be
        recorded as abandoned). A need_id with no verdict at all defaults
        to keep. This same test (not this same call) is applied again,
        continuously, to every *new* node a fast-repair retry's own
        rounds propose -- see WorkerReasoner.consolidate_graph's
        `enforce_alignment` parameter.
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
        prior_answer: str = "",
        grounded_updates: list[GroundedUpdate] | None = None,
    ) -> str:
        """`prior_answer` == "" (every gen0/slow-gen1 call, and every
        fast-repair call before this parameter existed) means full,
        independent synthesis from `evidence` -- byte-identical behavior
        to before this parameter existed. `prior_answer` != "" (only ever
        set by a fast-repair retry, carrying prior_state.answer) switches
        to Grounded Fast Repair's patch mode: revise `prior_answer`
        freely in wording, but only convert an uncertain/absent claim to
        a confident positive one where a `grounded_updates` entry names
        that specific claim (see GroundedUpdate, produced only by an
        approved WorkerReasoner.verify_evidence_upgrade verdict) --
        everywhere `grounded_updates` says nothing, the original epistemic
        commitment (still uncertain, still absent) must survive even
        while its sentence is rephrased.
        """
        ...

    def synthesize_coalition(
        self,
        *,
        question: str,
        worker_ids: list[str],
        evidence: list[Evidence],
        absence_proofs: list[AbsenceProof] | None = None,
        prior_answer: str = "",
        grounded_updates: list[GroundedUpdate] | None = None,
    ) -> str:
        """Same `prior_answer`/`grounded_updates` patch-mode contract as
        `synthesize` -- see its docstring."""
        ...


@runtime_checkable
class UsageReporter(Protocol):
    def drain_usage(self) -> object: ...
