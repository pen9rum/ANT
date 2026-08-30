from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Territory(BaseModel):
    id: str
    root: str
    files: list[str] = Field(default_factory=list)
    summary: str = ""


class WorkerCard(BaseModel):
    id: str
    territory_id: str
    name: str
    root: str
    responsibilities: list[str] = Field(default_factory=list)
    searchable_terms: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    symbols: list[CodeSymbol] = Field(default_factory=list)
    # Fixed-format ("territory + core capability + typical needs handled"),
    # short, persistent text generated once at birth/specialize/merge (see
    # CardGenerator.summarize_routing) -- what the Orchestrator planning
    # call reads for every worker every round instead of the full card
    # above. Every worker is shown to the Orchestrator every round with no
    # relevance-based prefiltering; this field exists purely to keep that
    # affordable at colony sizes of 30+ evolved workers by compressing each
    # worker's *representation*, not by excluding any worker from
    # consideration.
    routing_summary: str = ""


class CodeSymbol(BaseModel):
    name: str
    kind: str
    path: str
    line: int
    qualname: str = ""
    bases: list[str] = Field(default_factory=list)


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    estimated_cost_usd: float = 0.0


class Evidence(BaseModel):
    path: str
    line_start: int
    line_end: int
    quote: str
    reason: str
    claim: str = ""
    worker_id: str = ""
    symbols: list[str] = Field(default_factory=list)
    dense_score: float = 0.0


class WorkerAction(BaseModel):
    tool: str
    query: str
    result_count: int = 0
    rationale: str = ""


class ExecutionDiagnostic(BaseModel):
    kind: str
    message: str
    tool: str = ""
    suggested_terms: list[str] = Field(default_factory=list)


class WorkerObservation(BaseModel):
    worker_id: str
    territory_id: str
    evidence: list[Evidence] = Field(default_factory=list)
    unresolved_needs: list[UnresolvedNeed] = Field(default_factory=list)
    diagnostics: list[ExecutionDiagnostic] = Field(default_factory=list)
    actions: list[WorkerAction] = Field(default_factory=list)
    stop_reason: str = ""


class UnresolvedNeed(BaseModel):
    description: str
    kind: str = "missing_detail"
    need_type: str = "unknown"
    known: list[str] = Field(default_factory=list)
    missing: str = ""
    scope: str = "unknown"
    source_worker_id: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    suggested_terms: list[str] = Field(default_factory=list)
    suggested_territories: list[str] = Field(default_factory=list)
    relevant_symbols: list[str] = Field(default_factory=list)


class NeedResolution(BaseModel):
    """Verdict from reasoner.check_need_resolution(): whether evidence
    gathered since a need was raised actually satisfies it, generalizing
    the old heuristic closure check (which only understood 3 of ~6
    need_types, silently never closing the rest) to every need_type via
    real judgment instead of pattern-matching a definition/inheritance
    quote."""

    status: str = "unresolved"  # "resolved" | "partial" | "unresolved"
    # Only meaningful when status == "partial": a more specific need that
    # replaces the original in accumulated_needs, so a need that keeps
    # getting re-raised sharpens round over round instead of being asked
    # again verbatim.
    refined_need: UnresolvedNeed | None = None


class NeedNode(BaseModel):
    """One node in a task's Need Graph.

    `need_id` is assigned once at creation and is permanent: a replan may
    add children, add/remove edges, or close a node, but must never
    delete-and-recreate "the same underlying gap" under a new id -- that
    permanence is what lets recovery-attempt bookkeeping (see
    NeedGraph.recovery_streaks) survive across however many times a stuck
    subgraph gets reorganized.

    Three independently-sourced status dimensions, deliberately never
    collapsed into one "status" field -- a node can be partial + ready +
    stuck all at once, and each dimension has exactly one writer so they
    can never contradict each other:
    - `resolution`: written only by WorkerReasoner.check_need_resolution()
      (per-node, after its worker(s) execute; for a node with children,
      written only once as a one-time closure-verification pass after
      every child first reaches "resolved" -- see `children` below).
    - `execution`: written only by the Dependency Graph Analyzer's
      Kahn's-algorithm pass over `depends_on` edges. Meaningless (never
      updated) once `children` is non-empty -- a parent with children is
      not itself an execution target, see below.
    - `progress`: written only by the coordinator's own no-progress
      counter (`rounds_without_progress`), not by any LLM judgment.

    The Orchestrator (the one LLM planning call per round) writes none of
    these three dimensions. It only ever creates/decomposes nodes
    (`children`) and edits `depends_on`/`related_to` -- the semantic,
    judgment-requiring part of the graph, not the status bookkeeping.
    """

    need_id: str
    need: str
    depends_on: list[str] = Field(default_factory=list)
    related_to: list[str] = Field(default_factory=list)
    # Non-empty once the Orchestrator has decomposed this node into finer
    # sub-needs. A node with children is a pure hierarchical container: it
    # is excluded from the Analyzer's ready/blocked frontier and is never
    # itself assigned a worker or given its own per-round
    # check_need_resolution() call -- only leaf nodes (children == [])
    # participate in execution. Children resolving does not by itself mark
    # the parent resolved (a decomposition can be incomplete relative to
    # the parent's original scope); see `resolution`'s closure-check note
    # above.
    children: list[str] = Field(default_factory=list)

    resolution: str = "unresolved"  # "unresolved" | "partial" | "resolved"
    execution: str = "ready"  # "ready" | "blocked" -- meaningful only for leaf nodes
    progress: str = "not_stuck"  # "not_stuck" | "stuck"

    # Consecutive rounds this node has been touched with neither a
    # `resolution` advance (unresolved->partial/resolved) nor a dependency
    # release that produced a new ready node -- genuinely new evidence with
    # no accepted effect on resolution does NOT reset this. Once this
    # reaches the coordinator's stuck threshold, `progress` flips to
    # "stuck".
    rounds_without_progress: int = 0

    # The semantic content of the gap itself (what's missing, need_type,
    # scope, suggested terms/territories, ...) -- reuses UnresolvedNeed
    # rather than duplicating its fields, since check_need_resolution() and
    # the rest of the existing need-consuming machinery already operate on
    # UnresolvedNeed.
    detail: UnresolvedNeed


class NeedGraph(BaseModel):
    """The full Need Graph for one in-progress task, keyed by `need_id`.

    Deliberately pure problem structure only -- no execution/recovery
    bookkeeping (recovery-attempt streaks, which special tactics have
    already been tried, which persistent workers were already tried per
    node). That belongs to LocalCoordinator's own runtime RecoveryState
    instead, kept separate so the graph stays a clean record of the task's
    needs and their relationships, not a mix of problem structure and
    coordinator execution history.
    """

    nodes: dict[str, NeedNode] = Field(default_factory=dict)


class FrontierResult(BaseModel):
    """Output of the pure, deterministic Dependency Graph Analyzer
    (ant.coordinator.graph_analyzer.compute_frontier), computed fresh each
    round from a NeedGraph's `depends_on` edges via Kahn's-algorithm-style
    frontier computation and Tarjan's SCC for cycle detection -- never
    guessed by an LLM. Lives in ant.domain (not the analyzer module
    itself) purely so WorkerReasoner.plan_round() can reference its type
    without ant.providers depending on ant.coordinator.
    """

    # Leaf node ids (children == []) whose every depends_on target is
    # already resolved, and which are not themselves marked
    # progress=="stuck" -- this round's assignable frontier.
    ready: list[str] = Field(default_factory=list)
    # Leaf node ids that are not yet resolved, not stuck, and not (yet)
    # ready.
    blocked: list[str] = Field(default_factory=list)
    # Groups of node ids that together make up one genuinely stuck
    # subgraph: either a real depends_on cycle (should never occur in
    # validated state -- see find_cycles) or a chain of blocked nodes
    # whose root cause traces back to one node already marked
    # progress=="stuck". Always empty when `ready` is non-empty.
    stuck_subgraphs: list[list[str]] = Field(default_factory=list)


class RoundPlan(BaseModel):
    """Output of WorkerReasoner.plan_round(): the Orchestrator's single
    per-round decision. It never writes `resolution`/`execution`/`progress`
    -- each of those three dimensions has exactly one other writer (see
    NeedNode's docstring). It also never has the final word on whether a
    *new* need node actually comes into existence -- see `graph_updates`
    below and GraphConsolidationPlan/WorkerReasoner.consolidate_graph.
    """

    # Edits to nodes that already exist in the graph, AND proposals for
    # brand-new ones, both keyed by need_id -- the two are told apart
    # purely by whether that need_id is already in the graph. An existing
    # need_id here is applied directly (need/depends_on/related_to/children
    # edits -- the Orchestrator's own free judgment call, same as always).
    # A *new* need_id here is only a PROPOSAL: it is collected into this
    # round's Potential Needs Buffer alongside worker-observed needs, and
    # only becomes a real, permanent graph node if
    # WorkerReasoner.consolidate_graph decides "create"/"attach"/"relate"
    # for it -- "merge"/"subsume"/"drop" mean it never gets a node_id at
    # all. This is the fix for graph growth outrunning resolution: the
    # Orchestrator proposing a need no longer means it exists immediately;
    # see GraphConsolidationPlan for the actual per-proposal decision.
    graph_updates: dict[str, NeedNode] = Field(default_factory=dict)
    # need_id -> worker ids assigned this round. Only ready-frontier
    # need_ids (or, for a stuck subgraph handed to this call, one of its
    # members) should appear here -- a need_id that is only a proposal
    # this round (see graph_updates above) is never assignable yet, it
    # does not exist in the graph until consolidated.
    assignments: dict[str, list[str]] = Field(default_factory=dict)
    # Stuck-subgraph root need_id -> "temporary_bridge" | "global_fallback"
    # -- only present when the Orchestrator's recovery plan for that
    # subgraph (see FrontierResult.stuck_subgraphs) calls for one of the
    # two special, executor-mediated tactics. Every other kind of recovery
    # (reassign, redecompose, form a coalition) is expressed as ordinary
    # graph_updates/assignments above; no special flag needed for those.
    special_tactics: dict[str, str] = Field(default_factory=dict)


class ProposedNode(BaseModel):
    """One new-node candidate for this round's Potential Needs Buffer --
    from either the Orchestrator's own graph_updates (a new-id entry, see
    RoundPlan's docstring) or a worker's observed_needs. Never becomes a
    permanent NeedGraph node until WorkerReasoner.consolidate_graph
    decides what to do with it (see GraphConsolidationDecision).
    """

    proposal_id: str  # this round's own provisional label, never persisted as a real need_id
    need: str
    detail: UnresolvedNeed
    # Existing node ids OR other proposal_ids from this same round's
    # buffer -- resolved through the commit-order proposal_id -> real
    # node_id map as decisions are applied (see
    # LocalCoordinator._consolidate_and_commit).
    proposed_depends_on: list[str] = Field(default_factory=list)
    proposed_children: list[str] = Field(default_factory=list)
    proposed_related_to: list[str] = Field(default_factory=list)
    # Non-empty => the proposer suggests this become a child of that
    # existing node specifically (a hint only -- consolidate_graph can
    # still choose "attach" to a different target, or a different action
    # entirely).
    proposed_parent: str = ""
    source: str = "orchestrator"  # "orchestrator" | "worker_observed"


class GraphConsolidationDecision(BaseModel):
    """WorkerReasoner.consolidate_graph's verdict on one ProposedNode.

    - "create": genuinely new, independent node. Permanent id = the
      proposal's own proposal_id.
    - "attach": genuinely new, but wired as a `children` entry of
      target_node_id instead of standalone. Also gets a permanent id
      (the proposal's own proposal_id).
    - "relate": genuinely new and independent, but wired with a
      `related_to` edge to target_node_id (not a dependency/duplicate).
      Also gets a permanent id.
    - "merge": this proposal *is* an existing unresolved node, just
      reworded -- no new id; target_node_id's detail is enriched (union
      of relevant_symbols/suggested_terms) with the proposal's own.
    - "subsume": this proposal is a more specific restatement of an
      existing node's own scope -- no new id; target_node_id's own
      need/detail text is replaced by the proposal's sharper wording.
    - "drop": already covered by existing evidence/nodes -- discarded
      entirely, no id, no edit.
    """

    proposal_id: str
    action: str
    target_node_id: str = ""  # existing node id -- required for merge/subsume/attach/relate
    rationale: str = ""


class GraphConsolidationPlan(BaseModel):
    decisions: list[GraphConsolidationDecision] = Field(default_factory=list)


class NodeExecutionTrace(BaseModel):
    """One need node's execution within a PlanningRound: who worked it,
    how, and what it produced. `need_reduction` is deliberately *direct*
    only (1 if this specific node's own resolution became "resolved" this
    execution, 0 otherwise) -- a closure check resolving a parent whose
    children just finished is attributed at the PlanningRound level
    instead (see `derived_resolved_nodes`), not pinned onto whichever
    child happened to execute last within a multi-node round, which would
    only reflect execution order, not which child actually "caused" it.
    """

    need_id: str
    need: str = ""  # the node's own need text at the time of this execution
    worker_ids: list[str] = Field(default_factory=list)
    coalition_formed: bool = False
    resolution: str = "unresolved"  # this node's resolution status after this execution
    special_tactic: str = ""  # "" | "temporary_bridge" | "global_fallback"
    evidence_gain: int = 0
    need_reduction: int = 0  # 0 or 1, direct only -- see docstring above
    observations: list[WorkerObservation] = Field(default_factory=list)
    # This need's own retrieval-ranked candidate set for the round it was
    # planned in (see ant.coordinator.worker_retrieval.rank_workers and
    # LocalCoordinator._candidate_workers_for_round) -- independent of
    # which worker(s) actually ended up in `worker_ids` above, so a trace
    # can be audited after the fact: was the right worker even a
    # candidate this round, and did the Orchestrator pick correctly from
    # it. Empty for a stuck-subgraph/special-tactic execution, which is
    # not narrowed to any candidate set (see that method's docstring).
    candidate_worker_ids: list[str] = Field(default_factory=list)
    candidate_worker_ranks: dict[str, int] = Field(default_factory=dict)
    # How many probe anchors (see LocalCoordinator._probe_need_candidates)
    # each candidate turned up before this need's worker(s) were actually
    # committed to -- just the counts, not the anchors themselves: the
    # committed worker's own full AutonomousWorker.run() re-surfaces the
    # real evidence through the ordinary pipeline, so persisting the
    # probe's disposable anchors in full would be redundant weight for no
    # real audit gain beyond "how many did each candidate find".
    candidate_probe_anchor_counts: dict[str, int] = Field(default_factory=dict)


class GraphDelta(BaseModel):
    """Lightweight record of how the Need Graph's *structure* changed
    during one round -- separate from NodeExecutionTrace (which records
    execution outcomes, not structural edits) so a trajectory consumer
    (e.g. task-conditioned/fast-mode repair) can see how the graph was
    decomposed/rewired each round without re-deriving it from a full
    before/after NeedNode diff. Captures the whole round's structural
    change, including a gap-node the closure check itself creates on a
    "partial" parent verdict, not just plan.graph_updates.
    """

    # need_ids that did not exist in the graph before this round.
    created_nodes: list[str] = Field(default_factory=list)
    # need_id -> its new depends_on list, only for nodes whose depends_on
    # actually changed this round (existing nodes not mentioned are
    # unchanged; a newly-created node's initial edges are on
    # PlanningRound.node_executions/graph_updates already via
    # created_nodes, not duplicated here).
    dependency_changes: dict[str, list[str]] = Field(default_factory=dict)
    # need_id -> its new children list, only for nodes whose children
    # list actually changed this round (Orchestrator decomposition, or a
    # closure check appending a gap-node to an incomplete parent).
    created_children: dict[str, list[str]] = Field(default_factory=dict)
    # need_id -> worker ids assigned this round -- same content as
    # RoundPlan.assignments, kept here too so a trajectory consumer reading
    # graph_delta alone (without the full plan object) still has it.
    assignment_changes: dict[str, list[str]] = Field(default_factory=dict)
    # Parent need_ids closure-resolved this round -- same list as
    # PlanningRound.derived_resolved_nodes, duplicated here for the same
    # locality reason as assignment_changes.
    closure_results: list[str] = Field(default_factory=list)


class PlanningRound(BaseModel):
    """One round of the graph-based pipeline: exactly one
    WorkerReasoner.plan_round() call, fanning out to however many nodes
    its assignments/special_tactics touched. Replaces RecruitmentRound
    (which assumed one node per round) for this pipeline -- round_index
    stays unambiguous ("the Nth plan_round() call") even though a round
    can now execute several nodes at once.
    """

    round_index: int
    node_executions: list[NodeExecutionTrace] = Field(default_factory=list)
    # Parent need_ids whose closure check resolved them THIS round because
    # their children all finished -- kept separate from any child's own
    # NodeExecutionTrace.need_reduction (see that field's docstring).
    derived_resolved_nodes: list[str] = Field(default_factory=list)
    # Structural graph changes this round -- see GraphDelta's docstring.
    graph_delta: GraphDelta = Field(default_factory=GraphDelta)


class StuckEpisodeSnapshot(BaseModel):
    """Read-only end-of-task copy of one coordinator.local.StuckEpisode --
    that dataclass is coordinator-internal runtime bookkeeping (mutable
    sets, never serialized), this is what a trajectory consumer (fast-mode
    repair, offline analysis) actually gets to see."""

    episode_id: str
    members: list[str] = Field(default_factory=list)
    recovery_streak: int = 0
    used_special_tactics: list[str] = Field(default_factory=list)


class RecoverySnapshot(BaseModel):
    """Read-only end-of-task copy of coordinator.local.RecoveryState's
    recovery-relevant fields (not tried_workers_by_node's use during
    execution, which needs a mutable dict[str, set[str]] -- this is the
    post-hoc, list-ified view for anything reading the finished task's
    trajectory, not for the coordinator's own hot loop). Everything here
    defaults empty/zero for a task that never got stuck."""

    stuck_episodes: list[StuckEpisodeSnapshot] = Field(default_factory=list)
    abandoned_node_ids: list[str] = Field(default_factory=list)
    tried_workers_by_node: dict[str, list[str]] = Field(default_factory=dict)


class EvidenceState(BaseModel):
    question: str
    answer: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    unresolved_needs: list[UnresolvedNeed] = Field(default_factory=list)
    rounds: list[PlanningRound] = Field(default_factory=list)
    absence_proofs: list[AbsenceProof] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    # Full trajectory, always populated (see PlanningRound.graph_delta for
    # the per-round layer): every need_id's final resolution/execution/
    # progress/rounds_without_progress/depends_on/children, and the
    # recovery history (stuck episodes, abandonment, tried workers) that
    # produced it. Neither is read by slow/colony evolution (ColonyMemoryStore
    # only ever reads a compressed per-execution extraction of `rounds`, via
    # record_task_memory -- see its own docstring) -- this exists for
    # trajectory-conditioned analysis: task-conditioned/fast-mode repair,
    # debugging, and offline research use, none of which write back to the
    # persistent colony.
    final_need_graph: dict[str, NeedNode] = Field(default_factory=dict)
    final_recovery_state: RecoverySnapshot = Field(default_factory=RecoverySnapshot)

    def has_evidence(self) -> bool:
        return bool(self.evidence)


class AbsenceProof(BaseModel):
    query: str
    relevant_symbols: list[str] = Field(default_factory=list)
    searched_worker_ids: list[str] = Field(default_factory=list)
    searched_territories: list[str] = Field(default_factory=list)
    searched_paths: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    exhaustive: bool = False
    conclusion: str = "inconclusive"


def as_posix(path: Path) -> str:
    return path.as_posix()


class StuckNodeSummary(BaseModel):
    """One unresolved/partial need from a finished task's trajectory,
    condensed for a FastEvolutionReasoner -- everything it needs to judge
    "what should a retry of this specific task do differently", derived
    entirely from EvidenceState.final_need_graph/final_recovery_state/
    rounds (see assemble_trajectory_package). Deliberately excludes the
    reference answer/judge score -- fast-mode repair reasons only from the
    task's own trajectory, never supervision.
    """

    need_id: str
    need: str
    resolution: str
    depends_on: list[str] = Field(default_factory=list)
    children: list[str] = Field(default_factory=list)
    missing: str = ""
    suggested_terms: list[str] = Field(default_factory=list)
    suggested_territories: list[str] = Field(default_factory=list)
    tried_worker_ids: list[str] = Field(default_factory=list)
    tried_special_tactics: list[str] = Field(default_factory=list)
    no_progress_execution_count: int = 0
    evidence_claims: list[str] = Field(default_factory=list)
    is_abandoned: bool = False
    stuck_episode_id: str = ""


class TaskTrajectoryPackage(BaseModel):
    """The input to FastEvolutionReasoner.propose_repair(): a compact,
    purpose-built summary of one finished task's own trajectory, not the
    raw EvidenceState -- see assemble_trajectory_package. `prior_answer`
    is included as context (a retry may want to know what was already
    concluded) but is explicitly not a correctness signal; nothing here is
    the reference answer or a judge score."""

    question: str
    prior_answer: str = ""
    stuck_nodes: list[StuckNodeSummary] = Field(default_factory=list)
    graph_decomposition_log: list[GraphDelta] = Field(default_factory=list)


class RepairAction(BaseModel):
    """One instruction in a RepairPlan. `kind` is one of:
    reuse_assignment | replace_assignment | merge_needs | redecompose |
    change_dependency | form_local_bridge | force_global_search --
    see LocalCoordinator.retry_from_trajectory / ant.coordinator.repair for
    how each is applied. Every kind now has a real, observable effect on
    the retry, not just a hope the Orchestrator follows a suggestion:
    change_dependency/redecompose/merge_needs mechanically edit the
    retry's starting graph before round 0 begins; reuse_assignment/
    replace_assignment/form_local_bridge/force_global_search are forced to
    actually execute once at the retry's round 0, after which the
    Orchestrator regains ordinary freedom to route as it sees fit.
    """

    kind: str
    need_id: str
    worker_ids: list[str] = Field(default_factory=list)
    merge_with: list[str] = Field(default_factory=list)
    new_depends_on: list[str] | None = None
    rationale: str = ""


class RepairPlan(BaseModel):
    actions: list[RepairAction] = Field(default_factory=list)
