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
    per-round decision. This covers everything the Orchestrator is
    responsible for in the new graph-based pipeline -- decomposing or
    creating need nodes, editing `depends_on`/`related_to` edges, and
    assigning workers to this round's frontier (a single worker id means a
    plain follow-up/handoff, more than one means a coalition; both are the
    same kind of entry here, no separate escalation-tactic vocabulary
    needed for them). It never writes `resolution`/`execution`/`progress`
    -- each of those three dimensions has exactly one other writer (see
    NeedNode's docstring).
    """

    # New or updated nodes this call wants to add/change, keyed by
    # need_id. A node already in the existing graph that isn't mentioned
    # here is left untouched. need_id is permanent (see NeedNode): this
    # can introduce a brand-new id, or update an existing node's
    # `need`/`depends_on`/`related_to`/`children` -- it must never reuse
    # an existing need_id to mean something semantically different.
    graph_updates: dict[str, NeedNode] = Field(default_factory=dict)
    # need_id -> worker ids assigned this round. Only ready-frontier
    # need_ids (or, for a stuck subgraph handed to this call, one of its
    # members) should appear here.
    assignments: dict[str, list[str]] = Field(default_factory=dict)
    # Stuck-subgraph root need_id -> "temporary_bridge" | "global_fallback"
    # -- only present when the Orchestrator's recovery plan for that
    # subgraph (see FrontierResult.stuck_subgraphs) calls for one of the
    # two special, executor-mediated tactics. Every other kind of recovery
    # (reassign, redecompose, form a coalition) is expressed as ordinary
    # graph_updates/assignments above; no special flag needed for those.
    special_tactics: dict[str, str] = Field(default_factory=dict)
    # Indices into that call's own `observed_needs` argument that this plan
    # has acted on in some way (created a node from, merged into an
    # existing node's edit, or deliberately decided not to track) -- the
    # coordinator drops exactly these from its persistent observed-needs
    # buffer afterward, everything else stays pending for a future round.
    # Mirrors reasoner.select_evidence's index-based consumption pattern
    # rather than requiring content-matching heuristics to guess which
    # buffered need a graph_updates entry came from.
    resolved_observed_need_indices: list[str] = Field(default_factory=list)


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


class EvidenceState(BaseModel):
    question: str
    answer: str = ""
    evidence: list[Evidence] = Field(default_factory=list)
    unresolved_needs: list[UnresolvedNeed] = Field(default_factory=list)
    rounds: list[PlanningRound] = Field(default_factory=list)
    absence_proofs: list[AbsenceProof] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)

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
