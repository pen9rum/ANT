from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np

from ant.coordinator.graph_analyzer import compute_frontier, find_cycles
from ant.coordinator.repair import (
    apply_alignment_verdicts,
    assemble_trajectory_package,
    build_retry_starting_state,
    render_repair_guidance,
    resolve_repair_plan,
)
from ant.coordinator.worker_retrieval import WorkerIndex, build_worker_index, rank_workers
from ant.domain import (
    AbsenceProof,
    AnswerObligation,
    Evidence,
    EvidenceState,
    FrontierResult,
    GraphConsolidationDecision,
    GraphConsolidationPlan,
    GraphDelta,
    GroundedUpdate,
    NeedGraph,
    NeedNode,
    NeedResolution,
    NodeExecutionTrace,
    ObligationCoverage,
    PlanningRound,
    ProposedNode,
    RecoverySnapshot,
    RoundPlan,
    StuckEpisodeSnapshot,
    TokenUsage,
    UnresolvedNeed,
    WorkerAction,
    WorkerCard,
    WorkerObservation,
)
from ant.indexing.cards import template_routing_summary
from ant.memory import IndexStore, MemoryRoute
from ant.providers import (
    AnswerSynthesizer,
    FastEvolutionReasoner,
    MockLLMProvider,
    UsageReporter,
    WorkerReasoner,
)
from ant.retrieval import STOP_WORDS, TOKEN_RE, extract_terms, is_stem_match, score_evidence
from ant.retrieval.dense import (
    WORKER_CARDS_KEY,
    EmbeddingEntry,
    EmbeddingIndex,
    get_shared_embedder,
)
from ant.scoring_config import DEFAULT_SCORING_CONFIG
from ant.tools import LocalSearchTool
from ant.tools.path_prior import has_low_value_part, has_source_part
from ant.workers import AutonomousWorker, WorkerRunConfig

BASE_CLASS_RE = re.compile(r"\bclass\s+[A-Za-z_][A-Za-z0-9_]*\(([^)]*)\)")

# need_type values that a heuristic evidence match is allowed to close outright.
# Absence-shaped needs (negative_presence, unknown) are deliberately excluded:
# not finding a symbol is never proof it does not exist, so closing those on a
# heuristic match would let synthesis assert a false absence.
_CLOSABLE_BY_EVIDENCE = {"subclass_lookup", "implementation_location", "source_test_coalition"}

# Consecutive rounds an active need may go with no new evidence and no
# resolution progress before escalation (a qualitatively different tactic,
# not just "pick another worker again") kicks in. 2 rather than 1: a single
# quiet round is common and not yet evidence of being stuck, only a repeat
# of it is.
_STUCK_THRESHOLD = 2

# How many consecutive recovery attempts (any kind: reassign, redecompose,
# coalition, temporary_bridge, global_fallback) a stuck subgraph -- or an
# incomplete parent whose closure check keeps coming back unresolved -- may
# make with no progress before the coordinator gives up on it
# deterministically, instead of an LLM's own "I give up" judgment. An
# engineering bounded-computation safeguard, not a routing heuristic.
_MAX_CONSECUTIVE_FAILED_RECOVERIES = 3

# unresolved < partial < resolved: the one ordering "did resolution
# genuinely advance this round" is judged against, for both a single
# node's own stuck counter and a stuck episode's recovery streak
# (_resolution_advanced below) -- partial -> partial is NOT an advance,
# even though its status is "not unresolved".
_RESOLUTION_RANK = {"unresolved": 0, "partial": 1, "resolved": 2}

# Two-stage routing candidate-set size for a ready-frontier need, keyed by
# that need's own rounds_without_progress (0 -> fresh, >=1 -> one quiet
# round already, still on the ready frontier -- _STUCK_THRESHOLD == 2 is
# where it actually leaves the ready frontier and enters
# frontier.stuck_subgraphs instead, where temporary_bridge/global_fallback
# already operate at full, unscoped width; see
# _candidate_workers_for_round's own docstring). The one deliberate new
# hand-tuned pair in this change -- a structural candidate-set cutoff
# needs a number, same "flag it explicitly" treatment tonight's RRF k=60
# constant got.
_FRESH_NEED_CANDIDATE_LIMIT = 5
_ESCALATED_NEED_CANDIDATE_LIMIT = 10

# How many probe anchors (see _probe_need_candidates) a single candidate
# worker's cheap pre-commit search/dense_search call may surface -- a
# short look, not the full evidence-gathering AutonomousWorker.run() would
# do once actually committed to.
_PROBE_ANCHOR_LIMIT = 3

# How many existing active nodes get surfaced as a proposal's
# candidate_hints for Need Graph Consolidation (see
# _candidate_hints_for_proposals/WorkerReasoner.consolidate_graph) --
# embedding-nearest-neighbor retrieval, never itself the merge decision.
# Same order of magnitude as _FRESH_NEED_CANDIDATE_LIMIT; at today's
# observed graph sizes (<=8 active nodes) this is a no-op in practice, it
# only starts narrowing once a task's graph actually grows past it.
_CONSOLIDATION_CANDIDATE_LIMIT = 5

# Cosine similarity (normalized embeddings) above which two of THIS SAME
# round's own proposals are treated as the same underlying gap for
# intra-batch clustering (see _cluster_pending_proposals) -- deliberately
# stricter than typical retrieval thresholds, since a false-positive merge
# here silently discards a proposal the graph-admission LLM call never even
# saw, not just a slightly-worse-ranked retrieval result.
_PENDING_CLUSTER_SIMILARITY_THRESHOLD = 0.92


@dataclass
class StuckEpisode:
    """One persistent "this part of the graph is stuck" saga.

    `episode_id` is assigned ONCE, the first round any of its `members` is
    ever detected in a stuck subgraph, and never changes again for the
    lifetime of this episode -- even as `members` grows (a previously
    unrelated node joins the same stuck chain) or the Dependency Graph
    Analyzer's *dynamically recomputed* grouping of those same members
    shifts round to round (e.g. because the Orchestrator edited
    depends_on). Recovery bookkeeping keys off `episode_id`, never off
    whatever compute_frontier() happens to return this round -- that
    output is topology detection, not identity, and re-deriving an
    identity from it every round is exactly the bug this type exists to
    prevent (confirmed directly on a real qibo trace: the same stuck need
    got temporary_bridge'd 4+ rounds in a row because the "root" used as
    the streak's dict key silently drifted).
    """

    episode_id: str
    members: set[str] = field(default_factory=set)
    recovery_streak: int = 0
    # Persists for the episode's whole lifetime, NOT reset when a partial
    # streak-reset happens -- once a special tactic has been tried for
    # this episode, it stays tried; a fresh episode (after this one fully
    # resolves or gets abandoned) starts its own clean record.
    used_special_tactics: set[str] = field(default_factory=set)


@dataclass
class RecoveryState:
    """Coordinator-local execution bookkeeping for one task's ask() call --
    deliberately NOT part of NeedGraph (which stays pure problem
    structure/state, see its docstring).
    """

    # Every currently-open stuck episode, keyed by its own permanent
    # episode_id (see StuckEpisode's docstring) -- not by anything
    # recomputed per round.
    stuck_episodes: dict[str, StuckEpisode] = field(default_factory=dict)
    # need_id -> the episode_id it currently belongs to, for O(1) lookup
    # from a touched need_id back to its episode. A need_id is removed
    # from here (see _reconcile_stuck_episodes) once it's no longer part
    # of any stuck subgraph -- a stale mapping is otherwise harmless (its
    # resolution-rank diff each round is inert once actually resolved),
    # but keeping it accurate avoids acting on membership that no longer
    # reflects the graph's real structure.
    episode_by_need_id: dict[str, str] = field(default_factory=dict)
    # Persistent worker ids actually assigned to each need_id via ordinary
    # (non-ephemeral) assignments, accumulated unconditionally as
    # assignments execute -- looked up lazily (union over a stuck
    # episode's member ids) when a temporary_bridge is actually built, so
    # the bridge spans every real worker already tried for that episode.
    # A prior bridge's own ephemeral id is deliberately never recorded
    # here: it is not a real, reusable candidate for a future bridge.
    tried_workers_by_node: dict[str, set[str]] = field(default_factory=dict)
    # Consecutive rounds a parent has stayed in incomplete_parents (closure
    # check keeps saying its children don't fully cover it), keyed by the
    # parent's own need_id (permanent, same stability guarantee as
    # StuckEpisode -- a parent's own need_id never changes either). Same
    # abandonment threshold and rationale as a stuck episode's streak.
    incomplete_parent_streaks: dict[str, int] = field(default_factory=dict)
    # need_ids (leaf or parent) the coordinator has given up on -- excluded
    # from the frontier shown to future plan_round() calls and from the
    # closure-check pass, and surfaced honestly in the final
    # unresolved_needs output instead of silently vanishing.
    abandoned_node_ids: set[str] = field(default_factory=set)
    # Grounded Fast Repair: need_ids a NeedAlignmentVerdict judged would
    # never help answer the original question, discarded before round 0
    # (see ant.coordinator.repair.apply_alignment_verdicts). Deliberately
    # NOT abandoned_node_ids -- "this need was never a legitimate
    # question" is a different fact from "we tried and failed", and
    # abandoned_node_ids already feeds recovery-streak/evolution-memory
    # bookkeeping that a never-attempted need must not pollute. Excluded
    # from the frontier the same way (see _exclude_abandoned's call
    # sites), but never touches streaks, episodes, or GlobalMemoryStore.
    # Empty for every non-repair ask() call.
    discarded_misaligned_node_ids: set[str] = field(default_factory=set)
    # Grounded Fast Repair: need_id -> "open" | "absence_supported" |
    # "insufficient_evidence", the grounded epistemic standing a leaf need
    # had at the start of a fast-repair retry (see
    # StuckNodeSummary.epistemic_state / apply_alignment_verdicts). Only
    # ever populated by retry_from_trajectory; empty for every gen0/
    # slow-gen1 call. A need_id with no entry here (including one born
    # during the retry itself) is treated as "open" by
    # _apply_evidence_upgrade_gate's context lookup, never as "not
    # covered by the gate" -- see that function's own docstring for why
    # the gate's trigger condition does not key off membership in this
    # dict.
    epistemic_states: dict[str, str] = field(default_factory=dict)
    # Grounded Fast Repair: need_id -> the frozen "problem definition" a
    # leaf need was given the moment it entered a fast-repair retry --
    # either kept verbatim from prior_state (a node the Alignment Gate
    # said "keep") or the new reframed text (a node it said "reframe").
    # Confirmed live on a real yt-dlp trace: ordinary round-to-round
    # editing (plan_round's own free rewrite of an existing node's need
    # text, and check_need_resolution's own "partial" refined_need
    # rewrite) can legitimately narrow a leaf's wording several rounds
    # deep in the course of normal investigation -- e.g. "proxy
    # configuration validation" narrowing into "the make_socks_proxy_opts/
    # select_proxy helpers specifically" -- until a later round's
    # genuinely correct, DIRECTLY-responsive evidence (a general
    # _check_proxies method, not those two specific helpers) gets
    # rejected by check_need_resolution for not answering the
    # now-over-narrow wording, even though it squarely answers what the
    # node was originally about. _resolution_check_need swaps this frozen
    # anchor in for node.detail.description at every check_need_resolution/
    # verify_evidence_upgrade call site (never at a search/routing call
    # site -- investigation is still free to narrow, only the yardstick
    # used to JUDGE it against stays fixed), correcting a bug the
    # Alignment Gate cannot: that gate only catches a need pointed at the
    # wrong TOPIC entirely, not one still on-topic but narrowed too far to
    # correctly judge new evidence against. Deliberately NOT a NeedNode/
    # UnresolvedNeed schema field -- this is fast-repair-only ephemeral
    # bookkeeping, same as epistemic_states, until this mechanism has
    # actually proven out. A node born mid-retry gets its own anchor
    # seeded once, at the moment it is first committed to the graph (see
    # _consolidate_and_commit's call site in ask()) -- it has no gen0
    # framing to inherit, but still deserves the same protection against
    # narrowing over whatever rounds remain. Only ever updated again by an
    # explicit Alignment Gate reframe verdict (a deliberate, reviewed
    # redefinition of the problem), never by ordinary per-round editing.
    intent_anchors: dict[str, str] = field(default_factory=dict)

    @property
    def excluded_node_ids(self) -> set[str]:
        """Every need_id that must never appear in a frontier shown to
        plan_round() -- abandoned (tried and failed) union discarded-
        misaligned (Grounded Fast Repair's alignment gate said it would
        never help answer the original question). Both are excluded from
        the frontier identically; only abandoned_node_ids feeds recovery-
        streak/evolution-memory bookkeeping elsewhere (see
        discarded_misaligned_node_ids' own docstring for why they must
        stay two separate sets rather than one)."""
        return self.abandoned_node_ids | self.discarded_misaligned_node_ids


def _recovery_snapshot(recovery: RecoveryState) -> RecoverySnapshot:
    """Read-only, JSON-serializable copy of the recovery-relevant fields of
    a finished task's RecoveryState, for EvidenceState.final_recovery_state
    -- see that field's docstring for who reads this and why. Sets become
    sorted lists purely so the same input always serializes identically
    (stable trace diffs/snapshots), not because ordering is meaningful.
    """
    return RecoverySnapshot(
        stuck_episodes=[
            StuckEpisodeSnapshot(
                episode_id=episode.episode_id,
                members=sorted(episode.members),
                recovery_streak=episode.recovery_streak,
                used_special_tactics=sorted(episode.used_special_tactics),
            )
            for episode in recovery.stuck_episodes.values()
        ],
        abandoned_node_ids=sorted(recovery.abandoned_node_ids),
        tried_workers_by_node={
            need_id: sorted(worker_ids)
            for need_id, worker_ids in recovery.tried_workers_by_node.items()
        },
    )


class LocalCoordinator:
    def __init__(
        self,
        repo_root: Path,
        workers: list[WorkerCard],
        reasoner: WorkerReasoner | None = None,
        synthesizer: AnswerSynthesizer | None = None,
        memory_routes: list[MemoryRoute] | None = None,
        index_path: Path | None = None,
        cross_repo_experience: list[str] | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.workers = workers
        self.reasoner = reasoner or (
            cast(WorkerReasoner, synthesizer) if synthesizer is not None else MockLLMProvider()
        )
        self.synthesizer = synthesizer
        self.memory_routes = memory_routes or []
        self.index_path = index_path
        # Pre-fetched by the caller (see GlobalMemoryStore.retrieve_similar),
        # not owned by LocalCoordinator itself -- same separation as
        # memory_routes above, which the caller also fetches from
        # ColonyMemoryStore before construction rather than this class
        # reaching for either memory store directly.
        self.cross_repo_experience = cross_repo_experience or []

    def ask(
        self,
        question: str,
        max_rounds: int = 6,
        initial_graph: NeedGraph | None = None,
        initial_evidence: list[Evidence] | None = None,
        initial_recovery: RecoveryState | None = None,
        repair_guidance: str = "",
        forced_first_round_assignments: dict[str, list[str]] | None = None,
        forced_first_round_global_search_ids: set[str] | None = None,
        prior_answer: str = "",
        enforce_alignment: bool = False,
        coverage_gap_node_ids: list[str] | None = None,
        targeted_need_ids: set[str] | None = None,
    ) -> EvidenceState:
        # max_rounds is a blunt outer safety ceiling only -- the real
        # per-node/per-subgraph stopping logic is the Dependency Graph
        # Analyzer (ready/blocked/stuck) plus RecoveryState's bounded
        # recovery-attempt streaks below, not a fixed round count.
        #
        # initial_graph/initial_evidence/initial_recovery/repair_guidance/
        # forced_first_round_* all default to the values that reproduce
        # today's exact bootstrap (fresh root node, empty evidence, fresh
        # RecoveryState, no guidance, nothing forced) -- every existing
        # caller is unaffected. They exist for
        # LocalCoordinator.retry_from_trajectory (task-conditioned/"fast"
        # evolution, see ant.coordinator.repair): seeding a repaired graph
        # and the prior attempt's evidence pool means a retry only has to
        # make incremental progress on what was stuck, not re-discover
        # everything the first attempt already found.
        # forced_first_round_assignments/forced_first_round_global_search_ids
        # are honored ONLY at round_index == 0 (see below), which is the
        # entire enforcement mechanism -- a repair plan's execution-policy
        # actions run exactly once, deterministically, before the
        # Orchestrator regains its ordinary per-round freedom.
        #
        # prior_answer/enforce_alignment are Grounded Fast Repair's two
        # remaining knobs, both set ONLY by retry_from_trajectory, both
        # ""/False (i.e. today's exact behavior) for every other caller:
        # prior_answer, when non-empty, switches synthesize()/
        # synthesize_coalition() into patch mode (see _patch_instruction
        # in openai_provider.py) instead of full regeneration;
        # enforce_alignment, when True, makes every _consolidate_and_commit
        # call this round loop makes additionally reject a new-node
        # proposal that drifts from `question` (not just a duplicate), and
        # runs every round's new evidence through _apply_evidence_upgrade_gate
        # -- a claim-level admissibility check, independent of whatever
        # check_need_resolution decided about the need's own
        # resolved/partial/unresolved status this round (see that gate's
        # own docstring: Evidence admissibility != need completion).
        evidence: list[Evidence] = list(initial_evidence) if initial_evidence is not None else []
        seen_worker_ids: set[str] = set()
        search = LocalSearchTool(self.repo_root, index_path=self.index_path)
        worker_config = WorkerRunConfig(max_tool_calls=11)
        worker_by_id = {worker.id: worker for worker in self.workers}
        memory_hints = _memory_hints_from_routes(self.memory_routes)
        # Built once per ask() call, reused every round (self.workers never
        # changes mid-task) -- see WorkerIndex's own docstring.
        worker_index = build_worker_index(self.workers)
        worker_card_embedding_index = (
            IndexStore(self.index_path).load_embedding_index(WORKER_CARDS_KEY)
            if self.index_path is not None
            else None
        )
        recovery = initial_recovery if initial_recovery is not None else RecoveryState()
        observed_needs: list[UnresolvedNeed] = []
        incomplete_parents: list[str] = []
        # Grounded Fast Repair: accumulates across every round of this
        # ask() call, appended to only by _apply_evidence_upgrade_gate on
        # an approved verdict -- stays empty (and therefore inert) unless
        # enforce_alignment is True. Threaded to synthesize()'s patch mode
        # at the very end, alongside prior_answer.
        grounded_updates: list[GroundedUpdate] = []

        if initial_graph is not None:
            graph = initial_graph
        else:
            root = NeedNode(
                need_id="root", need=question, detail=UnresolvedNeed(description=question)
            )
            graph = NeedGraph(nodes={"root": root})
        resolution_results: dict[str, NeedResolution] = {}
        frontier = _exclude_abandoned(compute_frontier(graph), recovery.excluded_node_ids)
        _reconcile_stuck_episodes(recovery, frontier.stuck_subgraphs)
        rounds: list[PlanningRound] = []

        for round_index in range(max_rounds):
            if not frontier.ready and not frontier.blocked and not frontier.stuck_subgraphs \
                    and not incomplete_parents:
                break

            # Snapshot for this round's GraphDelta -- only depends_on/children
            # (list identity, not the object itself: both get mutated in place
            # during this round, e.g. node.children.append(...) below, so a
            # dict-of-node-references snapshot would silently "change" along
            # with the live graph instead of freezing the round's starting
            # point).
            nodes_before_round = {
                need_id: (list(node.depends_on), list(node.children))
                for need_id, node in graph.nodes.items()
            }

            stuck_tried_workers = {
                need_id: sorted(recovery.tried_workers_by_node.get(need_id, set()))
                for group in frontier.stuck_subgraphs
                for need_id in group
                if recovery.tried_workers_by_node.get(need_id)
            }
            candidate_workers, worker_relevance_rank, per_need_candidates = (
                self._candidate_workers_for_round(
                    question, graph, frontier, worker_index, worker_card_embedding_index
                )
            )
            candidate_probes = self._probe_need_candidates(
                question, graph, per_need_candidates, search
            )
            plan = _plan_round_with_cycle_validation(
                self.reasoner,
                question=question,
                graph=graph,
                resolution_results=resolution_results,
                evidence=evidence,
                workers=candidate_workers,
                memory_hints=memory_hints,
                frontier=frontier,
                incomplete_parents=incomplete_parents,
                cross_repo_experience=self.cross_repo_experience,
                repair_guidance=repair_guidance,
                stuck_tried_workers=stuck_tried_workers,
                candidate_probes=candidate_probes,
            )
            _enforce_no_repeat_stuck_assignment(plan, stuck_tried_workers)
            if round_index == 0 and forced_first_round_assignments:
                # Overrides whatever the Orchestrator itself proposed for
                # these need_ids this round -- a forced repair action must
                # actually run, not just be available for the Orchestrator
                # to ignore (see RepairSeed's docstring). Reusing the
                # ordinary plan.assignments slot means the forced
                # assignment gets the exact same execution treatment as
                # any other -- coalition detection, evidence dedup,
                # resolution check, tried_workers_by_node tracking -- for
                # free, no parallel code path to keep in sync.
                for need_id, worker_ids in forced_first_round_assignments.items():
                    plan.assignments[need_id] = list(worker_ids)
            # Captured against the graph as it stood entering this round --
            # a new-id graph_updates entry is a PROPOSAL, not a commit (see
            # RoundPlan's docstring); collected now, alongside this round's
            # worker-observed needs, for the consolidation step later in
            # the round, well after this round's own existing-node edits
            # below.
            orchestrator_new_nodes = {
                need_id: node
                for need_id, node in plan.graph_updates.items()
                if need_id not in graph.nodes
            }
            graph = _merge_plan_into_graph(graph, plan)

            # Baseline captured AFTER the graph edit, BEFORE execution: a
            # pure Orchestrator rewrite (e.g. dropping a depends_on edge)
            # must never itself count as this round's progress -- only
            # what execution actually produces should.
            pre_execution_frontier = _exclude_abandoned(
                compute_frontier(graph), recovery.excluded_node_ids
            )
            pre_resolution_status = {
                need_id: node.resolution for need_id, node in graph.nodes.items()
            }

            node_executions: list[NodeExecutionTrace] = []
            derived_resolved_nodes: list[str] = []
            pre_round_evidence_keys = {_evidence_key(item) for item in evidence}

            for need_id, worker_ids in plan.assignments.items():
                # Deterministic, not prompt-enforced: the Orchestrator may
                # only assign currently committed/active need_ids. A
                # need_id it proposed THIS round in graph_updates is not
                # merged into `graph` above (_merge_plan_into_graph never
                # adds new-id entries -- see that function's own
                # docstring), so `node` is None for it here regardless of
                # whether plan_round's own response also listed it in
                # assignments -- confirmed live via a controlled replay
                # that the model does sometimes do exactly that. Silently
                # skipped, not executed: a same-round proposal only
                # becomes assignable once a LATER consolidate_graph call
                # admits it and a subsequent round's frontier includes it.
                node = graph.nodes.get(need_id)
                selected = [worker_by_id[wid] for wid in worker_ids if wid in worker_by_id]
                if node is None or not selected:
                    continue

                recovery.tried_workers_by_node.setdefault(need_id, set()).update(
                    worker.id for worker in selected
                )

                query = self._query_from_needs(question, [node.detail])
                observations, round_needs = self._run_selected_workers(
                    selected,
                    query,
                    question,
                    search,
                    worker_config,
                    evidence,
                    seen_worker_ids,
                    need_id=need_id,
                )
                coalition_formed = len(selected) > 1
                if coalition_formed:
                    _add_coalition_cross_checks(observations, evidence)
                    joint_observation = _run_coalition_cross_check(
                        reasoner=self.reasoner,
                        question=question,
                        selected=selected,
                        evidence=evidence,
                        search=search,
                    )
                    if joint_observation is not None:
                        observations.append(joint_observation)
                        evidence.extend(joint_observation.evidence)
                        round_needs.extend(joint_observation.unresolved_needs)
                round_normalized = _normalize_coverage_needs(question, round_needs, self.workers)
                observed_needs = _merge_needs(observed_needs, round_normalized)

                new_evidence = [
                    item
                    for obs in observations
                    for item in obs.evidence
                    if _evidence_key(item) not in pre_round_evidence_keys
                ]
                resolution_check_need = _resolution_check_need(
                    node.detail, need_id, recovery, enforce_alignment
                )
                resolution_before = node.resolution
                cumulative_evidence = _cumulative_need_evidence(evidence, need_id)
                resolution = self.reasoner.check_need_resolution(
                    need=resolution_check_need, new_evidence=cumulative_evidence, question=question
                )
                resolution, grounded_update = self._apply_evidence_upgrade_gate(
                    resolution=resolution,
                    need=resolution_check_need,
                    need_id=need_id,
                    new_evidence=new_evidence,
                    recovery=recovery,
                    question=question,
                    enforce_alignment=enforce_alignment,
                )
                if grounded_update is not None:
                    grounded_updates.append(grounded_update)
                resolution_results[need_id] = resolution
                node.resolution = resolution.status
                if resolution.status == "partial" and resolution.refined_need is not None:
                    node.detail = resolution.refined_need
                    node.need = resolution.refined_need.description

                need_candidates = per_need_candidates.get(need_id, {})
                need_probes = candidate_probes.get(need_id, {})
                node_executions.append(
                    NodeExecutionTrace(
                        need_id=need_id,
                        need=node.need,
                        worker_ids=[worker.id for worker in selected],
                        coalition_formed=coalition_formed,
                        resolution=resolution.status,
                        evidence_gain=len(new_evidence),
                        cumulative_evidence_count=len(cumulative_evidence),
                        resolution_before=resolution_before,
                        resolution_after=resolution.status,
                        need_reduction=int(resolution.status == "resolved"),
                        observations=observations,
                        candidate_worker_ids=sorted(need_candidates),
                        candidate_worker_ranks=need_candidates,
                        candidate_probe_anchor_counts={
                            worker_id: len(anchors) for worker_id, anchors in need_probes.items()
                        },
                    )
                )

            if round_index == 0 and forced_first_round_global_search_ids:
                # force_global_search's forced execution: a broad,
                # unrestricted-territory search, same shape as the
                # existing "global_fallback" special tactic below but
                # triggered directly rather than through that tactic's
                # episode/used_special_tactics machinery -- this only ever
                # runs once, at round 0, by construction (nothing re-adds
                # to forced_first_round_global_search_ids on a later
                # round), so it needs no separate anti-repeat bookkeeping
                # of its own.
                for need_id in forced_first_round_global_search_ids:
                    node = graph.nodes.get(need_id)
                    if node is None:
                        continue
                    all_files = sorted(
                        {file for worker in self.workers for file in worker.files}
                    )
                    hits = search.search(
                        self._query_from_needs(question, [node.detail]), all_files, limit=8
                    )
                    new_evidence = [
                        item.model_copy(update={"need_ids": [need_id]})
                        for item in hits
                        if _evidence_key(item) not in pre_round_evidence_keys
                    ]
                    evidence.extend(new_evidence)
                    resolution_check_need = _resolution_check_need(
                        node.detail, need_id, recovery, enforce_alignment
                    )
                    resolution_before = node.resolution
                    cumulative_evidence = _cumulative_need_evidence(evidence, need_id)
                    resolution = self.reasoner.check_need_resolution(
                        need=resolution_check_need,
                        new_evidence=cumulative_evidence,
                        question=question,
                    )
                    resolution, grounded_update = self._apply_evidence_upgrade_gate(
                        resolution=resolution,
                        need=resolution_check_need,
                        need_id=need_id,
                        new_evidence=new_evidence,
                        recovery=recovery,
                        question=question,
                        enforce_alignment=enforce_alignment,
                    )
                    if grounded_update is not None:
                        grounded_updates.append(grounded_update)
                    resolution_results[need_id] = resolution
                    node.resolution = resolution.status
                    node_executions.append(
                        NodeExecutionTrace(
                            need_id=need_id,
                            need=node.need,
                            coalition_formed=False,
                            resolution=resolution.status,
                            special_tactic="global_fallback",
                            evidence_gain=len(new_evidence),
                            cumulative_evidence_count=len(cumulative_evidence),
                            resolution_before=resolution_before,
                            resolution_after=resolution.status,
                            need_reduction=int(resolution.status == "resolved"),
                            observations=[
                                WorkerObservation(
                                    worker_id="global-fallback",
                                    territory_id="global",
                                    evidence=hits,
                                    stop_reason="global_fallback",
                                )
                            ],
                        )
                    )
                    # Drop any special_tactic the Orchestrator's own
                    # plan_round() call independently proposed for this
                    # same need_id this round -- confirmed live on real
                    # qibo/seaborn traces: nothing previously stopped
                    # plan.special_tactics from also choosing
                    # global_fallback (or temporary_bridge) for a need_id
                    # already force-executed above, running global_fallback
                    # twice in the same round for no new evidence, at real
                    # API cost. The forced execution already happened; nothing
                    # left for the Orchestrator's own choice to add here.
                    plan.special_tactics.pop(need_id, None)

            for need_id, tactic in plan.special_tactics.items():
                episode = _episode_for_need(recovery, need_id)
                node = graph.nodes.get(need_id)
                if node is None or episode is None or tactic not in (
                    "temporary_bridge",
                    "global_fallback",
                ):
                    continue
                if tactic in episode.used_special_tactics:
                    # Already tried for this exact episode -- re-running it
                    # (same bridge/territory, since neither the tried-worker
                    # set nor the search scope has changed) would just spend
                    # tool-call budget to rediscover nothing new. Confirmed
                    # directly on a real qibo trace: the same need got
                    # temporary_bridge'd 4+ consecutive rounds, almost all
                    # ev_gain=0, because nothing stopped the identical tactic
                    # from being proposed and re-executed over and over.
                    # Record it as a real (touched, no-progress) recovery
                    # attempt -- the streak finalization below still counts
                    # it -- without actually re-running anything.
                    node_executions.append(
                        NodeExecutionTrace(
                            need_id=need_id,
                            need=node.need,
                            coalition_formed=False,
                            resolution=node.resolution,
                            special_tactic=tactic,
                            evidence_gain=0,
                            cumulative_evidence_count=len(
                                _cumulative_need_evidence(evidence, need_id)
                            ),
                            resolution_before=node.resolution,
                            resolution_after=node.resolution,
                            need_reduction=0,
                        )
                    )
                    # Every other node_executions.append() site sets this
                    # need_id's own entry in resolution_results right
                    # alongside its own trace -- this is the one branch
                    # that skipped it (nothing new happened, so nothing
                    # "to record"), but touched_this_round below is built
                    # from node_executions alone, and _resolution_advanced
                    # unconditionally indexes resolution_results for every
                    # touched need_id. Confirmed live: a real yt-dlp trace
                    # crashed with KeyError the first time a need hit this
                    # exact skip branch, because nothing else in this round
                    # happened to also touch it through a path that does
                    # set the entry. node.resolution is already this need's
                    # current, real status (unchanged by this no-op) --
                    # not a placeholder.
                    resolution_results[need_id] = NeedResolution(status=node.resolution)
                    continue
                if tactic == "temporary_bridge":
                    tried_worker_ids: set[str] = set()
                    for member_id in episode.members:
                        tried_worker_ids |= recovery.tried_workers_by_node.get(member_id, set())
                    tried_workers = [
                        worker_by_id[wid] for wid in tried_worker_ids if wid in worker_by_id
                    ]
                    if not tried_workers:
                        continue
                    bridge = _build_temporary_bridge(tried_workers)
                    observations, _ = self._run_selected_workers(
                        [bridge],
                        self._query_from_needs(question, [node.detail]),
                        question,
                        search,
                        worker_config,
                        evidence,
                        seen_worker_ids,
                        need_id=need_id,
                    )
                    worker_ids_used = [bridge.id]
                    # bridge.id is deliberately NOT written to
                    # tried_workers_by_node -- ephemeral, never a real
                    # persistent candidate for a future bridge.
                else:  # global_fallback
                    all_files = sorted({file for worker in self.workers for file in worker.files})
                    hits = [
                        item.model_copy(update={"need_ids": [need_id]})
                        for item in search.search(
                            self._query_from_needs(question, [node.detail]), all_files, limit=8
                        )
                    ]
                    evidence.extend(hits)
                    observations = [
                        WorkerObservation(
                            worker_id="global-fallback",
                            territory_id="global",
                            evidence=hits,
                            stop_reason="global_fallback",
                        )
                    ]
                    worker_ids_used = []
                episode.used_special_tactics.add(tactic)

                new_evidence = [
                    item
                    for obs in observations
                    for item in obs.evidence
                    if _evidence_key(item) not in pre_round_evidence_keys
                ]
                resolution_check_need = _resolution_check_need(
                    node.detail, need_id, recovery, enforce_alignment
                )
                resolution_before = node.resolution
                cumulative_evidence = _cumulative_need_evidence(evidence, need_id)
                resolution = self.reasoner.check_need_resolution(
                    need=resolution_check_need, new_evidence=cumulative_evidence, question=question
                )
                resolution, grounded_update = self._apply_evidence_upgrade_gate(
                    resolution=resolution,
                    need=resolution_check_need,
                    need_id=need_id,
                    new_evidence=new_evidence,
                    recovery=recovery,
                    question=question,
                    enforce_alignment=enforce_alignment,
                )
                if grounded_update is not None:
                    grounded_updates.append(grounded_update)
                resolution_results[need_id] = resolution
                node.resolution = resolution.status
                node_executions.append(
                    NodeExecutionTrace(
                        need_id=need_id,
                        need=node.need,
                        worker_ids=worker_ids_used,
                        coalition_formed=False,
                        resolution=resolution.status,
                        special_tactic=tactic,
                        evidence_gain=len(new_evidence),
                        cumulative_evidence_count=len(cumulative_evidence),
                        resolution_before=resolution_before,
                        resolution_after=resolution.status,
                        need_reduction=int(resolution.status == "resolved"),
                        observations=observations,
                    )
                )

            # Need Graph Consolidation: propose -> execute (above) ->
            # consolidate (here) -> closure check (below). This round's
            # own new-node proposals plus the persistent worker-observed-
            # needs buffer are unified and handed to the Graph Organizer
            # together -- neither becomes a real node on its own anymore
            # (see RoundPlan's docstring, GraphConsolidationPlan). The
            # buffer is fully consumed every round (a proposal with no
            # explicit decision now defaults to "drop", never "create" --
            # see _apply_consolidation_decisions), so nothing carries over.
            proposals = _collect_proposals(orchestrator_new_nodes, observed_needs)
            pre_consolidation_node_ids = set(graph.nodes)
            graph = self._consolidate_and_commit(
                question, graph, proposals, recovery, enforce_alignment=enforce_alignment
            )
            if enforce_alignment:
                # A node born mid-retry has no gen0 framing to inherit an
                # intent_anchor from (retry_from_trajectory only seeds
                # kept/reframed leaves) -- seed it here, once, at the
                # moment it is first committed, so it gets the same
                # protection against narrowing over whatever rounds
                # remain (see RecoveryState.intent_anchors' own
                # docstring).
                for new_node_id in set(graph.nodes) - pre_consolidation_node_ids:
                    recovery.intent_anchors.setdefault(
                        new_node_id, graph.nodes[new_node_id].detail.description
                    )
            graph = _prune_dangling_edges(graph)
            observed_needs = []

            # Closure check: a parent whose children just all resolved.
            # "unresolved" is explicitly handed back to the Orchestrator
            # (via incomplete_parents next round) instead of silently
            # stalling -- a parent with children is never itself
            # assignable, so without this it could sit forever with no
            # path forward once its (incomplete) children were all done.
            new_incomplete_parents: list[str] = []
            # list(...) snapshot: a "partial" verdict below adds a new gap
            # node straight into graph.nodes mid-loop (line ~554), which
            # would otherwise raise "dictionary changed size during
            # iteration" -- confirmed live on a real retry_from_trajectory
            # run. The snapshot is correct, not just crash-avoidance: a
            # gap node created this pass is freshly unresolved with no
            # children of its own, so it has nothing for this same
            # closure-check pass to evaluate anyway -- it's picked up on
            # its own merits in a later round once it might have children.
            for node in list(graph.nodes.values()):
                if node.need_id in recovery.excluded_node_ids:
                    continue
                if (
                    node.children
                    and node.resolution != "resolved"
                    and all(
                        graph.nodes[child_id].resolution == "resolved"
                        for child_id in node.children
                    )
                ):
                    closure = self.reasoner.check_need_resolution(
                        need=node.detail, new_evidence=evidence, question=question
                    )
                    if closure.status == "resolved":
                        node.resolution = "resolved"
                        derived_resolved_nodes.append(node.need_id)
                        recovery.incomplete_parent_streaks.pop(node.need_id, None)
                    elif closure.status == "partial" and closure.refined_need is not None:
                        gap = NeedNode(
                            need_id=f"{node.need_id}-gap-{len(node.children)}",
                            need=closure.refined_need.description,
                            detail=closure.refined_need,
                        )
                        graph.nodes[gap.need_id] = gap
                        node.children.append(gap.need_id)
                        recovery.incomplete_parent_streaks.pop(node.need_id, None)
                    else:
                        new_incomplete_parents.append(node.need_id)
                        streak = recovery.incomplete_parent_streaks.get(node.need_id, 0) + 1
                        recovery.incomplete_parent_streaks[node.need_id] = streak
                        if streak >= _MAX_CONSECUTIVE_FAILED_RECOVERIES:
                            recovery.abandoned_node_ids.add(node.need_id)
                            new_incomplete_parents.remove(node.need_id)
            incomplete_parents = new_incomplete_parents

            post_frontier = _exclude_abandoned(compute_frontier(graph), recovery.excluded_node_ids)
            newly_ready = set(post_frontier.ready) - set(pre_execution_frontier.ready)
            touched_this_round = {trace.need_id for trace in node_executions}

            # Per-node progress: a rank-increasing resolution transition
            # only (partial -> partial is NOT progress, even though its
            # status is "not unresolved") -- only touched nodes accumulate
            # this counter, since an untouched blocked node was never given
            # a chance to fail in the first place.
            for need_id in touched_this_round:
                node = graph.nodes[need_id]
                if _resolution_advanced(
                    need_id, touched_this_round, pre_resolution_status, resolution_results
                ):
                    node.rounds_without_progress = 0
                    node.progress = "not_stuck"
                else:
                    node.rounds_without_progress += 1
                    if node.rounds_without_progress >= _STUCK_THRESHOLD:
                        node.progress = "stuck"

            # Recovery-streak finalization: episode-LOCAL (see StuckEpisode),
            # and covers every kind of recovery attempt (special tactic,
            # ordinary reassignment/coalition, or redecomposition via
            # graph_updates) touching an episode's members this round -- not
            # just special_tactics, so "just keep replanning without a
            # special tactic" cannot dodge the streak counter. Uses the
            # PRE-round episode membership (frontier.stuck_subgraphs, as
            # reconciled at the end of the previous round) -- reconciling
            # against this round's post_frontier happens after, for the
            # *next* round, so a member that resolved this round (and so
            # would no longer appear in a freshly recomputed group) still
            # gets credited as progress for the episode it was part of when
            # the round started. Progress = resolution rank advanced for ANY
            # current member, or a dependency release produced a new ready
            # node among them -- matches every criterion a real recovery
            # attempt can succeed by, not just "fully resolved".
            touched_episode_ids: set[str] = set()
            for need_id in [*plan.special_tactics, *plan.assignments, *plan.graph_updates]:
                episode = _episode_for_need(recovery, need_id)
                if episode is not None:
                    touched_episode_ids.add(episode.episode_id)
            for episode_id in touched_episode_ids:
                episode = recovery.stuck_episodes.get(episode_id)
                if episode is None:
                    continue
                progressed = bool(newly_ready & episode.members) or any(
                    _resolution_advanced(
                        member_id, touched_this_round, pre_resolution_status, resolution_results
                    )
                    for member_id in episode.members
                )
                if progressed:
                    episode.recovery_streak = 0
                else:
                    episode.recovery_streak += 1
                    if episode.recovery_streak >= _MAX_CONSECUTIVE_FAILED_RECOVERIES:
                        recovery.abandoned_node_ids.update(episode.members)
                        del recovery.stuck_episodes[episode_id]
                        for member_id in episode.members:
                            recovery.episode_by_need_id.pop(member_id, None)

            graph_delta = GraphDelta(
                created_nodes=[
                    need_id for need_id in graph.nodes if need_id not in nodes_before_round
                ],
                dependency_changes={
                    need_id: list(node.depends_on)
                    for need_id, node in graph.nodes.items()
                    if need_id in nodes_before_round
                    and list(node.depends_on) != nodes_before_round[need_id][0]
                },
                created_children={
                    need_id: list(node.children)
                    for need_id, node in graph.nodes.items()
                    if need_id in nodes_before_round
                    and list(node.children) != nodes_before_round[need_id][1]
                },
                assignment_changes=dict(plan.assignments),
                closure_results=list(derived_resolved_nodes),
            )
            rounds.append(
                PlanningRound(
                    round_index=round_index,
                    node_executions=node_executions,
                    derived_resolved_nodes=derived_resolved_nodes,
                    graph_delta=graph_delta,
                )
            )
            frontier = _exclude_abandoned(post_frontier, recovery.excluded_node_ids)
            _reconcile_stuck_episodes(recovery, frontier.stuck_subgraphs)

        # Inheritance is a global structural fact, not a territory-scoped one:
        # recruitment routing only ever proves "this worker's own files have
        # no more subclasses", never "the repository has no more subclasses
        # elsewhere". Run one definitive scan across every indexed file so
        # completeness claims are actually true instead of hedged.
        inheritance_evidence, inheritance_proof = _verify_inheritance_completeness(
            question, self.workers, search
        )
        # Unconditional: this is the only dedup pass ever applied to the
        # full accumulated pool before ranking/synthesis (every per-round
        # evidence.extend() call above adds raw, undeduped worker output).
        # Gating this behind `if inheritance_evidence:` -- true only for
        # subclass/inheritance-lookup questions -- meant every other
        # question (the overwhelming majority) skipped it entirely.
        # Confirmed on a real qibo trace: the same 2-line `set_seed`
        # definition, independently rediscovered by the same worker across
        # different need executions, was kept 4 times in the final pool,
        # crowding out genuinely distinct evidence the run had also found.
        evidence = _dedupe_evidence([*evidence, *inheritance_evidence])

        # Keyed by need_id (not just a flat list) so _absence_proofs can
        # tag each proof with the specific node it's about -- built
        # independently of the flat unresolved_needs list below, since
        # that list is about to be extended with _coverage_needs'/
        # _close_resolved_needs' own synthetic entries, which never had a
        # real graph need_id to begin with (see _absence_proofs' own
        # docstring on why those must not get a guessed owner).
        unresolved_nodes_by_id = {
            node.need_id: _with_source_worker_fallback(node, recovery)
            for node in graph.nodes.values()
            # Grounded Fast Repair: a discarded-misaligned node is not a
            # real gap in the task (it was judged irrelevant to the
            # original question) -- excluded from the surfaced list
            # entirely, leaf or not, unlike an abandoned node (which IS a
            # genuine gap the coordinator gave up on and must be reported
            # honestly rather than silently vanishing).
            if node.need_id not in recovery.discarded_misaligned_node_ids
            and node.resolution != "resolved"
            and (not node.children or node.need_id in recovery.abandoned_node_ids)
        }
        unresolved_needs = list(unresolved_nodes_by_id.values())
        unresolved_needs = unresolved_needs + _coverage_needs(
            question, evidence, unresolved_needs, self.workers
        )
        unresolved_needs = _close_resolved_needs(unresolved_needs, evidence, question)

        if not evidence and not unresolved_needs:
            unresolved_needs.append(
                UnresolvedNeed(
                    description="No local evidence matched the question in the selected workers.",
                    kind="coverage_gap",
                    need_type=(
                        "implementation_location"
                        if _asks_for_source_implementation_text(question)
                        else "negative_presence"
                    ),
                    missing="Grounded evidence for the requested symbol, behavior, or absence.",
                    scope="unknown",
                    relevant_symbols=sorted(_relevant_symbols(question)),
                    suggested_terms=[
                        term
                        for term in TOKEN_RE.findall(question)
                        if term.lower() not in STOP_WORDS
                    ][:8],
                    suggested_territories=[worker.territory_id for worker in self.workers[:5]],
                )
            )

        absence_proofs = _absence_proofs(question, rounds, unresolved_nodes_by_id, self.workers)
        if inheritance_proof is not None:
            absence_proofs.append(inheritance_proof)

        answer = ""
        if enforce_alignment and prior_answer and not grounded_updates:
            # Grounded Fast Repair must be monotonic: if not one leaf
            # transition this retry earned a verified GroundedUpdate (the
            # Evidence Upgrade Gate approved zero upgrades all retry long),
            # nothing new was actually confirmed -- re-selecting evidence
            # and re-synthesizing anyway is pure downside, never upside.
            # _select_evidence's own LLM judgment call is a fresh,
            # independently-unstable decision every time it runs (not a
            # deterministic filter), and patch-mode re-synthesis is itself
            # a further source of wording drift -- both risk degrading an
            # already-good answer for no compensating chance of
            # improvement when there is nothing grounded to patch in.
            # Confirmed empirically across a 48-question audit: 7 retries
            # found nothing to target at all (0 rounds run) and still
            # re-selected evidence and re-synthesized regardless -- 5 of
            # those 7 scored WORSE than gen0's own answer, 1 flat, only 1
            # improved. Keep gen0's own answer and its own evidence pool
            # (not yet re-selected at this point) completely untouched
            # instead. This branch can only ever fire on a fast-repair
            # retry (enforce_alignment/prior_answer are both gen0/
            # slow-gen1-never-set), so gen0 and slow-gen1's own behavior
            # is unaffected.
            answer = prior_answer
        else:
            preserved_claim_descriptions: list[str] | None = None
            if enforce_alignment:
                # Per-claim evidence retention: an association with a need
                # this retry never reopened is preserved unconditionally,
                # full stop -- it never even reaches _select_evidence's
                # judgment. An association with a REOPENED need is kept
                # only if that need actually produced a GroundedUpdate
                # this retry (the existing Evidence Upgrade Gate, not
                # _select_evidence, is the real claim-support verifier --
                # _select_evidence only ever asks "is this chunk worth
                # showing at all", a display-budget cut, never "does this
                # chunk support this specific need"). Confirmed live this
                # distinction matters: a qibo retry correctly routed a
                # Question Coverage node to the right worker, but the
                # final _select_evidence call, judging the whole mixed
                # pool fresh with no memory of what was already grounded,
                # still picked the wrong evidence over it (-24 points); a
                # seaborn retry silently lost gen0's own already-good
                # evidence for an untouched need the same way (-14
                # points). An item shared between a reopened and an
                # untouched need keeps its untouched association
                # regardless of the reopened one's fate -- a rejection (or
                # simply no GroundedUpdate) for the reopened claim must
                # never delete support for the untouched one.
                reopened = _reopened_need_ids(
                    rounds, coverage_gap_node_ids or [], targeted_need_ids or set()
                )
                grounded_this_retry = {update.need_id for update in grounded_updates}

                # Provenance telemetry, not a gate: an item with need_ids=[]
                # can never be preserved (requires non-empty need_ids) nor
                # survive revalidation (an empty need_ids has nothing for
                # the kept_ids comprehension below to iterate), so it is
                # silently unrecoverable here regardless of whether its
                # owning need was ever grounded. A full audit of every
                # evidence entry point in this file found exactly one
                # accepted source for this: _verify_inheritance_completeness's
                # own `inheritance_evidence`, deliberately untagged because
                # it is a repository-wide structural fact, not any single
                # need's claim (see that function's own docstring) --
                # tagging it to whichever need happened to trigger this
                # ask() call would be fake provenance, not a fix. Anything
                # ELSE untagged reaching this point is a real gap in that
                # audit, not an accepted exception -- warn (never crash) so
                # a live run surfaces it instead of silently losing
                # evidence the same way the global_fallback bug did.
                untagged = [item for item in evidence if not item.need_ids]
                if untagged:
                    inheritance_keys = {_evidence_key(item) for item in inheritance_evidence}
                    unexpected = [
                        item for item in untagged if _evidence_key(item) not in inheritance_keys
                    ]
                    if unexpected:
                        warnings.warn(
                            f"{len(unexpected)}/{len(untagged)} untagged evidence item(s) "
                            "reached fast-repair final synthesis from an unaccounted-for "
                            "source (_verify_inheritance_completeness explains "
                            f"{len(untagged) - len(unexpected)} of them) -- these will be "
                            "unrecoverable under per-claim retention regardless of "
                            "grounding: "
                            + ", ".join(
                                f"{item.path}:{item.line_start}-{item.line_end}"
                                for item in unexpected[:5]
                            )
                            + ("..." if len(unexpected) > 5 else ""),
                            stacklevel=2,
                        )

                def _is_untouched(item: Evidence) -> bool:
                    return bool(item.need_ids) and not (set(item.need_ids) & reopened)

                preserved = [item for item in evidence if _is_untouched(item)]
                contested = [item for item in evidence if not _is_untouched(item)]
                revalidated: list[Evidence] = []
                for item in contested:
                    kept_ids = {
                        nid
                        for nid in item.need_ids
                        if nid not in reopened or nid in grounded_this_retry
                    }
                    if kept_ids:
                        revalidated.append(
                            item
                            if kept_ids == set(item.need_ids)
                            else item.model_copy(update={"need_ids": sorted(kept_ids)})
                        )
                ranked_evidence = _rank_global_evidence(revalidated, question)
                selected = _select_evidence(self.reasoner, question, ranked_evidence, search)
                evidence = _dedupe_evidence([*preserved, *selected])
                # Reinforcing addition on top of the evidence partition
                # above (which is what actually prevents the loss): name
                # the untouched needs explicitly in the patch prompt, so a
                # correctly-preserved claim can't still lose narrative
                # attention to newly-added evidence during synthesis
                # itself.
                preserved_need_ids = {
                    need_id for item in preserved for need_id in item.need_ids
                }
                preserved_claim_descriptions = [
                    graph.nodes[need_id].detail.description
                    for need_id in sorted(preserved_need_ids)
                    if need_id in graph.nodes
                ] or None
            else:
                ranked_evidence = _rank_global_evidence(evidence, question)
                evidence = _select_evidence(self.reasoner, question, ranked_evidence, search)
            if self.synthesizer and evidence:
                coalition_workers = _last_coalition_workers(rounds)
                if coalition_workers:
                    answer = self.synthesizer.synthesize_coalition(
                        question=question,
                        worker_ids=coalition_workers,
                        evidence=evidence,
                        absence_proofs=absence_proofs,
                        prior_answer=prior_answer,
                        grounded_updates=grounded_updates,
                        preserved_claims=preserved_claim_descriptions,
                    )
                else:
                    answer = self.synthesizer.synthesize(
                        question=question,
                        evidence=evidence,
                        absence_proofs=absence_proofs,
                        prior_answer=prior_answer,
                        grounded_updates=grounded_updates,
                        preserved_claims=preserved_claim_descriptions,
                    )
        if self.synthesizer and not answer.strip():
            # A blank final answer is never a legitimate output once a real
            # synthesizer was actually in play -- it arises either when
            # `evidence` ended up empty (nothing to synthesize from at all,
            # e.g. routing never reached the right part of the repo) or
            # when the synthesizer had evidence but judged none of it
            # directly responsive and produced empty text rather than
            # saying so (confirmed live: a real gen0 trace had 16 evidence
            # items, none of them relevant -- all from doc/tutorial-builder
            # and example scripts, none from the actual source module the
            # question was about -- and synthesize() returned "" instead of
            # an honest hedge). An explicit abstention is strictly more
            # useful than silence in both cases, and this applies uniformly
            # to every caller with a synthesizer configured (gen0,
            # slow-gen1, fast-gen1 alike), not just Grounded Fast Repair.
            # Guarded on `self.synthesizer`: a coordinator with none (e.g.
            # run_batch's own heuristic-only mode) never attempts synthesis
            # at all, and "" there is the deliberate signal
            # _fallback_prediction's caller already relies on -- not a bug.
            answer = (
                "The available evidence does not directly and reliably answer this "
                "question -- no claim could be confidently supported from what was found."
            )
        usage = (
            self.synthesizer.drain_usage() if isinstance(self.synthesizer, UsageReporter) else None
        )

        return EvidenceState(
            question=question,
            answer=answer,
            evidence=evidence,
            unresolved_needs=unresolved_needs,
            rounds=rounds,
            absence_proofs=absence_proofs,
            usage=usage if isinstance(usage, TokenUsage) else TokenUsage(),
            final_need_graph=dict(graph.nodes),
            final_recovery_state=_recovery_snapshot(recovery),
        )

    def retry_from_trajectory(
        self,
        prior_state: EvidenceState,
        fast_reasoner: FastEvolutionReasoner,
        max_rounds: int = 6,
    ) -> EvidenceState:
        """Task-conditioned ("fast") evolution: repairs `prior_state`'s own
        Need Graph from its own trajectory and retries the same question.
        Entirely ephemeral -- this method (and everything it calls) never
        writes to IndexStore/ColonyMemoryStore/GlobalMemoryStore. Nothing
        here reads a reference answer or judge score, only what
        `prior_state` already recorded about its own attempt. Nothing else
        calls this method; it is opt-in, never part of ask()'s own flow.

        Grounded Fast Repair: Need Alignment (gates ①+②) runs first, on
        `prior_state`'s own graph, before propose_repair even sees it --
        so repair targeting (which worker to force, what to merge) always
        reasons about the CORRECTED framing, never a drifted one. The
        rest of ask() (enforce_alignment=True) keeps applying the same
        alignment test to every node the retry's own rounds propose, and
        gates every leaf resolution behind direct-evidence verification
        (see WorkerReasoner.consolidate_graph's enforce_alignment
        parameter and LocalCoordinator._apply_evidence_upgrade_gate).
        Every leaf's resolution is judged against a frozen intent_anchor,
        not whatever its own wording has drifted into by the time new
        evidence arrives (see RecoveryState.intent_anchors and
        _resolution_check_need) -- catches a need that is still on-topic
        but has narrowed too far to correctly judge new evidence against,
        which the Alignment Gate above cannot (it only catches a need
        pointed at the wrong topic entirely).
        """
        package = assemble_trajectory_package(prior_state)
        alignment_plan = fast_reasoner.assess_need_alignment(
            question=prior_state.question, package=package
        )
        prior_epistemic_states = {
            node.need_id: node.epistemic_state for node in package.stuck_nodes
        }
        aligned_graph, epistemic_states, discarded_misaligned_ids, reframed_need_ids = (
            apply_alignment_verdicts(
                NeedGraph(nodes=prior_state.final_need_graph),
                alignment_plan,
                prior_epistemic_states,
            )
        )
        aligned_state = prior_state.model_copy(
            update={"final_need_graph": aligned_graph.nodes}
        )
        # Recomputed from the ALIGNED graph, not reused from `package`
        # above, so propose_repair sees reframed need text and never sees
        # a dropped need as a live stuck node at all.
        aligned_package = assemble_trajectory_package(aligned_state)

        plan = fast_reasoner.propose_repair(package=aligned_package)
        seed = resolve_repair_plan(plan)
        initial_graph, initial_evidence = build_retry_starting_state(aligned_state, seed)

        # Question Coverage Contract: independent of whatever the Need
        # Graph itself thinks is resolved -- a node reading "resolved"
        # means only that node's OWN (possibly narrower) question got
        # enough evidence, never that the original question is fully
        # answered (see AnswerObligation's own docstring for the live case
        # this was found from: a "which subclasses, and their overridden
        # methods" question had the subclass-identity half accepted as
        # resolved while the overridden-methods half was only ever
        # honestly hedged as unknown -- unresolved_needs read 0, so the
        # retry ran zero rounds and blindly re-synthesized the same
        # incomplete answer). Checked against the retry's own STARTING
        # evidence pool, before any new round runs, so a gap already
        # covered by gen0's own evidence never spuriously creates a node.
        obligations = fast_reasoner.extract_answer_obligations(question=prior_state.question)
        coverage = fast_reasoner.check_obligation_coverage(
            question=prior_state.question, obligations=obligations, evidence=initial_evidence
        )
        initial_graph, coverage_gap_node_ids = _inject_coverage_gap_nodes(
            initial_graph, obligations, coverage
        )

        prior_tried_workers = prior_state.final_recovery_state.tried_workers_by_node
        # Every kept/reframed leaf's frozen "problem definition" (see
        # RecoveryState.intent_anchors' own docstring) -- aligned_graph
        # already carries the right text for both: unchanged (gen0's own
        # wording) for a "keep" verdict, rewritten for a "reframe" one.
        # A discarded-misaligned node gets none -- excluded from the
        # frontier either way, it will never be judged against anything.
        # A newly-injected coverage-gap node gets its own description as
        # its anchor, same as any other freshly-born fast-repair node.
        intent_anchors = {
            need_id: node.detail.description
            for need_id, node in aligned_graph.nodes.items()
            if need_id not in discarded_misaligned_ids
        }
        intent_anchors.update(
            {
                need_id: initial_graph.nodes[need_id].detail.description
                for need_id in coverage_gap_node_ids
            }
        )
        initial_recovery = RecoveryState(
            tried_workers_by_node={
                need_id: set(worker_ids)
                for need_id, worker_ids in prior_tried_workers.items()
                # A reframed need is a fresh investigation (correction 2)
                # -- "worker X already tried and failed" is a fact about
                # the OLD framing, must not carry forward onto the new one.
                if need_id not in reframed_need_ids
            },
            abandoned_node_ids={
                need_id
                for need_id in prior_state.final_recovery_state.abandoned_node_ids
                if need_id not in seed.targeted_need_ids and need_id not in reframed_need_ids
            },
            discarded_misaligned_node_ids=discarded_misaligned_ids,
            epistemic_states=epistemic_states,
            intent_anchors=intent_anchors,
        )

        return self.ask(
            prior_state.question,
            max_rounds=max_rounds,
            initial_graph=initial_graph,
            initial_evidence=initial_evidence,
            initial_recovery=initial_recovery,
            repair_guidance=render_repair_guidance(aligned_package, seed),
            forced_first_round_assignments=seed.forced_assignments or None,
            forced_first_round_global_search_ids=seed.forced_global_search_ids or None,
            prior_answer=prior_state.answer,
            enforce_alignment=True,
            coverage_gap_node_ids=coverage_gap_node_ids,
            targeted_need_ids=seed.targeted_need_ids,
        )

    def _run_selected_workers(
        self,
        selected: list[WorkerCard],
        query: str,
        question: str,
        search: LocalSearchTool,
        worker_config: WorkerRunConfig,
        evidence: list[Evidence],
        seen_worker_ids: set[str],
        need_id: str = "",
    ) -> tuple[list[WorkerObservation], list[UnresolvedNeed]]:
        """Runs each selected worker in turn, mutating `evidence` and
        `seen_worker_ids` in place as it goes rather than batching updates
        until the end, so each subsequent worker's own observe() call --
        and an escalation tactic reusing this helper later in the same
        task -- sees what earlier workers already found this round, exactly
        as the original inline round loop did.

        `need_id`: stamped onto every Evidence item gathered this call as
        its provenance (see Evidence.need_ids' own docstring) -- runs for
        every ask() call, gen0/slow-gen1/fast-gen1 alike, so a later
        fast-repair retry always has real per-claim provenance to work
        with, even for evidence gen0 itself gathered. gen0/slow-gen1 never
        read this field. Optional/empty for callers with no single owning
        need (e.g. a global-fallback search not scoped to one leaf) --
        an untagged item simply can't be preserved by fast-repair's
        per-claim retention later and falls back to today's behavior.
        """
        observations: list[WorkerObservation] = []
        round_needs: list[UnresolvedNeed] = []
        for worker in selected:
            observation = AutonomousWorker(
                self.repo_root, worker, search, reasoner=self.reasoner
            ).run(
                query,
                config=worker_config,
            )
            worker_evidence = [
                item.model_copy(
                    update={
                        "worker_id": worker.id,
                        "need_ids": [need_id] if need_id else [],
                    }
                )
                for item in observation.evidence
            ]
            observation.evidence = worker_evidence
            evidence.extend(worker_evidence)
            # Give the reasoner the full shared evidence state, not just this
            # worker's own findings, so it does not re-raise a need another
            # worker already grounded (e.g. worker B finding the subclass
            # worker A's need was about).
            reasoner_context = _dedupe_evidence([*worker_evidence, *evidence])
            reasoner_observation = self.reasoner.observe(
                question=question,
                worker_id=worker.id,
                territory_id=worker.territory_id,
                evidence=reasoner_context,
            )
            observation.unresolved_needs.extend(reasoner_observation.unresolved_needs)
            observations.append(observation)
            round_needs.extend(observation.unresolved_needs)
            seen_worker_ids.add(worker.id)
        return observations, round_needs

    def _candidate_workers_for_round(
        self,
        question: str,
        graph: NeedGraph,
        frontier: FrontierResult,
        worker_index: WorkerIndex,
        embedding_index: EmbeddingIndex | None,
    ) -> tuple[list[WorkerCard], dict[str, int], dict[str, dict[str, int]]]:
        """Two-stage routing: retrieval decides *recall* (which workers are
        even shown to the Orchestrator this round), the Orchestrator keeps
        *composition* authority (which of those, alone or as a coalition,
        actually gets assigned) -- a structural narrowing, not a prompt-text
        suggestion. Confirmed live, on two different models, that an
        advisory-only rank annotation does not reliably beat a surface
        lexical association (a "gates" question pulling the Orchestrator to
        worker-src-qibo-gates over a rank-1-annotated worker-src-qibo-models):
        the fix is to not show the alternative at all, not to ask more
        persuasively.

        Only narrows ready-frontier (not-yet-stuck) needs, sized by that
        need's own rounds_without_progress: fresh (0) gets
        _FRESH_NEED_CANDIDATE_LIMIT, one quiet round (>=1, still on the
        ready frontier -- _STUCK_THRESHOLD==2 is where it actually leaves
        the ready frontier) gets the wider _ESCALATED_NEED_CANDIDATE_LIMIT.
        Once a need is genuinely stuck, this narrows nothing -- it already
        gets full worker visibility via the union below, and
        temporary_bridge/global_fallback (ant.coordinator.local's own
        special-tactic handling) already operate at full, unscoped width.
        That existing machinery *is* this ladder's final tier; this method
        does not duplicate or touch it.

        A need whose own query yields zero ranked candidates (rank_workers
        found no signal in any channel -- e.g. an all-stopword need text)
        falls back to the full worker list for that need alone: a need must
        never end up with zero candidates because retrieval happened to
        find nothing, the same "never let a filter zero out a legitimate
        scope" principle behind tonight's _territory_index corpus-exclusion
        fix, one level up.

        Returns (candidate_workers, worker_relevance_rank, per_need_candidates):
        - candidate_workers: the union to actually pass to plan_round as
          its `workers` argument (this is what gives the narrowing real
          teeth -- _parse_round_plan already validates every assignment's
          worker_id against exactly this list, dropping anything outside
          it, unchanged from tonight's advisory-rank version).
        - worker_relevance_rank: per worker id, the best (lowest/minimum)
          rank any ready need gave it -- stays meaningful as a prompt
          annotation when more than one ready need shares a candidate.
        - per_need_candidates: need_id -> {worker_id: rank}, this need's
          own top-K before the round-level union -- purely for
          NodeExecutionTrace's audit fields, not used for planning itself.
        """
        per_need_candidates: dict[str, dict[str, int]] = {}
        candidate_ids: set[str] = set()
        worker_relevance_rank: dict[str, int] = {}

        for need_id in frontier.ready:
            node = graph.nodes.get(need_id)
            if node is None:
                continue
            limit = (
                _FRESH_NEED_CANDIDATE_LIMIT
                if node.rounds_without_progress == 0
                else _ESCALATED_NEED_CANDIDATE_LIMIT
            )
            ranks = rank_workers(
                self._query_from_needs(question, [node.detail]),
                self.workers,
                worker_index,
                embedding_index,
            )
            top = dict(sorted(ranks.items(), key=lambda item: item[1])[:limit])
            if not top:
                # No retrieval signal for this need at all -- never leave
                # it with zero candidates.
                top = {worker.id: index for index, worker in enumerate(self.workers, start=1)}
            per_need_candidates[need_id] = top
            candidate_ids.update(top)
            for worker_id, rank in top.items():
                best_so_far = worker_relevance_rank.get(worker_id)
                if best_so_far is None or rank < best_so_far:
                    worker_relevance_rank[worker_id] = rank

        if frontier.stuck_subgraphs:
            # A stuck need's ordinary (non-special-tactic) reassignment
            # still needs full visibility, exactly like before this
            # change -- only ready-frontier needs are narrowed above.
            candidate_ids.update(worker.id for worker in self.workers)

        candidate_workers = (
            [worker for worker in self.workers if worker.id in candidate_ids]
            if candidate_ids
            else list(self.workers)
        )
        return candidate_workers, worker_relevance_rank, per_need_candidates

    def _probe_need_candidates(
        self,
        question: str,
        graph: NeedGraph,
        per_need_candidates: dict[str, dict[str, int]],
        search: LocalSearchTool,
    ) -> dict[str, dict[str, list[Evidence]]]:
        """Probe-then-commit: before the Orchestrator picks which candidate
        to actually assign, give each of a ready need's own narrowed
        candidates (see _candidate_workers_for_round) one cheap, local look
        into its own territory -- the same search()/dense_search() calls
        AutonomousWorker.run() itself opens with, just without the full
        reasoner-driven tool-use loop that follows them. No LLM call, no
        extra cost beyond retrieval this repo already runs everywhere else.

        Confirmed live that a rank number or a name/routing_summary that
        merely sounds relevant is not reliable enough on its own (the
        circuit-drawing question: worker-src-qibo-gates outranked by
        worker-src-qibo-models yet still picked, apparently pulled by the
        literal word "gates"). A worker that actually surfaces a matching
        anchor in its own files is stronger, harder-to-fake evidence than
        either signal -- so plan_round's prompt shows what was *found*,
        not a prior guess about what's likely.

        Scoped identically to the narrowing itself: only ready-frontier
        needs' own candidates are probed, never the stuck-subgraph full-list
        union (those needs already have richer recovery machinery and full,
        unprobed visibility -- unchanged by this method).
        """
        probes: dict[str, dict[str, list[Evidence]]] = {}
        for need_id, candidates in per_need_candidates.items():
            node = graph.nodes.get(need_id)
            if node is None:
                continue
            query = self._query_from_needs(question, [node.detail])
            worker_probes: dict[str, list[Evidence]] = {}
            for worker_id in candidates:
                worker = next((w for w in self.workers if w.id == worker_id), None)
                if worker is None:
                    continue
                lexical = search.search(query, worker.files, limit=_PROBE_ANCHOR_LIMIT)
                dense = search.dense_search(query, worker.files, limit=_PROBE_ANCHOR_LIMIT)
                worker_probes[worker_id] = _dedupe_evidence([*lexical, *dense])[
                    :_PROBE_ANCHOR_LIMIT
                ]
            probes[need_id] = worker_probes
        return probes

    def _apply_evidence_upgrade_gate(
        self,
        *,
        resolution: NeedResolution,
        need: UnresolvedNeed,
        need_id: str,
        new_evidence: list[Evidence],
        recovery: RecoveryState,
        question: str,
        enforce_alignment: bool,
    ) -> tuple[NeedResolution, GroundedUpdate | None]:
        """Grounded Fast Repair's Evidence Upgrade Gate: claim-level
        evidence ADMISSIBILITY, deliberately decoupled from
        check_need_resolution's need-level COMPLETION judgment above --
        two different questions that used to share one threshold
        (`resolution.status in ("resolved", "partial")`), which meant a
        need stuck at "unresolved" could never contribute a single
        verified claim to the final answer even when a specific piece of
        its new evidence was, on its own, directly and correctly
        supported. Confirmed live on a real seaborn trace: a worker found
        `fit_poly`'s direct call into `bootstrap()` -- textbook-clean
        supporting evidence for "how does bootstrap resampling flow
        through the regression CI pipeline" -- but the broader
        `boot_method` need itself never reached "resolved" (still missing
        the full picture), so the coupled gate discarded the evidence
        entirely instead of admitting just that claim. An audit across
        three repos found this is not an edge case: 16/18 fast-repair
        retries that did any real re-synthesis hit "found new evidence
        for a need, need never reached resolved/partial, evidence
        excluded" at least once.

        `resolution` passes through this gate completely untouched in
        every case -- this method only ever decides whether new_evidence
        produces a GroundedUpdate, never whether resolution is
        resolved/partial/unresolved (that remains check_need_resolution's
        sole responsibility; investigation continues or stops purely on
        its own verdict, independent of what this gate approves). No-op
        (no GroundedUpdate) unless `enforce_alignment` is True (i.e. this
        is a Grounded Fast Repair retry) AND there is new_evidence to
        judge this round -- nothing new to verify needs no verification
        call. Covers gen0-carried-over AND retry-born leaf needs
        identically (`recovery.epistemic_states.get(need_id, "open")` --
        a need never seeded there, e.g. one born mid-retry, is simply
        "open", not exempt from the gate), and runs regardless of
        whether check_need_resolution called this same round judged the
        need resolved, partial, or still unresolved.

        An approved verdict returns a GroundedUpdate for the caller to
        accumulate -- the only channel through which synthesize()'s patch
        mode may later strengthen this need's claim in the final answer,
        and the only source `_reopened_need_ids`'s partition logic (see
        `ask()`) treats as "this reopened need's evidence may survive
        into the answer." An unapproved verdict is a pure no-op: no
        GroundedUpdate, and `resolution` is returned exactly as passed
        in.
        """
        if not enforce_alignment or not new_evidence:
            return resolution, None
        epistemic_state = recovery.epistemic_states.get(need_id, "open")
        verdict = self.reasoner.verify_evidence_upgrade(
            need=need,
            epistemic_state=epistemic_state,
            new_evidence=new_evidence,
            question=question,
        )
        if not verdict.approved:
            return resolution, None
        return resolution, GroundedUpdate(
            need_id=need_id,
            supported_claim=verdict.supported_claim,
            evidence_ids=verdict.evidence_ids,
            prior_epistemic_state=epistemic_state,
        )

    def _consolidate_and_commit(
        self,
        question: str,
        graph: NeedGraph,
        proposals: list[ProposedNode],
        recovery: RecoveryState,
        enforce_alignment: bool = False,
    ) -> NeedGraph:
        """Need Graph Consolidation: the one place a *new* node actually
        comes into existence (see WorkerReasoner.consolidate_graph's own
        docstring). A no-op when this round proposed nothing -- no LLM
        call spent on an empty buffer.

        `enforce_alignment` is forwarded straight to
        WorkerReasoner.consolidate_graph -- True only inside a Grounded
        Fast Repair retry (see LocalCoordinator.ask), where it makes
        consolidation additionally reject a proposal that drifts from
        the original question, not just a duplicate. This is what keeps
        a node born mid-retry from drifting the same way a gen0 node
        could -- the one-time pre-retry NeedAlignmentPlan pass alone
        would not catch it.

        Intra-batch consolidation runs first (see
        _cluster_pending_proposals): only one representative per cluster
        of this round's own near-duplicate proposals ever reaches the LLM
        graph-admission call, and _expand_cluster_decisions fans its
        verdict back out afterward -- confirmed necessary on a real
        yt-dlp fast-repair retry where a single round's raw proposal
        count (hundreds, several workers each independently observing the
        same gap) silently truncated consolidate_graph's own response.
        """
        if not proposals:
            return graph
        active_nodes = {
            node_id: node
            for node_id, node in graph.nodes.items()
            if node.resolution != "resolved" and node_id not in recovery.excluded_node_ids
        }
        clusters = _cluster_pending_proposals(proposals)
        representatives = [cluster.representative for cluster in clusters]
        candidate_hints = _candidate_hints_for_proposals(active_nodes, representatives)
        consolidation_plan = self.reasoner.consolidate_graph(
            question=question,
            active_nodes=active_nodes,
            proposals=representatives,
            candidate_hints=candidate_hints,
            enforce_alignment=enforce_alignment,
        )
        expanded_plan = _expand_cluster_decisions(clusters, consolidation_plan, graph)
        # Representative before its own members, per cluster, so pass 1 of
        # _apply_consolidation_decisions always mints/resolves a
        # representative's own node before a member tries to merge into it
        # (both live in the same batch, applied in list order).
        # cluster.member_ids always starts with the representative's own
        # id (see ProposalCluster's docstring/construction), so iterating
        # it directly already yields representative-then-members order.
        proposals_by_id = {proposal.proposal_id: proposal for proposal in proposals}
        ordered_proposals = [
            proposals_by_id[proposal_id]
            for cluster in clusters
            for proposal_id in cluster.member_ids
        ]
        return _apply_consolidation_decisions(graph, ordered_proposals, expanded_plan)

    @staticmethod
    def _query_from_needs(question: str, needs: list[UnresolvedNeed]) -> str:
        # Keep the original question as a stable lexical anchor in every round's
        # query, not just as a fallback for when the need text is empty. Round 2+
        # queries are built from an LLM-generated UnresolvedNeed, and that call has
        # no temperature/seed pinning -- its wording varies between otherwise
        # identical runs. Dropping the original question's terms each round made
        # routing and retrieval fully dependent on that one call's phrasing; keeping
        # them present means a lucky/unlucky word choice can only add or subtract
        # recall, not erase the anchor entirely.
        parts = [question]
        for need in needs:
            parts.append(_need_query_text(need))
        query = " ".join(part for part in parts if part).strip()
        return query or question


def _build_temporary_bridge(tried_workers: list[WorkerCard]) -> WorkerCard:
    """An ephemeral, in-task-only worker for _escalate_stuck_need's third
    tactic: the union of every territory already tried for a stuck need,
    so a single search pass can see across their boundary instead of two
    separate territory-scoped workers each missing the other's half. Never
    saved to IndexStore -- persistent reorganization is evolve_workers'
    job, at a slow cross-task timescale earned by *repeated* need, not
    granted to a single task's stuck attempt.
    """
    files = sorted({file for worker in tried_workers for file in worker.files})
    terms = sorted({term for worker in tried_workers for term in worker.searchable_terms})[:32]
    ids = sorted(worker.id.removeprefix("worker-") for worker in tried_workers)
    bridge_id = f"worker-temp-bridge-{'-'.join(ids)}"[:96]
    names = " + ".join(worker.name for worker in tried_workers[:3])
    card = WorkerCard(
        id=bridge_id,
        territory_id="temp-bridge",
        name=f"{names} temporary bridge",
        root="",
        responsibilities=[
            "Ephemeral, task-scoped interface specialist -- not part of persistent colony state.",
            f"Combines territories: {', '.join(worker.id for worker in tried_workers)}.",
        ],
        searchable_terms=terms,
        files=files,
    )
    # Zero-cost template only, never an LLM call: this worker is built and
    # thrown away within a single task, never persisted, so it is not
    # worth spending a routing_summary generation call on -- but it still
    # needs a non-empty one, since the Orchestrator planning call reads
    # only routing_summary, not the full card, and this worker becomes a
    # candidate this same round.
    return card.model_copy(update={"routing_summary": template_routing_summary(card)})


def _plan_round_with_cycle_validation(
    reasoner: WorkerReasoner,
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
    repair_guidance: str = "",
    stuck_tried_workers: dict[str, list[str]] | None = None,
    candidate_probes: dict[str, dict[str, list[Evidence]]] | None = None,
) -> RoundPlan:
    """Calls reasoner.plan_round() and validates the graph its
    graph_updates would produce -- merged onto the existing graph -- has
    no new depends_on cycle (see graph_analyzer.find_cycles) before
    accepting it. depends_on is entirely LLM-drawn, so a cycle here is
    presumptively a planning mistake, not evidence the task's needs are
    genuinely circular: rejected and retried once with the offending cycle
    described back to the reasoner, rather than silently accepted into
    state or auto-repaired by guessing which edge to drop. Accepts
    whatever the retry returns even if still cyclic -- one retry only, so
    a round has a bounded worst case; a persistently cyclic response still
    surfaces defensively in the next round's frontier computation (see
    compute_frontier).
    """
    plan = reasoner.plan_round(
        question=question,
        graph=graph,
        resolution_results=resolution_results,
        evidence=evidence,
        workers=workers,
        memory_hints=memory_hints,
        frontier=frontier,
        incomplete_parents=incomplete_parents,
        cross_repo_experience=cross_repo_experience,
        repair_guidance=repair_guidance,
        stuck_tried_workers=stuck_tried_workers,
        candidate_probes=candidate_probes,
    )
    cycles = find_cycles(_merge_graph_updates(graph, plan))
    if not cycles:
        return plan
    cycle = cycles[0]
    feedback = (
        "Your graph_updates produced a dependency cycle: "
        + " -> ".join([*cycle, cycle[0]])
        + ". depends_on must describe a strict prerequisite order with no "
        "cycles (a node must never, directly or transitively, depend on "
        "itself) -- revise the edges so the graph is acyclic."
    )
    return reasoner.plan_round(
        question=question,
        graph=graph,
        resolution_results=resolution_results,
        evidence=evidence,
        workers=workers,
        memory_hints=memory_hints,
        frontier=frontier,
        incomplete_parents=incomplete_parents,
        cross_repo_experience=cross_repo_experience,
        validation_feedback=feedback,
        repair_guidance=repair_guidance,
        stuck_tried_workers=stuck_tried_workers,
        candidate_probes=candidate_probes,
    )


def _enforce_no_repeat_stuck_assignment(
    plan: RoundPlan, stuck_tried_workers: dict[str, list[str]]
) -> None:
    """Routing self-correction: `stuck_tried_workers` (see plan_round's own
    docstring) is advisory information, and the Orchestrator's own choice
    to repeat an already-tried worker is respected when it names even one
    worker outside that tried set (e.g. keeping a tried worker in a new
    coalition alongside someone new) -- this only overrides the one
    pattern that is never a deliberate choice: an assignment for a
    still-stuck need_id made up *entirely* of workers RecoveryState already
    recorded as tried-with-no-progress on it, which just re-runs the exact
    same thing again. Confirmed on real traces (see this session's own
    diagnosis) that a stochastic planner can do this repeatedly, several
    rounds in a row, with no self-correction -- it has no memory of its
    own past assignments unless told, and stuck_tried_workers alone (a
    prompt hint) does not reliably stop it.

    Downgrades to global_fallback rather than picking a specific
    alternative worker algorithmically: that judgment call -- which
    *particular* other worker is a good complementary choice -- belongs to
    the Orchestrator's own reasoning (now that it has been told what
    failed), not to a mechanical substitution here. This only fires when
    the Orchestrator did not diversify despite that information; the
    existing global_fallback tactic (unscoped repo-wide search) is the
    one safe, already-tested escape hatch that requires no such judgment.
    Mutates `plan` in place -- called right after plan_round returns, same
    pattern as the forced_first_round_assignments override.
    """
    for need_id, tried in stuck_tried_workers.items():
        assigned = plan.assignments.get(need_id)
        if not assigned or not tried:
            continue
        if set(assigned) - set(tried):
            continue  # at least one new worker in the mix -- a real choice
        del plan.assignments[need_id]
        plan.special_tactics.setdefault(need_id, "global_fallback")


def _merge_graph_updates(graph: NeedGraph, plan: RoundPlan) -> NeedGraph:
    """Validation-only merge: used purely to check whether accepting
    `plan.graph_updates` as-is would introduce a depends_on cycle (see
    _plan_round_with_cycle_validation), not to actually advance coordinator
    state -- that is _merge_plan_into_graph's job, which additionally
    preserves each existing node's resolution/execution/progress instead
    of blindly overwriting with the Orchestrator's (necessarily
    default-valued) copy.
    """
    return graph.model_copy(update={"nodes": {**graph.nodes, **plan.graph_updates}})


def _merge_plan_into_graph(graph: NeedGraph, plan: RoundPlan) -> NeedGraph:
    """Applies plan.graph_updates onto `graph` for *existing* nodes only:
    need/depends_on/related_to/children/detail get overwritten -- its
    resolution/execution/progress/rounds_without_progress are preserved
    untouched, since the Orchestrator never legitimately writes those
    three dimensions (see NeedNode's docstring) and its own copy of an
    existing node only ever carries their default values, not real ones.

    A *new* need_id in plan.graph_updates is deliberately NOT added here
    -- see RoundPlan's own docstring: a new-id entry is only a proposal
    for this round's Potential Needs Buffer, collected separately (see
    LocalCoordinator._collect_proposals) and only becomes a real node if
    WorkerReasoner.consolidate_graph decides to create/attach/relate it.
    """
    nodes = dict(graph.nodes)
    for need_id, update in plan.graph_updates.items():
        existing = nodes.get(need_id)
        if existing is None:
            continue
        nodes[need_id] = existing.model_copy(
            update={
                "need": update.need,
                "depends_on": update.depends_on,
                "related_to": update.related_to,
                "children": update.children,
                "detail": update.detail,
            }
        )
    return graph.model_copy(update={"nodes": nodes})


def _reconcile_stuck_episodes(recovery: RecoveryState, stuck_subgraphs: list[list[str]]) -> None:
    """Updates RecoveryState.stuck_episodes/episode_by_need_id from this
    round's freshly-detected stuck_subgraphs (pure topology, recomputed
    every round by compute_frontier -- see FrontierResult), WITHOUT ever
    inventing a new identity for a group that already has one.

    For each detected group: if any of its members already belong to an
    existing episode, that group is folded into that episode (the
    EARLIEST-created one, by episode_id, if more than one existing episode
    is being merged) -- episode_id itself never changes. Only a group with
    no existing-episode member at all gets a brand-new episode, anchored
    at its own smallest need_id (deterministic, but chosen exactly once,
    at creation -- not recomputed on a later round the way the old
    per-round "root" was).
    """
    for group in stuck_subgraphs:
        existing_ids = {
            recovery.episode_by_need_id[need_id]
            for need_id in group
            if need_id in recovery.episode_by_need_id
        }
        if not existing_ids:
            episode_id = min(group)
            episode = StuckEpisode(episode_id=episode_id, members=set(group))
            recovery.stuck_episodes[episode_id] = episode
            for need_id in group:
                recovery.episode_by_need_id[need_id] = episode_id
            continue

        target_id = min(existing_ids)
        target = recovery.stuck_episodes[target_id]
        for other_id in existing_ids - {target_id}:
            other = recovery.stuck_episodes.pop(other_id)
            target.members |= other.members
            # A streak only means something once merged nodes are treated
            # as one problem -- keep whichever count is further along
            # (closer to abandonment), not reset either side's history.
            target.recovery_streak = max(target.recovery_streak, other.recovery_streak)
            target.used_special_tactics |= other.used_special_tactics
        target.members |= set(group)
        for need_id in group:
            recovery.episode_by_need_id[need_id] = target_id


def _episode_for_need(recovery: RecoveryState, need_id: str) -> StuckEpisode | None:
    episode_id = recovery.episode_by_need_id.get(need_id)
    if episode_id is None:
        return None
    return recovery.stuck_episodes.get(episode_id)


def _resolution_advanced(
    need_id: str,
    touched_this_round: set[str],
    pre_resolution_status: dict[str, str],
    resolution_results: dict[str, NeedResolution],
) -> bool:
    """True if `need_id` was actually touched this round and its
    resolution rank strictly increased (unresolved->partial/resolved, or
    partial->resolved) -- partial->partial, or a node untouched this
    round, is never progress. Shared by the per-node stuck counter and a
    stuck episode's recovery streak, which both need exactly this same
    judgment.
    """
    if need_id not in touched_this_round:
        return False
    before = pre_resolution_status.get(need_id, "unresolved")
    after = resolution_results[need_id].status
    return _RESOLUTION_RANK[after] > _RESOLUTION_RANK[before]


def _exclude_abandoned(frontier: FrontierResult, abandoned_node_ids: set[str]) -> FrontierResult:
    """Filters a coordinator-given-up-on need_id out of every part of a
    FrontierResult before it's shown to the next plan_round() call or used
    for the round loop's own termination check -- compute_frontier() has
    no knowledge of RecoveryState.abandoned_node_ids (deliberately, see
    RecoveryState's docstring), so this is applied by the coordinator
    itself on the analyzer's raw output.
    """
    if not abandoned_node_ids:
        return frontier
    return FrontierResult(
        ready=[need_id for need_id in frontier.ready if need_id not in abandoned_node_ids],
        blocked=[need_id for need_id in frontier.blocked if need_id not in abandoned_node_ids],
        stuck_subgraphs=[
            [need_id for need_id in group if need_id not in abandoned_node_ids]
            for group in frontier.stuck_subgraphs
            if any(need_id not in abandoned_node_ids for need_id in group)
        ],
    )


def _with_source_worker_fallback(node: NeedNode, recovery: RecoveryState) -> UnresolvedNeed:
    """Graph-node needs (node.detail) never get source_worker_id populated
    by the Orchestrator -- _parse_graph_update has no notion of "which
    worker" for a need it is authoring, only observe()-sourced needs in
    the observed_needs buffer ever carried one. Backfill it here, from
    whichever persistent worker(s) were actually tried on this need (see
    RecoveryState.tried_workers_by_node), so record_task_memory's per-need
    struggling-worker route (which requires a non-empty source_worker_id)
    keeps working for graph-authored needs too, not just observed ones.
    """
    if node.detail.source_worker_id:
        return node.detail
    tried = sorted(recovery.tried_workers_by_node.get(node.need_id, set()))
    if not tried:
        return node.detail
    return node.detail.model_copy(update={"source_worker_id": tried[0]})


def _memory_hints_from_routes(routes: list[MemoryRoute]) -> dict[str, str]:
    """Surfaces repo-local memory routes as text the Orchestrator can
    actually read in its prompt, the same "text, not an invisible score"
    principle _llm_reorder already applied to the old routing-score path.
    One line per worker id, keeping whichever route scores highest when
    more than one references the same worker.
    """
    best: dict[str, tuple[float, str]] = {}
    for route in routes:
        terms = ", ".join(route.need_terms[:6]) or "(no terms recorded)"
        text = f"resolved a similar past need (terms: {terms}; weight={route.weight:.1f})"
        for worker_id in route.worker_ids:
            if worker_id not in best or route.weight > best[worker_id][0]:
                best[worker_id] = (route.weight, text)
    return {worker_id: text for worker_id, (_, text) in best.items()}


def _select_evidence(
    reasoner: WorkerReasoner,
    question: str,
    ranked_evidence: list[Evidence],
    search: LocalSearchTool,
) -> list[Evidence]:
    """The final evidence cut before synthesis: zero relevance-based
    truncation -- every dedup'd piece of evidence gathered across every
    round is shown to the reasoner, not a score-ranked top-N slice. Verified
    empirically before removing the old pool cap (see git history): qibo/
    seaborn traces never exceeded the old 40-item cap at all (zero effect
    there), but sanic/yt-dlp traces did, and a sanic trace concretely lost
    two evidence items mentioning a directly-relevant `THRESHOLD` timeout
    setting to that cap before the reasoner ever got to see them -- exactly
    the failure mode this removal exists to close, and one that only gets
    more likely as a colony evolves toward more workers and wider per-round
    searches. The ranked order is kept purely as presentation order (best
    guesses first), not as a cut. Cost control instead comes from
    compressing each item's own *representation* (see
    OpenAIProvider.select_evidence's quote truncation), not from excluding
    candidates -- same principle as WorkerCard.routing_summary showing
    every worker to the Orchestrator every round.

    Per-worker collection deliberately does no filtering pass of its own
    (AutonomousWorker's evidence_limit is a generous safety cap, not a
    relevance decision), so this single pass over the full accumulated
    evidence is the one place spent on real judgment. The same call also
    flags kept items whose quote is too narrow to answer from; those get
    their full source region reopened here (Shared Evidence State:
    "evidence compression is reversible") rather than only at coalition
    cross-check time.
    """
    if not ranked_evidence:
        return []
    keep_limit = DEFAULT_SCORING_CONFIG.routing.llm_evidence_keep_limit
    pool = ranked_evidence
    keep_ids, expand_ids = reasoner.select_evidence(
        question=question, evidence=pool, limit=keep_limit
    )
    kept_indices: list[int] = []
    for raw_index in keep_ids:
        try:
            index = int(raw_index)
        except ValueError:
            continue
        if 0 <= index < len(pool):
            kept_indices.append(index)
    if not kept_indices:
        kept_indices = list(range(min(keep_limit, len(pool))))
    reopened_map = _reopen_evidence_by_index(expand_ids, pool, search) if expand_ids else {}
    return [reopened_map.get(index, pool[index]) for index in kept_indices[:keep_limit]]


def _matches_term(query_term: str, terms: set[str]) -> bool:
    for term in terms:
        if query_term == term:
            return True
        parts = _compound_parts(term)
        if query_term in parts:
            return True
        if is_stem_match(query_term, term):
            return True
        # Confirmed on a real seaborn trace: a symbol named
        # `_assign_variables_wideform` compound-splits to {"assign",
        # "variables", "wideform"} -- "wideform" stays one un-split token
        # because the source didn't put an underscore between "wide" and
        # "form". A query term "wide" then matched neither the full term
        # (stem-match checked only the whole "assign_variables_wideform",
        # which doesn't start with "wide") nor the compound-part set
        # (exact membership only, "wide" != "wideform"). Meanwhile a
        # tutorial file literally named `wide_form_violinplot.py` split
        # into {"wide", "form", "violinplot"} and got credit "wide" never
        # could. Stem-matching against each compound part too closes that
        # gap: "wide" is a >=4-char prefix of "wideform".
        if any(is_stem_match(query_term, part) for part in parts):
            return True
    return False


def _compound_parts(term: str) -> set[str]:
    if "_" not in term:
        return set()
    return {part for part in term.split("_") if len(part) > 2}


# check_need_resolution's refined_need.missing/description are free text
# the LLM is explicitly asked to write more specifically each round, with
# no length control -- confirmed on a real qibo trace: round over round
# this grew to 800+ characters of accumulated prose (a full "here's what
# we now know, here's what's still unclear" paragraph, once even a
# stringified list of sub-questions) becoming the literal search()/
# dense_search() query text. Unweighted BM25 term-overlap scoring is
# diluted, not sharpened, by hundreds of words of narrative -- the
# meaningful signal (suggested_terms, relevant_symbols, a handful of the
# actual gap) gets buried in it. Capped here, at the point this text
# becomes a search query, not upstream: the full text is still what
# plan_round's prompt shows the Orchestrator (see _node_prompt_line),
# where verbose reasoning is exactly what's wanted.
_QUERY_SNIPPET_MAX_CHARS = 200


def _query_snippet(text: str) -> str:
    text = text.strip()
    if len(text) <= _QUERY_SNIPPET_MAX_CHARS:
        return text
    truncated = text[:_QUERY_SNIPPET_MAX_CHARS]
    return truncated.rsplit(" ", 1)[0] if " " in truncated else truncated


def _need_query_text(need: UnresolvedNeed | None) -> str:
    if need is None:
        return ""
    return " ".join(
        part
        for part in [
            _query_snippet(need.missing),
            _query_snippet(need.description),
            "" if need.need_type == "unknown" else need.need_type,
            " ".join(need.relevant_symbols),
            " ".join(need.suggested_terms),
            " ".join(need.suggested_territories),
        ]
        if part
    )


def _term_set(text: str) -> set[str]:
    return {
        term.lower()
        for term in TOKEN_RE.findall(text)
        if len(term) > 2 and term.lower() not in STOP_WORDS
    }


def _relevant_symbols(text: str, need: UnresolvedNeed | None = None) -> set[str]:
    symbols = set(need.relevant_symbols if need else [])
    for token in TOKEN_RE.findall(text):
        if "_" in token or any(character.isupper() for character in token):
            symbols.add(token)
    return {symbol for symbol in symbols if len(symbol) > 2}


def _asks_for_source_implementation_text(text: str) -> bool:
    text = text.lower()
    indicators = [
        # "implement" as a stem catches implement/implements/implementing/
        # implementation/implemented -- "How does X implement Y" (no -ation
        # or -ed suffix) previously matched none of the older, longer-form
        # indicators and so never triggered the coverage-gap safety net.
        "implement",
        "code path",
        "source code",
        "code location",
        "definition",
        "defines",
        "function",
        "class",
        "module",
    ]
    return any(indicator in text for indicator in indicators)


def _rank_global_evidence(evidence: list[Evidence], question: str) -> list[Evidence]:
    terms = extract_terms(question)

    def score(item: Evidence) -> int:
        return score_evidence(
            quote=item.quote,
            path=item.path,
            reason=item.reason,
            terms=terms,
            dense_score=item.dense_score,
            symbol_name=item.symbols[0] if item.symbols else "",
        )

    return sorted(evidence, key=score, reverse=True)


def _inject_coverage_gap_nodes(
    graph: NeedGraph,
    obligations: list[AnswerObligation],
    coverage: list[ObligationCoverage],
) -> tuple[NeedGraph, list[str]]:
    """Question Coverage Contract's other half of LocalCoordinator.
    retry_from_trajectory: for every obligation check_obligation_coverage
    found NOT covered by the retry's starting evidence, adds a real, live
    leaf NeedNode for it -- root-level (no depends_on, no children, so it
    lands directly in round 0's ready frontier via compute_frontier),
    distinct from _coverage_needs elsewhere in this module, which only
    ever produces a post-hoc UnresolvedNeed for the FINAL reported list,
    never a graph node the round loop can actually act on. Without this,
    an uncovered obligation stays invisible to fast-repair's own targeting
    (unresolved_needs can read 0 even though part of the original question
    was never investigated) -- see AnswerObligation's own docstring for
    the live case this was found from.

    Idempotent against an id already present in `graph` (e.g. a second
    fast-repair pass over the same prior_state) -- never clobbers an
    existing node under the same id.
    """
    obligations_by_id = {item.obligation_id: item for item in obligations}
    nodes = dict(graph.nodes)
    injected_ids: list[str] = []
    for verdict in coverage:
        if verdict.covered:
            continue
        obligation = obligations_by_id.get(verdict.obligation_id)
        if obligation is None:
            continue
        need_id = f"coverage-{obligation.obligation_id}"
        if need_id in nodes:
            continue
        nodes[need_id] = NeedNode(
            need_id=need_id,
            need=obligation.description,
            detail=UnresolvedNeed(
                description=obligation.description,
                kind="coverage_gap",
                missing=obligation.description,
                scope="unknown",
            ),
        )
        injected_ids.append(need_id)
    if not injected_ids:
        return graph, []
    return NeedGraph(nodes=nodes), injected_ids


def _coverage_needs(
    question: str,
    evidence: list[Evidence],
    existing_needs: list[UnresolvedNeed],
    workers: list[WorkerCard],
) -> list[UnresolvedNeed]:
    needs = []
    existing_types = {need.need_type for need in existing_needs}
    question_symbols = sorted(_relevant_symbols(question))
    if (
        "subclass_lookup" not in existing_types
        and _asks_for_inheritance_text(question)
        and not _has_subclass_evidence_for(set(question_symbols), evidence)
    ):
        needs.append(
            UnresolvedNeed(
                description=(
                    "Need subclass definitions that inherit from the requested base symbol."
                ),
                kind="coverage_gap",
                need_type="subclass_lookup",
                missing="Subclass definitions and their base-class relationship.",
                scope="unknown",
                relevant_symbols=question_symbols,
                suggested_terms=[*question_symbols, "subclass", "inherit"],
                suggested_territories=[worker.territory_id for worker in workers[:5]],
            )
        )
    if (
        "implementation_location" not in existing_types
        and _asks_for_source_implementation_text(question)
        and not _has_source_definition_evidence(evidence)
    ):
        needs.append(
            UnresolvedNeed(
                description="Need source implementation definitions, not only references or tests.",
                kind="coverage_gap",
                need_type="implementation_location",
                missing="Source code definitions that implement the requested behavior.",
                scope="unknown",
                relevant_symbols=question_symbols,
                suggested_terms=[*question_symbols, "implementation", "definition"],
                suggested_territories=[
                    worker.territory_id
                    for worker in workers
                    if any(has_source_part(file) for file in worker.files)
                ][:5],
            )
        )
    if _asks_for_source_test_relationship(question):
        has_source = _has_source_definition_evidence(evidence)
        has_test = any(_is_test_evidence(item) for item in evidence)
        if not (has_source and has_test) and "source_test_coalition" not in existing_types:
            missing_side = "source implementation" if not has_source else "test coverage"
            needs.append(
                UnresolvedNeed(
                    description=(
                        "Need evidence from both source and tests before making a coverage claim."
                    ),
                    kind="coverage_gap",
                    need_type="source_test_coalition",
                    missing=missing_side,
                    scope="cross_territory",
                    relevant_symbols=question_symbols,
                    suggested_terms=[*question_symbols, "test", "implementation"],
                    suggested_territories=[worker.territory_id for worker in workers[:5]],
                )
            )
    return needs


def _asks_for_source_test_relationship(question: str) -> bool:
    lowered = question.lower()
    mentions_test = any(word in lowered for word in ["test", "tests", "tested", "coverage"])
    return mentions_test and _asks_for_source_implementation_text(question)


def _is_test_evidence(item: Evidence) -> bool:
    parts = set(item.path.replace("\\", "/").lower().split("/"))
    return bool({"test", "tests"} & parts) or Path(item.path).name.startswith("test_")


def _normalize_coverage_needs(
    question: str,
    needs: list[UnresolvedNeed],
    workers: list[WorkerCard],
) -> list[UnresolvedNeed]:
    symbols = sorted(_relevant_symbols(question))
    territories = [worker.territory_id for worker in workers[:5]]
    normalized = []
    for need in needs:
        if need.kind != "missing_evidence":
            normalized.append(need)
            continue
        need_type = need.need_type
        if need_type == "unknown":
            need_type = (
                "implementation_location"
                if _asks_for_source_implementation_text(question)
                else "negative_presence"
            )
        normalized.append(
            need.model_copy(
                update={
                    "kind": "coverage_gap",
                    "need_type": need_type,
                    "relevant_symbols": need.relevant_symbols or symbols,
                    "suggested_territories": need.suggested_territories or territories,
                }
            )
        )
    return normalized


def _absence_proofs(
    question: str,
    rounds: list[PlanningRound],
    needs: dict[str, UnresolvedNeed],
    workers: list[WorkerCard],
) -> list[AbsenceProof]:
    """`needs` is keyed by the originating NeedNode's own need_id (not a
    flat list) specifically so each produced AbsenceProof can be tagged
    with `need_id` -- see AbsenceProof's own docstring and
    ant.coordinator.repair._epistemic_state_for, which is the sole
    consumer that cares. One proof per matching need (not one combined
    proof unioning every matching need's relevant_symbols, as this
    function did before Grounded Fast Repair needed a real per-need
    owner) -- searched_worker_ids/searched_territories/searched_paths/
    tools/exhaustive reflect the whole task's execution and are
    therefore identical across every proof this call produces; only
    relevant_symbols and need_id differ per need.
    """
    negative_needs = {
        need_id: need
        for need_id, need in needs.items()
        if need.kind == "coverage_gap"
        and need.need_type in {"negative_presence", "implementation_location"}
    }
    if not negative_needs:
        return []
    searched_workers = list(
        dict.fromkeys(
            worker_id
            for round_ in rounds
            for trace in round_.node_executions
            for worker_id in trace.worker_ids
        )
    )
    workers_by_id = {worker.id: worker for worker in workers}
    searched_paths = sorted(
        {
            path
            for worker_id in searched_workers
            if worker_id in workers_by_id
            for path in workers_by_id[worker_id].files
        }
    )
    tools = sorted(
        {
            action.tool
            for round_ in rounds
            for trace in round_.node_executions
            for observation in trace.observations
            for action in observation.actions
        }
    )
    searched_territories = sorted(
        {
            workers_by_id[item].territory_id
            for item in searched_workers
            if item in workers_by_id
        }
    )
    exhaustive = bool(workers) and set(searched_workers) == set(workers_by_id)
    conclusion = "not_found" if searched_workers else "inconclusive"
    return [
        AbsenceProof(
            query=question,
            need_id=need_id,
            relevant_symbols=sorted(need.relevant_symbols),
            searched_worker_ids=searched_workers,
            searched_territories=searched_territories,
            searched_paths=searched_paths,
            tools=tools,
            exhaustive=exhaustive,
            conclusion=conclusion,
        )
        for need_id, need in negative_needs.items()
    ]


def _asks_for_inheritance_text(text: str) -> bool:
    lowered = text.lower()
    return any(
        indicator in lowered
        for indicator in [
            "subclass",
            "subclasses",
            "inherit",
            "inherits",
            "inherited",
            "base class",
            "derived",
        ]
    )


def _verify_inheritance_completeness(
    question: str,
    workers: list[WorkerCard],
    search: LocalSearchTool,
) -> tuple[list[Evidence], AbsenceProof | None]:
    """Run one definitive, repository-wide subclass lookup for an inheritance
    question, instead of relying on whichever worker recruitment happened to
    reach. `subclasses()` is an AST-index lookup, not a heuristic search, so
    scanning the union of every worker's files (not just the routed worker's
    territory) makes "these are all the subclasses" an actually-true claim,
    and the resulting AbsenceProof gives the synthesizer real grounds to
    state that instead of hedging about territories it never visited.
    """
    if not _asks_for_inheritance_text(question):
        return [], None
    symbols = sorted(_relevant_symbols(question))
    if not symbols:
        return [], None
    all_files = sorted({file for worker in workers for file in worker.files})
    if not all_files:
        return [], None

    found: list[Evidence] = []
    for symbol in symbols:
        found.extend(search.subclasses(symbol, all_files, limit=50))
    found = _dedupe_evidence(found)

    proof = AbsenceProof(
        query=question,
        relevant_symbols=symbols,
        searched_worker_ids=sorted(worker.id for worker in workers),
        searched_territories=sorted({worker.territory_id for worker in workers}),
        searched_paths=all_files,
        tools=["subclasses"],
        exhaustive=True,
        conclusion=(
            f"found_{len(found)}_subclass{'es' if len(found) != 1 else ''}"
            if found
            else "not_found"
        ),
    )
    return found, proof


def _has_subclass_evidence(text: str) -> bool:
    return bool(re.search(r"\bclass\s+[A-Za-z_][A-Za-z0-9_]*\([^)]*[A-Za-z_]", text))


def _has_subclass_evidence_for(candidates: set[str], evidence: list[Evidence]) -> bool:
    """True if evidence contains a class definition whose base matches a candidate.

    Scoped matching (base name equals one of `candidates`) is preferred over the
    generic "some class has some base" check so that evidence for an unrelated
    base class cannot be mistaken for coverage of the base the need is about.
    Falls back to the generic check only when no specific base symbol is known.
    """
    if not candidates:
        return any(_has_subclass_evidence(item.quote) for item in evidence)
    lowered = {candidate.lower() for candidate in candidates}
    for item in evidence:
        for bases in BASE_CLASS_RE.findall(item.quote):
            for base in bases.split(","):
                base_name = base.strip().split(".")[-1]
                if base_name.lower() in lowered:
                    return True
    return False


def _has_source_definition_evidence(evidence: list[Evidence]) -> bool:
    return any(
        has_source_part(item.path)
        and not has_low_value_part(item.path)
        and ("class " in item.quote or "def " in item.quote)
        for item in evidence
    )


def _has_source_definition_evidence_for(candidates: set[str], evidence: list[Evidence]) -> bool:
    """Scoped variant of `_has_source_definition_evidence`: the definition must
    actually define one of `candidates`, not merely sit near source-looking code."""
    if not candidates:
        return _has_source_definition_evidence(evidence)
    for item in evidence:
        if not has_source_part(item.path) or has_low_value_part(item.path):
            continue
        quote = item.quote
        if "class " not in quote and "def " not in quote:
            continue
        if any(_symbol_defined_in_quote(candidate, quote) for candidate in candidates):
            return True
    return False


def _symbol_defined_in_quote(symbol: str, quote: str) -> bool:
    return bool(re.search(rf"\b(?:class|def)\s+{re.escape(symbol)}\b", quote))


def _need_symbol_candidates(need: UnresolvedNeed, question: str) -> set[str]:
    return _relevant_symbols(f"{_need_query_text(need)} {question}", need)


def _is_need_satisfied(need: UnresolvedNeed, evidence: list[Evidence], question: str) -> bool:
    if need.need_type not in _CLOSABLE_BY_EVIDENCE:
        return False
    candidates = _need_symbol_candidates(need, question)
    if need.need_type == "subclass_lookup":
        return _has_subclass_evidence_for(candidates, evidence)
    if need.need_type == "implementation_location":
        return _has_source_definition_evidence_for(candidates, evidence)
    if need.need_type == "source_test_coalition":
        return _has_source_definition_evidence(evidence) and any(
            _is_test_evidence(item) for item in evidence
        )
    return False


def _close_resolved_needs(
    needs: list[UnresolvedNeed],
    evidence: list[Evidence],
    question: str,
) -> list[UnresolvedNeed]:
    """Drop needs whose target evidence is now present in the cumulative evidence
    state. Re-run after every round so a need raised while evidence was sparse
    does not linger once a later round (possibly from a different worker)
    actually grounds it."""
    return [need for need in needs if not _is_need_satisfied(need, evidence, question)]


def _need_identity(need: UnresolvedNeed) -> tuple[str, str]:
    text = (need.missing or need.description or "").strip().lower()
    return (need.need_type, text)


def _merge_needs(
    existing: list[UnresolvedNeed],
    new_needs: list[UnresolvedNeed],
) -> list[UnresolvedNeed]:
    """Accumulate needs across rounds instead of overwriting. A need that keeps
    getting re-raised for the same reason is merged (union of symbols/terms)
    rather than duplicated, so a stale reasoner that never self-closes a need
    does not flood the final list with repeats of the same gap."""
    merged = list(existing)
    index_by_key = {_need_identity(need): index for index, need in enumerate(merged)}
    for need in new_needs:
        key = _need_identity(need)
        if key in index_by_key:
            merged[index_by_key[key]] = _merge_need_pair(merged[index_by_key[key]], need)
        else:
            index_by_key[key] = len(merged)
            merged.append(need)
    return merged


def _merge_need_pair(old: UnresolvedNeed, new: UnresolvedNeed) -> UnresolvedNeed:
    return new.model_copy(
        update={
            "relevant_symbols": sorted(set(old.relevant_symbols) | set(new.relevant_symbols)),
            "suggested_terms": sorted(set(old.suggested_terms) | set(new.suggested_terms)),
            "suggested_territories": sorted(
                set(old.suggested_territories) | set(new.suggested_territories)
            ),
            "evidence_ids": sorted(set(old.evidence_ids) | set(new.evidence_ids)),
        }
    )


def _collect_proposals(
    orchestrator_new_nodes: dict[str, NeedNode],
    observed: list[UnresolvedNeed],
) -> list[ProposedNode]:
    """Unifies this round's Potential Needs Buffer: the Orchestrator's own
    new-id graph_updates entries (proposal_id = its own suggested need_id,
    kept as-is -- see GraphConsolidationDecision's "create" semantics) plus
    the coordinator's worker-observed-needs buffer (proposal_id minted
    here, since an UnresolvedNeed carries no id of its own). Neither
    becomes a real node until WorkerReasoner.consolidate_graph decides.
    """
    proposals = [
        ProposedNode(
            proposal_id=need_id,
            need=node.need,
            detail=node.detail,
            proposed_depends_on=list(node.depends_on),
            proposed_children=list(node.children),
            proposed_related_to=list(node.related_to),
            source="orchestrator",
        )
        for need_id, node in orchestrator_new_nodes.items()
    ]
    proposals.extend(
        ProposedNode(
            proposal_id=f"obs-{index}",
            need=need.missing or need.description,
            detail=need,
            source="worker_observed",
        )
        for index, need in enumerate(observed)
    )
    return proposals


@dataclass
class ProposalCluster:
    """One group of this round's own proposals judged (deterministically,
    never by an LLM) to be the same underlying gap -- see
    _cluster_pending_proposals. `representative` is the only member the
    graph-admission LLM call (consolidate_graph) actually sees;
    `member_ids` is lossless (every original proposal_id, representative's
    own included first) so _expand_cluster_decisions can fan the
    representative's verdict back out to the rest afterward.
    """

    representative: ProposedNode
    member_ids: list[str]


def _normalize_proposal_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _cluster_pending_proposals(proposals: list[ProposedNode]) -> list[ProposalCluster]:
    """Intra-batch consolidation: groups THIS ROUND's own proposals that
    are near-duplicates of EACH OTHER, before any of them ever reaches the
    LLM-judged graph-admission step (WorkerReasoner.consolidate_graph).
    Confirmed live on a real yt-dlp fast-repair retry: several workers
    assigned to the same stuck subgraph each independently observe "the
    same gap," worded slightly differently, and Need Graph Consolidation's
    own dedup judgment does not reliably catch all of them once a single
    round's proposal batch grows into the hundreds -- consolidate_graph's
    prompt/response both scale with batch size, and its max_output_tokens
    is fixed, so a large-enough batch's response silently truncates,
    leaving most proposals with no decision at all. Clustering here first
    shrinks what the LLM ever has to judge (hundreds of raw observations
    down to a few dozen distinct gaps), which is a structural fix for that
    failure mode, not just a cheaper one.

    Deliberately NOT an LLM call -- two free, deterministic passes:
    1. Exact/normalized-text dedup (case/whitespace-insensitive) -- free,
       unambiguous, always correct.
    2. Embedding-similarity clustering (connected components over a cosine
       similarity graph at _PENDING_CLUSTER_SIMILARITY_THRESHOLD) among
       what's left, when a shared embedder is available.
    A no-op (one singleton cluster per proposal, in original order) when
    there is no shared embedder or fewer than two distinct texts --
    degrades gracefully to today's one-proposal-per-cluster behavior,
    matching dense_search's own "never block on a missing embedder"
    posture.
    """
    if not proposals:
        return []
    groups_by_text: dict[str, list[ProposedNode]] = {}
    order: list[str] = []
    for proposal in proposals:
        key = _normalize_proposal_text(proposal.need)
        if key not in groups_by_text:
            groups_by_text[key] = []
            order.append(key)
        groups_by_text[key].append(proposal)
    normalized_groups = [groups_by_text[key] for key in order]

    if len(normalized_groups) <= 1:
        return [
            ProposalCluster(
                representative=group[0], member_ids=[p.proposal_id for p in group]
            )
            for group in normalized_groups
        ]

    embedder = get_shared_embedder()
    if embedder is None:
        return [
            ProposalCluster(
                representative=group[0], member_ids=[p.proposal_id for p in group]
            )
            for group in normalized_groups
        ]
    vectors = embedder.embed([group[0].need for group in normalized_groups])
    if not vectors:
        return [
            ProposalCluster(
                representative=group[0], member_ids=[p.proposal_id for p in group]
            )
            for group in normalized_groups
        ]

    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = array / norms
    similarity = normed @ normed.T

    n = len(normalized_groups)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    for i in range(n):
        for j in range(i + 1, n):
            if similarity[i, j] >= _PENDING_CLUSTER_SIMILARITY_THRESHOLD:
                union(i, j)

    members_by_root: dict[int, list[ProposedNode]] = {}
    for i, group in enumerate(normalized_groups):
        members_by_root.setdefault(find(i), []).extend(group)

    return [
        ProposalCluster(representative=members[0], member_ids=[p.proposal_id for p in members])
        for members in members_by_root.values()
    ]


def _expand_cluster_decisions(
    clusters: list[ProposalCluster],
    plan: GraphConsolidationPlan,
    graph: NeedGraph,
) -> GraphConsolidationPlan:
    """Fans WorkerReasoner.consolidate_graph's per-representative verdict
    back out to every proposal_id it stood in for -- a cluster member
    never itself reached the LLM, so it inherits its representative's
    verdict verbatim, with one adjustment: a representative that mints a
    brand-new node (create/attach/relate, or a merge/subsume whose
    target_node_id turns out not to be a real node -- see
    _apply_consolidation_decisions' own "falls back to create" rule, which
    this mirrors so a member's synthesized target always matches what the
    representative will ACTUALLY resolve to) means its members are now
    duplicates of THAT new node specifically, not of whatever the
    representative's raw decision literally named -- so members merge
    into the representative's own new id. A representative with no
    decision at all (the graph-admission response never mentioned it --
    still possible for a very large cluster count, just far less likely
    now that clustering already shrank the batch) leaves every member
    undecided too, deferring to _apply_consolidation_decisions' own
    never-auto-create default for an unjudged proposal.
    """
    decisions_by_id = {decision.proposal_id: decision for decision in plan.decisions}
    expanded: list[GraphConsolidationDecision] = []
    for cluster in clusters:
        representative_id = cluster.representative.proposal_id
        decision = decisions_by_id.get(representative_id)
        if decision is not None:
            expanded.append(decision)
        if decision is None:
            continue
        mints_new_node = decision.action in ("create", "attach", "relate") or (
            decision.action in ("merge", "subsume") and decision.target_node_id not in graph.nodes
        )
        for member_id in cluster.member_ids:
            if member_id == representative_id:
                continue
            if mints_new_node:
                expanded.append(
                    GraphConsolidationDecision(
                        proposal_id=member_id,
                        action="merge",
                        target_node_id=representative_id,
                        rationale=f"intra-batch duplicate of {representative_id}",
                    )
                )
            else:
                expanded.append(
                    GraphConsolidationDecision(
                        proposal_id=member_id,
                        action=decision.action,
                        target_node_id=decision.target_node_id,
                        rationale=f"intra-batch duplicate, same cluster as {representative_id}",
                    )
                )
    return GraphConsolidationPlan(decisions=expanded)


def _candidate_hints_for_proposals(
    active_nodes: dict[str, NeedNode],
    proposals: list[ProposedNode],
) -> dict[str, list[str]]:
    """Retrieval-only step for Need Graph Consolidation: dense-embedding
    nearest neighbors of each proposal's need text among the currently
    active (unresolved, not abandoned) nodes -- never itself the merge
    decision, exactly the same "narrows candidates, does not decide"
    posture as ant.coordinator.worker_retrieval.rank_workers. A no-op
    (empty dict) when there's no shared embedder available (same
    degrade-gracefully behavior as dense_search) or nothing to compare
    against -- consolidate_graph still runs, just without hints for that
    round, it never blocks on this.
    """
    if not proposals or not active_nodes:
        return {}
    embedder = get_shared_embedder()
    if embedder is None:
        return {}
    node_ids = list(active_nodes)
    node_vectors = embedder.embed([active_nodes[node_id].need for node_id in node_ids])
    if not node_vectors:
        return {}
    entries = [
        EmbeddingEntry(path=node_id, line_start=0, line_end=0, quote=active_nodes[node_id].need)
        for node_id in node_ids
    ]
    array = np.asarray(node_vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    index = EmbeddingIndex(entries=entries, vectors=array / norms)

    proposal_vectors = embedder.embed([proposal.need for proposal in proposals])
    if not proposal_vectors:
        return {}
    hints: dict[str, list[str]] = {}
    for proposal, vector in zip(proposals, proposal_vectors, strict=True):
        results = index.search(vector, limit=_CONSOLIDATION_CANDIDATE_LIMIT)
        hints[proposal.proposal_id] = [entry.path for _score, entry in results]
    return hints


def _apply_consolidation_decisions(
    graph: NeedGraph,
    proposals: list[ProposedNode],
    plan: GraphConsolidationPlan,
) -> NeedGraph:
    """Commits WorkerReasoner.consolidate_graph's verdicts for real.
    Two passes, not one, so a proposal's depends_on/children/related_to
    can reference *another* proposal from this same batch regardless of
    which one is processed first:

    Pass 1 -- decide each proposal's own committed identity (mint a
    permanent node for create/attach/relate, or resolve to an existing
    target for merge/subsume, applying that enrichment immediately since
    it only ever touches a pre-existing node) and record it in `remap`.
    A proposal with no decision at all defaults to "drop", never "create"
    -- a proposal the graph-admission LLM call never actually reviewed
    (a truncated/malformed response omitting it, most commonly) must
    never silently become a real node either; it simply isn't admitted
    this round; a worker-observed need that's still a real gap gets
    re-observed and re-proposed in a later round regardless. A merge/
    subsume that WAS explicitly decided but whose target_node_id isn't a
    real node still falls back to "create" -- that is a real, reviewed
    verdict with a malformed target, not an unreviewed proposal, and
    silently losing it would be the actual data loss this whole function
    exists to avoid.

    Pass 2 -- now that `remap` is complete, resolve every proposal_id
    reference through it, in EVERY node's children/depends_on/related_to
    -- not just newly-minted proposals' own edges, but also any
    pre-existing node whose own list was set by the Orchestrator's direct
    existing-node edit (_merge_plan_into_graph, applied before this
    function runs) to name a *new* id it is simultaneously proposing this
    same round via graph_updates. Confirmed live: a root node's children
    edit named a proposal that this round's Organizer decided to merge
    away, leaving a dangling child id and crashing the closure check with
    a KeyError once this ran unpatched -- a reference to anything that
    isn't a committed node (dropped, merged/subsumed with no live remap
    entry, or simply never real) is dropped from the list rather than
    left dangling. Attach/relate target wiring (parent gains a child,
    a minted node gains a related_to edge) is handled separately after,
    since those are edges arriving AT a minted node from its own
    decision, not outgoing references to resolve.
    """
    decisions_by_id = {decision.proposal_id: decision for decision in plan.decisions}
    nodes = dict(graph.nodes)
    remap: dict[str, str] = {}
    minted: list[tuple[ProposedNode, GraphConsolidationDecision | None]] = []

    for proposal in proposals:
        decision = decisions_by_id.get(proposal.proposal_id)
        action = decision.action if decision else "drop"
        target = decision.target_node_id if decision else ""

        if action == "drop":
            continue
        if action in ("merge", "subsume") and target in nodes:
            existing = nodes[target]
            if action == "merge":
                nodes[target] = existing.model_copy(
                    update={"detail": _merge_need_pair(proposal.detail, existing.detail)}
                )
            else:  # subsume
                nodes[target] = existing.model_copy(
                    update={"need": proposal.need, "detail": proposal.detail}
                )
            remap[proposal.proposal_id] = target
            continue

        # create / attach / relate, or a merge/subsume with no real target
        # (falls back to create rather than silently losing the proposal).
        nodes[proposal.proposal_id] = NeedNode(
            need_id=proposal.proposal_id,
            need=proposal.need,
            detail=proposal.detail,
            depends_on=list(proposal.proposed_depends_on),
            children=list(proposal.proposed_children),
            related_to=list(proposal.proposed_related_to),
        )
        remap[proposal.proposal_id] = proposal.proposal_id
        minted.append((proposal, decision))

    proposal_ids = {proposal.proposal_id for proposal in proposals}

    def resolve_ref(ref: str) -> str | None:
        if ref in remap:
            return remap[ref]
        if ref in proposal_ids:
            return None  # dropped, or a merge/subsume with no real target
        if ref in nodes:
            return ref
        return None  # a stale/unknown id -- discard rather than leave dangling

    def _resolved(values: list[str]) -> list[str]:
        result: list[str] = []
        for ref in values:
            resolved = resolve_ref(ref)
            if resolved and resolved not in result:
                result.append(resolved)
        return result

    for node_id, node in list(nodes.items()):
        new_children = _resolved(node.children)
        new_depends_on = _resolved(node.depends_on)
        new_related_to = _resolved(node.related_to)
        if (
            new_children != node.children
            or new_depends_on != node.depends_on
            or new_related_to != node.related_to
        ):
            nodes[node_id] = node.model_copy(
                update={
                    "children": new_children,
                    "depends_on": new_depends_on,
                    "related_to": new_related_to,
                }
            )

    for proposal, decision in minted:
        node_id = remap[proposal.proposal_id]
        # decision is never None here -- an undecided proposal now drops
        # before ever being minted (see pass 1's "drop" default above), so
        # only a proposal with a real create/attach/relate/fallback
        # decision ever reaches this loop.
        action = decision.action if decision else "create"

        parent_id = None
        if action == "attach" and decision and decision.target_node_id:
            parent_id = resolve_ref(decision.target_node_id)
        elif proposal.proposed_parent:
            parent_id = resolve_ref(proposal.proposed_parent)
        if action == "relate" and decision and decision.target_node_id:
            related_target = resolve_ref(decision.target_node_id)
            if related_target and related_target in nodes:
                current = nodes[node_id]
                if related_target not in current.related_to:
                    nodes[node_id] = current.model_copy(
                        update={"related_to": [*current.related_to, related_target]}
                    )

        if parent_id and parent_id in nodes and node_id not in nodes[parent_id].children:
            parent = nodes[parent_id]
            nodes[parent_id] = parent.model_copy(update={"children": [*parent.children, node_id]})

    return graph.model_copy(update={"nodes": nodes})


def _resolution_check_need(
    detail: UnresolvedNeed,
    need_id: str,
    recovery: RecoveryState,
    enforce_alignment: bool,
) -> UnresolvedNeed:
    """Substitutes a leaf's frozen intent_anchor (see
    RecoveryState.intent_anchors' own docstring) for its own, possibly
    since-narrowed, description -- ONLY for what a call site uses to
    JUDGE new evidence (check_need_resolution, verify_evidence_upgrade),
    never for what drives search/routing (worker assignment, query
    construction), which must keep seeing the node's real, freely-
    narrowing current text so investigation itself is not blocked from
    getting more specific. A no-op (returns `detail` unchanged) outside
    fast-repair mode, or for a need_id with no anchor seeded yet (a node
    not yet committed to the graph when this is called, or a call made
    before this mechanism existed on an older RecoveryState) -- the
    caller then falls back to today's exact behavior.
    """
    if not enforce_alignment:
        return detail
    anchor = recovery.intent_anchors.get(need_id)
    if not anchor:
        return detail
    return detail.model_copy(update={"description": anchor})


def _prune_dangling_edges(graph: NeedGraph) -> NeedGraph:
    """Safety net run every round right after _consolidate_and_commit:
    drops any children/depends_on/related_to reference that doesn't
    resolve to a real node in the graph. _apply_consolidation_decisions
    already resolves same-round proposal_id references through its own
    remap, but that only accounts for reference MISMATCHES it can
    recognize as such -- confirmed live on a real yt-dlp run that it is
    not exhaustive: the Orchestrator emits graph_updates' new-node keys
    and an existing node's children list as two independently-generated
    JSON fields in the same plan_round response, and nothing enforces
    they name the exact same string (e.g. a hyphen/underscore mismatch).
    That produced a child id with no matching proposal_id and no matching
    existing node either, which crashed the closure check with KeyError
    once _apply_consolidation_decisions had already run and found nothing
    to fix. This function is the general invariant enforcement instead of
    chasing each individual mismatch source: after every round, every
    node's children/depends_on/related_to must name only ids that are
    actually keys in graph.nodes, full stop.
    """
    nodes = graph.nodes
    updates: dict[str, NeedNode] = {}
    for need_id, node in nodes.items():
        children = [ref for ref in node.children if ref in nodes]
        depends_on = [ref for ref in node.depends_on if ref in nodes]
        related_to = [ref for ref in node.related_to if ref in nodes]
        changed = (
            children != node.children
            or depends_on != node.depends_on
            or related_to != node.related_to
        )
        if changed:
            updates[need_id] = node.model_copy(
                update={"children": children, "depends_on": depends_on, "related_to": related_to}
            )
    if not updates:
        return graph
    return graph.model_copy(update={"nodes": {**nodes, **updates}})


def _evidence_key(item: Evidence) -> tuple[str, int, int, str]:
    return (item.path, item.line_start, item.line_end, item.quote)


def _reopened_need_ids(
    rounds: list[PlanningRound],
    coverage_gap_node_ids: list[str],
    targeted_need_ids: set[str],
) -> set[str]:
    """Which need_ids this fast-repair retry actually reopened -- pure,
    deterministic, no new LLM call. A need counts as reopened if it
    received real worker execution this retry (search activity actually
    happened against it), if it's a newly-injected Question Coverage gap
    node (reopened by construction -- see _inject_coverage_gap_nodes), or
    if it was explicitly targeted by the repair plan (seed.
    targeted_need_ids). Everything else is untouched this retry, and its
    evidence must be preserved -- see the final-synthesis block in ask()
    for how this feeds the per-claim evidence retention gate.
    """
    executed = {
        trace.need_id for round_state in rounds for trace in round_state.node_executions
    }
    return executed | set(coverage_gap_node_ids) | targeted_need_ids


def _dedupe_evidence(evidence: list[Evidence]) -> list[Evidence]:
    # See the matching note on autonomous._dedupe: keep the first-seen copy
    # of a (path, lines, quote) duplicate, but merge a later duplicate's
    # dense_score forward rather than silently dropping it, so a chunk two
    # different workers/rounds both surfaced -- one lexically, one via
    # dense_search -- doesn't lose the dense signal to whichever copy
    # happened to arrive first. need_ids is unioned the same way -- the
    # same code region can legitimately be gathered for more than one
    # need (once per round it's independently surfaced), and fast-repair's
    # per-claim evidence retention (see local.py's final-synthesis block)
    # depends on this union being complete: keeping only the first
    # duplicate's need_ids would let a later need's real association with
    # this exact chunk silently disappear.
    index_by_key: dict[tuple[str, int, int, str], int] = {}
    deduped: list[Evidence] = []
    for item in evidence:
        key = _evidence_key(item)
        if key in index_by_key:
            existing = deduped[index_by_key[key]]
            if item.dense_score > existing.dense_score or item.need_ids:
                deduped[index_by_key[key]] = existing.model_copy(
                    update={
                        "dense_score": max(item.dense_score, existing.dense_score),
                        "need_ids": sorted(set(existing.need_ids) | set(item.need_ids)),
                    }
                )
            continue
        index_by_key[key] = len(deduped)
        deduped.append(item)
    return deduped


def _cumulative_need_evidence(evidence: list[Evidence], need_id: str) -> list[Evidence]:
    """Every evidence item linked to `need_id` gathered so far this task,
    deduped -- what check_need_resolution should judge a need's status
    against (see that method's own docstring for the bug this fixes).
    Provenance comes from Evidence.need_ids, stamped once at collection
    time (_run_selected_workers) and never rewritten later, so this
    correctly accumulates across every round that has touched this
    need_id so far, not just the round currently executing -- confirmed
    live via controlled replay on a real qibo trace
    (b93c31147ace128d/root): evidence gathered across round 0 and round 1
    was jointly sufficient to resolve the need, but round 1's own new
    evidence alone was not, and the need stayed unresolved for 5 wasted
    rounds because the old call only ever saw that round's own new items.

    Deliberately NOT the whole task's global `evidence` pool -- a sibling
    need's own findings must never count toward THIS need's resolution,
    which is what keeps this a correctness fix rather than a loosening of
    the check. `evidence` here is the shared list `ask()`'s round loop
    already mutates in place, so it already contains this round's own
    finds by the time this is called.
    """
    return _dedupe_evidence([item for item in evidence if need_id in item.need_ids])


def _last_coalition_workers(rounds: list[PlanningRound]) -> list[str]:
    """Every distinct worker id used up through and including the last
    node execution (in chronological round/execution order) that formed a
    coalition -- same semantics as before, adapted to a round now
    potentially fanning out to several node executions instead of always
    exactly one.
    """
    all_worker_ids = [
        worker_id
        for round_ in rounds
        for trace in round_.node_executions
        for worker_id in trace.worker_ids
    ]
    last_coalition_index = None
    running_index = 0
    for round_ in rounds:
        for trace in round_.node_executions:
            if trace.coalition_formed:
                last_coalition_index = running_index + len(trace.worker_ids) - 1
            running_index += len(trace.worker_ids)
    if last_coalition_index is None:
        return []
    return list(dict.fromkeys(all_worker_ids[: last_coalition_index + 1]))


def _add_coalition_cross_checks(observations, prior_evidence: list[Evidence]) -> None:
    for observation in observations:
        peer_claims = sorted(
            {
                evidence.claim
                for evidence in prior_evidence
                if evidence.worker_id != observation.worker_id and evidence.claim
            }
        )
        peer_paths = sorted(
            {
                evidence.path
                for evidence in prior_evidence
                if evidence.worker_id != observation.worker_id
            }
        )
        query = ", ".join(peer_claims[:6]) if peer_claims else ", ".join(peer_paths[:6])
        observation.actions.append(
            WorkerAction(
                tool="cross_check",
                query=query,
                result_count=len(peer_claims) or len(peer_paths),
                rationale="One-pass coalition cross-check against peer evidence claims.",
            )
        )


def _run_coalition_cross_check(
    *,
    reasoner: WorkerReasoner,
    question: str,
    selected: list[WorkerCard],
    evidence: list[Evidence],
    search: LocalSearchTool,
) -> WorkerObservation | None:
    """Coalitions exist to reason jointly across territories, not just to log
    that a cross-check happened. Run one real reasoning pass over the pooled
    coalition evidence so the reasoner can catch a conflict or an
    under-specified claim between peers' evidence that no single worker would
    see on its own; if it flags a gap tied to a specific piece of evidence,
    reopen that evidence's full source region (Evidence Compression Is
    Reversible) instead of only trusting the compressed quote already held.
    """
    joint_evidence = _dedupe_evidence(evidence)
    coalition_worker_id = "coalition:" + "-".join(sorted(worker.id for worker in selected))
    joint_observation = reasoner.observe(
        question=question,
        worker_id=coalition_worker_id,
        territory_id="coalition",
        evidence=joint_evidence,
    )
    if not joint_observation.unresolved_needs:
        return None
    reopened = _reopen_referenced_evidence(
        joint_observation.unresolved_needs, joint_evidence, search
    )
    return WorkerObservation(
        worker_id=coalition_worker_id,
        territory_id="coalition",
        evidence=reopened,
        unresolved_needs=joint_observation.unresolved_needs,
        stop_reason="coalition_cross_check",
    )


def _reopen_referenced_evidence(
    needs: list[UnresolvedNeed],
    evidence_pool: list[Evidence],
    search: LocalSearchTool,
    context_lines: int = 30,
) -> list[Evidence]:
    """Preserves `source.need_ids` on the reopened region -- a fresh
    `search.read_region()` object defaults to none of its own, and
    without copying the source's forward, a chunk that already carried
    real per-claim provenance (see Evidence.need_ids) would silently lose
    it the moment coalition cross-check widens its context, later
    misreading it as never-tagged in ask()'s per-claim evidence
    retention partition.
    """
    reopened: list[Evidence] = []
    seen: set[tuple[str, int]] = set()
    for need in needs:
        for raw_index in need.evidence_ids:
            try:
                index = int(raw_index)
            except ValueError:
                continue
            if not (0 <= index < len(evidence_pool)):
                continue
            source = evidence_pool[index]
            key = (source.path, source.line_start)
            if key in seen:
                continue
            seen.add(key)
            try:
                region = search.read_region(source.path, source.line_start, context_lines)
            except OSError:
                continue
            reopened.append(
                region.model_copy(
                    update={
                        "worker_id": source.worker_id,
                        "reason": f"Reopened for coalition cross-check: {source.reason}".strip(),
                        "need_ids": list(source.need_ids),
                    }
                )
            )
    return reopened


def _reopen_evidence_by_index(
    indices: list[str],
    pool: list[Evidence],
    search: LocalSearchTool,
    context_lines: int = 30,
) -> dict[int, Evidence]:
    """Maps pool index -> reopened Evidence (larger source region), for
    _select_evidence's own expand requests. Kept separate from
    _reopen_referenced_evidence above (which dedupes by source location
    across several needs' evidence_ids) since this one resolves a single
    selection call's own flagged indices. Also preserves `source.need_ids`
    on the reopened region for the same reason as that function's own
    docstring: an item can only reach this expand step after already
    surviving ask()'s per-claim retention partition, and losing its
    need_ids here would erase that already-settled provenance right
    before the item lands in the final answer.
    """
    reopened: dict[int, Evidence] = {}
    for raw_index in indices:
        try:
            index = int(raw_index)
        except ValueError:
            continue
        if not (0 <= index < len(pool)):
            continue
        source = pool[index]
        try:
            region = search.read_region(source.path, source.line_start, context_lines)
        except OSError:
            continue
        reopened[index] = region.model_copy(
            update={
                "worker_id": source.worker_id,
                "reason": f"Reopened for deeper context: {source.reason}".strip(),
                "need_ids": list(source.need_ids),
            }
        )
    return reopened
