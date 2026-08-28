"""Task-conditioned ("fast") evolution: repair a single finished task's own
Need Graph from its own trajectory and retry it. Deliberately separate
from the rest of coordinator/local.py -- this is the fast/ephemeral
adaptation timescale, never the slow/persistent colony one (see
ant.evolution for that). Nothing in this module writes to IndexStore,
ColonyMemoryStore, or GlobalMemoryStore; it only ever produces the
starting materials for one more LocalCoordinator.ask() call. Nothing here
reads a reference answer or judge score -- only what a finished
EvidenceState already recorded about its own attempt.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ant.domain import (
    Evidence,
    EvidenceState,
    NeedGraph,
    RepairPlan,
    StuckNodeSummary,
    TaskTrajectoryPackage,
)


def assemble_trajectory_package(state: EvidenceState) -> TaskTrajectoryPackage:
    """Condenses a finished EvidenceState into the compact, purpose-built
    summary a FastEvolutionReasoner actually reasons over -- not the raw
    trace. Every field here is derived from data this session's trajectory
    instrumentation already captures (final_need_graph, final_recovery_state,
    rounds); no new data capture is needed.
    """
    stuck_episode_by_member: dict[str, str] = {
        member: episode.episode_id
        for episode in state.final_recovery_state.stuck_episodes
        for member in episode.members
    }
    stuck_nodes: list[StuckNodeSummary] = []
    for need_id, node in state.final_need_graph.items():
        if node.resolution == "resolved":
            continue
        executions = [
            trace
            for round_ in state.rounds
            for trace in round_.node_executions
            if trace.need_id == need_id
        ]
        tried_special_tactics = sorted(
            {trace.special_tactic for trace in executions if trace.special_tactic}
        )
        no_progress_execution_count = sum(
            1 for trace in executions if trace.evidence_gain == 0 and trace.need_reduction == 0
        )
        evidence_claims: list[str] = []
        seen_claims: set[str] = set()
        for trace in executions:
            for observation in trace.observations:
                for item in observation.evidence:
                    claim = item.claim or item.reason
                    if claim and claim not in seen_claims:
                        seen_claims.add(claim)
                        evidence_claims.append(claim)
        stuck_nodes.append(
            StuckNodeSummary(
                need_id=need_id,
                need=node.need,
                resolution=node.resolution,
                depends_on=list(node.depends_on),
                children=list(node.children),
                missing=node.detail.missing,
                suggested_terms=list(node.detail.suggested_terms),
                suggested_territories=list(node.detail.suggested_territories),
                tried_worker_ids=list(
                    state.final_recovery_state.tried_workers_by_node.get(need_id, [])
                ),
                tried_special_tactics=tried_special_tactics,
                no_progress_execution_count=no_progress_execution_count,
                evidence_claims=evidence_claims,
                is_abandoned=need_id in state.final_recovery_state.abandoned_node_ids,
                stuck_episode_id=stuck_episode_by_member.get(need_id, ""),
            )
        )
    return TaskTrajectoryPackage(
        question=state.question,
        prior_answer=state.answer,
        stuck_nodes=stuck_nodes,
        graph_decomposition_log=[round_.graph_delta for round_ in state.rounds],
    )


@dataclass
class RepairSeed:
    """What a RepairPlan resolves into for LocalCoordinator.ask()'s new
    initial_graph/initial_recovery/repair_guidance parameters, plus which
    need_ids a repair plan actually targeted -- meant to be applied to a
    fresh copy of the prior graph/RecoveryState by the caller
    (LocalCoordinator.retry_from_trajectory), not mutated here, so this
    stays a pure transformation from (prior state, repair plan) to
    instructions.
    """

    dependency_changes: dict[str, list[str]] = field(default_factory=dict)
    redecompose_node_ids: set[str] = field(default_factory=set)
    guidance_lines: list[str] = field(default_factory=list)
    # Every need_id any action targeted, regardless of kind -- a repair
    # plan proposing *anything* for a need_id (not just redecompose) means
    # "give this another shot", so retry_from_trajectory un-abandons all of
    # these, not just the ones with a structural redecompose action.
    targeted_need_ids: set[str] = field(default_factory=set)


def resolve_repair_plan(plan: RepairPlan) -> RepairSeed:
    """Splits a RepairPlan's actions into the two treatments described in
    the fast-evolution design: change_dependency/redecompose are
    unambiguous structural edits, applied mechanically; everything else
    (reuse_assignment, replace_assignment, merge_needs, form_local_bridge,
    force_global_search) becomes advisory text for the Orchestrator to
    weigh each round, not a forced round-0 assignment -- a suggestion that
    doesn't pan out on retry either should be overridable.
    """
    seed = RepairSeed()
    for action in plan.actions:
        seed.targeted_need_ids.add(action.need_id)
        if action.kind == "change_dependency" and action.new_depends_on is not None:
            seed.dependency_changes[action.need_id] = list(action.new_depends_on)
            continue
        if action.kind == "redecompose":
            seed.redecompose_node_ids.add(action.need_id)
            continue
        line = f"- {action.kind} on {action.need_id!r}"
        if action.worker_ids:
            line += f" (workers: {', '.join(action.worker_ids)})"
        if action.merge_with:
            line += f" (merge with: {', '.join(action.merge_with)})"
        if action.rationale:
            line += f": {action.rationale}"
        seed.guidance_lines.append(line)
    return seed


def render_repair_guidance(package: TaskTrajectoryPackage, seed: RepairSeed) -> str:
    """Human-readable guidance text threaded into every plan_round() call
    of the retry -- advisory only, never mechanically enforced (see
    RepairSeed's own docstring)."""
    if not seed.guidance_lines:
        return ""
    header = (
        "A prior attempt on this exact question got stuck; a repair analysis of "
        "that attempt's own trajectory suggests:\n"
    )
    return header + "\n".join(seed.guidance_lines)


def build_retry_starting_state(
    prior_state: EvidenceState, seed: RepairSeed
) -> tuple[NeedGraph, list[Evidence]]:
    """Builds the retry's starting graph (prior_state.final_need_graph,
    carrying forward resolved nodes too so dependents' chains stay
    consistent, with change_dependency seed edits applied) and starting
    evidence pool (the prior attempt's full evidence, so the retry only
    has to make incremental progress on what was stuck, not re-discover
    everything). RecoveryState seeding (tried_workers_by_node carried
    forward, abandoned nodes cleared for targeted needs) is the caller's
    job (LocalCoordinator.retry_from_trajectory) since RecoveryState is
    coordinator-local, not a domain type this module depends on.

    Every targeted need_id (not just redecompose ones) gets its
    progress/rounds_without_progress reset to fresh, not only
    redecompose_node_ids: a stale progress="stuck" carried over from the
    prior attempt would keep the node out of the retry's own ready
    frontier (compute_frontier routes a still-"stuck" node to
    stuck_subgraphs, not ready) regardless of which worker a
    reuse_assignment/replace_assignment action suggests -- any action
    targeting a need_id means "give this another shot", the same
    reasoning already applied to abandoned_node_ids in
    retry_from_trajectory.
    """
    nodes = dict(prior_state.final_need_graph)
    for need_id, new_depends_on in seed.dependency_changes.items():
        if need_id in nodes:
            nodes[need_id] = nodes[need_id].model_copy(update={"depends_on": new_depends_on})
    for need_id in seed.targeted_need_ids:
        if need_id in nodes:
            nodes[need_id] = nodes[need_id].model_copy(
                update={"progress": "not_stuck", "rounds_without_progress": 0}
            )
    return NeedGraph(nodes=nodes), list(prior_state.evidence)
