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
    AbsenceProof,
    Evidence,
    EvidenceState,
    NeedAlignmentPlan,
    NeedGraph,
    NeedNode,
    RepairPlan,
    StuckNodeSummary,
    TaskTrajectoryPackage,
)

# The canonical root need_id, hardcoded at LocalCoordinator.ask()'s initial
# graph construction (NeedGraph(nodes={"root": ...})) -- never anything
# else. apply_alignment_verdicts uses this to defensively refuse a "drop"
# verdict on the root: dropping it would abandon the whole retry, never a
# sound verdict regardless of what a reasoner said.
ROOT_NEED_ID = "root"


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
    # need_id -> its own AbsenceProof, when one exists that is genuinely
    # about that one need (proof.need_id != "") -- a proof that couldn't be
    # tied to one need (e.g. _verify_inheritance_completeness's
    # question-level scan) simply matches nothing here and every need it
    # might be relevant to falls through to "open" below, rather than
    # guessing an owner.
    absence_proof_by_need: dict[str, AbsenceProof] = {
        proof.need_id: proof for proof in state.absence_proofs if proof.need_id
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
                epistemic_state=_epistemic_state_for(absence_proof_by_need.get(need_id)),
            )
        )
    return TaskTrajectoryPackage(
        question=state.question,
        prior_answer=state.answer,
        stuck_nodes=stuck_nodes,
        graph_decomposition_log=[round_.graph_delta for round_ in state.rounds],
    )


def _epistemic_state_for(proof: AbsenceProof | None) -> str:
    """Deterministic, no reasoner call involved -- Grounded Fast Repair's
    whole point is that epistemic_state is grounded fact, never an LLM
    guess (see NeedAlignmentVerdict's docstring). No matching proof at
    all -> "open" (may exist, nobody has determined otherwise). A matching
    proof that is exhaustive and concludes "not_found" -> "absence_
    supported". Any other matching proof (non-exhaustive, or exhaustive
    but inconclusive) -> "insufficient_evidence".
    """
    if proof is None:
        return "open"
    if proof.exhaustive and proof.conclusion == "not_found":
        return "absence_supported"
    return "insufficient_evidence"


@dataclass
class RepairSeed:
    """What a RepairPlan resolves into for LocalCoordinator.ask()'s new
    initial_graph/initial_recovery/repair_guidance/forced_first_round_*
    parameters, plus which need_ids a repair plan actually targeted --
    meant to be applied to a fresh copy of the prior graph/RecoveryState by
    the caller (LocalCoordinator.retry_from_trajectory), not mutated here,
    so this stays a pure transformation from (prior state, repair plan) to
    instructions.

    Every action kind resolves into exactly one of two treatments -- there
    is no longer a third "hope the Orchestrator listens" bucket:
    - Hard structural repairs (change_dependency, redecompose, merge_needs)
      are unambiguous graph edits, applied directly to the retry's starting
      graph before round 0 even begins.
    - Execution-policy repairs (reuse_assignment, replace_assignment,
      form_local_bridge, force_global_search) are forced to actually run
      once, at the retry's round 0 (see LocalCoordinator.ask's
      forced_first_round_assignments/forced_first_round_global_search_ids),
      before the Orchestrator regains ordinary freedom to route as it sees
      fit from round 1 onward. This was a deliberate correction: leaving
      these as advisory-only text made it impossible to tell, from a
      retry's outcome, whether a repair plan was bad or simply never
      followed -- forcing the execution makes the effect of the repair
      itself observable and testable, independent of whether the
      Orchestrator would have chosen the same thing on its own.
    """

    dependency_changes: dict[str, list[str]] = field(default_factory=dict)
    redecompose_node_ids: set[str] = field(default_factory=set)
    # primary need_id -> need_ids to fold into it (see _apply_merges).
    merges: dict[str, list[str]] = field(default_factory=dict)
    # need_id -> worker ids to force-assign at the retry's round 0 --
    # reuse_assignment/replace_assignment/form_local_bridge all resolve
    # here (a coalition is just >1 worker id in the same slot, no separate
    # representation needed).
    forced_assignments: dict[str, list[str]] = field(default_factory=dict)
    # need_ids to force a broad, unrestricted-territory search on at the
    # retry's round 0 -- force_global_search. Only ever forced once, at
    # round 0, by construction (nothing re-adds to this set on later
    # rounds), which is itself the "at most once per repaired need"
    # session-local guard: without it, a temporary_bridge/global_fallback
    # style tactic could spin identically every round for no new evidence,
    # the exact failure mode the RecoveryState streak machinery already
    # exists to prevent.
    forced_global_search_ids: set[str] = field(default_factory=set)
    # Informational only, for repair_guidance text: documents what the
    # execution-policy actions above already did, so the Orchestrator has
    # that context from round 1 onward (e.g. not to blindly re-request the
    # same worker a forced replace_assignment already tried). Hard
    # structural repairs don't get a line here -- the graph edit itself is
    # what the Orchestrator sees, no separate narration needed.
    guidance_lines: list[str] = field(default_factory=list)
    # Every need_id any action targeted, regardless of kind -- a repair
    # plan proposing *anything* for a need_id means "give this another
    # shot", so retry_from_trajectory un-abandons all of these, not just
    # the ones with a structural redecompose action.
    targeted_need_ids: set[str] = field(default_factory=set)


def apply_alignment_verdicts(
    graph: NeedGraph,
    plan: NeedAlignmentPlan,
    prior_epistemic_states: dict[str, str],
) -> tuple[NeedGraph, dict[str, str], set[str], set[str]]:
    """Grounded Fast Repair's Need Alignment Gate, applied: turns a
    NeedAlignmentPlan (FastEvolutionReasoner.assess_need_alignment's
    verdicts) into an aligned copy of `graph` plus the bookkeeping the
    caller (LocalCoordinator.retry_from_trajectory) needs to seed
    RecoveryState correctly. `prior_epistemic_states` is
    `{s.need_id: s.epistemic_state for s in package.stuck_nodes}` -- the
    grounded values assemble_trajectory_package already computed; this
    function only ever carries them forward or resets them, never
    invents one. Returns (aligned_graph, epistemic_states,
    discarded_misaligned_ids, reframed_need_ids):

    - "keep" (or no verdict at all, same default): node untouched;
      epistemic_state carries over from `prior_epistemic_states` (or
      "open" if the need_id has no entry there -- e.g. a resolved gen0
      node, which assemble_trajectory_package never summarizes, being
      kept as a plain pass-through graph node).
    - "reframe": treated as a fresh investigation, not a text edit on
      stale state (correction 2) -- `node.need` and
      `node.detail.description`/`missing` become `reframed_need`, and
      `children`/`depends_on` are cleared (any decomposition done under
      the old framing answers the wrong question, so a reframed node
      starts as a fresh leaf). epistemic_state unconditionally resets to
      "open" (there is, by construction, never an existing proof for
      wording nothing has searched under yet). Its id is returned in
      `reframed_need_ids` so the caller also clears its
      `tried_workers_by_node` entry and any stuck-episode membership --
      "worker X already tried and failed" and "this need is part of
      stuck episode Y" are both facts about the *old* framing.
    - "drop": node is left in the graph (structure/dependents intact,
      matching how ordinary consolidation "drop" already behaves) but its
      id is returned in `discarded_misaligned_ids`, NEVER
      `abandoned_node_ids` -- "this need was never a legitimate question"
      is a different fact from "we tried and failed", and the latter
      already feeds recovery-streak/evolution-memory bookkeeping that a
      misaligned-but-never-attempted need must not pollute. The caller is
      responsible for excluding `discarded_misaligned_ids` from the
      frontier (unioned with `abandoned_node_ids` there) without treating
      it as abandoned for any other purpose. Its epistemic_state is
      omitted -- meaningless once excluded from the frontier either way.
    - A "drop" verdict naming the graph's root need (see ROOT_NEED_ID) is
      coerced to keep -- dropping the root would abandon the whole retry,
      never a sound verdict regardless of what a reasoner said.
    - A verdict naming an unknown need_id is ignored.
    """
    nodes = dict(graph.nodes)
    epistemic_states: dict[str, str] = {}
    discarded_misaligned_ids: set[str] = set()
    reframed_need_ids: set[str] = set()
    verdict_by_need_id = {v.need_id: v for v in plan.verdicts}

    for need_id, node in graph.nodes.items():
        verdict = verdict_by_need_id.get(need_id)
        action = verdict.verdict if verdict is not None else "keep"
        if action == "drop" and need_id == ROOT_NEED_ID:
            action = "keep"

        if action == "keep":
            epistemic_states[need_id] = prior_epistemic_states.get(need_id, "open")
            continue
        if action == "drop":
            discarded_misaligned_ids.add(need_id)
            continue
        # action == "reframe"
        assert verdict is not None
        nodes[need_id] = node.model_copy(
            update={
                "need": verdict.reframed_need,
                "detail": node.detail.model_copy(
                    update={
                        "description": verdict.reframed_need,
                        "missing": verdict.reframed_need,
                    }
                ),
                "children": [],
                "depends_on": [],
            }
        )
        epistemic_states[need_id] = "open"
        reframed_need_ids.add(need_id)

    return (
        graph.model_copy(update={"nodes": nodes}),
        epistemic_states,
        discarded_misaligned_ids,
        reframed_need_ids,
    )


def resolve_repair_plan(plan: RepairPlan) -> RepairSeed:
    """Splits a RepairPlan's actions into the two treatments described in
    RepairSeed's docstring: change_dependency/redecompose/merge_needs are
    unambiguous structural edits, applied mechanically; the remaining four
    kinds (reuse_assignment, replace_assignment, form_local_bridge,
    force_global_search) are forced to execute once at the retry's round 0
    rather than left as text the Orchestrator might ignore.
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
        if action.kind == "merge_needs" and action.merge_with:
            seed.merges.setdefault(action.need_id, []).extend(action.merge_with)
            continue
        if (
            action.kind in ("reuse_assignment", "replace_assignment", "form_local_bridge")
            and action.worker_ids
        ):
            seed.forced_assignments[action.need_id] = list(action.worker_ids)
            line = (
                f"- {action.kind} was force-executed on {action.need_id!r} this retry's "
                f"round 0 with worker(s) {', '.join(action.worker_ids)} (already ran once)"
            )
            if action.rationale:
                line += f": {action.rationale}"
            seed.guidance_lines.append(line)
            continue
        if action.kind == "force_global_search":
            seed.forced_global_search_ids.add(action.need_id)
            line = (
                f"- force_global_search was force-executed on {action.need_id!r} this "
                "retry's round 0 (already ran once)"
            )
            if action.rationale:
                line += f": {action.rationale}"
            seed.guidance_lines.append(line)
            continue
        # Malformed/incomplete action for its kind (e.g. replace_assignment
        # with no worker_ids) -- ignored rather than failing the whole
        # plan, same degrade-gracefully posture as propose_repair's own
        # empty-actions fallback.
    return seed


def render_repair_guidance(package: TaskTrajectoryPackage, seed: RepairSeed) -> str:
    """Human-readable context threaded into every plan_round() call of the
    retry -- purely informational (see RepairSeed's own docstring): every
    action already had its effect applied before round 0 (a mechanical
    graph edit or a forced one-time execution), this is not a request for
    the Orchestrator to still act on.
    """
    if not seed.guidance_lines:
        return ""
    header = (
        "A prior attempt on this exact question got stuck; before this retry began, "
        "a repair analysis of that attempt's own trajectory already took the "
        "following actions:\n"
    )
    return header + "\n".join(seed.guidance_lines)


def _apply_merges(
    nodes: dict[str, NeedNode], merges: dict[str, list[str]]
) -> dict[str, NeedNode]:
    """Folds each merge_with id into its primary need_id: any node
    referencing a merged-away id in depends_on/children is redirected to
    the primary instead (deduplicated), then the merged-away node is
    dropped from the graph entirely. A self-loop redirection could
    introduce (a node ending up depending on/parenting itself) is filtered
    out. A primary id that doesn't exist, or a merge target that's already
    gone (e.g. named by two different merge_needs actions in the same
    plan), is skipped rather than raising -- same tolerant-of-a-single-bad-
    action posture as resolve_repair_plan.
    """
    nodes = dict(nodes)
    for primary_id, merge_with in merges.items():
        if primary_id not in nodes:
            continue
        merged_away = {
            other_id for other_id in merge_with if other_id in nodes and other_id != primary_id
        }
        if not merged_away:
            continue
        for need_id, node in list(nodes.items()):
            new_depends_on = [d if d not in merged_away else primary_id for d in node.depends_on]
            new_depends_on = [d for d in dict.fromkeys(new_depends_on) if d != need_id]
            new_children = [c if c not in merged_away else primary_id for c in node.children]
            new_children = [c for c in dict.fromkeys(new_children) if c != need_id]
            if new_depends_on != node.depends_on or new_children != node.children:
                nodes[need_id] = node.model_copy(
                    update={"depends_on": new_depends_on, "children": new_children}
                )
        for other_id in merged_away:
            nodes.pop(other_id, None)
    return nodes


def build_retry_starting_state(
    prior_state: EvidenceState, seed: RepairSeed
) -> tuple[NeedGraph, list[Evidence]]:
    """Builds the retry's starting graph (prior_state.final_need_graph,
    carrying forward resolved nodes too so dependents' chains stay
    consistent, with merge_needs/change_dependency seed edits applied) and
    starting evidence pool (the prior attempt's full evidence, so the
    retry only has to make incremental progress on what was stuck, not
    re-discover everything). RecoveryState seeding (tried_workers_by_node
    carried forward, abandoned nodes cleared for targeted needs) and the
    forced_assignments/forced_global_search_ids execution itself are the
    caller's job (LocalCoordinator.retry_from_trajectory /
    LocalCoordinator.ask) since RecoveryState is coordinator-local and
    forcing an execution needs the coordinator's search/worker machinery,
    neither of which this module depends on.

    Every targeted need_id (not just redecompose ones) gets its
    progress/rounds_without_progress reset to fresh: a stale
    progress="stuck" carried over from the prior attempt would keep the
    node out of the retry's own ready frontier (compute_frontier routes a
    still-"stuck" node to stuck_subgraphs, not ready) regardless of which
    worker a reuse_assignment/replace_assignment action forces -- any
    action targeting a need_id means "give this another shot", the same
    reasoning already applied to abandoned_node_ids in
    retry_from_trajectory.
    """
    nodes = _apply_merges(dict(prior_state.final_need_graph), seed.merges)
    for need_id, new_depends_on in seed.dependency_changes.items():
        if need_id in nodes:
            nodes[need_id] = nodes[need_id].model_copy(update={"depends_on": new_depends_on})
    for need_id in seed.targeted_need_ids:
        if need_id in nodes:
            nodes[need_id] = nodes[need_id].model_copy(
                update={"progress": "not_stuck", "rounds_without_progress": 0}
            )
    return NeedGraph(nodes=nodes), list(prior_state.evidence)
