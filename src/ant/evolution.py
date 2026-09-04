from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from ant.domain import Territory, WorkerCard
from ant.indexing.cards import build_worker_cards, template_routing_summary
from ant.memory import ColonyMemoryStore, IndexStore
from ant.memory.colony import EpisodeAggregate, MemoryRoute
from ant.providers import EvolutionReasoner
from ant.retrieval.dense import DenseEmbedder, get_shared_embedder
from ant.scoring_config import DEFAULT_SCORING_CONFIG
from ant.tools.local import LocalSearchTool

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")

# Episode-driven strengthen_route/birth_bridge (_apply_episode_actions):
# a recurring pattern is only "actual progress", not just "got called a
# lot", once at least half its recorded occurrences succeeded. Confirmed
# necessary on a real qibo run: an aggregate with 13/35 successes (37%) --
# recurring mostly because it kept NOT working, not because it worked --
# still got birth_bridge'd by the reasoner. This is a structural floor
# underneath that judgment call, not a replacement for it.
_MIN_EPISODE_SUCCESS_RATIO = 0.5

# No event kind actually removes a worker from the pool anymore (Phase 3
# of the multi-generation organizational evolution redesign: birth,
# specialize, and merge are all non-destructive overlays now -- a
# specialize ADDS children alongside its still-present parent, a merge ADDS
# a composite alongside its still-present two sources; only "retire" drops
# a worker, and it's added to removed_worker_ids directly at the retire
# site, not via this set). Kept as an explicit allowlist (currently empty)
# rather than an exclusion list, and kept as its own named constant rather
# than inlined, so a future destructive lifecycle stage (dormant workers
# actually being dropped -- see WorkerCard.lifecycle_state's own docstring
# on why that's deliberately not implemented yet) has one obvious place to
# register itself instead of scattering removal logic across each
# operator's own branch again.
_WORKER_REMOVING_EVENT_KINDS: frozenset[str] = frozenset()


def _removed_worker_ids_from(events: list[EvolutionEvent]) -> set[str]:
    """Regression fix for a real bug found live on qibo: evolve_workers used
    to union source_worker_ids from EVERY episode-driven event kind into
    removed_worker_ids, including "strengthen_route" and "birth" -- neither
    of which removes a worker. mark_stale(removed_worker_ids) then
    invalidated those still-present workers' OTHER, perfectly valid routes
    as pure collateral damage (confirmed live: 9 real strengthen_route
    events on qibo wrongly staled 49 routes, including all 3 of gen0's own
    legitimate high-quality ones, while every "removed" worker was still
    present in the final worker list, unchanged). See
    _WORKER_REMOVING_EVENT_KINDS' own docstring for which kinds legitimately
    belong here.
    """
    return {
        worker_id
        for event in events
        if event.kind in _WORKER_REMOVING_EVENT_KINDS
        for worker_id in event.source_worker_ids
    }


# --- TEMPORARY DIAGNOSTIC TELEMETRY (2026-09-03) ---------------------------
# Purely additive print-based tracing to root-cause a live discrepancy: DB
# side effects (mark_stale rows, weight=5.0 strengthen_route rows) prove
# evolve_workers executed real structural/episode actions on a real qibo
# run, but the EvolutionResult it returned reported events=[] and an
# unchanged worker_count. No control flow, gating, or return value below is
# changed by any _tel(...) call -- remove this block and every _tel(...)
# call site once the discrepancy is root-caused.
def _tel(*parts: object) -> None:
    print("[EVOTEL]", *parts, flush=True)


# --- END TEMPORARY DIAGNOSTIC TELEMETRY -------------------------------------


def _with_routing_summary(card: WorkerCard, reasoner: EvolutionReasoner | None) -> WorkerCard:
    """Generate/refresh routing_summary for a card that was just born,
    specialized, or merged -- a real LLM call via the reasoner when one is
    available, or the same zero-cost deterministic template used at
    initial index time otherwise (see indexing.cards.template_routing_summary).
    Every WorkerCard construction site in this module routes through this
    rather than leaving routing_summary at its default "" -- an empty
    routing_summary would make that worker effectively invisible in the
    Orchestrator's per-round routing context, which reads only this field,
    not the full card.
    """
    summary = (
        reasoner.summarize_routing(card=card)
        if reasoner is not None
        else template_routing_summary(card)
    )
    return card.model_copy(update={"routing_summary": summary})


class EvolutionEvent(BaseModel):
    kind: str
    worker_id: str
    reason: str
    source_worker_ids: list[str] = Field(default_factory=list)


class EvolutionResult(BaseModel):
    events: list[EvolutionEvent] = Field(default_factory=list)
    worker_count: int = 0


def evolve_workers(
    index_path: Path,
    repo_root: Path | None = None,
    reasoner: EvolutionReasoner | None = None,
    min_coalition_count: int = DEFAULT_SCORING_CONFIG.evolution.min_coalition_count,
    retire_empty: bool = True,
    merge_overlap: float = DEFAULT_SCORING_CONFIG.evolution.merge_overlap_threshold,
    min_specialization_routes: int = DEFAULT_SCORING_CONFIG.evolution.min_specialization_routes,
    min_specialization_group_routes: int = (
        DEFAULT_SCORING_CONFIG.evolution.min_specialization_group_routes
    ),
    min_routes_for_health_check: int = DEFAULT_SCORING_CONFIG.evolution.min_routes_for_health_check,
    healthy_route_ratio: float = DEFAULT_SCORING_CONFIG.evolution.healthy_route_ratio,
    negative_presence_gate_ratio: float = (
        DEFAULT_SCORING_CONFIG.evolution.negative_presence_gate_ratio
    ),
    min_episode_count: int = DEFAULT_SCORING_CONFIG.evolution.min_episode_count,
    semantic_cluster_similarity_threshold: float = (
        DEFAULT_SCORING_CONFIG.evolution.semantic_cluster_similarity_threshold
    ),
    min_semantic_cluster_file_support: int = (
        DEFAULT_SCORING_CONFIG.evolution.min_semantic_cluster_file_support
    ),
    generation: int = 0,
    needs_overlap_threshold: float = DEFAULT_SCORING_CONFIG.evolution.needs_overlap_threshold,
) -> EvolutionResult:
    # `generation` is purely descriptive lineage metadata (stamped onto
    # WorkerCard.generation_created for every worker this call creates) --
    # it does not gate or change any decision this function makes. 0 (the
    # default) means "caller doesn't track generations"; gen_compare's own
    # cumulative slow-gen{k} loop passes its real k.
    store = IndexStore(index_path)
    memory = ColonyMemoryStore(index_path)
    workers = store.load_workers()
    events: list[EvolutionEvent] = []
    removed_worker_ids: set[str] = set()

    routes = memory.all_routes()
    routes_by_worker: dict[str, list[MemoryRoute]] = defaultdict(list)
    for route in routes:
        for worker_id in route.worker_ids:
            routes_by_worker[worker_id].append(route)

    _tel("ENTRY workers_before=", len(workers), "ids=", sorted(w.id for w in workers))
    _tel(
        "ENTRY candidate recurring_coalitions=",
        len(memory.recurring_coalitions(min_count=min_coalition_count)),
        "candidate aggregate_episodes=",
        len(memory.aggregate_episodes(min_count=min_episode_count)),
    )

    workers, specialize_events = _specialize_overloaded_workers(
        workers,
        routes_by_worker,
        min_routes=min_specialization_routes,
        min_group_routes=min_specialization_group_routes,
        min_health_routes=min_routes_for_health_check,
        healthy_ratio=healthy_route_ratio,
        negative_presence_gate_ratio=negative_presence_gate_ratio,
        repo_root=repo_root,
        reasoner=reasoner,
        semantic_cluster_similarity_threshold=semantic_cluster_similarity_threshold,
        min_semantic_cluster_file_support=min_semantic_cluster_file_support,
        generation=generation,
    )
    events.extend(specialize_events)
    removed_worker_ids.update(_removed_worker_ids_from(specialize_events))
    _tel(
        "AFTER specialize: events+=", len(specialize_events),
        "removed_worker_ids=", sorted(removed_worker_ids),
        "workers_after=", len(workers),
    )

    # recurring_coalitions() is no longer consulted here -- it stays a
    # candidate GENERATOR (per-task-deduplicated raw co-occurrence), but a
    # candidate it finds is now merged into _apply_episode_actions' own
    # aggregate_episodes list (see _merge_coalition_candidates_into_
    # aggregates) so it gets the exact same task-level-recurrence signal
    # and exactly one authoritative decide_episode_action decision as any
    # other birth candidate -- never a separate, cruder count-only birth
    # that a richer-signal decision elsewhere might have refused the same
    # cycle. Regression fix for a real bug: see that helper's own
    # docstring for the live qibo case this closes.

    if retire_empty:
        kept = []
        for worker in workers:
            if worker.files:
                kept.append(worker)
                continue
            _tel("RETIRE:", worker.id, "(no owned files after refresh)")
            events.append(
                EvolutionEvent(
                    kind="retire",
                    worker_id=worker.id,
                    reason="Worker has no owned files after refresh.",
                )
            )
            removed_worker_ids.add(worker.id)
        workers = kept
    _tel(
        "AFTER retire: workers=", len(workers),
        "events total=", len(events),
        "removed_worker_ids so far=", sorted(removed_worker_ids),
    )

    workers_before_merge_ids = sorted(w.id for w in workers)
    workers, merge_events = _merge_overlapping_workers(
        workers,
        threshold=merge_overlap,
        routes_by_worker=routes_by_worker,
        min_health_routes=min_routes_for_health_check,
        healthy_ratio=healthy_route_ratio,
        reasoner=reasoner,
        generation=generation,
        needs_overlap_threshold=needs_overlap_threshold,
    )
    events.extend(merge_events)
    removed_worker_ids.update(_removed_worker_ids_from(merge_events))
    _tel(
        "AFTER merge: merge_events=", [e.model_dump() for e in merge_events],
        "workers_before_ids=", workers_before_merge_ids,
        "workers_after_ids=", sorted(w.id for w in workers),
        "events total=", len(events),
        "removed_worker_ids so far=", sorted(removed_worker_ids),
    )

    if reasoner is not None:
        workers_before_episode_ids = sorted(w.id for w in workers)
        workers, episode_events = _apply_episode_actions(
            workers,
            memory,
            reasoner,
            min_episode_count,
            merge_overlap,
            min_coalition_count=min_coalition_count,
            routes_by_worker=routes_by_worker,
            min_health_routes=min_routes_for_health_check,
            healthy_ratio=healthy_route_ratio,
            generation=generation,
        )
        events.extend(episode_events)
        removed_worker_ids.update(_removed_worker_ids_from(episode_events))
        _tel(
            "AFTER _apply_episode_actions: episode_events=",
            [e.model_dump() for e in episode_events],
            "workers_before_ids=", workers_before_episode_ids,
            "workers_after_ids=", sorted(w.id for w in workers),
            "events total=", len(events),
            "removed_worker_ids so far=", sorted(removed_worker_ids),
        )
    else:
        _tel("SKIPPED _apply_episode_actions: reasoner is None")

    territories = [
        Territory(
            id=worker.territory_id,
            root=worker.root,
            files=worker.files,
            summary="Synthetic territory from current worker directory.",
        )
        for worker in workers
    ]
    _tel(
        "PRE-RETURN: events=", len(events), "will_save=", bool(events),
        "worker_count=", len(workers), "worker_ids=", sorted(w.id for w in workers),
        "removed_worker_ids=", sorted(removed_worker_ids),
        "will_mark_stale=", bool(removed_worker_ids),
    )
    if events:
        store.save(territories, workers)
        _tel("store.save() called with", len(workers), "workers")
    else:
        _tel("store.save() SKIPPED (events is empty)")
    if removed_worker_ids:
        marked = ColonyMemoryStore(index_path).mark_stale(
            sorted(removed_worker_ids),
            reason="Worker retired/specialized/merged away by colony evolution.",
        )
        _tel("mark_stale() called for", sorted(removed_worker_ids), "-> rows marked stale:", marked)
    result = EvolutionResult(events=events, worker_count=len(workers))
    _tel(
        "RETURN: events=", [e.model_dump() for e in result.events],
        "worker_count=", result.worker_count,
    )
    return result


def _merge_coalition_candidates_into_aggregates(
    aggregates: list[EpisodeAggregate],
    memory: ColonyMemoryStore,
    min_coalition_count: int,
) -> list[EpisodeAggregate]:
    """recurring_coalitions() stays a valid CANDIDATE GENERATOR -- it sees
    per-task-deduplicated raw co-occurrence that a genuinely recurring
    coalition might not otherwise surface if it fell short of
    min_episode_count (e.g. custom configs where min_coalition_count is more
    lenient) -- but it no longer gets to decide anything on its own. Every
    candidate it finds is folded into the SAME aggregate_episodes list
    _apply_episode_actions already runs through decide_episode_action, so a
    coalition discovered this way gets exactly the same task-level-recurrence
    signal and exactly one authoritative decision as one discovered via
    episode aggregation -- never a separate, cruder count-only birth.

    Regression fix for a real bug found live on qibo: a coalition
    (worker-src/qibo/gates, worker-src/qibo/models) recurring 21 times
    across only 3 tasks -- of which just 1 actually resolved a need --
    still got born as a permanent bridge worker via the OLD raw-coalition
    path's own unconditional birth, seconds before the episode-level path
    (with full visibility into that weak task-level support) correctly
    said no_change for the exact same pair. Two independent authorities
    disagreeing and the cruder one winning by running first.

    A coalition worker-set already represented among `aggregates` with
    strategy="coalition" (the common case: every coalition-forming
    execution also records a "coalition"-strategy episode for the same
    task/need, so the two data sources normally coincide) is skipped here
    -- the richer episode data already covers it. Only a candidate with NO
    matching episode-level entry gets a synthesized, deliberately
    conservative stand-in (successes=0, no unique-task/progress detail
    beyond what recurring_coalitions itself already deduplicates per task)
    -- honest about having zero outcome signal for it, which the existing
    low-success gate below already treats as "not enough evidence to
    promote," not a free pass.
    """
    existing_coalition_keys = {
        tuple(sorted(aggregate.workers))
        for aggregate in aggregates
        if aggregate.strategy == "coalition"
    }
    merged = list(aggregates)
    for worker_ids, count in memory.recurring_coalitions(min_count=min_coalition_count):
        key = tuple(sorted(worker_ids))
        if key in existing_coalition_keys:
            continue
        merged.append(
            EpisodeAggregate(
                strategy="coalition",
                workers=list(key),
                need_terms=[],
                occurrences=count,
                successes=0,
                total_evidence_gain=0,
                total_need_reduction=0,
                unique_task_count=count,
                tasks_with_progress=0,
                tasks_with_need_reduction=0,
            )
        )
    merged.sort(key=lambda aggregate: aggregate.occurrences, reverse=True)
    return merged


def _apply_episode_actions(
    workers: list[WorkerCard],
    memory: ColonyMemoryStore,
    reasoner: EvolutionReasoner,
    min_episode_count: int,
    merge_overlap: float,
    min_coalition_count: int,
    routes_by_worker: dict[str, list[MemoryRoute]],
    min_health_routes: int,
    healthy_ratio: float,
    generation: int = 0,
) -> tuple[list[WorkerCard], list[EvolutionEvent]]:
    """Acts on ColonyMemoryStore.aggregate_episodes' recurring (strategy,
    worker-set) patterns -- e.g. "temporary_bridge kept resolving a
    proxy-validation-shaped need across 3 separate tasks" -- via
    EvolutionReasoner.decide_episode_action. This is a richer signal than
    raw worker co-occurrence alone (which worker(s) got selected together,
    not which specific temporary adaptation actually worked or how often):
    the same slow, cross-task timescale as the rest of evolve_workers, never
    reacting to a single task's own outcome. recurring_coalitions'
    candidates are merged in via _merge_coalition_candidates_into_aggregates
    -- this is now the ONLY place any birth (coalition-discovered or
    episode-discovered) actually happens.
    """
    worker_by_id = {worker.id: worker for worker in workers}
    events: list[EvolutionEvent] = []
    consumed: set[str] = set()
    aggregates = _merge_coalition_candidates_into_aggregates(
        memory.aggregate_episodes(min_count=min_episode_count), memory, min_coalition_count
    )
    for aggregate in aggregates:
        source_workers = [
            worker_by_id[worker_id]
            for worker_id in aggregate.workers
            if worker_id in worker_by_id and worker_id not in consumed
        ]
        if len(source_workers) != len(aggregate.workers):
            # At least one involved worker no longer exists (already
            # retired/specialized/merged away this cycle, or by a prior
            # aggregate in this same loop) -- nothing to act on.
            continue
        action = reasoner.decide_episode_action(
            strategy=aggregate.strategy,
            need_terms=aggregate.need_terms,
            occurrences=aggregate.occurrences,
            successes=aggregate.successes,
            total_evidence_gain=aggregate.total_evidence_gain,
            total_need_reduction=aggregate.total_need_reduction,
            unique_task_count=aggregate.unique_task_count,
            tasks_with_progress=aggregate.tasks_with_progress,
            tasks_with_need_reduction=aggregate.tasks_with_need_reduction,
            workers=aggregate.workers,
        )
        reason = (
            f"Episode-driven: '{aggregate.strategy}' succeeded in "
            f"{aggregate.successes}/{aggregate.occurrences} recorded tasks "
            f"(total evidence gain {aggregate.total_evidence_gain})."
        )
        success_ratio = aggregate.successes / aggregate.occurrences
        low_success = success_ratio < _MIN_EPISODE_SUCCESS_RATIO
        if action in ("strengthen_route", "birth_bridge") and low_success:
            # Recurring often is not the same as working often -- see the
            # constant's docstring. Downgrades rather than raising: the
            # occurrence-count gate (min_episode_count) already established
            # this pattern is real and recurring, it just isn't grounds to
            # structurally promote or reinforce it yet.
            continue
        if action == "birth_bridge" and len(set(aggregate.workers)) < 2:
            # A "bridge" born from a single worker (a "normal"-strategy
            # episode, not an actual coalition) is just a same-files clone
            # under a new id, not a cross-territory specialist -- confirmed
            # on a real qibo run, this produced 3 such duplicates in one
            # evolve call. The single-worker case this pattern is real
            # evidence for is "this worker is doing well", which
            # strengthen_route already expresses without growing the pool.
            action = "strengthen_route"
        if action == "no_change":
            continue
        if action == "strengthen_route":
            memory.save_route(
                MemoryRoute(
                    need_terms=aggregate.need_terms,
                    worker_ids=aggregate.workers,
                    weight=5.0,
                    is_high_quality=True,
                )
            )
            events.append(
                EvolutionEvent(
                    kind="strengthen_route",
                    worker_id=",".join(aggregate.workers),
                    reason=reason,
                    source_worker_ids=aggregate.workers,
                )
            )
        elif action == "birth_bridge":
            if all(
                _is_worker_healthy(worker.id, routes_by_worker, min_health_routes, healthy_ratio)
                for worker in source_workers
            ):
                # Same rationale as the old raw-coalition path's own health
                # gate (now unified here instead of duplicated): every
                # source worker already has a track record of good
                # answers, so birthing a dedicated bridge is for coalitions
                # that are struggling together, not ones that already work
                # fine split apart.
                continue
            bridge_suffix = "-".join(
                worker_id.removeprefix("worker-") for worker_id in aggregate.workers
            )
            bridge_id = f"worker-bridge-{bridge_suffix}"
            if bridge_id in worker_by_id:
                continue
            files = sorted({file for worker in source_workers for file in worker.files})
            if _overlaps_existing_worker(files, workers, merge_overlap):
                continue
            terms = sorted(
                {term for worker in source_workers for term in worker.searchable_terms}
            )[:32]
            interface_responsibility = reasoner.describe_interface_responsibility(
                source_workers=[
                    (worker.id, "; ".join(worker.responsibilities[:2]) or worker.name)
                    for worker in source_workers
                ],
                representative_needs=memory.representative_needs(
                    aggregate.strategy, aggregate.workers
                ),
                need_terms=aggregate.need_terms,
                unique_task_count=aggregate.unique_task_count,
                occurrences=aggregate.occurrences,
            )
            redundant_with = _interface_subsumed_by_existing(
                interface_responsibility,
                files,
                workers,
                reasoner,
            )
            if redundant_with is not None:
                # Same treatment as the single-source-worker downgrade
                # above: this recurring pattern is real signal, just
                # misassigned to "birth a clone" instead of "reinforce the
                # worker that already covers this specialty" -- reinforces
                # the specific *existing* worker found redundant, not
                # aggregate.workers (the sources that would have been
                # combined into the now-skipped clone).
                memory.save_route(
                    MemoryRoute(
                        need_terms=aggregate.need_terms,
                        worker_ids=[redundant_with],
                        weight=5.0,
                        is_high_quality=True,
                    )
                )
                events.append(
                    EvolutionEvent(
                        kind="strengthen_route",
                        worker_id=redundant_with,
                        reason=f"{reason} Interface responsibility "
                        f"({interface_responsibility!r}) already fully owned by existing "
                        f"worker {redundant_with}; reinforced instead of birthing a clone.",
                        source_worker_ids=aggregate.workers,
                    )
                )
                continue
            bridge = _with_routing_summary(
                WorkerCard(
                    id=bridge_id,
                    territory_id=bridge_id.removeprefix("worker-"),
                    name=" + ".join(worker.name for worker in source_workers[:3]) + " bridge",
                    root="",
                    responsibilities=[
                        interface_responsibility,
                        "Permanent bridge born from a recurring successful temporary adaptation.",
                    ],
                    searchable_terms=terms,
                    files=files,
                    parent_worker_ids=aggregate.workers,
                    structural_action="birth",
                    generation_created=generation,
                    lifecycle_state="probationary",
                ),
                reasoner,
            )
            # Non-destructive overlay (Phase 3): source_workers stay in the
            # pool exactly as before, the bridge is added alongside them --
            # never a replacement. Routing must still be able to recruit an
            # original worker even after a structural sibling exists for
            # it.
            workers = [*workers, bridge]
            worker_by_id[bridge.id] = bridge
            events.append(
                EvolutionEvent(
                    kind="birth",
                    worker_id=bridge.id,
                    reason=reason,
                    source_worker_ids=aggregate.workers,
                )
            )
        elif action == "merge" and len(source_workers) == 2:
            first, second = source_workers
            merged_worker = _with_routing_summary(
                WorkerCard(
                    id=f"worker-merge-{first.id.removeprefix('worker-')}"
                    f"-{second.id.removeprefix('worker-')}",
                    territory_id=f"merge-{first.territory_id}-{second.territory_id}",
                    name=f"{first.name} / {second.name} merged",
                    root=first.root or second.root,
                    responsibilities=[*first.responsibilities, *second.responsibilities, reason],
                    searchable_terms=sorted(
                        set(first.searchable_terms) | set(second.searchable_terms)
                    ),
                    files=sorted(set(first.files) | set(second.files)),
                    parent_worker_ids=[first.id, second.id],
                    structural_action="merge",
                    generation_created=generation,
                    lifecycle_state="probationary",
                ),
                reasoner,
            )
            # Non-destructive overlay (Phase 3): first/second stay in the
            # pool -- the composite is a virtual merged role added
            # alongside them, not a replacement. `consumed` still tracks
            # them so a LATER aggregate this same cycle can't spend either
            # one on a second structural action, independent of whether
            # they physically remain in `workers`.
            workers = [*workers, merged_worker]
            worker_by_id[merged_worker.id] = merged_worker
            consumed.update({first.id, second.id})
            events.append(
                EvolutionEvent(
                    kind="merge",
                    worker_id=merged_worker.id,
                    reason=reason,
                    source_worker_ids=[first.id, second.id],
                )
            )
    return workers, events


def _needs_overlap(
    worker_a_routes: list[MemoryRoute], worker_b_routes: list[MemoryRoute]
) -> float:
    """Jaccard overlap of two workers' own recorded need_terms vocabularies
    -- a route/responsibility-overlap merge-candidacy signal independent of
    file overlap (see EvolutionConfig.needs_overlap_threshold's own
    docstring for why this deliberately does not need embedding
    similarity). 0.0 whenever either side has no route history yet -- no
    signal, not "definitely distinct".
    """
    terms_a = {term for route in worker_a_routes for term in route.need_terms}
    terms_b = {term for route in worker_b_routes for term in route.need_terms}
    if not terms_a or not terms_b:
        return 0.0
    union = terms_a | terms_b
    return len(terms_a & terms_b) / len(union)


def _merge_overlapping_workers(
    workers: list[WorkerCard],
    threshold: float,
    routes_by_worker: dict[str, list[MemoryRoute]],
    min_health_routes: int,
    healthy_ratio: float,
    reasoner: EvolutionReasoner | None = None,
    generation: int = 0,
    needs_overlap_threshold: float = DEFAULT_SCORING_CONFIG.evolution.needs_overlap_threshold,
) -> tuple[list[WorkerCard], list[EvolutionEvent]]:
    # Non-destructive overlay (Phase 3): every original worker in
    # `workers` stays in the returned pool unchanged -- a composite is
    # ADDED alongside its two sources, never a replacement. `consumed`
    # still exists purely to stop the same worker being spent on two
    # different composites in one pass (A+B and A+C in the same cycle
    # would both claim A), independent of the fact that neither A nor B
    # nor C is ever actually removed.
    events: list[EvolutionEvent] = []
    consumed: set[str] = set()
    composites: list[WorkerCard] = []
    for index, worker in enumerate(workers):
        if worker.id in consumed:
            continue
        worker_files = set(worker.files)
        partner = None
        for other in workers[index + 1 :]:
            if other.id in consumed:
                continue
            # A worker already structurally descended from (or ancestor
            # of) the other is never a merge candidate against it -- a
            # freshly specialized parent/child pair, or two siblings,
            # legitimately share files by design (see
            # _specialize_overloaded_workers' own non-destructive overlay);
            # merging them back together the same cycle would just undo
            # the specialization that just happened.
            if worker.id in other.parent_worker_ids or other.id in worker.parent_worker_ids:
                continue
            # Neither side of a merge may already have a track record of
            # good answers: folding an unrelated (even if file-overlapping)
            # worker's territory into a worker that's already answering
            # well can only dilute it, never improve something that wasn't
            # broken. Structural overlap alone is not sufficient grounds to
            # touch a worker with demonstrated quality.
            if _is_worker_healthy(
                worker.id, routes_by_worker, min_health_routes, healthy_ratio
            ) or _is_worker_healthy(other.id, routes_by_worker, min_health_routes, healthy_ratio):
                continue
            other_files = set(other.files)
            union = worker_files | other_files
            file_overlap = len(worker_files & other_files) / len(union) if union else 0.0
            # File overlap is one candidacy signal now, not the sole
            # prerequisite (Phase 5): a pair with near-zero file overlap
            # but a highly overlapping recurring need_terms vocabulary --
            # two workers repeatedly answering the same KIND of question
            # from genuinely disjoint files -- is still a real merge
            # candidate worth asking the reasoner about.
            needs_overlap = _needs_overlap(
                routes_by_worker.get(worker.id, []), routes_by_worker.get(other.id, [])
            )
            if file_overlap < threshold and needs_overlap < needs_overlap_threshold:
                continue
            # Neither structural signal alone can tell "behaviorally
            # substitutable" apart from "coincidentally overlaps but is
            # conceptually distinct" (file overlap) or "frequently
            # recruited together because each contributes a complementary
            # half of a joint question" (needs overlap -- that shape is
            # birth pressure, not merge pressure) -- this is the judgment
            # on top of either structural candidacy signal.
            if reasoner is not None and not reasoner.should_merge(
                worker_a_id=worker.id,
                worker_a_summary="; ".join(worker.responsibilities[:2]) or worker.name,
                worker_b_id=other.id,
                worker_b_summary="; ".join(other.responsibilities[:2]) or other.name,
            ):
                continue
            partner = other
            break
        if partner is None:
            continue
        consumed.update({worker.id, partner.id})
        merged_worker = _with_routing_summary(
            WorkerCard(
                id=(
                    f"worker-merge-{worker.id.removeprefix('worker-')}"
                    f"-{partner.id.removeprefix('worker-')}"
                ),
                territory_id=f"merge-{worker.territory_id}-{partner.territory_id}",
                name=f"{worker.name} / {partner.name} merged",
                root=worker.root or partner.root,
                responsibilities=worker.responsibilities + partner.responsibilities,
                searchable_terms=sorted(
                    set(worker.searchable_terms) | set(partner.searchable_terms)
                ),
                files=sorted(worker_files | set(partner.files)),
                parent_worker_ids=[worker.id, partner.id],
                structural_action="merge",
                generation_created=generation,
                lifecycle_state="probationary",
            ),
            reasoner,
        )
        composites.append(merged_worker)
        events.append(
            EvolutionEvent(
                kind="merge",
                worker_id=merged_worker.id,
                reason="Workers have highly overlapping file ownership.",
                source_worker_ids=[worker.id, partner.id],
            )
        )
    return [*workers, *composites], events


def _specialize_overloaded_workers(
    workers: list[WorkerCard],
    routes_by_worker: dict[str, list[MemoryRoute]],
    min_routes: int,
    min_group_routes: int,
    min_health_routes: int,
    healthy_ratio: float,
    negative_presence_gate_ratio: float,
    repo_root: Path | None,
    reasoner: EvolutionReasoner | None = None,
    semantic_cluster_similarity_threshold: float = (
        DEFAULT_SCORING_CONFIG.evolution.semantic_cluster_similarity_threshold
    ),
    min_semantic_cluster_file_support: int = (
        DEFAULT_SCORING_CONFIG.evolution.min_semantic_cluster_file_support
    ),
    generation: int = 0,
) -> tuple[list[WorkerCard], list[EvolutionEvent]]:
    """Split a coarse worker into finer workers along its existing directory
    substructure once recurring routes show it is being asked about at least
    two genuinely different sub-areas often enough. This is the "Specialization
    / Birth" mechanism from the design: worker count is not fixed up front, it
    grows where task experience shows a single worker is covering topics that
    do not actually belong together.

    When a worker's territory has no usable subdirectory structure to
    split along at all (_subdirectory_groups returns <2 groups -- e.g.
    yt-dlp's yt_dlp/extractor/, 1010 files flat in one directory), falls
    back to _semantic_groups: clusters the worker's own route HISTORY by
    embedding similarity over need_terms (what keeps recurring), then
    MATERIALIZES each recurring cluster's territory via live, full-
    territory retrieval against the worker's current files (what that
    workload corresponds to now) -- see _semantic_groups' own docstring
    for why this doesn't require storing historical evidence paths.
    """
    events: list[EvolutionEvent] = []
    result: list[WorkerCard] = []
    for worker in workers:
        worker_routes = routes_by_worker.get(worker.id, [])
        # sum(occurrence_count), not len(worker_routes): route consolidation
        # (Phase 6) means the same recurring pattern accumulates onto one
        # row's occurrence_count instead of growing row count -- every
        # threshold here must weigh by occurrence_count or consolidation
        # would silently starve this gate no matter how much real
        # recurrence accumulated.
        total_route_occurrences = sum(route.occurrence_count for route in worker_routes)
        if (
            total_route_occurrences < min_routes
            or _is_worker_healthy(worker.id, routes_by_worker, min_health_routes, healthy_ratio)
            or _mostly_negative_presence(worker_routes, negative_presence_gate_ratio)
        ):
            result.append(worker)
            continue

        # Semantic clustering (what recurring workload actually shows,
        # regardless of filesystem shape) is tried FIRST and preferred --
        # directory structure is one candidate signal, not a prerequisite
        # (Phase 4 of the multi-generation organizational evolution
        # redesign). _subdirectory_groups is the fallback for a worker with
        # too little route history to cluster yet, or no shared embedder
        # available at all -- structural grouping some evolve_workers()
        # environments (tests, no dense extra installed) can still use with
        # zero setup.
        groups = (
            _semantic_groups(
                worker,
                worker_routes,
                repo_root,
                min_group_routes=min_group_routes,
                similarity_threshold=semantic_cluster_similarity_threshold,
                min_file_support=min_semantic_cluster_file_support,
            )
            if repo_root is not None
            else {}
        )
        if len(groups) < 2:
            groups = _subdirectory_groups(worker)
        groups = _fold_colliding_groups(groups, worker, workers)
        if len(groups) < 2:
            result.append(worker)
            continue

        group_counts: dict[str, int] = defaultdict(int)
        group_terms: dict[str, set[str]] = defaultdict(set)
        for route in worker_routes:
            group = _assign_route_to_group(route, groups)
            if group:
                # occurrence_count, not +1 -- see _is_worker_healthy's own
                # docstring on why every threshold here must weigh by it
                # (route consolidation, Phase 6).
                group_counts[group] += route.occurrence_count
                group_terms[group].update(route.need_terms[:6])
        qualifying_groups = {
            group for group, count in group_counts.items() if count >= min_group_routes
        }
        if len(qualifying_groups) < 2:
            result.append(worker)
            continue

        if reasoner is not None and not reasoner.should_specialize(
            worker_id=worker.id,
            worker_summary="; ".join(worker.responsibilities[:2]) or worker.name,
            candidate_groups={
                group: sorted(group_terms[group])[:10] for group in sorted(qualifying_groups)
            },
            route_summaries=[
                f"{group_counts[group]} recurring needs mentioning: "
                f"{', '.join(sorted(group_terms[group])[:8]) or '(no terms recorded)'}"
                for group in sorted(qualifying_groups)
            ],
        ):
            result.append(worker)
            continue

        children = [
            _child_worker(
                worker, group, files, group_counts.get(group, 0), repo_root, reasoner, generation
            )
            for group, files in sorted(groups.items())
        ]
        # Non-destructive overlay (Phase 3): the parent stays in the pool
        # as a fallback, children are ADDED alongside it -- routing must
        # still be able to recruit the original worker even after these
        # more specific siblings exist.
        result.append(worker)
        result.extend(children)
        for child in children:
            events.append(
                EvolutionEvent(
                    kind="specialize",
                    worker_id=child.id,
                    reason=(
                        f"Split from {worker.id}: recurring needs concentrated on "
                        f"{len(qualifying_groups)} distinct substructures under it."
                    ),
                    source_worker_ids=[worker.id],
                )
            )
    return result, events


def _overlaps_existing_worker(
    candidate_files: list[str], workers: list[WorkerCard], threshold: float
) -> bool:
    """True when `candidate_files` (a birth candidate's resulting file set)
    is a near-duplicate -- by the same file-overlap ratio _merge_overlapping_
    workers uses to decide two *existing* workers should be merged -- of any
    single current worker. A birth whose source list includes an existing
    bridge/merge worker can end up with a file set that's effectively just
    that worker's again (its other source already inside the bridge
    contributes nothing new); checking against every current worker, not
    just the immediate sources, catches this regardless of which source
    carried the redundancy, without needing to track birth provenance
    recursively.
    """
    candidate = set(candidate_files)
    if not candidate:
        return False
    for worker in workers:
        worker_files = set(worker.files)
        union = candidate | worker_files
        if not union:
            continue
        if len(candidate & worker_files) / len(union) >= threshold:
            return True
    return False


def _interface_subsumed_by_existing(
    interface_responsibility: str,
    candidate_files: list[str],
    workers: list[WorkerCard],
    reasoner: EvolutionReasoner | None,
) -> str | None:
    """Returns the existing worker's id that already fully owns this birth
    candidate's cross-boundary interface responsibility (per
    reasoner.assess_interface_subsumption), or None. _overlaps_existing_worker
    above catches file-set duplication; this catches a distinct file set
    whose actual FUNCTION already belongs entirely to one existing worker.

    Deliberately NOT should_merge's "is this the same specialty" question:
    a candidate built by combining source worker A's own responsibilities
    will always share vocabulary with A, so that question made redundancy
    near-tautological -- confirmed live on a real qibo run, where every
    tested birth candidate got vetoed this way, 100% of the vetoes landing
    on the colony's handful of large package-level "hub" workers (the ones
    most likely to be a source of any coalition in the first place).
    assess_interface_subsumption asks the narrower, correct question:
    does this existing worker, alone, already perform the JOINT function --
    which a hub worker sharing a topic with one source can still
    legitimately fail, since the interface itself (not either side of it)
    is what must already be covered.

    Deliberately checks EVERY worker sharing a file with the candidate,
    including its own source workers -- per describe_interface_responsibility's
    own docstring, a source worker CAN legitimately turn out to already own
    the joint interface on its own; skipping source workers here would just
    trade one systematic blind spot for another. Only asks about workers
    that share at least one file with the candidate -- a cheap structural
    prefilter before spending an LLM call, same shape as every other
    file-overlap gate in this module. No reasoner means no LLM-judged veto
    is possible (same rule the structural-only gates already follow), so
    this returns None.
    """
    if reasoner is None:
        return None
    candidate_files_set = set(candidate_files)
    for worker in workers:
        if not (candidate_files_set & set(worker.files)):
            continue
        worker_summary = (
            worker.routing_summary or "; ".join(worker.responsibilities[:2]) or worker.name
        )
        if reasoner.assess_interface_subsumption(
            interface_responsibility=interface_responsibility,
            existing_worker_id=worker.id,
            existing_worker_summary=worker_summary,
        ):
            return worker.id
    return None


def _is_worker_healthy(
    worker_id: str,
    routes_by_worker: dict[str, list[MemoryRoute]],
    min_routes: int,
    healthy_ratio: float,
) -> bool:
    """A worker counts as "already working well enough to leave alone" once
    it has at least `min_routes` recorded route OCCURRENCES and at least
    `healthy_ratio` of them are flagged is_high_quality. With fewer than
    `min_routes` occurrences there isn't enough evidence to call it either
    way, so this returns False (not healthy, not exempt) rather than True
    -- a worker with no track record yet should not be shielded from
    evolution just because it also hasn't failed yet.

    Counts occurrence_count, not row count (see MemoryRoute.occurrence_count
    and ColonyMemoryStore.save_route's own docstring -- route consolidation
    means a worker recruited for the same recurring pattern 10 times now
    stores as ONE row with occurrence_count=10, not 10 rows; every
    threshold here must weigh by occurrence_count or consolidation would
    silently starve every route-count-based gate in this module).
    """
    routes = routes_by_worker.get(worker_id, [])
    total = sum(route.occurrence_count for route in routes)
    if total < min_routes:
        return False
    healthy = sum(route.occurrence_count for route in routes if route.is_high_quality)
    return healthy / total >= healthy_ratio


def _mostly_negative_presence(routes: list[MemoryRoute], gate_ratio: float) -> bool:
    """True once most of a worker's *typed* struggling routes are
    need_type="negative_presence" -- the answer genuinely isn't in this
    repo (confirmed directly: a qibo question about a tool that doesn't
    exist in that codebase, a seaborn question about doc-build performance
    with no matching implementation to point at). Reorganizing territory
    boundaries cannot fix "the information was never here" -- specializing
    a worker whose struggles are dominated by this need_type would spend an
    evolution cycle on a problem this mechanism has no power over.

    Only routes with a known need_type count (the task-level aggregate
    route, and any route predating this field, carry ""); with none, there
    is nothing to judge, so this returns False rather than gating on an
    empty sample. Weighs by occurrence_count, not row count -- see
    _is_worker_healthy's own docstring on why every threshold here must
    (route consolidation, Phase 6).
    """
    typed = [route for route in routes if route.need_type]
    if not typed:
        return False
    total = sum(route.occurrence_count for route in typed)
    negative = sum(
        route.occurrence_count for route in typed if route.need_type == "negative_presence"
    )
    return negative / total >= gate_ratio


def _subdirectory_groups(worker: WorkerCard) -> dict[str, list[str]]:
    root_parts = Path(worker.root).parts if worker.root else ()
    groups: dict[str, list[str]] = defaultdict(list)
    for file in worker.files:
        remainder = Path(file).parts[len(root_parts) :]
        key = worker.root if len(remainder) <= 1 else "/".join([*root_parts, remainder[0]])
        groups[key].append(file)
    return dict(groups)


def _fold_colliding_groups(
    groups: dict[str, list[str]], worker: WorkerCard, all_workers: list[WorkerCard]
) -> dict[str, list[str]]:
    """A candidate child's id (`worker-{_slug(group)}`) can collide with an
    unrelated, already-existing worker -- confirmed live on sphinx:
    worker-sphinx's own file list contained a single stray file under
    sphinx/locale/, a directory already fully owned by a separate,
    pre-existing 67-file worker-sphinx-locale (a `_merge_tiny_groups`
    artifact from initial indexing -- a lone file that natural-roots to a
    subdirectory another territory already fully claims can still end up
    counted under a coarser worker's own file list). Specializing
    worker-sphinx then tried to create a second, wrong-scope (1-file)
    worker-sphinx-locale, crashing IndexStore.save's UNIQUE constraint on
    territories.id -- the first time specialize had ever actually fired
    against real accumulated route data, so this path had never been
    exercised live before.

    Rather than drop a colliding group's files (silently orphaning them --
    no worker would own them after `worker` itself is replaced by its
    children) or crash, folds them into `groups[worker.root]` -- the
    group `_subdirectory_groups` already uses for a file that isn't under
    any further subdirectory, i.e. this specialize pass's own natural
    "residual" bucket for `worker`.
    """
    other_ids = {other.id for other in all_workers if other.id != worker.id}
    colliding = [group for group in groups if f"worker-{_slug(group)}" in other_ids]
    if not colliding:
        return groups
    folded = dict(groups)
    for group in colliding:
        folded.setdefault(worker.root, []).extend(folded.pop(group))
    return folded


def _semantic_groups(
    worker: WorkerCard,
    worker_routes: list[MemoryRoute],
    repo_root: Path,
    min_group_routes: int,
    similarity_threshold: float,
    min_file_support: int,
) -> dict[str, list[str]]:
    """Fallback for _subdirectory_groups when a worker's territory has no
    usable subdirectory structure (e.g. yt-dlp's yt_dlp/extractor/, 1010
    files flat in one directory -- _subdirectory_groups can only ever
    return one group for it, so specialization could never trigger there
    no matter how much route history accumulated).

    Two separate responsibilities, deliberately not conflated:
    1. Route history says WHAT keeps recurring: cluster this worker's own
       routes by embedding similarity over need_terms (connected
       components over a cosine-similarity graph, same pattern as
       ant.coordinator.local._cluster_pending_proposals) -- a cluster
       that doesn't clear min_group_routes isn't a real recurring niche,
       just noise.
    2. CURRENT full-territory retrieval says WHERE that workload lands
       NOW: for each qualifying cluster, run search()/dense_search()
       (the same uncapped machinery a worker's own AutonomousWorker.run()
       already uses once assigned -- no per-card term cap applies here)
       once per route in the cluster against this worker's own complete
       file list, and keep only files with RECURRING support -- hit by
       at least `min_file_support` of the cluster's own routes, not just
       whichever files one route's own top hit happened to surface.

    Deliberately does not read or store historical evidence paths (see
    this session's own design discussion): a route's need_terms describe
    a recurring WORKLOAD, but where that workload's answer lives is a
    property of the current repo checkout, not a historical fact worth
    persisting -- code moves, and a live query stays correct as the
    codebase evolves in a way a stored path list would not.

    Returns {} (never a single-group dict) whenever there's nothing
    genuinely separable: no shared embedder, fewer than two distinct
    routes worth clustering, or fewer than two clusters clear both the
    route-count and file-support bars -- the same "no split" signal
    _subdirectory_groups gives by returning a dict of length < 2.

    Route-count bars here are weighed by occurrence_count, not row count
    -- see _is_worker_healthy's own docstring on why (route consolidation,
    Phase 6: a big recurring cluster can now be a single row with a high
    occurrence_count instead of many rows).
    """
    embedder = get_shared_embedder()
    total_route_occurrences = sum(route.occurrence_count for route in worker_routes)
    if embedder is None or total_route_occurrences < min_group_routes * 2:
        return {}

    clusters = _cluster_routes_by_need_terms(worker_routes, embedder, similarity_threshold)
    qualifying = [
        cluster
        for cluster in clusters
        if sum(route.occurrence_count for route in cluster) >= min_group_routes
    ]
    if len(qualifying) < 2:
        return {}

    tools = LocalSearchTool(repo_root)
    groups: dict[str, list[str]] = {}
    for cluster in qualifying:
        support: Counter[str] = Counter()
        for route in cluster:
            query = " ".join(route.need_terms)
            if not query.strip():
                continue
            hit_paths = {evidence.path for evidence in tools.search(query, worker.files, limit=6)}
            hit_paths |= {
                evidence.path for evidence in tools.dense_search(query, worker.files, limit=6)
            }
            # Weighted by occurrence_count, not a flat +1 per row -- see
            # _is_worker_healthy's own docstring on why (route
            # consolidation, Phase 6: a route this worker was actually
            # recruited for N times is N routes' worth of support for
            # whatever files its own query retrieves, even though it is
            # now one row, not N).
            for path in hit_paths:
                support[path] += route.occurrence_count
        supported_files = sorted(
            path for path, count in support.items() if count >= min_file_support
        )
        if not supported_files:
            continue
        label = _cluster_representative_query(cluster)
        group_key = f"{worker.root}/{_slug(label)}" if worker.root else _slug(label)
        groups[group_key] = supported_files
    return groups


def _cluster_routes_by_need_terms(
    routes: list[MemoryRoute], embedder: DenseEmbedder, similarity_threshold: float
) -> list[list[MemoryRoute]]:
    """Connected components over a cosine-similarity graph of each route's
    need_terms -- same deterministic-clustering shape as
    ant.coordinator.local._cluster_pending_proposals, just over
    MemoryRoute.need_terms instead of ProposedNode.need text. A route with
    no need_terms at all can't be embedded meaningfully and is skipped.
    """
    embeddable = [route for route in routes if route.need_terms]
    if len(embeddable) < 2:
        return [[route] for route in embeddable]

    texts = [" ".join(route.need_terms) for route in embeddable]
    vectors = embedder.embed(texts)
    if not vectors:
        return [[route] for route in embeddable]

    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = array / norms
    similarity = normed @ normed.T

    n = len(embeddable)
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
            if similarity[i, j] >= similarity_threshold:
                union(i, j)

    members_by_root: dict[int, list[MemoryRoute]] = defaultdict(list)
    for i, route in enumerate(embeddable):
        members_by_root[find(i)].append(route)
    return list(members_by_root.values())


def _cluster_representative_query(cluster: list[MemoryRoute]) -> str:
    """Deterministic, non-LLM label for a semantic cluster: the terms
    shared across the most of its own routes, not every route's full
    need_terms concatenated together (which balloons into a long, noisy
    query as a cluster grows). Used both as the retrieval query fed to
    search()/dense_search() for routes without their own usable
    need_terms and as this cluster's group-key slug.
    """
    counts: Counter[str] = Counter()
    for route in cluster:
        counts.update({term.lower() for term in route.need_terms})
    top_terms = [term for term, _ in counts.most_common(6)]
    return " ".join(top_terms) if top_terms else "cluster"


def _assign_route_to_group(route: MemoryRoute, groups: dict[str, list[str]]) -> str | None:
    need_terms = {term.lower() for term in route.need_terms}
    if not need_terms:
        return None
    best_group: str | None = None
    best_score = 0
    for group, files in groups.items():
        score = len(need_terms & _path_terms(files))
        if score > best_score:
            best_group, best_score = group, score
    return best_group


def _child_worker(
    worker: WorkerCard,
    group: str,
    files: list[str],
    route_count: int,
    repo_root: Path | None,
    reasoner: EvolutionReasoner | None = None,
    generation: int = 0,
) -> WorkerCard:
    territory_id = _slug(group)
    provenance = (
        f"Specialized from {worker.id} after {route_count} recurring needs "
        "concentrated on this substructure."
    )
    lineage = {
        "parent_worker_ids": [worker.id],
        "structural_action": "specialize",
        "generation_created": generation,
        "lifecycle_state": "probationary",
    }
    if repo_root is not None:
        # Reuse the same real term-frequency/README/symbol extraction used
        # for a repo's *initial* worker cards (see generate_worker_cards),
        # instead of a static templated sentence. Confirmed this mattered
        # in practice: the reasoner-driven routing added this session
        # (select_workers) decides who to recruit by reading a worker
        # card's responsibilities/terms -- a specialized child whose card
        # says nothing but "specialized after N needs" gives that judgment
        # call almost nothing to go on, degrading routing quality
        # specifically for the workers evolution just created.
        territory = Territory(id=territory_id, root=group, files=sorted(files), summary=provenance)
        # build_worker_cards already fills routing_summary with the
        # zero-cost template; re-derive via the reasoner when one is
        # available so a specialized child's routing_summary gets the same
        # LLM-quality treatment as its should_specialize judgment just did.
        child = build_worker_cards(repo_root, [territory])[0].model_copy(update=lineage)
        return _with_routing_summary(child, reasoner)
    # Fallback when no repo_root is supplied (existing tests, and any
    # caller that hasn't opted in): the original mechanical derivation from
    # already-indexed worker-card metadata only, no filesystem access.
    child_symbols = [symbol for symbol in worker.symbols if symbol.path in files]
    path_terms = _path_terms(files)
    inherited_terms = {term for term in worker.searchable_terms if term.lower() in path_terms}
    symbol_terms = {symbol.name for symbol in child_symbols} | {
        symbol.qualname for symbol in child_symbols if symbol.qualname
    }
    terms = sorted(inherited_terms | symbol_terms) or sorted(path_terms)
    return _with_routing_summary(
        WorkerCard(
            id=f"worker-{territory_id}",
            territory_id=territory_id,
            name=f"{group or 'root'} worker",
            root=group,
            responsibilities=[provenance],
            searchable_terms=terms[:32],
            files=sorted(files),
            symbols=child_symbols,
            **lineage,
        ),
        reasoner,
    )


def _path_terms(files: list[str]) -> set[str]:
    return {token.lower() for file in files for token in TOKEN_RE.findall(file)}


def _slug(value: str) -> str:
    normalized = "".join(character if character.isalnum() else "-" for character in value.lower())
    return normalized.strip("-") or "root"
