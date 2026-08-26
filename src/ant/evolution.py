from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from ant.domain import Territory, WorkerCard
from ant.indexing.cards import build_worker_cards
from ant.memory import ColonyMemoryStore, IndexStore
from ant.memory.colony import MemoryRoute
from ant.providers import EvolutionReasoner
from ant.scoring_config import DEFAULT_SCORING_CONFIG

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


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
) -> EvolutionResult:
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
    )
    events.extend(specialize_events)
    removed_worker_ids.update(
        worker_id for event in specialize_events for worker_id in event.source_worker_ids
    )

    worker_by_id = {worker.id: worker for worker in workers}
    for worker_ids, count in memory.recurring_coalitions(min_count=min_coalition_count):
        bridge_suffix = "-".join(worker_id.removeprefix("worker-") for worker_id in worker_ids)
        bridge_id = f"worker-bridge-{bridge_suffix}"
        if bridge_id in worker_by_id:
            continue
        source_workers = [
            worker_by_id[worker_id] for worker_id in worker_ids if worker_id in worker_by_id
        ]
        if len(source_workers) < 2:
            continue
        if all(
            _is_worker_healthy(
                worker.id, routes_by_worker, min_routes_for_health_check, healthy_route_ratio
            )
            for worker in source_workers
        ):
            # Every worker in this recurring coalition already has a track
            # record of good answers -- birthing a dedicated bridge worker
            # is for coalitions that are *struggling* together, not ones
            # that already work fine split apart.
            continue
        files = sorted({file for worker in source_workers for file in worker.files})
        terms = sorted({term for worker in source_workers for term in worker.searchable_terms})[:32]
        bridge = WorkerCard(
            id=bridge_id,
            territory_id=bridge_id.removeprefix("worker-"),
            name=" + ".join(worker.name for worker in source_workers[:3]) + " bridge",
            root="",
            responsibilities=[
                "Cross-territory specialist born from recurring temporary coalition.",
                f"Coalition recurred {count} times.",
            ],
            searchable_terms=terms,
            files=files,
        )
        workers.append(bridge)
        worker_by_id[bridge.id] = bridge
        events.append(
            EvolutionEvent(
                kind="birth",
                worker_id=bridge.id,
                reason=f"Recurring coalition observed {count} times.",
                source_worker_ids=worker_ids,
            )
        )

    if retire_empty:
        kept = []
        for worker in workers:
            if worker.files:
                kept.append(worker)
                continue
            events.append(
                EvolutionEvent(
                    kind="retire",
                    worker_id=worker.id,
                    reason="Worker has no owned files after refresh.",
                )
            )
            removed_worker_ids.add(worker.id)
        workers = kept

    workers, merge_events = _merge_overlapping_workers(
        workers,
        threshold=merge_overlap,
        routes_by_worker=routes_by_worker,
        min_health_routes=min_routes_for_health_check,
        healthy_ratio=healthy_route_ratio,
        reasoner=reasoner,
    )
    events.extend(merge_events)
    removed_worker_ids.update(
        worker_id for event in merge_events for worker_id in event.source_worker_ids
    )

    if reasoner is not None:
        workers, episode_events = _apply_episode_actions(
            workers, memory, reasoner, min_episode_count
        )
        events.extend(episode_events)
        removed_worker_ids.update(
            worker_id for event in episode_events for worker_id in event.source_worker_ids
        )

    territories = [
        Territory(
            id=worker.territory_id,
            root=worker.root,
            files=worker.files,
            summary="Synthetic territory from current worker directory.",
        )
        for worker in workers
    ]
    if events:
        store.save(territories, workers)
    if removed_worker_ids:
        ColonyMemoryStore(index_path).mark_stale(
            sorted(removed_worker_ids),
            reason="Worker retired/specialized/merged away by colony evolution.",
        )
    return EvolutionResult(events=events, worker_count=len(workers))


def _apply_episode_actions(
    workers: list[WorkerCard],
    memory: ColonyMemoryStore,
    reasoner: EvolutionReasoner,
    min_episode_count: int,
) -> tuple[list[WorkerCard], list[EvolutionEvent]]:
    """Acts on ColonyMemoryStore.aggregate_episodes' recurring (strategy,
    worker-set) patterns -- e.g. "temporary_bridge kept resolving a
    proxy-validation-shaped need across 3 separate tasks" -- via
    EvolutionReasoner.decide_episode_action. This is a richer signal than
    recurring_coalitions (which only sees raw worker co-occurrence, not
    which specific temporary adaptation actually worked or how often): the
    same slow, cross-task timescale as the rest of evolve_workers, never
    reacting to a single task's own outcome.
    """
    worker_by_id = {worker.id: worker for worker in workers}
    events: list[EvolutionEvent] = []
    consumed: set[str] = set()
    for aggregate in memory.aggregate_episodes(min_count=min_episode_count):
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
            workers=aggregate.workers,
        )
        reason = (
            f"Episode-driven: '{aggregate.strategy}' succeeded in "
            f"{aggregate.successes}/{aggregate.occurrences} recorded tasks "
            f"(total evidence gain {aggregate.total_evidence_gain})."
        )
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
            bridge_suffix = "-".join(
                worker_id.removeprefix("worker-") for worker_id in aggregate.workers
            )
            bridge_id = f"worker-bridge-{bridge_suffix}"
            if bridge_id in worker_by_id:
                continue
            files = sorted({file for worker in source_workers for file in worker.files})
            terms = sorted(
                {term for worker in source_workers for term in worker.searchable_terms}
            )[:32]
            bridge = WorkerCard(
                id=bridge_id,
                territory_id=bridge_id.removeprefix("worker-"),
                name=" + ".join(worker.name for worker in source_workers[:3]) + " bridge",
                root="",
                responsibilities=[
                    "Permanent bridge born from a recurring successful temporary adaptation.",
                    reason,
                ],
                searchable_terms=terms,
                files=files,
            )
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
            merged_worker = WorkerCard(
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
            )
            workers = [w for w in workers if w.id not in (first.id, second.id)]
            workers.append(merged_worker)
            worker_by_id = {w.id: w for w in workers}
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


def _merge_overlapping_workers(
    workers: list[WorkerCard],
    threshold: float,
    routes_by_worker: dict[str, list[MemoryRoute]],
    min_health_routes: int,
    healthy_ratio: float,
    reasoner: EvolutionReasoner | None = None,
) -> tuple[list[WorkerCard], list[EvolutionEvent]]:
    events: list[EvolutionEvent] = []
    consumed: set[str] = set()
    merged: list[WorkerCard] = []
    for index, worker in enumerate(workers):
        if worker.id in consumed:
            continue
        worker_files = set(worker.files)
        partner = None
        for other in workers[index + 1 :]:
            if other.id in consumed:
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
            if not union:
                continue
            overlap = len(worker_files & other_files) / len(union)
            if overlap < threshold:
                continue
            # File overlap alone cannot tell "same specialty duplicated
            # across two workers" apart from "coincidentally touches a lot
            # of shared files but is conceptually distinct" -- this is the
            # judgment on top of the structural overlap gate.
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
            merged.append(worker)
            continue
        consumed.update({worker.id, partner.id})
        merged_worker = WorkerCard(
            id=f"worker-merge-{worker.id.removeprefix('worker-')}-{partner.id.removeprefix('worker-')}",
            territory_id=f"merge-{worker.territory_id}-{partner.territory_id}",
            name=f"{worker.name} / {partner.name} merged",
            root=worker.root or partner.root,
            responsibilities=worker.responsibilities + partner.responsibilities,
            searchable_terms=sorted(set(worker.searchable_terms) | set(partner.searchable_terms)),
            files=sorted(worker_files | set(partner.files)),
        )
        merged.append(merged_worker)
        events.append(
            EvolutionEvent(
                kind="merge",
                worker_id=merged_worker.id,
                reason="Workers have highly overlapping file ownership.",
                source_worker_ids=[worker.id, partner.id],
            )
        )
    return merged, events


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
) -> tuple[list[WorkerCard], list[EvolutionEvent]]:
    """Split a coarse worker into finer workers along its existing directory
    substructure once recurring routes show it is being asked about at least
    two genuinely different sub-areas often enough. This is the "Specialization
    / Birth" mechanism from the design: worker count is not fixed up front, it
    grows where task experience shows a single worker is covering topics that
    do not actually belong together.
    """
    events: list[EvolutionEvent] = []
    result: list[WorkerCard] = []
    for worker in workers:
        groups = _subdirectory_groups(worker)
        worker_routes = routes_by_worker.get(worker.id, [])
        if (
            len(groups) < 2
            or len(worker_routes) < min_routes
            or _is_worker_healthy(worker.id, routes_by_worker, min_health_routes, healthy_ratio)
            or _mostly_negative_presence(worker_routes, negative_presence_gate_ratio)
        ):
            result.append(worker)
            continue

        group_counts: dict[str, int] = defaultdict(int)
        group_terms: dict[str, set[str]] = defaultdict(set)
        for route in worker_routes:
            group = _assign_route_to_group(route, groups)
            if group:
                group_counts[group] += 1
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
            _child_worker(worker, group, files, group_counts.get(group, 0), repo_root)
            for group, files in sorted(groups.items())
        ]
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


def _is_worker_healthy(
    worker_id: str,
    routes_by_worker: dict[str, list[MemoryRoute]],
    min_routes: int,
    healthy_ratio: float,
) -> bool:
    """A worker counts as "already working well enough to leave alone" once
    it has at least `min_routes` recorded routes and at least
    `healthy_ratio` of them are flagged is_high_quality. With fewer than
    `min_routes` routes there isn't enough evidence to call it either way,
    so this returns False (not healthy, not exempt) rather than True -- a
    worker with no track record yet should not be shielded from evolution
    just because it also hasn't failed yet.
    """
    routes = routes_by_worker.get(worker_id, [])
    if len(routes) < min_routes:
        return False
    healthy = sum(1 for route in routes if route.is_high_quality)
    return healthy / len(routes) >= healthy_ratio


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
    empty sample.
    """
    typed = [route for route in routes if route.need_type]
    if not typed:
        return False
    negative = sum(1 for route in typed if route.need_type == "negative_presence")
    return negative / len(typed) >= gate_ratio


def _subdirectory_groups(worker: WorkerCard) -> dict[str, list[str]]:
    root_parts = Path(worker.root).parts if worker.root else ()
    groups: dict[str, list[str]] = defaultdict(list)
    for file in worker.files:
        remainder = Path(file).parts[len(root_parts) :]
        key = worker.root if len(remainder) <= 1 else "/".join([*root_parts, remainder[0]])
        groups[key].append(file)
    return dict(groups)


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
) -> WorkerCard:
    territory_id = _slug(group)
    provenance = (
        f"Specialized from {worker.id} after {route_count} recurring needs "
        "concentrated on this substructure."
    )
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
        return build_worker_cards(repo_root, [territory])[0]
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
    return WorkerCard(
        id=f"worker-{territory_id}",
        territory_id=territory_id,
        name=f"{group or 'root'} worker",
        root=group,
        responsibilities=[provenance],
        searchable_terms=terms[:32],
        files=sorted(files),
        symbols=child_symbols,
    )


def _path_terms(files: list[str]) -> set[str]:
    return {token.lower() for file in files for token in TOKEN_RE.findall(file)}


def _slug(value: str) -> str:
    normalized = "".join(character if character.isalnum() else "-" for character in value.lower())
    return normalized.strip("-") or "root"
