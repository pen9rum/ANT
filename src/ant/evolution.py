from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from ant.domain import Territory, WorkerCard
from ant.memory import ColonyMemoryStore, IndexStore
from ant.memory.colony import MemoryRoute
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
    min_coalition_count: int = DEFAULT_SCORING_CONFIG.evolution.min_coalition_count,
    retire_empty: bool = True,
    merge_overlap: float = DEFAULT_SCORING_CONFIG.evolution.merge_overlap_threshold,
    min_specialization_routes: int = DEFAULT_SCORING_CONFIG.evolution.min_specialization_routes,
    min_specialization_group_routes: int = (
        DEFAULT_SCORING_CONFIG.evolution.min_specialization_group_routes
    ),
) -> EvolutionResult:
    store = IndexStore(index_path)
    memory = ColonyMemoryStore(index_path)
    workers = store.load_workers()
    events: list[EvolutionEvent] = []
    removed_worker_ids: set[str] = set()

    workers, specialize_events = _specialize_overloaded_workers(
        workers,
        memory.all_routes(),
        min_routes=min_specialization_routes,
        min_group_routes=min_specialization_group_routes,
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

    workers, merge_events = _merge_overlapping_workers(workers, threshold=merge_overlap)
    events.extend(merge_events)
    removed_worker_ids.update(
        worker_id for event in merge_events for worker_id in event.source_worker_ids
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


def _merge_overlapping_workers(
    workers: list[WorkerCard],
    threshold: float,
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
            other_files = set(other.files)
            union = worker_files | other_files
            if not union:
                continue
            overlap = len(worker_files & other_files) / len(union)
            if overlap >= threshold:
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
    routes: list[MemoryRoute],
    min_routes: int,
    min_group_routes: int,
) -> tuple[list[WorkerCard], list[EvolutionEvent]]:
    """Split a coarse worker into finer workers along its existing directory
    substructure once recurring routes show it is being asked about at least
    two genuinely different sub-areas often enough. This is the "Specialization
    / Birth" mechanism from the design: worker count is not fixed up front, it
    grows where task experience shows a single worker is covering topics that
    do not actually belong together.
    """
    routes_by_worker: dict[str, list[MemoryRoute]] = defaultdict(list)
    for route in routes:
        for worker_id in route.worker_ids:
            routes_by_worker[worker_id].append(route)

    events: list[EvolutionEvent] = []
    result: list[WorkerCard] = []
    for worker in workers:
        groups = _subdirectory_groups(worker)
        worker_routes = routes_by_worker.get(worker.id, [])
        if len(groups) < 2 or len(worker_routes) < min_routes:
            result.append(worker)
            continue

        group_counts: dict[str, int] = defaultdict(int)
        for route in worker_routes:
            group = _assign_route_to_group(route, groups)
            if group:
                group_counts[group] += 1
        qualifying_groups = {
            group for group, count in group_counts.items() if count >= min_group_routes
        }
        if len(qualifying_groups) < 2:
            result.append(worker)
            continue

        children = [
            _child_worker(worker, group, files, group_counts.get(group, 0))
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


def _child_worker(worker: WorkerCard, group: str, files: list[str], route_count: int) -> WorkerCard:
    child_symbols = [symbol for symbol in worker.symbols if symbol.path in files]
    path_terms = _path_terms(files)
    inherited_terms = {term for term in worker.searchable_terms if term.lower() in path_terms}
    symbol_terms = {symbol.name for symbol in child_symbols} | {
        symbol.qualname for symbol in child_symbols if symbol.qualname
    }
    terms = sorted(inherited_terms | symbol_terms) or sorted(path_terms)
    territory_id = _slug(group)
    return WorkerCard(
        id=f"worker-{territory_id}",
        territory_id=territory_id,
        name=f"{group or 'root'} worker",
        root=group,
        responsibilities=[
            f"Specialized from {worker.id} after {route_count} recurring needs "
            "concentrated on this substructure.",
        ],
        searchable_terms=terms[:32],
        files=sorted(files),
        symbols=child_symbols,
    )


def _path_terms(files: list[str]) -> set[str]:
    return {token.lower() for file in files for token in TOKEN_RE.findall(file)}


def _slug(value: str) -> str:
    normalized = "".join(character if character.isalnum() else "-" for character in value.lower())
    return normalized.strip("-") or "root"
