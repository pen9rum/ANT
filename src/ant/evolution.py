from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ant.domain import Territory, WorkerCard
from ant.memory import ColonyMemoryStore, IndexStore


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
    min_coalition_count: int = 2,
    retire_empty: bool = True,
    merge_overlap: float = 0.9,
) -> EvolutionResult:
    store = IndexStore(index_path)
    memory = ColonyMemoryStore(index_path)
    workers = store.load_workers()
    worker_by_id = {worker.id: worker for worker in workers}
    events: list[EvolutionEvent] = []

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
        workers = kept

    workers, merge_events = _merge_overlapping_workers(workers, threshold=merge_overlap)
    events.extend(merge_events)

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
