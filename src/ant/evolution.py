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


def evolve_workers(index_path: Path, min_coalition_count: int = 2) -> EvolutionResult:
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
