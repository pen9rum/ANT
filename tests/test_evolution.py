from pathlib import Path

from ant.domain import Territory, WorkerCard
from ant.evolution import evolve_workers
from ant.memory import CoalitionRecord, ColonyMemoryStore, IndexStore


def test_evolve_workers_births_bridge_from_recurring_coalition(tmp_path: Path) -> None:
    index_path = tmp_path / ".ant"
    workers = [
        WorkerCard(id="worker-a", territory_id="a", name="a", root="a", files=["a.py"]),
        WorkerCard(id="worker-b", territory_id="b", name="b", root="b", files=["b.py"]),
    ]
    territories = [
        Territory(id="a", root="a", files=["a.py"]),
        Territory(id="b", root="b", files=["b.py"]),
    ]
    IndexStore(index_path).save(territories, workers)
    memory = ColonyMemoryStore(index_path)
    memory.record_coalition(
        CoalitionRecord(
            worker_ids=["worker-a", "worker-b"],
            question="q1",
            evidence_count=2,
            unresolved_need_count=0,
        )
    )
    memory.record_coalition(
        CoalitionRecord(
            worker_ids=["worker-a", "worker-b"],
            question="q2",
            evidence_count=2,
            unresolved_need_count=0,
        )
    )

    result = evolve_workers(index_path, min_coalition_count=2)

    assert result.events[0].kind == "birth"
    stored_workers = IndexStore(index_path).load_workers()
    assert any(worker.id.startswith("worker-bridge") for worker in stored_workers)


def test_evolve_workers_retires_empty_and_merges_overlap(tmp_path: Path) -> None:
    index_path = tmp_path / ".ant"
    workers = [
        WorkerCard(id="worker-empty", territory_id="empty", name="empty", root="", files=[]),
        WorkerCard(id="worker-a", territory_id="a", name="a", root="a", files=["shared.py"]),
        WorkerCard(id="worker-b", territory_id="b", name="b", root="b", files=["shared.py"]),
    ]
    territories = [
        Territory(id=worker.territory_id, root=worker.root, files=worker.files)
        for worker in workers
    ]
    IndexStore(index_path).save(territories, workers)

    result = evolve_workers(index_path, min_coalition_count=99, merge_overlap=0.9)

    assert {event.kind for event in result.events} == {"retire", "merge"}
