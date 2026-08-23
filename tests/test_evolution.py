from pathlib import Path

from ant.domain import CodeSymbol, Territory, WorkerCard
from ant.evolution import evolve_workers
from ant.memory import CoalitionRecord, ColonyMemoryStore, IndexStore, MemoryRoute


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


def test_evolve_workers_specializes_worker_overloaded_with_diverse_needs(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / ".ant"
    worker = WorkerCard(
        id="worker-mixed",
        territory_id="mixed",
        name="mixed worker",
        root="pkg",
        searchable_terms=["auth", "billing", "service"],
        files=["pkg/auth/service.py", "pkg/billing/service.py"],
        symbols=[
            CodeSymbol(name="AuthService", kind="class", path="pkg/auth/service.py", line=1),
            CodeSymbol(name="BillingService", kind="class", path="pkg/billing/service.py", line=1),
        ],
    )
    territories = [Territory(id="mixed", root="pkg", files=worker.files)]
    IndexStore(index_path).save(territories, [worker])
    memory = ColonyMemoryStore(index_path)
    for _ in range(2):
        memory.save_route(
            MemoryRoute(need_terms=["auth"], worker_ids=["worker-mixed"], weight=2.0)
        )
    for _ in range(2):
        memory.save_route(
            MemoryRoute(need_terms=["billing"], worker_ids=["worker-mixed"], weight=2.0)
        )

    result = evolve_workers(
        index_path,
        min_coalition_count=99,
        retire_empty=False,
        merge_overlap=0.99,
        min_specialization_routes=4,
        min_specialization_group_routes=2,
    )

    specialize_events = [event for event in result.events if event.kind == "specialize"]
    assert {event.worker_id for event in specialize_events} == {
        "worker-pkg-auth",
        "worker-pkg-billing",
    }
    assert all(event.source_worker_ids == ["worker-mixed"] for event in specialize_events)

    stored_workers = {worker.id: worker for worker in IndexStore(index_path).load_workers()}
    assert "worker-mixed" not in stored_workers
    auth_worker = stored_workers["worker-pkg-auth"]
    billing_worker = stored_workers["worker-pkg-billing"]
    assert auth_worker.files == ["pkg/auth/service.py"]
    assert [symbol.name for symbol in auth_worker.symbols] == ["AuthService"]
    assert billing_worker.files == ["pkg/billing/service.py"]
    assert [symbol.name for symbol in billing_worker.symbols] == ["BillingService"]

    # The old worker id is now stale: any memory that pointed at it should not
    # be served until it is revalidated (and it will be discarded then, since
    # worker-mixed no longer exists in the colony).
    assert memory.matching_routes(["auth"]) == []
    current_worker_ids = {worker.id for worker in IndexStore(index_path).load_workers()}
    outcome = memory.revalidate_stale(current_worker_ids)
    assert outcome["discarded"] == 4


def test_evolve_workers_does_not_specialize_without_enough_route_diversity(
    tmp_path: Path,
) -> None:
    index_path = tmp_path / ".ant"
    worker = WorkerCard(
        id="worker-mixed",
        territory_id="mixed",
        name="mixed worker",
        root="pkg",
        files=["pkg/auth/service.py", "pkg/billing/service.py"],
    )
    territories = [Territory(id="mixed", root="pkg", files=worker.files)]
    IndexStore(index_path).save(territories, [worker])
    memory = ColonyMemoryStore(index_path)
    for _ in range(4):
        memory.save_route(
            MemoryRoute(need_terms=["auth"], worker_ids=["worker-mixed"], weight=2.0)
        )

    result = evolve_workers(
        index_path,
        min_coalition_count=99,
        retire_empty=False,
        merge_overlap=0.99,
        min_specialization_routes=4,
        min_specialization_group_routes=2,
    )

    assert not [event for event in result.events if event.kind == "specialize"]
    stored_workers = {worker.id for worker in IndexStore(index_path).load_workers()}
    assert stored_workers == {"worker-mixed"}
