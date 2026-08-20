from pathlib import Path

from ant.domain import EvidenceState, Territory, WorkerCard
from ant.memory import IndexStore


def test_index_store_persists_workers_and_traces(tmp_path: Path) -> None:
    territory = Territory(id="src", root="src", files=["src/app.py"], summary="Owns source.")
    worker = WorkerCard(
        id="worker-src",
        territory_id="src",
        name="src worker",
        root="src",
        files=["src/app.py"],
    )
    store = IndexStore(tmp_path / ".ant")

    store.save([territory], [worker])
    trace_id = store.save_trace(EvidenceState(question="Where is app?"))

    assert store.load_workers() == [worker]
    assert trace_id == 1
    assert (tmp_path / ".ant" / "ant.sqlite3").exists()
