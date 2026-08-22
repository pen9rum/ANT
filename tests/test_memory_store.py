from pathlib import Path

from typer.testing import CliRunner

from ant.cli import app
from ant.domain import CodeSymbol, EvidenceState, Territory, WorkerCard
from ant.memory import ColonyMemoryStore, IndexStore, MemoryRoute


def test_index_store_persists_workers_and_traces(tmp_path: Path) -> None:
    territory = Territory(id="src", root="src", files=["src/app.py"], summary="Owns source.")
    worker = WorkerCard(
        id="worker-src",
        territory_id="src",
        name="src worker",
        root="src",
        files=["src/app.py"],
        symbols=[
            CodeSymbol(
                name="App",
                kind="class",
                path="src/app.py",
                line=1,
                qualname="App",
            )
        ],
    )
    store = IndexStore(tmp_path / ".ant")

    store.save([territory], [worker])
    trace_id = store.save_trace(EvidenceState(question="Where is app?"))

    assert store.load_workers() == [worker]
    assert trace_id == 1
    assert (tmp_path / ".ant" / "ant.sqlite3").exists()
    assert (tmp_path / ".ant" / "symbols.json").exists()
    assert store.load_symbols()[0].name == "App"

    result = CliRunner().invoke(
        app, ["symbols", "--index", str(tmp_path / ".ant"), "--query", "App"]
    )
    assert result.exit_code == 0
    assert '"worker_id": "worker-src"' in result.stdout


def test_colony_memory_returns_matching_routes(tmp_path: Path) -> None:
    memory = ColonyMemoryStore(tmp_path)
    memory.save_route(
        MemoryRoute(
            need_terms=["measurement", "sample_shots"],
            worker_ids=["worker-backends"],
            weight=2.5,
        )
    )
    memory.save_route(
        MemoryRoute(
            need_terms=["draw"],
            worker_ids=["worker-models"],
            weight=3.0,
        )
    )

    routes = memory.matching_routes(["How", "measurement", "works"])

    assert len(routes) == 1
    assert routes[0].worker_ids == ["worker-backends"]
