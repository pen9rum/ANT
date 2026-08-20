from pathlib import Path

from ant.coordinator import LocalCoordinator
from ant.domain import WorkerCard


def test_local_coordinator_returns_grounded_evidence(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text(
        "def authenticate_user():\n    return True\n",
        encoding="utf-8",
    )
    worker = WorkerCard(
        id="worker-src",
        territory_id="src",
        name="src worker",
        root="src",
        responsibilities=["Owns source files."],
        searchable_terms=["authenticate", "user"],
        files=["src/auth.py"],
    )

    state = LocalCoordinator(tmp_path, [worker]).ask("Where is authenticate handled?")

    assert state.has_evidence()
    assert state.evidence[0].path == "src/auth.py"
    assert state.rounds
    assert state.rounds[0].selected_worker_ids == ["worker-src"]


def test_local_coordinator_records_unresolved_needs(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("Release notes only\n", encoding="utf-8")
    worker = WorkerCard(
        id="worker-docs",
        territory_id="docs",
        name="docs worker",
        root="docs",
        responsibilities=["Owns docs."],
        searchable_terms=["release"],
        files=["docs/guide.md"],
    )

    state = LocalCoordinator(tmp_path, [worker]).ask("authentication pipeline", max_rounds=1)

    assert not state.has_evidence()
    assert state.unresolved_needs
    assert state.rounds[0].observations[0].worker_id == "worker-docs"
