from pathlib import Path

from ant.coordinator import LocalCoordinator
from ant.domain import WorkerCard
from ant.tools import LocalSearchTool


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


def test_local_search_returns_context_windows(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "coordinator.py").write_text(
        "\n".join(
            [
                "class LocalCoordinator:",
                "    def ask(self):",
                "        selected = self._select_workers('query')",
                "        return selected",
                "",
                "    def _select_workers(self, query):",
                "        return []",
            ]
        ),
        encoding="utf-8",
    )

    evidence = LocalSearchTool(tmp_path).search(
        "Where is worker selection handled?",
        ["src/coordinator.py"],
        limit=1,
        context_lines=2,
    )

    assert evidence[0].line_end > evidence[0].line_start
    assert "def _select_workers" in evidence[0].quote or "selected =" in evidence[0].quote


def test_local_search_can_navigate_to_definition_block(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "coordinator.py").write_text(
        "\n".join(
            [
                "class LocalCoordinator:",
                "    def ask(self):",
                "        return self._select_workers('query')",
                "",
                "    def _select_workers(self, query):",
                "        selected = []",
                "        return selected",
            ]
        ),
        encoding="utf-8",
    )

    evidence = LocalSearchTool(tmp_path).navigate("_select_workers", ["src/coordinator.py"])

    assert evidence
    assert "def _select_workers" in evidence[0].quote
    assert "return selected" in evidence[0].quote


def test_local_search_finds_references(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "model.py").write_text(
        "class QAOA:\n    pass\n\nqaoa = QAOA()\n",
        encoding="utf-8",
    )

    evidence = LocalSearchTool(tmp_path).references("QAOA", ["src/model.py"])

    assert evidence
    assert any("QAOA()" in item.quote for item in evidence)


def test_local_search_finds_callers_callees_and_assignments(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "model.py").write_text(
        "\n".join(
            [
                "def helper():",
                "    return 1",
                "",
                "def target():",
                "    value = helper()",
                "    return value",
                "",
                "def caller():",
                "    return target()",
            ]
        ),
        encoding="utf-8",
    )
    tool = LocalSearchTool(tmp_path)

    assert any("def caller" in item.quote for item in tool.callers("target", ["src/model.py"]))
    assert any("def helper" in item.quote for item in tool.callees("target", ["src/model.py"]))
    assert any(
        "value = helper()" in item.quote
        for item in tool.assignments("value", ["src/model.py"])
    )
