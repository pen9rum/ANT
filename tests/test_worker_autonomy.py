from pathlib import Path

from ant.domain import WorkerCard
from ant.retrieval import BM25Index
from ant.tools import LocalSearchTool
from ant.workers import AutonomousWorker, WorkerRunConfig


def test_bm25_ranks_matching_document() -> None:
    index = BM25Index(
        [
            ("a", "quantum algorithm qaoa"),
            ("b", "database connection pool"),
        ]
    )

    assert index.search(["qaoa"], limit=1)[0][1] == "a"


def test_autonomous_worker_records_tool_actions(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "model.py").write_text(
        "\n".join(
            [
                "class QAOA:",
                "    def minimize(self):",
                "        return None",
                "",
                "class FALQON(QAOA):",
                "    def minimize(self):",
                "        return super().minimize()",
            ]
        ),
        encoding="utf-8",
    )
    card = WorkerCard(
        id="worker-src",
        territory_id="src",
        name="src worker",
        root="src",
        searchable_terms=["qaoa", "falqon", "minimize"],
        files=["src/model.py"],
    )

    observation = AutonomousWorker(tmp_path, card, LocalSearchTool(tmp_path)).run(
        "What subclasses inherit from QAOA?",
        WorkerRunConfig(max_tool_calls=6, evidence_limit=6),
    )

    assert observation.evidence
    assert [action.tool for action in observation.actions][:2] == ["search", "navigate"]
    assert any("FALQON" in item.quote for item in observation.evidence)


def test_budget_exhaustion_creates_unresolved_need(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "backend.py").write_text(
        "class Backend:\n    def measurement(self):\n        return None\n",
        encoding="utf-8",
    )
    card = WorkerCard(
        id="worker-src",
        territory_id="src",
        name="src worker",
        root="src",
        searchable_terms=["backend", "measurement"],
        files=["src/backend.py"],
    )

    observation = AutonomousWorker(tmp_path, card, LocalSearchTool(tmp_path)).run(
        "Where does measurement data flow to final outcomes?",
        WorkerRunConfig(max_tool_calls=1, evidence_limit=4),
    )

    assert observation.stop_reason == "budget_exhausted"
    assert observation.unresolved_needs


def test_worker_ranks_repo_symbols_above_question_words(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "backend.py").write_text(
        "class Backend:\n    def measurement(self):\n        return None\n",
        encoding="utf-8",
    )
    card = WorkerCard(
        id="worker-src",
        territory_id="src",
        name="src worker",
        root="src",
        searchable_terms=["backend", "measurement"],
        files=["src/backend.py"],
    )

    observation = AutonomousWorker(tmp_path, card, LocalSearchTool(tmp_path)).run(
        "Where does Backend measurement happen?",
        WorkerRunConfig(max_tool_calls=10, evidence_limit=4),
    )

    navigated = [action.query for action in observation.actions if action.tool == "navigate"]
    assert "Backend" in navigated
    assert "Where" not in navigated
