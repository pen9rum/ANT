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
