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
    assert [action.tool for action in observation.actions][:3] == [
        "search",
        "dense_search",
        "subclasses",
    ]
    assert any("FALQON" in item.quote for item in observation.evidence)
    assert any(
        action.tool == "subclasses" and action.query == "QAOA"
        for action in observation.actions
    )


def test_local_search_resolves_symbols_and_subclasses_with_ast_index(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "models.py").write_text(
        "class QAOA:\n"
        "    pass\n\n"
        "class FALQON(QAOA):\n"
        "    def minimize(self):\n"
        "        return None\n\n"
        "def helper():\n"
        "    return FALQON()\n",
        encoding="utf-8",
    )
    tool = LocalSearchTool(tmp_path)

    definitions = tool.resolve_symbol("QAOA", ["src/models.py"])
    subclasses = tool.subclasses("QAOA", ["src/models.py"])
    callers = tool.indexed_callers("FALQON", ["src/models.py"])

    assert definitions[0].line_start == 1
    assert "class FALQON(QAOA)" in subclasses[0].quote
    assert subclasses[0].claim == "Defines class FALQON inheriting from QAOA."
    assert callers[0].symbols[:2] == ["helper", "helper"]


def test_local_search_resolves_import_to_definition(tmp_path: Path) -> None:
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "models.py").write_text(
        "class CircuitResult:\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "src" / "pkg" / "backend.py").write_text(
        "from .models import CircuitResult\n", encoding="utf-8"
    )
    files = ["src/pkg/models.py", "src/pkg/backend.py"]

    evidence = LocalSearchTool(tmp_path).resolve_import(
        "CircuitResult", "src/pkg/backend.py", files
    )

    assert evidence[0].path == "src/pkg/models.py"
    assert evidence[0].line_start == 1


def test_budget_exhaustion_creates_execution_diagnostic(tmp_path: Path) -> None:
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
    assert not observation.unresolved_needs
    assert observation.diagnostics[0].kind == "budget_exhausted"


def test_bm25_matches_terms_across_an_implementation_block(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "pipeline.py").write_text(
        "def produce_report():\n"
        "    measurement = collect()\n"
        "    return final_outcome(measurement)\n",
        encoding="utf-8",
    )

    evidence = LocalSearchTool(tmp_path).search(
        "measurement final_outcome", ["src/pipeline.py"], limit=1, context_lines=0
    )

    assert "measurement = collect()" in evidence[0].quote
    assert "final_outcome(measurement)" in evidence[0].quote


def test_assignments_include_downstream_data_flow_uses(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "pipeline.py").write_text(
        "def run():\n    value = collect()\n    result = transform(value)\n    return value\n",
        encoding="utf-8",
    )

    evidence = LocalSearchTool(tmp_path).assignments("value", ["src/pipeline.py"])

    assert any("transform(value)" in item.quote for item in evidence)
    assert all("data-flow" in item.reason for item in evidence)


def test_bm25_uses_method_region_instead_of_whole_class(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    filler = "\n".join(f"        filler_{index} = {index}" for index in range(40))
    (tmp_path / "src" / "backend.py").write_text(
        "class Backend:\n"
        "    def unrelated(self):\n"
        f"{filler}\n"
        "        return None\n\n"
        "    def sample_shots(self, probabilities):\n"
        "        return draw_samples(probabilities)\n",
        encoding="utf-8",
    )

    evidence = LocalSearchTool(tmp_path).search(
        "sample_shots probabilities", ["src/backend.py"], limit=1, context_lines=0
    )

    assert "def sample_shots" in evidence[0].quote
    assert "filler_0" not in evidence[0].quote


def test_symbol_ranking_is_conditioned_on_current_need(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "backend.py").write_text(
        "def set_threads():\n    pass\n\ndef sample_shots():\n    pass\n",
        encoding="utf-8",
    )
    tool = LocalSearchTool(tmp_path)

    ranked = tool.rank_symbols(
        ["set_threads", "sample_shots"],
        ["src/backend.py"],
        need="How do probabilities become sampled outcomes using sample_shots?",
    )

    assert ranked[0] == "sample_shots"


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


def test_worker_uses_call_and_data_flow_tools_for_flow_questions(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "backend.py").write_text(
        "from .states import CircuitResult\n\n"
        "def sample_shots(probabilities):\n"
        "    samples = draw_samples(probabilities)\n"
        "    return samples\n\n"
        "def execute_circuit():\n"
        "    probabilities = calculate_probabilities()\n"
        "    samples = sample_shots(probabilities)\n"
        "    return CircuitResult(samples)\n",
        encoding="utf-8",
    )
    card = WorkerCard(
        id="worker-backends",
        territory_id="backends",
        name="backend worker",
        root="src",
        searchable_terms=["sample_shots", "probabilities", "CircuitResult"],
        files=["src/backend.py"],
    )

    observation = AutonomousWorker(tmp_path, card, LocalSearchTool(tmp_path)).run(
        "How do probabilities flow into sampled measurement outcomes through sample_shots?",
        WorkerRunConfig(max_tool_calls=10, evidence_limit=8),
    )

    tools = [action.tool for action in observation.actions]
    assert "imports" in tools
    assert "callers" in tools
    assert "assignments" in tools
    assert any("execute_circuit" in item.quote for item in observation.evidence)
