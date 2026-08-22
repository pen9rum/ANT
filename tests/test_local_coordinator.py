from pathlib import Path

from ant.coordinator import LocalCoordinator
from ant.coordinator.local import _matches_term
from ant.domain import UnresolvedNeed, WorkerCard, WorkerObservation
from ant.memory import MemoryRoute
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
    assert state.rounds[0].routing_scores[0].worker_id == "worker-src"
    assert "authenticate" in state.rounds[0].routing_scores[0].query_hits


def test_routing_matcher_avoids_arbitrary_substrings() -> None:
    assert not _matches_term("into", {"quantum_info"})
    assert not _matches_term("fusedgate", {"gate"})
    assert _matches_term("sample", {"sample_shots"})
    assert _matches_term("fusedgate", {"fusedgate"})


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


class NeedRoutingReasoner:
    def observe(self, *, question, worker_id, territory_id, evidence):
        needs = []
        if worker_id == "worker-api":
            needs.append(
                UnresolvedNeed(
                    description="Need the persistence implementation.",
                    kind="missing_implementation",
                    suggested_terms=["persist_record"],
                    suggested_territories=["storage"],
                    scope="cross_territory",
                    source_worker_id="worker-api",
                )
            )
        return WorkerObservation(
            worker_id=worker_id, territory_id=territory_id, unresolved_needs=needs
        )


def test_semantic_need_routes_follow_up_and_forms_coalition(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    (tmp_path / "storage").mkdir()
    (tmp_path / "api" / "service.py").write_text(
        "def submit_record():\n    return persist_record()\n", encoding="utf-8"
    )
    (tmp_path / "storage" / "repo.py").write_text(
        "def persist_record():\n    return 'stored'\n", encoding="utf-8"
    )
    workers = [
        WorkerCard(
            id="worker-api",
            territory_id="api",
            name="api",
            root="api",
            searchable_terms=["submit"],
            files=["api/service.py"],
        ),
        WorkerCard(
            id="worker-storage",
            territory_id="storage",
            name="storage",
            root="storage",
            searchable_terms=["persist_record"],
            files=["storage/repo.py"],
        ),
    ]

    state = LocalCoordinator(tmp_path, workers, reasoner=NeedRoutingReasoner()).ask(
        "How does submit work?", max_rounds=2
    )

    assert state.rounds[0].selected_worker_ids == ["worker-api"]
    assert not state.rounds[0].coalition_formed
    assert state.rounds[1].input_need == "Need the persistence implementation."
    assert state.rounds[1].selected_worker_ids == ["worker-storage"]
    assert state.rounds[1].coalition_formed


def test_parallel_initial_recruitment_is_not_a_coalition(tmp_path: Path) -> None:
    for root in ("api", "ui"):
        (tmp_path / root).mkdir()
        (tmp_path / root / "feature.py").write_text(
            f"def shared_feature():\n    return '{root}'\n", encoding="utf-8"
        )
    workers = [
        WorkerCard(
            id=f"worker-{root}",
            territory_id=root,
            name=root,
            root=root,
            searchable_terms=["shared", "feature"],
            files=[f"{root}/feature.py"],
        )
        for root in ("api", "ui")
    ]

    state = LocalCoordinator(tmp_path, workers).ask("Where is shared feature?", max_rounds=1)

    assert len(state.rounds[0].selected_worker_ids) == 2
    assert not state.rounds[0].coalition_formed
    assert all(
        action.tool != "cross_check"
        for observation in state.rounds[0].observations
        for action in observation.actions
    )


class LocalContinuationReasoner:
    def __init__(self) -> None:
        self.calls = 0

    def observe(self, *, question, worker_id, territory_id, evidence):
        self.calls += 1
        needs = []
        if self.calls == 1:
            needs.append(
                UnresolvedNeed(
                    description="Need to connect probabilities to sampled outcomes.",
                    missing="How probabilities become sampled outcomes.",
                    suggested_terms=["sample_shots", "frequencies"],
                    suggested_territories=[territory_id],
                    scope="local",
                    source_worker_id=worker_id,
                )
            )
        return WorkerObservation(
            worker_id=worker_id, territory_id=territory_id, unresolved_needs=needs
        )


def test_local_need_can_continue_the_same_worker(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "backend.py").write_text(
        "def probabilities():\n    return [0.5, 0.5]\n\ndef sample_shots():\n    return {'0': 1}\n",
        encoding="utf-8",
    )
    worker = WorkerCard(
        id="worker-backend",
        territory_id="backend",
        name="backend",
        root="src/backend",
        searchable_terms=["probabilities", "sample_shots", "frequencies"],
        files=["src/backend.py"],
    )
    reasoner = LocalContinuationReasoner()

    state = LocalCoordinator(tmp_path, [worker], reasoner=reasoner).ask(
        "How do probabilities become outcomes?", max_rounds=2
    )

    assert [round_.selected_worker_ids for round_ in state.rounds] == [
        ["worker-backend"],
        ["worker-backend"],
    ]
    assert "Continued" in state.rounds[1].selection_reason
    assert not state.rounds[1].coalition_formed


class MissingFocusedReasoner:
    def observe(self, *, question, worker_id, territory_id, evidence):
        needs = []
        if worker_id == "worker-docs":
            needs.append(
                UnresolvedNeed(
                    description="Need the implementation of FusedGate merging.",
                    missing="Implementation of FusedGate merging.",
                    known=[
                        "The original question and docs mention tutorials, examples, and README."
                    ],
                    suggested_terms=["FusedGate", "fuse"],
                    suggested_territories=["gate optimization"],
                    scope="cross_territory",
                    source_worker_id=worker_id,
                )
            )
        return WorkerObservation(
            worker_id=worker_id, territory_id=territory_id, unresolved_needs=needs
        )


def test_follow_up_routing_uses_missing_information_over_original_question(
    tmp_path: Path,
) -> None:
    for root in ("docs", "optimizers", "examples"):
        (tmp_path / root).mkdir()
    (tmp_path / "docs" / "guide.py").write_text(
        "def tutorial_entry():\n    return 'automatic gate fusion docs'\n",
        encoding="utf-8",
    )
    (tmp_path / "optimizers" / "fusion.py").write_text(
        "class FusedGate:\n    pass\n\ndef fuse():\n    return FusedGate()\n",
        encoding="utf-8",
    )
    (tmp_path / "examples" / "demo.py").write_text(
        "def tutorial_example():\n    return 'docs examples readme'\n",
        encoding="utf-8",
    )
    workers = [
        WorkerCard(
            id="worker-docs",
            territory_id="docs",
            name="docs",
            root="docs",
            searchable_terms=["automatic", "gate", "fusion", "tutorial"],
            files=["docs/guide.py"],
        ),
        WorkerCard(
            id="worker-optimizers",
            territory_id="optimizers",
            name="gate optimization",
            root="optimizers",
            searchable_terms=["FusedGate", "fuse", "merge"],
            files=["optimizers/fusion.py"],
        ),
        WorkerCard(
            id="worker-examples",
            territory_id="examples",
            name="examples",
            root="examples",
            searchable_terms=["tutorial", "examples", "readme"],
            files=["examples/demo.py"],
        ),
    ]

    state = LocalCoordinator(tmp_path, workers, reasoner=MissingFocusedReasoner()).ask(
        "How does the tutorial explain automatic gate fusion examples?", max_rounds=2
    )

    assert state.rounds[0].selected_worker_ids == ["worker-docs"]
    assert state.rounds[1].query == (
        "Implementation of FusedGate merging. Need the implementation of FusedGate merging. "
        "FusedGate fuse gate optimization"
    )
    assert "tutorial" not in state.rounds[1].query
    assert "examples" not in state.rounds[1].query
    assert state.rounds[1].selected_worker_ids == ["worker-optimizers"]
    assert "worker-examples" not in state.rounds[1].selected_worker_ids


class FuzzyTerritoryReasoner:
    def observe(self, *, question, worker_id, territory_id, evidence):
        needs = []
        if worker_id == "worker-api":
            needs.append(
                UnresolvedNeed(
                    description="Need backend sampling implementation.",
                    missing="Backend sampling implementation.",
                    suggested_terms=["sample_shots"],
                    suggested_territories=["backend architecture"],
                    scope="cross_territory",
                    source_worker_id=worker_id,
                )
            )
        return WorkerObservation(
            worker_id=worker_id, territory_id=territory_id, unresolved_needs=needs
        )


def test_territory_hints_are_fuzzy_soft_scores(tmp_path: Path) -> None:
    for root in ("api", "src/qibo/backends", "tools"):
        (tmp_path / root).mkdir(parents=True)
    (tmp_path / "api" / "circuit.py").write_text(
        "def execute_circuit():\n    return 'measurement'\n", encoding="utf-8"
    )
    (tmp_path / "src" / "qibo" / "backends" / "numpy.py").write_text(
        "def sample_shots():\n    return 'samples'\n", encoding="utf-8"
    )
    (tmp_path / "tools" / "sample.py").write_text(
        "def sample_shots():\n    return 'tool samples'\n", encoding="utf-8"
    )
    workers = [
        WorkerCard(
            id="worker-api",
            territory_id="api",
            name="api",
            root="api",
            searchable_terms=["execute_circuit", "measurement"],
            files=["api/circuit.py"],
        ),
        WorkerCard(
            id="worker-backends",
            territory_id="src/qibo/backends",
            name="qibo backend worker",
            root="src/qibo/backends",
            searchable_terms=["sample_shots"],
            files=["src/qibo/backends/numpy.py"],
        ),
        WorkerCard(
            id="worker-tools",
            territory_id="tools",
            name="tools",
            root="tools",
            searchable_terms=["sample_shots"],
            files=["tools/sample.py"],
        ),
    ]

    state = LocalCoordinator(tmp_path, workers, reasoner=FuzzyTerritoryReasoner()).ask(
        "Where does measurement data flow?", max_rounds=2
    )

    assert state.rounds[1].selected_worker_ids == ["worker-backends"]
    assert state.rounds[1].candidate_worker_ids[:2] == ["worker-backends", "worker-tools"]


class ImplementationSourceReasoner:
    def observe(self, *, question, worker_id, territory_id, evidence):
        needs = []
        if worker_id == "worker-tests":
            needs.append(
                UnresolvedNeed(
                    description="Need the source code implementation of draw symbols.",
                    missing="Source code definition for rendering labels.",
                    suggested_terms=["render_gate_labels", "Circuit.draw", "symbol mapping"],
                    suggested_territories=["src package implementation"],
                    scope="cross_territory",
                    source_worker_id=worker_id,
                )
            )
        return WorkerObservation(
            worker_id=worker_id, territory_id=territory_id, unresolved_needs=needs
        )


def test_cross_territory_source_need_prefers_implementation_over_tests(
    tmp_path: Path,
) -> None:
    for root in ("src/tests", "src/models"):
        (tmp_path / root).mkdir(parents=True)
    (tmp_path / "src" / "tests" / "test_draw.py").write_text(
        "def test_draw_symbols():\n    assert 'symbol mapping'\n", encoding="utf-8"
    )
    (tmp_path / "src" / "models" / "renderer.py").write_text(
        "def render_gate_labels():\n    return {'H': 'H'}\n", encoding="utf-8"
    )
    workers = [
        WorkerCard(
            id="worker-tests",
            territory_id="src-tests",
            name="tests",
            root="src/tests",
            searchable_terms=["draw_symbols", "Circuit.draw", "symbol mapping"],
            files=["src/tests/test_draw.py"],
        ),
        WorkerCard(
            id="worker-models",
            territory_id="src-models",
            name="src package implementation",
            root="src/models",
            searchable_terms=["render_gate_labels", "Circuit.draw"],
            files=["src/models/renderer.py"],
        ),
    ]

    state = LocalCoordinator(tmp_path, workers, reasoner=ImplementationSourceReasoner()).ask(
        "Where are draw symbols tested?", max_rounds=2
    )

    assert state.rounds[0].selected_worker_ids == ["worker-tests"]
    assert state.rounds[1].selected_worker_ids == ["worker-models"]


def test_initial_implementation_question_prefers_source_over_tests(
    tmp_path: Path,
) -> None:
    for root in ("src/models", "tests"):
        (tmp_path / root).mkdir(parents=True)
    (tmp_path / "src" / "models" / "variational.py").write_text(
        "class QAOA:\n    pass\n\nclass FALQON(QAOA):\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_variational.py").write_text(
        "def test_qaoa_falqon():\n    assert 'QAOA FALQON subclass implementation'\n",
        encoding="utf-8",
    )
    workers = [
        WorkerCard(
            id="worker-tests",
            territory_id="tests",
            name="tests",
            root="tests",
            searchable_terms=["QAOA", "FALQON", "subclass", "implementation"],
            files=["tests/test_variational.py"],
        ),
        WorkerCard(
            id="worker-src-models",
            territory_id="src-models",
            name="models",
            root="src/models",
            searchable_terms=["QAOA", "FALQON"],
            files=["src/models/variational.py"],
        ),
    ]

    state = LocalCoordinator(tmp_path, workers).ask(
        "Where is the QAOA subclass implementation?", max_rounds=1
    )

    assert state.rounds[0].selected_worker_ids == ["worker-src-models"]


def test_memory_route_soft_bonus_can_select_known_worker(tmp_path: Path) -> None:
    for root in ("backends", "docs"):
        (tmp_path / root).mkdir()
    (tmp_path / "backends" / "numpy.py").write_text(
        "def sample_shots(probabilities):\n    return probabilities\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "guide.py").write_text(
        "def measurement_notes():\n    return 'measurement overview'\n",
        encoding="utf-8",
    )
    workers = [
        WorkerCard(
            id="worker-docs",
            territory_id="docs",
            name="docs",
            root="docs",
            searchable_terms=["measurement"],
            files=["docs/guide.py"],
        ),
        WorkerCard(
            id="worker-backends",
            territory_id="backends",
            name="backends",
            root="backends",
            searchable_terms=["sample_shots"],
            files=["backends/numpy.py"],
        ),
    ]

    state = LocalCoordinator(
        tmp_path,
        workers,
        memory_routes=[
            MemoryRoute(
                need_terms=["measurement"],
                worker_ids=["worker-backends"],
                weight=3.0,
            )
        ],
    ).ask("How are measurement samples produced?", max_rounds=1)

    assert state.rounds[0].selected_worker_ids == ["worker-backends"]
    assert state.rounds[0].routing_scores[0].memory_route_bonus > 0


def test_inheritance_question_reports_coverage_gap_without_subclass_evidence(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "models.py").write_text(
        "class QAOA:\n    pass\n",
        encoding="utf-8",
    )
    worker = WorkerCard(
        id="worker-src",
        territory_id="src",
        name="src",
        root="src",
        searchable_terms=["QAOA"],
        files=["src/models.py"],
    )

    state = LocalCoordinator(tmp_path, [worker]).ask(
        "What subclasses inherit from QAOA?", max_rounds=1
    )

    assert state.unresolved_needs
    assert state.unresolved_needs[0].need_type == "subclass_lookup"
    assert "QAOA" in state.unresolved_needs[0].relevant_symbols


def test_no_evidence_returns_structured_negative_presence_need(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "models.py").write_text(
        "class Present:\n    pass\n",
        encoding="utf-8",
    )
    worker = WorkerCard(
        id="worker-src",
        territory_id="src",
        name="src",
        root="src",
        searchable_terms=["Present"],
        files=["src/models.py"],
    )

    state = LocalCoordinator(tmp_path, [worker]).ask(
        "Is MissingVisualizer implemented?", max_rounds=1
    )

    assert state.unresolved_needs[0].kind == "coverage_gap"
    assert state.unresolved_needs[0].need_type == "implementation_location"
    assert "MissingVisualizer" in state.unresolved_needs[0].relevant_symbols
    assert state.absence_proofs[0].searched_worker_ids == ["worker-src"]
    assert state.absence_proofs[0].searched_paths == ["src/models.py"]
    assert state.absence_proofs[0].conclusion == "not_found"


def test_source_test_question_requires_evidence_from_both_sides(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "service.py").write_text(
        "def authenticate():\n    return True\n", encoding="utf-8"
    )
    worker = WorkerCard(
        id="worker-src",
        territory_id="src",
        name="src",
        root="src",
        searchable_terms=["authenticate"],
        files=["src/service.py"],
    )

    state = LocalCoordinator(tmp_path, [worker]).ask(
        "Where is authenticate implemented and tested?", max_rounds=1
    )

    coalition_need = next(
        need for need in state.unresolved_needs if need.need_type == "source_test_coalition"
    )
    assert coalition_need.kind == "coverage_gap"
    assert coalition_need.scope == "cross_territory"
    assert coalition_need.missing == "test coverage"


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
        "value = helper()" in item.quote for item in tool.assignments("value", ["src/model.py"])
    )
