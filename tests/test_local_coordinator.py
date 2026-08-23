from pathlib import Path

from ant.coordinator import LocalCoordinator
from ant.coordinator.local import (
    _close_resolved_needs,
    _matches_term,
    _merge_needs,
    _relevant_symbol_weights,
    _reopen_referenced_evidence,
)
from ant.domain import Evidence, UnresolvedNeed, WorkerCard, WorkerObservation
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


def test_query_from_needs_keeps_the_original_question_as_a_stable_anchor() -> None:
    # Regression test: round 2+ queries used to fully replace the original
    # question with LLM-generated need text. That LLM call has no
    # temperature/seed pinning, so its exact wording varies between
    # otherwise-identical runs -- and a lucky/unlucky word choice could
    # erase recall for a term the original question used but the need
    # happened not to repeat. The anchor must survive even when a need
    # fully replaces the topic-specific portion of the query.
    need = UnresolvedNeed(
        description="Need the seed handling.",
        missing="Seed handling implementation.",
        source_worker_id="worker-a",
    )
    query = LocalCoordinator._query_from_needs(
        "How are reproducible measurements sampled across backends?", [need]
    )
    assert query.startswith("How are reproducible measurements sampled across backends?")
    assert "Seed handling implementation." in query


def test_routing_matcher_avoids_arbitrary_substrings() -> None:
    assert not _matches_term("into", {"quantum_info"})
    assert not _matches_term("fusedgate", {"gate"})
    assert _matches_term("sample", {"sample_shots"})
    assert _matches_term("fusedgate", {"fusedgate"})


def test_routing_matcher_matches_a_common_grammatical_stem() -> None:
    # Regression test: a question asking "where is drawing implemented"
    # previously missed a worker whose only matching vocabulary was the
    # method name "draw" -- same concept, different inflection -- because
    # matching required an exact string or an underscore-split component.
    assert _matches_term("drawing", {"draw"})
    assert _matches_term("backends", {"backend"})
    # Must not match on shared prefixes shorter than 4 chars.
    assert not _matches_term("ingest", {"in"})


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
    # The original question stays present as a stable lexical anchor (it is
    # never fully dropped, even once a specific need takes over routing),
    # but the need-derived text is what actually decides the worker pick
    # below: worker-optimizers wins on "FusedGate"/"fuse" overlap despite
    # the anchor's "tutorial"/"examples" wording nominally favoring
    # worker-examples/worker-docs just as strongly on raw term count.
    assert state.rounds[1].query == (
        "How does the tutorial explain automatic gate fusion examples? "
        "Implementation of FusedGate merging. Need the implementation of FusedGate merging. "
        "FusedGate fuse gate optimization"
    )
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


def test_memory_route_bonus_ignores_a_single_generic_word_shared_with_an_unrelated_route(
    tmp_path: Path,
) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "visualization").mkdir()
    (tmp_path / "models" / "algorithms.py").write_text(
        "class QAOA:\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "visualization" / "bloch.py").write_text(
        "def paint_bloch_sphere():\n    return 'quantum state plot'\n", encoding="utf-8"
    )
    workers = [
        WorkerCard(
            id="worker-models",
            territory_id="models",
            name="models",
            root="models",
            searchable_terms=["QAOA"],
            files=["models/algorithms.py"],
        ),
        WorkerCard(
            id="worker-visualization",
            territory_id="visualization",
            name="visualization",
            root="visualization",
            searchable_terms=["bloch", "sphere"],
            files=["visualization/bloch.py"],
        ),
    ]
    # Recorded from answering an unrelated earlier question about QAOA
    # subclasses -- its vocabulary shares only the single generic word
    # "quantum" with the new, unrelated Bloch-sphere question below.
    unrelated_route = MemoryRoute(
        need_terms=[
            "algorithm",
            "class",
            "extended",
            "inheriting",
            "methods",
            "overridden",
            "qaoa",
            "quantum",
            "specialized",
            "subclasses",
            "variational",
        ],
        worker_ids=["worker-models"],
        weight=5.0,
    )

    state = LocalCoordinator(
        tmp_path, workers, memory_routes=[unrelated_route]
    ).ask("How does the quantum state get rendered on the Bloch sphere?", max_rounds=1)

    # Without the stale memory bonus, worker-models has no signal at all for
    # this question and should not even be a candidate -- previously the
    # capped +12 bonus from the unrelated route alone would have put it
    # ahead of the genuinely relevant worker.
    assert "worker-models" not in state.rounds[0].candidate_worker_ids
    assert state.rounds[0].selected_worker_ids == ["worker-visualization"]


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


def test_implement_stem_without_suffix_triggers_implementation_coverage_gap(
    tmp_path: Path,
) -> None:
    # Regression test: "How does X implement Y" (bare stem, no -ation/-ed
    # suffix) previously matched none of _asks_for_source_implementation_text's
    # indicators, so the question fell back to a "negative_presence" need
    # even when it was clearly asking for an implementation -- meaning
    # nothing ever flagged that the actual algorithm/code was missing.
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
        "How does the library implement automatic gate fusion?", max_rounds=1
    )

    assert state.unresolved_needs[0].need_type == "implementation_location"


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


def test_navigate_does_not_swallow_a_deep_method_inside_a_large_class(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    filler_methods = "\n\n".join(
        f"    def filler_{index}(self):\n        return {index}" for index in range(60)
    )
    (tmp_path / "src" / "backend.py").write_text(
        "class NumpyBackend:\n"
        f"{filler_methods}\n\n"
        "    def calculate_probabilities(self, state, qubits, nqubits):\n"
        "        return state\n",
        encoding="utf-8",
    )

    tool = LocalSearchTool(tmp_path)
    class_hits = tool.navigate("NumpyBackend", ["src/backend.py"])
    method_hits = tool.navigate("calculate_probabilities", ["src/backend.py"])

    # Navigating to the class must not return its entire body (which would
    # get flat-truncated before reaching a method deep inside it, even
    # though line_start/line_end would still claim full coverage); it must
    # stay capped to a short header region.
    assert class_hits
    assert (class_hits[0].line_end - class_hits[0].line_start) < 20
    # The specific method must still be reachable on its own -- previously
    # it would be silently dropped because it fell entirely inside the
    # class's own (uncapped) range and _merge_windows treats "contained
    # within an already-added range" as a duplicate to skip.
    assert method_hits
    assert "def calculate_probabilities" in method_hits[0].quote


def test_resolve_symbol_expands_a_large_class_into_its_own_members(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    filler_methods = "\n\n".join(
        f"    def filler_{index}(self):\n        return {index}" for index in range(60)
    )
    (tmp_path / "src" / "circuit.py").write_text(
        "class Circuit:\n"
        f"{filler_methods}\n\n"
        "    def draw(self):\n"
        "        return 'diagram'\n",
        encoding="utf-8",
    )

    evidence = LocalSearchTool(tmp_path).resolve_symbol(
        "Circuit", ["src/circuit.py"], need="Where is circuit drawing implemented?"
    )

    # Regression test: resolving a class this large used to return only the
    # class itself as one flat-truncated blob, so a method deep inside it
    # (here `draw`, defined after 60 unrelated filler methods) never
    # actually appeared in the visible quote text even though the class was
    # correctly found. It must now surface as its own relevance-ranked
    # evidence item -- not get lost either behind the truncation point or
    # behind 60 more-numerous but irrelevant siblings.
    assert any(
        "def draw" in item.quote and item.quote.strip().startswith("def draw")
        for item in evidence
    )


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


def _subclass_need(relevant_symbols: list[str] | None = None) -> UnresolvedNeed:
    return UnresolvedNeed(
        description="Need subclass definitions that inherit from the requested base symbol.",
        kind="coverage_gap",
        need_type="subclass_lookup",
        missing="Subclass definitions and their base-class relationship.",
        scope="unknown",
        relevant_symbols=relevant_symbols or ["QAOA"],
        suggested_terms=["QAOA", "subclass", "inherit"],
    )


def test_close_resolved_needs_closes_subclass_lookup_once_scoped_evidence_exists() -> None:
    need = _subclass_need(["QAOA"])
    evidence = [
        Evidence(
            path="src/models/variational.py",
            line_start=549,
            line_end=575,
            quote="class FALQON(QAOA):\n    pass",
            reason="Subclass lookup for base symbol QAOA.",
        )
    ]

    remaining = _close_resolved_needs([need], evidence, "What subclasses inherit from QAOA?")

    assert remaining == []


def test_close_resolved_needs_does_not_close_on_an_unrelated_base_class() -> None:
    need = _subclass_need(["QAOA"])
    evidence = [
        Evidence(
            path="src/models/other.py",
            line_start=1,
            line_end=2,
            quote="class Widget(BaseComponent):\n    pass",
            reason="Unrelated subclass in a different territory.",
        )
    ]

    remaining = _close_resolved_needs([need], evidence, "What subclasses inherit from QAOA?")

    assert remaining == [need]


def test_close_resolved_needs_closes_implementation_location_on_scoped_definition() -> None:
    need = UnresolvedNeed(
        description="Need source implementation definitions, not only references or tests.",
        kind="coverage_gap",
        need_type="implementation_location",
        missing="Source code definitions that implement the requested behavior.",
        scope="unknown",
        relevant_symbols=["render_gate_labels"],
        suggested_terms=["render_gate_labels", "implementation"],
    )
    unrelated_evidence = [
        Evidence(
            path="src/other.py",
            line_start=1,
            line_end=2,
            quote="def unrelated_helper():\n    return None",
            reason="Different symbol.",
        )
    ]
    matching_evidence = [
        *unrelated_evidence,
        Evidence(
            path="src/models/renderer.py",
            line_start=10,
            line_end=11,
            quote="def render_gate_labels():\n    return {'H': 'H'}",
            reason="Definition for render_gate_labels.",
        ),
    ]

    assert _close_resolved_needs(
        [need], unrelated_evidence, "Where is render_gate_labels implemented?"
    ) == [need]
    assert (
        _close_resolved_needs(
            [need], matching_evidence, "Where is render_gate_labels implemented?"
        )
        == []
    )


def test_close_resolved_needs_never_auto_closes_absence_type_needs() -> None:
    need = UnresolvedNeed(
        description="Need grounded evidence for MissingVisualizer.",
        kind="coverage_gap",
        need_type="negative_presence",
        missing="Grounded evidence for MissingVisualizer.",
        scope="unknown",
        relevant_symbols=["MissingVisualizer"],
    )
    plenty_of_unrelated_evidence = [
        Evidence(
            path="src/models/variational.py",
            line_start=549,
            line_end=575,
            quote="class FALQON(QAOA):\n    def render_gate_labels(self):\n        return {}",
            reason="Unrelated evidence should never resolve an absence claim.",
        )
    ]

    remaining = _close_resolved_needs(
        [need], plenty_of_unrelated_evidence, "Is MissingVisualizer implemented?"
    )

    assert remaining == [need]


def test_merge_needs_deduplicates_same_gap_across_rounds() -> None:
    first_round = [_subclass_need(["QAOA"])]
    second_round = [_subclass_need(["QAOA", "FALQON"])]

    merged = _merge_needs(first_round, second_round)

    assert len(merged) == 1
    assert sorted(merged[0].relevant_symbols) == ["FALQON", "QAOA"]


class StaleSubclassNeedReasoner:
    """Simulates an LLM reasoner that keeps flagging the same completeness
    doubt about QAOA's subclasses every round and never notices on its own
    that a later round's evidence has already grounded it. The coordinator's
    own closure check -- not the reasoner -- is what must drop the need."""

    def observe(self, *, question, worker_id, territory_id, evidence):
        return WorkerObservation(
            worker_id=worker_id,
            territory_id=territory_id,
            unresolved_needs=[
                UnresolvedNeed(
                    description="Need to confirm all subclasses of QAOA are covered.",
                    kind="coverage_gap",
                    need_type="subclass_lookup",
                    missing="Subclass definitions and their base-class relationship.",
                    scope="unknown",
                    relevant_symbols=["QAOA"],
                    suggested_terms=["QAOA", "subclass", "inherit"],
                    suggested_territories=["variational subclasses"],
                    source_worker_id=worker_id,
                )
            ],
        )


def test_subclass_lookup_need_closes_once_a_later_round_grounds_it(tmp_path: Path) -> None:
    (tmp_path / "src" / "base").mkdir(parents=True)
    (tmp_path / "src" / "variants").mkdir(parents=True)
    (tmp_path / "src" / "base" / "qaoa.py").write_text(
        "class QAOA:\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "src" / "variants" / "falqon.py").write_text(
        "class FALQON(QAOA):\n    pass\n", encoding="utf-8"
    )
    workers = [
        WorkerCard(
            id="worker-base",
            territory_id="src-base",
            name="base algorithms",
            root="src/base",
            searchable_terms=["QAOA"],
            files=["src/base/qaoa.py"],
        ),
        WorkerCard(
            id="worker-variants",
            territory_id="src-variants",
            name="variational subclasses",
            root="src/variants",
            searchable_terms=["FALQON"],
            files=["src/variants/falqon.py"],
        ),
    ]

    state = LocalCoordinator(
        tmp_path, workers, reasoner=StaleSubclassNeedReasoner()
    ).ask("What subclasses inherit from QAOA?", max_rounds=2)

    assert state.rounds[0].selected_worker_ids == ["worker-base"]
    assert state.rounds[1].selected_worker_ids == ["worker-variants"]
    assert any(
        "FALQON" in item.quote and "QAOA" in item.quote for item in state.evidence
    )
    assert not any(need.need_type == "subclass_lookup" for need in state.unresolved_needs)


def test_reopen_referenced_evidence_pulls_a_larger_region(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    lines = [f"# filler {index}" for index in range(1, 41)]
    lines[19] = "def target():"
    lines[20] = "    return 1"
    (tmp_path / "src" / "mod.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    search = LocalSearchTool(tmp_path)
    narrow = Evidence(
        path="src/mod.py",
        line_start=18,
        line_end=22,
        quote="def target():\n    return 1",
        reason="Initial narrow hit.",
        worker_id="worker-a",
    )
    need = UnresolvedNeed(description="Need more context around target().", evidence_ids=["0"])

    reopened = _reopen_referenced_evidence([need], [narrow], search, context_lines=15)

    assert len(reopened) == 1
    assert reopened[0].path == "src/mod.py"
    assert reopened[0].worker_id == "worker-a"
    assert (reopened[0].line_end - reopened[0].line_start) > (narrow.line_end - narrow.line_start)
    assert "Reopened for coalition cross-check" in reopened[0].reason


def test_reopen_referenced_evidence_ignores_invalid_or_out_of_range_ids(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    search = LocalSearchTool(tmp_path)
    need = UnresolvedNeed(description="...", evidence_ids=["not-a-number", "99"])

    assert _reopen_referenced_evidence([need], [], search) == []


class CoalitionJointReasoner:
    """Only worker-api raises the cross-territory need that triggers a
    coalition; a distinct "coalition" territory_id call represents the joint
    reasoning pass, referencing evidence index 0 to exercise the reopen path.
    """

    def observe(self, *, question, worker_id, territory_id, evidence):
        if territory_id == "coalition":
            return WorkerObservation(
                worker_id=worker_id,
                territory_id=territory_id,
                unresolved_needs=[
                    UnresolvedNeed(
                        description="Need to confirm what submit_record forwards.",
                        kind="missing_detail",
                        scope="cross_territory",
                        evidence_ids=["0"],
                    )
                ],
            )
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


def test_coalition_runs_joint_cross_check_and_reopens_referenced_evidence(
    tmp_path: Path,
) -> None:
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

    state = LocalCoordinator(tmp_path, workers, reasoner=CoalitionJointReasoner()).ask(
        "How does submit work?", max_rounds=2
    )

    assert state.rounds[1].coalition_formed
    coalition_observations = [
        observation
        for observation in state.rounds[1].observations
        if observation.territory_id == "coalition"
    ]
    assert len(coalition_observations) == 1
    assert coalition_observations[0].stop_reason == "coalition_cross_check"
    assert any("Reopened for coalition cross-check" in item.reason for item in state.evidence)


def test_relevant_symbol_weights_downweight_terms_common_across_the_colony() -> None:
    qibo_workers = [
        WorkerCard(
            id=f"worker-qibo-{index}",
            territory_id=f"qibo-{index}",
            name=f"qibo module {index}",
            root=f"src/qibo/mod{index}",
            files=[f"src/qibo/mod{index}/file.py"],
        )
        for index in range(6)
    ]
    examples_worker = WorkerCard(
        id="worker-examples-bloch",
        territory_id="examples-bloch",
        name="bloch example",
        root="examples/bloch",
        searchable_terms=["Bloch"],
        files=["examples/bloch/plot.py"],
    )
    workers = [*qibo_workers, examples_worker]

    weights = _relevant_symbol_weights({"Qibo", "Bloch"}, workers)

    # "Qibo" matches 6 of 7 workers purely via the repo-name-in-every-path
    # fallback and must be heavily discounted; "Bloch" matches exactly one
    # worker for a real reason and must keep close to full weight.
    assert weights["Qibo"] < weights["Bloch"]
    assert weights["Bloch"] == 6


def test_repo_name_matching_every_src_path_does_not_bury_the_real_match(
    tmp_path: Path,
) -> None:
    for index in range(6):
        (tmp_path / "src" / "qibo" / f"mod{index}").mkdir(parents=True)
        (tmp_path / "src" / "qibo" / f"mod{index}" / "file.py").write_text(
            "def unrelated():\n    return 1\n", encoding="utf-8"
        )
    (tmp_path / "examples" / "bloch").mkdir(parents=True)
    (tmp_path / "examples" / "bloch" / "plot.py").write_text(
        "def paint_bloch_sphere():\n    return 'bloch'\n", encoding="utf-8"
    )
    qibo_workers = [
        WorkerCard(
            id=f"worker-qibo-{index}",
            territory_id=f"qibo-{index}",
            name=f"qibo module {index}",
            root=f"src/qibo/mod{index}",
            searchable_terms=["qibo"],
            files=[f"src/qibo/mod{index}/file.py"],
        )
        for index in range(6)
    ]
    examples_worker = WorkerCard(
        id="worker-examples-bloch",
        territory_id="examples-bloch",
        name="bloch example",
        root="examples/bloch",
        searchable_terms=["Bloch", "paint_bloch_sphere"],
        files=["examples/bloch/plot.py"],
    )

    state = LocalCoordinator(tmp_path, [*qibo_workers, examples_worker]).ask(
        "How does Qibo's visualization render the Bloch sphere?", max_rounds=1
    )

    assert state.rounds[0].selected_worker_ids == ["worker-examples-bloch"]


def test_inheritance_scan_finds_subclass_in_a_territory_never_recruited(
    tmp_path: Path,
) -> None:
    (tmp_path / "base").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "base" / "qaoa.py").write_text("class QAOA:\n    pass\n", encoding="utf-8")
    (tmp_path / "other" / "falqon.py").write_text(
        "class FALQON(QAOA):\n    pass\n", encoding="utf-8"
    )
    workers = [
        WorkerCard(
            id="worker-base",
            territory_id="base",
            name="base",
            root="base",
            searchable_terms=["QAOA"],
            files=["base/qaoa.py"],
        ),
        WorkerCard(
            id="worker-other",
            territory_id="other",
            name="other",
            root="other",
            searchable_terms=["something_unrelated"],
            files=["other/falqon.py"],
        ),
    ]

    state = LocalCoordinator(tmp_path, workers).ask(
        "What subclasses inherit from QAOA?", max_rounds=1
    )

    # Routing only ever recruited worker-base: worker-other's own terms don't
    # match the question at all, so this evidence could only have come from
    # the repo-wide completeness scan, not from need-conditioned recruitment.
    assert [round_.selected_worker_ids for round_ in state.rounds] == [["worker-base"]]
    assert any(
        "FALQON" in item.quote and "QAOA" in item.quote for item in state.evidence
    )
    assert not any(need.need_type == "subclass_lookup" for need in state.unresolved_needs)
    exhaustive_proofs = [proof for proof in state.absence_proofs if proof.exhaustive]
    assert exhaustive_proofs
    assert exhaustive_proofs[0].conclusion == "found_1_subclass"
    assert "QAOA" in exhaustive_proofs[0].relevant_symbols
