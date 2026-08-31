from pathlib import Path

from ant.domain import Evidence, WorkerCard
from ant.retrieval import BM25Index
from ant.tools import LocalSearchTool
from ant.tools.local import _symbol_path_channel_rank
from ant.workers import AutonomousWorker, WorkerRunConfig
from ant.workers.autonomous import _rank_evidence


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


def _boilerplate_extractor(class_name: str) -> str:
    # Mirrors the real yt-dlp extractor shape this regression targets:
    # every sibling file shares the same generic vocabulary (error,
    # unable, find, url, extractor), so those terms are common across the
    # whole territory and should NOT be what decides the ranking.
    return (
        f"class {class_name}(BaseIE):\n"
        "    def _real_extract(self, url):\n"
        "        webpage = self._download_webpage(url, None)\n"
        "        if not webpage:\n"
        "            raise ExtractorError('Unable to find video URL')\n"
        "        return self._extract_from_webpage(webpage)\n"
    )


def test_territory_wide_bm25_ranks_a_rare_term_above_common_boilerplate_across_files(
    tmp_path: Path,
) -> None:
    # Regression test for the yt-dlp Teachable incident: search()'s old
    # per-file-scoped BM25 (a fresh BM25Index per file, so its own IDF
    # never saw document frequency across the other files) let a file
    # matching several common, territory-wide boilerplate terms outrank
    # the one file containing the query's actual discriminative term.
    # Lowercase "teachable" in the query on purpose -- the fix must not
    # depend on the query preserving a symbol's original capitalization.
    (tmp_path / "extractor").mkdir()
    sibling_names = ["Soundcloud", "Pornhub", "Rai", "Tiktok", "Archiveorg", "Tnaflix"]
    for name in sibling_names:
        (tmp_path / "extractor" / f"{name.lower()}.py").write_text(
            _boilerplate_extractor(f"{name}IE"), encoding="utf-8"
        )
    (tmp_path / "extractor" / "teachable.py").write_text(
        _boilerplate_extractor("TeachableIE"), encoding="utf-8"
    )
    files = [f"extractor/{name.lower()}.py" for name in [*sibling_names, "teachable"]]

    evidence = LocalSearchTool(tmp_path).search(
        "teachable extractor unable to find video url", files, limit=3
    )

    assert evidence
    assert evidence[0].path == "extractor/teachable.py"


def test_territory_wide_bm25_orders_the_rare_term_file_above_common_term_only_files(
    tmp_path: Path,
) -> None:
    # Same shape, checking relative order specifically rather than just
    # top-1 presence: every sibling file matches the query's common terms
    # (error/unable/find/url/extractor) about equally well, so if the rare
    # term "teachable" carried no extra weight, teachable.py would just be
    # one of several ~tied files, not reliably first.
    (tmp_path / "extractor").mkdir()
    sibling_names = ["Soundcloud", "Pornhub", "Rai", "Tiktok", "Archiveorg", "Tnaflix"]
    for name in sibling_names:
        (tmp_path / "extractor" / f"{name.lower()}.py").write_text(
            _boilerplate_extractor(f"{name}IE"), encoding="utf-8"
        )
    (tmp_path / "extractor" / "teachable.py").write_text(
        _boilerplate_extractor("TeachableIE"), encoding="utf-8"
    )
    files = [f"extractor/{name.lower()}.py" for name in [*sibling_names, "teachable"]]

    evidence = LocalSearchTool(tmp_path).search(
        "teachable extractor unable to find video url", files, limit=len(files)
    )

    ranked_paths = [item.path for item in evidence]
    assert ranked_paths[0] == "extractor/teachable.py"


def test_symbol_channel_matches_a_class_name_regardless_of_query_capitalization(
    tmp_path: Path,
) -> None:
    # The old "symbol bonus" only fired when the query string happened to
    # preserve a symbol's original capitalization (_query_symbols required
    # an uppercase char in the *query* token itself). A need's search
    # query is Orchestrator-authored free text, not guaranteed to echo a
    # proper noun's exact casing -- confirmed on a real trace, the auto-
    # generated need_id for the Teachable question was already all-
    # lowercase. This asserts the fix: an all-lowercase query still finds
    # the class through the symbol/path channel.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "teachable.py").write_text(
        "class TeachableIE(BaseIE):\n    def _real_extract(self, url):\n        pass\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "unrelated.py").write_text(
        "class UnrelatedIE(BaseIE):\n    def _real_extract(self, url):\n        pass\n",
        encoding="utf-8",
    )

    evidence = LocalSearchTool(tmp_path).search(
        "teachable video extraction", ["src/teachable.py", "src/unrelated.py"], limit=1
    )

    assert evidence
    assert evidence[0].path == "src/teachable.py"


def test_symbol_channel_weights_a_shared_path_component_below_a_rare_term_match() -> None:
    # Regression test for a bug found only via a live smoke test against
    # real yt-dlp data (the synthetic tests above used too small a corpus
    # to expose it): every file in a territory sharing one parent
    # directory (e.g. "extractor/") makes that directory name match
    # *every* region in the territory through the same symbol_term_index
    # this channel uses for filenames/symbols. A flat +1-per-matched-term
    # count let nine sibling regions, each additionally matching three
    # common symbol-level terms ("find"/"video"/"url", from a shared
    # helper name), outrank the one region whose only extra match was the
    # query's actual rare, discriminative term ("teachable") -- the exact
    # "common terms drown the rare term" failure this whole retrieval
    # rewrite exists to remove, just relocated into this channel instead
    # of the old _line_score. The fix weights each matched term by
    # 1/(regions it matches), so "extractor" (all 10 regions) contributes
    # almost nothing while "teachable" (1 region) dominates.
    regions = [("extractor/teachable.py", 0, ["class TeachableIE"])] + [
        (f"extractor/sibling{i}.py", 0, [f"class Sibling{i}IE"]) for i in range(9)
    ]
    symbol_term_index: dict[str, set[int]] = {
        "extractor": set(range(10)),
        "teachable": {0},
        "find": set(range(1, 10)),
        "video": set(range(1, 10)),
        "url": set(range(1, 10)),
    }

    ranked = _symbol_path_channel_rank(
        regions, symbol_term_index, ["teachable", "extractor", "find", "video", "url"], limit=10
    )

    assert ranked[0][0] == "extractor/teachable.py"


def test_search_finds_results_when_a_workers_entire_scope_sits_under_a_low_value_directory_name(
    tmp_path: Path,
) -> None:
    # Regression test for a real bug found only via a live smoke test
    # against real qibo data, not any synthetic test: an earlier version
    # of _territory_index skipped is_low_value_path/has_low_value_part
    # files entirely when building the territory-wide corpus (intended to
    # keep generated/vendored noise out). But has_low_value_part flags
    # entire directory names like "examples"/"test"/"tests"/"doc"/"docs"
    # as low-value -- and a worker's whole assigned file scope can, and
    # in this case did, live entirely under one of those names
    # (worker-examples-reuploading-classifier in the real qibo repo). The
    # filter then excluded every single file the worker owned, leaving an
    # empty corpus and zero search results, unconditionally, no matter
    # the query. A worker's `files` list is already its deliberately
    # assigned scope, not something a generic noise filter should be
    # allowed to zero out entirely -- so the filter was removed rather
    # than special-cased.
    (tmp_path / "examples" / "reuploading_classifier").mkdir(parents=True)
    (tmp_path / "examples" / "reuploading_classifier" / "qlassifier.py").write_text(
        "def paint_world_map(self):\n"
        "    fig, ax = world_map_template()\n"
        "    ax.scatter(laea_x(angles), laea_y(angles))\n",
        encoding="utf-8",
    )

    evidence = LocalSearchTool(tmp_path).search(
        "bloch sphere visualization world map",
        ["examples/reuploading_classifier/qlassifier.py"],
        limit=8,
    )

    assert evidence
    assert evidence[0].path == "examples/reuploading_classifier/qlassifier.py"


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


def test_rank_evidence_credits_a_symbol_name_matching_the_need() -> None:
    # Regression test for a real observed failure: asking "where is X
    # implemented" for a method whose *name* is the only thing that matches
    # the query (its body text shares no literal words with the question)
    # used to score identically to a handful of completely unrelated
    # sibling methods, because _rank_evidence built score_evidence() calls
    # without passing symbol_name -- even though the exact same Evidence
    # items already carry it in their `symbols` field (populated by
    # _definitions_to_evidence). A real trace showed this: qibo's
    # Circuit.draw() tied for last place with Circuit._shallow_copy() and
    # Circuit.decompose() when asked "where is drawing implemented", and
    # lost the final evidence_limit cut to less-relevant-but-lexically-tied
    # methods as a result.
    unrelated = Evidence(
        path="src/a.py",
        line_start=10,
        line_end=20,
        quote="def _shallow_copy(self):\n    return copy.copy(self)",
        reason="Resolved symbol definition for Widget.",
        symbols=["_shallow_copy", "Widget._shallow_copy"],
    )
    target = Evidence(
        path="src/a.py",
        line_start=30,
        line_end=40,
        quote="def draw(self, line_wrap=70):\n    return _render(self)",
        reason="Resolved symbol definition for Widget.",
        symbols=["draw", "Widget.draw"],
    )

    ranked = _rank_evidence([unrelated, target], "Where is drawing implemented?")

    assert ranked[0] is target


class _DenylistReasoner:
    """Stand-in for a real LLM reasoner: filters out one specific candidate,
    passing everything else through unchanged -- enough to prove
    AutonomousWorker actually wires a supplied reasoner's select_lookups()
    into which candidates get explored, without needing a real model call.
    """

    def __init__(self, denied: str) -> None:
        self.denied = denied
        self.seen_candidates: list[str] = []

    def observe(self, *, question, worker_id, territory_id, evidence):
        raise AssertionError("this test does not exercise observe()")

    def select_lookups(self, *, need, evidence, candidates):
        self.seen_candidates = list(candidates)
        return [item for item in candidates if item != self.denied]

    def select_workers(self, *, query, need, candidates, limit, memory_hints):
        raise AssertionError("this test does not exercise select_workers()")

    def select_evidence(self, *, question, evidence, limit):
        raise AssertionError("this test does not exercise select_evidence()")

    def plan_worker_actions(
        self, *, need, evidence, candidate_symbols, available_tools, hints, max_actions
    ):
        # Mirrors the old fixed pipeline's default per-symbol tool order
        # closely enough for this test's purpose: it only cares that a
        # denied candidate never gets a tool call, not the exact plan.
        all_tools = ("navigate", "references", "callers", "callees", "assignments")
        tools = [t for t in all_tools if t in available_tools]
        return [(tool, symbol) for symbol in candidate_symbols for tool in tools][:max_actions]

    def should_continue_recruiting(self, *, question, need, evidence, rounds_completed):
        raise AssertionError("this test does not exercise should_continue_recruiting()")

    def check_need_resolution(self, *, need, new_evidence, question):
        raise AssertionError("this test does not exercise check_need_resolution()")

    def verify_evidence_upgrade(self, *, need, epistemic_state, new_evidence, question):
        raise AssertionError("this test does not exercise verify_evidence_upgrade()")

    def decide_local_action(self, *, need, evidence, worker_progress, worker):
        raise AssertionError("this test does not exercise decide_local_action()")

    def plan_round(
        self,
        *,
        question,
        graph,
        resolution_results,
        evidence,
        workers,
        memory_hints,
        frontier,
        incomplete_parents,
        cross_repo_experience,
        validation_feedback="",
        repair_guidance="",
        stuck_tried_workers=None,
        candidate_probes=None,
    ):
        raise AssertionError("this test does not exercise plan_round()")

    def consolidate_graph(
        self, *, question, active_nodes, proposals, candidate_hints, enforce_alignment=False
    ):
        raise AssertionError("this test does not exercise consolidate_graph()")

    def summarize_task_experience(self, *, question, rounds, unresolved_needs, evidence_count):
        raise AssertionError("this test does not exercise summarize_task_experience()")


def test_autonomous_worker_filters_candidate_symbols_through_a_supplied_reasoner(
    tmp_path: Path,
) -> None:
    # Regression test for a real observed failure: _candidate_symbols()
    # treats any capitalized, length>3 token in the need text as a
    # lookup-worthy symbol -- including ordinary English words like
    # "Information" that a real need's free text happens to capitalize.
    # Without a reasoner, that noise is unavoidable (see
    # test_autonomous_worker_records_tool_actions and friends, which rely on
    # exactly this mechanical behavior). With one supplied, the reasoner's
    # judgment -- not the regex -- decides what actually gets a tool call.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "model.py").write_text(
        "class TargetSymbol:\n    def run(self):\n        return None\n",
        encoding="utf-8",
    )
    card = WorkerCard(
        id="worker-src",
        territory_id="src",
        name="src worker",
        root="src",
        searchable_terms=["targetsymbol"],
        files=["src/model.py"],
    )
    reasoner = _DenylistReasoner(denied="Information")

    worker = AutonomousWorker(tmp_path, card, LocalSearchTool(tmp_path), reasoner=reasoner)
    observation = worker.run(
        "Information about TargetSymbol behavior",
        WorkerRunConfig(max_tool_calls=8, evidence_limit=8),
    )

    # The mechanical extractor did surface the junk candidate (proving this
    # test would have failed before the fix, not just trivially pass)...
    assert "Information" in reasoner.seen_candidates
    # ...but no tool call was ever made against it once the reasoner said no.
    navigate_queries = [action.query for action in observation.actions if action.tool == "navigate"]
    assert "Information" not in navigate_queries
    assert "TargetSymbol" in navigate_queries
