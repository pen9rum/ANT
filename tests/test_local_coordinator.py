from pathlib import Path

from ant.coordinator import LocalCoordinator
from ant.coordinator.local import (
    _ESCALATED_NEED_CANDIDATE_LIMIT,
    _FRESH_NEED_CANDIDATE_LIMIT,
    RecoveryState,
    StuckEpisode,
    _build_temporary_bridge,
    _close_resolved_needs,
    _matches_term,
    _merge_needs,
    _plan_round_with_cycle_validation,
    _rank_global_evidence,
    _reopen_referenced_evidence,
    _select_evidence,
)
from ant.coordinator.worker_retrieval import build_worker_index
from ant.domain import (
    CodeSymbol,
    Evidence,
    FrontierResult,
    NeedGraph,
    NeedNode,
    NeedResolution,
    RepairAction,
    RepairPlan,
    RoundPlan,
    UnresolvedNeed,
    WorkerCard,
    WorkerObservation,
)
from ant.indexing.cards import template_routing_summary
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
    assert state.rounds[0].node_executions
    assert state.rounds[0].node_executions[0].worker_ids == ["worker-src"]


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


def test_query_from_needs_caps_a_verbose_missing_and_description_field() -> None:
    # Regression test: check_need_resolution's refined_need.missing/
    # description are free text the LLM is asked to write more
    # specifically each round, with no length control -- confirmed on a
    # real qibo trace, this became an 800+ character narrative paragraph
    # (once even a stringified list of sub-questions) as the literal
    # search()/dense_search() query text, round over round, diluting
    # unweighted BM25 term-overlap scoring instead of sharpening it.
    long_missing = (
        "We now know FALQON is a subclass of QAOA, but it is still unclear which "
        "methods, if any, FALQON or any other QAOA subclasses override or extend. "
        "There is also still no information about other possible subclasses of QAOA "
        "or their method specializations, and this paragraph keeps going well past "
        "two hundred characters on purpose to prove the cap actually bites."
    )
    assert len(long_missing) > 200
    need = UnresolvedNeed(description="short description", missing=long_missing)

    query = LocalCoordinator._query_from_needs("question", [need])

    assert long_missing not in query
    assert "short description" in query
    # The missing text's own contribution is capped well below its
    # original length -- not just "somewhat shorter", genuinely bounded.
    assert len(query) < len(long_missing)


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


def test_routing_matcher_stems_against_a_compound_part_not_just_the_whole_term() -> None:
    # Regression test for a real seaborn trace: the symbol
    # `assign_variables_wideform` compound-splits to {"assign", "variables",
    # "wideform"} -- "wideform" stays one token because the source didn't
    # underscore-separate "wide" and "form". A query term "wide" used to
    # match neither the whole term (doesn't start with "wide") nor the
    # compound-part set (exact membership only), while a tutorial file
    # literally named `wide_form_violinplot.py` (which DOES underscore-split
    # into "wide"/"form"/"violinplot") got credit the real implementation
    # never could -- a token-boundary accident, not a real relevance
    # difference. Only the prefix side of this closes: "wide" is a >=4-char
    # prefix of "wideform", so it now matches; "form" is a suffix, not a
    # prefix, of "wideform", and is_stem_match is deliberately prefix-only
    # (a suffix-only match would make "form" match "platform"/"uniform"/
    # "transform" too, trading one narrow miss for a much broader false-
    # positive class), so it still does not match here -- this fix recovers
    # part of the routing signal, not all of it.
    assert _matches_term("wide", {"assign_variables_wideform"})
    assert not _matches_term("form", {"assign_variables_wideform"})
    # Must not match an unrelated compound part on a short accidental prefix
    # (same 4-char stem-match floor as the whole-term case above).
    assert not _matches_term("was", {"assign_variables_wideform"})


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
    assert state.rounds[0].node_executions[0].observations[0].worker_id == "worker-docs"


def test_cross_repo_experience_reaches_plan_round(tmp_path: Path) -> None:
    # Pre-fetched by the caller (see GlobalMemoryStore.retrieve_similar) and
    # passed straight through to every round's plan_round() call as
    # reference text -- LocalCoordinator itself never touches
    # GlobalMemoryStore directly (same separation as memory_routes).
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text("def authenticate():\n    pass\n", encoding="utf-8")
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/auth.py"]
    )
    received: list[list[str]] = []

    class _RecordingReasoner(_PassthroughLookupsReasoner):
        def observe(self, *, question, worker_id, territory_id, evidence):
            raise AssertionError("this test does not exercise observe()")

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
            observed_needs,
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            worker_relevance_rank=None,
        ):
            received.append(cross_repo_experience)
            return RoundPlan()

    LocalCoordinator(
        tmp_path,
        [worker],
        reasoner=_RecordingReasoner(),
        cross_repo_experience=["a transferable pattern from another repo"],
    ).ask("Where is authenticate?", max_rounds=1)

    assert received == [["a transferable pattern from another repo"]]


def test_ask_threads_a_retrieval_based_worker_relevance_rank_into_plan_round(
    tmp_path: Path,
) -> None:
    # ant.coordinator.worker_retrieval.rank_workers is built from
    # WorkerCard.symbols (the AST's real, non-truncated definition list),
    # not searchable_terms -- confirm ask() actually computes and threads
    # this rank through to plan_round() every round, and that the worker
    # whose symbol table contains the question's own rare term ranks
    # ahead of an unrelated sibling.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "circuit.py").write_text("class FALQON:\n    pass\n", encoding="utf-8")
    (tmp_path / "src" / "other.py").write_text("class Unrelated:\n    pass\n", encoding="utf-8")
    target = WorkerCard(
        id="worker-models",
        territory_id="models",
        name="models",
        root="src",
        files=["src/circuit.py"],
        symbols=[
            CodeSymbol(
                name="FALQON", kind="class", path="src/circuit.py", line=1, qualname="FALQON"
            )
        ],
    )
    sibling = WorkerCard(
        id="worker-other",
        territory_id="other",
        name="other",
        root="src",
        files=["src/other.py"],
        symbols=[
            CodeSymbol(
                name="Unrelated", kind="class", path="src/other.py", line=1, qualname="Unrelated"
            )
        ],
    )
    received: list[dict[str, int] | None] = []

    class _RecordingReasoner(_PassthroughLookupsReasoner):
        def observe(self, *, question, worker_id, territory_id, evidence):
            raise AssertionError("this test does not exercise observe()")

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
            observed_needs,
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            worker_relevance_rank=None,
        ):
            received.append(worker_relevance_rank)
            return RoundPlan()

    LocalCoordinator(tmp_path, [target, sibling], reasoner=_RecordingReasoner()).ask(
        "What class is FALQON?", max_rounds=1
    )

    # worker-other genuinely has no retrieval signal for this query (its
    # corpus -- symbol/file/responsibility text -- shares no term with
    # "FALQON"), so it correctly has no entry at all (see rank_workers'
    # docstring: no entry means "no signal", not "worst possible").
    # worker-models, whose symbol table contains the query's own rare
    # term, must be ranked.
    assert received
    assert received[0]
    assert received[0].get("worker-models") == 1
    assert "worker-other" not in received[0]


def _numbered_workers(tmp_path: Path, count: int) -> list[WorkerCard]:
    (tmp_path / "src").mkdir(exist_ok=True)
    workers = []
    for i in range(count):
        relative = f"src/m{i:02d}.py"
        (tmp_path / relative).write_text(f"class Term{i:02d}:\n    pass\n", encoding="utf-8")
        workers.append(
            WorkerCard(
                id=f"worker-{i:02d}",
                territory_id=f"t{i:02d}",
                name=f"w{i:02d}",
                root="src",
                files=[relative],
                symbols=[
                    CodeSymbol(
                        name=f"Term{i:02d}",
                        kind="class",
                        path=relative,
                        line=1,
                        qualname=f"Term{i:02d}",
                    )
                ],
            )
        )
    return workers


def test_candidate_workers_for_round_narrows_a_fresh_ready_need(tmp_path: Path) -> None:
    # Two-stage routing: retrieval gets structural authority over the
    # candidate set for a fresh (rounds_without_progress == 0) ready need
    # -- capped at _FRESH_NEED_CANDIDATE_LIMIT, not the full worker list.
    # 12 workers, each with one symbol matching exactly one query term
    # (uniform signal, more than either candidate limit), so truncation is
    # actually exercised rather than accidentally passing because there
    # weren't enough ranked candidates to truncate in the first place.
    workers = _numbered_workers(tmp_path, 12)
    question = " ".join(f"term{i:02d}" for i in range(12))
    coordinator = LocalCoordinator(tmp_path, workers)
    root_node = NeedNode(need_id="root", need=question, detail=UnresolvedNeed(description=question))
    graph = NeedGraph(nodes={"root": root_node})
    frontier = FrontierResult(ready=["root"], blocked=[], stuck_subgraphs=[])
    worker_index = build_worker_index(workers)

    candidates, ranks, per_need = coordinator._candidate_workers_for_round(
        question, graph, frontier, worker_index, None
    )

    assert len(candidates) == _FRESH_NEED_CANDIDATE_LIMIT
    assert len(ranks) == _FRESH_NEED_CANDIDATE_LIMIT
    assert per_need["root"] == ranks


def test_candidate_workers_for_round_widens_after_one_quiet_round(tmp_path: Path) -> None:
    # Same need, but rounds_without_progress == 1 -- still on the ready
    # frontier (_STUCK_THRESHOLD == 2), gets the wider escalation limit.
    workers = _numbered_workers(tmp_path, 12)
    question = " ".join(f"term{i:02d}" for i in range(12))
    coordinator = LocalCoordinator(tmp_path, workers)
    node = NeedNode(
        need_id="root",
        need=question,
        detail=UnresolvedNeed(description=question),
        rounds_without_progress=1,
    )
    graph = NeedGraph(nodes={"root": node})
    frontier = FrontierResult(ready=["root"], blocked=[], stuck_subgraphs=[])
    worker_index = build_worker_index(workers)

    candidates, ranks, per_need = coordinator._candidate_workers_for_round(
        question, graph, frontier, worker_index, None
    )

    assert len(candidates) == _ESCALATED_NEED_CANDIDATE_LIMIT
    assert len(ranks) == _ESCALATED_NEED_CANDIDATE_LIMIT


def test_candidate_workers_for_round_falls_back_to_full_list_with_no_retrieval_signal(
    tmp_path: Path,
) -> None:
    # A need must never end up with zero candidates just because
    # retrieval found nothing for its (here: all-stopword) query text --
    # same "never let a filter zero out a legitimate scope" principle as
    # tonight's _territory_index corpus-exclusion fix, one level up.
    workers = _numbered_workers(tmp_path, 12)
    coordinator = LocalCoordinator(tmp_path, workers)
    node = NeedNode(
        need_id="root", need="the and of", detail=UnresolvedNeed(description="the and of")
    )
    graph = NeedGraph(nodes={"root": node})
    frontier = FrontierResult(ready=["root"], blocked=[], stuck_subgraphs=[])
    worker_index = build_worker_index(workers)

    candidates, ranks, per_need = coordinator._candidate_workers_for_round(
        "the and of", graph, frontier, worker_index, None
    )

    assert len(candidates) == len(workers)
    assert len(ranks) == len(workers)
    assert len(per_need["root"]) == len(workers)


def test_candidate_workers_for_round_shows_everyone_when_a_stuck_subgraph_exists(
    tmp_path: Path,
) -> None:
    # A stuck need's ordinary reassignment still needs full visibility,
    # unchanged from before this change -- only ready-frontier needs are
    # narrowed.
    workers = _numbered_workers(tmp_path, 12)
    question = " ".join(f"term{i:02d}" for i in range(12))
    coordinator = LocalCoordinator(tmp_path, workers)
    fresh_node = NeedNode(
        need_id="fresh", need=question, detail=UnresolvedNeed(description=question)
    )
    stuck_node = NeedNode(
        need_id="stuck",
        need="stuck thing",
        detail=UnresolvedNeed(description="stuck thing"),
        progress="stuck",
    )
    graph = NeedGraph(nodes={"fresh": fresh_node, "stuck": stuck_node})
    frontier = FrontierResult(ready=["fresh"], blocked=[], stuck_subgraphs=[["stuck"]])
    worker_index = build_worker_index(workers)

    candidates, ranks, per_need = coordinator._candidate_workers_for_round(
        question, graph, frontier, worker_index, None
    )

    assert len(candidates) == len(workers)
    # The fresh need's own top-K is still recorded for audit purposes even
    # though the round-level union ends up showing everyone.
    assert len(per_need["fresh"]) == _FRESH_NEED_CANDIDATE_LIMIT


def test_ask_narrows_plan_round_workers_and_records_candidates_in_the_trace(
    tmp_path: Path,
) -> None:
    # End-to-end: ask()'s round loop actually calls
    # _candidate_workers_for_round and threads its output both into
    # plan_round's `workers` argument (this is what gives the narrowing
    # real teeth -- _parse_round_plan already validates every assignment's
    # worker_id against exactly this list, see
    # test_plan_round_drops_an_assignment_to_a_worker_id_not_in_the_candidate_list
    # in test_openai_provider.py) and into NodeExecutionTrace's new audit
    # fields.
    workers = _numbered_workers(tmp_path, 12)
    question = " ".join(f"term{i:02d}" for i in range(12))
    captured_worker_counts: list[int] = []

    class _Reasoner(_PassthroughLookupsReasoner):
        def observe(self, *, question, worker_id, territory_id, evidence):
            return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

        def check_need_resolution(self, *, need, new_evidence, question):
            return NeedResolution(status="resolved")

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
            observed_needs,
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            worker_relevance_rank=None,
        ):
            captured_worker_counts.append(len(workers))
            return RoundPlan(assignments={need_id: [workers[0].id] for need_id in frontier.ready})

    state = LocalCoordinator(tmp_path, workers, reasoner=_Reasoner()).ask(question, max_rounds=1)

    assert captured_worker_counts == [_FRESH_NEED_CANDIDATE_LIMIT]
    execution = state.rounds[0].node_executions[0]
    assert execution.need_id == "root"
    assert len(execution.candidate_worker_ids) == _FRESH_NEED_CANDIDATE_LIMIT
    assert execution.candidate_worker_ranks


class _PassthroughLookupsReasoner:
    """Shared by every scenario-specific test reasoner below: they each
    exist to test observe()'s behavior for one routing/coalition scenario,
    not select_lookups(), so this just satisfies the WorkerReasoner protocol
    with the same no-op AutonomousWorker itself uses when no reasoner is
    configured -- no filtering, so a scenario's expected tool calls/evidence
    aren't affected by symbol-lookup filtering it isn't testing.
    """

    def select_lookups(self, *, need, evidence, candidates):
        return candidates

    def select_workers(self, *, query, need, candidates, limit, memory_hints):
        return [worker.id for worker in candidates]

    def select_evidence(self, *, question, evidence, limit):
        return [str(index) for index in range(len(evidence))][:limit], []

    def plan_worker_actions(
        self, *, need, evidence, candidate_symbols, available_tools, hints, max_actions
    ):
        # Mirrors the pre-existing fixed tool sequence (inheritance/flow
        # hints first, then navigate/references/callers/callees/assignments
        # per symbol) as a plan instead of inline execution, so scenarios
        # written against the old always-run-every-tool behavior keep
        # exercising the same evidence-gathering shape.
        plan: list[tuple[str, str]] = []
        available = set(available_tools)
        is_inheritance = any("inheritance" in hint.lower() for hint in hints)
        is_flow = any("flow" in hint.lower() or "call path" in hint.lower() for hint in hints)
        if is_inheritance and "subclasses" in available:
            plan.extend(("subclasses", symbol) for symbol in candidate_symbols)
        if is_flow:
            for symbol in candidate_symbols[:4]:
                plan.extend(
                    (tool, symbol)
                    for tool in ("imports", "callers", "assignments")
                    if tool in available
                )
        for symbol in candidate_symbols:
            plan.extend(
                (tool, symbol)
                for tool in ("navigate", "references", "callers", "callees", "assignments")
                if tool in available
            )
        return plan[:max_actions]

    def should_continue_recruiting(self, *, question, need, evidence, rounds_completed):
        return True

    def check_need_resolution(self, *, need, new_evidence, question):
        # Always "unresolved": these scenario reasoners exist to test one
        # specific routing/coalition behavior each, not the new resolution-
        # check/escalation layer -- returning "unresolved" keeps every
        # existing scenario's need lifecycle exactly as it was before this
        # method existed (matches MockLLMProvider's own rationale).
        return NeedResolution(status="unresolved")

    def decide_local_action(self, *, need, evidence, worker_progress, worker):
        # Always "continue": same rationale as MockLLMProvider -- these
        # scenario reasoners exist to test one specific behavior each, not
        # the local-continuation decision layer.
        return "continue"

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
        observed_needs,
        incomplete_parents,
        cross_repo_experience,
        validation_feedback="",
        repair_guidance="",
        stuck_tried_workers=None,
        worker_relevance_rank=None,
    ):
        # TODO(Phase 7): every scenario reasoner below this class exercises
        # the old routing/escalation-specific ask() mechanics, which the
        # Phase 6 graph-based round loop replaced outright -- ask() now
        # always calls plan_round(), so these scenario tests need
        # rewriting against the new pipeline (or removing, where they only
        # ever tested since-deleted heuristics), not just a signature fix.
        raise AssertionError(
            "this scenario reasoner predates the Phase 6 graph-based round loop and needs "
            "rewriting against it (see TODO) -- plan_round() is unconditionally called now"
        )

    def summarize_task_experience(self, *, question, rounds, unresolved_needs, evidence_count):
        raise AssertionError("this test does not exercise summarize_task_experience()")


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


def test_rank_global_evidence_credits_a_symbol_name_matching_the_question() -> None:
    # Coordinator-level counterpart to the same fix in
    # ant.workers.autonomous._rank_evidence: the final cross-worker ranking
    # pass must also credit an Evidence item's carried symbol name, not just
    # its literal quote text, or a correctly-recruited method whose name is
    # the only thing matching the question loses to unrelated siblings tied
    # at the same baseline score.
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

    ranked = _rank_global_evidence([unrelated, target], "Where is drawing implemented?")

    assert ranked[0] is target


def test_build_temporary_bridge_gets_a_nonempty_routing_summary_via_template() -> None:
    # Ephemeral, never persisted -- not worth an LLM call -- but it still
    # becomes a candidate in the same round it's built, and the
    # Orchestrator planning call reads only routing_summary, not the full
    # card, so it must not be left at the WorkerCard default "".
    tried = [
        WorkerCard(
            id="worker-a",
            territory_id="a",
            name="a",
            root="a",
            files=["a.py"],
            searchable_terms=["alpha"],
        ),
        WorkerCard(
            id="worker-b",
            territory_id="b",
            name="b",
            root="b",
            files=["b.py"],
            searchable_terms=["beta"],
        ),
    ]

    bridge = _build_temporary_bridge(tried)

    assert bridge.routing_summary
    assert bridge.routing_summary == template_routing_summary(
        bridge.model_copy(update={"routing_summary": ""})
    )


def test_select_evidence_shows_the_reasoner_every_item_with_no_relevance_cap(
    tmp_path: Path,
) -> None:
    # Regression-shaped: a previous version scored evidence with
    # score_evidence()/_rank_global_evidence and cut to a fixed top-40
    # before the reasoner ever saw the rest. Verified empirically on real
    # sanic/yt-dlp traces that this cap discarded relevant evidence the
    # reasoner never got a chance to judge -- _select_evidence must now
    # pass every deduped item through, regardless of pool size.
    seen_counts: list[int] = []

    class _RecordingReasoner(_PassthroughLookupsReasoner):
        def observe(self, *, question, worker_id, territory_id, evidence):
            raise AssertionError("this test does not exercise observe()")

        def select_evidence(self, *, question, evidence, limit):
            seen_counts.append(len(evidence))
            return [str(index) for index in range(len(evidence))][:limit], []

    evidence = [
        Evidence(
            path=f"src/file_{index}.py",
            line_start=1,
            line_end=2,
            quote=f"def function_{index}(): pass",
            reason="match",
        )
        for index in range(55)  # well past the old 40-item pool cap
    ]

    _select_evidence(_RecordingReasoner(), "question", evidence, LocalSearchTool(tmp_path))

    assert seen_counts == [55]


def test_plan_round_accepts_an_acyclic_plan_without_retrying() -> None:
    graph = NeedGraph(
        nodes={
            "n1": NeedNode(
                need_id="n1", need="n1 text", detail=UnresolvedNeed(description="n1 text")
            )
        }
    )
    frontier = FrontierResult(ready=["n1"], blocked=[], stuck_subgraphs=[])
    calls: list[str] = []

    class _AcyclicReasoner(_PassthroughLookupsReasoner):
        def observe(self, *, question, worker_id, territory_id, evidence):
            raise AssertionError("this test does not exercise observe()")


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
            observed_needs,
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            worker_relevance_rank=None,
        ):
            calls.append(validation_feedback)
            return RoundPlan(assignments={"n1": ["worker-a"]})

    plan = _plan_round_with_cycle_validation(
        _AcyclicReasoner(),
        question="q",
        graph=graph,
        resolution_results={},
        evidence=[],
        workers=[],
        memory_hints={},
        frontier=frontier,
        observed_needs=[],
        incomplete_parents=[],
        cross_repo_experience=[],
    )

    assert plan.assignments == {"n1": ["worker-a"]}
    assert calls == [""]  # exactly one call, no retry


def test_plan_round_rejects_a_cyclic_plan_and_retries_with_the_cycle_described() -> None:
    graph = NeedGraph(nodes={})
    frontier = FrontierResult(ready=[], blocked=[], stuck_subgraphs=[])
    calls: list[str] = []

    class _FixesOnRetryReasoner(_PassthroughLookupsReasoner):
        def observe(self, *, question, worker_id, territory_id, evidence):
            raise AssertionError("this test does not exercise observe()")


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
            observed_needs,
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            worker_relevance_rank=None,
        ):
            calls.append(validation_feedback)
            if not validation_feedback:
                # First attempt: a genuine self-loop cycle.
                return RoundPlan(
                    graph_updates={
                        "n1": NeedNode(
                            need_id="n1",
                            need="n1",
                            depends_on=["n1"],
                            detail=UnresolvedNeed(description="n1"),
                        )
                    }
                )
            # Retry, informed by the feedback: acyclic.
            return RoundPlan(
                graph_updates={
                    "n1": NeedNode(need_id="n1", need="n1", detail=UnresolvedNeed(description="n1"))
                }
            )

    plan = _plan_round_with_cycle_validation(
        _FixesOnRetryReasoner(),
        question="q",
        graph=graph,
        resolution_results={},
        evidence=[],
        workers=[],
        memory_hints={},
        frontier=frontier,
        observed_needs=[],
        incomplete_parents=[],
        cross_repo_experience=[],
    )

    assert len(calls) == 2
    assert calls[0] == ""
    assert "cycle" in calls[1].lower()
    assert plan.graph_updates["n1"].depends_on == []


def test_plan_round_accepts_the_retry_even_if_it_is_still_cyclic() -> None:
    # Bounded worst case: exactly one retry, then accept whatever comes
    # back even if still cyclic -- a round must not loop indefinitely
    # waiting for the reasoner to self-correct.
    graph = NeedGraph(nodes={})
    frontier = FrontierResult(ready=[], blocked=[], stuck_subgraphs=[])
    calls: list[str] = []

    class _NeverFixesReasoner(_PassthroughLookupsReasoner):
        def observe(self, *, question, worker_id, territory_id, evidence):
            raise AssertionError("this test does not exercise observe()")


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
            observed_needs,
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            worker_relevance_rank=None,
        ):
            calls.append(validation_feedback)
            return RoundPlan(
                graph_updates={
                    "n1": NeedNode(
                        need_id="n1",
                        need="n1",
                        depends_on=["n1"],
                        detail=UnresolvedNeed(description="n1"),
                    )
                }
            )

    plan = _plan_round_with_cycle_validation(
        _NeverFixesReasoner(),
        question="q",
        graph=graph,
        resolution_results={},
        evidence=[],
        workers=[],
        memory_hints={},
        frontier=frontier,
        observed_needs=[],
        incomplete_parents=[],
        cross_repo_experience=[],
    )

    assert len(calls) == 2
    assert plan.graph_updates["n1"].depends_on == ["n1"]




class _AlwaysStuckAndAlwaysProposesBridgeReasoner(_PassthroughLookupsReasoner):
    """Regression fixture for a real bug found on a live qibo trace: a stuck
    need got temporary_bridge'd 4+ consecutive rounds, almost all with
    ev_gain=0, because the old recovery-streak keying (a "root" recomputed
    fresh from compute_frontier()'s stuck_subgraphs every round) silently
    drifted and never accumulated to the abandonment threshold. This
    reasoner reproduces the failure condition directly: check_need_resolution
    NEVER advances (guaranteed stuck), and plan_round ALWAYS re-proposes
    temporary_bridge for every currently-stuck need_id, every round, for as
    long as the coordinator keeps showing it one.
    """

    def observe(self, *, question, worker_id, territory_id, evidence):
        return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

    def check_need_resolution(self, *, need, new_evidence, question):
        return NeedResolution(status="unresolved")

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
        observed_needs,
        incomplete_parents,
        cross_repo_experience,
        validation_feedback="",
        repair_guidance="",
        stuck_tried_workers=None,
        worker_relevance_rank=None,
    ):
        assignments = {need_id: [workers[0].id] for need_id in frontier.ready}
        special_tactics = {
            need_id: "temporary_bridge"
            for group in frontier.stuck_subgraphs
            for need_id in group
        }
        return RoundPlan(assignments=assignments, special_tactics=special_tactics)


def test_a_stuck_need_gets_abandoned_after_three_failed_recoveries_not_retried_forever(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("def unrelated():\n    pass\n", encoding="utf-8")
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )

    state = LocalCoordinator(
        tmp_path,
        [worker],
        reasoner=_AlwaysStuckAndAlwaysProposesBridgeReasoner(),
    ).ask("What is the meaning of this codebase?", max_rounds=10)

    bridge_traces = [
        trace
        for round_state in state.rounds
        for trace in round_state.node_executions
        if trace.special_tactic == "temporary_bridge"
    ]
    real_bridge_runs = [trace for trace in bridge_traces if trace.worker_ids]

    # Proposed at most _MAX_CONSECUTIVE_FAILED_RECOVERIES (3) times total --
    # the 4th proposal the old bug allowed must never happen, because the
    # need is abandoned (excluded from frontier.stuck_subgraphs) right after
    # the 3rd failed attempt.
    assert len(bridge_traces) <= 3
    # And only the FIRST of those actually ran a worker -- every repeat
    # proposal for the same episode is deduped (no worker_ids, no tool
    # calls spent) rather than rebuilding and re-running an identical
    # bridge against an unchanged territory.
    assert len(real_bridge_runs) == 1

    # The task terminates instead of looping until max_rounds regardless:
    # once the stuck need is abandoned, nothing is left in the frontier for
    # it, so ask() stops well before round 10.
    assert len(state.rounds) < 10


def test_final_recovery_state_records_the_abandoned_need(tmp_path: Path) -> None:
    # EvidenceState.final_recovery_state must actually reflect an
    # abandonment, not just exist with empty defaults -- reuses the same
    # always-stuck scenario as the recovery-streak regression test above,
    # this time asserting on the new trajectory snapshot instead of the
    # round traces.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("def unrelated():\n    pass\n", encoding="utf-8")
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )

    state = LocalCoordinator(
        tmp_path,
        [worker],
        reasoner=_AlwaysStuckAndAlwaysProposesBridgeReasoner(),
    ).ask("What is the meaning of this codebase?", max_rounds=10)

    assert "root" in state.final_recovery_state.abandoned_node_ids
    assert "root" in state.final_need_graph
    # The abandoned episode's bookkeeping must have actually run (not just
    # left at zero/empty defaults) before the episode got deleted on
    # abandonment -- stuck_episodes itself is empty afterward (see
    # RecoveryState's own abandonment cleanup), so this asserts on the
    # trace evidence of that instead: at least one real (non-deduped)
    # temporary_bridge execution happened before the abandonment fired.
    real_bridge_runs = [
        trace
        for round_state in state.rounds
        for trace in round_state.node_executions
        if trace.special_tactic == "temporary_bridge" and trace.worker_ids
    ]
    assert len(real_bridge_runs) == 1


class _DecomposesRootIntoTwoChildrenReasoner(_PassthroughLookupsReasoner):
    """Round 0: splits root into child-a (no deps) and child-b (depends on
    child-a), assigning nothing. Round 1: edits child-b's depends_on to add
    a made-up extra dependency, purely to exercise GraphDelta.
    dependency_changes on an *existing* node (as opposed to a newly-created
    one's initial edges, which show up via created_nodes instead).
    """

    def __init__(self) -> None:
        self._round = 0

    def observe(self, *, question, worker_id, territory_id, evidence):
        return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

    def check_need_resolution(self, *, need, new_evidence, question):
        return NeedResolution(status="unresolved")

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
        observed_needs,
        incomplete_parents,
        cross_repo_experience,
        validation_feedback="",
        repair_guidance="",
        stuck_tried_workers=None,
        worker_relevance_rank=None,
    ):
        self._round += 1
        if self._round == 1:
            root = graph.nodes["root"]
            return RoundPlan(
                graph_updates={
                    "root": root.model_copy(update={"children": ["child-a", "child-b"]}),
                    "child-a": NeedNode(
                        need_id="child-a", need="a", detail=UnresolvedNeed(description="a")
                    ),
                    "child-b": NeedNode(
                        need_id="child-b",
                        need="b",
                        depends_on=["child-a"],
                        detail=UnresolvedNeed(description="b"),
                    ),
                }
            )
        child_b = graph.nodes["child-b"]
        return RoundPlan(
            graph_updates={
                "child-b": child_b.model_copy(update={"depends_on": ["child-a", "root"]}),
            }
        )


def test_graph_delta_records_created_nodes_children_and_dependency_changes(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("def f():\n    pass\n", encoding="utf-8")
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )

    state = LocalCoordinator(
        tmp_path, [worker], reasoner=_DecomposesRootIntoTwoChildrenReasoner()
    ).ask("root question", max_rounds=2)

    assert len(state.rounds) == 2
    round0, round1 = state.rounds[0].graph_delta, state.rounds[1].graph_delta

    assert sorted(round0.created_nodes) == ["child-a", "child-b"]
    assert round0.created_children == {"root": ["child-a", "child-b"]}
    # child-b's initial depends_on=["child-a"] is part of its creation
    # (created_nodes), not a "change" -- only an already-existing node's
    # depends_on being edited counts as dependency_changes.
    assert round0.dependency_changes == {}

    assert round1.created_nodes == []
    assert round1.dependency_changes == {"child-b": ["child-a", "root"]}
    assert round1.created_children == {}

    assert state.final_need_graph["root"].children == ["child-a", "child-b"]
    assert state.final_need_graph["child-b"].depends_on == ["child-a", "root"]


class _AssignsSameWorkerToTwoIndependentNeedsReasoner(_PassthroughLookupsReasoner):
    """Regression fixture for a real bug found on a live qibo trace
    (question: "statistical sampling architecture"): the same worker was
    independently assigned to several different need_ids across rounds and,
    each time, rediscovered the same symbol's definition via its own
    navigate/references lookups -- producing literal (path, line_start,
    line_end, quote) duplicates in the final evidence pool (one 2-line
    `set_seed` definition kept 4 times), crowding out genuinely distinct
    evidence the same run had also found elsewhere. This reasoner creates
    that exact shape directly: round 1 assigns `root` to the only worker;
    round 2 adds one independent leaf need and assigns it to the *same*
    worker, which searches the *same* one-symbol file again.
    """

    def __init__(self) -> None:
        self._added_aux_need = False

    def observe(self, *, question, worker_id, territory_id, evidence):
        return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

    def check_need_resolution(self, *, need, new_evidence, question):
        return NeedResolution(status="unresolved")

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
        observed_needs,
        incomplete_parents,
        cross_repo_experience,
        validation_feedback="",
        repair_guidance="",
        stuck_tried_workers=None,
        worker_relevance_rank=None,
    ):
        worker_id = workers[0].id
        if not self._added_aux_need:
            self._added_aux_need = True
            return RoundPlan(
                assignments={"root": [worker_id]},
                graph_updates={
                    "aux-need": NeedNode(
                        need_id="aux-need",
                        need="Independently re-inspect the same symbol.",
                        detail=UnresolvedNeed(
                            description="Independently re-inspect the same symbol."
                        ),
                    )
                },
            )
        return RoundPlan(assignments={"aux-need": [worker_id]})


def test_final_evidence_pool_is_deduped_even_without_any_inheritance_evidence(
    tmp_path: Path,
) -> None:
    # Regression test: _dedupe_evidence on the final evidence pool used to
    # only run inside `if inheritance_evidence:` (local.py, right before
    # _rank_global_evidence/_select_evidence) -- for any question that
    # isn't a subclass/inheritance lookup (the overwhelming majority),
    # inheritance_evidence is empty and that branch never executes, so the
    # raw pool -- which every worker execution appends to via
    # evidence.extend() with no dedup of its own -- reached synthesis with
    # literal duplicates whenever two different need executions
    # independently rediscovered the same (path, line_start, line_end,
    # quote) region. _dedupe_evidence itself was never broken; it just
    # wasn't being called on this path.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "backend.py").write_text(
        "def set_seed(seed):\n    np.random.seed(seed)\n",
        encoding="utf-8",
    )
    worker = WorkerCard(
        id="worker-src",
        territory_id="src",
        name="src worker",
        root="src",
        searchable_terms=["set_seed"],
        files=["src/backend.py"],
    )

    state = LocalCoordinator(
        tmp_path, [worker], reasoner=_AssignsSameWorkerToTwoIndependentNeedsReasoner()
    ).ask("Where is set_seed defined for reproducible sampling?", max_rounds=2)

    keys = [(item.path, item.line_start, item.line_end, item.quote) for item in state.evidence]
    assert keys, "expected the worker to find real evidence in this test"
    assert len(keys) == len(set(keys)), (
        f"duplicate (path, lines, quote) evidence reached synthesis: {keys}"
    )


def test_closure_check_survives_a_partial_verdict_creating_a_gap_node(tmp_path: Path) -> None:
    # Regression test: a "partial" closure verdict inserts a new gap node
    # straight into graph.nodes (local.py's closure-check block) while
    # that same block is iterating graph.nodes.values() -- this used to
    # raise "RuntimeError: dictionary changed size during iteration",
    # confirmed live on a real retry_from_trajectory run against seaborn
    # (a parent whose only child resolved this round, closure check
    # returned partial). A single leaf resolving into a parent whose
    # closure check comes back partial is enough to reproduce it -- the
    # crash happens on the iterator's very next() call after the
    # in-progress iteration mutates the dict, regardless of how many
    # other nodes are also present.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "def target_function():\n    pass\n", encoding="utf-8"
    )
    worker = WorkerCard(
        id="worker-src",
        territory_id="src",
        name="src",
        root="src",
        searchable_terms=["target_function"],
        files=["src/mod.py"],
    )
    seeded_graph = NeedGraph(
        nodes={
            "root": NeedNode(
                need_id="root",
                need="root question",
                children=["leaf"],
                detail=UnresolvedNeed(description="root question"),
            ),
            "leaf": NeedNode(
                need_id="leaf",
                need="Where is target_function defined?",
                detail=UnresolvedNeed(description="Where is target_function defined?"),
            ),
        }
    )

    class _ResolvesLeafThenPartialsRootReasoner(_PassthroughLookupsReasoner):
        def observe(self, *, question, worker_id, territory_id, evidence):
            return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

        def check_need_resolution(self, *, need, new_evidence, question):
            if need.description == "root question":
                return NeedResolution(
                    status="partial",
                    refined_need=UnresolvedNeed(description="a refined follow-up need"),
                )
            return NeedResolution(status="resolved" if new_evidence else "unresolved")

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
            observed_needs,
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            worker_relevance_rank=None,
        ):
            return RoundPlan(assignments={need_id: [workers[0].id] for need_id in frontier.ready})

    state = LocalCoordinator(
        tmp_path, [worker], reasoner=_ResolvesLeafThenPartialsRootReasoner()
    ).ask(
        "root question",
        max_rounds=1,
        initial_graph=seeded_graph,
    )

    assert state.rounds, "expected round 0 to actually run and hit the closure check"
    assert "root" in state.final_need_graph
    assert any(
        need_id.startswith("root-gap-") for need_id in state.final_need_graph
    ), "expected the partial verdict to have created a gap node"


def test_ask_with_seeded_initial_state_only_works_the_unresolved_part(tmp_path: Path) -> None:
    # LocalCoordinator.ask()'s new initial_graph/initial_evidence params
    # (added for retry_from_trajectory) must be genuinely usable standalone
    # too: seeding an already-resolved leaf alongside an unresolved one
    # dependent on it should never re-assign the resolved one (nothing to
    # do -- it's not in the ready frontier), and the seeded evidence must
    # survive to the final state untouched.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "def target_function():\n    pass\n", encoding="utf-8"
    )
    worker = WorkerCard(
        id="worker-src",
        territory_id="src",
        name="src",
        root="src",
        searchable_terms=["target_function"],
        files=["src/mod.py"],
    )
    seeded_evidence = [
        Evidence(
            path="root.py", line_start=1, line_end=1, quote="x", reason="from prior attempt"
        )
    ]
    seeded_graph = NeedGraph(
        nodes={
            "root": NeedNode(
                need_id="root",
                need="root question",
                resolution="resolved",
                detail=UnresolvedNeed(description="root question"),
            ),
            "child": NeedNode(
                need_id="child",
                need="Where is target_function defined?",
                depends_on=["root"],
                detail=UnresolvedNeed(description="Where is target_function defined?"),
            ),
        }
    )
    assigned_need_ids: list[str] = []

    class _RecordsAssignmentsReasoner(_PassthroughLookupsReasoner):
        def observe(self, *, question, worker_id, territory_id, evidence):
            return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

        def check_need_resolution(self, *, need, new_evidence, question):
            return NeedResolution(status="resolved" if new_evidence else "unresolved")

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
            observed_needs,
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            worker_relevance_rank=None,
        ):
            assigned_need_ids.extend(frontier.ready)
            return RoundPlan(assignments={need_id: [workers[0].id] for need_id in frontier.ready})

    state = LocalCoordinator(tmp_path, [worker], reasoner=_RecordsAssignmentsReasoner()).ask(
        "root question",
        max_rounds=2,
        initial_graph=seeded_graph,
        initial_evidence=seeded_evidence,
    )

    assert "root" not in assigned_need_ids
    assert "child" in assigned_need_ids
    assert any(item.reason == "from prior attempt" for item in state.evidence)


def test_ask_forces_the_given_assignment_at_round_0_only(tmp_path: Path) -> None:
    # forced_first_round_assignments (added for retry_from_trajectory's
    # execution-policy repair actions -- reuse_assignment/
    # replace_assignment/form_local_bridge) must override whatever the
    # Orchestrator itself proposes at round 0, and must NOT keep
    # overriding on later rounds -- the Orchestrator regains ordinary
    # freedom from round 1 onward.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def a():\n    pass\n", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("def b():\n    pass\n", encoding="utf-8")
    worker_a = WorkerCard(
        id="worker-a", territory_id="a", name="a", root="src", files=["src/a.py"]
    )
    worker_b = WorkerCard(
        id="worker-b", territory_id="b", name="b", root="src", files=["src/b.py"]
    )

    class _AlwaysPicksWorkerAAndNeverResolvesReasoner(_PassthroughLookupsReasoner):
        def observe(self, *, question, worker_id, territory_id, evidence):
            return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

        def check_need_resolution(self, *, need, new_evidence, question):
            return NeedResolution(status="unresolved")

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
            observed_needs,
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            worker_relevance_rank=None,
        ):
            return RoundPlan(assignments={need_id: ["worker-a"] for need_id in frontier.ready})

    state = LocalCoordinator(
        tmp_path, [worker_a, worker_b], reasoner=_AlwaysPicksWorkerAAndNeverResolvesReasoner()
    ).ask(
        "question",
        max_rounds=2,
        forced_first_round_assignments={"root": ["worker-b"]},
    )

    assert state.rounds[0].node_executions[0].worker_ids == ["worker-b"]
    assert state.rounds[1].node_executions[0].worker_ids == ["worker-a"]


def test_ask_forces_a_global_search_at_round_0_with_no_stuck_episode_needed(
    tmp_path: Path,
) -> None:
    # forced_first_round_global_search_ids (force_global_search's forced
    # execution) must run even though the ordinary special_tactics
    # executor path requires a RecoveryState stuck episode to exist for
    # that need_id (_episode_for_need) -- a freshly-repaired retry node
    # has no such episode (its progress/abandonment was just reset),
    # so the forced path must not depend on one.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "findme.py").write_text(
        "def target_function():\n    pass\n", encoding="utf-8"
    )
    worker = WorkerCard(
        id="worker-src",
        territory_id="src",
        name="src",
        root="src",
        searchable_terms=["target_function"],
        files=["src/findme.py"],
    )

    class _NeverAssignsAnythingReasoner(_PassthroughLookupsReasoner):
        def observe(self, *, question, worker_id, territory_id, evidence):
            return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

        def check_need_resolution(self, *, need, new_evidence, question):
            return NeedResolution(status="resolved" if new_evidence else "unresolved")

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
            observed_needs,
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            worker_relevance_rank=None,
        ):
            return RoundPlan()

    state = LocalCoordinator(
        tmp_path, [worker], reasoner=_NeverAssignsAnythingReasoner()
    ).ask(
        "Where is target_function defined?",
        max_rounds=1,
        forced_first_round_global_search_ids={"root"},
    )

    assert state.rounds[0].node_executions
    execution = state.rounds[0].node_executions[0]
    assert execution.special_tactic == "global_fallback"
    assert execution.resolution == "resolved"
    assert state.final_need_graph["root"].resolution == "resolved"


def test_ask_does_not_double_execute_global_fallback_when_orchestrator_also_picks_it(
    tmp_path: Path,
) -> None:
    # Regression test: forced_first_round_global_search_ids's forced
    # execution and the Orchestrator's own independently-chosen
    # plan.special_tactics used to both run global_fallback for the same
    # need_id in the same round -- confirmed live on real qibo/seaborn
    # traces (one need_id's round-0 node_executions showed global_fallback
    # twice). This needs a real stuck episode to exist (unlike the
    # no-episode-needed test above) so the Orchestrator's own
    # special_tactics loop is actually capable of executing at all --
    # otherwise _episode_for_need's own None-guard would prevent the
    # second run for an unrelated reason, and this test would pass
    # without the fix doing anything.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "findme.py").write_text(
        "def target_function():\n    pass\n", encoding="utf-8"
    )
    worker = WorkerCard(
        id="worker-src",
        territory_id="src",
        name="src",
        root="src",
        searchable_terms=["target_function"],
        files=["src/findme.py"],
    )

    class _AlsoPicksGlobalFallbackReasoner(_PassthroughLookupsReasoner):
        def observe(self, *, question, worker_id, territory_id, evidence):
            return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

        def check_need_resolution(self, *, need, new_evidence, question):
            return NeedResolution(status="resolved" if new_evidence else "unresolved")

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
            observed_needs,
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            worker_relevance_rank=None,
        ):
            return RoundPlan(special_tactics={"root": "global_fallback"})

    recovery = RecoveryState(
        stuck_episodes={"root": StuckEpisode(episode_id="root", members={"root"})},
        episode_by_need_id={"root": "root"},
    )

    state = LocalCoordinator(
        tmp_path, [worker], reasoner=_AlsoPicksGlobalFallbackReasoner()
    ).ask(
        "Where is target_function defined?",
        max_rounds=1,
        initial_recovery=recovery,
        forced_first_round_global_search_ids={"root"},
    )

    root_executions = [ne for ne in state.rounds[0].node_executions if ne.need_id == "root"]
    assert len(root_executions) == 1
    assert root_executions[0].special_tactic == "global_fallback"


class _StubbornlyReassignsTriedWorkerReasoner(_PassthroughLookupsReasoner):
    """Never escalates a stuck need on its own -- keeps proposing a plain
    reassignment of the same single known worker for ready AND
    stuck-subgraph-member need_ids alike, ignoring the stuck_tried_workers
    hint plan_round receives entirely. Exists to prove
    LocalCoordinator.ask() enforces routing self-correction mechanically
    (see _enforce_no_repeat_stuck_assignment) rather than only hoping a
    stochastic planner notices the hint and diversifies on its own."""

    def observe(self, *, question, worker_id, territory_id, evidence):
        return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

    def check_need_resolution(self, *, need, new_evidence, question):
        return NeedResolution(status="unresolved")

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
        observed_needs,
        incomplete_parents,
        cross_repo_experience,
        validation_feedback="",
        repair_guidance="",
        stuck_tried_workers=None,
        worker_relevance_rank=None,
    ):
        stuck_members = {need_id for group in frontier.stuck_subgraphs for need_id in group}
        targets = set(frontier.ready) | stuck_members
        return RoundPlan(assignments={need_id: [workers[0].id] for need_id in targets})


def test_ask_overrides_a_stuck_reassignment_of_only_already_tried_workers(
    tmp_path: Path,
) -> None:
    # Routing self-correction: once a need is stuck (>= _STUCK_THRESHOLD
    # rounds without progress), an assignment made up entirely of workers
    # already recorded as tried-with-no-progress on it must not execute as
    # a plain repeat -- the coordinator overrides it with a forced
    # global_fallback rather than trusting the planner to diversify on its
    # own, since this fixture's reasoner deliberately never does.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def a():\n    pass\n", encoding="utf-8")
    worker = WorkerCard(
        id="worker-a", territory_id="a", name="a", root="src", files=["src/a.py"]
    )

    state = LocalCoordinator(
        tmp_path, [worker], reasoner=_StubbornlyReassignsTriedWorkerReasoner()
    ).ask("question", max_rounds=4)

    # Rounds 0-2: not yet stuck (progress flips to "stuck" only after
    # post_frontier for the *next* round is already computed -- see
    # _STUCK_THRESHOLD's own bookkeeping -- so the override's earliest
    # possible round is one later than rounds_without_progress alone would
    # suggest). By round 3 the reasoner is still proposing the same
    # already-tried worker_a for the now-stuck root -- confirmed overridden.
    assert len(state.rounds) == 4
    for round_index in range(3):
        assert state.rounds[round_index].node_executions[0].special_tactic == ""
    round3_executions = state.rounds[3].node_executions
    assert round3_executions, "expected round 3 to still execute something for the stuck need"
    assert round3_executions[0].need_id == "root"
    assert round3_executions[0].special_tactic == "global_fallback"
    assert round3_executions[0].worker_ids == []


class _AlwaysUnresolvedSingleWorkerReasoner(_PassthroughLookupsReasoner):
    """Produces a real, reproducible abandonment: assigns the only worker
    it knows about to whatever's ready/stuck every round, but never
    accepts any evidence as resolving anything -- same shape as
    _AlwaysStuckAndAlwaysProposesBridgeReasoner but without proposing
    special tactics, so root is abandoned purely by incomplete-parent-style
    non-progress (simpler prior trajectory to retry from)."""

    def observe(self, *, question, worker_id, territory_id, evidence):
        return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

    def check_need_resolution(self, *, need, new_evidence, question):
        return NeedResolution(status="unresolved")

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
        observed_needs,
        incomplete_parents,
        cross_repo_experience,
        validation_feedback="",
        repair_guidance="",
        stuck_tried_workers=None,
        worker_relevance_rank=None,
    ):
        assignments = {need_id: [workers[0].id] for need_id in frontier.ready}
        special_tactics = {
            need_id: "temporary_bridge"
            for group in frontier.stuck_subgraphs
            for need_id in group
        }
        return RoundPlan(assignments=assignments, special_tactics=special_tactics)


class _SuggestsReplacementWorkerReasoner:
    """Stub FastEvolutionReasoner: always proposes replacing whichever
    node is stuck with worker-fixed, regardless of package content -- this
    test only needs to prove the plan reaches and is applied by
    retry_from_trajectory, not exercise real repair judgment (that's
    OpenAIProvider.propose_repair's job)."""

    def propose_repair(self, *, package):
        stuck_id = package.stuck_nodes[0].need_id if package.stuck_nodes else "root"
        return RepairPlan(
            actions=[
                RepairAction(
                    kind="replace_assignment",
                    need_id=stuck_id,
                    worker_ids=["worker-fixed"],
                    rationale="tried workers failed; try a different one",
                )
            ]
        )


class _AlwaysPrefersBrokenWorkerReasoner(_PassthroughLookupsReasoner):
    """The retry's own WorkerReasoner: accepts real evidence as resolving a
    need (unlike the original attempt's always-unresolved reasoner), but
    -- deliberately, unconditionally -- keeps preferring worker-broken
    (the exact worker that already failed), ignoring repair_guidance
    entirely. Proves retry_from_trajectory's replace_assignment repair is
    FORCED to execute once at round 0 regardless of what the Orchestrator
    itself would have chosen, not merely a suggestion it might or might
    not follow -- if root still resolves here, it's only because the
    forced assignment overrode this reasoner's own (bad) round-0 choice."""

    def observe(self, *, question, worker_id, territory_id, evidence):
        return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

    def check_need_resolution(self, *, need, new_evidence, question):
        return NeedResolution(status="resolved" if new_evidence else "unresolved")

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
        observed_needs,
        incomplete_parents,
        cross_repo_experience,
        validation_feedback="",
        repair_guidance="",
        stuck_tried_workers=None,
        worker_relevance_rank=None,
    ):
        return RoundPlan(assignments={need_id: ["worker-broken"] for need_id in frontier.ready})


def test_retry_from_trajectory_resolves_a_need_the_original_attempt_abandoned(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "broken.py").write_text("def unrelated():\n    pass\n", encoding="utf-8")
    (tmp_path / "src" / "fixed.py").write_text(
        "def target_function():\n    pass\n", encoding="utf-8"
    )
    worker_broken = WorkerCard(
        id="worker-broken", territory_id="broken", name="broken", root="src",
        files=["src/broken.py"],
    )
    worker_fixed = WorkerCard(
        id="worker-fixed",
        territory_id="fixed",
        name="fixed",
        root="src",
        searchable_terms=["target_function"],
        files=["src/fixed.py"],
    )
    question = "Where is target_function defined?"

    original = LocalCoordinator(
        tmp_path, [worker_broken], reasoner=_AlwaysUnresolvedSingleWorkerReasoner()
    ).ask(question, max_rounds=10)

    assert "root" in original.final_recovery_state.abandoned_node_ids, (
        "test setup assumption: the original attempt must actually abandon root"
    )

    retried = LocalCoordinator(
        tmp_path,
        [worker_broken, worker_fixed],
        reasoner=_AlwaysPrefersBrokenWorkerReasoner(),
    ).retry_from_trajectory(
        original, fast_reasoner=_SuggestsReplacementWorkerReasoner(), max_rounds=3
    )

    assert retried.final_need_graph["root"].resolution == "resolved"
    assert "root" not in retried.final_recovery_state.abandoned_node_ids
    # The repair plan's replace_assignment must have actually run at round
    # 0 -- worker-fixed, not worker-broken (what this retry's own
    # Orchestrator reasoner always prefers) -- proving the fast-repair
    # action was forced, not merely offered as text.
    assert retried.rounds[0].node_executions[0].worker_ids == ["worker-fixed"]
