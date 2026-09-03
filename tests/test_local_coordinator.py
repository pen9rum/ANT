from pathlib import Path

import pytest

from ant.coordinator import LocalCoordinator
from ant.coordinator.local import (
    _ESCALATED_NEED_CANDIDATE_LIMIT,
    _FRESH_NEED_CANDIDATE_LIMIT,
    ProposalCluster,
    RecoveryState,
    StuckEpisode,
    _apply_consolidation_decisions,
    _build_temporary_bridge,
    _candidate_hints_for_proposals,
    _close_resolved_needs,
    _cluster_pending_proposals,
    _collect_proposals,
    _dedupe_evidence,
    _expand_cluster_decisions,
    _matches_term,
    _merge_needs,
    _plan_round_with_cycle_validation,
    _prune_dangling_edges,
    _rank_global_evidence,
    _reopen_referenced_evidence,
    _resolution_check_need,
    _select_evidence,
)
from ant.coordinator.worker_retrieval import build_worker_index
from ant.domain import (
    AnswerObligation,
    CodeSymbol,
    Evidence,
    EvidenceState,
    EvidenceUpgradeVerdict,
    FrontierResult,
    GraphConsolidationDecision,
    GraphConsolidationPlan,
    GroundedUpdate,
    NeedAlignmentPlan,
    NeedAlignmentVerdict,
    NeedGraph,
    NeedNode,
    NeedResolution,
    ObligationCoverage,
    ProposedNode,
    RecoverySnapshot,
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
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            candidate_probes=None,
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


def test_ask_threads_probe_results_into_plan_round(tmp_path: Path) -> None:
    # ant.coordinator.local.LocalCoordinator._probe_need_candidates gives
    # each of a ready need's narrowed candidates one cheap search()/
    # dense_search() look into its own territory before the Orchestrator
    # commits -- confirm ask() actually computes and threads this through
    # to plan_round() every round: the worker whose file actually contains
    # the query's own rare term ("FALQON") gets a real probe anchor, an
    # unrelated sibling's probe comes back empty (found nothing, not
    # simply absent).
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
    received: list[dict[str, dict[str, list[Evidence]]] | None] = []

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
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            candidate_probes=None,
        ):
            received.append(candidate_probes)
            return RoundPlan()

    LocalCoordinator(tmp_path, [target, sibling], reasoner=_RecordingReasoner()).ask(
        "What class is FALQON?", max_rounds=1
    )

    # worker-other genuinely has no retrieval signal for this query (its
    # corpus shares no term with "FALQON"), so it never became a
    # candidate in the first place -- nothing to probe, no entry at all
    # (same "no signal, not a bad one" distinction rank_workers itself
    # makes -- see _candidate_workers_for_round).
    assert received
    assert received[0]
    root_probes = received[0]["root"]
    assert root_probes["worker-models"]
    assert root_probes["worker-models"][0].path == "src/circuit.py"
    assert "worker-other" not in root_probes


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
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            candidate_probes=None,
        ):
            captured_worker_counts.append(len(workers))
            return RoundPlan(assignments={need_id: [workers[0].id] for need_id in frontier.ready})

    state = LocalCoordinator(tmp_path, workers, reasoner=_Reasoner()).ask(question, max_rounds=1)

    assert captured_worker_counts == [_FRESH_NEED_CANDIDATE_LIMIT]
    execution = state.rounds[0].node_executions[0]
    assert execution.need_id == "root"
    assert len(execution.candidate_worker_ids) == _FRESH_NEED_CANDIDATE_LIMIT
    assert execution.candidate_worker_ranks
    assert len(execution.candidate_probe_anchor_counts) == _FRESH_NEED_CANDIDATE_LIMIT


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

    def consolidate_graph(
        self, *, question, active_nodes, proposals, candidate_hints, enforce_alignment=False
    ):
        # Create-everything passthrough, matching MockLLMProvider's own
        # default -- preserves every existing scenario's graph-shape
        # assertions from before Need Graph Consolidation existed. Only a
        # test specifically about merge/subsume/attach/relate/drop
        # overrides this.
        return GraphConsolidationPlan(
            decisions=[
                GraphConsolidationDecision(proposal_id=proposal.proposal_id, action="create")
                for proposal in proposals
            ]
        )

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

    def verify_evidence_upgrade(self, *, need, epistemic_state, new_evidence, question):
        # Deliberately unapproved by default, same rationale as
        # check_need_resolution always returning "unresolved" above --
        # never called anyway unless a subclass overrides
        # check_need_resolution to actually return resolved/partial (see
        # _AlwaysPrefersBrokenWorkerReasoner for the one that does).
        return EvidenceUpgradeVerdict(approved=False)

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
        incomplete_parents,
        cross_repo_experience,
        validation_feedback="",
        repair_guidance="",
        stuck_tried_workers=None,
        candidate_probes=None,
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
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            candidate_probes=None,
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
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            candidate_probes=None,
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
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            candidate_probes=None,
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
        incomplete_parents,
        cross_repo_experience,
        validation_feedback="",
        repair_guidance="",
        stuck_tried_workers=None,
        candidate_probes=None,
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
        incomplete_parents,
        cross_repo_experience,
        validation_feedback="",
        repair_guidance="",
        stuck_tried_workers=None,
        candidate_probes=None,
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
        incomplete_parents,
        cross_repo_experience,
        validation_feedback="",
        repair_guidance="",
        stuck_tried_workers=None,
        candidate_probes=None,
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


def test_dedupe_evidence_unions_need_ids_instead_of_keeping_only_the_first_seen() -> None:
    # Regression test for per-claim evidence retention (Grounded Fast
    # Repair): the same (path, lines, quote) chunk can legitimately be
    # gathered once per need it answers -- keeping only the first
    # duplicate's need_ids would let a later need's real association with
    # this exact chunk silently disappear, breaking the "an untouched
    # need's evidence must survive" invariant for any need whose only
    # supporting occurrence happened to arrive second.
    first = Evidence(
        path="src/mod.py", line_start=1, line_end=2, quote="def f():", reason="r",
        need_ids=["need-a"],
    )
    second = Evidence(
        path="src/mod.py", line_start=1, line_end=2, quote="def f():", reason="r",
        need_ids=["need-b"],
    )

    deduped = _dedupe_evidence([first, second])

    assert len(deduped) == 1
    assert set(deduped[0].need_ids) == {"need-a", "need-b"}


class _GroundsOnlyOneNamedNeedReasoner(_PassthroughLookupsReasoner):
    """Assigns the sole worker to every ready need each round, but only
    ever grounds the ONE need whose description is `grounded_description`
    via verify_evidence_upgrade -- every other assigned need's evidence
    upgrade is rejected (the round loop still records it as executed, so
    it still counts as "reopened" for _reopened_need_ids, just never
    grounded). check_need_resolution stays "unresolved" for everything
    (deliberately -- these tests exercise the decoupled gate: a need can
    ground a specific claim while investigation itself never closes, see
    _apply_evidence_upgrade_gate's own docstring), which the gate no
    longer treats as a reason to skip verification. Lets a test put a
    real GroundedUpdate into ask()'s grounded_updates (required to even
    reach the per-claim partition branch -- see the monotonic gate this
    sits behind) while keeping a SEPARATE need under test un-grounded.
    Records every select_evidence call's own input pool so a test can
    assert exactly what the final display-budget cut was, and was not,
    shown."""

    def __init__(self, grounded_description: str) -> None:
        self.grounded_description = grounded_description
        self.select_evidence_calls: list[list[Evidence]] = []

    def observe(self, *, question, worker_id, territory_id, evidence):
        return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

    def check_need_resolution(self, *, need, new_evidence, question):
        return NeedResolution(status="unresolved")

    def verify_evidence_upgrade(self, *, need, epistemic_state, new_evidence, question):
        return EvidenceUpgradeVerdict(
            approved=need.description == self.grounded_description,
            supported_claim="grounded this retry",
            evidence_ids=["0"],
        )

    def select_evidence(self, *, question, evidence, limit):
        self.select_evidence_calls.append(list(evidence))
        return [str(index) for index in range(len(evidence))][:limit], []

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
        return RoundPlan(assignments={need_id: [workers[0].id] for need_id in frontier.ready})


class _RecordsFinalEvidenceSynthesizer:
    """Stub AnswerSynthesizer that records the exact evidence list it was
    handed for final synthesis, so a test can assert what per-claim
    retention did and did not let through -- not just what ask() itself
    returns (which is the same list, but recording the call directly
    proves synthesis genuinely saw it, not merely that it survived some
    unrelated post-processing)."""

    def __init__(self) -> None:
        self.synthesize_evidence: list[Evidence] | None = None

    def synthesize(self, **kwargs):
        self.synthesize_evidence = kwargs["evidence"]
        return "a freshly synthesized answer"

    def synthesize_coalition(self, **kwargs):
        self.synthesize_evidence = kwargs["evidence"]
        return "a freshly synthesized coalition answer"


class _ReturnsBlankTextSynthesizer:
    """Stub AnswerSynthesizer that always returns an empty string -- models
    a real gen0 trace where evidence was gathered but none of it was
    actually relevant (routing landed entirely on doc/example files for a
    question about actual source code) and synthesize() produced "" rather
    than an honest hedge."""

    def synthesize(self, **kwargs):
        return ""

    def synthesize_coalition(self, **kwargs):
        return ""


class _NeverAssignsAnyWorkerReasoner(_PassthroughLookupsReasoner):
    """Every round's plan_round assigns nothing -- evidence stays
    completely empty for the whole task, exercising the "nothing to
    synthesize from at all" abstention path (as opposed to "synthesized,
    but got blank text back")."""

    def observe(self, *, question, worker_id, territory_id, evidence):
        return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

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
        return RoundPlan(assignments={})


class _AssignsWhateverIsReadyReasoner(_PassthroughLookupsReasoner):
    """Plain, non-fast-repair reasoner: assigns the sole worker to every
    ready need each round, keeps everything select_evidence sees. Used to
    get real evidence gathered and through to synthesis without any of
    the enforce_alignment-specific machinery."""

    def observe(self, *, question, worker_id, territory_id, evidence):
        return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

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
        return RoundPlan(assignments={need_id: [workers[0].id] for need_id in frontier.ready})


def test_ask_abstains_instead_of_returning_a_blank_answer_when_no_evidence_survives(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("def unrelated():\n    pass\n", encoding="utf-8")
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )

    result = LocalCoordinator(
        tmp_path,
        [worker],
        reasoner=_NeverAssignsAnyWorkerReasoner(),
        synthesizer=_RecordsFinalEvidenceSynthesizer(),
    ).ask("some question nothing ever gets assigned to", max_rounds=1)

    assert result.answer != ""
    assert "does not directly and reliably answer" in result.answer


def test_ask_leaves_answer_blank_when_no_synthesizer_is_configured(tmp_path: Path) -> None:
    # Regression test: run_batch's own heuristic-only mode (no OpenAIProvider
    # passed) deliberately never attempts synthesis, and its own
    # _fallback_prediction(state.evidence) relies on `state.answer` staying
    # "" (falsy) to know to substitute the raw evidence text for scoring --
    # the abstention fallback must not fire when there was never a real
    # synthesizer to abstain FROM in the first place.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("def unrelated():\n    pass\n", encoding="utf-8")
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )

    result = LocalCoordinator(
        tmp_path, [worker], reasoner=_NeverAssignsAnyWorkerReasoner()
    ).ask("some question nothing ever gets assigned to", max_rounds=1)

    assert result.answer == ""


def test_ask_abstains_instead_of_returning_a_blank_answer_when_synthesis_itself_returns_blank(
    tmp_path: Path,
) -> None:
    # Regression test for a real gen0 trace: 16 evidence items gathered
    # (from doc/sphinxext and examples/, not the actual source module the
    # question asked about), and synthesize() returned "" rather than an
    # honest "insufficient evidence" hedge.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "def some_symbol():\n    pass\n", encoding="utf-8"
    )
    worker = WorkerCard(
        id="worker-src",
        territory_id="src",
        name="src",
        root="src",
        files=["src/mod.py"],
        searchable_terms=["some_symbol"],
    )

    result = LocalCoordinator(
        tmp_path,
        [worker],
        reasoner=_AssignsWhateverIsReadyReasoner(),
        synthesizer=_ReturnsBlankTextSynthesizer(),
    ).ask("find some_symbol", max_rounds=1)

    assert result.answer != ""
    assert "does not directly and reliably answer" in result.answer


def test_ask_preserves_untouched_evidence_when_a_sibling_need_is_grounded_this_retry(
    tmp_path: Path,
) -> None:
    # Regression test for the seaborn `regression.py` loss (-14 points):
    # a need nothing this retry ever reopened must have its evidence
    # survive completely untouched, even though a SIBLING need earning a
    # real GroundedUpdate this same retry pushes ask() past the monotonic
    # gate and into the partition-then-select branch.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "def reopened_target():\n    pass\n", encoding="utf-8"
    )
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )
    graph = NeedGraph(
        nodes={
            "root": NeedNode(
                need_id="root",
                need="original question",
                resolution="resolved",
                children=["untouched-need", "reopened-need"],
                detail=UnresolvedNeed(description="original question"),
            ),
            "untouched-need": NeedNode(
                need_id="untouched-need",
                need="untouched need",
                resolution="resolved",
                detail=UnresolvedNeed(description="untouched need"),
            ),
            "reopened-need": NeedNode(
                need_id="reopened-need",
                need="find reopened_target",
                resolution="unresolved",
                detail=UnresolvedNeed(description="find reopened_target"),
            ),
        }
    )
    untouched_evidence = Evidence(
        path="src/untouched.py",
        line_start=1,
        line_end=1,
        quote="def untouched_target():",
        reason="gen0's own evidence for the untouched need",
        need_ids=["untouched-need"],
    )
    # The reasoner's own verify_evidence_upgrade approval is keyed off this
    # exact description -- and the worker must actually find real evidence
    # for `find reopened_target` (matching `reopened_target()` in mod.py
    # above) or _apply_evidence_upgrade_gate's `not new_evidence` guard
    # skips verification entirely, same as any need with zero evidence_gain.
    reasoner = _GroundsOnlyOneNamedNeedReasoner(grounded_description="find reopened_target")
    synthesizer = _RecordsFinalEvidenceSynthesizer()

    result = LocalCoordinator(
        tmp_path, [worker], reasoner=reasoner, synthesizer=synthesizer
    ).ask(
        "original question",
        max_rounds=1,
        initial_graph=graph,
        initial_evidence=[untouched_evidence],
        prior_answer="gen0's own verbatim answer",
        enforce_alignment=True,
    )

    assert any(
        item.path == "src/untouched.py" and item.need_ids == ["untouched-need"]
        for item in result.evidence
    )
    # The untouched association bypassed _select_evidence's judgment
    # entirely -- it never appears in any pool the reasoner was asked to
    # select from.
    for pool in reasoner.select_evidence_calls:
        assert all(item.path != "src/untouched.py" for item in pool)


def test_ask_keeps_a_reopened_associations_need_id_intact_when_it_is_grounded_this_retry(
    tmp_path: Path,
) -> None:
    # An item whose ONLY association is a reopened need that DID produce
    # a GroundedUpdate this retry must survive with that need_id intact --
    # proving the grounded-verifier signal, not _select_evidence's own
    # relevance judgment, is what re-validates a reopened association.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "def reopened_target():\n    pass\n", encoding="utf-8"
    )
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )
    graph = NeedGraph(
        nodes={
            "root": NeedNode(
                need_id="root",
                need="original question",
                resolution="resolved",
                children=["reopened-need"],
                detail=UnresolvedNeed(description="original question"),
            ),
            "reopened-need": NeedNode(
                need_id="reopened-need",
                need="find reopened_target",
                resolution="unresolved",
                detail=UnresolvedNeed(description="find reopened_target"),
            ),
        }
    )
    reopened_evidence = Evidence(
        path="src/reopened.py",
        line_start=1,
        line_end=1,
        quote="def reopened_target():",
        reason="prior evidence for the reopened need, up for re-verification",
        need_ids=["reopened-need"],
    )
    # Must actually find real evidence for `find reopened_target` this
    # retry (matching mod.py above), or the gate's `not new_evidence`
    # guard skips verification and grounded_updates stays empty --
    # falling through the monotonic gate instead of the partition branch
    # this test exists to exercise.
    reasoner = _GroundsOnlyOneNamedNeedReasoner(grounded_description="find reopened_target")

    result = LocalCoordinator(tmp_path, [worker], reasoner=reasoner).ask(
        "original question",
        max_rounds=1,
        initial_graph=graph,
        initial_evidence=[reopened_evidence],
        prior_answer="gen0's own verbatim answer",
        enforce_alignment=True,
    )

    assert any(
        item.path == "src/reopened.py" and item.need_ids == ["reopened-need"]
        for item in result.evidence
    )


def test_ask_excludes_a_reopened_association_never_grounded_before_select_evidence_runs(
    tmp_path: Path,
) -> None:
    # An item whose ONLY association is a reopened need that produced NO
    # GroundedUpdate this retry must be excluded from the final evidence,
    # and must never even reach _select_evidence -- a coarse relevance
    # judgment cannot substitute for real grounding (the qibo -24 point
    # regression: the final _select_evidence call, judging the whole
    # mixed pool fresh, picked the wrong evidence over what was correctly
    # routed-to but never itself verified).
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "def grounds_target():\n    pass\n\n\ndef reopened_target():\n    pass\n",
        encoding="utf-8",
    )
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )
    graph = NeedGraph(
        nodes={
            "root": NeedNode(
                need_id="root",
                need="original question",
                resolution="resolved",
                children=["grounds-need", "reopened-need"],
                detail=UnresolvedNeed(description="original question"),
            ),
            "grounds-need": NeedNode(
                need_id="grounds-need",
                need="find grounds_target",
                resolution="unresolved",
                detail=UnresolvedNeed(description="find grounds_target"),
            ),
            "reopened-need": NeedNode(
                need_id="reopened-need",
                need="find reopened_target",
                resolution="unresolved",
                detail=UnresolvedNeed(description="find reopened_target"),
            ),
        }
    )
    ungrounded_evidence = Evidence(
        path="src/reopened_only.py",
        line_start=1,
        line_end=1,
        quote="def reopened_target():",
        reason="old evidence for the reopened need, never reverified",
        need_ids=["reopened-need"],
    )
    # `grounds-need` must actually find real evidence (matching
    # `grounds_target()` in mod.py above) so grounded_updates is non-empty
    # and the trace reaches the partition branch this test targets --
    # `reopened-need` deliberately investigates real evidence too (its own
    # matching symbol above) but is never approved, exercising the exact
    # "found evidence, need's association still excluded" pattern.
    reasoner = _GroundsOnlyOneNamedNeedReasoner(grounded_description="find grounds_target")

    result = LocalCoordinator(tmp_path, [worker], reasoner=reasoner).ask(
        "original question",
        max_rounds=1,
        initial_graph=graph,
        initial_evidence=[ungrounded_evidence],
        prior_answer="gen0's own verbatim answer",
        enforce_alignment=True,
    )

    assert all(item.path != "src/reopened_only.py" for item in result.evidence)
    for pool in reasoner.select_evidence_calls:
        assert all(item.path != "src/reopened_only.py" for item in pool)


def test_ask_reduces_shared_evidence_to_its_untouched_need_id_when_its_sibling_is_ungrounded(
    tmp_path: Path,
) -> None:
    # The shared-evidence rescue: an item supporting both a reopened need
    # with NO GroundedUpdate and an untouched need must still survive in
    # the final evidence pool, with need_ids reduced to just the
    # untouched need -- proving the untouched claim's support cannot be
    # deleted by its reopened sibling's own fate.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "def grounds_target():\n    pass\n\n\ndef reopened_target():\n    pass\n",
        encoding="utf-8",
    )
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )
    graph = NeedGraph(
        nodes={
            "root": NeedNode(
                need_id="root",
                need="original question",
                resolution="resolved",
                children=["grounds-need", "reopened-need", "untouched-need"],
                detail=UnresolvedNeed(description="original question"),
            ),
            "grounds-need": NeedNode(
                need_id="grounds-need",
                need="find grounds_target",
                resolution="unresolved",
                detail=UnresolvedNeed(description="find grounds_target"),
            ),
            "reopened-need": NeedNode(
                need_id="reopened-need",
                need="find reopened_target",
                resolution="unresolved",
                detail=UnresolvedNeed(description="find reopened_target"),
            ),
            "untouched-need": NeedNode(
                need_id="untouched-need",
                need="untouched need",
                resolution="resolved",
                detail=UnresolvedNeed(description="untouched need"),
            ),
        }
    )
    shared_evidence = Evidence(
        path="src/shared.py",
        line_start=1,
        line_end=1,
        quote="def shared_target():",
        reason="supports both the reopened and the untouched need",
        need_ids=["reopened-need", "untouched-need"],
    )
    # `grounds-need` must actually find real evidence (matching
    # `grounds_target()` in mod.py above) so grounded_updates is non-empty
    # and the trace reaches the partition branch this test targets.
    reasoner = _GroundsOnlyOneNamedNeedReasoner(grounded_description="find grounds_target")

    result = LocalCoordinator(tmp_path, [worker], reasoner=reasoner).ask(
        "original question",
        max_rounds=1,
        initial_graph=graph,
        initial_evidence=[shared_evidence],
        prior_answer="gen0's own verbatim answer",
        enforce_alignment=True,
    )

    assert any(
        item.path == "src/shared.py" and item.need_ids == ["untouched-need"]
        for item in result.evidence
    )


def test_ask_warns_on_untagged_evidence_from_an_unaccounted_for_source(tmp_path: Path) -> None:
    # Provenance telemetry: an audit of every evidence entry point found
    # exactly one accepted untagged source (_verify_inheritance_completeness
    # -- deliberately untagged, a repository-wide fact, not any one need's
    # claim). Anything else reaching final synthesis with need_ids=[] is an
    # unaudited gap -- must warn (never crash) rather than silently and
    # permanently lose that evidence the way the global_fallback bug did.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "def grounds_target():\n    pass\n", encoding="utf-8"
    )
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )
    graph = NeedGraph(
        nodes={
            "root": NeedNode(
                need_id="root",
                need="original question",
                resolution="resolved",
                children=["grounds-need"],
                detail=UnresolvedNeed(description="original question"),
            ),
            "grounds-need": NeedNode(
                need_id="grounds-need",
                need="find grounds_target",
                resolution="unresolved",
                detail=UnresolvedNeed(description="find grounds_target"),
            ),
        }
    )
    # No inheritance-asking language in the question, so
    # _verify_inheritance_completeness contributes nothing this call --
    # this item's empty need_ids has no accepted explanation.
    mystery_evidence = Evidence(
        path="src/mystery.py",
        line_start=1,
        line_end=1,
        quote="def mystery_target():",
        reason="from some untagged source",
    )
    reasoner = _GroundsOnlyOneNamedNeedReasoner(grounded_description="find grounds_target")

    with pytest.warns(UserWarning, match="untagged evidence"):
        LocalCoordinator(tmp_path, [worker], reasoner=reasoner).ask(
            "original question",
            max_rounds=1,
            initial_graph=graph,
            initial_evidence=[mystery_evidence],
            prior_answer="gen0's own verbatim answer",
            enforce_alignment=True,
        )


def test_ask_does_not_warn_on_inheritance_completeness_evidence(
    tmp_path: Path, recwarn: pytest.WarningsRecorder
) -> None:
    # The accepted exception itself must NOT trip the telemetry warning --
    # _verify_inheritance_completeness's own untagged evidence is deliberate
    # (a repository-wide structural fact, not any one need's claim), not a
    # gap the audit missed.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "class BaseThing:\n    pass\n\n\nclass SubThing(BaseThing):\n    pass\n\n\n"
        "def grounds_target():\n    pass\n",
        encoding="utf-8",
    )
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )
    graph = NeedGraph(
        nodes={
            "root": NeedNode(
                need_id="root",
                need="original question",
                resolution="resolved",
                children=["grounds-need"],
                detail=UnresolvedNeed(description="original question"),
            ),
            "grounds-need": NeedNode(
                need_id="grounds-need",
                need="find grounds_target",
                resolution="unresolved",
                detail=UnresolvedNeed(description="find grounds_target"),
            ),
        }
    )
    reasoner = _GroundsOnlyOneNamedNeedReasoner(grounded_description="find grounds_target")

    LocalCoordinator(tmp_path, [worker], reasoner=reasoner).ask(
        "What are the subclasses of BaseThing? find grounds_target",
        max_rounds=1,
        initial_graph=graph,
        prior_answer="gen0's own verbatim answer",
        enforce_alignment=True,
    )

    assert not any(
        "untagged evidence" in str(warning.message) for warning in recwarn.list
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
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            candidate_probes=None,
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
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            candidate_probes=None,
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
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            candidate_probes=None,
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
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            candidate_probes=None,
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
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            candidate_probes=None,
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


def test_ask_tags_the_global_fallback_special_tactics_evidence_with_need_ids(
    tmp_path: Path,
) -> None:
    # Regression test: the Orchestrator's own special_tactics-chosen
    # "global_fallback" (distinct from forced_first_round_global_search_ids,
    # which already tagged its own hits) built its Evidence items straight
    # from search.search()'s raw output and extended `evidence` with them
    # untagged -- need_ids stayed [] forever, so per-claim evidence
    # retention could never preserve this need's association with this
    # evidence even if the need was later grounded (an untagged item can
    # neither be "preserved" -- requires non-empty need_ids -- nor survive
    # the reopened-but-grounded check, whose kept_ids comprehension has
    # nothing to iterate over an empty list).
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

    class _PicksGlobalFallbackReasoner(_PassthroughLookupsReasoner):
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
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            candidate_probes=None,
        ):
            return RoundPlan(special_tactics={"root": "global_fallback"})

    recovery = RecoveryState(
        stuck_episodes={"root": StuckEpisode(episode_id="root", members={"root"})},
        episode_by_need_id={"root": "root"},
    )

    state = LocalCoordinator(
        tmp_path, [worker], reasoner=_PicksGlobalFallbackReasoner()
    ).ask(
        "Where is target_function defined?",
        max_rounds=1,
        initial_recovery=recovery,
    )

    assert state.evidence
    assert all(item.need_ids == ["root"] for item in state.evidence)


def test_ask_does_not_crash_when_a_different_episode_member_hits_an_already_tried_tactic(
    tmp_path: Path,
) -> None:
    # Regression test for a real crash on a fresh yt-dlp gen0/fast-gen1
    # run (KeyError: 'need2'): `episode.used_special_tactics` is tracked
    # per EPISODE, not per need_id, and a stuck episode can have several
    # member need_ids. Round 0: need_a (a real member) executes
    # global_fallback for real -- this also sets resolution_results
    # ["need_a"] via the normal per-need epilogue, and marks
    # "global_fallback" used for the WHOLE episode. Round 1: need_b (a
    # DIFFERENT member of the SAME episode, touched for the very first
    # time) gets proposed the same already-used tactic -- hits the
    # "already tried, skip" branch (`if tactic in
    # episode.used_special_tactics`), which appends a NodeExecutionTrace
    # but, unlike every other node_executions.append() site, never sets
    # need_b's own entry in resolution_results. touched_this_round is
    # built from node_executions alone, and _resolution_advanced
    # unconditionally indexes resolution_results for every touched
    # need_id -- need_b has no entry at all (its only prior state was
    # never set), so this raised KeyError and crashed the whole run.
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
    graph = NeedGraph(
        nodes={
            "need_a": NeedNode(
                need_id="need_a",
                need="Where is target_function defined? (a)",
                detail=UnresolvedNeed(description="Where is target_function defined? (a)"),
            ),
            "need_b": NeedNode(
                need_id="need_b",
                need="Where is target_function defined? (b)",
                detail=UnresolvedNeed(description="Where is target_function defined? (b)"),
            ),
        }
    )

    class _RunsOneMemberThenTheOtherReasoner(_PassthroughLookupsReasoner):
        def __init__(self) -> None:
            self.round_index = 0

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
            incomplete_parents,
            cross_repo_experience,
            validation_feedback="",
            repair_guidance="",
            stuck_tried_workers=None,
            candidate_probes=None,
        ):
            # Round 0: need_a executes the tactic for real. Round 1:
            # need_b (never touched before) is proposed the SAME
            # already-used tactic for the first time -- by round index,
            # not frontier state (need_a is still "ready" in round 1 too,
            # one quiet round short of leaving the frontier).
            target = "need_a" if self.round_index == 0 else "need_b"
            self.round_index += 1
            return RoundPlan(special_tactics={target: "global_fallback"})

    recovery = RecoveryState(
        stuck_episodes={
            "ep1": StuckEpisode(episode_id="ep1", members={"need_a", "need_b"})
        },
        episode_by_need_id={"need_a": "ep1", "need_b": "ep1"},
    )

    state = LocalCoordinator(
        tmp_path, [worker], reasoner=_RunsOneMemberThenTheOtherReasoner()
    ).ask(
        "Where is target_function defined?",
        max_rounds=2,
        initial_graph=graph,
        initial_recovery=recovery,
    )

    need_b_executions = [
        ne for ne in state.rounds[1].node_executions if ne.need_id == "need_b"
    ]
    assert len(need_b_executions) == 1
    assert need_b_executions[0].evidence_gain == 0


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
        incomplete_parents,
        cross_repo_experience,
        validation_feedback="",
        repair_guidance="",
        stuck_tried_workers=None,
        candidate_probes=None,
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
        incomplete_parents,
        cross_repo_experience,
        validation_feedback="",
        repair_guidance="",
        stuck_tried_workers=None,
        candidate_probes=None,
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

    def assess_need_alignment(self, *, question, package):
        # Empty plan -- every stuck node defaults to "keep" (see
        # NeedAlignmentPlan's own "no verdict" default), matching this
        # test's real point: proving replace_assignment's forced
        # execution, not exercising alignment judgment.
        return NeedAlignmentPlan()

    def extract_answer_obligations(self, *, question):
        return []

    def check_obligation_coverage(self, *, question, obligations, evidence):
        return []


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

    def verify_evidence_upgrade(self, *, need, epistemic_state, new_evidence, question):
        # This test's own point is that root DOES resolve once the
        # forced replace_assignment runs (see class docstring) -- unlike
        # the base class's conservative default, this must approve so the
        # Evidence Upgrade Gate (live for every retry_from_trajectory
        # call now) doesn't itself block the very resolution this test
        # exists to prove.
        return EvidenceUpgradeVerdict(
            approved=True, supported_claim="worker-fixed found it", evidence_ids=["0"]
        )

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


class _RecordsEvidenceUpgradeCallsReasoner(_PassthroughLookupsReasoner):
    """Scripted per-need verify_evidence_upgrade verdicts (keyed by the
    need's own description, since the reasoner protocol does not pass
    need_id), recording each call's epistemic_state argument -- lets a
    single test assert both what context a call received and how the
    gate's outcome depends on it."""

    def __init__(self, verdicts: dict[str, EvidenceUpgradeVerdict]) -> None:
        self.verdicts = verdicts
        self.calls: list[tuple[str, str]] = []

    def observe(self, *, question, worker_id, territory_id, evidence):
        return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

    def verify_evidence_upgrade(self, *, need, epistemic_state, new_evidence, question):
        self.calls.append((need.description, epistemic_state))
        return self.verdicts[need.description]


def test_apply_evidence_upgrade_gate_covers_gen0_and_retry_born_nodes_and_is_inert_unenforced(
    tmp_path: Path,
) -> None:
    reasoner = _RecordsEvidenceUpgradeCallsReasoner(
        {
            "gen0 need": EvidenceUpgradeVerdict(
                approved=True, supported_claim="X directly implements Y", evidence_ids=["0"]
            ),
            "retry-born need": EvidenceUpgradeVerdict(approved=False),
        }
    )
    coordinator = LocalCoordinator(tmp_path, [], reasoner=reasoner)
    # Only the gen0-carried-over need has a seeded epistemic_state -- a
    # need born mid-retry (e.g. via decomposition) has no entry at all,
    # per correction 7's "no membership bypass".
    recovery = RecoveryState(epistemic_states={"gen0-need": "absence_supported"})
    evidence = [Evidence(path="a.py", line_start=1, line_end=2, quote="x", reason="r")]

    # A gen0-carried-over node: its real seeded epistemic_state reaches
    # the gate as context, and an approved verdict passes the resolution
    # through, building a GroundedUpdate directly from the verdict's own
    # fields (correction 8 -- no reconstruction, no second call).
    resolution, update = coordinator._apply_evidence_upgrade_gate(
        resolution=NeedResolution(status="resolved"),
        need=UnresolvedNeed(description="gen0 need"),
        need_id="gen0-need",
        new_evidence=evidence,
        recovery=recovery,
        question="q",
        enforce_alignment=True,
    )
    assert resolution.status == "resolved"
    assert update == GroundedUpdate(
        need_id="gen0-need",
        supported_claim="X directly implements Y",
        evidence_ids=["0"],
        prior_epistemic_state="absence_supported",
    )

    # A node with no entry in recovery.epistemic_states at all -- the gate
    # still fires (no membership check gets in its way), defaulting its
    # context to "open"; an unapproved verdict is a pure no-op -- it must
    # NOT revert the resolution check_need_resolution already produced
    # (Evidence admissibility != need completion: this gate only ever
    # decides whether a GroundedUpdate exists, never resolved/partial/
    # unresolved, which stays check_need_resolution's sole call).
    resolution, update = coordinator._apply_evidence_upgrade_gate(
        resolution=NeedResolution(status="partial"),
        need=UnresolvedNeed(description="retry-born need"),
        need_id="retry-born-need",
        new_evidence=evidence,
        recovery=recovery,
        question="q",
        enforce_alignment=True,
    )
    assert resolution.status == "partial"
    assert update is None
    assert reasoner.calls == [
        ("gen0 need", "absence_supported"),
        ("retry-born need", "open"),
    ]

    # Outside fast-repair mode, the gate is fully inert: resolution passes
    # through unchanged and verify_evidence_upgrade is never even called,
    # regardless of resolution.status.
    reasoner.calls.clear()
    resolution, update = coordinator._apply_evidence_upgrade_gate(
        resolution=NeedResolution(status="resolved"),
        need=UnresolvedNeed(description="gen0 need"),
        need_id="gen0-need",
        new_evidence=evidence,
        recovery=recovery,
        question="q",
        enforce_alignment=False,
    )
    assert resolution.status == "resolved"
    assert update is None
    assert reasoner.calls == []


def test_apply_evidence_upgrade_gate_can_ground_a_still_unresolved_need(tmp_path: Path) -> None:
    # The core new capability this decoupling exists for -- a direct
    # regression test for the live seaborn trace that motivated it: a
    # worker found fit_poly's own direct call into bootstrap(), a
    # specific, directly-supported claim -- but the broader `boot_method`
    # need itself never reached resolved/partial (still missing the full
    # picture), so check_need_resolution correctly keeps it unresolved
    # (investigation should continue). Under the OLD coupled gate, an
    # "unresolved" resolution.status short-circuited the gate entirely,
    # so this specific, correct claim could never produce a GroundedUpdate
    # and was silently unusable in the final answer. It must now.
    reasoner = _RecordsEvidenceUpgradeCallsReasoner(
        {"boot_method": EvidenceUpgradeVerdict(
            approved=True,
            supported_claim="fit_poly calls algo.bootstrap directly",
            evidence_ids=["0"],
        )}
    )
    coordinator = LocalCoordinator(tmp_path, [], reasoner=reasoner)
    evidence = [
        Evidence(path="seaborn/regression.py", line_start=1, line_end=2, quote="x", reason="r")
    ]

    resolution, update = coordinator._apply_evidence_upgrade_gate(
        resolution=NeedResolution(status="unresolved"),
        need=UnresolvedNeed(description="boot_method"),
        need_id="boot_method",
        new_evidence=evidence,
        recovery=RecoveryState(),
        question="q",
        enforce_alignment=True,
    )

    # Investigation state is untouched -- still unresolved, so the round
    # loop keeps searching -- while the claim itself is grounded.
    assert resolution.status == "unresolved"
    assert update == GroundedUpdate(
        need_id="boot_method",
        supported_claim="fit_poly calls algo.bootstrap directly",
        evidence_ids=["0"],
        prior_epistemic_state="open",
    )


def test_apply_evidence_upgrade_gate_skips_the_verifier_call_with_no_new_evidence(
    tmp_path: Path,
) -> None:
    # Nothing new to verify needs no verification call -- avoids a
    # pointless LLM call every round for a need with zero evidence_gain,
    # now that the gate is no longer implicitly guarded by
    # resolution.status ever reaching resolved/partial.
    reasoner = _RecordsEvidenceUpgradeCallsReasoner({})
    coordinator = LocalCoordinator(tmp_path, [], reasoner=reasoner)

    resolution, update = coordinator._apply_evidence_upgrade_gate(
        resolution=NeedResolution(status="unresolved"),
        need=UnresolvedNeed(description="some need"),
        need_id="n1",
        new_evidence=[],
        recovery=RecoveryState(),
        question="q",
        enforce_alignment=True,
    )

    assert resolution.status == "unresolved"
    assert update is None
    assert reasoner.calls == []


def test_resolution_check_need_is_a_passthrough_outside_fast_repair_mode() -> None:
    detail = UnresolvedNeed(description="narrowed live wording")
    recovery = RecoveryState(intent_anchors={"n1": "broad original wording"})

    result = _resolution_check_need(detail, "n1", recovery, enforce_alignment=False)

    assert result is detail


def test_resolution_check_need_is_a_passthrough_with_no_anchor_seeded() -> None:
    detail = UnresolvedNeed(description="narrowed live wording")
    recovery = RecoveryState()

    result = _resolution_check_need(detail, "n1", recovery, enforce_alignment=True)

    assert result is detail


def test_resolution_check_need_substitutes_the_frozen_anchor_for_the_description() -> None:
    detail = UnresolvedNeed(
        description="narrowed live wording", missing="a live detail that keeps updating"
    )
    recovery = RecoveryState(intent_anchors={"n1": "broad original wording"})

    result = _resolution_check_need(detail, "n1", recovery, enforce_alignment=True)

    assert result.description == "broad original wording"
    # Only description is swapped -- every other field (used for
    # search/routing, not for judging evidence) passes through unchanged.
    assert result.missing == "a live detail that keeps updating"


class _KeepsEverythingFastReasoner:
    def propose_repair(self, *, package):
        return RepairPlan(actions=[])

    def assess_need_alignment(self, *, question, package):
        return NeedAlignmentPlan()

    def extract_answer_obligations(self, *, question):
        return []

    def check_obligation_coverage(self, *, question, obligations, evidence):
        return []


class _NarrowsWordingThenChecksResolutionReasoner(_PassthroughLookupsReasoner):
    """Round 0: narrows the ready leaf's own wording via an ordinary
    existing-node graph_updates edit (exactly what plan_round does in
    practice, and what check_need_resolution's own "partial" refined_need
    rewrite also does) -- then executes and checks it the SAME round, so
    check_need_resolution sees whatever _resolution_check_need hands it
    at that point. Records every description it was actually asked to
    judge, so the test can assert it saw the frozen original wording, not
    the already-narrowed live one."""

    def __init__(self) -> None:
        self.seen_descriptions: list[str] = []

    def observe(self, *, question, worker_id, territory_id, evidence):
        return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

    def check_need_resolution(self, *, need, new_evidence, question):
        self.seen_descriptions.append(need.description)
        return NeedResolution(status="unresolved")

    def verify_evidence_upgrade(self, *, need, epistemic_state, new_evidence, question):
        return EvidenceUpgradeVerdict(approved=False)

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
        graph_updates = {}
        for need_id in frontier.ready:
            existing = graph.nodes[need_id]
            if existing.need == "broad original wording":
                graph_updates[need_id] = existing.model_copy(
                    update={
                        "need": "narrow specific wording",
                        "detail": existing.detail.model_copy(
                            update={"description": "narrow specific wording"}
                        ),
                    }
                )
        return RoundPlan(
            assignments={need_id: [workers[0].id] for need_id in frontier.ready},
            graph_updates=graph_updates,
        )


def test_retry_from_trajectory_judges_a_narrowed_leaf_against_its_frozen_intent_anchor(
    tmp_path: Path,
) -> None:
    # Regression test for a real yt-dlp trace: a kept leaf's own wording
    # narrowed over the retry's own rounds (e.g. "proxy configuration
    # validation" -> "the make_socks_proxy_opts/select_proxy helpers
    # specifically") until a later round's genuinely correct, directly-
    # responsive evidence got judged against the now-over-narrow wording
    # instead of what the node was originally about, and rejected.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("def unrelated():\n    pass\n", encoding="utf-8")
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )
    prior_state = EvidenceState(
        question="original question",
        final_need_graph={
            "root": NeedNode(
                need_id="root",
                need="original question",
                resolution="resolved",
                children=["target-need"],
                detail=UnresolvedNeed(description="original question"),
            ),
            "target-need": NeedNode(
                need_id="target-need",
                need="broad original wording",
                resolution="unresolved",
                detail=UnresolvedNeed(description="broad original wording"),
            ),
        },
    )
    reasoner = _NarrowsWordingThenChecksResolutionReasoner()

    LocalCoordinator(tmp_path, [worker], reasoner=reasoner).retry_from_trajectory(
        prior_state, fast_reasoner=_KeepsEverythingFastReasoner(), max_rounds=2
    )

    assert "broad original wording" in reasoner.seen_descriptions
    assert "narrow specific wording" not in reasoner.seen_descriptions


class _BirthsThenNarrowsANewNodeReasoner(_PassthroughLookupsReasoner):
    """Round 0: proposes a brand-new node (via graph_updates, source
    already committed through consolidation since this reasoner's
    consolidate_graph -- inherited from _PassthroughLookupsReasoner --
    creates everything). Round 1: narrows THAT SAME node's own wording,
    then checks its resolution -- proving a node born mid-retry gets its
    own intent_anchor seeded once, at birth, not just gen0-carried-over
    nodes."""

    def __init__(self) -> None:
        self.seen_descriptions: list[str] = []

    def observe(self, *, question, worker_id, territory_id, evidence):
        return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

    def check_need_resolution(self, *, need, new_evidence, question):
        self.seen_descriptions.append(need.description)
        return NeedResolution(status="unresolved")

    def verify_evidence_upgrade(self, *, need, epistemic_state, new_evidence, question):
        return EvidenceUpgradeVerdict(approved=False)

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
        assignments = {need_id: [workers[0].id] for need_id in frontier.ready}
        if "born-mid-retry" not in graph.nodes:
            return RoundPlan(
                assignments=assignments,
                graph_updates={
                    "born-mid-retry": NeedNode(
                        need_id="born-mid-retry",
                        need="birth wording",
                        detail=UnresolvedNeed(description="birth wording"),
                    )
                },
            )
        existing = graph.nodes["born-mid-retry"]
        if existing.need == "birth wording":
            assignments["born-mid-retry"] = [workers[0].id]
            return RoundPlan(
                assignments=assignments,
                graph_updates={
                    "born-mid-retry": existing.model_copy(
                        update={
                            "need": "narrowed after birth",
                            "detail": existing.detail.model_copy(
                                update={"description": "narrowed after birth"}
                            ),
                        }
                    )
                },
            )
        return RoundPlan(assignments=assignments)


def test_retry_from_trajectory_seeds_an_intent_anchor_for_a_node_born_mid_retry(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("def unrelated():\n    pass\n", encoding="utf-8")
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )
    prior_state = EvidenceState(
        question="original question",
        final_need_graph={
            "root": NeedNode(
                need_id="root",
                need="original question",
                resolution="unresolved",
                detail=UnresolvedNeed(description="original question"),
            ),
        },
    )
    reasoner = _BirthsThenNarrowsANewNodeReasoner()

    LocalCoordinator(tmp_path, [worker], reasoner=reasoner).retry_from_trajectory(
        prior_state, fast_reasoner=_KeepsEverythingFastReasoner(), max_rounds=3
    )

    assert "birth wording" in reasoner.seen_descriptions
    assert "narrowed after birth" not in reasoner.seen_descriptions


def _fast_repair_graph_fixture() -> dict[str, NeedNode]:
    return {
        "root": NeedNode(
            need_id="root",
            need="original question",
            resolution="resolved",
            children=["kept-need", "reframe-need", "drop-need"],
            detail=UnresolvedNeed(description="original question"),
        ),
        "kept-need": NeedNode(
            need_id="kept-need",
            need="kept as worded",
            resolution="unresolved",
            detail=UnresolvedNeed(description="kept as worded"),
        ),
        "reframe-need": NeedNode(
            need_id="reframe-need",
            need="wrong framing",
            resolution="unresolved",
            children=["stale-child"],
            depends_on=["kept-need"],
            detail=UnresolvedNeed(description="wrong framing"),
        ),
        "stale-child": NeedNode(
            need_id="stale-child",
            need="stale child",
            resolution="unresolved",
            detail=UnresolvedNeed(description="stale child"),
        ),
        "drop-need": NeedNode(
            need_id="drop-need",
            need="irrelevant tangent",
            resolution="unresolved",
            detail=UnresolvedNeed(description="irrelevant tangent"),
        ),
    }


class _ReframesOneDropsAnotherReasoner:
    """Stub FastEvolutionReasoner: keeps kept-need as worded, reframes
    reframe-need onto new wording, drops drop-need as never having been
    able to help answer the original question. propose_repair returns an
    empty RepairPlan on purpose -- this scenario's own point is that the
    alignment pass (not repair targeting) is what reshapes the graph."""

    def propose_repair(self, *, package):
        return RepairPlan(actions=[])

    def assess_need_alignment(self, *, question, package):
        return NeedAlignmentPlan(
            verdicts=[
                NeedAlignmentVerdict(need_id="kept-need", verdict="keep"),
                NeedAlignmentVerdict(
                    need_id="reframe-need",
                    verdict="reframe",
                    reframed_need="the corrected framing",
                ),
                NeedAlignmentVerdict(need_id="drop-need", verdict="drop"),
            ]
        )

    def extract_answer_obligations(self, *, question):
        return []

    def check_obligation_coverage(self, *, question, obligations, evidence):
        return []


class _NeverResolvesAssignsEverythingReasoner(_PassthroughLookupsReasoner):
    """The retry's own WorkerReasoner: assigns worker-src to every ready
    need every round but never resolves anything -- keeps this scenario
    to exactly the graph-shape effects of alignment, with no evidence-
    upgrade or abandonment machinery in play."""

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
        incomplete_parents,
        cross_repo_experience,
        validation_feedback="",
        repair_guidance="",
        stuck_tried_workers=None,
        candidate_probes=None,
    ):
        return RoundPlan(assignments={need_id: [workers[0].id] for need_id in frontier.ready})


def test_retry_from_trajectory_reframes_and_drops_needs_without_abandoning_the_dropped_one(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("def unrelated():\n    pass\n", encoding="utf-8")
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )
    prior_state = EvidenceState(
        question="original question",
        answer="a tentative partial answer",
        final_need_graph=_fast_repair_graph_fixture(),
        final_recovery_state=RecoverySnapshot(
            tried_workers_by_node={"reframe-need": ["worker-src"]}
        ),
    )

    retried = LocalCoordinator(
        tmp_path, [worker], reasoner=_NeverResolvesAssignsEverythingReasoner()
    ).retry_from_trajectory(
        prior_state, fast_reasoner=_ReframesOneDropsAnotherReasoner(), max_rounds=1
    )

    # Reframe (correction 2): the node's own wording is rewritten, and its
    # stale old-framing edges (children/depends_on) are cleared rather
    # than carried into the new framing.
    reframed = retried.final_need_graph["reframe-need"]
    assert reframed.need == "the corrected framing"
    assert reframed.children == []
    assert reframed.depends_on == []

    # Drop (correction 5): the node's own structure is untouched (still a
    # real node, still worded the same) -- only the caller's bookkeeping
    # changes. It must never land in abandoned_node_ids (a different fact:
    # "never a legitimate question" vs. "we tried and failed").
    assert "drop-need" in retried.final_need_graph
    assert retried.final_need_graph["drop-need"].need == "irrelevant tangent"
    assert "drop-need" not in retried.final_recovery_state.abandoned_node_ids

    # A discarded-misaligned node is excluded from the frontier entirely
    # -- no round ever touches it -- and from the surfaced unresolved-
    # needs list, unlike an ordinary unresolved need (kept-need, or
    # reframe-need under its NEW wording), which is reported honestly.
    executed_need_ids = {
        trace.need_id for round_state in retried.rounds for trace in round_state.node_executions
    }
    assert "drop-need" not in executed_need_ids
    unresolved_descriptions = {need.description for need in retried.unresolved_needs}
    assert "irrelevant tangent" not in unresolved_descriptions
    assert "kept as worded" in unresolved_descriptions
    assert "the corrected framing" in unresolved_descriptions
    assert "wrong framing" not in unresolved_descriptions


class _ProposesAnOffTopicNodeReasoner(_PassthroughLookupsReasoner):
    """The retry's own WorkerReasoner: round 0 both assigns the one real
    need AND proposes a brand-new node via graph_updates -- reproducing a
    need born mid-retry (e.g. via decomposition), not carried over from
    gen0. consolidate_graph records every enforce_alignment value it was
    called with and rejects the off-topic proposal specifically because
    enforce_alignment is set, proving the per-round alignment check (not
    just the one-time pre-retry pass) is what catches it."""

    def __init__(self) -> None:
        self.enforce_alignment_values: list[bool] = []

    def observe(self, *, question, worker_id, territory_id, evidence):
        return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

    def check_need_resolution(self, *, need, new_evidence, question):
        return NeedResolution(status="unresolved")

    def consolidate_graph(
        self, *, question, active_nodes, proposals, candidate_hints, enforce_alignment=False
    ):
        self.enforce_alignment_values.append(enforce_alignment)
        return GraphConsolidationPlan(
            decisions=[
                GraphConsolidationDecision(
                    proposal_id=proposal.proposal_id,
                    action="drop" if enforce_alignment and proposal.need == "off-topic tangent"
                    else "create",
                )
                for proposal in proposals
            ]
        )

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
        graph_updates = {}
        if "off-topic-node" not in graph.nodes:
            graph_updates["off-topic-node"] = NeedNode(
                need_id="off-topic-node",
                need="off-topic tangent",
                detail=UnresolvedNeed(description="off-topic tangent"),
            )
        return RoundPlan(
            assignments={need_id: [workers[0].id] for need_id in frontier.ready},
            graph_updates=graph_updates,
        )


class _EmptyRepairPlanKeepsEverythingReasoner:
    """Stub FastEvolutionReasoner: an empty RepairPlan (repair_guidance
    renders empty text) and every stuck node defaults to keep -- this
    scenario's own point is that enforce_alignment must still reach
    consolidate_graph as True regardless (correction 3), not inferred
    from bool(repair_guidance)."""

    def propose_repair(self, *, package):
        return RepairPlan(actions=[])

    def assess_need_alignment(self, *, question, package):
        return NeedAlignmentPlan()

    def extract_answer_obligations(self, *, question):
        return []

    def check_obligation_coverage(self, *, question, obligations, evidence):
        return []


def test_retry_from_trajectory_enforces_alignment_mid_retry_with_an_empty_repair_plan(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("def unrelated():\n    pass\n", encoding="utf-8")
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )
    prior_state = EvidenceState(
        question="original question",
        answer="a tentative partial answer",
        final_need_graph={
            "root": NeedNode(
                need_id="root",
                need="original question",
                resolution="resolved",
                children=["target-need"],
                detail=UnresolvedNeed(description="original question"),
            ),
            "target-need": NeedNode(
                need_id="target-need",
                need="target need",
                resolution="unresolved",
                detail=UnresolvedNeed(description="target need"),
            ),
        },
    )
    reasoner = _ProposesAnOffTopicNodeReasoner()

    retried = LocalCoordinator(tmp_path, [worker], reasoner=reasoner).retry_from_trajectory(
        prior_state, fast_reasoner=_EmptyRepairPlanKeepsEverythingReasoner(), max_rounds=1
    )

    # The new node consolidate_graph rejected under enforce_alignment
    # never becomes a real graph node.
    assert "off-topic-node" not in retried.final_need_graph
    # enforce_alignment reached consolidate_graph as True even though
    # propose_repair returned an empty RepairPlan -- proving it is an
    # explicit parameter, not inferred from repair_guidance's own text.
    assert True in reasoner.enforce_alignment_values


class _RecordingSynthesizer:
    """Stub AnswerSynthesizer that records every call -- used to prove
    the monotonic-repair gate skips synthesis entirely (not just skips
    the upgrade), rather than calling it and discarding the result."""

    def __init__(self) -> None:
        self.synthesize_calls = 0
        self.synthesize_coalition_calls = 0

    def synthesize(self, **kwargs):
        self.synthesize_calls += 1
        return "a freshly synthesized answer"

    def synthesize_coalition(self, **kwargs):
        self.synthesize_coalition_calls += 1
        return "a freshly synthesized coalition answer"


def test_retry_from_trajectory_is_monotonic_when_nothing_is_verified(tmp_path: Path) -> None:
    # Regression test for a real, empirically-confirmed finding (a
    # 48-question audit): a fast-repair retry that earns zero verified
    # GroundedUpdates (the Evidence Upgrade Gate approved no upgrade all
    # retry long -- here because _NeverResolvesAssignsEverythingReasoner
    # never lets anything reach resolved/partial at all) must leave gen0's
    # own answer completely untouched -- no re-selection of evidence, no
    # re-synthesis. 7 real retries that found nothing to target still
    # re-synthesized regardless before this fix; 5 of those 7 scored worse
    # than gen0's own answer purely from that unforced re-synthesis.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("def unrelated():\n    pass\n", encoding="utf-8")
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )
    prior_state = EvidenceState(
        question="original question",
        answer="gen0's own verbatim answer",
        evidence=[
            Evidence(
                path="src/mod.py",
                line_start=1,
                line_end=1,
                quote="def unrelated():",
                reason="gen0's own evidence",
            )
        ],
        final_need_graph={
            "target-need": NeedNode(
                need_id="target-need",
                need="target need",
                resolution="unresolved",
                detail=UnresolvedNeed(description="target need"),
            ),
        },
    )
    synthesizer = _RecordingSynthesizer()

    retried = LocalCoordinator(
        tmp_path,
        [worker],
        reasoner=_NeverResolvesAssignsEverythingReasoner(),
        synthesizer=synthesizer,
    ).retry_from_trajectory(
        prior_state, fast_reasoner=_EmptyRepairPlanKeepsEverythingReasoner(), max_rounds=1
    )

    assert retried.answer == "gen0's own verbatim answer"
    assert synthesizer.synthesize_calls == 0
    assert synthesizer.synthesize_coalition_calls == 0


class _ProposesOneUncoveredObligationReasoner:
    """Stub FastEvolutionReasoner: an empty RepairPlan (nothing in the
    existing graph is targeted) but one answer obligation the starting
    evidence pool does not cover -- reproducing the live qibo case this
    mechanism exists for (a "which subclasses, and their overridden
    methods" question where check_need_resolution accepted "found the one
    subclass" as resolving the whole node, leaving the overridden-methods
    half uncovered and only ever honestly hedged as unknown)."""

    def propose_repair(self, *, package):
        return RepairPlan(actions=[])

    def assess_need_alignment(self, *, question, package):
        return NeedAlignmentPlan()

    def extract_answer_obligations(self, *, question):
        return [AnswerObligation(obligation_id="obligation-0", description="overridden methods")]

    def check_obligation_coverage(self, *, question, obligations, evidence):
        return [
            ObligationCoverage(obligation_id=item.obligation_id, covered=False)
            for item in obligations
        ]


def test_retry_from_trajectory_injects_a_node_for_an_uncovered_obligation(tmp_path: Path) -> None:
    # Regression test for a real, empirically-confirmed finding: a node
    # can read resolution="resolved" -- unresolved_needs reads 0, nothing
    # for propose_repair to target -- while still leaving part of the
    # ORIGINAL question uncovered (see AnswerObligation's own docstring).
    # Before this mechanism, that gap was invisible to fast-repair
    # targeting; confirmed live it drove a real -6 point regression (gen0
    # honestly hedged, fast-gen1 re-synthesized the same incomplete
    # evidence and still scored worse). Question Coverage must inject a
    # real, live leaf node for the gap so the round loop actually
    # searches for it, not just report it after the fact the way
    # _coverage_needs elsewhere in this module already does.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("def unrelated():\n    pass\n", encoding="utf-8")
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )
    prior_state = EvidenceState(
        question="original question",
        answer="gen0's own incomplete but honestly-hedged answer",
        final_need_graph={
            "root": NeedNode(
                need_id="root",
                need="original question",
                resolution="resolved",
                detail=UnresolvedNeed(description="original question"),
            ),
        },
    )

    retried = LocalCoordinator(
        tmp_path, [worker], reasoner=_NeverResolvesAssignsEverythingReasoner()
    ).retry_from_trajectory(
        prior_state, fast_reasoner=_ProposesOneUncoveredObligationReasoner(), max_rounds=1
    )

    assert "coverage-obligation-0" in retried.final_need_graph
    executed_need_ids = {
        trace.need_id for round_state in retried.rounds for trace in round_state.node_executions
    }
    assert "coverage-obligation-0" in executed_need_ids


def _proposal(
    proposal_id: str,
    need: str = "some gap",
    *,
    depends_on: list[str] | None = None,
    children: list[str] | None = None,
    related_to: list[str] | None = None,
    parent: str = "",
    source: str = "orchestrator",
) -> ProposedNode:
    return ProposedNode(
        proposal_id=proposal_id,
        need=need,
        detail=UnresolvedNeed(description=need),
        proposed_depends_on=depends_on or [],
        proposed_children=children or [],
        proposed_related_to=related_to or [],
        proposed_parent=parent,
        source=source,
    )


def test_collect_proposals_combines_orchestrator_new_nodes_and_observed_needs() -> None:
    orchestrator_new_nodes = {
        "new-gap": NeedNode(
            need_id="new-gap", need="a new gap", detail=UnresolvedNeed(description="a new gap")
        ),
    }
    observed = [UnresolvedNeed(description="worker saw this", missing="worker saw this")]

    proposals = _collect_proposals(orchestrator_new_nodes, observed)

    assert len(proposals) == 2
    orchestrator_proposal = next(p for p in proposals if p.source == "orchestrator")
    assert orchestrator_proposal.proposal_id == "new-gap"
    assert orchestrator_proposal.need == "a new gap"
    worker_proposal = next(p for p in proposals if p.source == "worker_observed")
    assert worker_proposal.need == "worker saw this"
    # A worker-observed proposal_id is minted fresh (UnresolvedNeed carries
    # no id of its own) and must not collide with a real node id.
    assert worker_proposal.proposal_id not in orchestrator_new_nodes


def test_apply_consolidation_decisions_create_mints_a_permanent_node() -> None:
    graph = NeedGraph(nodes={})
    proposals = [_proposal("proposal-1", "brand new gap")]
    plan = GraphConsolidationPlan(
        decisions=[GraphConsolidationDecision(proposal_id="proposal-1", action="create")]
    )

    result = _apply_consolidation_decisions(graph, proposals, plan)

    assert set(result.nodes) == {"proposal-1"}
    assert result.nodes["proposal-1"].need == "brand new gap"


def test_apply_consolidation_decisions_defaults_an_undecided_proposal_to_drop() -> None:
    # A proposal the graph-admission LLM call never actually reviewed
    # (most commonly a truncated/malformed response omitting it at scale)
    # must never silently become a real node -- confirmed live on a real
    # yt-dlp fast-repair retry that the OLD "defaults to create" behavior
    # is exactly what let hundreds of never-reviewed worker-observed
    # duplicates through. A still-real gap gets re-observed and
    # re-proposed in a later round regardless -- dropping here is not a
    # permanent loss.
    graph = NeedGraph(nodes={})
    proposals = [_proposal("proposal-1", "no decision reached")]
    plan = GraphConsolidationPlan(decisions=[])

    result = _apply_consolidation_decisions(graph, proposals, plan)

    assert set(result.nodes) == set()


def test_apply_consolidation_decisions_merge_produces_no_new_id_and_enriches_target() -> None:
    existing = NeedNode(
        need_id="existing-gap",
        need="the existing gap",
        detail=UnresolvedNeed(description="the existing gap", relevant_symbols=["QAOA"]),
    )
    graph = NeedGraph(nodes={"existing-gap": existing})
    proposals = [
        ProposedNode(
            proposal_id="proposal-1",
            need="a reworded version of the existing gap",
            detail=UnresolvedNeed(
                description="a reworded version of the existing gap",
                relevant_symbols=["FALQON"],
            ),
        )
    ]
    plan = GraphConsolidationPlan(
        decisions=[
            GraphConsolidationDecision(
                proposal_id="proposal-1", action="merge", target_node_id="existing-gap"
            )
        ]
    )

    result = _apply_consolidation_decisions(graph, proposals, plan)

    assert set(result.nodes) == {"existing-gap"}
    assert sorted(result.nodes["existing-gap"].detail.relevant_symbols) == ["FALQON", "QAOA"]
    # The target's own wording survives a merge (only subsume replaces it).
    assert result.nodes["existing-gap"].need == "the existing gap"


def test_apply_consolidation_decisions_subsume_replaces_target_wording_with_no_new_id() -> None:
    existing = NeedNode(
        need_id="existing-gap",
        need="vague existing gap",
        detail=UnresolvedNeed(description="vague existing gap"),
    )
    graph = NeedGraph(nodes={"existing-gap": existing})
    proposals = [_proposal("proposal-1", "sharper restatement of the same gap")]
    plan = GraphConsolidationPlan(
        decisions=[
            GraphConsolidationDecision(
                proposal_id="proposal-1", action="subsume", target_node_id="existing-gap"
            )
        ]
    )

    result = _apply_consolidation_decisions(graph, proposals, plan)

    assert set(result.nodes) == {"existing-gap"}
    assert result.nodes["existing-gap"].need == "sharper restatement of the same gap"


def test_apply_consolidation_decisions_attach_becomes_a_child_of_the_named_node() -> None:
    parent = NeedNode(
        need_id="parent-node", need="parent", detail=UnresolvedNeed(description="parent")
    )
    graph = NeedGraph(nodes={"parent-node": parent})
    proposals = [_proposal("proposal-1", "child gap")]
    plan = GraphConsolidationPlan(
        decisions=[
            GraphConsolidationDecision(
                proposal_id="proposal-1", action="attach", target_node_id="parent-node"
            )
        ]
    )

    result = _apply_consolidation_decisions(graph, proposals, plan)

    assert set(result.nodes) == {"parent-node", "proposal-1"}
    assert result.nodes["parent-node"].children == ["proposal-1"]


def test_apply_consolidation_decisions_drop_produces_nothing() -> None:
    graph = NeedGraph(nodes={})
    proposals = [_proposal("proposal-1", "already covered by existing evidence")]
    plan = GraphConsolidationPlan(
        decisions=[GraphConsolidationDecision(proposal_id="proposal-1", action="drop")]
    )

    result = _apply_consolidation_decisions(graph, proposals, plan)

    assert result.nodes == {}


def test_apply_consolidation_decisions_resolves_cross_proposal_references() -> None:
    # Two brand-new proposals in the same round, each naming the *other*
    # by its provisional proposal_id -- must resolve regardless of commit
    # order, per _apply_consolidation_decisions' two-pass design.
    graph = NeedGraph(nodes={})
    proposals = [
        _proposal("proposal-1", "gap one", depends_on=["proposal-2"]),
        _proposal("proposal-2", "gap two", related_to=["proposal-1"]),
    ]
    plan = GraphConsolidationPlan(
        decisions=[
            GraphConsolidationDecision(proposal_id="proposal-1", action="create"),
            GraphConsolidationDecision(proposal_id="proposal-2", action="create"),
        ]
    )

    result = _apply_consolidation_decisions(graph, proposals, plan)

    assert result.nodes["proposal-1"].depends_on == ["proposal-2"]
    assert result.nodes["proposal-2"].related_to == ["proposal-1"]


def test_apply_consolidation_decisions_redirects_a_reference_to_a_merged_proposal() -> None:
    # proposal-2 depends on proposal-1, but proposal-1 gets merged into an
    # existing node rather than minted fresh -- the edge must follow the
    # remap to the real target, not dangle on a proposal_id that never
    # became a node.
    existing = NeedNode(
        need_id="existing-gap", need="existing", detail=UnresolvedNeed(description="existing")
    )
    graph = NeedGraph(nodes={"existing-gap": existing})
    proposals = [
        _proposal("proposal-1", "duplicate of existing"),
        _proposal("proposal-2", "depends on the duplicate", depends_on=["proposal-1"]),
    ]
    plan = GraphConsolidationPlan(
        decisions=[
            GraphConsolidationDecision(
                proposal_id="proposal-1", action="merge", target_node_id="existing-gap"
            ),
            GraphConsolidationDecision(proposal_id="proposal-2", action="create"),
        ]
    )

    result = _apply_consolidation_decisions(graph, proposals, plan)

    assert set(result.nodes) == {"existing-gap", "proposal-2"}
    assert result.nodes["proposal-2"].depends_on == ["existing-gap"]


def test_apply_consolidation_decisions_drops_a_dangling_child_ref_on_an_existing_node() -> None:
    # Regression test: confirmed live on a real qibo gen0 run --
    # _merge_plan_into_graph trusts the Orchestrator's direct edit of an
    # EXISTING node's children list as-is, and that edit can legitimately
    # name a brand-new id the Orchestrator is simultaneously proposing via
    # graph_updates this same round. If the Organizer decides "merge" (or
    # "drop") for that proposal, it never becomes a real node -- without
    # this fix the parent's children list still pointed at the vanished
    # proposal_id, and the closure-check loop's
    # graph.nodes[child_id].resolution lookup raised KeyError.
    root = NeedNode(
        need_id="root",
        need="root need",
        detail=UnresolvedNeed(description="root need"),
        children=["proposal-1"],
    )
    existing = NeedNode(
        need_id="existing-gap", need="existing", detail=UnresolvedNeed(description="existing")
    )
    graph = NeedGraph(nodes={"root": root, "existing-gap": existing})
    proposals = [_proposal("proposal-1", "same as existing-gap, reworded")]
    plan = GraphConsolidationPlan(
        decisions=[
            GraphConsolidationDecision(
                proposal_id="proposal-1", action="merge", target_node_id="existing-gap"
            )
        ]
    )

    result = _apply_consolidation_decisions(graph, proposals, plan)

    assert set(result.nodes) == {"root", "existing-gap"}
    # The dangling reference is redirected to the merge target, not left
    # pointing at a node_id that no longer (and never did) exist.
    assert result.nodes["root"].children == ["existing-gap"]


def test_apply_consolidation_decisions_drops_a_child_ref_to_a_dropped_proposal() -> None:
    root = NeedNode(
        need_id="root",
        need="root need",
        detail=UnresolvedNeed(description="root need"),
        children=["proposal-1"],
    )
    graph = NeedGraph(nodes={"root": root})
    proposals = [_proposal("proposal-1", "already covered")]
    plan = GraphConsolidationPlan(
        decisions=[GraphConsolidationDecision(proposal_id="proposal-1", action="drop")]
    )

    result = _apply_consolidation_decisions(graph, proposals, plan)

    assert set(result.nodes) == {"root"}
    assert result.nodes["root"].children == []


def test_prune_dangling_edges_drops_a_child_ref_with_no_matching_node() -> None:
    # Regression test: confirmed live on a real yt-dlp run --
    # plan_round's graph_updates keys and an existing node's children list
    # are two independently-generated JSON fields in the same response,
    # and nothing enforces they name the exact same string (e.g. a
    # hyphen/underscore mismatch). _apply_consolidation_decisions'
    # proposal_id-remap fix only catches mismatches it can recognize as
    # referring to a known proposal -- this is the general safety net
    # that runs after it every round: any children/depends_on/related_to
    # entry naming an id that isn't an actual graph.nodes key is dropped,
    # full stop, regardless of why it went stale.
    root = NeedNode(
        need_id="root",
        need="root need",
        detail=UnresolvedNeed(description="root need"),
        children=["root.child-real", "root.child_ghost"],
        depends_on=["also-ghost"],
        related_to=["still-ghost"],
    )
    child = NeedNode(
        need_id="root.child-real",
        need="real child",
        detail=UnresolvedNeed(description="real child"),
    )
    graph = NeedGraph(nodes={"root": root, "root.child-real": child})

    result = _prune_dangling_edges(graph)

    assert result.nodes["root"].children == ["root.child-real"]
    assert result.nodes["root"].depends_on == []
    assert result.nodes["root"].related_to == []
    assert result.nodes["root.child-real"] == child


def test_prune_dangling_edges_is_a_no_op_when_everything_already_resolves() -> None:
    root = NeedNode(
        need_id="root",
        need="root need",
        detail=UnresolvedNeed(description="root need"),
        children=["child"],
    )
    child = NeedNode(need_id="child", need="child", detail=UnresolvedNeed(description="child"))
    graph = NeedGraph(nodes={"root": root, "child": child})

    result = _prune_dangling_edges(graph)

    assert result.nodes["root"].children == ["child"]
    assert result is graph


def test_candidate_hints_for_proposals_degrades_gracefully_with_no_embedder(monkeypatch) -> None:
    monkeypatch.setattr("ant.coordinator.local.get_shared_embedder", lambda: None)
    active_nodes = {
        "existing-gap": NeedNode(
            need_id="existing-gap", need="existing", detail=UnresolvedNeed(description="existing")
        )
    }
    proposals = [_proposal("proposal-1", "some new gap")]

    assert _candidate_hints_for_proposals(active_nodes, proposals) == {}


def test_cluster_pending_proposals_groups_exact_normalized_duplicates_with_no_embedder(
    monkeypatch,
) -> None:
    # Regression test: several workers independently observing the same
    # gap, worded with only whitespace/case differences, must cluster
    # together even with no embedder available -- this pass is free and
    # unambiguous, it must never depend on the embedder being present.
    monkeypatch.setattr("ant.coordinator.local.get_shared_embedder", lambda: None)
    proposals = [
        _proposal("p1", "  Find HLS parser   behavior"),
        _proposal("p2", "find hls parser behavior"),
        _proposal("p3", "a completely different gap"),
    ]

    clusters = _cluster_pending_proposals(proposals)

    by_representative = {c.representative.proposal_id: c.member_ids for c in clusters}
    assert by_representative == {"p1": ["p1", "p2"], "p3": ["p3"]}


def test_cluster_pending_proposals_is_a_noop_with_no_embedder_and_distinct_text(
    monkeypatch,
) -> None:
    monkeypatch.setattr("ant.coordinator.local.get_shared_embedder", lambda: None)
    proposals = [_proposal("p1", "gap one"), _proposal("p2", "gap two")]

    clusters = _cluster_pending_proposals(proposals)

    assert {c.representative.proposal_id: c.member_ids for c in clusters} == {
        "p1": ["p1"],
        "p2": ["p2"],
    }


class _FakeEmbedder:
    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self.vectors_by_text = vectors_by_text

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors_by_text[text] for text in texts]


def test_cluster_pending_proposals_groups_near_duplicates_via_embedding_similarity(
    monkeypatch,
) -> None:
    # Three workers each independently observe "the same gap" worded
    # differently enough that exact-text dedup alone would miss it --
    # confirmed live as the actual failure mode on a real yt-dlp
    # fast-repair retry (hundreds of worker_observed proposals, most
    # never merged). p1/p2 are near-identical embeddings (same cluster);
    # p3 is orthogonal (its own cluster).
    embedder = _FakeEmbedder(
        {
            "trace hls manifest extraction": [1.0, 0.0],
            "understand m3u8 parsing path": [0.999, 0.045],
            "something about the downloader retry logic": [0.0, 1.0],
        }
    )
    monkeypatch.setattr("ant.coordinator.local.get_shared_embedder", lambda: embedder)
    proposals = [
        _proposal("p1", "trace hls manifest extraction"),
        _proposal("p2", "understand m3u8 parsing path"),
        _proposal("p3", "something about the downloader retry logic"),
    ]

    clusters = _cluster_pending_proposals(proposals)

    by_representative = {c.representative.proposal_id: sorted(c.member_ids) for c in clusters}
    assert by_representative == {"p1": ["p1", "p2"], "p3": ["p3"]}


def test_cluster_pending_proposals_keeps_dissimilar_embeddings_separate(monkeypatch) -> None:
    embedder = _FakeEmbedder(
        {
            "gap one": [1.0, 0.0],
            "gap two": [0.0, 1.0],
        }
    )
    monkeypatch.setattr("ant.coordinator.local.get_shared_embedder", lambda: embedder)
    proposals = [_proposal("p1", "gap one"), _proposal("p2", "gap two")]

    clusters = _cluster_pending_proposals(proposals)

    assert {c.representative.proposal_id: c.member_ids for c in clusters} == {
        "p1": ["p1"],
        "p2": ["p2"],
    }


def test_expand_cluster_decisions_fans_a_create_verdict_out_as_merge_into_the_representative() -> (
    None
):
    cluster = ProposalCluster(
        representative=_proposal("p1", "trace hls manifest extraction"),
        member_ids=["p1", "p2", "p3"],
    )
    plan = GraphConsolidationPlan(
        decisions=[GraphConsolidationDecision(proposal_id="p1", action="create")]
    )

    expanded = _expand_cluster_decisions([cluster], plan, NeedGraph(nodes={}))

    by_id = {d.proposal_id: d for d in expanded.decisions}
    assert by_id["p1"].action == "create"
    assert by_id["p2"].action == "merge"
    assert by_id["p2"].target_node_id == "p1"
    assert by_id["p3"].action == "merge"
    assert by_id["p3"].target_node_id == "p1"


def test_expand_cluster_decisions_fans_a_merge_verdict_out_identically() -> None:
    graph = NeedGraph(
        nodes={
            "existing-gap": NeedNode(
                need_id="existing-gap",
                need="existing",
                detail=UnresolvedNeed(description="existing"),
            )
        }
    )
    cluster = ProposalCluster(
        representative=_proposal("p1", "a reworded version of the existing gap"),
        member_ids=["p1", "p2"],
    )
    plan = GraphConsolidationPlan(
        decisions=[
            GraphConsolidationDecision(
                proposal_id="p1", action="merge", target_node_id="existing-gap"
            )
        ]
    )

    expanded = _expand_cluster_decisions([cluster], plan, graph)

    by_id = {d.proposal_id: d for d in expanded.decisions}
    assert by_id["p1"].action == "merge"
    assert by_id["p1"].target_node_id == "existing-gap"
    # A member of the SAME cluster merges into the same existing target,
    # not into the representative itself (the representative never became
    # a new node in this branch).
    assert by_id["p2"].action == "merge"
    assert by_id["p2"].target_node_id == "existing-gap"


def test_expand_cluster_decisions_treats_an_invalid_merge_target_like_apply_will() -> None:
    # Mirrors _apply_consolidation_decisions' own "merge/subsume with no
    # real target falls back to create" rule -- a member must merge into
    # whatever the representative will ACTUALLY resolve to (its own new
    # id), not into the invalid target literally named in the decision,
    # or applying this plan would mint one node per member instead of one
    # node total.
    cluster = ProposalCluster(
        representative=_proposal("p1", "gap"), member_ids=["p1", "p2"]
    )
    plan = GraphConsolidationPlan(
        decisions=[
            GraphConsolidationDecision(
                proposal_id="p1", action="merge", target_node_id="does-not-exist"
            )
        ]
    )

    expanded = _expand_cluster_decisions([cluster], plan, NeedGraph(nodes={}))

    by_id = {d.proposal_id: d for d in expanded.decisions}
    assert by_id["p2"].action == "merge"
    assert by_id["p2"].target_node_id == "p1"


def test_expand_cluster_decisions_leaves_every_member_undecided_with_no_verdict() -> None:
    cluster = ProposalCluster(
        representative=_proposal("p1", "gap"), member_ids=["p1", "p2", "p3"]
    )
    plan = GraphConsolidationPlan(decisions=[])

    expanded = _expand_cluster_decisions([cluster], plan, NeedGraph(nodes={}))

    assert expanded.decisions == []


def test_consolidate_and_commit_merges_duplicate_proposals_before_the_reasoner_sees_them(
    tmp_path: Path, monkeypatch
) -> None:
    # Integration-level regression test for the real yt-dlp cost/scale
    # bug: several near-duplicate proposals from the same round must
    # collapse to ONE representative before consolidate_graph is ever
    # called, and the final graph must end up with exactly one real node
    # for them, not one per proposal.
    embedder = _FakeEmbedder(
        {
            "trace hls manifest extraction": [1.0, 0.0],
            "understand m3u8 parsing path": [0.999, 0.045],
            "an unrelated gap": [0.0, 1.0],
        }
    )
    monkeypatch.setattr("ant.coordinator.local.get_shared_embedder", lambda: embedder)

    seen_proposal_counts: list[int] = []

    class _RecordsProposalCountReasoner(_PassthroughLookupsReasoner):
        def observe(self, *, question, worker_id, territory_id, evidence):
            return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

        def consolidate_graph(
            self, *, question, active_nodes, proposals, candidate_hints, enforce_alignment=False
        ):
            seen_proposal_counts.append(len(proposals))
            return GraphConsolidationPlan(
                decisions=[
                    GraphConsolidationDecision(proposal_id=p.proposal_id, action="create")
                    for p in proposals
                ]
            )

    coordinator = LocalCoordinator(
        tmp_path, [], reasoner=_RecordsProposalCountReasoner()
    )
    proposals = [
        _proposal("p1", "trace hls manifest extraction"),
        _proposal("p2", "understand m3u8 parsing path"),
        _proposal("p3", "an unrelated gap"),
    ]

    result_graph = coordinator._consolidate_and_commit(
        "question", NeedGraph(nodes={}), proposals, RecoveryState()
    )

    # The reasoner only ever saw 2 proposals (the two distinct clusters'
    # representatives), not the raw 3.
    assert seen_proposal_counts == [2]
    assert set(result_graph.nodes) == {"p1", "p3"}
