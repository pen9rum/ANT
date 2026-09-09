"""Boundary tests for the no-self-evolution ANT runtime.

The new paper/system boundary: ANT adapts coordination only within a
single repository-understanding episode. No completed question may
permanently change the worker population, routing policy, memory, or
organization used by future questions. This file is the dedicated
regression suite for that boundary specifically -- it does not re-test
mechanisms (Need Graph consolidation, recovery, facet rescue, ...)
already covered exhaustively in test_local_coordinator.py, except where a
boundary-specific angle (e.g. "and nothing here writes to colony memory")
needs its own assertion.

Letters below match the required-tests list from the no-self-evolution
implementation task:
  A - no cross-task memory read
  B - no cross-task memory write
  (C, D - base-worker local exhaustion / reframe-reset: see
   test_local_coordinator.py's test_base_worker_attempts_are_tracked_the_
   same_as_evolved_ones / test_a_reframed_need_resets_local_exhaustion_for_
   the_same_worker -- the mechanism has no worker-type branch left to test
   separately here)
  E - exhaustion is task-local
  F - runtime Need Graph structural revision survives the cleanup
  G - routing remains task-local
  H - task-local recovery still works, and writes nothing persistent
  I - FAST repair (retry_from_trajectory) remains memory-free
  J - worker population invariant across independent questions
"""

from __future__ import annotations

from pathlib import Path

from ant.coordinator import LocalCoordinator
from ant.coordinator.local import RecoveryState, _record_worker_need_attempt
from ant.coordinator.worker_retrieval import build_worker_index, rank_workers
from ant.domain import (
    AnswerObligation,
    Evidence,
    EvidenceState,
    NeedAlignmentPlan,
    NeedGraph,
    NeedNode,
    NeedResolution,
    ObligationCoverage,
    RepairPlan,
    RoundPlan,
    UnresolvedNeed,
    WorkerCard,
    WorkerObservation,
)
from ant.environment import RepoEnvironment
from ant.evaluation.datasets import EvalExample
from ant.evaluation.runner import run_batch
from ant.indexing import build_worker_cards, discover_territories
from ant.memory import ColonyMemoryStore, IndexStore, default_global_memory_path
from ant.providers.mock import MockLLMProvider


def _index_a_trivial_repo(repo: Path, index_path: Path) -> None:
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "auth.py").write_text(
        "def authenticate_user():\n    return True\n", encoding="utf-8"
    )
    environment = RepoEnvironment(repo)
    territories = discover_territories(environment)
    workers = build_worker_cards(environment.root, territories)
    IndexStore(index_path).save(territories, workers)


# --- A/B: no cross-task memory read or write ---------------------------


def test_a_ordinary_runtime_does_not_read_cross_task_memory(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    index_path = tmp_path / ".ant"
    _index_a_trivial_repo(repo, index_path)

    captured: list[tuple[list, list]] = []
    real_coordinator_cls = LocalCoordinator

    class _SpyCoordinator(real_coordinator_cls):
        def __init__(self, *args, memory_routes=None, cross_repo_experience=None, **kwargs):
            captured.append((list(memory_routes or []), list(cross_repo_experience or [])))
            super().__init__(
                *args,
                memory_routes=memory_routes,
                cross_repo_experience=cross_repo_experience,
                **kwargs,
            )

    monkeypatch.setattr("ant.evaluation.runner.LocalCoordinator", _SpyCoordinator)

    run_batch(
        examples=[EvalExample(id="q1", question="authenticate", answer="auth.py")],
        repo_root=repo,
        index_path=index_path,
        out_path=tmp_path / "results1.jsonl",
    )
    run_batch(
        examples=[EvalExample(id="q2", question="authenticate again", answer="auth.py")],
        repo_root=repo,
        index_path=index_path,
        out_path=tmp_path / "results2.jsonl",
    )

    assert len(captured) == 2
    for memory_routes, cross_repo_experience in captured:
        # Question 2 must not receive anything from Question 1 -- but
        # neither question should receive anything at all, since the
        # ordinary runtime never reads ColonyMemoryStore/GlobalMemoryStore
        # in the first place (see run_batch's own docstring).
        assert memory_routes == []
        assert cross_repo_experience == []


def test_b_ordinary_runtime_does_not_write_cross_task_memory(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    index_path = tmp_path / ".ant"
    _index_a_trivial_repo(repo, index_path)
    global_memory_path = tmp_path / "global_memory"
    monkeypatch.setenv("ANT_GLOBAL_MEMORY_PATH", str(global_memory_path))
    assert default_global_memory_path() == global_memory_path

    for example_id, question in [("q1", "authenticate"), ("q2", "authenticate again")]:
        run_batch(
            examples=[EvalExample(id=example_id, question=question, answer="auth.py")],
            repo_root=repo,
            index_path=index_path,
            out_path=tmp_path / f"{example_id}-results.jsonl",
        )

    # No collaboration episode, no route -- record_task_memory was never
    # called. (IndexStore.save_trace still uses the same ant.sqlite3 file
    # for its own, unrelated per-question trace table -- saving a trace is
    # not cross-task memory, nothing reads it back for routing/prompt
    # hints -- so its existence alone would not be evidence either way;
    # checking the colony tables directly is the real assertion.)
    assert ColonyMemoryStore(index_path).all_routes() == []
    assert ColonyMemoryStore(index_path).aggregate_episodes() == []
    # No global cross-repo experience recorded either.
    assert not global_memory_path.exists()


# --- E: exhaustion is task-local -----------------------------------------


def test_e_local_exhaustion_does_not_survive_into_a_new_question() -> None:
    worker = WorkerCard(id="worker-a", territory_id="a", name="a", root="a", files=["a.py"])
    same_item = Evidence(
        path="a.py", line_start=1, line_end=2, quote="q1", reason="r", worker_id=worker.id
    )

    question_one_recovery = RecoveryState()
    _record_worker_need_attempt(question_one_recovery, worker, "root", "the need", [same_item])
    _record_worker_need_attempt(question_one_recovery, worker, "root", "the need", [same_item])
    assert (
        question_one_recovery.worker_need_attempts[(worker.id, "root")].locally_exhausted is True
    )

    # A new question starts a fresh RecoveryState -- never seeded from a
    # previous one anywhere in the ordinary runtime (ask() defaults
    # initial_recovery=None -> RecoveryState()). Nothing about
    # question_one_recovery is threaded into it.
    question_two_recovery = RecoveryState()
    assert question_two_recovery.worker_need_attempts == {}
    assert (
        _record_worker_need_attempt(
            question_two_recovery, worker, "root", "the need", [same_item]
        )
        is None
    )
    attempt = question_two_recovery.worker_need_attempts[(worker.id, "root")]
    assert attempt.attempt_count == 1
    assert attempt.locally_exhausted is False


# --- F: runtime Need Graph structural revision survives the cleanup -----


class _ProposesAStandaloneNewNodeReasoner:
    """Round 0: assigns root, and separately proposes a brand-new,
    independent node via graph_updates (a new-id entry -- see RoundPlan's
    own docstring). The default consolidate_graph below admits every
    proposal ("create"), so this exercises the full pipeline: Potential
    Needs Buffer -> consolidation/admission gate -> a real, permanent
    graph node distinct from anything the initial graph seeded.
    """

    def select_lookups(self, *, need, evidence, candidates):
        return candidates

    def select_evidence(self, *, question, evidence, limit):
        return [str(index) for index in range(len(evidence))][:limit], []

    def consolidate_graph(
        self, *, question, active_nodes, proposals, candidate_hints, enforce_alignment=False
    ):
        from ant.domain import GraphConsolidationDecision, GraphConsolidationPlan

        return GraphConsolidationPlan(
            decisions=[
                GraphConsolidationDecision(proposal_id=proposal.proposal_id, action="create")
                for proposal in proposals
            ]
        )

    def observe(self, *, question, worker_id, territory_id, evidence):
        return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

    def check_need_resolution(self, *, need, new_evidence, question):
        return NeedResolution(status="resolved" if new_evidence else "unresolved")

    def plan_worker_actions(
        self, *, need, evidence, candidate_symbols, available_tools, hints, max_actions
    ):
        return []

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
        worker_need_attempt_states=None,
    ):
        return RoundPlan(
            assignments={need_id: [workers[0].id] for need_id in frontier.ready},
            graph_updates={
                "discovered-need": NeedNode(
                    need_id="discovered-need",
                    need="a newly discovered need",
                    detail=UnresolvedNeed(description="a newly discovered need"),
                ),
            },
        )


def test_f_a_newly_proposed_need_is_admitted_as_a_real_graph_node(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("def target_function():\n    pass\n", encoding="utf-8")
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )

    state = LocalCoordinator(
        tmp_path, [worker], reasoner=_ProposesAStandaloneNewNodeReasoner()
    ).ask("root question", max_rounds=1)

    assert "discovered-need" in state.final_need_graph
    node = state.final_need_graph["discovered-need"]
    assert node.need == "a newly discovered need"
    # Gap-node creation from a "partial" closure verdict is the other half
    # of genuine structural revision -- already covered, and still passing
    # after this cleanup, by
    # test_closure_check_survives_a_partial_verdict_creating_a_gap_node in
    # test_local_coordinator.py; not duplicated here.


def test_f_a_dependency_edit_on_an_existing_node_is_applied(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text("def target_function():\n    pass\n", encoding="utf-8")
    worker = WorkerCard(
        id="worker-src", territory_id="src", name="src", root="src", files=["src/mod.py"]
    )
    seeded_graph = NeedGraph(
        nodes={
            "a": NeedNode(need_id="a", need="need a", detail=UnresolvedNeed(description="need a")),
            "b": NeedNode(need_id="b", need="need b", detail=UnresolvedNeed(description="need b")),
        }
    )

    class _EditsBsDependencyReasoner(_ProposesAStandaloneNewNodeReasoner):
        def plan_round(self, *, graph, frontier, workers, **kwargs):
            if not graph.nodes["b"].depends_on:
                # Existing-id graph_updates entry -> applied directly, not
                # a proposal (see RoundPlan's own docstring).
                return RoundPlan(
                    graph_updates={
                        "b": graph.nodes["b"].model_copy(update={"depends_on": ["a"]}),
                    },
                    assignments={"a": [workers[0].id]},
                )
            return RoundPlan(assignments={need_id: [workers[0].id] for need_id in frontier.ready})

    state = LocalCoordinator(
        tmp_path, [worker], reasoner=_EditsBsDependencyReasoner()
    ).ask("root question", max_rounds=1, initial_graph=seeded_graph)

    assert state.final_need_graph["b"].depends_on == ["a"]


# --- G: routing remains task-local ---------------------------------------


def test_g_rank_workers_is_a_pure_function_of_cards_and_query() -> None:
    workers = [
        WorkerCard(
            id="worker-auth",
            territory_id="auth",
            name="auth",
            root="auth",
            files=["auth/login.py"],
            searchable_terms=["authenticate", "login"],
        ),
        WorkerCard(
            id="worker-db",
            territory_id="db",
            name="db",
            root="db",
            files=["db/models.py"],
            searchable_terms=["query", "model"],
        ),
    ]
    index = build_worker_index(workers)

    first = rank_workers("authenticate", workers, index, embedding_index=None)
    second = rank_workers("authenticate", workers, index, embedding_index=None)

    # Same WorkerCards, same query, same result every time -- no hidden
    # state (no route memory, no prior-question history) can be feeding
    # this. rank_workers' own signature also carries no such parameter.
    assert first == second
    assert "worker-auth" in first
    assert "worker-db" not in first  # no channel found any signal for it

    db_ranked = rank_workers("query model", workers, index, embedding_index=None)
    assert "worker-db" in db_ranked
    assert "worker-auth" not in db_ranked

    import inspect

    signature = inspect.signature(rank_workers)
    assert "memory_routes" not in signature.parameters
    assert "cross_repo_experience" not in signature.parameters


# --- H: task-local recovery still works, and writes nothing persistent --


class _NeverResolvesSingleWorkerReasoner:
    def select_lookups(self, *, need, evidence, candidates):
        return candidates

    def select_evidence(self, *, question, evidence, limit):
        return [str(index) for index in range(len(evidence))][:limit], []

    def consolidate_graph(
        self, *, question, active_nodes, proposals, candidate_hints, enforce_alignment=False
    ):
        from ant.domain import GraphConsolidationPlan

        return GraphConsolidationPlan(decisions=[])

    def observe(self, *, question, worker_id, territory_id, evidence):
        return WorkerObservation(worker_id=worker_id, territory_id=territory_id)

    def check_need_resolution(self, *, need, new_evidence, question):
        return NeedResolution(status="unresolved")

    def plan_worker_actions(
        self, *, need, evidence, candidate_symbols, available_tools, hints, max_actions
    ):
        return []

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
        worker_need_attempt_states=None,
    ):
        targets = set(frontier.ready) | {n for group in frontier.stuck_subgraphs for n in group}
        return RoundPlan(assignments={need_id: [workers[0].id] for need_id in targets})


def test_h_a_stalled_need_reroutes_instead_of_repeating_and_writes_nothing_persistent(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def a():\n    pass\n", encoding="utf-8")
    worker = WorkerCard(id="worker-a", territory_id="a", name="a", root="src", files=["src/a.py"])
    index_path = tmp_path / ".ant"

    state = LocalCoordinator(
        tmp_path, [worker], reasoner=_NeverResolvesSingleWorkerReasoner(), index_path=index_path
    ).ask("question", max_rounds=4)

    # Ordinary blind repetition is prevented: at least one later round is
    # diverted to global_fallback (an unrestricted, repo-wide search)
    # instead of mechanically reassigning the same exhausted worker again.
    tactics = [
        trace.special_tactic
        for round_state in state.rounds
        for trace in round_state.node_executions
    ]
    assert "global_fallback" in tactics
    # Stuck state was genuinely detected and tracked (final_recovery_state
    # is coordinator-local bookkeeping, thrown away at the end of ask() --
    # see EvidenceState.final_recovery_state's own docstring).
    assert state.final_recovery_state is not None
    # Nothing about this recovery touched persistent memory -- ask() never
    # calls record_task_memory itself; only a caller explicitly opting in
    # (see run_batch's use_cross_task_memory) does.
    assert not (index_path / "ant.sqlite3").exists()


# --- I: FAST repair (retry_from_trajectory) remains memory-free ---------


class _NoOpFastRepairReasoner:
    """Minimal FastEvolutionReasoner: no alignment changes, no repair
    actions, no obligations -- a valid "just retry with carried-forward
    state" verdict (see propose_repair's own docstring)."""

    def assess_need_alignment(self, *, question, package):
        return NeedAlignmentPlan(verdicts=[])

    def propose_repair(self, *, package):
        return RepairPlan(actions=[])

    def extract_answer_obligations(self, *, question):
        return [AnswerObligation(obligation_id="o1", description="cover the question")]

    def check_obligation_coverage(self, *, question, obligations, evidence):
        return [
            ObligationCoverage(obligation_id=o.obligation_id, covered=True) for o in obligations
        ]


def test_i_retry_from_trajectory_writes_nothing_to_any_persistent_store(
    tmp_path: Path, monkeypatch
) -> None:
    import ant.coordinator.repair as repair_module

    # Structural proof, not just behavioral: the module implementing FAST
    # repair does not even import the SLOW colony-evolution entry point.
    assert not hasattr(repair_module, "evolve_workers")

    index_path = tmp_path / ".ant"
    global_memory_path = tmp_path / "global_memory"
    monkeypatch.setenv("ANT_GLOBAL_MEMORY_PATH", str(global_memory_path))

    (tmp_path / "a.py").write_text("def target_function():\n    pass\n", encoding="utf-8")
    worker = WorkerCard(id="worker-a", territory_id="a", name="a", root="a", files=["a.py"])
    prior_state = EvidenceState(
        question="Where is target_function defined?",
        answer="a prior, already-complete answer",
        evidence=[],
        final_need_graph={
            "root": NeedNode(
                need_id="root",
                need="Where is target_function defined?",
                resolution="unresolved",
                detail=UnresolvedNeed(description="Where is target_function defined?"),
            ),
        },
    )
    workers_before = [worker.model_copy()]

    coordinator = LocalCoordinator(
        tmp_path, [worker], reasoner=MockLLMProvider(), index_path=index_path
    )
    coordinator.retry_from_trajectory(
        prior_state, fast_reasoner=_NoOpFastRepairReasoner(), max_rounds=1
    )

    assert not (index_path / "ant.sqlite3").exists()
    assert not global_memory_path.exists()
    # The worker population itself is never touched by a fast-repair call.
    assert coordinator.workers == workers_before


# --- J: worker population invariant across independent questions --------


def test_j_worker_population_is_identical_before_and_after_independent_questions(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    index_path = tmp_path / ".ant"
    _index_a_trivial_repo(repo, index_path)

    before = IndexStore(index_path).load_workers()

    for example_id, question in [
        ("q1", "authenticate"),
        ("q2", "who authenticates users"),
        ("q3", "authentication flow"),
    ]:
        run_batch(
            examples=[EvalExample(id=example_id, question=question, answer="auth.py")],
            repo_root=repo,
            index_path=index_path,
            out_path=tmp_path / f"{example_id}-results.jsonl",
        )

    after = IndexStore(index_path).load_workers()

    assert {w.id for w in before} == {w.id for w in after}
    assert len(before) == len(after)
    before_by_id = {w.id: w for w in before}
    for worker in after:
        original = before_by_id[worker.id]
        assert worker.files == original.files
        assert worker.root == original.root
        assert worker.routing_summary == original.routing_summary
        assert worker.lifecycle_state == original.lifecycle_state
