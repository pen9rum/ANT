from ant.coordinator.repair import (
    ROOT_NEED_ID,
    apply_alignment_verdicts,
    assemble_trajectory_package,
    build_retry_starting_state,
    render_repair_guidance,
    resolve_repair_plan,
)
from ant.domain import (
    AbsenceProof,
    Evidence,
    EvidenceState,
    GraphDelta,
    NeedAlignmentPlan,
    NeedAlignmentVerdict,
    NeedGraph,
    NeedNode,
    NodeExecutionTrace,
    PlanningRound,
    RecoverySnapshot,
    RepairAction,
    RepairPlan,
    StuckEpisodeSnapshot,
    UnresolvedNeed,
    WorkerObservation,
)


def _synthetic_state() -> EvidenceState:
    nodes = {
        "root": NeedNode(
            need_id="root",
            need="root question",
            resolution="resolved",
            children=["stuck-need", "blocked-need"],
            detail=UnresolvedNeed(description="root question"),
        ),
        "stuck-need": NeedNode(
            need_id="stuck-need",
            need="the stuck part",
            resolution="unresolved",
            detail=UnresolvedNeed(
                description="the stuck part",
                missing="the actual implementation location",
                suggested_terms=["foo"],
                suggested_territories=["src-foo"],
            ),
        ),
        "blocked-need": NeedNode(
            need_id="blocked-need",
            need="depends on the stuck part",
            resolution="unresolved",
            depends_on=["stuck-need"],
            detail=UnresolvedNeed(description="depends on the stuck part"),
        ),
    }
    rounds = [
        PlanningRound(
            round_index=0,
            node_executions=[
                NodeExecutionTrace(
                    need_id="stuck-need",
                    need="the stuck part",
                    worker_ids=["worker-a"],
                    resolution="unresolved",
                    evidence_gain=0,
                    need_reduction=0,
                    observations=[
                        WorkerObservation(
                            worker_id="worker-a",
                            territory_id="foo",
                            evidence=[
                                Evidence(
                                    path="a.py",
                                    line_start=1,
                                    line_end=2,
                                    quote="def f(): pass",
                                    reason="looked relevant",
                                    claim="found nothing useful",
                                )
                            ],
                        )
                    ],
                )
            ],
            graph_delta=GraphDelta(created_nodes=["stuck-need", "blocked-need"]),
        ),
        PlanningRound(
            round_index=1,
            node_executions=[
                NodeExecutionTrace(
                    need_id="stuck-need",
                    need="the stuck part",
                    worker_ids=[],
                    resolution="unresolved",
                    special_tactic="temporary_bridge",
                    evidence_gain=0,
                    need_reduction=0,
                )
            ],
            graph_delta=GraphDelta(),
        ),
    ]
    return EvidenceState(
        question="root question",
        answer="a tentative partial answer",
        rounds=rounds,
        final_need_graph=nodes,
        final_recovery_state=RecoverySnapshot(
            stuck_episodes=[
                StuckEpisodeSnapshot(
                    episode_id="ep1",
                    members=["stuck-need"],
                    recovery_streak=3,
                    used_special_tactics=["temporary_bridge"],
                )
            ],
            abandoned_node_ids=["stuck-need"],
            tried_workers_by_node={"stuck-need": ["worker-a", "worker-b"]},
        ),
    )


def test_assemble_trajectory_package_excludes_resolved_nodes() -> None:
    package = assemble_trajectory_package(_synthetic_state())
    need_ids = {node.need_id for node in package.stuck_nodes}
    assert need_ids == {"stuck-need", "blocked-need"}
    assert package.question == "root question"
    assert package.prior_answer == "a tentative partial answer"


def test_assemble_trajectory_package_populates_stuck_node_detail() -> None:
    package = assemble_trajectory_package(_synthetic_state())
    stuck = next(node for node in package.stuck_nodes if node.need_id == "stuck-need")

    assert stuck.tried_worker_ids == ["worker-a", "worker-b"]
    assert stuck.tried_special_tactics == ["temporary_bridge"]
    # Both of stuck-need's two executions had evidence_gain=0 and
    # need_reduction=0 -- neither counts as progress.
    assert stuck.no_progress_execution_count == 2
    assert stuck.is_abandoned is True
    assert stuck.stuck_episode_id == "ep1"
    assert stuck.missing == "the actual implementation location"
    assert stuck.suggested_terms == ["foo"]
    assert "found nothing useful" in stuck.evidence_claims


def test_assemble_trajectory_package_leaves_an_unstuck_dependent_node_alone() -> None:
    package = assemble_trajectory_package(_synthetic_state())
    blocked = next(node for node in package.stuck_nodes if node.need_id == "blocked-need")

    assert blocked.depends_on == ["stuck-need"]
    assert blocked.is_abandoned is False
    assert blocked.stuck_episode_id == ""
    assert blocked.tried_worker_ids == []


def test_assemble_trajectory_package_carries_the_full_decomposition_log() -> None:
    package = assemble_trajectory_package(_synthetic_state())
    assert len(package.graph_decomposition_log) == 2
    assert package.graph_decomposition_log[0].created_nodes == ["stuck-need", "blocked-need"]


def test_resolve_repair_plan_splits_structural_from_forced_execution_actions() -> None:
    plan = RepairPlan(
        actions=[
            RepairAction(
                kind="change_dependency", need_id="blocked-need", new_depends_on=[]
            ),
            RepairAction(kind="redecompose", need_id="stuck-need"),
            RepairAction(
                kind="replace_assignment",
                need_id="stuck-need",
                worker_ids=["worker-c"],
                rationale="worker-a/worker-b both failed",
            ),
        ]
    )

    seed = resolve_repair_plan(plan)

    assert seed.dependency_changes == {"blocked-need": []}
    assert seed.redecompose_node_ids == {"stuck-need"}
    assert seed.forced_assignments == {"stuck-need": ["worker-c"]}
    assert seed.targeted_need_ids == {"blocked-need", "stuck-need"}
    # change_dependency/redecompose are pure graph edits -- no narration
    # needed, the graph itself shows the Orchestrator what changed. Only
    # the forced-execution action gets a guidance line.
    assert len(seed.guidance_lines) == 1
    assert "replace_assignment" in seed.guidance_lines[0]
    assert "worker-c" in seed.guidance_lines[0]


def test_resolve_repair_plan_treats_merge_needs_as_structural_not_forced_execution() -> None:
    plan = RepairPlan(
        actions=[
            RepairAction(
                kind="merge_needs",
                need_id="stuck-need",
                merge_with=["blocked-need"],
                rationale="same underlying gap",
            )
        ]
    )

    seed = resolve_repair_plan(plan)

    assert seed.merges == {"stuck-need": ["blocked-need"]}
    assert seed.forced_assignments == {}
    assert seed.guidance_lines == []


def test_resolve_repair_plan_collects_force_global_search_ids() -> None:
    plan = RepairPlan(actions=[RepairAction(kind="force_global_search", need_id="stuck-need")])

    seed = resolve_repair_plan(plan)

    assert seed.forced_global_search_ids == {"stuck-need"}
    assert len(seed.guidance_lines) == 1
    assert "force_global_search" in seed.guidance_lines[0]


def test_resolve_repair_plan_drops_a_malformed_action_without_worker_ids() -> None:
    plan = RepairPlan(actions=[RepairAction(kind="replace_assignment", need_id="stuck-need")])

    seed = resolve_repair_plan(plan)

    assert seed.forced_assignments == {}
    assert seed.guidance_lines == []


def test_render_repair_guidance_is_empty_when_nothing_advisory_was_proposed() -> None:
    plan = RepairPlan(
        actions=[RepairAction(kind="change_dependency", need_id="blocked-need", new_depends_on=[])]
    )
    seed = resolve_repair_plan(plan)
    package = assemble_trajectory_package(_synthetic_state())

    assert render_repair_guidance(package, seed) == ""


def test_render_repair_guidance_includes_the_advisory_lines() -> None:
    plan = RepairPlan(
        actions=[
            RepairAction(
                kind="replace_assignment", need_id="stuck-need", worker_ids=["worker-c"]
            )
        ]
    )
    seed = resolve_repair_plan(plan)
    package = assemble_trajectory_package(_synthetic_state())

    guidance = render_repair_guidance(package, seed)
    assert "replace_assignment" in guidance
    assert "worker-c" in guidance


def test_build_retry_starting_state_carries_forward_resolved_nodes_and_evidence() -> None:
    state = _synthetic_state()
    state = state.model_copy(
        update={
            "evidence": [
                Evidence(
                    path="root.py", line_start=1, line_end=1, quote="x", reason="root evidence"
                )
            ]
        }
    )
    plan = RepairPlan(
        actions=[RepairAction(kind="change_dependency", need_id="blocked-need", new_depends_on=[])]
    )
    seed = resolve_repair_plan(plan)

    graph, evidence = build_retry_starting_state(state, seed)

    assert set(graph.nodes) == {"root", "stuck-need", "blocked-need"}
    assert graph.nodes["root"].resolution == "resolved"
    assert graph.nodes["blocked-need"].depends_on == []
    assert len(evidence) == 1
    assert evidence[0].reason == "root evidence"


def test_build_retry_starting_state_applies_merge_needs_and_redirects_dependents() -> None:
    state = _synthetic_state()
    plan = RepairPlan(
        actions=[
            RepairAction(kind="merge_needs", need_id="stuck-need", merge_with=["blocked-need"])
        ]
    )
    seed = resolve_repair_plan(plan)

    graph, _ = build_retry_starting_state(state, seed)

    # blocked-need is folded into stuck-need and removed entirely.
    assert set(graph.nodes) == {"root", "stuck-need"}
    # root's children referenced blocked-need -- redirected to stuck-need,
    # deduplicated against the stuck-need entry already there.
    assert graph.nodes["root"].children == ["stuck-need"]


def test_build_retry_starting_state_ignores_a_merge_naming_a_nonexistent_primary() -> None:
    state = _synthetic_state()
    plan = RepairPlan(
        actions=[
            RepairAction(
                kind="merge_needs", need_id="does-not-exist", merge_with=["blocked-need"]
            )
        ]
    )
    seed = resolve_repair_plan(plan)

    graph, _ = build_retry_starting_state(state, seed)

    assert set(graph.nodes) == {"root", "stuck-need", "blocked-need"}


def test_build_retry_starting_state_resets_progress_for_redecompose_targets() -> None:
    state = _synthetic_state()
    state.final_need_graph["stuck-need"].progress = "stuck"
    state.final_need_graph["stuck-need"].rounds_without_progress = 3
    plan = RepairPlan(actions=[RepairAction(kind="redecompose", need_id="stuck-need")])
    seed = resolve_repair_plan(plan)

    graph, _ = build_retry_starting_state(state, seed)

    assert graph.nodes["stuck-need"].progress == "not_stuck"
    assert graph.nodes["stuck-need"].rounds_without_progress == 0


def test_assemble_trajectory_package_computes_epistemic_state_deterministically() -> None:
    # No reasoner/LLM call involved at all -- purely a lookup against the
    # state's own absence_proofs by need_id (see AbsenceProof.need_id and
    # _epistemic_state_for). This is the fix for Grounded Fast Repair's
    # first review correction: epistemic_state must be grounded fact,
    # never an LLM guess.
    state = _synthetic_state()
    state = state.model_copy(
        update={
            "absence_proofs": [
                AbsenceProof(
                    query="root question",
                    need_id="stuck-need",
                    exhaustive=True,
                    conclusion="not_found",
                ),
                AbsenceProof(
                    query="root question",
                    need_id="blocked-need",
                    exhaustive=False,
                    conclusion="inconclusive",
                ),
                # A proof with no need_id at all (e.g. a question-level
                # completeness check like _verify_inheritance_completeness)
                # must not be guessed onto any need -- matches nothing.
                AbsenceProof(query="root question", exhaustive=True, conclusion="not_found"),
            ]
        }
    )

    package = assemble_trajectory_package(state)

    by_id = {node.need_id: node for node in package.stuck_nodes}
    assert by_id["stuck-need"].epistemic_state == "absence_supported"
    assert by_id["blocked-need"].epistemic_state == "insufficient_evidence"


def test_assemble_trajectory_package_epistemic_state_defaults_to_open_with_no_proof() -> None:
    package = assemble_trajectory_package(_synthetic_state())
    assert all(node.epistemic_state == "open" for node in package.stuck_nodes)


def _graph_with(*nodes: NeedNode) -> NeedGraph:
    return NeedGraph(nodes={node.need_id: node for node in nodes})


def test_apply_alignment_verdicts_keep_carries_epistemic_state_over_unchanged() -> None:
    graph = _graph_with(
        NeedNode(need_id="a", need="need a", detail=UnresolvedNeed(description="need a"))
    )
    plan = NeedAlignmentPlan(
        verdicts=[NeedAlignmentVerdict(need_id="a", verdict="keep")]
    )

    aligned, epistemic_states, discarded, reframed = apply_alignment_verdicts(
        graph, plan, {"a": "absence_supported"}
    )

    assert aligned.nodes["a"].need == "need a"
    assert epistemic_states == {"a": "absence_supported"}
    assert discarded == set()
    assert reframed == set()


def test_apply_alignment_verdicts_no_verdict_defaults_to_keep() -> None:
    graph = _graph_with(
        NeedNode(need_id="a", need="need a", detail=UnresolvedNeed(description="need a"))
    )

    aligned, epistemic_states, discarded, reframed = apply_alignment_verdicts(
        graph, NeedAlignmentPlan(), {"a": "insufficient_evidence"}
    )

    assert aligned.nodes["a"].need == "need a"
    assert epistemic_states == {"a": "insufficient_evidence"}
    assert discarded == set()


def test_apply_alignment_verdicts_reframe_rewrites_text_clears_edges_and_resets_to_open() -> None:
    node = NeedNode(
        need_id="a",
        need="role resolution inheritance",
        detail=UnresolvedNeed(description="role resolution inheritance", missing="old missing"),
        children=["a-child"],
        depends_on=["a-dep"],
    )
    graph = _graph_with(
        node,
        NeedNode(need_id="a-child", need="child", detail=UnresolvedNeed(description="child")),
        NeedNode(need_id="a-dep", need="dep", detail=UnresolvedNeed(description="dep")),
    )
    plan = NeedAlignmentPlan(
        verdicts=[
            NeedAlignmentVerdict(
                need_id="a", verdict="reframe", reframed_need="TLS/auth inheritance"
            )
        ]
    )

    aligned, epistemic_states, discarded, reframed = apply_alignment_verdicts(
        graph, plan, {"a": "absence_supported"}
    )

    reframed_node = aligned.nodes["a"]
    assert reframed_node.need == "TLS/auth inheritance"
    assert reframed_node.detail.description == "TLS/auth inheritance"
    assert reframed_node.detail.missing == "TLS/auth inheritance"
    # A reframed need is a fresh investigation (correction 2): old-framing
    # edges are cleared, not carried into the new framing.
    assert reframed_node.children == []
    assert reframed_node.depends_on == []
    # Never inherits the old framing's proof -- resets to open
    # unconditionally, regardless of what prior_epistemic_states said.
    assert epistemic_states["a"] == "open"
    assert reframed == {"a"}
    assert discarded == set()


def test_apply_alignment_verdicts_drop_discards_without_abandoning() -> None:
    graph = _graph_with(
        NeedNode(need_id="a", need="need a", detail=UnresolvedNeed(description="need a"))
    )
    plan = NeedAlignmentPlan(verdicts=[NeedAlignmentVerdict(need_id="a", verdict="drop")])

    aligned, epistemic_states, discarded, reframed = apply_alignment_verdicts(
        graph, plan, {"a": "open"}
    )

    # Node structure is left alone (matches ordinary consolidation "drop"
    # behavior) -- only the caller's bookkeeping changes.
    assert "a" in aligned.nodes
    assert discarded == {"a"}
    assert "a" not in epistemic_states
    assert reframed == set()


def test_apply_alignment_verdicts_coerces_a_root_drop_to_keep() -> None:
    graph = _graph_with(
        NeedNode(
            need_id=ROOT_NEED_ID,
            need="original question",
            detail=UnresolvedNeed(description="original question"),
        )
    )
    plan = NeedAlignmentPlan(
        verdicts=[NeedAlignmentVerdict(need_id=ROOT_NEED_ID, verdict="drop")]
    )

    aligned, epistemic_states, discarded, reframed = apply_alignment_verdicts(
        graph, plan, {ROOT_NEED_ID: "open"}
    )

    assert discarded == set()
    assert epistemic_states[ROOT_NEED_ID] == "open"


def test_apply_alignment_verdicts_ignores_an_unknown_need_id() -> None:
    graph = _graph_with(
        NeedNode(need_id="a", need="need a", detail=UnresolvedNeed(description="need a"))
    )
    plan = NeedAlignmentPlan(
        verdicts=[NeedAlignmentVerdict(need_id="does-not-exist", verdict="drop")]
    )

    aligned, epistemic_states, discarded, reframed = apply_alignment_verdicts(
        graph, plan, {"a": "open"}
    )

    assert discarded == set()
    assert epistemic_states == {"a": "open"}
