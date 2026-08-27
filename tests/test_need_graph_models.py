from ant.domain import NeedGraph, NeedNode, UnresolvedNeed, WorkerCard


def _node(need_id: str, **overrides) -> NeedNode:
    defaults = {
        "need_id": need_id,
        "need": f"need text for {need_id}",
        "detail": UnresolvedNeed(description=f"need text for {need_id}"),
    }
    defaults.update(overrides)
    return NeedNode(**defaults)


def test_need_node_defaults_are_a_fresh_unresolved_ready_leaf() -> None:
    node = _node("n1")

    assert node.resolution == "unresolved"
    assert node.execution == "ready"
    assert node.progress == "not_stuck"
    assert node.children == []
    assert node.depends_on == []
    assert node.rounds_without_progress == 0


def test_need_node_round_trips_through_json() -> None:
    node = _node(
        "n1",
        depends_on=["n0"],
        related_to=["n2"],
        children=["n1a", "n1b"],
        resolution="partial",
        execution="blocked",
        progress="stuck",
        rounds_without_progress=2,
    )

    restored = NeedNode.model_validate_json(node.model_dump_json())

    assert restored == node


def test_need_graph_holds_nodes_keyed_by_need_id() -> None:
    # NeedGraph is deliberately pure problem structure -- recovery-attempt
    # bookkeeping (streaks, used special tactics, tried workers) lives in
    # the coordinator's own RecoveryState instead, not on the graph.
    parent = _node("n1", children=["n1a", "n1b"])
    child_a = _node("n1a")
    child_b = _node("n1b")
    graph = NeedGraph(nodes={node.need_id: node for node in (parent, child_a, child_b)})

    assert set(graph.nodes) == {"n1", "n1a", "n1b"}
    assert graph.nodes["n1"].children == ["n1a", "n1b"]


def test_worker_card_routing_summary_defaults_to_empty_and_is_independent_of_full_card() -> None:
    card = WorkerCard(
        id="worker-a",
        territory_id="a",
        name="a",
        root="a",
        responsibilities=["does a full job of describing worker a in detail"],
        searchable_terms=["alpha", "beta"],
    )

    assert card.routing_summary == ""

    summarized = card.model_copy(update={"routing_summary": "territory a: alpha/beta specialist"})

    assert summarized.responsibilities == card.responsibilities
    assert summarized.routing_summary == "territory a: alpha/beta specialist"
