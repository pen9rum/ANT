from ant.coordinator.graph_analyzer import compute_frontier, find_cycles
from ant.domain import NeedGraph, NeedNode, UnresolvedNeed


def _node(need_id: str, **overrides) -> NeedNode:
    defaults = {
        "need_id": need_id,
        "need": f"need text for {need_id}",
        "detail": UnresolvedNeed(description=f"need text for {need_id}"),
    }
    defaults.update(overrides)
    return NeedNode(**defaults)


def _graph(*nodes: NeedNode) -> NeedGraph:
    return NeedGraph(nodes={node.need_id: node for node in nodes})


def test_independent_unresolved_leaves_are_all_ready() -> None:
    graph = _graph(_node("n1"), _node("n2"))

    result = compute_frontier(graph)

    assert result.ready == ["n1", "n2"]
    assert result.blocked == []
    assert result.stuck_subgraphs == []


def test_node_depending_on_an_unresolved_node_is_blocked_not_ready() -> None:
    graph = _graph(
        _node("n1"),
        _node("n2", depends_on=["n1"]),
    )

    result = compute_frontier(graph)

    assert result.ready == ["n1"]
    assert result.blocked == ["n2"]


def test_node_becomes_ready_once_its_dependency_resolves() -> None:
    graph = _graph(
        _node("n1", resolution="resolved"),
        _node("n2", depends_on=["n1"]),
    )

    result = compute_frontier(graph)

    assert result.ready == ["n2"]
    assert result.blocked == []


def test_resolved_leaves_and_parent_container_nodes_are_excluded_entirely() -> None:
    graph = _graph(
        _node("n1", resolution="resolved"),
        _node("parent", children=["c1", "c2"]),
        _node("c1"),
        _node("c2"),
    )

    result = compute_frontier(graph)

    # n1 is done (resolved) and "parent" is a pure hierarchical container --
    # neither is a ready/blocked candidate; only its leaf children are.
    assert "n1" not in result.ready
    assert "n1" not in result.blocked
    assert "parent" not in result.ready
    assert "parent" not in result.blocked
    assert set(result.ready) == {"c1", "c2"}


def test_self_loop_counts_as_a_cycle() -> None:
    graph = _graph(_node("n1", depends_on=["n1"]))

    cycles = find_cycles(graph)

    assert cycles == [["n1"]]


def test_mutual_dependency_is_detected_as_a_cycle() -> None:
    graph = _graph(
        _node("n1", depends_on=["n2"]),
        _node("n2", depends_on=["n1"]),
    )

    cycles = find_cycles(graph)

    assert cycles == [["n1", "n2"]]


def test_acyclic_chain_has_no_cycles() -> None:
    graph = _graph(
        _node("n1"),
        _node("n2", depends_on=["n1"]),
        _node("n3", depends_on=["n2"]),
    )

    assert find_cycles(graph) == []


def test_blocked_chain_rooted_in_a_stuck_node_is_one_stuck_subgraph() -> None:
    # N1 stuck -> N2 blocked -> N3 blocked. No cycle anywhere, so
    # find_cycles alone would miss this entirely -- compute_frontier must
    # still surface it via the upstream-stuck-root walk.
    graph = _graph(
        _node("n1", progress="stuck"),
        _node("n2", depends_on=["n1"]),
        _node("n3", depends_on=["n2"]),
    )

    result = compute_frontier(graph)

    assert result.ready == []
    assert set(result.blocked) == {"n2", "n3"}
    assert result.stuck_subgraphs == [["n1", "n2", "n3"]]


def test_blocked_node_behind_a_not_yet_stuck_upstream_node_is_not_a_stuck_subgraph() -> None:
    # N2 is blocked on N1, but N1 hasn't been flagged stuck yet (still
    # progressing normally) -- this is ordinary blocking, not a
    # Temporary-Reorganization-worthy deadlock.
    graph = _graph(
        _node("n1"),
        _node("n2", depends_on=["n1"]),
    )

    result = compute_frontier(graph)

    assert result.blocked == ["n2"]
    assert result.stuck_subgraphs == []


def test_a_stuck_subgraph_elsewhere_does_not_suppress_a_real_ready_frontier() -> None:
    graph = _graph(
        _node("independent"),
        _node("n1", progress="stuck"),
        _node("n2", depends_on=["n1"]),
    )

    result = compute_frontier(graph)

    assert result.ready == ["independent"]
    # Normal planning can proceed this round; stuck_subgraphs is only
    # surfaced once there is genuinely nothing else to do.
    assert result.stuck_subgraphs == []
