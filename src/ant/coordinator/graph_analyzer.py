"""Pure, deterministic analysis over a task's NeedGraph -- no LLM call.

This is the "Dependency Graph Analyzer" from the design: it owns two of
the Need Graph's three orthogonal status dimensions (`execution` and,
indirectly, which nodes make up a genuinely stuck subgraph), computed
fresh from `depends_on` edges each round via classic graph algorithms
(Kahn's-style frontier computation, Tarjan's SCC for cycle detection) --
never guessed by an LLM. The Orchestrator only ever creates/decomposes
nodes and edits `depends_on`/`related_to`; this module is what turns that
edge structure into "what can run this round" and "what genuinely cannot
proceed right now".
"""

from __future__ import annotations

from ant.domain import FrontierResult, NeedGraph


def compute_frontier(graph: NeedGraph) -> FrontierResult:
    """Runs once per round on the coordinator's current, already-validated
    (see find_cycles) NeedGraph.

    A node's raw `execution` field (ready/blocked) is purely a function of
    `depends_on` satisfaction -- but the frontier this function returns for
    round-branching purposes (normal planning vs. Temporary Reorganization)
    additionally excludes any leaf already marked `progress=="stuck"`, even
    one whose dependencies are otherwise satisfied: "stuck" means normal
    round-by-round re-assignment already failed repeatedly on that exact
    node, so counting it as part of an ordinary ready frontier would just
    trigger the same failing approach again. Every stuck leaf instead
    surfaces via `stuck_subgraphs` -- on its own if nothing else depends on
    it, grouped with everything transitively blocked behind it otherwise.
    """
    resolution_by_id = {node_id: node.resolution for node_id, node in graph.nodes.items()}
    unfinished_leaves = [
        node for node in graph.nodes.values() if not node.children and node.resolution != "resolved"
    ]

    ready: list[str] = []
    blocked: list[str] = []
    stuck_leaf_ids: set[str] = set()
    for node in unfinished_leaves:
        if node.progress == "stuck":
            stuck_leaf_ids.add(node.need_id)
        elif _dependencies_satisfied(node.depends_on, resolution_by_id):
            ready.append(node.need_id)
        else:
            blocked.append(node.need_id)

    if ready:
        # Normal planning can proceed on at least part of the graph this
        # round; a stuck subgraph elsewhere does not need surfacing until
        # nothing else is left to do.
        return FrontierResult(ready=sorted(ready), blocked=sorted(blocked), stuck_subgraphs=[])

    cycles = find_cycles(graph)
    cycle_members = {node_id for cycle in cycles for node_id in cycle}
    chains = _blocked_chains_rooted_in_stuck(graph, blocked, stuck_leaf_ids, cycle_members)
    grouped_members = {node_id for chain in chains for node_id in chain}
    isolated_stuck = [
        [node_id] for node_id in sorted(stuck_leaf_ids) if node_id not in grouped_members
    ]
    stuck_subgraphs = _dedupe_groups([*cycles, *chains, *isolated_stuck])
    return FrontierResult(ready=[], blocked=sorted(blocked), stuck_subgraphs=stuck_subgraphs)


def _dependencies_satisfied(depends_on: list[str], resolution_by_id: dict[str, str]) -> bool:
    return all(resolution_by_id.get(dep_id) == "resolved" for dep_id in depends_on)


def find_cycles(graph: NeedGraph) -> list[list[str]]:
    """Tarjan's strongly-connected-components algorithm over the
    `depends_on` edges (node A depends_on B == directed edge A -> B, "A
    needs B resolved first"). Returns every group of node ids forming a
    cycle: an SCC of size > 1, or a single node whose own `depends_on`
    includes itself (a self-loop also counts, per design).

    `depends_on` is LLM-drawn (the Orchestrator's own structured output),
    so a cycle found here is presumptively a planning error -- see the
    Orchestrator wiring that calls this immediately after every planning
    call and rejects/retries rather than accepting a cyclic graph into
    state. Accepted state should therefore always be acyclic; this
    function is also called from compute_frontier() as a defensive
    re-check, not because cycles are expected to occur there.
    """
    index_counter = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    sccs: list[list[str]] = []

    def strongconnect(node_id: str) -> None:
        nonlocal index_counter
        indices[node_id] = index_counter
        lowlink[node_id] = index_counter
        index_counter += 1
        stack.append(node_id)
        on_stack.add(node_id)

        node = graph.nodes.get(node_id)
        successors = node.depends_on if node is not None else []
        for successor_id in successors:
            if successor_id not in graph.nodes:
                continue
            if successor_id not in indices:
                strongconnect(successor_id)
                lowlink[node_id] = min(lowlink[node_id], lowlink[successor_id])
            elif successor_id in on_stack:
                lowlink[node_id] = min(lowlink[node_id], indices[successor_id])

        if lowlink[node_id] == indices[node_id]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == node_id:
                    break
            sccs.append(component)

    for node_id in graph.nodes:
        if node_id not in indices:
            strongconnect(node_id)

    cycles: list[list[str]] = []
    for component in sccs:
        if len(component) > 1:
            cycles.append(sorted(component))
        elif len(component) == 1:
            only_id = component[0]
            node = graph.nodes.get(only_id)
            if node is not None and only_id in node.depends_on:
                cycles.append(component)
    return cycles


def _blocked_chains_rooted_in_stuck(
    graph: NeedGraph,
    blocked_ids: list[str],
    stuck_leaf_ids: set[str],
    cycle_members: set[str],
) -> list[list[str]]:
    """Groups blocked leaves (dependency-unsatisfied, not themselves stuck)
    by the single upstream node whose own progress=="stuck" is the actual
    reason none of them can proceed -- e.g. N1 stuck -> N2 blocked -> N3
    blocked (no cycle, so find_cycles alone would miss it entirely). Every
    blocked node tracing back to the same stuck root is one subgraph,
    since reorganizing that root is what would unblock all of them at
    once. A stuck leaf with nothing depending on it still needs a group of
    its own -- compute_frontier adds those as trivial single-node
    subgraphs for whichever stuck ids this function's groups don't already
    cover.
    """
    groups: dict[str, set[str]] = {}
    candidates = [*blocked_ids, *stuck_leaf_ids]
    for node_id in candidates:
        if node_id in cycle_members:
            continue
        root = _find_stuck_root(graph, node_id, cycle_members)
        if root is None:
            continue
        groups.setdefault(root, set()).update({root, node_id})
    return [sorted(members) for members in groups.values()]


def _find_stuck_root(graph: NeedGraph, start_id: str, cycle_members: set[str]) -> str | None:
    """Walks depends_on outward from `start_id` (toward what it needs
    resolved first) looking for the nearest ancestor already marked
    progress=="stuck" and not yet resolved. Assumes an acyclic graph
    (guaranteed by validation, see find_cycles) -- `cycle_members` is only
    a defensive guard against an infinite walk if that assumption is ever
    violated.
    """
    seen: set[str] = set()
    queue = [start_id]
    while queue:
        current_id = queue.pop()
        if current_id in seen or current_id in cycle_members:
            continue
        seen.add(current_id)
        node = graph.nodes.get(current_id)
        if node is None:
            continue
        if node.progress == "stuck" and node.resolution != "resolved":
            return current_id
        queue.extend(node.depends_on)
    return None


def _dedupe_groups(groups: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    deduped: list[list[str]] = []
    for group in groups:
        key = tuple(sorted(group))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(list(key))
    return deduped
