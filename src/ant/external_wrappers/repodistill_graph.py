"""RepoDistill component 1/3: GraphRAG (paper section 2.1).

Paper: Xin Yin, Zixiang Ding, Yiang Zhang, Qiang Wang, Rui Wang, Chao Ni,
Zhe Cui. "RepoDistill: Distilling Repository Knowledge through
Compression-Aware Budget Allocation and Policy Optimization." Findings of
ACL 2026, pp. 4425-4443. https://aclanthology.org/2026.findings-acl.217/

REIMPLEMENTED FROM THE PUBLISHED PAPER DESCRIPTION, NOT PORTED. The
paper's own code link is an Anonymous-GitHub mirror that is dead/
unreachable as of this writing, and no trained checkpoints are published
anywhere. Every algorithmic detail below traces to a specific sentence or
numbered equation in section 2.1 of the PDF (fetched from the ACL
Anthology and read directly before implementation); nothing here is
"inspired by" the method at a distance.

Section 2.1 in one paragraph, as implemented:

  2.1.1 Repository Graph Construction -- "Each source file f_i in F is
  parsed into a Concrete Syntax Tree (CST) using tree-sitter. Candidate
  units are extracted via CST traversal, and we record metadata including
  file path, line number, name, and code snippet." Two relation types:
  "(1) CONTAIN edges link classes to their contained functions or inner
  classes, reflecting hierarchical structure; (2) INVOKE edges connect a
  caller unit to its callee by resolving function calls within CST
  subtrees, producing a candidate set of relations."

  2.1.2 Repository Graph Retrieval -- Node Localization picks the Top-K
  anchors by cosine similarity (Equation 1); Reasoning Chain Mining
  expands via first-order INVOKE subtrees and shortest multi-hop paths
  between anchor pairs over INVOKE+CONTAIN; Re-Ranking scores every
  candidate with Equation 2,

      S(q, c) = lambda * sim(q, c) + (1 - lambda) * max_{v in c} sim(q, v)

  with lambda = 0.5.

================================================================
DEVIATIONS FROM THE PAPER'S EXACT SPEC (official -> adapted -> reason)
================================================================

1. EMBEDDING MODEL.
   Official setting: Qwen3-Embedding-0.6B (Zhang et al., 2025), cosine
   similarity, for sim(q, v) in Equations 1 and 2.
   Adapted setting: this codebase's existing `ant.retrieval.dense.
   DenseEmbedder`, i.e. BAAI/bge-small-en-v1.5 served locally through
   fastembed/onnxruntime.
   Reason: fastembed -- the only local embedding runtime this project
   depends on -- has no Qwen3-Embedding-0.6B build. Its supported-model
   list was enumerated directly in this environment
   (`TextEmbedding.list_supported_models()`, 30 models) and contains no
   Qwen model of any size. Obtaining Qwen3-Embedding-0.6B would mean
   adding a second, parallel embedding runtime (sentence-transformers or
   a hand-built ONNX export) purely for this one baseline, which is
   exactly the "heavy extra engineering" the governing task scoped out.
   Reusing DenseEmbedder additionally keeps RepoDistill's retrieval on
   the same embedding substrate as this suite's frozen Dense Retrieval
   baseline, so a RepoDistill-vs-Dense comparison isolates the method
   (graph expansion + compression) rather than confounding it with an
   embedding-model swap. Cost profile is unchanged either way: both are
   local, free, and never touch a paid API.

2. TREE-SITTER GRAMMAR COVERAGE.
   Official setting: "Each source file f_i in F is parsed ... using
   tree-sitter", with no language restriction stated.
   Adapted setting: Python only (`tree_sitter_python`). Files that are
   not `.py`, or that fail to parse, contribute no graph nodes.
   Reason: both target benchmarks are Python-centric (RepoProbe-PYTHON
   is 8 Python repos by construction; SWE-QA-Pro's repo set is Python),
   and each additional grammar needs its own CONTAIN/INVOKE extraction
   rules -- node-type names and call syntax differ per language, so it
   is real per-language implementation work, not a grammar swap. Scoped
   and disclosed rather than half-done across many languages. Note this
   makes RepoDistill's file universe NARROWER than this suite's Sparse/
   Dense baselines (which index every text file via
   `EvalRepoEnvironment`); that is a property of the method as published
   (it is a code-dependency-graph method), not an unfair restriction
   introduced here.

3. TOP-K (the anchor-set size K in Equation 1).
   Official setting: the paper writes "Top-K" throughout but never
   states K numerically, in section 2.1, the Implementation paragraph of
   section 3, or the appendix (searched directly).
   Adapted setting: K = 8 (`ANCHOR_TOP_K`).
   Reason: 8 is this evaluation suite's own frozen retrieval width --
   Sparse Retrieval's `search(..., limit=8)` and Dense Retrieval's
   `TOP_K = 8` both use it. Picking the suite's existing value keeps
   RepoDistill's anchor budget comparable to the baselines it is being
   compared against, instead of inventing a third number.

4. MULTI-HOP PATH SEARCH DIRECTIONALITY AND DEPTH.
   Official setting: "We mine the shortest paths of anchor pairs v_i,
   v_j in N_TopK using INVOKE and CONTAIN edges, forming a path set P."
   Direction handling and any hop cap are unspecified.
   Adapted setting: shortest path over the UNDIRECTED view of the union
   of INVOKE and CONTAIN edges, with a hop cap (`MAX_PATH_HOPS = 6`).
   Reason: on the directed view, most anchor pairs in a real repo have
   no path at all (a caller-to-caller pair is only connected through a
   shared callee, i.e. an undirected connection), which would make the
   multi-hop half of Reasoning Chain Mining a no-op -- clearly not the
   intent of a step whose stated purpose is "to capture code logic and
   dependencies" between anchors. The hop cap bounds an otherwise
   unbounded all-pairs BFS over graphs with tens of thousands of nodes;
   paths longer than 6 hops are not plausibly a coherent "reasoning
   chain" anyway.

5. INVOKE RESOLUTION FOR AMBIGUOUS CALLEE NAMES.
   Official setting: "resolving function calls within CST subtrees,
   producing a candidate set of relations" -- the resolution rule for a
   bare name matching several definitions is not specified.
   Adapted setting: same-file definitions win outright; only if the name
   is undefined in the caller's own file do repo-wide matches apply, and
   those are capped at `MAX_GLOBAL_INVOKE_TARGETS = 3` (lowest node id
   first, so the choice is deterministic).
   Reason: a bare name like `run` or `get` can match hundreds of methods
   in a large repo; linking all of them would turn INVOKE into a
   near-complete bipartite blob and destroy the graph's signal. The
   same-file-first rule is the cheapest sound approximation of Python
   scoping without building a full import/type resolver (which the paper
   does not claim to do either -- "candidate set" is its own wording).

6. FIRST-ORDER SUBTREE WIDTH.
   Official setting: "For each anchor v in N_TopK, we retrieve ALL nodes
   directly connected via INVOKE edges" (emphasis added).
   Adapted setting: the `MAX_SUBTREE_NEIGHBOURS = 16` most
   query-similar such neighbours, not all of them.
   Reason: measured directly on a real benchmark repository
   (RepoProbe-Python's FieldStation42, 791 graph nodes), first-order
   INVOKE degree is 1 at the median and 16 at p99, but its maximum is
   68 -- and large repos are worse. An anchor that happens to be a
   utility hub would otherwise produce a single candidate containing
   dozens of mostly-irrelevant units, which (a) crowds every other
   candidate out of the downstream unit list, since `candidates_to_units`
   flattens in rank order, and (b) inflates the number of PAID CAPO turns
   for that one question. Ranking the neighbourhood by sim(q, v) and
   keeping the top 16 preserves the step's purpose -- pulling in the
   call-graph context around an anchor -- while making its cost
   independent of whether the anchor happens to be a popular helper. 16
   is 2x the anchor budget K, so a subtree can still be substantially
   wider than the anchor set itself. Disclosed rather than silently
   capped; the number is exposed as a module constant so it can be swept.

NO GOLD LEAKAGE. Nothing in this module ever reads
`TaskExample.reference`, `TaskExample.metadata['checklist']`, gold
relevant-file lists, or any other benchmark answer metadata. The graph is
a pure function of repository file CONTENT, and retrieval is a pure
function of (graph, question text). See
`tests/test_repodistill.py::test_no_gold_reference_or_checklist_reaches_the_pipeline`.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# --- Frozen hyperparameters (paper section 2.1; see deviation notes above
# for the two -- K and MAX_PATH_HOPS -- the paper leaves unspecified). ---

# Equation 2's lambda, stated explicitly in the paper: "where lambda is a
# hyperparameter, set to 0.5."
RERANK_LAMBDA = 0.5

# Equation 1's K. Paper says "Top-K" without a number -- see deviation 3.
ANCHOR_TOP_K = 8

# Hop cap for the multi-hop path mining step -- see deviation 4.
MAX_PATH_HOPS = 6

# Ambiguous-callee cap -- see deviation 5.
MAX_GLOBAL_INVOKE_TARGETS = 3

# First-order subtree width cap -- see deviation 6.
MAX_SUBTREE_NEIGHBOURS = 16

# How many re-ranked candidates are handed downstream to CAPO/CABA. The
# paper produces a "final ranked list" without stating a cut-off; this is
# the practical cut this implementation applies before the (paid) CAPO
# turns see anything, and it is the single biggest driver of per-question
# LLM cost -- see `repodistill.py`'s own cost documentation.
FINAL_CANDIDATE_LIMIT = 24

CONTAIN = "CONTAIN"
INVOKE = "INVOKE"


@dataclass(frozen=True)
class CodeUnit:
    """One node of the repository graph: a "code structural unit" in the
    paper's terms (a function or a class), carrying exactly the metadata
    section 2.1.1 says to record -- "file path, line number, name, and
    code snippet"."""

    node_id: str
    path: str
    kind: str  # "function" | "class"
    name: str
    qualname: str
    line_start: int
    line_end: int
    snippet: str

    def embedding_text(self) -> str:
        """What sim(q, v) is computed against for this node. Qualname is
        prepended to the body for the same reason `build_embedding_index`
        already does it for symbols: a short function body frequently
        carries none of the vocabulary its own NAME carries, and the name
        is usually the most query-alignable token a code unit has."""
        return f"{self.qualname}\n{self.snippet}"


@dataclass
class RepoGraph:
    """G = (V, E) with E subset of V x R x V, R = {CONTAIN, INVOKE}.

    Adjacency is stored as two separate id->ids maps rather than one edge
    list: every consumer below needs "the neighbours of v under relation
    r", never "all edges", and the two relations are traversed under
    different rules (INVOKE alone for first-order subtrees; INVOKE +
    CONTAIN together for multi-hop paths).
    """

    units: dict[str, CodeUnit] = field(default_factory=dict)
    contain: dict[str, list[str]] = field(default_factory=dict)
    invoke: dict[str, list[str]] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.units)

    @property
    def n_edges(self) -> int:
        return sum(len(v) for v in self.contain.values()) + sum(
            len(v) for v in self.invoke.values()
        )

    def invoked_by(self) -> dict[str, list[str]]:
        """Reverse INVOKE index (callee -> callers), built once and cached.

        Without this, `first_order_invoke` had to scan every INVOKE edge to
        find one node's callers, making any bulk traversal O(V*E) -- on
        adk-python (7.7k nodes, 21.8k edges) that is ~170M comparisons for
        a single full pass. The forward map is only ever mutated during
        `build_repository_graph`, which finishes before any retrieval
        happens, so caching here is safe; `save`/`load` deliberately do not
        persist it, since rebuilding costs one linear pass.
        """
        cached = getattr(self, "_invoked_by", None)
        if cached is None:
            cached = {}
            for caller, callees in self.invoke.items():
                for callee in callees:
                    cached.setdefault(callee, []).append(caller)
            object.__setattr__(self, "_invoked_by", cached)
        return cached

    def first_order_invoke(self, node_id: str) -> list[str]:
        """"For each anchor v in N_TopK, we retrieve all nodes directly
        connected via INVOKE edges." Both directions count as "directly
        connected": a function's callers are as much a part of its
        first-order neighbourhood as its callees, and the paper's phrase
        is connection, not out-degree."""
        return sorted(
            {*self.invoke.get(node_id, ()), *self.invoked_by().get(node_id, ())}
        )

    def undirected_adjacency(self) -> dict[str, set[str]]:
        adjacency: dict[str, set[str]] = {node_id: set() for node_id in self.units}
        for source, targets in (*self.contain.items(), *self.invoke.items()):
            if source not in adjacency:
                continue
            for target in targets:
                if target not in adjacency:
                    continue
                adjacency[source].add(target)
                adjacency[target].add(source)
        return adjacency

    def shortest_path(
        self,
        source: str,
        target: str,
        adjacency: dict[str, set[str]] | None = None,
        max_hops: int = MAX_PATH_HOPS,
    ) -> list[str]:
        """Breadth-first shortest path over the undirected INVOKE+CONTAIN
        view (deviation 4). Returns [] when no path within `max_hops`
        exists. Neighbours are visited in sorted order so the returned
        path is deterministic when several shortest paths tie."""
        if source == target or source not in self.units or target not in self.units:
            return []
        adjacency = adjacency if adjacency is not None else self.undirected_adjacency()
        previous: dict[str, str] = {source: ""}
        frontier: deque[tuple[str, int]] = deque([(source, 0)])
        while frontier:
            node_id, depth = frontier.popleft()
            if depth >= max_hops:
                continue
            for neighbour in sorted(adjacency.get(node_id, ())):
                if neighbour in previous:
                    continue
                previous[neighbour] = node_id
                if neighbour == target:
                    path = [target]
                    while previous[path[-1]]:
                        path.append(previous[path[-1]])
                    return list(reversed(path))
                frontier.append((neighbour, depth + 1))
        return []

    # --- persistence (one-time per repo checkout, reused across questions) ---

    def save(self, index_dir: Path, key: str = "graph") -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "units": [asdict(unit) for unit in self.units.values()],
            "contain": self.contain,
            "invoke": self.invoke,
        }
        tmp = index_dir / f"{key}.json.tmp"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(index_dir / f"{key}.json")

    @classmethod
    def load(cls, index_dir: Path, key: str = "graph") -> RepoGraph | None:
        path = index_dir / f"{key}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        units = {item["node_id"]: CodeUnit(**item) for item in payload["units"]}
        return cls(units=units, contain=payload["contain"], invoke=payload["invoke"])


# ---------------------------------------------------------------------------
# 2.1.1 Repository Graph Construction
# ---------------------------------------------------------------------------

# tree-sitter node types for the two "code structural unit" kinds the paper
# names (functions and classes). `decorated_definition` wraps either of
# them in the Python grammar and is traversed through, not treated as a
# unit itself.
_FUNCTION_NODES = {"function_definition"}
_CLASS_NODES = {"class_definition"}
_UNIT_NODES = _FUNCTION_NODES | _CLASS_NODES


def _python_parser():
    """Built per call rather than cached in a module global: tree-sitter
    `Parser` objects are not documented as thread-safe, and graph builds
    can legitimately run concurrently (this suite already runs agents
    across repos in parallel). Parser construction is microseconds; the
    Language object underneath is the expensive part and IS cached by
    tree_sitter_python itself."""
    from tree_sitter import Language, Parser
    from tree_sitter_python import language

    return Parser(Language(language()))


def _node_name(node) -> str:
    child = node.child_by_field_name("name")
    if child is None:
        return ""
    return child.text.decode("utf-8", errors="replace")


def _callee_name(call_node) -> str:
    """The called symbol's own name, from a `call` CST node.

    `foo(...)`        -> function field is an `identifier`  -> "foo"
    `obj.foo(...)`    -> function field is an `attribute`   -> "foo"
    `a.b.foo(...)`    -> attribute's own `attribute` field  -> "foo"

    Only the final name segment is kept, because that is the only part
    resolvable against a definition's own `name` without a type resolver
    -- see deviation 5.
    """
    function_node = call_node.child_by_field_name("function")
    if function_node is None:
        return ""
    if function_node.type == "identifier":
        return function_node.text.decode("utf-8", errors="replace")
    if function_node.type == "attribute":
        attribute = function_node.child_by_field_name("attribute")
        if attribute is not None:
            return attribute.text.decode("utf-8", errors="replace")
    return ""


def _parse_file_units(
    relative_path: str, source: str
) -> tuple[list[CodeUnit], dict[str, list[str]], dict[str, str]]:
    """Parse one file into (units, calls_made_by_unit, parent_of_unit).

    Implements section 2.1.1's "Candidate units are extracted via CST
    traversal, and we record metadata including file path, line number,
    name, and code snippet", plus the per-unit call collection that
    INVOKE extraction consumes.

    Calls are attributed to the INNERMOST enclosing unit only: a call
    inside a nested function belongs to that nested function, not also to
    its parent. The paper's phrase is "resolving function calls within
    CST subtrees" applied to a caller UNIT, and double-attributing every
    nested call up the whole enclosing chain would make an outer class
    appear to invoke everything any of its methods ever touches.
    """
    try:
        tree = _python_parser().parse(source.encode("utf-8"))
    except Exception:  # noqa: BLE001 -- a single unparseable file must never
        # abort a whole repo's graph build; it simply contributes no nodes.
        return [], {}, {}

    lines = source.splitlines()
    units: list[CodeUnit] = []
    calls: dict[str, list[str]] = {}
    parents: dict[str, str] = {}

    def walk(node, unit_stack: list[CodeUnit]) -> None:
        current = unit_stack[-1] if unit_stack else None
        if node.type == "call" and current is not None:
            name = _callee_name(node)
            if name:
                calls.setdefault(current.node_id, []).append(name)
        if node.type in _UNIT_NODES:
            name = _node_name(node)
            if name:
                line_start = node.start_point[0] + 1
                line_end = node.end_point[0] + 1
                qualname = ".".join([*(u.name for u in unit_stack), name])
                unit = CodeUnit(
                    node_id=f"{relative_path}::{qualname}::{line_start}",
                    path=relative_path,
                    kind="class" if node.type in _CLASS_NODES else "function",
                    name=name,
                    qualname=qualname,
                    line_start=line_start,
                    line_end=line_end,
                    snippet="\n".join(lines[line_start - 1 : line_end]),
                )
                units.append(unit)
                if current is not None:
                    parents[unit.node_id] = current.node_id
                for child in node.children:
                    walk(child, [*unit_stack, unit])
                return
        for child in node.children:
            walk(child, unit_stack)

    walk(tree.root_node, [])
    return units, calls, parents


def build_repository_graph(root: Path, relative_paths: list[str]) -> RepoGraph:
    """Section 2.1.1's three sequential steps: Code Parsing, Dependency
    Relation Extraction, Graph Building.

    ONE-TIME PER REPO CHECKOUT, not per question -- every question asked
    against the same pinned checkout reuses the identical graph (see
    `repodistill.py`'s disk cache). This mirrors how this suite's
    large-repo Dense Retrieval baseline already separates one-time
    per-repo index construction from per-question retrieval.
    """
    graph = RepoGraph()
    calls_by_unit: dict[str, list[str]] = {}
    units_by_name: dict[str, list[str]] = {}
    units_by_file_and_name: dict[tuple[str, str], list[str]] = {}

    # --- Code Parsing ---
    for relative in sorted(relative_paths):
        if not relative.endswith(".py"):
            # Deviation 2: Python grammar only.
            continue
        try:
            source = (root / relative).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        units, calls, parents = _parse_file_units(relative, source)
        for unit in units:
            graph.units[unit.node_id] = unit
            units_by_name.setdefault(unit.name, []).append(unit.node_id)
            units_by_file_and_name.setdefault((unit.path, unit.name), []).append(unit.node_id)
        calls_by_unit.update(calls)
        # --- Dependency Relation Extraction, part 1: CONTAIN edges.
        # "CONTAIN edges link classes to their contained functions or
        # inner classes" -- so only a CLASS parent produces one; a
        # function nesting a closure is lexical nesting, not the
        # hierarchical containment the paper describes.
        for child_id, parent_id in parents.items():
            if graph.units[parent_id].kind == "class":
                graph.contain.setdefault(parent_id, []).append(child_id)

    # --- Dependency Relation Extraction, part 2: INVOKE edges ---
    for caller_id, callee_names in calls_by_unit.items():
        caller_path = graph.units[caller_id].path
        targets: list[str] = []
        for callee_name in callee_names:
            local = units_by_file_and_name.get((caller_path, callee_name))
            if local:
                targets.extend(local)
                continue
            # Deviation 5: repo-wide fallback, deterministically capped.
            globals_ = sorted(units_by_name.get(callee_name, ()))
            targets.extend(globals_[:MAX_GLOBAL_INVOKE_TARGETS])
        targets = sorted({t for t in targets if t != caller_id})
        if targets:
            graph.invoke[caller_id] = targets

    # --- Graph Building --- (nodes/edges already instantiated in place)
    for key in list(graph.contain):
        graph.contain[key] = sorted(set(graph.contain[key]))
    return graph


# ---------------------------------------------------------------------------
# 2.1.2 Repository Graph Retrieval
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """One element of the candidate set C: "the anchors, their first-order
    subtrees, and the mined multi-hop paths".

    A candidate is a SET of nodes, not a single node -- which is what
    makes Equation 2's `max_{v in c} sim(q, v)` term meaningful (a max
    over one element would be identical to sim(q, c) and the whole
    re-ranking formula would collapse to sim(q, c)).
    """

    kind: str  # "anchor" | "subtree" | "path"
    node_ids: list[str]
    score: float = 0.0
    similarity: float = 0.0
    max_member_similarity: float = 0.0

    @property
    def primary_node_id(self) -> str:
        return self.node_ids[0]


def _cosine(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return matrix @ vector


def retrieve_candidates(
    graph: RepoGraph,
    question: str,
    node_ids: list[str],
    node_vectors: np.ndarray,
    query_vector: np.ndarray,
    *,
    top_k: int = ANCHOR_TOP_K,
    rerank_lambda: float = RERANK_LAMBDA,
    limit: int = FINAL_CANDIDATE_LIMIT,
) -> tuple[list[Candidate], list[str]]:
    """Section 2.1.2 end to end. Returns (ranked candidates, anchor ids).

    `node_vectors` must be L2-normalized rows aligned with `node_ids`
    (built once per repo, see `repodistill.py`); `query_vector` is the
    embedding of `question` alone. The question text is the ONLY
    task-derived input -- no reference answer, no checklist, no gold file
    list is reachable from this signature.
    """
    if not node_ids or node_vectors.size == 0:
        return [], []

    similarities = _cosine(node_vectors, query_vector)
    similarity_by_id = {node_id: float(similarities[i]) for i, node_id in enumerate(node_ids)}

    # --- Node Localization (Equation 1) ---
    # arg max over subsets of size K of the SUM of sim(q, v) is, for a
    # fixed cardinality with no interaction term, exactly the K
    # individually-highest-similarity nodes -- so this is the equation,
    # not an approximation of it.
    order = np.argsort(similarities)[::-1][:top_k]
    anchors = [node_ids[int(i)] for i in order]

    # --- Reasoning Chain Mining ---
    candidates: list[Candidate] = []
    seen: set[tuple[str, ...]] = set()

    def add(kind: str, members: list[str]) -> None:
        members = [m for m in members if m in similarity_by_id]
        if not members:
            return
        key = tuple(members)
        if key in seen:
            return
        seen.add(key)
        candidates.append(Candidate(kind=kind, node_ids=members))

    for anchor in anchors:
        add("anchor", [anchor])

    # (a) First-Order Subtrees: "For each anchor v in N_TopK, we retrieve
    # all nodes directly connected via INVOKE edges." Width-capped by
    # query similarity -- see deviation 6.
    for anchor in anchors:
        neighbours = [n for n in graph.first_order_invoke(anchor) if n in similarity_by_id]
        if not neighbours:
            continue
        neighbours.sort(key=lambda n: (-similarity_by_id[n], n))
        add("subtree", [anchor, *neighbours[:MAX_SUBTREE_NEIGHBOURS]])

    # (b) Multi-Hop Paths: "We mine the shortest paths of anchor pairs
    # v_i, v_j in N_TopK using INVOKE and CONTAIN edges, forming a path
    # set P."
    adjacency = graph.undirected_adjacency()
    for i, source in enumerate(anchors):
        for target in anchors[i + 1 :]:
            path = graph.shortest_path(source, target, adjacency=adjacency)
            if len(path) > 1:
                add("path", path)

    # --- Re-Ranking (Equation 2) ---
    # sim(q, c) for a multi-node candidate is the mean of its members'
    # vectors re-normalized and dotted with the query -- i.e. the
    # similarity of the candidate AS A WHOLE, which is the only reading
    # under which Equation 2's two terms differ from each other. For a
    # single-node anchor candidate this reduces exactly to sim(q, v).
    index_by_id = {node_id: i for i, node_id in enumerate(node_ids)}
    for candidate in candidates:
        rows = node_vectors[[index_by_id[n] for n in candidate.node_ids]]
        centroid = rows.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        candidate.similarity = float(centroid @ query_vector / norm) if norm > 0 else 0.0
        candidate.max_member_similarity = max(similarity_by_id[n] for n in candidate.node_ids)
        candidate.score = (
            rerank_lambda * candidate.similarity
            + (1.0 - rerank_lambda) * candidate.max_member_similarity
        )

    # Ties broken by node id so a re-run over the same checkout produces a
    # byte-identical ranked list (this suite requires deterministic
    # retrieval -- see the Dense Retrieval baseline's own determinism test).
    candidates.sort(key=lambda c: (-c.score, c.primary_node_id))
    return candidates[:limit], anchors


def candidates_to_units(graph: RepoGraph, candidates: list[Candidate]) -> list[CodeUnit]:
    """Flatten the ranked candidate list into the deduplicated, rank-ordered
    unit list CAPO's turns and CABA's compression actually operate on.

    Order is preserved from the candidate ranking (a unit first seen in a
    higher-scoring candidate keeps that position), because CAPO chunks
    this list in order and the paper's whole point is that the model sees
    the most relevant context first.
    """
    out: list[CodeUnit] = []
    seen: set[str] = set()
    for candidate in candidates:
        for node_id in candidate.node_ids:
            if node_id in seen or node_id not in graph.units:
                continue
            seen.add(node_id)
            out.append(graph.units[node_id])
    return out
