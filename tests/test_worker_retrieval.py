from ant.coordinator.worker_retrieval import build_worker_index, rank_workers
from ant.domain import CodeSymbol, WorkerCard


def _worker(
    worker_id: str,
    *,
    files: list[str],
    symbol_names: list[str],
    responsibilities: list[str] | None = None,
    searchable_terms: list[str] | None = None,
) -> WorkerCard:
    return WorkerCard(
        id=worker_id,
        territory_id=worker_id.removeprefix("worker-"),
        name=worker_id,
        root=files[0].rsplit("/", 1)[0] if "/" in files[0] else "",
        files=files,
        responsibilities=responsibilities or [],
        searchable_terms=searchable_terms or [],
        symbols=[
            CodeSymbol(name=name, kind="class", path=files[0], line=index, qualname=name)
            for index, name in enumerate(symbol_names, start=1)
        ],
    )


def test_rank_workers_finds_a_symbol_absent_from_searchable_terms() -> None:
    # Regression test for the real qibo case: `searchable_terms` is built by
    # indexing.cards._top_terms's round-robin sampling, capped at a fixed
    # length -- confirmed directly that a worker's own defining class
    # (FALQON) sat past the slice the Orchestrator's prompt actually shows,
    # even though it is present, in full, on WorkerCard.symbols. This
    # module must find it through `symbols`, not `searchable_terms`.
    #
    # File names/responsibilities are deliberately uninformative (no "qaoa"/
    # "falqon" substring anywhere but the symbol name itself) so this test
    # actually isolates the symbols-vs-searchable_terms question -- an
    # earlier version of this test let the file stem "variational.py" leak
    # the answer even when the corpus was built from searchable_terms
    # alone, silently passing for the wrong reason.
    target = _worker(
        "worker-src-qibo-models",
        files=["src/qibo/models/impl.py"],
        symbol_names=["VQE", "QAOA", "FALQON"],
        searchable_terms=["StateEvolution", "Grover", "qPDF", "StyleQGAN", "TSP"],
    )
    sibling = _worker(
        "worker-src-qibo-gates",
        files=["src/qibo/gates/impl2.py"],
        symbol_names=["Gate", "Hamiltonian"],
        searchable_terms=["Gate", "Hamiltonian", "apply", "matrix"],
    )
    index = build_worker_index([target, sibling])

    ranks = rank_workers("subclasses of FALQON overriding minimize", [target, sibling], index)

    assert ranks
    assert min(ranks, key=lambda worker_id: ranks[worker_id]) == "worker-src-qibo-models"


def test_rank_workers_weights_a_shared_path_component_below_a_rare_symbol() -> None:
    # Same IDF-weighting regression covered for the file-level symbol/path
    # channel tonight (ant.tools.local._symbol_path_channel_rank), applied
    # here at worker granularity: every worker lives under the same parent
    # directory ("extractor/"), so that path component matches every
    # worker equally, and the 9 siblings additionally share a
    # "handle_error" symbol that matches 2 more query terms each -- under
    # flat per-term counting the 9 siblings (3 matched terms: extractor +
    # handle + error) would each outrank the target (2 matched terms:
    # extractor + teachable), even though only the target is actually
    # relevant to "teachable". IDF-style weighting (1/matches per term)
    # must let the target's one truly rare term win instead.
    target = _worker(
        "worker-teachable",
        files=["extractor/teachable.py"],
        symbol_names=["TeachableIE"],
    )
    siblings = [
        _worker(
            f"worker-sibling-{i}",
            files=[f"extractor/sibling{i}.py"],
            symbol_names=["BaseIE", "handle_error"],
        )
        for i in range(9)
    ]
    workers = [target, *siblings]
    index = build_worker_index(workers)

    ranks = rank_workers("teachable extractor handle error", workers, index)

    assert ranks
    assert min(ranks, key=lambda worker_id: ranks[worker_id]) == "worker-teachable"


def test_rank_workers_returns_empty_for_a_query_with_no_content_terms() -> None:
    worker = _worker("worker-a", files=["a.py"], symbol_names=["Foo"])
    index = build_worker_index([worker])

    assert rank_workers("the and of", [worker], index) == {}


def test_rank_workers_is_stable_across_repeated_calls_with_a_cached_index() -> None:
    # build_worker_index is meant to be built once per LocalCoordinator.ask()
    # call and reused every round -- confirm the same WorkerIndex object
    # produces identical rankings across repeated rank_workers() calls
    # (nothing mutates it).
    target = _worker("worker-x", files=["x.py"], symbol_names=["Widget"])
    other = _worker("worker-y", files=["y.py"], symbol_names=["Gadget"])
    index = build_worker_index([target, other])

    first = rank_workers("Widget lookup", [target, other], index)
    second = rank_workers("Widget lookup", [target, other], index)

    assert first == second
