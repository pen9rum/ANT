from pathlib import Path
from typing import cast

from ant.domain import EvidenceState
from ant.memory.global_memory import (
    GlobalMemoryStore,
    TaskExperience,
    record_global_experience,
    record_global_experience_safe,
    retrieve_cross_repo_experience_safe,
)
from ant.providers import WorkerReasoner
from ant.retrieval.dense import DenseEmbedder


class _FakeEmbedder:
    """Hand-written, fixed vectors keyed by exact text -- no real model
    load, same "fake embedder, real cosine math" pattern used for the rest
    of this codebase's dense-retrieval tests.
    """

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


def _embedder(vectors: dict[str, list[float]]) -> DenseEmbedder:
    return cast(DenseEmbedder, _FakeEmbedder(vectors))


def test_retrieve_similar_returns_empty_list_when_store_is_empty(tmp_path: Path) -> None:
    store = GlobalMemoryStore(tmp_path)
    embedder = _embedder({"query": [1.0, 0.0]})

    assert store.retrieve_similar("query", embedder) == []


def test_record_experience_skips_an_empty_summary(tmp_path: Path) -> None:
    store = GlobalMemoryStore(tmp_path)
    embedder = _embedder({})  # would KeyError if record_experience tried to embed anything

    store.record_experience(TaskExperience(summary="   ", repo="repo-a"), embedder)

    assert store.retrieve_similar("anything", _embedder({"anything": [1.0, 0.0]})) == []


def test_retrieve_similar_ranks_by_cosine_similarity_not_recency(tmp_path: Path) -> None:
    store = GlobalMemoryStore(tmp_path)
    embedder = _embedder(
        {
            "close match": [1.0, 0.0],
            "far match": [0.0, 1.0],
            "query": [0.9, 0.1],
        }
    )
    # Recorded in an order that would be wrong if retrieval just returned
    # insertion order instead of actually ranking by similarity.
    store.record_experience(TaskExperience(summary="far match", repo="repo-a"), embedder)
    store.record_experience(TaskExperience(summary="close match", repo="repo-b"), embedder)

    results = store.retrieve_similar("query", embedder, limit=5)

    assert results[0] == "close match"
    assert results[1] == "far match"


def test_retrieval_never_filters_or_privileges_by_repo(tmp_path: Path) -> None:
    # The entire point of cross-repo memory: a pattern learned on repo-a
    # must surface for a query from repo-b just as readily as one recorded
    # under the same repo would -- retrieve_similar takes no repo argument
    # at all, provenance (`repo`) is metadata only.
    store = GlobalMemoryStore(tmp_path)
    embedder = _embedder({"a pattern": [1.0, 0.0], "query": [1.0, 0.0]})
    store.record_experience(
        TaskExperience(summary="a pattern", repo="totally-unrelated-repo"), embedder
    )

    assert store.retrieve_similar("query", embedder) == ["a pattern"]


class _StubReasoner:
    def __init__(self, summary: str) -> None:
        self._summary = summary
        self.received_kwargs: dict | None = None

    def summarize_task_experience(self, *, question, rounds, unresolved_needs, evidence_count):
        self.received_kwargs = {
            "question": question,
            "rounds": rounds,
            "unresolved_needs": unresolved_needs,
            "evidence_count": evidence_count,
        }
        return self._summary


def test_record_global_experience_skips_when_reasoner_judges_nothing_worth_remembering(
    tmp_path: Path,
) -> None:
    store = GlobalMemoryStore(tmp_path)
    embedder = _embedder({})  # would KeyError if a summary were (wrongly) embedded
    reasoner = _StubReasoner(summary="")
    state = EvidenceState(question="q")

    record_global_experience(
        store, cast(WorkerReasoner, reasoner), embedder, "q", state, repo="repo-a"
    )

    assert reasoner.received_kwargs is not None  # the reasoner WAS asked
    assert store.retrieve_similar("q", _embedder({"q": [1.0, 0.0]})) == []


def test_record_global_experience_records_the_reasoners_summary(tmp_path: Path) -> None:
    store = GlobalMemoryStore(tmp_path)
    embedder = _embedder(
        {"a transferable pattern": [1.0, 0.0], "similar query": [0.9, 0.1]}
    )
    reasoner = _StubReasoner(summary="a transferable pattern")
    state = EvidenceState(question="q")

    record_global_experience(
        store, cast(WorkerReasoner, reasoner), embedder, "q", state, repo="repo-a"
    )

    assert store.retrieve_similar("similar query", embedder) == ["a transferable pattern"]


def test_retrieve_cross_repo_experience_safe_degrades_to_empty_without_an_embedder(
    tmp_path: Path, monkeypatch
) -> None:
    import ant.memory.global_memory as global_memory_module

    monkeypatch.setattr(global_memory_module, "get_shared_embedder", lambda: None)
    store = GlobalMemoryStore(tmp_path)

    assert retrieve_cross_repo_experience_safe(store, "any query") == []


def test_record_global_experience_safe_degrades_to_a_no_op_without_an_embedder(
    tmp_path: Path, monkeypatch
) -> None:
    import ant.memory.global_memory as global_memory_module

    monkeypatch.setattr(global_memory_module, "get_shared_embedder", lambda: None)
    store = GlobalMemoryStore(tmp_path)
    reasoner = _StubReasoner(summary="should never be recorded")
    state = EvidenceState(question="q")

    record_global_experience_safe(store, cast(WorkerReasoner, reasoner), "q", state, repo="repo-a")

    # get_shared_embedder() returning None means the reasoner is never even
    # consulted -- no wasted LLM call when dense retrieval isn't available.
    assert reasoner.received_kwargs is None
