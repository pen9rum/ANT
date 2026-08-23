import time
from pathlib import Path

import numpy as np
import pytest

from ant.domain import Evidence, WorkerCard
from ant.retrieval import dense as dense_module
from ant.retrieval.dense import (
    DenseEmbedder,
    EmbeddingEntry,
    EmbeddingIndex,
    build_and_cache_in_background,
    build_embedding_index,
)
from ant.retrieval.relevance import extract_terms, score_evidence
from ant.tools import LocalSearchTool
from ant.tools import local as local_module
from ant.workers import AutonomousWorker, WorkerRunConfig
from ant.workers.autonomous import _dedupe


class _FakeEmbedder(DenseEmbedder):
    """Deterministic, network-free stand-in for DenseEmbedder: one fixed
    2-D vector per call, enough to exercise the batching/save/load pipeline
    without loading a real model. Subclasses DenseEmbedder (rather than just
    duck-typing it) so it satisfies the concrete DenseEmbedder type hints
    used throughout dense.py, and skips the real __init__ (which would try
    to import fastembed and load a model) entirely.
    """

    def __init__(self) -> None:
        pass  # deliberately not calling super().__init__()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def _fake_index() -> EmbeddingIndex:
    # Hand-written vectors, no real model load: keeps this test fast and
    # network-free like the rest of the suite. Two orthogonal directions
    # stand in for "about seeds" vs. "about something else".
    entries = [
        EmbeddingEntry(
            path="src/a.py", line_start=1, line_end=3, quote="def set_seed(self, seed):"
        ),
        EmbeddingEntry(path="src/b.py", line_start=1, line_end=3, quote="def unrelated_helper():"),
    ]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    return EmbeddingIndex(entries=entries, vectors=vectors)


def test_embedding_index_search_ranks_by_cosine_similarity() -> None:
    index = _fake_index()
    hits = index.search([0.9, 0.1], limit=2)
    assert hits[0][1].path == "src/a.py"
    assert hits[0][0] > hits[1][0]


def test_embedding_index_search_filters_to_the_given_paths() -> None:
    index = _fake_index()
    hits = index.search([1.0, 0.0], limit=2, paths={"src/b.py"})
    assert [entry.path for _, entry in hits] == ["src/b.py"]


def test_embedding_index_save_and_load_round_trips(tmp_path: Path) -> None:
    index = _fake_index()
    index.save(tmp_path, "some-territory")

    loaded = EmbeddingIndex.load(tmp_path, "some-territory")

    assert loaded is not None
    assert [entry.path for entry in loaded.entries] == ["src/a.py", "src/b.py"]
    assert loaded.vectors.shape == (2, 2)


def test_embedding_index_load_returns_none_when_absent(tmp_path: Path) -> None:
    assert EmbeddingIndex.load(tmp_path, "some-territory") is None


def test_embedding_index_keys_are_isolated(tmp_path: Path) -> None:
    # Two different territories saved under the same directory must not
    # collide -- each worker's lazily-built cache is independent.
    _fake_index().save(tmp_path, "territory-a")

    assert EmbeddingIndex.load(tmp_path, "territory-b") is None
    assert EmbeddingIndex.load(tmp_path, "territory-a") is not None


def test_score_evidence_lets_a_paraphrase_only_match_compete() -> None:
    # A candidate sharing zero lexical terms with the query is invisible to
    # every bonus except dense_score -- that's the entire point of dense
    # retrieval feeding into the same reranker instead of a separate path.
    terms = extract_terms("reproducible measurement sampling across backends")
    lexical_only = score_evidence(
        quote="def unrelated_helper(): pass",
        path="src/b.py",
        terms=terms,
    )
    dense_paraphrase = score_evidence(
        quote="def set_seed(self, seed): self.np.random.seed(seed)",
        path="src/a.py",
        terms=terms,
        dense_score=0.8,
    )
    assert dense_paraphrase > lexical_only


def test_dedupe_merges_dense_score_onto_the_kept_duplicate() -> None:
    # Regression test: search() and dense_search() can both surface the same
    # (path, line_start, line_end) chunk in one round -- a paraphrase that
    # also happens to share a lexical term. Keeping only the first-seen copy
    # (the lexical one, since search() runs first) used to silently drop
    # that item's dense_score, undercutting the fused reranker's signal for
    # exactly the items both channels agree on.
    lexical = Evidence(
        path="src/a.py",
        line_start=2,
        line_end=6,
        quote="def set_seed(self, seed):",
        reason="Scored local retrieval match (19) around: def set_seed",
    )
    dense = Evidence(
        path="src/a.py",
        line_start=2,
        line_end=6,
        quote="def set_seed(self, seed):",
        reason="Dense semantic match (score=0.72) for: ...",
        dense_score=0.72,
    )

    deduped = _dedupe([lexical, dense])

    assert len(deduped) == 1
    assert deduped[0].dense_score == 0.72
    assert deduped[0].reason == lexical.reason


def test_dense_search_returns_empty_without_a_built_index(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def f():\n    pass\n", encoding="utf-8")

    tool = LocalSearchTool(tmp_path)  # no index_path -> no embedding index on disk
    assert tool.dense_search("anything", ["src/a.py"]) == []


def test_autonomous_worker_run_does_not_require_an_embedding_index(tmp_path: Path) -> None:
    # dense_search runs unconditionally every round (see AutonomousWorker.run);
    # it must degrade to a no-op, not an error, whenever no dense index has
    # been built for this repo (the default/common case).
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text(
        "def authenticate_user():\n    return True\n", encoding="utf-8"
    )
    card = WorkerCard(
        id="worker-src",
        territory_id="src",
        name="src worker",
        root="src",
        searchable_terms=["authenticate"],
        files=["src/a.py"],
    )

    observation = AutonomousWorker(tmp_path, card, LocalSearchTool(tmp_path)).run(
        "Where is authenticate handled?", WorkerRunConfig(max_tool_calls=6, evidence_limit=6)
    )

    assert any(action.tool == "dense_search" for action in observation.actions)


def test_build_embedding_index_is_one_entry_per_symbol_not_per_paragraph(
    tmp_path: Path,
) -> None:
    # Regression test: embedding every paragraph/definition-block region (the
    # same granularity BM25 uses) scales with file length, not with how much
    # is actually there to point to -- a handful of large, dense files could
    # still produce thousands of chunks even though the worker's own file
    # list is short. Symbol count is a tighter bound.
    (tmp_path / "src").mkdir()
    filler = "\n\n".join(f"# comment paragraph {i}\nx = {i}" for i in range(30))
    (tmp_path / "src" / "big.py").write_text(
        f"{filler}\n\n\ndef only_symbol():\n    return 1\n", encoding="utf-8"
    )

    index = build_embedding_index(tmp_path, ["src/big.py"], _FakeEmbedder())

    assert len(index.entries) == 1
    assert "def only_symbol" in index.entries[0].quote


def test_build_and_cache_in_background_does_not_block_and_saves_eventually(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    dense_dir = tmp_path / "dense"

    started = time.time()
    build_and_cache_in_background(tmp_path, ["src/a.py"], dense_dir, "key-a", _FakeEmbedder())
    elapsed = time.time() - started

    assert elapsed < 1.0  # returns immediately, does not wait for the build

    for _ in range(50):
        if EmbeddingIndex.load(dense_dir, "key-a") is not None:
            break
        time.sleep(0.05)
    loaded = EmbeddingIndex.load(dense_dir, "key-a")
    assert loaded is not None
    assert len(loaded.entries) == 1


def test_build_and_cache_in_background_does_not_duplicate_an_inflight_build(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    dense_dir = tmp_path / "dense"
    calls = []

    class _CountingEmbedder(_FakeEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            calls.append(texts)
            return super().embed(texts)

    embedder = _CountingEmbedder()
    build_and_cache_in_background(tmp_path, ["src/a.py"], dense_dir, "key-b", embedder)
    build_and_cache_in_background(tmp_path, ["src/a.py"], dense_dir, "key-b", embedder)

    for _ in range(50):
        if EmbeddingIndex.load(dense_dir, "key-b") is not None:
            break
        time.sleep(0.05)

    assert len(calls) == 1


def test_dense_search_returns_empty_immediately_for_an_uncached_territory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The whole point of the background-build design: a territory nobody has
    # queried before must never make *this* query wait on the build.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    index_path = tmp_path / ".ant"
    index_path.mkdir()

    monkeypatch.setattr(local_module, "get_shared_embedder", lambda: _FakeEmbedder())

    tool = LocalSearchTool(tmp_path, index_path=index_path)
    assert tool.dense_search("anything", ["src/a.py"]) == []

    for _ in range(50):
        if (index_path / "dense").exists():
            break
        time.sleep(0.05)
    key = dense_module.territory_key(["src/a.py"])
    assert EmbeddingIndex.load(index_path / "dense", key) is not None
