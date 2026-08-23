from pathlib import Path

import numpy as np

from ant.domain import Evidence, WorkerCard
from ant.retrieval.dense import EmbeddingEntry, EmbeddingIndex
from ant.retrieval.relevance import extract_terms, score_evidence
from ant.tools import LocalSearchTool
from ant.workers import AutonomousWorker, WorkerRunConfig
from ant.workers.autonomous import _dedupe


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
