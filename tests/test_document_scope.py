"""Tests for the shared long-document/multi-document substrate
(ant.evaluation_suite.document_scope): materialization, the
EvalDocumentEnvironment RepoEnvironment-compatible view, deterministic
boundary-respecting chunking, and the Lost-in-the-Middle positional-
perturbation utility (implemented but not run as a formal evaluation this
pass -- see docs/long_context_dataset_audit.md and Section 15 of the
long-context evaluation spec)."""
from __future__ import annotations

from pathlib import Path

import pytest

from ant.evaluation_suite.document_scope import (
    DocumentRecord,
    EvalDocumentEnvironment,
    build_positional_variants,
    chunk_documents,
    materialize_documents,
)


def _docs() -> list[DocumentRecord]:
    return [
        DocumentRecord(doc_id="doc0", title="Alpha", text="alpha body text"),
        DocumentRecord(doc_id="doc1", title="Beta", text="beta body text"),
        DocumentRecord(doc_id="doc2", title="", text="gamma body text, no title"),
    ]


def test_materialize_documents_writes_in_original_order_with_positional_filenames(
    tmp_path: Path,
) -> None:
    paths = materialize_documents(_docs(), tmp_path)

    assert [p.name for p in paths.values()] == ["doc_0000.txt", "doc_0001.txt", "doc_0002.txt"]
    content = (tmp_path / "doc_0000.txt").read_text(encoding="utf-8")
    assert content == "Title: Alpha\n\nalpha body text"
    # Empty title: no "Title: " line prefixed.
    assert (tmp_path / "doc_0002.txt").read_text(encoding="utf-8") == "gamma body text, no title"


def test_materialize_documents_filename_never_encodes_title_or_relevance(tmp_path: Path) -> None:
    # Purely positional filenames -- a method that only sees file NAMES
    # (not yet opened content) gets zero signal about which document is
    # which topic or whether it's gold-relevant.
    docs = [
        DocumentRecord(doc_id="doc0", title="VERY RELEVANT SECRET ANSWER", text="x"),
        DocumentRecord(doc_id="doc1", title="irrelevant", text="y"),
    ]
    paths = materialize_documents(docs, tmp_path)
    for path in paths.values():
        assert "RELEVANT" not in path.name
        assert "SECRET" not in path.name


def test_eval_document_environment_round_trips_doc_id_and_relative_path(tmp_path: Path) -> None:
    docs = _docs()
    materialize_documents(docs, tmp_path)
    environment = EvalDocumentEnvironment(tmp_path, docs)

    assert environment.relative_path_for("doc1") == "doc_0001.txt"
    assert environment.doc_id_for_relative_path("doc_0001.txt") == "doc1"
    assert environment.doc_id_for_relative_path("no-such-file.txt") is None
    doc1 = environment.document("doc1")
    assert doc1 is not None
    assert doc1.title == "Beta"
    assert environment.document("doc-missing") is None
    assert [d.doc_id for d in environment.ordered_documents()] == ["doc0", "doc1", "doc2"]
    expected_names = ["doc_0000.txt", "doc_0001.txt", "doc_0002.txt"]
    assert [p.name for p in environment.iter_files()] == expected_names


def test_chunk_documents_never_crosses_a_document_boundary() -> None:
    docs = [
        DocumentRecord(doc_id="doc0", title="", text="word " * 50),
        DocumentRecord(doc_id="doc1", title="", text="word " * 50),
    ]
    chunks = chunk_documents(docs, chunk_size_tokens=10)

    assert len(chunks) > 2  # each document needs multiple chunks at this size
    # Every chunk belongs to exactly one document_id; no chunk's text mixes
    # both documents' content (verified structurally: chunk_index resets to
    # 0 at each new document_id, and global_chunk_index is monotonic).
    seen_docs: list[str] = []
    for chunk in chunks:
        if not seen_docs or seen_docs[-1] != chunk.document_id:
            assert chunk.chunk_index == 0
            seen_docs.append(chunk.document_id)
    assert seen_docs == ["doc0", "doc1"]


def test_chunk_documents_global_chunk_index_is_monotonic_and_contiguous() -> None:
    docs = [
        DocumentRecord(doc_id="doc0", title="", text="word " * 20),
        DocumentRecord(doc_id="doc1", title="", text="word " * 20),
    ]
    chunks = chunk_documents(docs, chunk_size_tokens=5)
    assert [c.global_chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_documents_empty_document_yields_one_empty_chunk() -> None:
    docs = [DocumentRecord(doc_id="doc0", title="", text="")]
    chunks = chunk_documents(docs, chunk_size_tokens=10)
    assert len(chunks) == 1
    assert chunks[0].text == ""
    assert chunks[0].token_start == 0 and chunks[0].token_end == 0


def test_chunk_documents_is_deterministic_across_repeated_calls() -> None:
    docs = _docs()
    first = chunk_documents(docs, chunk_size_tokens=4)
    second = chunk_documents(docs, chunk_size_tokens=4)
    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]


def test_build_positional_variants_preserves_document_set_text_and_relative_order() -> None:
    docs = [DocumentRecord(doc_id=f"doc{i}", title=f"T{i}", text=f"body {i}") for i in range(6)]
    supporting = ["doc2", "doc4"]

    variants = build_positional_variants(docs, supporting)
    positions = {v.position for v in variants}
    assert positions == {"early", "middle", "late"}

    for variant in variants:
        assert len(variant.documents) == len(docs)
        assert {d.doc_id for d in variant.documents} == {d.doc_id for d in docs}
        for doc in variant.documents:
            original = next(d for d in docs if d.doc_id == doc.doc_id)
            assert doc.text == original.text
            assert doc.title == original.title
        distractor_order = [d.doc_id for d in variant.documents if d.doc_id not in supporting]
        assert distractor_order == ["doc0", "doc1", "doc3", "doc5"]

    early = next(v for v in variants if v.position == "early")
    assert [d.doc_id for d in early.documents[:2]] == supporting
    late = next(v for v in variants if v.position == "late")
    assert [d.doc_id for d in late.documents[-2:]] == supporting
    middle = next(v for v in variants if v.position == "middle")
    middle_ids = [d.doc_id for d in middle.documents]
    assert middle_ids[0] != "doc2" and middle_ids[-1] != "doc4"  # not at either edge


def test_build_positional_variants_requires_at_least_one_supporting_document() -> None:
    docs = _docs()
    with pytest.raises(ValueError, match="at least one supporting document"):
        build_positional_variants(docs, supporting_doc_ids=[])
