from __future__ import annotations

from pathlib import Path

from ant.evaluation_suite.document_scope import (
    DocumentRecord,
    EvalDocumentEnvironment,
    chunk_documents,
    materialize_documents,
)
from ant.tools.document_tools import chunk_by_global_index, navigate_chunk, view_document


def _sample_documents() -> list[DocumentRecord]:
    return [
        DocumentRecord(doc_id="doc0", title="First", text="alpha beta gamma"),
        DocumentRecord(doc_id="doc1", title="", text="delta epsilon"),
    ]


def test_view_document_returns_title_and_text_verbatim(tmp_path: Path) -> None:
    documents = _sample_documents()
    materialize_documents(documents, tmp_path)
    environment = EvalDocumentEnvironment(tmp_path, documents)

    assert view_document(environment, "doc0") == "Title: First\n\nalpha beta gamma"
    # Empty title: no "Title: " prefix line, body only.
    assert view_document(environment, "doc1") == "delta epsilon"


def test_view_document_unknown_doc_id_returns_empty_string_not_a_crash(tmp_path: Path) -> None:
    documents = _sample_documents()
    materialize_documents(documents, tmp_path)
    environment = EvalDocumentEnvironment(tmp_path, documents)

    assert view_document(environment, "doc-nonexistent") == ""


def test_navigate_chunk_next_and_previous_step_across_document_boundaries() -> None:
    documents = [
        DocumentRecord(doc_id="doc0", title="", text="word " * 5),
        DocumentRecord(doc_id="doc1", title="", text="word " * 5),
    ]
    chunks = chunk_documents(documents, chunk_size_tokens=3)
    assert len(chunks) > 2  # at least 2 chunks per document

    first = chunks[0]
    forward = navigate_chunk(chunks, first.global_chunk_index, "next")
    assert forward is not None
    assert forward.global_chunk_index == first.global_chunk_index + 1

    back_again = navigate_chunk(chunks, forward.global_chunk_index, "previous")
    assert back_again is not None
    assert back_again.global_chunk_index == first.global_chunk_index


def test_navigate_chunk_returns_none_past_either_end() -> None:
    documents = [DocumentRecord(doc_id="doc0", title="", text="word " * 5)]
    chunks = chunk_documents(documents, chunk_size_tokens=3)

    assert navigate_chunk(chunks, 0, "previous") is None
    assert navigate_chunk(chunks, len(chunks) - 1, "next") is None


def test_navigate_chunk_rejects_an_unknown_direction_or_out_of_range_index() -> None:
    documents = [DocumentRecord(doc_id="doc0", title="", text="word " * 5)]
    chunks = chunk_documents(documents, chunk_size_tokens=3)

    assert navigate_chunk(chunks, 0, "sideways") is None
    assert navigate_chunk(chunks, 999, "next") is None
    assert navigate_chunk(chunks, -1, "next") is None


def test_chunk_by_global_index_finds_the_matching_chunk_or_returns_none() -> None:
    documents = [DocumentRecord(doc_id="doc0", title="", text="word " * 5)]
    chunks = chunk_documents(documents, chunk_size_tokens=3)

    found = chunk_by_global_index(chunks, chunks[-1].global_chunk_index)
    assert found is chunks[-1]
    assert chunk_by_global_index(chunks, 9999) is None
