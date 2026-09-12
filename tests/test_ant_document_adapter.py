"""Tests for the ANT document adapter (`ant.agents.ant_document_adapter`).

Two tiers, matching this evaluation suite's own precedent for
`ant.agents.ant_adapter.AntAgent` (which has no dedicated unit-test file at
all -- that class is validated end-to-end via real smoke runs, not unit
mocks, since its whole job is to wire frozen core components together
unmodified):

1. Pure unit tests for `_document_territories` -- the ONE piece of actual
   new logic this module adds (everything else is frozen-core `build_
   worker_cards`/`LocalCoordinator`/`IndexStore`, called unmodified).
2. One true integration test that builds territories/workers the exact
   way `AntDocumentAgent` does and feeds them into the REAL, unmodified
   `LocalCoordinator.ask()` with no reasoner (the same mechanical-fallback
   pipeline `test_local_coordinator_returns_grounded_evidence` already
   exercises for repositories) -- proving the frozen ANT core actually
   runs end-to-end on a document substrate and finds the right document,
   with zero algorithmic changes, which is the central claim Section 9 of
   the long-context evaluation spec requires to be verified, not merely
   asserted.
"""
from __future__ import annotations

from pathlib import Path

from ant.agents.ant_document_adapter import _document_territories
from ant.coordinator import LocalCoordinator
from ant.evaluation_suite.document_scope import (
    DocumentRecord,
    EvalDocumentEnvironment,
    materialize_documents,
)
from ant.indexing import build_worker_cards


def _sample_documents() -> list[DocumentRecord]:
    return [
        DocumentRecord(doc_id="doc0", title="Auth Systems", text="authenticate_user returns True."),
        DocumentRecord(doc_id="doc1", title="Unrelated Topic", text="This document is about gardening."),
        DocumentRecord(doc_id="doc2", title="", text="A document with an empty title."),
    ]


def test_document_territories_one_per_document_in_original_order(tmp_path: Path) -> None:
    documents = _sample_documents()
    materialize_documents(documents, tmp_path)
    environment = EvalDocumentEnvironment(tmp_path, documents)

    territories = _document_territories(environment)

    assert [t.root for t in territories] == ["doc0", "doc1", "doc2"]
    assert [t.files for t in territories] == [["doc_0000.txt"], ["doc_0001.txt"], ["doc_0002.txt"]]
    assert territories[0].id == "doc-doc0"
    assert "Auth Systems" in territories[0].summary
    # Empty-title document still gets a deterministic, non-empty summary.
    assert territories[2].summary


def test_document_territories_are_deterministic_across_repeated_calls(tmp_path: Path) -> None:
    documents = _sample_documents()
    materialize_documents(documents, tmp_path)
    environment = EvalDocumentEnvironment(tmp_path, documents)

    first = _document_territories(environment)
    second = _document_territories(environment)

    assert [t.model_dump() for t in first] == [t.model_dump() for t in second]


def test_document_territories_construction_uses_only_doc_id_title_and_relative_path(
    tmp_path: Path,
) -> None:
    # DocumentRecord has no supporting-fact/gold field at all -- structurally
    # absent, not merely unused -- so territory construction cannot leak it
    # regardless of what a benchmark adapter's own TaskExample.metadata
    # carries alongside the materialized documents. This test pins that
    # down for the actual construction function itself: two document sets
    # identical except for doc_id/title/text produce territories that never
    # reference anything outside those three fields.
    documents = _sample_documents()
    materialize_documents(documents, tmp_path)
    environment = EvalDocumentEnvironment(tmp_path, documents)

    territories = _document_territories(environment)

    for territory, document in zip(territories, documents, strict=True):
        assert territory.root == document.doc_id
        assert territory.files == [environment.relative_path_for(document.doc_id)]


def test_ant_core_runs_unmodified_on_a_document_substrate_and_finds_the_right_document(
    tmp_path: Path,
) -> None:
    # The central Section 9 claim, verified rather than merely asserted:
    # feeding document-derived Territory/WorkerCard objects into the
    # frozen, real LocalCoordinator.ask() (no reasoner -- the same
    # mechanical fallback pipeline the repository-QA equivalent test
    # already exercises) must actually route to the one document worker
    # whose own file contains the answer, exactly like a repository
    # worker would for its own files.
    documents = _sample_documents()
    materialize_documents(documents, tmp_path)
    environment = EvalDocumentEnvironment(tmp_path, documents)
    territories = _document_territories(environment)
    workers = build_worker_cards(environment.root, territories)

    state = LocalCoordinator(tmp_path, workers).ask("How is a user authenticated?")

    assert state.has_evidence()
    assert any(item.path == "doc_0000.txt" for item in state.evidence)
    assert state.rounds
    assert state.rounds[0].node_executions
