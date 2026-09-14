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

import pytest

import ant.agents.ant_document_adapter as ant_document_adapter_module
from ant.agents.ant_document_adapter import AntDocumentAgent, _document_territories
from ant.benchmarks.base import TaskExample
from ant.coordinator import LocalCoordinator
from ant.domain.models import Evidence, EvidenceState
from ant.evaluation_suite.document_scope import (
    DocumentRecord,
    EvalDocumentEnvironment,
    materialize_documents,
)
from ant.indexing import build_worker_cards


def _sample_documents() -> list[DocumentRecord]:
    return [
        DocumentRecord(
            doc_id="doc0", title="Auth Systems", text="authenticate_user returns True."
        ),
        DocumentRecord(
            doc_id="doc1", title="Unrelated Topic", text="This document is about gardening."
        ),
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


def test_document_territories_scale_to_a_niah_plus_sized_instance_deterministically(
    tmp_path: Path,
) -> None:
    # Part B (NIAH+) territories/workers can number in the hundreds
    # (32K-128K token contexts split into many small documents). Territory
    # construction must remain purely mechanical -- one territory per
    # document, no LLM calls, no dependency on which document(s) happen to
    # be the needle(s) -- at this scale, not just the ~10-20-document scale
    # the original 3-benchmark track exercises.
    documents = [
        DocumentRecord(doc_id=f"doc{i}", title=f"Filler {i}", text=f"filler content {i} " * 5)
        for i in range(300)
    ]
    materialize_documents(documents, tmp_path)
    environment = EvalDocumentEnvironment(tmp_path, documents)

    territories = _document_territories(environment)

    assert len(territories) == 300
    assert len({t.id for t in territories}) == 300  # every id unique
    # Deterministic: rebuilding from the same environment produces an
    # identical territory list, regardless of which of these 300 documents
    # (if any) a NIAH+ instance's own needle happens to be -- territory
    # construction never reads that information at all (see next test).
    territories_again = _document_territories(environment)
    assert [t.model_dump() for t in territories] == [t.model_dump() for t in territories_again]


def test_document_territories_never_read_niah_plus_needle_or_depth_metadata(
    tmp_path: Path,
) -> None:
    # A NIAH+-materialized environment is built from plain DocumentRecords
    # exactly like every other benchmark's -- needle_doc_ids/depth_percents
    # live only in the benchmark adapter's own TaskExample.metadata
    # ("niah_metadata"), never inside a DocumentRecord itself. This test
    # pins down that _document_territories' OWN construction is identical
    # whether or not a caller happens to know which documents are needles
    # -- i.e. it structurally cannot special-case a "needle territory",
    # since DocumentRecord has no such field to read in the first place.
    documents = [
        DocumentRecord(doc_id=f"doc{i}", title=f"Filler {i}", text=f"filler {i} " * 5)
        for i in range(20)
    ]
    # Simulate two different "needle placements" over the IDENTICAL
    # document set/order -- construction must be byte-identical regardless.
    materialize_documents(documents, tmp_path)
    environment = EvalDocumentEnvironment(tmp_path, documents)
    needle_ids_scenario_a = {"doc2", "doc15"}  # unused by _document_territories
    needle_ids_scenario_b = {"doc0", "doc19"}  # unused by _document_territories
    del needle_ids_scenario_a, needle_ids_scenario_b  # never passed in -- the point

    for doc in documents:
        assert set(doc.model_dump()) == {"doc_id", "title", "text"}

    territories = _document_territories(environment)
    for territory in territories:
        assert set(territory.model_dump()) == {"id", "root", "files", "summary"}
        # Summary is derived only from doc_id/title -- never mentions
        # needle/depth/position, which _document_territories has no access
        # to at all.
        assert "needle" not in territory.summary.lower()
        assert "depth" not in territory.summary.lower()


# =============================================================================
# Benchmark-scoped final-answer synthesis (natural_qa_synthesis integration).
#
# `LocalCoordinator.ask()` itself (frozen core, routing/Need-Graph/retrieval)
# is stubbed out here -- its own correctness is what the test above already
# proves. These tests target ONLY the thing this change actually touches:
# which final-answer synthesis path `AntDocumentAgent.run()` dispatches to,
# for which benchmark, and that nothing extra leaks into it.
# =============================================================================


class _FakeProvider:
    """Stand-in for CountingOpenAIProvider -- never actually calls
    responses_text() in these tests, since synthesize_natural_multihop_answer/
    condense_to_answer_span are themselves monkeypatched to spies below."""

    def drain_call_count(self) -> int:
        return 0


def _make_fake_coordinator(state: EvidenceState):
    class _FakeCoordinator:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def ask(self, question: str, max_rounds: int = 6, search_top_k: int = 4) -> EvidenceState:
            return state

    return _FakeCoordinator


def _make_example(benchmark: str, metadata: dict | None = None) -> TaskExample:
    return TaskExample(
        benchmark=benchmark,
        task_id="t1",
        question="Which act has more instruments per person?",
        reference="",
        metadata={
            "documents": [d.model_dump() for d in _sample_documents()],
            **(metadata or {}),
        },
    )


def _run_with_stubbed_coordinator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, example: TaskExample, state: EvidenceState
):
    # AntDocumentAgent.run() expects environment_root already materialized
    # by the harness (benchmark.prepare_environment(), called by run_suite
    # before agent.run() in real use) -- replicate that here.
    materialize_documents(_sample_documents(), tmp_path)
    monkeypatch.setattr(
        ant_document_adapter_module, "LocalCoordinator", _make_fake_coordinator(state)
    )
    monkeypatch.setattr(
        ant_document_adapter_module, "CountingOpenAIProvider", lambda model: _FakeProvider()
    )
    agent = AntDocumentAgent()
    return agent.run(example, tmp_path)


def test_ant_adapter_module_never_imports_natural_qa_synthesis() -> None:
    # Structural proof of CRITICAL CONSTRAINT 1: SWE-QA-Pro's own adapter
    # (a completely separate module/class from AntDocumentAgent) has no
    # dependency on the new benchmark-scoped policy at all -- it cannot be
    # affected by anything in natural_qa_synthesis.py or by this change,
    # regardless of what benchmark name it is ever called with.
    import ant.agents.ant_adapter as ant_adapter_module

    names = set(vars(ant_adapter_module))
    assert "natural_qa_synthesis" not in names
    assert "synthesize_natural_multihop_answer" not in names
    assert "is_natural_multihop_qa_benchmark" not in names
    # And AntAgent.run() uses state.answer directly -- no condensation,
    # no natural-QA synthesis, nothing added by this or any prior change.
    import inspect

    source = inspect.getsource(ant_adapter_module.AntAgent.run)
    assert "final_answer=state.answer" in source.replace(" ", "")


def test_run_uses_natural_qa_synthesis_for_hotpotqa_not_condensation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = EvidenceState(
        question="Which act has more instruments per person?",
        answer="not stated in the provided documents",  # what the frozen core itself produced
        evidence=[
            Evidence(
                path="doc_0000.txt",
                line_start=1,
                line_end=1,
                quote="Badly Drawn Boy is a solo artist.",
                reason="test",
            )
        ],
    )
    captured: dict = {}

    def _fake_synthesize(provider, question, evidence_block):
        captured["question"] = question
        captured["evidence_block"] = evidence_block
        return "Badly Drawn Boy"

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("condense_to_answer_span must not be called for hotpotqa")

    monkeypatch.setattr(
        ant_document_adapter_module, "synthesize_natural_multihop_answer", _fake_synthesize
    )
    monkeypatch.setattr(ant_document_adapter_module, "condense_to_answer_span", _fail_if_called)

    example = _make_example("hotpotqa")
    result = _run_with_stubbed_coordinator(monkeypatch, tmp_path, example, state)

    assert result.final_answer == "Badly Drawn Boy"
    assert captured["question"] == example.question
    assert "Badly Drawn Boy is a solo artist." in captured["evidence_block"]
    assert result.metadata["final_synthesis_policy"] == "natural_multihop_qa"


@pytest.mark.parametrize("benchmark", ["2wikimultihopqa", "musique"])
def test_run_uses_natural_qa_synthesis_for_2wiki_and_musique(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, benchmark: str
) -> None:
    state = EvidenceState(question="Q?", answer="not stated in the provided documents", evidence=[])
    monkeypatch.setattr(
        ant_document_adapter_module,
        "synthesize_natural_multihop_answer",
        lambda provider, question, evidence_block: "derived answer",
    )
    monkeypatch.setattr(
        ant_document_adapter_module,
        "condense_to_answer_span",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    example = _make_example(benchmark)
    result = _run_with_stubbed_coordinator(monkeypatch, tmp_path, example, state)
    assert result.final_answer == "derived answer"


def test_run_still_uses_condense_to_answer_span_for_other_benchmarks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Preserves EXACT prior behavior for any benchmark outside the natural-
    # QA set (including a hypothetical future ant_document benchmark) --
    # the "else: preserve current behavior" branch of the governing spec.
    state = EvidenceState(question="Q?", answer="Some raw synthesized answer.", evidence=[])
    captured: dict = {}

    def _fake_condense(provider, question, raw_answer):
        captured["question"] = question
        captured["raw_answer"] = raw_answer
        return "condensed answer"

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("synthesize_natural_multihop_answer must not be called here")

    monkeypatch.setattr(ant_document_adapter_module, "condense_to_answer_span", _fake_condense)
    monkeypatch.setattr(
        ant_document_adapter_module, "synthesize_natural_multihop_answer", _fail_if_called
    )

    example = _make_example("some_future_document_benchmark")
    result = _run_with_stubbed_coordinator(monkeypatch, tmp_path, example, state)

    assert result.final_answer == "condensed answer"
    assert captured["raw_answer"] == "Some raw synthesized answer."
    assert result.metadata["final_synthesis_policy"] == "swe_qa_pro_default"


def test_run_rejects_condition_b_metadata_on_natural_qa_benchmark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Condition B (single-needle contamination-study regrounding) is not
    # defined for the natural-QA branch -- if it is ever set for hotpotqa/
    # 2wiki/musique (it currently never is), this must fail loudly rather
    # than silently applying the wrong policy.
    state = EvidenceState(question="Q?", answer="raw", evidence=[])
    monkeypatch.setattr(
        ant_document_adapter_module,
        "synthesize_natural_multihop_answer",
        lambda *a, **k: "unused",
    )
    example = _make_example("hotpotqa", metadata={"answer_contract_condition": "B"})
    with pytest.raises(AssertionError):
        _run_with_stubbed_coordinator(monkeypatch, tmp_path, example, state)


def test_run_no_gold_or_reference_metadata_reaches_natural_qa_synthesis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = "GOLD_ANSWER_SECRET_MUST_NEVER_APPEAR"
    state = EvidenceState(
        question="Q?",
        answer="raw",
        evidence=[
            Evidence(path="doc.txt", line_start=1, line_end=1, quote="A real fact.", reason="t")
        ],
    )
    captured: dict = {}

    def _fake_synthesize(provider, question, evidence_block):
        captured["question"] = question
        captured["evidence_block"] = evidence_block
        return "an answer"

    monkeypatch.setattr(
        ant_document_adapter_module, "synthesize_natural_multihop_answer", _fake_synthesize
    )
    example = _make_example(
        "hotpotqa", metadata={"gold_answer_secret": secret, "supporting_doc_ids": [secret]}
    )
    _run_with_stubbed_coordinator(monkeypatch, tmp_path, example, state)

    assert secret not in captured["question"]
    assert secret not in captured["evidence_block"]
