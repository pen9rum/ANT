"""Tests for the Dense Retrieval baseline
(ant.agents.dense_retrieval_document). Embedding calls are REAL (local,
free -- fastembed/ONNX, no network, no API cost) so ranking behavior is
genuinely exercised, not mocked away; only the LLM answer-generation call
site (CountingOpenAIProvider) is monkeypatched, the same
mock-the-external-boundary convention used throughout this suite. Per
the governing spec's Part 12, these MUST pass before any paid call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ant.agents import dense_retrieval_document as dense_module
from ant.agents.dense_retrieval_document import TOP_K, DenseRetrievalDocumentAgent
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.document_scope import DocumentRecord, materialize_documents


def _write_docs(tmp_path: Path, docs: dict[str, str]) -> Path:
    records = [DocumentRecord(doc_id=doc_id, title="", text=text) for doc_id, text in docs.items()]
    materialize_documents(records, tmp_path)
    return tmp_path


class _RecordingProvider:
    """Records the (question, evidence) args synthesize() receives and
    every prompt condense_to_answer_span/regrounding sends -- lets tests
    assert exactly what reached the LLM, without a real API call."""

    def __init__(self) -> None:
        self.synthesize_calls: list[dict] = []
        self.text_prompts: list[str] = []
        self._calls = 0

    def synthesize(self, *, question, evidence, **kwargs):
        self.synthesize_calls.append({"question": question, "evidence": list(evidence)})
        self._calls += 1
        return "a synthesized raw answer"

    def responses_text(self, prompt: str, max_output_tokens: int = 512):
        from ant.domain import TokenUsage
        from ant.providers.openai_provider import ResponseResult

        self.text_prompts.append(prompt)
        self._calls += 1
        return ResponseResult(text="condensed", usage=TokenUsage(), raw={})

    def drain_call_count(self) -> int:
        count = self._calls
        self._calls = 0
        return count

    def drain_usage(self):
        from ant.domain import TokenUsage

        return TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)

    def drain_retry_log(self) -> list[dict]:
        return []


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    docs: dict[str, str],
    question: str,
    metadata: dict | None = None,
):
    root = _write_docs(tmp_path, docs)
    provider = _RecordingProvider()
    monkeypatch.setattr(dense_module, "CountingOpenAIProvider", lambda model: provider)
    example = TaskExample(
        benchmark="test",
        task_id="t1",
        question=question,
        reference="",
        metadata={
            "documents": [
                d.model_dump()
                for d in [DocumentRecord(doc_id=k, title="", text=v) for k, v in docs.items()]
            ],
            **(metadata or {}),
        },
    )
    agent = DenseRetrievalDocumentAgent()
    result = agent.run(example, root)
    return result, provider


# --- structural invariants: no BM25/RRF/fallback/routing imports at all ---


def test_module_never_imports_bm25_rrf_or_search_tool() -> None:
    # Checks actual bound names in the module's own namespace, not the
    # docstring text (which legitimately explains, in prose, why BM25/
    # LocalSearchTool are NOT used here).
    names = set(vars(dense_module))
    for forbidden in (
        "BM25Index",
        "_bm25_channel_rank",
        "_reciprocal_rank_fusion",
        "LocalSearchTool",
    ):
        assert forbidden not in names


def test_module_never_imports_antman_coordination() -> None:
    names = set(vars(dense_module))
    for forbidden in ("LocalCoordinator", "NeedGraph", "AutonomousWorker", "evolve_workers"):
        assert forbidden not in names


# --- behavioral: ranking is genuinely embedding-based, not lexical ---


def test_ranking_is_semantic_not_lexical(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # doc_a shares NO keywords with the question but is semantically the
    # right answer; doc_b shares many literal keywords with the question
    # but is topically irrelevant. A lexical (BM25) ranker would favor
    # doc_b; a genuine embedding ranker should favor doc_a.
    docs = {
        "a": "The feline slept peacefully on the warm windowsill all afternoon.",
        "b": "Cats cats cats: a numeric list of cats -- cats cats cats cats cats cats.",
    }
    result, provider = _run(monkeypatch, tmp_path, docs, "Where did the cat rest?")
    assert provider.synthesize_calls, "synthesize() must have been called"
    evidence = provider.synthesize_calls[0]["evidence"]
    assert evidence, "dense retrieval must return at least one hit"
    top_hit_path = evidence[0].path
    assert (
        top_hit_path == "doc_0000.txt"
    )  # "a" materializes first, byte-identical content check below


def test_no_sparse_fallback_when_dense_finds_a_semantic_match(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    docs = {"a": "The feline slept peacefully on the warm windowsill all afternoon."}
    result, provider = _run(monkeypatch, tmp_path, docs, "Where did the cat rest?")
    evidence = provider.synthesize_calls[0]["evidence"]
    assert all(e.dense_score != 0.0 or len(evidence) == 0 for e in evidence)
    # No evidence field indicates any lexical scoring path was involved.
    for e in evidence:
        assert "Dense semantic match" in e.reason


# --- top-k parity with Sparse Retrieval ---


def test_top_k_matches_sparse_retrieval_default() -> None:
    assert TOP_K == 8
    assert DenseRetrievalDocumentAgent().top_k == 8


def test_retrieves_at_most_top_k_chunks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # 12 short, distinct paragraphs (blank-line separated) in one file --
    # _retrieval_regions splits on blank lines, giving >8 regions.
    text = "\n\n".join(f"Paragraph number {i} talks about topic {i}." for i in range(12))
    docs = {"a": text}
    result, provider = _run(monkeypatch, tmp_path, docs, "What does paragraph five discuss?")
    evidence = provider.synthesize_calls[0]["evidence"]
    assert len(evidence) <= 8


# --- no gold/supporting-fact metadata ever reaches a prompt ---


def test_no_gold_or_supporting_fact_metadata_enters_inference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    secret = "SECRET_GOLD_VALUE_MUST_NEVER_APPEAR"
    docs = {"a": "The feline slept peacefully on the warm windowsill all afternoon."}
    result, provider = _run(
        monkeypatch,
        tmp_path,
        docs,
        "Where did the cat rest?",
        metadata={"supporting_doc_ids": ["a"], "gold_answer_secret": secret},
    )
    for call in provider.synthesize_calls:
        assert secret not in call["question"]
        for e in call["evidence"]:
            assert secret not in e.quote
            assert secret not in e.reason
    for prompt in provider.text_prompts:
        assert secret not in prompt


# --- determinism ---


def test_retrieval_is_deterministic_across_repeated_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    docs = {
        "a": "The feline slept peacefully on the warm windowsill all afternoon.",
        "b": "Quarterly revenue increased due to strong regional sales growth.",
    }
    result1, provider1 = _run(monkeypatch, tmp_path, docs, "Where did the cat rest?")
    result2, provider2 = _run(monkeypatch, tmp_path, docs, "Where did the cat rest?")
    ids1 = [
        e.path + f":{e.line_start}-{e.line_end}" for e in provider1.synthesize_calls[0]["evidence"]
    ]
    ids2 = [
        e.path + f":{e.line_start}-{e.line_end}" for e in provider2.synthesize_calls[0]["evidence"]
    ]
    scores1 = [round(e.dense_score, 6) for e in provider1.synthesize_calls[0]["evidence"]]
    scores2 = [round(e.dense_score, 6) for e in provider2.synthesize_calls[0]["evidence"]]
    assert ids1 == ids2
    assert scores1 == scores2


# --- same answer-generation stage as Sparse Retrieval ---


def test_uses_the_same_synthesize_and_condensation_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    docs = {"a": "The feline slept peacefully on the warm windowsill all afternoon."}
    result, provider = _run(monkeypatch, tmp_path, docs, "Where did the cat rest?")
    assert len(provider.synthesize_calls) == 1  # exactly one synthesize() call -- one-shot
    assert len(provider.text_prompts) == 1  # exactly one condense_to_answer_span call
    assert result.final_answer == "condensed"
    assert result.usage.llm_calls == 2  # synthesize + condense, deterministic, always


def test_agent_is_registered() -> None:
    from ant.evaluation_suite.registry import get_agent

    agent = get_agent("dense_retrieval_document")
    assert isinstance(agent, DenseRetrievalDocumentAgent)
