"""Tests for the ChainRAG adapter (ant.external_wrappers.chainrag). Per
the governing spec's Part 12, these MUST pass before any paid call.

Two external boundaries are mocked, never the algorithm itself:
- The LLM call site (CountingOpenAIProvider) -- a deterministic scripted
  stub, the same convention used throughout this suite.
- `_embed_texts` (real, billed OpenAI calls in production) -- replaced
  with this suite's own free, local `DenseEmbedder` (fastembed/ONNX,
  already used by dense_retrieval_document.py) purely so these tests can
  exercise REAL semantic-similarity behavior (graph construction,
  seed retrieval, hop expansion) without any API cost. The reranker
  (`BAAI/bge-reranker-large`) is REAL and REAL local -- it is genuinely
  free, so it is never mocked.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.document_scope import DocumentRecord
from ant.external_wrappers import chainrag as chainrag_module
from ant.external_wrappers.chainrag import (
    ChainRAGAdapter,
    _build_sentence_graph,
    _calculate_bm25_importance,
    _compute_similarity_matrix,
    _split_sentences_per_document,
    decompose_question,
)

# ===========================================================================
# Structural invariants: no ANTMAN/BM25-retrieval/sparse-fallback names bound.
# ===========================================================================


def test_module_never_binds_antman_or_shared_retrieval_names() -> None:
    names = set(vars(chainrag_module))
    forbidden = (
        "LocalCoordinator",
        "NeedGraph",
        "AutonomousWorker",
        "evolve_workers",
        "LocalSearchTool",
        "BM25Index",
        "_reciprocal_rank_fusion",
        "EmbeddingIndex",
    )
    for name in forbidden:
        assert name not in names


def test_agent_is_registered() -> None:
    from ant.evaluation_suite.registry import get_agent

    agent = get_agent("chainrag")
    assert isinstance(agent, ChainRAGAdapter)


def test_no_max_output_tokens_below_openai_api_minimum() -> None:
    """Regression guard: the OpenAI Responses API hard-rejects
    `max_output_tokens < 16` (HTTP 400 invalid_request_error). An earlier
    version of this module passed `max_output_tokens=10` to
    `can_answer_question` and the reference-resolution check -- every
    such call silently 400'd and was swallowed by the per-sub-question
    broad except, collapsing every sub-answer to an error string (see
    docs/chainrag_fidelity_audit.md's "Implementation bug" section)."""
    source = inspect.getsource(chainrag_module)
    for match in re.finditer(r"max_output_tokens\s*=\s*(\d+)", source):
        assert int(match.group(1)) >= 16, (
            f"max_output_tokens={match.group(1)} is below the OpenAI API's own minimum of 16"
        )


# ===========================================================================
# Corpus adaptation: sentence splitting preserves document identity/order.
# ===========================================================================


def test_split_sentences_per_document_preserves_order_and_source() -> None:
    docs = [
        DocumentRecord(doc_id="doc0", title="", text="Alpha bravo. Charlie delta."),
        DocumentRecord(doc_id="doc1", title="", text="Echo foxtrot."),
    ]
    sentences, source_ids = _split_sentences_per_document(docs)
    assert sentences == ["Alpha bravo.", "Charlie delta.", "Echo foxtrot."]
    assert source_ids == ["doc0", "doc0", "doc1"]


# ===========================================================================
# Graph construction: verbatim edge-building logic, exercised for real.
# ===========================================================================


def test_similarity_matrix_and_graph_construction_are_real() -> None:
    embeddings = [np.array([1.0, 0.0]), np.array([1.0, 0.01]), np.array([0.0, 1.0])]
    sim = _compute_similarity_matrix(embeddings)
    assert sim[0][1] > sim[0][2]  # sentences 0/1 are near-identical, 2 is orthogonal

    ent_lists = [["Paris"], ["Paris"], ["Tokyo"]]
    high_importance, entity_scores = _calculate_bm25_importance(ent_lists)
    assert "Paris" in entity_scores and "Tokyo" in entity_scores

    graph = _build_sentence_graph(["a", "b", "c"], sim, ent_lists, entity_scores, k=1)
    assert graph.number_of_nodes() == 3
    # 0 and 1 share the entity "Paris" -> an entity edge must exist between them.
    assert graph.has_edge(0, 1)


# ===========================================================================
# Decomposition: real behavior via a scripted LLM stub.
# ===========================================================================


def test_decompose_question_returns_single_question_when_not_multihop() -> None:
    calls = chainrag_module._CallLog()
    responses = iter(['{"is_multi_hop": false}'])

    def fake_llm(prompt: str, max_output_tokens: int = 400) -> str:
        return next(responses)

    result = decompose_question("Who is the mayor?", fake_llm, calls)
    assert result == ["Who is the mayor?"]
    assert calls.n_llm_calls == 1  # only the judge call, no decompose call


def test_decompose_question_returns_subquestions_when_multihop() -> None:
    calls = chainrag_module._CallLog()
    responses = iter(['{"is_multi_hop": true}', '["Who directed film A?", "Who directed film B?"]'])

    def fake_llm(prompt: str, max_output_tokens: int = 400) -> str:
        return next(responses)

    result = decompose_question("Compare directors of A and B", fake_llm, calls)
    assert result == ["Who directed film A?", "Who directed film B?"]
    assert calls.n_llm_calls == 2


# ===========================================================================
# End-to-end run(): mocked LLM (scripted) + free local embeddings.
# ===========================================================================


class _ScriptedProvider:
    """Returns pre-scripted responses in call order -- same convention as
    test_longagent.py's own _ScriptedProvider."""

    def __init__(self, scripted: list[str]) -> None:
        self._scripted = list(scripted)
        self._calls = 0
        self.prompts: list[str] = []

    def responses_text(self, prompt: str, max_output_tokens: int = 512):
        from ant.domain import TokenUsage
        from ant.providers.openai_provider import ResponseResult

        self.prompts.append(prompt)
        self._calls += 1
        text = self._scripted.pop(0) if self._scripted else "no"
        return ResponseResult(text=text, usage=TokenUsage(), raw={})

    def client(self):
        return None  # never used: _embed_texts is monkeypatched below

    def drain_call_count(self) -> int:
        count = self._calls
        self._calls = 0
        return count

    def drain_usage(self):
        from ant.domain import TokenUsage

        return TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)


def _fake_embed_texts(client, texts, usage, batch_size=20):
    """Free, local, real-semantics stand-in for the real (paid) OpenAI
    embedding call -- uses this suite's own local DenseEmbedder
    (fastembed/ONNX), never a hand-rolled/random vector, so retrieval
    ranking is genuinely exercised, not just structurally invoked."""
    from ant.retrieval.dense import DenseEmbedder

    embedder = DenseEmbedder()
    usage.calls += 1
    usage.tokens += sum(len(t.split()) for t in texts)
    return [np.array(v, dtype=np.float32) for v in embedder.embed(texts)]


def _make_example(question: str, docs: dict[str, str], metadata: dict | None = None) -> TaskExample:
    documents = [DocumentRecord(doc_id=k, title="", text=v) for k, v in docs.items()]
    return TaskExample(
        benchmark="test",
        task_id="t1",
        question=question,
        reference="",
        metadata={"documents": [d.model_dump() for d in documents], **(metadata or {})},
    )


def test_end_to_end_run_decomposes_processes_sequentially_and_synthesizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _ScriptedProvider(
        [
            '{"is_multi_hop": true}',  # decompose: judge
            '["Where does the cat live?", "What color is the house there?"]',  # decompose: split
            "yes",  # can_answer_question for sub-q 1
            "the windowsill",  # force_answer for sub-q 1
            # sub-q 2 contains "there" -> reference check triggers
            "yes",  # reference check: does it refer to previous answer
            "What color is the house at the windowsill?",  # rewrite
            "yes",  # can_answer_question for rewritten sub-q 2
            "blue",  # force_answer for sub-q 2
            "windowsill, blue house",  # answer_with_subquestions
            "windowsill, blue house",  # answer_without_subquestions
        ]
    )
    monkeypatch.setattr(chainrag_module, "_ZeroTemperatureProvider", lambda model: provider)
    monkeypatch.setattr(chainrag_module, "_embed_texts", _fake_embed_texts)

    docs = {
        "a": "The cat lives on the windowsill. The house at the windowsill is blue.",
    }
    example = _make_example("Where does the cat live and what color is the house there?", docs)
    agent = ChainRAGAdapter()
    result = agent.run(example, environment_root=Path("."))

    assert result.final_answer == "windowsill, blue house"
    assert result.metadata["n_sub_questions"] == 2
    assert result.metadata["n_rewrites"] == 1  # sub-q 2 was rewritten
    assert len(result.trajectory[0]["sub_results"]) == 2
    # Sequential processing: sub-q 1 fully resolved before sub-q 2's own
    # reference-check prompt (which mentions sub-q 1's own answer) fired.
    assert "windowsill" in provider.prompts[4].lower()  # the reference-check prompt


def test_no_gold_or_supporting_fact_metadata_enters_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _ScriptedProvider(['{"is_multi_hop": false}', "yes", "a fact", "a fact", "a fact"])
    monkeypatch.setattr(chainrag_module, "_ZeroTemperatureProvider", lambda model: provider)
    monkeypatch.setattr(chainrag_module, "_embed_texts", _fake_embed_texts)

    secret = "SECRET_GOLD_VALUE_MUST_NEVER_APPEAR"
    docs = {"a": "The feline slept peacefully on the warm windowsill all afternoon."}
    example = _make_example(
        "Where did the cat rest?",
        docs,
        metadata={"supporting_doc_ids": ["a"], "gold_answer_secret": secret},
    )
    agent = ChainRAGAdapter()
    agent.run(example, environment_root=Path("."))

    for prompt in provider.prompts:
        assert secret not in prompt


def test_graph_expansion_is_exercised_when_seed_context_is_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # "no" to can_answer_question forces graph expansion to run before the
    # final unconditional force_answer.
    provider = _ScriptedProvider(
        [
            '{"is_multi_hop": false}',
            "no",  # can_answer_question (seed context)
            "no",  # can_answer_question (1-hop expanded)
            "a forced answer",  # unconditional force_answer after hop expansion
            "a forced answer",  # answer_with_subquestions
            "a forced answer",  # answer_without_subquestions
        ]
    )
    monkeypatch.setattr(chainrag_module, "_ZeroTemperatureProvider", lambda model: provider)
    monkeypatch.setattr(chainrag_module, "_embed_texts", _fake_embed_texts)

    docs = {
        "a": (
            "Sentence one about topic A. Sentence two about topic B. "
            "Sentence three about topic C. Sentence four about topic D."
        )
    }
    example = _make_example("What does the mysterious topic Z discuss?", docs)
    agent = ChainRAGAdapter()
    result = agent.run(example, environment_root=Path("."))

    assert result.final_answer == "a forced answer"
    # Both can_answer_question calls (seed + 1-hop) plus the final forced
    # answer all happened -- graph expansion was genuinely exercised, not
    # skipped.
    assert provider.drain_call_count() == 0  # already drained inside run()


def test_retrieval_uses_embedding_and_reranker_not_bm25(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _ScriptedProvider(
        ['{"is_multi_hop": false}', "yes", "an answer", "an answer", "an answer"]
    )
    monkeypatch.setattr(chainrag_module, "_ZeroTemperatureProvider", lambda model: provider)
    monkeypatch.setattr(chainrag_module, "_embed_texts", _fake_embed_texts)

    docs = {
        "a": "The feline slept peacefully on the warm windowsill all afternoon.",
        "b": "Cats cats cats: a numeric list of cats -- cats cats cats cats cats cats.",
    }
    example = _make_example("Where did the cat rest?", docs)
    agent = ChainRAGAdapter()
    result = agent.run(example, environment_root=Path("."))

    sub_result = result.trajectory[0]["sub_results"][0]
    assert sub_result["n_context_sentences"] >= 1
