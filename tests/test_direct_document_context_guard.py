"""Tests for DirectDocumentAgent's context-window guard (Section 8 of the
Part B spec / Section 5 of the original long-context spec): if the full
benchmark context plus prompt overhead exceeds the model's supported
context window, the agent must STOP and report rather than silently
truncate any document's text.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ant.agents import direct_document as direct_document_module
from ant.agents.direct_document import DirectDocumentAgent
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.document_scope import DocumentRecord, materialize_documents


def test_direct_document_stops_and_reports_when_context_exceeds_the_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Lower the limit far below what a real call would need, rather than
    # constructing an actually-enormous document set -- exercises the same
    # guard branch without a slow multi-hundred-thousand-token test.
    monkeypatch.setattr(direct_document_module, "MAX_CONTEXT_TOKENS", 50)
    documents = [DocumentRecord(doc_id="doc0", title="T", text="word " * 200)]
    materialize_documents(documents, tmp_path)
    example = TaskExample(
        benchmark="test",
        task_id="t1",
        question="What does this say?",
        reference="[]",
        metadata={"documents": [d.model_dump() for d in documents]},
    )

    result = DirectDocumentAgent().run(example, tmp_path)

    assert result.termination_reason == "context_window_exceeded"
    assert result.final_answer == ""
    assert result.usage.llm_calls == 0  # no call was made -- not a truncated one
    assert result.metadata["prompt_tokens_estimate"] > 50
    assert result.metadata["max_context_tokens"] == 50


def test_direct_document_never_truncates_document_text_when_over_the_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The guard must reject the WHOLE task rather than silently keep only
    # a prefix of the documents -- verified by confirming no physical call
    # was attempted at all (a truncate-and-continue implementation would
    # still show llm_calls == 1).
    monkeypatch.setattr(direct_document_module, "MAX_CONTEXT_TOKENS", 10)
    documents = [
        DocumentRecord(doc_id="doc0", title="A", text="alpha " * 50),
        DocumentRecord(doc_id="doc1", title="B", text="beta " * 50),
    ]
    materialize_documents(documents, tmp_path)
    example = TaskExample(
        benchmark="test",
        task_id="t1",
        question="Q?",
        reference="[]",
        metadata={"documents": [d.model_dump() for d in documents]},
    )

    result = DirectDocumentAgent().run(example, tmp_path)

    assert result.termination_reason == "context_window_exceeded"
    assert result.usage.llm_calls == 0


def test_direct_document_under_the_limit_runs_normally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    class _FakeProvider:
        def responses_text(self, prompt: str, max_output_tokens: int = 512):
            from ant.domain import TokenUsage
            from ant.providers.openai_provider import ResponseResult

            calls.append(prompt)
            return ResponseResult(text="a real answer", usage=TokenUsage(), raw={})

        def drain_call_count(self) -> int:
            return len(calls)

        def drain_usage(self):
            from ant.domain import TokenUsage

            return TokenUsage()

        def drain_retry_log(self) -> list[dict]:
            return []

    monkeypatch.setattr(
        direct_document_module, "CountingOpenAIProvider", lambda model: _FakeProvider()
    )
    documents = [DocumentRecord(doc_id="doc0", title="T", text="short document text")]
    materialize_documents(documents, tmp_path)
    example = TaskExample(
        benchmark="test",
        task_id="t1",
        question="Q?",
        reference="[]",
        metadata={"documents": [d.model_dump() for d in documents]},
    )

    result = DirectDocumentAgent().run(example, tmp_path)

    assert result.termination_reason == "single_call_complete"
    assert len(calls) == 2  # 1 for the answer, 1 for the concise-answer condensation
