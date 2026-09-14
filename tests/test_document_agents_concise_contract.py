"""Verifies the Part A shared short-answer contract
(ant.evaluation_suite.answer_contract.condense_to_answer_span) is actually
wired into all five long-context methods identically -- not selectively
applied to only some of them, and never inserted earlier than the very
last step of each method's own pipeline (see answer_contract.py's own
module docstring for why). No real API calls: each method's own
`CountingOpenAIProvider` reference is monkeypatched with a small fake that
distinguishes the condensation call from every other call by the
condensation prompt's own distinctive marker text, the same
mock-the-external-boundary convention used throughout this evaluation
suite (see tests/test_longagent.py's own module docstring).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ant.agents import ant_document_adapter as ant_document_adapter_module
from ant.agents import direct_document as direct_document_module
from ant.agents import matched_react_document as matched_react_document_module
from ant.agents import retrieval_document as retrieval_document_module
from ant.benchmarks.base import TaskExample
from ant.domain import TokenUsage
from ant.evaluation_suite.document_scope import DocumentRecord, materialize_documents
from ant.providers.openai_provider import ResponseResult

RAW_ANSWER = "A long discursive paragraph that eventually concludes: yes."
CONDENSED_ANSWER = "yes"
REGROUNDED_ANSWER = "the context-grounded answer"


class _FakeProvider:
    """Stands in for CountingOpenAIProvider across all three document
    agents that call `responses_text`/`responses_json`/`synthesize`
    directly (Direct, Retrieval, Matched ReAct) -- distinguishes the
    condensation call from every other call by looking for
    condense_to_answer_span's own distinctive prompt marker text, so this
    single fake works for all three without per-agent branching logic.
    """

    def __init__(self, decisions: list[str] | None = None, regrounded_answer: str = "") -> None:
        self._decisions = list(decisions or [])
        self.condense_calls: list[str] = []
        self.reground_calls: list[str] = []
        self._regrounded_answer = regrounded_answer or REGROUNDED_ANSWER
        self._calls = 0

    def responses_text(self, prompt: str, max_output_tokens: int = 512) -> ResponseResult:
        self._calls += 1
        if "to be condensed, not replaced" in prompt:
            self.condense_calls.append(prompt)
            return ResponseResult(text=CONDENSED_ANSWER, usage=TokenUsage(), raw={})
        if "strictly grounded in the context/evidence above" in prompt:
            self.reground_calls.append(prompt)
            return ResponseResult(text=self._regrounded_answer, usage=TokenUsage(), raw={})
        return ResponseResult(text=RAW_ANSWER, usage=TokenUsage(), raw={})

    def responses_json(self, prompt: str, max_output_tokens: int = 512) -> ResponseResult:
        self._calls += 1
        text = self._decisions.pop(0) if self._decisions else '{"enough": true}'
        return ResponseResult(text=text, usage=TokenUsage(), raw={})

    def synthesize(self, *, question: str, evidence: list) -> str:
        del question, evidence
        return RAW_ANSWER

    def drain_call_count(self) -> int:
        count = self._calls
        self._calls = 0
        return count

    def drain_usage(self) -> TokenUsage:
        return TokenUsage()

    def drain_retry_log(self) -> list[dict]:
        return []


def _example_with_documents(tmp_path: Path, condition: str | None = None) -> TaskExample:
    documents = [DocumentRecord(doc_id="doc0", title="T", text="alpha beta gamma")]
    materialize_documents(documents, tmp_path)
    metadata: dict = {"documents": [d.model_dump() for d in documents]}
    if condition is not None:
        metadata["answer_contract_condition"] = condition
    return TaskExample(
        benchmark="test",
        task_id="t1",
        question="Is this a question?",
        reference="[]",
        metadata=metadata,
    )


def test_direct_document_condenses_the_raw_answer_as_the_last_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeProvider()
    monkeypatch.setattr(direct_document_module, "CountingOpenAIProvider", lambda model: fake)
    example = _example_with_documents(tmp_path)

    result = direct_document_module.DirectDocumentAgent().run(example, tmp_path)

    assert result.final_answer == CONDENSED_ANSWER
    assert result.metadata["raw_answer_before_condensation"] == RAW_ANSWER
    assert len(fake.condense_calls) == 1


def test_retrieval_document_condenses_the_synthesized_answer_as_the_last_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeProvider(decisions=['{"enough": true}'])
    monkeypatch.setattr(retrieval_document_module, "CountingOpenAIProvider", lambda model: fake)
    example = _example_with_documents(tmp_path)

    result = retrieval_document_module.RetrievalDocumentAgent().run(example, tmp_path)

    assert result.final_answer == CONDENSED_ANSWER
    assert result.metadata["raw_answer_before_condensation"] == RAW_ANSWER
    assert len(fake.condense_calls) == 1


def test_matched_react_document_condenses_the_finish_answer_as_the_last_step(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeProvider(decisions=[f'{{"finish": "{RAW_ANSWER}"}}'])
    monkeypatch.setattr(matched_react_document_module, "CountingOpenAIProvider", lambda model: fake)
    example = _example_with_documents(tmp_path)

    result = matched_react_document_module.MatchedReActDocumentAgent().run(example, tmp_path)

    assert result.final_answer == CONDENSED_ANSWER
    assert result.metadata["raw_answer_before_condensation"] == RAW_ANSWER
    assert len(fake.condense_calls) == 1


def test_matched_react_document_condenses_the_budget_exhausted_fallback_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # No "finish" ever offered -- forces the budget-exhausted synthesize()
    # fallback path, which must ALSO be condensed (not just the "finish"
    # path) -- this is the "no method-specific answer-format advantage"
    # requirement applied within a single method's own two exit paths.
    fake = _FakeProvider(decisions=[])
    monkeypatch.setattr(matched_react_document_module, "CountingOpenAIProvider", lambda model: fake)
    example = _example_with_documents(tmp_path)

    agent = matched_react_document_module.MatchedReActDocumentAgent(tool_call_budget=2)
    result = agent.run(example, tmp_path)

    assert result.final_answer == CONDENSED_ANSWER
    assert len(fake.condense_calls) == 1


def test_ant_document_adapter_calls_the_shared_condensation_function_after_ask_returns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # ant_document_adapter.run() wires condense_to_answer_span in directly.
    # Rather than mocking the full WorkerReasoner protocol a real
    # CountingOpenAIProvider would need LocalCoordinator.ask() to drive
    # (test_ant_document_adapter.py already exercises that path for real,
    # unmocked, against the unmodified core), LocalCoordinator itself is
    # stubbed here to isolate exactly one thing: that AntDocumentAgent.run()
    # calls the shared condensation function exactly once, AFTER
    # coordinator.ask() has already returned, with that exact returned
    # answer, and uses its return value as final_answer -- i.e. the wiring
    # is correct, independent of whatever LocalCoordinator's own internals
    # decide to do.
    from ant.domain import EvidenceState

    calls: list[tuple[object, str, str]] = []

    def _fake_condense(provider: object, question: str, raw_answer: str) -> str:
        calls.append((provider, question, raw_answer))
        return CONDENSED_ANSWER

    class _StubCoordinator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def ask(self, question: str, max_rounds: int = 6, search_top_k: int = 4) -> EvidenceState:
            return EvidenceState(question=question, answer=RAW_ANSWER)

    monkeypatch.setattr(ant_document_adapter_module, "condense_to_answer_span", _fake_condense)
    monkeypatch.setattr(ant_document_adapter_module, "LocalCoordinator", _StubCoordinator)
    documents = [
        DocumentRecord(doc_id="doc0", title="Auth", text="authenticate_user returns True.")
    ]
    materialize_documents(documents, tmp_path)
    example = TaskExample(
        benchmark="test",
        task_id="t1",
        question="How is a user authenticated?",
        reference="[]",
        metadata={"documents": [d.model_dump() for d in documents]},
    )

    result = ant_document_adapter_module.AntDocumentAgent().run(example, tmp_path)

    assert result.final_answer == CONDENSED_ANSWER
    assert len(calls) == 1
    _, question, raw_answer = calls[0]
    assert question == example.question
    assert raw_answer == RAW_ANSWER
    assert result.metadata["raw_answer_before_condensation"] == raw_answer


# ---------------------------------------------------------------------------
# Single-Needle Contamination Study Condition B: apply_context_authoritative_
# regrounding must run BEFORE condensation, and ONLY when
# example.metadata["answer_contract_condition"] == "B" -- a no-op for every
# other condition/track, since that key is absent everywhere else.
# ---------------------------------------------------------------------------


def test_direct_document_condition_b_regrounds_before_condensing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeProvider()
    monkeypatch.setattr(direct_document_module, "CountingOpenAIProvider", lambda model: fake)
    example = _example_with_documents(tmp_path, condition="B")

    result = direct_document_module.DirectDocumentAgent().run(example, tmp_path)

    assert len(fake.reground_calls) == 1
    assert len(fake.condense_calls) == 1
    assert result.final_answer == CONDENSED_ANSWER
    assert result.metadata["answer_contract_condition"] == "B"
    assert result.metadata["grounded_answer_after_regrounding"] == REGROUNDED_ANSWER


def test_direct_document_condition_none_never_regrounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeProvider()
    monkeypatch.setattr(direct_document_module, "CountingOpenAIProvider", lambda model: fake)
    example = _example_with_documents(tmp_path)  # no condition set

    result = direct_document_module.DirectDocumentAgent().run(example, tmp_path)

    assert len(fake.reground_calls) == 0
    assert result.metadata["answer_contract_condition"] is None
    assert result.metadata["grounded_answer_after_regrounding"] is None


def test_retrieval_document_condition_b_regrounds_before_condensing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeProvider()
    monkeypatch.setattr(retrieval_document_module, "CountingOpenAIProvider", lambda model: fake)
    example = _example_with_documents(tmp_path, condition="B")

    result = retrieval_document_module.RetrievalDocumentAgent().run(example, tmp_path)

    assert len(fake.reground_calls) == 1
    assert len(fake.condense_calls) == 1
    assert result.final_answer == CONDENSED_ANSWER


def test_matched_react_document_condition_b_regrounds_before_condensing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _FakeProvider(decisions=[f'{{"finish": "{RAW_ANSWER}"}}'])
    monkeypatch.setattr(matched_react_document_module, "CountingOpenAIProvider", lambda model: fake)
    example = _example_with_documents(tmp_path, condition="B")

    result = matched_react_document_module.MatchedReActDocumentAgent().run(example, tmp_path)

    assert len(fake.reground_calls) == 1
    assert len(fake.condense_calls) == 1
    assert result.final_answer == CONDENSED_ANSWER


def test_ant_document_adapter_condition_b_regrounds_before_condensing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from ant.domain import Evidence, EvidenceState

    reground_calls: list[tuple[str, str]] = []
    condense_calls: list[str] = []

    def _fake_reground(provider: object, question: str, raw_answer: str, context_text: str) -> str:
        del provider
        reground_calls.append((raw_answer, context_text))
        return REGROUNDED_ANSWER

    def _fake_condense(provider: object, question: str, raw_answer: str) -> str:
        del provider, question
        condense_calls.append(raw_answer)
        return CONDENSED_ANSWER

    class _StubCoordinator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def ask(self, question: str, max_rounds: int = 6, search_top_k: int = 4) -> EvidenceState:
            return EvidenceState(
                question=question,
                answer=RAW_ANSWER,
                evidence=[
                    Evidence(path="doc0.txt", line_start=1, line_end=1, quote="fact", reason="r")
                ],
            )

    monkeypatch.setattr(
        ant_document_adapter_module, "apply_context_authoritative_regrounding", _fake_reground
    )
    monkeypatch.setattr(ant_document_adapter_module, "condense_to_answer_span", _fake_condense)
    monkeypatch.setattr(ant_document_adapter_module, "LocalCoordinator", _StubCoordinator)
    example = _example_with_documents(tmp_path, condition="B")

    result = ant_document_adapter_module.AntDocumentAgent().run(example, tmp_path)

    assert len(reground_calls) == 1
    assert reground_calls[0][0] == RAW_ANSWER
    assert "fact" in reground_calls[0][1]  # built from state.evidence
    assert condense_calls == [REGROUNDED_ANSWER]  # condensation sees the REGROUNDED answer
    assert result.final_answer == CONDENSED_ANSWER
    assert result.metadata["answer_contract_condition"] == "B"
