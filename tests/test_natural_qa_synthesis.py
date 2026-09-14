"""Tests for ant.evaluation_suite.natural_qa_synthesis -- the benchmark-
scoped final-answer synthesis policy that fixes ANTMAN's benchmark-policy
leakage (SWE-QA-Pro's abstention-tolerant grounding was previously
inherited, unmodified, by HotpotQA/2WikiMultihopQA/MuSiQue). Per the
governing spec's VALIDATION section:

A. is covered in tests/test_ant_document_adapter.py (SWE-QA-Pro's own
   adapter module never imports this one at all).
B/C. Real LLM composition/abstention behavior cannot be asserted
   deterministically without a live API call (this suite's established
   convention -- see every other agent's own test file -- is to mock the
   LLM boundary, not call it for real in the automated suite), so these
   tests instead pin down the POLICY itself: the prompt text explicitly
   permits compositional inference and explicitly still permits
   abstention on genuinely insufficient evidence, and the function is a
   pure pass-through of whatever the model returns (no post-hoc filter
   that could suppress a correct composed answer or force an abstention).
   The real behavioral claim is verified empirically by the 90-example
   re-answer experiment itself, whose results are the actual evidence for
   B/C at scale.
D. is covered by a structural signature check plus a scripted-provider
   check that only question/evidence_block ever reach the prompt.
"""
from __future__ import annotations

import inspect

import pytest

from ant.domain import TokenUsage
from ant.domain.models import Evidence
from ant.evaluation_suite.natural_qa_synthesis import (
    ABSTENTION_TEXT,
    NATURAL_MULTIHOP_QA_BENCHMARKS,
    format_evidence_block,
    is_natural_multihop_qa_benchmark,
    synthesize_natural_multihop_answer,
)
from ant.providers.openai_provider import ResponseResult


def test_is_natural_multihop_qa_benchmark_exact_three() -> None:
    assert NATURAL_MULTIHOP_QA_BENCHMARKS == {"hotpotqa", "2wikimultihopqa", "musique"}
    for name in ("hotpotqa", "2wikimultihopqa", "musique"):
        assert is_natural_multihop_qa_benchmark(name)
    for name in ("swe_qa_pro", "webwalkerqa", "repoprobe-python", "", "HotpotQA"):
        assert not is_natural_multihop_qa_benchmark(name)


def test_format_evidence_block_dicts_and_model_instances_are_equivalent() -> None:
    as_dicts = [
        {"path": "doc_0000.txt", "line_start": 1, "line_end": 1, "quote": "Alpha is blue."},
        {"path": "doc_0001.txt", "line_start": 3, "line_end": 3, "quote": "Beta is red."},
    ]
    as_models = [
        Evidence(path="doc_0000.txt", line_start=1, line_end=1, quote="Alpha is blue.", reason=""),
        Evidence(path="doc_0001.txt", line_start=3, line_end=3, quote="Beta is red.", reason=""),
    ]
    assert format_evidence_block(as_dicts) == format_evidence_block(as_models)
    assert format_evidence_block(as_dicts) == (
        "[doc_0000.txt:1-1] Alpha is blue.\n[doc_0001.txt:3-3] Beta is red."
    )


def test_format_evidence_block_empty_list_is_empty_string() -> None:
    assert format_evidence_block([]) == ""


class _ScriptedProvider:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[tuple[str, int]] = []

    def responses_text(self, prompt: str, max_output_tokens: int = 512) -> ResponseResult:
        self.calls.append((prompt, max_output_tokens))
        return ResponseResult(text=self.response_text, usage=TokenUsage(), raw={})


def test_synthesize_returns_abstention_without_any_llm_call_when_evidence_is_empty() -> None:
    provider = _ScriptedProvider("should never be seen")
    result = synthesize_natural_multihop_answer(provider, "Any question?", "")
    assert result == ABSTENTION_TEXT
    assert provider.calls == []  # zero-cost short-circuit, matching condense_to_answer_span's own


def test_synthesize_prompt_explicitly_permits_compositional_inference() -> None:
    provider = _ScriptedProvider("Badly Drawn Boy")
    evidence_block = format_evidence_block(
        [
            {
                "path": "doc_0006.txt",
                "line_start": 3,
                "line_end": 3,
                "quote": "Badly Drawn Boy is the stage name of a solo English singer-songwriter.",
            },
            {
                "path": "doc_0000.txt",
                "line_start": 3,
                "line_end": 3,
                "quote": "Wolf Alice are a four-piece alternative rock band.",
            },
        ]
    )
    question = "Which act has a higher instrument-to-person ratio, Badly Drawn Boy or Wolf Alice?"
    result = synthesize_natural_multihop_answer(provider, question, evidence_block)

    assert result == "Badly Drawn Boy"  # pure pass-through of the model's own composed answer
    assert len(provider.calls) == 1
    prompt, max_output_tokens = provider.calls[0]
    assert max_output_tokens > 0
    assert "does NOT need to appear verbatim" in prompt
    assert "compositional inference" in prompt
    assert "comparisons" in prompt and "counting" in prompt and "arithmetic" in prompt
    assert "Do not refuse to answer merely because" in prompt
    assert question in prompt
    assert "Badly Drawn Boy is the stage name" in prompt
    assert "Wolf Alice are a four-piece" in prompt


def test_synthesize_prompt_still_permits_abstention_on_insufficient_evidence() -> None:
    provider = _ScriptedProvider(ABSTENTION_TEXT)
    evidence_block = format_evidence_block(
        [
            {
                "path": "doc_0009.txt",
                "line_start": 3,
                "line_end": 3,
                "quote": "Mike Park is part of The Bruce Lee Band.",
            }
        ]
    )
    question = "What record label did the person who is part of The Bruce Lee Band start?"
    result = synthesize_natural_multihop_answer(provider, question, evidence_block)

    # Pure pass-through: the policy does not force an answer when the
    # model itself judges the evidence insufficient -- a true routing miss
    # (the second-hop fact was never retrieved) must still surface as a
    # failure, not get hallucinated around.
    assert result == ABSTENTION_TEXT
    prompt, _ = provider.calls[0]
    assert "Do not invent or guess a fact the evidence does not support" in prompt


def test_synthesize_falls_back_to_abstention_text_on_blank_model_output() -> None:
    provider = _ScriptedProvider("   ")
    result = synthesize_natural_multihop_answer(provider, "Q?", "[doc:1-1] some fact.")
    assert result == ABSTENTION_TEXT


def test_synthesize_function_signature_has_no_gold_or_metadata_channel() -> None:
    # Structural guarantee, not just an empirical check: the function
    # literally cannot accept a gold answer, supporting-fact list, or any
    # other example metadata -- there is no parameter for it.
    params = list(inspect.signature(synthesize_natural_multihop_answer).parameters)
    assert params == ["provider", "question", "evidence_block"]


def test_synthesize_prompt_contains_only_the_supplied_question_and_evidence() -> None:
    secret = "GOLD_ANSWER_SECRET_MUST_NEVER_APPEAR"
    provider = _ScriptedProvider("an answer")
    evidence_block = format_evidence_block(
        [{"path": "doc.txt", "line_start": 1, "line_end": 1, "quote": "Some unrelated fact."}]
    )
    synthesize_natural_multihop_answer(provider, "A clean question?", evidence_block)
    prompt, _ = provider.calls[0]
    assert secret not in prompt


@pytest.mark.parametrize("bad_evidence", [None, "not a list"])
def test_format_evidence_block_rejects_non_iterable_input_loudly(bad_evidence) -> None:
    with pytest.raises((TypeError, AttributeError)):
        format_evidence_block(bad_evidence)  # type: ignore[arg-type]
