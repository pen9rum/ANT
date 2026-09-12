"""Tests for the strict, method-agnostic answer-span extractor
(ant.evaluation_suite.answer_extraction). Covers all 7 cases (A-G) the
governing spec requires. No real API calls: the LLM extraction path is
exercised via a monkeypatched `_ZeroTemperatureProvider`, the same
mock-the-external-boundary convention used throughout this suite.
"""

from __future__ import annotations

import pytest

from ant.evaluation_suite import answer_extraction as ae_module
from ant.evaluation_suite.answer_extraction import (
    _is_verbatim_substring,
    extract_answer_span,
)


class _StubProvider:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls = 0

    def responses_text(self, prompt: str, max_output_tokens: int = 512):
        from ant.domain import TokenUsage
        from ant.providers.openai_provider import ResponseResult

        self.calls += 1
        return ResponseResult(text=self._response_text, usage=TokenUsage(), raw={})

    def drain_call_count(self) -> int:
        count = self.calls
        self.calls = 0
        return count

    def drain_usage(self):
        from ant.domain import TokenUsage

        return TokenUsage()


def _patch_provider(monkeypatch: pytest.MonkeyPatch, response_text: str) -> _StubProvider:
    stub = _StubProvider(response_text)
    monkeypatch.setattr(ae_module, "_ZeroTemperatureProvider", lambda model: stub)
    return stub


# --- A: "Yes, both were American." -> "yes" (deterministic, no LLM call) ---


def test_a_yes_no_deterministic_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _patch_provider(monkeypatch, "should not be called")
    result = extract_answer_span("Were both American?", "Yes, both were American.")
    assert result.extracted_answer == "yes"
    assert result.used_llm is False
    assert stub.calls == 0


# --- B: "No, the two locations are in different neighborhoods." -> "no" ---


def test_b_no_deterministic_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _patch_provider(monkeypatch, "should not be called")
    result = extract_answer_span(
        "Are they in the same place?", "No, the two locations are in different neighborhoods."
    )
    assert result.extracted_answer == "no"
    assert result.used_llm is False
    assert stub.calls == 0


# --- C: entity-phrase extraction via the LLM path, verbatim substring ---


def test_c_entity_phrase_extraction_via_llm() -> None:
    import json

    def fake_provider(model: str):
        return _StubProvider(json.dumps({"extracted_span": "Greenwich Village, New York City"}))

    import ant.evaluation_suite.answer_extraction as mod

    original = mod._ZeroTemperatureProvider
    mod._ZeroTemperatureProvider = fake_provider
    try:
        result = extract_answer_span(
            "What city?", "The answer is Greenwich Village, New York City."
        )
        assert result.extracted_answer == "Greenwich Village, New York City"
        assert result.used_llm is True
        assert result.rejected_hallucination is False
    finally:
        mod._ZeroTemperatureProvider = original


# --- D: multiple candidate entities -- must select a substring, never invent ---


def test_d_selects_a_substring_among_several_candidates() -> None:
    import json

    def fake_provider(model: str):
        return _StubProvider(json.dumps({"extracted_span": "Chief of Protocol"}))

    import ant.evaluation_suite.answer_extraction as mod

    original = mod._ZeroTemperatureProvider
    mod._ZeroTemperatureProvider = fake_provider
    try:
        raw = "He served as ambassador, Chief of Protocol, and later governor."
        result = extract_answer_span("What position?", raw)
        assert result.extracted_answer == "Chief of Protocol"
        assert result.extracted_answer in raw
        assert result.rejected_hallucination is False
    finally:
        mod._ZeroTemperatureProvider = original


# --- E: wrong raw answer -- extractor must NOT repair it ---


def test_e_does_not_repair_a_wrong_answer() -> None:
    import json

    # The raw answer is factually wrong (gold would be "Paris"), but the
    # extractor's only job is format, not correctness -- it must return
    # exactly what's in the raw text, never silently fix it to "Paris".
    def fake_provider(model: str):
        return _StubProvider(json.dumps({"extracted_span": "London"}))

    import ant.evaluation_suite.answer_extraction as mod

    original = mod._ZeroTemperatureProvider
    mod._ZeroTemperatureProvider = fake_provider
    try:
        result = extract_answer_span(
            "What is the capital of France?", "The capital of France is London."
        )
        assert result.extracted_answer == "London"  # NOT "Paris" -- not repaired
    finally:
        mod._ZeroTemperatureProvider = original


# --- F: no clean short answer present -- safe fallback to the raw answer ---


def test_f_falls_back_to_raw_answer_when_llm_returns_the_full_text() -> None:
    import json

    raw = "It is genuinely unclear from the provided documents which entity is meant."

    def fake_provider(model: str):
        return _StubProvider(json.dumps({"extracted_span": raw}))

    import ant.evaluation_suite.answer_extraction as mod

    original = mod._ZeroTemperatureProvider
    mod._ZeroTemperatureProvider = fake_provider
    try:
        result = extract_answer_span("What entity?", raw)
        assert result.extracted_answer == raw
        assert result.rejected_hallucination is False
    finally:
        mod._ZeroTemperatureProvider = original


# --- G: attempted non-substring hallucination -- validation rejects it ---


def test_g_rejects_a_non_substring_hallucination_and_falls_back_to_raw() -> None:
    import json

    raw = "The document discusses several unrelated topics without a clear answer."

    def fake_provider(model: str):
        # The "extraction" invents an entity that never appeared in raw.
        return _StubProvider(json.dumps({"extracted_span": "Invented Entity Name"}))

    import ant.evaluation_suite.answer_extraction as mod

    original = mod._ZeroTemperatureProvider
    mod._ZeroTemperatureProvider = fake_provider
    try:
        result = extract_answer_span("What entity?", raw)
        assert result.extracted_answer == raw  # fell back, hallucination rejected
        assert result.rejected_hallucination is True
        assert result.used_llm is True  # the call still happened and is still logged/costed
    finally:
        mod._ZeroTemperatureProvider = original


# --- Additional coverage: the verbatim-substring helper itself ---


def test_is_verbatim_substring_case_and_whitespace_insensitive() -> None:
    assert _is_verbatim_substring("chief of protocol", "He was Chief   of Protocol.")
    assert not _is_verbatim_substring("prime minister", "He was Chief of Protocol.")
    assert not _is_verbatim_substring("", "anything")


def test_extract_answer_span_blank_input_returns_unchanged_without_any_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _patch_provider(monkeypatch, "should not be called")
    result = extract_answer_span("Q?", "")
    assert result.extracted_answer == ""
    assert result.used_llm is False
    assert stub.calls == 0


def test_extract_answer_span_malformed_llm_json_falls_back_to_raw() -> None:
    def fake_provider(model: str):
        return _StubProvider("not valid json at all")

    import ant.evaluation_suite.answer_extraction as mod

    original = mod._ZeroTemperatureProvider
    mod._ZeroTemperatureProvider = fake_provider
    try:
        raw = "Some answer text with no clean JSON response from the extractor."
        result = extract_answer_span("Q?", raw)
        assert result.extracted_answer == raw
        assert result.rejected_hallucination is True
    finally:
        mod._ZeroTemperatureProvider = original


def test_zero_temperature_provider_injects_temperature_into_kwargs() -> None:
    from ant.evaluation_suite.answer_extraction import _ZeroTemperatureProvider

    provider = _ZeroTemperatureProvider(model="gpt-4.1")
    kwargs = provider._responses_kwargs("prompt", 128)
    assert kwargs["temperature"] == 0
    assert kwargs["model"] == "gpt-4.1"
