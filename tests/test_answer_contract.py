from __future__ import annotations

from ant.evaluation_suite.answer_contract import (
    CONCISE_ANSWER_INSTRUCTION,
    CONTEXT_AUTHORITATIVE_INSTRUCTION,
    MAX_REGROUND_CONTEXT_CHARS,
    apply_context_authoritative_regrounding,
    condense_to_answer_span,
)


class _StubProvider:
    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[tuple[str, int]] = []

    def responses_text(self, prompt: str, max_output_tokens: int = 512):
        from ant.domain import TokenUsage
        from ant.providers.openai_provider import ResponseResult

        self.calls.append((prompt, max_output_tokens))
        return ResponseResult(text=self._response_text, usage=TokenUsage(), raw={})


def test_condense_extracts_short_span_from_a_verbose_answer() -> None:
    provider = _StubProvider("yes")
    result = condense_to_answer_span(
        provider,
        "Were A and B the same nationality?",
        "A long discursive paragraph concluding yes.",
    )
    assert result == "yes"
    assert len(provider.calls) == 1


def test_condense_prompt_includes_the_shared_instruction_and_original_answer() -> None:
    provider = _StubProvider("yes")
    condense_to_answer_span(provider, "Q?", "raw answer text")
    prompt, max_tokens = provider.calls[0]
    assert CONCISE_ANSWER_INSTRUCTION in prompt
    assert "raw answer text" in prompt
    assert "Q?" in prompt
    assert max_tokens == 64


def test_condense_returns_blank_input_unchanged_without_calling_the_provider() -> None:
    provider = _StubProvider("should not be used")
    result = condense_to_answer_span(provider, "Q?", "")
    assert result == ""
    assert provider.calls == []

    result_whitespace = condense_to_answer_span(provider, "Q?", "   ")
    assert result_whitespace == "   "
    assert provider.calls == []


def test_condense_falls_back_to_raw_answer_if_the_model_returns_nothing() -> None:
    provider = _StubProvider("   ")
    result = condense_to_answer_span(provider, "Q?", "the original answer")
    assert result == "the original answer"


def test_reground_includes_the_shared_instruction_question_answer_and_context() -> None:
    provider = _StubProvider("the counterfactual entity")
    result = apply_context_authoritative_regrounding(
        provider, "Who appeared?", "the memorized entity", "context says: the counterfactual entity"
    )
    assert result == "the counterfactual entity"
    prompt, max_tokens = provider.calls[0]
    assert CONTEXT_AUTHORITATIVE_INSTRUCTION in prompt
    assert "Who appeared?" in prompt
    assert "the memorized entity" in prompt
    assert "context says: the counterfactual entity" in prompt
    assert max_tokens == 200


def test_reground_returns_blank_input_unchanged_without_calling_the_provider() -> None:
    provider = _StubProvider("should not be used")
    result = apply_context_authoritative_regrounding(provider, "Q?", "", "some context")
    assert result == ""
    assert provider.calls == []


def test_reground_falls_back_to_raw_answer_if_the_model_returns_nothing() -> None:
    provider = _StubProvider("   ")
    result = apply_context_authoritative_regrounding(provider, "Q?", "original answer", "context")
    assert result == "original answer"


def test_reground_caps_the_included_context_length() -> None:
    provider = _StubProvider("short answer")
    huge_context = "x" * (MAX_REGROUND_CONTEXT_CHARS * 2)
    apply_context_authoritative_regrounding(provider, "Q?", "answer", huge_context)
    prompt, _ = provider.calls[0]
    assert huge_context not in prompt
    assert "x" * MAX_REGROUND_CONTEXT_CHARS in prompt
