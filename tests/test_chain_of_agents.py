"""Tests for the Chain-of-Agents (CoA) adapter
(ant.external_wrappers.chain_of_agents). No real API calls: the LLM call
site (CountingOpenAIProvider) is monkeypatched with a deterministic
recording stub, the same mock-the-external-boundary convention used by
test_longagent.py. Per the governing spec's Part D, these MUST pass
before any paid smoke test is run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tiktoken

from ant.benchmarks.base import TaskExample
from ant.external_wrappers import chain_of_agents as coa_module
from ant.external_wrappers.chain_of_agents import (
    MAX_CU_OUTPUT_TOKENS,
    SAFETY_MARGIN_TOKENS,
    ChainOfAgentsAdapter,
    build_chunks,
    compute_chunk_token_budget,
    split_into_sentences,
    verify_full_coverage,
)
from ant.external_wrappers.longagent import serialize_repository

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text(
        "Alpha bravo charlie delta echo foxtrot golf hotel india juno.\n"
        "Kilo lima mike november oscar papa quebec romeo sierra tango.\n"
        "Uniform victor whiskey xray yankee zulu alpha beta gamma delta.\n",
        encoding="utf-8",
    )
    return tmp_path


def _make_isolation_repo(tmp_path: Path) -> Path:
    # Three short, clearly distinct sentences, each comfortably under the
    # isolation tests' own target per-chunk budget on its own, so normal
    # greedy packing (not the disclosed oversized-sentence hard-split
    # fallback, exercised separately) produces one sentence per chunk.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text(
        "Alpha bravo charlie.\nDelta echo foxtrot.\nGolf hotel india.\n",
        encoding="utf-8",
    )
    return tmp_path


# ===========================================================================
# 1/2/3/9/10: pure chunking invariants -- no API calls, no mocking needed.
# ===========================================================================


def test_split_into_sentences_preserves_word_content() -> None:
    text = "First sentence here. Second sentence follows! Third one too?"
    sentences = split_into_sentences(text)
    assert "".join(sentences).replace(" ", "") != ""
    assert len(sentences) == 3


def test_build_chunks_complete_source_coverage_no_drop_no_duplicate() -> None:
    text = ("Sentence number one is here. " * 3) + ("Sentence number two follows now. " * 3)
    chunks = build_chunks(text, budget_tokens=15, encoding=_ENCODING)
    assert verify_full_coverage(text, [c.text for c in chunks])


def test_build_chunks_preserves_source_order() -> None:
    text = "AAAA marker one. BBBB marker two. CCCC marker three."
    chunks = build_chunks(text, budget_tokens=6, encoding=_ENCODING)
    combined = " ".join(c.text for c in chunks)
    assert combined.index("AAAA") < combined.index("BBBB") < combined.index("CCCC")


def test_build_chunks_is_deterministic() -> None:
    text = "One two three four five six seven eight nine ten. " * 5
    a = build_chunks(text, budget_tokens=12, encoding=_ENCODING)
    b = build_chunks(text, budget_tokens=12, encoding=_ENCODING)
    assert [c.text for c in a] == [c.text for c in b]


def test_build_chunks_respects_token_budget() -> None:
    text = "Short sentence one. Short sentence two. Short sentence three. " * 4
    budget = 10
    chunks = build_chunks(text, budget_tokens=budget, encoding=_ENCODING)
    for chunk in chunks:
        assert chunk.token_count <= budget


def test_build_chunks_oversized_single_sentence_is_hard_split_not_dropped() -> None:
    # One sentence alone exceeds the budget -- must be split, never silently
    # dropped or truncated away.
    huge_sentence = "word " * 100 + "."
    chunks = build_chunks(huge_sentence, budget_tokens=10, encoding=_ENCODING)
    assert verify_full_coverage(huge_sentence, [c.text for c in chunks])
    for chunk in chunks:
        assert chunk.token_count <= 10


def test_compute_chunk_token_budget_accounts_for_question_and_overhead() -> None:
    short_q_budget = compute_chunk_token_budget(agent_window_tokens=4000, question="Q?")
    long_q_budget = compute_chunk_token_budget(agent_window_tokens=4000, question="Q? " * 200)
    assert long_q_budget < short_q_budget


def test_compute_chunk_token_budget_raises_on_non_positive_budget() -> None:
    with pytest.raises(ValueError, match="non-positive budget"):
        compute_chunk_token_budget(agent_window_tokens=MAX_CU_OUTPUT_TOKENS, question="Q?")


# ===========================================================================
# 4/5/6/7/8/11: end-to-end run() wiring, via a recording stub provider.
# ===========================================================================


class _RecordingProvider:
    """Records every prompt it receives, in call order, and returns a
    synthetic, input-independent "CU-<call index>" response -- this makes
    it possible to assert exactly what each call DID and DID NOT receive,
    regardless of how many chunks the real budget arithmetic produces."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self._calls_since_drain = 0

    def responses_text(self, prompt: str, max_output_tokens: int = 512):
        from ant.domain import TokenUsage
        from ant.providers.openai_provider import ResponseResult

        index = len(self.prompts)
        self.prompts.append(prompt)
        self._calls_since_drain += 1
        return ResponseResult(text=f"CU-{index}", usage=TokenUsage(), raw={})

    def drain_call_count(self) -> int:
        count = self._calls_since_drain
        self._calls_since_drain = 0
        return count

    def drain_usage(self):
        from ant.domain import TokenUsage

        return TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2)

    def drain_retry_log(self) -> list[dict]:
        return []


def _small_window_for(question: str, target_budget: int) -> int:
    """Back-solves an agent_window_tokens that yields approximately
    `target_budget` tokens of chunk-text budget, using the module's own
    budget formula (never a hardcoded guess independent of it)."""
    question_tokens = len(_ENCODING.encode(question, disallowed_special=()))
    instruction_tokens = coa_module._worker_instruction_overhead_tokens(_ENCODING)
    return (
        question_tokens
        + instruction_tokens
        + MAX_CU_OUTPUT_TOKENS
        + SAFETY_MARGIN_TOKENS
        + target_budget
    )


def test_n_workers_equals_n_chunks_and_exactly_one_manager_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _make_isolation_repo(tmp_path)
    provider = _RecordingProvider()
    monkeypatch.setattr(coa_module, "CountingOpenAIProvider", lambda model: provider)
    question = "What does alpha do?"
    example = TaskExample(benchmark="test", task_id="t1", question=question, reference="")
    adapter = ChainOfAgentsAdapter(agent_window_tokens=_small_window_for(question, 9))

    result = adapter.run(example, root)

    assert result.metadata["n_workers"] == result.metadata["n_chunks"]
    assert result.metadata["n_workers"] == len(result.metadata["chunk_token_counts"])
    assert result.metadata["worker_calls"] == result.metadata["n_chunks"]
    assert result.metadata["manager_calls"] == 1


def test_worker_and_manager_prompt_isolation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _make_isolation_repo(tmp_path)
    provider = _RecordingProvider()
    monkeypatch.setattr(coa_module, "CountingOpenAIProvider", lambda model: provider)
    question = "What does alpha do?"
    example = TaskExample(benchmark="test", task_id="t1", question=question, reference="")
    window = _small_window_for(question, 9)
    adapter = ChainOfAgentsAdapter(agent_window_tokens=window)

    adapter.run(example, root)

    source_text, _ = serialize_repository(root)
    budget = compute_chunk_token_budget(agent_window_tokens=window, question=question)
    expected_chunks = build_chunks(source_text, budget, encoding=_ENCODING)
    n_chunks = len(expected_chunks)
    assert n_chunks >= 2, "test requires the fixture to actually split into >=2 chunks"

    worker_prompts = provider.prompts[:n_chunks]
    manager_prompt = provider.prompts[n_chunks]

    # 7: Wi receives its own chunk, the previous CU, and the question --
    # and NEVER a future chunk's text.
    for i, prompt in enumerate(worker_prompts):
        if expected_chunks[i].text.strip():
            assert expected_chunks[i].text in prompt
        assert question in prompt
        if i == 0:
            assert "none -- this is the first chunk" in prompt
        else:
            assert f"CU-{i - 1}" in prompt
        for j in range(i + 1, n_chunks):
            if expected_chunks[j].text.strip():
                assert expected_chunks[j].text not in prompt

    # 6: the manager receives ONLY CU_l (the last worker's output) and the
    # question -- never any raw chunk text.
    assert f"CU-{n_chunks - 1}" in manager_prompt
    assert question in manager_prompt
    for chunk in expected_chunks:
        if chunk.text.strip():
            assert chunk.text not in manager_prompt


def test_no_gold_or_needle_metadata_enters_any_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _make_repo(tmp_path)
    provider = _RecordingProvider()
    monkeypatch.setattr(coa_module, "CountingOpenAIProvider", lambda model: provider)
    secret = "SECRET_GOLD_VALUE_MUST_NEVER_APPEAR"
    example = TaskExample(
        benchmark="test",
        task_id="t1",
        question="What does alpha do?",
        reference="",
        metadata={
            "documents": [],
            "niah_metadata": {"needle_doc_ids": ["doc0"], "original_answer_replaced": secret},
            "gold_answer_secret": secret,
        },
    )
    adapter = ChainOfAgentsAdapter()
    adapter.run(example, root)

    for prompt in provider.prompts:
        assert secret not in prompt


def test_physical_calls_equals_worker_plus_manager_plus_condensation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _make_repo(tmp_path)
    provider = _RecordingProvider()
    monkeypatch.setattr(coa_module, "CountingOpenAIProvider", lambda model: provider)
    question = "What does alpha do?"
    example = TaskExample(benchmark="test", task_id="t1", question=question, reference="")
    adapter = ChainOfAgentsAdapter(apply_concise_answer_contract=True)

    result = adapter.run(example, root)

    n_chunks = result.metadata["n_chunks"]
    # +1 for the shared condense_to_answer_span call (apply_concise_answer_contract=True).
    expected = n_chunks + 1 + 1
    assert result.usage.llm_calls == expected
    assert len(provider.prompts) == expected


def test_apply_concise_answer_contract_false_skips_condensation_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _make_repo(tmp_path)
    provider = _RecordingProvider()
    monkeypatch.setattr(coa_module, "CountingOpenAIProvider", lambda model: provider)
    question = "What does alpha do?"
    example = TaskExample(benchmark="test", task_id="t1", question=question, reference="")
    adapter = ChainOfAgentsAdapter(apply_concise_answer_contract=False)

    result = adapter.run(example, root)

    n_chunks = result.metadata["n_chunks"]
    assert result.usage.llm_calls == n_chunks + 1
    assert result.final_answer == "CU-" + str(n_chunks)  # raw manager answer, uncondensed


def test_run_asserts_full_source_coverage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Regression guard: run() itself asserts verify_full_coverage right
    # after chunking -- if that ever silently regresses, this test (not
    # just the isolated build_chunks unit tests) would catch it too.
    root = _make_repo(tmp_path)
    provider = _RecordingProvider()
    monkeypatch.setattr(coa_module, "CountingOpenAIProvider", lambda model: provider)
    example = TaskExample(
        benchmark="test", task_id="t1", question="What does alpha do?", reference=""
    )
    adapter = ChainOfAgentsAdapter()
    result = adapter.run(example, root)  # must not raise
    assert result.metadata["n_chunks"] >= 1


def test_agent_is_registered() -> None:
    from ant.evaluation_suite.registry import get_agent

    agent = get_agent("chain_of_agents")
    assert isinstance(agent, ChainOfAgentsAdapter)
