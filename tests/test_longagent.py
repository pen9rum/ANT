"""Tests for the LongAgent repository adapter. No real API calls are made
here -- CountingOpenAIProvider.responses_json is monkeypatched with a
deterministic, scripted sequence of leader/member decisions, following the
same monkeypatch-the-boundary convention used by test_sweqa_pro_native_agent.py
(mock the one external call site, exercise everything else for real).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ant.benchmarks.base import TaskExample
from ant.external_wrappers import longagent as longagent_module
from ant.external_wrappers.longagent import (
    LongAgentAdapter,
    chunk_document,
    serialize_repository,
)
from ant.providers.openai_provider import ResponseResult


def _make_repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (tmp_path / "src" / "b.py").write_text("def beta():\n    return 2\n", encoding="utf-8")
    return tmp_path


def test_serialize_repository_is_deterministically_ordered_and_marks_file_boundaries(
    tmp_path: Path,
) -> None:
    root = _make_repo(tmp_path)

    text, n_files = serialize_repository(root)

    assert n_files == 2
    # Stable sorted-relative-path order: src/a.py before src/b.py.
    assert text.index("===== FILE: src\\a.py =====") < text.index("===== FILE: src\\b.py =====") \
        or text.index("===== FILE: src/a.py =====") < text.index("===== FILE: src/b.py =====")
    assert "def alpha" in text
    assert "def beta" in text


def test_serialize_repository_preserves_file_text_verbatim_no_summarization(
    tmp_path: Path,
) -> None:
    root = _make_repo(tmp_path)
    text, _ = serialize_repository(root)
    assert "def alpha():\n    return 1" in text


def test_chunk_document_splits_at_the_requested_token_count() -> None:
    # 50 repeated words is comfortably more than 10 tokens under any BPE
    # encoding; splitting at chunk_size=10 must yield more than one chunk.
    text = "word " * 200
    chunks = chunk_document(text, chunk_size_tokens=10)
    assert len(chunks) > 1


def test_chunk_document_single_chunk_when_text_is_smaller_than_chunk_size() -> None:
    chunks = chunk_document("a short document", chunk_size_tokens=2000)
    assert len(chunks) == 1


def test_chunk_document_empty_text_yields_one_empty_chunk() -> None:
    assert chunk_document("", chunk_size_tokens=2000) == [""]


class _ScriptedProvider:
    """Stands in for CountingOpenAIProvider: returns pre-scripted JSON
    responses in sequence, one per responses_json call, and implements the
    same drain_* read-and-reset accounting contract."""

    def __init__(self, scripted_texts: list[str]) -> None:
        self._scripted = list(scripted_texts)
        self._calls = 0

    def responses_json(self, prompt: str, max_output_tokens: int = 512) -> ResponseResult:
        self._calls += 1
        from ant.domain import TokenUsage

        fallback = '{"type": "answer", "content": "fallback"}'
        text = self._scripted.pop(0) if self._scripted else fallback
        return ResponseResult(text=text, usage=TokenUsage(), raw={})

    def drain_call_count(self) -> int:
        count = self._calls
        self._calls = 0
        return count

    def drain_usage(self):
        from ant.domain import TokenUsage

        return TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2, estimated_cost_usd=0.0)

    def drain_retry_log(self) -> list[dict]:
        return []


def _run_with_scripted_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scripted_texts: list[str], **adapter_kwargs
):
    root = _make_repo(tmp_path)
    provider = _ScriptedProvider(scripted_texts)
    monkeypatch.setattr(longagent_module, "CountingOpenAIProvider", lambda model: provider)
    adapter = LongAgentAdapter(**adapter_kwargs)
    example = TaskExample(
        benchmark="test", task_id="t1", question="What does alpha do?", reference=""
    )
    result = adapter.run(example, root)
    return result, provider


def test_answer_action_terminates_immediately_with_leader_answer_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # First leader call (history empty) is scripted as an immediate answer.
    # The adapter's own loop always issues the FIRST call as if it can only
    # be new_state (per the initial-round prompt), but the leader's own
    # text could in principle still say "answer" -- the parser must honor
    # whatever the leader actually returned, not force new_state.
    result, _ = _run_with_scripted_provider(
        monkeypatch,
        tmp_path,
        ['{"type": "answer", "content": "alpha returns 1"}'],
    )
    assert result.termination_reason == "leader_answer"
    assert result.final_answer == "alpha returns 1"
    assert result.metadata["leader_calls"] == 1
    assert result.metadata["member_calls"] == 0
    assert result.metadata["new_state_used"] is False


def test_new_state_broadcasts_to_every_member_then_leader_can_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    n_members_expected = 1  # tiny repo -> 1 chunk at the default 2000-token size
    scripted = [
        '{"type": "new_state", "content": "Find what alpha does"}',
        '{"type": "response", "content": "alpha returns 1"}',  # member response
        '{"type": "answer", "content": "alpha returns 1"}',
    ]
    result, _ = _run_with_scripted_provider(monkeypatch, tmp_path, scripted)
    assert result.termination_reason == "leader_answer"
    assert result.metadata["leader_rounds"] == 1
    assert result.metadata["member_calls"] == n_members_expected
    assert result.metadata["new_state_used"] is True
    assert result.metadata["n_members"] == n_members_expected


def test_conflict_action_shares_chunks_pairwise_and_does_not_advance_round_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Force exactly 2 members deterministically (real tokenization of a
    # real tiny repo won't reliably land on exactly 2 chunks for any
    # chosen chunk_size_tokens) by monkeypatching chunk_document itself.
    monkeypatch.setattr(
        longagent_module,
        "chunk_document",
        lambda text, chunk_size_tokens: ["chunk-0-text", "chunk-1-text"],
    )
    scripted = [
        '{"type": "new_state", "content": "Find what alpha and beta do"}',
        '{"type": "response", "content": "alpha returns 1"}',
        '{"type": "response", "content": "alpha returns 2"}',  # conflicting member response
        '{"type": "conflict", "content": "members disagree", "conflict_member_ids": [0, 1]}',
        '{"type": "response", "content": "confirmed: alpha returns 1"}',
        '{"type": "response", "content": "confirmed: alpha returns 1"}',
        '{"type": "answer", "content": "alpha returns 1"}',
    ]
    result, _ = _run_with_scripted_provider(monkeypatch, tmp_path, scripted)
    assert result.metadata["conflict_used"] is True
    assert result.metadata["conflict_events"] == 1
    # leader_rounds must NOT have advanced for the conflict step itself --
    # only new_state increments it.
    assert result.metadata["leader_rounds"] == 1
    assert result.termination_reason == "leader_answer"


def test_degenerate_conflict_decision_is_skipped_not_crashed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A leader claiming "conflict" with fewer than 2 member ids (or before
    # any round has happened) is a parser-defect-adjacent malformed
    # decision, not a real conflict -- must be skipped gracefully, not
    # crash the run.
    scripted = [
        '{"type": "conflict", "content": "bogus", "conflict_member_ids": [0]}',
        '{"type": "answer", "content": "done"}',
    ]
    result, _ = _run_with_scripted_provider(monkeypatch, tmp_path, scripted)
    assert result.termination_reason == "leader_answer"
    assert result.final_answer == "done"


def test_round_budget_exhausted_forces_a_final_answer_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Every leader decision is new_state forever -- never answers on its
    # own. With max_leader_decisions=2, the loop must force-terminate and
    # make one additional forced-answer call.
    scripted = [
        '{"type": "new_state", "content": "round 1"}',
        '{"type": "response", "content": "r1"}',
        '{"type": "new_state", "content": "round 2"}',
        '{"type": "response", "content": "r2"}',
        '{"type": "answer", "content": "forced final answer"}',
    ]
    result, _ = _run_with_scripted_provider(
        monkeypatch, tmp_path, scripted, max_leader_decisions=2
    )
    assert result.termination_reason == "round_budget_exhausted"
    assert result.final_answer == "forced final answer"
    # 2 leader decisions (both new_state) + 1 forced-answer call = 3.
    assert result.metadata["leader_calls"] == 3


def test_malformed_json_from_a_member_does_not_crash_the_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scripted = [
        '{"type": "new_state", "content": "go"}',
        "this is not json at all",  # malformed member response
        '{"type": "answer", "content": "done anyway"}',
    ]
    result, _ = _run_with_scripted_provider(monkeypatch, tmp_path, scripted)
    assert result.final_answer == "done anyway"
    assert result.termination_reason == "leader_answer"


def test_metadata_always_discloses_the_paper_faithful_label_and_no_upstream_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, _ = _run_with_scripted_provider(
        monkeypatch, tmp_path, ['{"type": "answer", "content": "x"}']
    )
    assert result.metadata["implementation_label"] == "paper-faithful LongAgent adaptation"
    assert result.metadata["upstream_commit"] is None
    assert "runnable LongAgent implementation exists" in result.metadata["upstream_repo_note"]


def test_unique_files_inspected_equals_the_full_eligible_file_universe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # LongAgent is a static-partition baseline: it always serializes every
    # eligible file, never a relevance-filtered subset -- unlike Retrieval/
    # Matched ReAct, whose unique_files_inspected reflects only what was
    # actually searched/read.
    result, _ = _run_with_scripted_provider(
        monkeypatch, tmp_path, ['{"type": "answer", "content": "x"}']
    )
    assert result.usage.unique_files_inspected == 2  # src/a.py, src/b.py
