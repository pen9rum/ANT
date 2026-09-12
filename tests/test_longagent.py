"""Tests for the LongAgent repository adapter. No real API calls are made
here -- CountingOpenAIProvider.responses_json is monkeypatched with a
deterministic, scripted sequence of leader/member decisions, following the
same monkeypatch-the-boundary convention used by test_sweqa_pro_native_agent.py
(mock the one external call site, exercise everything else for real).
"""
from __future__ import annotations

import threading
import time
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

    def responses_text(self, prompt: str, max_output_tokens: int = 512) -> ResponseResult:
        # Only ever reached by the Part A short-answer contract's
        # condensation call (apply_concise_answer_contract=True) -- every
        # scripted-order test above leaves that flag at its default False,
        # so this is never exercised except by the dedicated concise-
        # contract test below.
        self._calls += 1
        from ant.domain import TokenUsage

        return ResponseResult(text="condensed", usage=TokenUsage(), raw={})

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
    # Sequential dispatch for every scripted-order test below: these tests
    # rely on `_ScriptedProvider` handing out its texts in exact list
    # order (`list.pop(0)` per call), which only stays deterministic under
    # single-threaded dispatch against one shared mock instance. Real
    # concurrent dispatch (member_concurrency > 1) is exercised separately
    # by the dedicated concurrency tests below, using a keyed-not-ordered
    # scripted provider so no test ever depends on thread scheduling order.
    adapter_kwargs.setdefault("member_concurrency", 1)
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


class _KeyedScriptedProvider:
    """Deterministic scripted provider keyed by a substring of the PROMPT
    itself rather than by call order -- required to exercise genuine
    concurrent member dispatch (member_concurrency > 1): several member
    calls race against this one shared mock instance at once (matching
    how CountingOpenAIProvider(model=...) resolves to the same monkeypatched
    instance regardless of which thread constructs it), so no test built on
    this mock may depend on which caller happens to arrive first. An
    optional `delay_seconds` widens the concurrency window enough to
    observe genuine overlap between calls, and `max_concurrent` (tracked
    under `_lock`) records the highest number of calls actually in flight
    at once, to assert the thread pool both parallelizes AND stays within
    its configured bound.
    """

    def __init__(self, responses_by_substring: dict[str, str], delay_seconds: float = 0.0) -> None:
        self._responses_by_substring = responses_by_substring
        self._delay_seconds = delay_seconds
        self._calls = 0
        self._in_flight = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def responses_json(self, prompt: str, max_output_tokens: int = 512) -> ResponseResult:
        from ant.domain import TokenUsage

        with self._lock:
            self._calls += 1
            self._in_flight += 1
            self.max_concurrent = max(self.max_concurrent, self._in_flight)
        try:
            if self._delay_seconds:
                time.sleep(self._delay_seconds)
            for substring, text in self._responses_by_substring.items():
                if substring in prompt:
                    return ResponseResult(text=text, usage=TokenUsage(), raw={})
            return ResponseResult(
                text='{"type": "answer", "content": "fallback"}', usage=TokenUsage(), raw={}
            )
        finally:
            with self._lock:
                self._in_flight -= 1

    def drain_call_count(self) -> int:
        with self._lock:
            count = self._calls
            self._calls = 0
        return count

    def drain_usage(self):
        from ant.domain import TokenUsage

        return TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2, estimated_cost_usd=0.0)

    def drain_retry_log(self) -> list[dict]:
        return []


_FOUR_CHUNK_LEADER_SCRIPT = {
    "FIRST round": '{"type": "new_state", "content": "What does each chunk say?"}',
    "CHUNK-A": '{"type": "response", "content": "answer-from-A"}',
    "CHUNK-B": '{"type": "response", "content": "answer-from-B"}',
    "CHUNK-C": '{"type": "response", "content": "answer-from-C"}',
    "CHUNK-D": '{"type": "response", "content": "answer-from-D"}',
    "Discussion history so far": '{"type": "answer", "content": "combined answer"}',
}


def _run_four_member_round(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: _KeyedScriptedProvider,
    member_concurrency: int,
):
    root = _make_repo(tmp_path)
    monkeypatch.setattr(
        longagent_module,
        "chunk_document",
        lambda text, chunk_size_tokens: ["CHUNK-A", "CHUNK-B", "CHUNK-C", "CHUNK-D"],
    )
    monkeypatch.setattr(longagent_module, "CountingOpenAIProvider", lambda model: provider)
    adapter = LongAgentAdapter(member_concurrency=member_concurrency)
    example = TaskExample(
        benchmark="test", task_id="t1", question="What do the chunks say?", reference=""
    )
    return adapter.run(example, root)


def test_concurrent_member_dispatch_preserves_correct_per_member_mapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Real concurrency (>1) against a shared mock, keyed by prompt content
    # rather than call order -- directly verifies that bounded concurrent
    # dispatch does not scramble which member's response lands under which
    # member_id, even when several member calls are genuinely in flight
    # together (member_concurrency=4 == n_members, so all four race at once).
    provider = _KeyedScriptedProvider(dict(_FOUR_CHUNK_LEADER_SCRIPT), delay_seconds=0.01)
    result = _run_four_member_round(monkeypatch, tmp_path, provider, member_concurrency=4)

    assert result.termination_reason == "leader_answer"
    assert result.final_answer == "combined answer"
    assert result.metadata["n_members"] == 4
    assert result.metadata["member_calls"] == 4
    assert result.metadata["member_concurrency"] == 4

    members_step = next(step for step in result.trajectory if step.get("role") == "members")
    assert members_step["responses"] == {
        0: "answer-from-A",
        1: "answer-from-B",
        2: "answer-from-C",
        3: "answer-from-D",
    }


def test_concurrent_member_dispatch_respects_the_configured_concurrency_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A small delay per call widens the window in which several calls are
    # simultaneously in flight; asserting max_concurrent both (a) exceeds 1
    # (proving calls genuinely overlap in time, i.e. concurrency is real,
    # not just correctness-preserving under a sequential disguise) and
    # (b) never exceeds the configured bound (proving the bound is
    # honored, not just a label with no effect).
    provider = _KeyedScriptedProvider(dict(_FOUR_CHUNK_LEADER_SCRIPT), delay_seconds=0.05)
    _run_four_member_round(monkeypatch, tmp_path, provider, member_concurrency=2)

    assert provider.max_concurrent > 1
    assert provider.max_concurrent <= 2


def test_member_concurrency_one_is_effectively_sequential(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = _KeyedScriptedProvider(dict(_FOUR_CHUNK_LEADER_SCRIPT), delay_seconds=0.01)
    _run_four_member_round(monkeypatch, tmp_path, provider, member_concurrency=1)

    assert provider.max_concurrent == 1


def test_apply_concise_answer_contract_false_by_default_leaves_answer_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Default False preserves this class's existing repository-QA
    # behavior byte-for-byte -- no caller that doesn't explicitly opt in
    # (i.e. every existing repository-QA/document-track caller from
    # before this change) sees any difference at all.
    result, _ = _run_with_scripted_provider(
        monkeypatch, tmp_path, ['{"type": "answer", "content": "the leader'"'"'s own answer"}']
    )
    assert result.final_answer == "the leader's own answer"
    assert result.metadata["apply_concise_answer_contract"] is False


def test_apply_concise_answer_contract_true_condenses_the_final_answer_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, provider = _run_with_scripted_provider(
        monkeypatch,
        tmp_path,
        ['{"type": "answer", "content": "the leader\'s own verbose answer"}'],
        member_concurrency=1,
        apply_concise_answer_contract=True,
    )
    assert result.final_answer == "condensed"
    assert result.metadata["raw_answer_before_condensation"] == "the leader's own verbose answer"
    assert result.metadata["apply_concise_answer_contract"] is True
    # Leader/member protocol itself is untouched: still exactly one leader
    # call (the immediate "answer"), zero member calls -- condensation adds
    # ONE extra physical call on top, not a change to round semantics.
    assert result.metadata["leader_calls"] == 1
    assert result.metadata["member_calls"] == 0
