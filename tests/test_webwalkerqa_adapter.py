"""Tests for ant.benchmarks.webwalkerqa (the TaskExample/BenchmarkAdapter
layer). Dataset loading and the judge call are both monkeypatched -- zero
real network calls, zero LLM calls, per the governing spec.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ant.agents.base import AgentResult
from ant.benchmarks import webwalkerqa as webwalkerqa_module
from ant.benchmarks.webwalkerqa import WebWalkerQaAdapter
from ant.evaluation_suite.usage import UsageStats
from ant.evaluation_suite.webwalkerqa.loader import WebWalkerQaRecord

SECRET_ANSWER = "SECRET_GOLD_ANSWER_MUST_NEVER_LEAK"
SECRET_SOURCE = "SECRET_SOURCE_WEBSITE_MUST_NEVER_LEAK"
SECRET_PATH_STEP = "SECRET_GOLDEN_PATH_STEP_MUST_NEVER_LEAK"


def _fake_records() -> list[WebWalkerQaRecord]:
    return [
        WebWalkerQaRecord(
            example_id="webwalkerqa-abc123",
            question="What is the keynote speaker's affiliation?",
            gold_answer=SECRET_ANSWER,
            root_url="http://conf.example.com/",
            language="en",
            domain="conference",
            hop_type="single_source",
            difficulty="easy",
            source_websites=[SECRET_SOURCE],
            golden_path=[SECRET_PATH_STEP],
        ),
        WebWalkerQaRecord(
            example_id="webwalkerqa-def456",
            question="谁是主讲人?",
            gold_answer="something",
            root_url="http://conf.example.com/",
            language="zh",
            domain="conference",
            hop_type="single_source",
            difficulty="easy",
        ),
    ]


def _adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> WebWalkerQaAdapter:
    monkeypatch.setattr(webwalkerqa_module, "load_webwalkerqa_records", _fake_records)
    return WebWalkerQaAdapter(cache_root=tmp_path)


# --- load_examples: English-only, leakage-free ---


def test_load_examples_filters_to_english_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _adapter(monkeypatch, tmp_path)

    examples = adapter.load_examples()

    assert len(examples) == 1
    assert examples[0].task_id == "webwalkerqa-abc123"


def test_load_examples_never_exposes_gold_fields_outside_audit_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _adapter(monkeypatch, tmp_path)

    [example] = adapter.load_examples()

    assert SECRET_ANSWER not in example.question
    for key, value in example.metadata.items():
        if key == "_audit_only":
            continue
        assert SECRET_ANSWER not in json.dumps(value)
        assert SECRET_SOURCE not in json.dumps(value)
        assert SECRET_PATH_STEP not in json.dumps(value)
    # the gold fields DO exist, but only under the one clearly-marked key
    assert example.metadata["_audit_only"]["source_websites"] == [SECRET_SOURCE]
    assert example.metadata["_audit_only"]["golden_path"] == [SECRET_PATH_STEP]
    # reference carries the gold answer -- read only by .score(), never by generation
    assert example.reference == SECRET_ANSWER


def test_load_examples_metadata_carries_root_url_and_no_gold_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _adapter(monkeypatch, tmp_path)

    [example] = adapter.load_examples()

    assert example.metadata["root_url"] == "http://conf.example.com/"
    assert "gold_answer" not in example.metadata
    assert "source_websites" not in example.metadata  # only inside _audit_only
    assert "golden_path" not in example.metadata


def test_load_examples_domain_filter_reuses_repo_filter_param(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(webwalkerqa_module, "load_webwalkerqa_records", lambda: _fake_records())
    adapter = WebWalkerQaAdapter(cache_root=tmp_path)

    examples = adapter.load_examples(repo_filter="conference")
    assert len(examples) == 1

    examples_none = adapter.load_examples(repo_filter="education")
    assert examples_none == []


# --- prepare_environment: local cache dir only, no live crawl ---


def test_prepare_environment_creates_a_local_cache_dir_and_fetches_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _adapter(monkeypatch, tmp_path)
    [example] = adapter.load_examples()

    env_root = adapter.prepare_environment(example)

    assert env_root.exists()
    assert env_root.is_dir()
    assert list(env_root.iterdir()) == []  # no page was ever fetched


# --- score(): LLM_BINARY, judge only sees question/reference/candidate ---


class _FakeJudgeResult:
    def __init__(self, text: str) -> None:
        self.text = text
        self.estimated_cost_usd = 0.001


def test_score_uses_llm_binary_majority_vote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _adapter(monkeypatch, tmp_path)
    [example] = adapter.load_examples()

    calls = []

    def _fake_call_judge(*, system, user, **kwargs):
        calls.append(user)
        return _FakeJudgeResult(json.dumps({"correct": True, "reasoning": "matches"}))

    monkeypatch.setattr(webwalkerqa_module, "call_judge", _fake_call_judge)

    result = AgentResult(
        benchmark="webwalkerqa",
        task_id=example.task_id,
        method="test_method",
        final_answer="Stanford University",
        usage=UsageStats(),
    )
    metric = adapter.score(example, result)

    assert len(calls) == 3  # n_binary_calls=3
    assert metric.native_score == 1.0
    assert metric.normalized_score == 100.0
    assert metric.metadata["judge_prompt_is_official"] is False


def test_score_judge_prompt_never_contains_source_websites_or_golden_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    adapter = _adapter(monkeypatch, tmp_path)
    [example] = adapter.load_examples()

    prompts = []

    def _fake_call_judge(*, system, user, **kwargs):
        prompts.append(user)
        return _FakeJudgeResult(json.dumps({"correct": False, "reasoning": "no match"}))

    monkeypatch.setattr(webwalkerqa_module, "call_judge", _fake_call_judge)
    result = AgentResult(
        benchmark="webwalkerqa",
        task_id=example.task_id,
        method="m",
        final_answer="wrong",
        usage=UsageStats(),
    )

    adapter.score(example, result)

    for prompt in prompts:
        assert SECRET_SOURCE not in prompt
        assert SECRET_PATH_STEP not in prompt


def test_adapter_is_registered() -> None:
    from ant.evaluation_suite.registry import get_benchmark

    adapter = get_benchmark("webwalkerqa")
    assert isinstance(adapter, WebWalkerQaAdapter)
