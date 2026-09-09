"""Harness tests for the benchmark-agnostic evaluation suite (Phase K):
schema round-trips, registry, run_suite's resume/failure-isolation
behavior. No real API calls anywhere in this file -- see
test_evaluation_suite_baseline_invariants.py for the mocked-provider
per-method invariant tests, and the smoke-test scripts (not part of the
regular test suite) for real, budget-spending runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.manifests import finish_and_save, start_manifest
from ant.evaluation_suite.registry import (
    get_agent,
    get_benchmark,
    register_agent,
    register_benchmark,
)
from ant.evaluation_suite.runner import run_suite
from ant.evaluation_suite.scoring import (
    JudgeType,
    MetricResult,
    normalize_0_100,
    run_judge_noise_protocol,
)
from ant.evaluation_suite.usage import UsageStats


def test_task_example_round_trips_through_json() -> None:
    example = TaskExample(
        benchmark="sweqa_pro",
        task_id="q1",
        question="Where is X?",
        reference="In file.py",
        metadata={"repo": "owner/repo", "commit_id": "abc123"},
    )
    restored = TaskExample.model_validate_json(example.model_dump_json())
    assert restored == example


def test_agent_result_round_trips_and_defaults_empty_usage() -> None:
    result = AgentResult(
        benchmark="sweqa_pro", task_id="q1", method="direct", final_answer="the answer"
    )
    assert result.usage == UsageStats()
    restored = AgentResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_metric_result_round_trips_with_grader_runs() -> None:
    metric = MetricResult(
        benchmark="sweqa_pro",
        task_id="q1",
        native_score=42.0,
        normalized_score=82.2,
        submetrics={"correctness": 8.0},
        grader_runs=[{"raw_text": "{}", "scores": {"correctness": 8}}],
    )
    restored = MetricResult.model_validate_json(metric.model_dump_json())
    assert restored == metric


def test_usage_stats_addition_sums_counters_and_maxes_uniques() -> None:
    a = UsageStats(llm_calls=2, tool_calls=3, total_tokens=100, unique_files_inspected=4)
    b = UsageStats(llm_calls=1, tool_calls=5, total_tokens=50, unique_files_inspected=9)
    total = a + b
    assert total.llm_calls == 3
    assert total.tool_calls == 8
    assert total.total_tokens == 150
    # unique_files_inspected is a max, not a sum -- summing would double-count
    # the same repo inspected from two different accounting sites.
    assert total.unique_files_inspected == 9


def test_registry_round_trip() -> None:
    class _StubBenchmark:
        name = "stub_benchmark_for_test"

        def load_examples(self, limit=None, repo_filter=None):
            return []

        def prepare_environment(self, example):
            return Path(".")

        def score(self, example, result):
            return MetricResult(
                benchmark=self.name, task_id=example.task_id, native_score=0, normalized_score=0
            )

    class _StubAgent:
        name = "stub_agent_for_test"

        def run(self, example, environment_root):
            return AgentResult(
                benchmark=example.benchmark,
                task_id=example.task_id,
                method=self.name,
                final_answer="stub",
            )

    register_benchmark(_StubBenchmark())
    register_agent(_StubAgent())
    assert get_benchmark("stub_benchmark_for_test").name == "stub_benchmark_for_test"
    assert get_agent("stub_agent_for_test").name == "stub_agent_for_test"


def test_manifest_records_ant_commit_and_timing(tmp_path: Path) -> None:
    manifest = start_manifest(benchmark="sweqa_pro", method="direct", example_count=4)
    assert manifest.ant_commit  # non-empty, either a real SHA or "unknown"
    assert manifest.finished_at == ""
    out_path = tmp_path / "results.jsonl"
    finish_and_save(manifest, out_path.with_suffix(".manifest.json"))
    saved = json.loads(out_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert saved["finished_at"] != ""
    assert saved["example_count"] == 4


class _FakeBenchmarkForRunner:
    name = "fake_bench"

    def __init__(self, fail_on: set[str] | None = None) -> None:
        self.fail_on = fail_on or set()
        self.prepared: list[str] = []

    def load_examples(self, limit=None, repo_filter=None):
        return []

    def prepare_environment(self, example: TaskExample) -> Path:
        self.prepared.append(example.task_id)
        if example.task_id in self.fail_on:
            raise RuntimeError(f"simulated environment failure for {example.task_id}")
        return Path(".")

    def score(self, example: TaskExample, result: AgentResult) -> MetricResult:
        return MetricResult(
            benchmark=self.name,
            task_id=example.task_id,
            native_score=1.0,
            normalized_score=100.0,
        )


class _FakeAgentForRunner:
    name = "fake_agent"

    def __init__(self) -> None:
        self.ran: list[str] = []

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        self.ran.append(example.task_id)
        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=f"answer for {example.task_id}",
        )


def _examples(ids: list[str]) -> list[TaskExample]:
    return [
        TaskExample(benchmark="fake_bench", task_id=tid, question="q", reference="r")
        for tid in ids
    ]


def test_run_suite_isolates_a_failing_example_and_continues(tmp_path: Path) -> None:
    benchmark = _FakeBenchmarkForRunner(fail_on={"q2"})
    agent = _FakeAgentForRunner()
    out_path = tmp_path / "results.jsonl"

    results = run_suite(
        benchmark=benchmark,
        agent=agent,
        examples=_examples(["q1", "q2", "q3"]),
        out_path=out_path,
        resume=False,
    )

    assert [r.task_id for r in results] == ["q1", "q2", "q3"]
    assert results[1].status.startswith("error: RuntimeError")
    assert results[1].metric.native_score == 0.0
    # q1 and q3 still ran normally -- one bad example must not sink the batch.
    assert results[0].status == "completed"
    assert results[2].status == "completed"
    written = out_path.read_text(encoding="utf-8").splitlines()
    assert len(written) == 3


def test_run_suite_resumes_and_skips_already_done_task_ids(tmp_path: Path) -> None:
    benchmark = _FakeBenchmarkForRunner()
    agent = _FakeAgentForRunner()
    out_path = tmp_path / "results.jsonl"

    run_suite(
        benchmark=benchmark, agent=agent, examples=_examples(["q1", "q2"]), out_path=out_path
    )
    assert agent.ran == ["q1", "q2"]

    # Second call, same out_path, resume=True (default): q1/q2 already
    # written -- only the new q3 should actually run the agent again.
    agent.ran.clear()
    results = run_suite(
        benchmark=benchmark,
        agent=agent,
        examples=_examples(["q1", "q2", "q3"]),
        out_path=out_path,
    )
    assert agent.ran == ["q3"]
    assert len(results) == 1
    written = out_path.read_text(encoding="utf-8").splitlines()
    assert len(written) == 3


def test_run_suite_dumps_full_trajectory_when_asked(tmp_path: Path) -> None:
    benchmark = _FakeBenchmarkForRunner()
    agent = _FakeAgentForRunner()
    dump_dir = tmp_path / "trajectories"

    run_suite(
        benchmark=benchmark,
        agent=agent,
        examples=_examples(["q1"]),
        out_path=tmp_path / "results.jsonl",
        trajectory_dump_dir=dump_dir,
        resume=False,
    )

    dumped = json.loads((dump_dir / "fake_agent-q1.json").read_text(encoding="utf-8"))
    assert dumped["final_answer"] == "answer for q1"


def test_judge_noise_protocol_deterministic_calls_exactly_once() -> None:
    calls = []

    def single_call():
        calls.append(1)
        return {"score": 7.0}

    score, runs = run_judge_noise_protocol(
        judge_type=JudgeType.DETERMINISTIC,
        single_call=single_call,
        native_score_from_call=lambda r: r["score"],
    )
    assert score == 7.0
    assert len(calls) == 1
    assert len(runs) == 1


def test_judge_noise_protocol_scalar_averages_three_calls() -> None:
    values = iter([6.0, 8.0, 10.0])

    def single_call():
        return {"score": next(values)}

    score, runs = run_judge_noise_protocol(
        judge_type=JudgeType.LLM_SCALAR,
        single_call=single_call,
        native_score_from_call=lambda r: r["score"],
    )
    assert score == 8.0  # mean of 6, 8, 10
    assert len(runs) == 3


def test_judge_noise_protocol_binary_majority_vote() -> None:
    values = iter([1.0, 0.0, 1.0])

    def single_call():
        return {"correct": next(values)}

    score, runs = run_judge_noise_protocol(
        judge_type=JudgeType.LLM_BINARY,
        single_call=single_call,
        native_score_from_call=lambda r: r["correct"],
    )
    assert score == 1.0  # 2 of 3 calls said correct
    assert len(runs) == 3
    # judge-1-only comparability is always recoverable from grader_runs[0].
    assert runs[0]["correct"] == 1.0


def test_normalize_0_100_is_clamped_and_linear() -> None:
    assert normalize_0_100(5.0, native_min=5.0, native_max=50.0) == 0.0
    assert normalize_0_100(50.0, native_min=5.0, native_max=50.0) == 100.0
    assert normalize_0_100(27.5, native_min=5.0, native_max=50.0) == 50.0
    # Out-of-range native scores never produce an out-of-range display value.
    assert normalize_0_100(1000.0, native_min=5.0, native_max=50.0) == 100.0
    assert normalize_0_100(-1000.0, native_min=5.0, native_max=50.0) == 0.0
