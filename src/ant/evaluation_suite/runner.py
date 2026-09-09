from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel

from ant.agents.base import AgentAdapter, AgentResult
from ant.benchmarks.base import BenchmarkAdapter, TaskExample
from ant.evaluation_suite.manifests import finish_and_save, start_manifest
from ant.evaluation_suite.scoring import MetricResult
from ant.evaluation_suite.usage import UsageStats


class SuiteResult(BaseModel):
    """One row of a run_suite() call -- the generalized counterpart of
    ant.evaluation.runner.BatchResult, covering any (benchmark, method)
    pair rather than only ANT-on-SWE-QA-Pro."""

    benchmark: str
    task_id: str
    method: str
    final_answer: str
    metric: MetricResult
    usage: UsageStats
    status: str = "completed"
    elapsed_seconds: float = 0.0


def run_suite(
    *,
    benchmark: BenchmarkAdapter,
    agent: AgentAdapter,
    examples: list[TaskExample],
    out_path: Path,
    trajectory_dump_dir: Path | None = None,
    resume: bool = True,
) -> list[SuiteResult]:
    """Benchmark-agnostic counterpart of ant.evaluation.runner.run_batch:
    for each example, prepare the benchmark's own environment, run the
    agent, score with the benchmark's own native scorer, and persist one
    JSONL row per example -- same per-example failure isolation as the
    existing run_batch (one bad example must not sink the whole run), same
    resume-by-already-written-task_id convention as gen_compare.py's own
    `_load_or_rebuild_scores`.

    `trajectory_dump_dir`, when given, also writes each example's full
    AgentResult JSON to `trajectory_dump_dir / f"{method}-{task_id}.json"`
    -- the evaluation-suite's own version of run_batch's `state_dump_dir`.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already_done: set[str] = set()
    if resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                already_done.add(json.loads(line)["task_id"])
            except (json.JSONDecodeError, KeyError):
                continue

    manifest = start_manifest(
        benchmark=benchmark.name, method=agent.name, example_count=len(examples)
    )

    results: list[SuiteResult] = []
    mode = "a" if resume and out_path.exists() else "w"
    with out_path.open(mode, encoding="utf-8") as handle:
        for example in examples:
            if example.task_id in already_done:
                continue
            started = time.time()
            try:
                environment_root = benchmark.prepare_environment(example)
                agent_result = agent.run(example, environment_root)
                metric = benchmark.score(example, agent_result)
                status = "completed"
            except Exception as exc:  # noqa: BLE001 - one bad example must not sink the run
                agent_result = AgentResult(
                    benchmark=benchmark.name,
                    task_id=example.task_id,
                    method=agent.name,
                    final_answer="",
                    termination_reason=f"error: {type(exc).__name__}: {exc}",
                )
                metric = MetricResult(
                    benchmark=benchmark.name,
                    task_id=example.task_id,
                    native_score=0.0,
                    normalized_score=0.0,
                    metadata={"error": f"{type(exc).__name__}: {exc}"},
                )
                status = f"error: {type(exc).__name__}: {exc}"

            row = SuiteResult(
                benchmark=benchmark.name,
                task_id=example.task_id,
                method=agent.name,
                final_answer=agent_result.final_answer,
                metric=metric,
                usage=agent_result.usage,
                status=status,
                elapsed_seconds=round(time.time() - started, 2),
            )
            results.append(row)
            handle.write(json.dumps(row.model_dump(), ensure_ascii=True) + "\n")
            handle.flush()

            if trajectory_dump_dir is not None:
                trajectory_dump_dir.mkdir(parents=True, exist_ok=True)
                (trajectory_dump_dir / f"{agent.name}-{example.task_id}.json").write_text(
                    json.dumps(agent_result.model_dump(), indent=2), encoding="utf-8"
                )

    finish_and_save(manifest, out_path.with_suffix(".manifest.json"))
    return results
