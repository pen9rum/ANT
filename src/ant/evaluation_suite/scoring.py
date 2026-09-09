from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """Benchmark-agnostic scoring output, per the evaluation-suite audit's
    canonical schema. `native_score` and `submetrics` are always in the
    BENCHMARK'S OWN scale (e.g. SWE-QA-Pro's 5-50 rubric sum, RepoProbe's
    0-10 checklist total) -- `normalized_score` is a display-only 0-100
    transform computed from `native_score`, never a replacement rubric
    (see JudgeNoiseProtocol's own docstring for exactly how each benchmark
    type computes it). `grader_runs` holds one entry per actual judge call
    made (1 for deterministic/single-pass, 3 for the stabilized protocol),
    each a raw, unmodified record of that call -- this is what lets a
    later reader recover "judge-1 only" for literal official-leaderboard
    comparability without re-running anything.
    """

    benchmark: str
    task_id: str
    native_score: float
    normalized_score: float
    submetrics: dict[str, float] = Field(default_factory=dict)
    grader_runs: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeType(StrEnum):
    """Per the evaluation-suite audit's Part 5 findings, verified against
    each benchmark's own official evaluator code/paper -- not a guess:
    - DETERMINISTIC: AssistantBench (DROP-style F1 / log-distance / dict-F1,
      no LLM judge at all).
    - LLM_SCALAR: SWE-QA-Pro (5-axis, 1-10 each), RepoProbe (weighted
      0-10 checklist, effectively scalar output despite per-item structure).
    - LLM_BINARY: WebWalkerQA, BrowseComp-Plus, FRAMES (correct/incorrect).
    """

    DETERMINISTIC = "deterministic"
    LLM_SCALAR = "llm_scalar"
    LLM_BINARY = "llm_binary"


def run_judge_noise_protocol(
    *,
    judge_type: JudgeType,
    single_call: Callable[[], dict[str, Any]],
    native_score_from_call: Callable[[dict[str, Any]], float],
    n_scalar_calls: int = 3,
    n_binary_calls: int = 3,
) -> tuple[float, list[dict[str, Any]]]:
    """Implements the stabilization protocol from the evaluation-suite
    audit's Part 5, verified per-benchmark against official semantics:

    - DETERMINISTIC: `single_call` is invoked exactly once. There is
      nothing to stabilize (AssistantBench's scorer is pure arithmetic/
      string-matching); calling it more than once would just waste budget
      and could never change the answer, so this path deliberately never
      retries.
    - LLM_SCALAR (SWE-QA-Pro, RepoProbe): `single_call` is invoked
      `n_scalar_calls` times (default 3, matching SWE-QA-Pro's own
      documented protocol exactly); the returned native_score is the MEAN
      across calls. `grader_runs` retains all raw calls, so a caller who
      wants literal single-leaderboard-pass comparability can always read
      `grader_runs[0]` back out instead of the mean -- this function never
      discards that.
    - LLM_BINARY (WebWalkerQA, BrowseComp-Plus, FRAMES): `single_call` is
      invoked `n_binary_calls` times; the returned native_score is the
      MAJORITY-VOTE outcome (as 1.0/0.0), not a mean of some already-binary
      quantity averaged into a fake decimal -- ties (impossible at n=3,
      kept general for a caller passing an even n) fall back to the first
      call's own verdict.

    This function never regenerates the underlying model answer between
    calls -- `single_call` must close over the SAME saved answer text each
    time it's invoked; only the judge call itself repeats.
    """
    if judge_type is JudgeType.DETERMINISTIC:
        record = single_call()
        return native_score_from_call(record), [record]

    n = n_scalar_calls if judge_type is JudgeType.LLM_SCALAR else n_binary_calls
    records = [single_call() for _ in range(max(n, 1))]
    values = [native_score_from_call(record) for record in records]

    if judge_type is JudgeType.LLM_SCALAR:
        return sum(values) / len(values), records

    # LLM_BINARY: majority vote over {0.0, 1.0}-valued outcomes.
    counts = Counter(values)
    winner, winner_count = counts.most_common(1)[0]
    ties = [value for value, count in counts.items() if count == winner_count]
    if len(ties) > 1:
        return values[0], records
    return winner, records


def normalize_0_100(native_score: float, *, native_min: float, native_max: float) -> float:
    """Linear rescale of a native score into a 0-100 DISPLAY value only --
    never fed back into any decision, never a replacement for the native
    rubric (per the audit's Part 4/16: "the normalized score is only a
    display transformation, do not alter the rubric"). Clamped to [0, 100]
    so a judge's occasional out-of-range response (already validated/
    rejected upstream in normal operation) can never silently produce a
    display value outside the advertised scale.
    """
    if native_max <= native_min:
        raise ValueError("native_max must be greater than native_min")
    raw = (native_score - native_min) / (native_max - native_min) * 100.0
    return max(0.0, min(100.0, raw))
