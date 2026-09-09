from __future__ import annotations

from pydantic import BaseModel


class UsageStats(BaseModel):
    """Benchmark-agnostic resource accounting for one (agent, task) run.

    ANT's own `TokenUsage` (ant.domain) already covers tokens/cost/latency
    for a single synthesizer call; this is the superset an evaluation-suite
    comparison actually needs -- LLM/tool call *counts* (not just tokens,
    since a method's number of round-trips is itself a fairness dimension
    -- see the Matched ReAct budget-fairness requirement) and, for
    repository tasks specifically, how much of the repo was actually
    touched. `unique_files_inspected`/`unique_symbols_inspected` default to
    0 and are simply left at 0 for non-repository benchmarks rather than
    becoming Optional -- a web-benchmark run has nothing meaningful to put
    there, and 0 reads correctly as "not applicable to this task" without
    needing a separate has-repo flag.
    """

    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    wall_clock_seconds: float = 0.0
    unique_files_inspected: int = 0
    unique_symbols_inspected: int = 0

    def __add__(self, other: UsageStats) -> UsageStats:
        return UsageStats(
            llm_calls=self.llm_calls + other.llm_calls,
            tool_calls=self.tool_calls + other.tool_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            estimated_cost_usd=self.estimated_cost_usd + other.estimated_cost_usd,
            wall_clock_seconds=self.wall_clock_seconds + other.wall_clock_seconds,
            unique_files_inspected=max(
                self.unique_files_inspected, other.unique_files_inspected
            ),
            unique_symbols_inspected=max(
                self.unique_symbols_inspected, other.unique_symbols_inspected
            ),
        )
