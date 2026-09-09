from __future__ import annotations

import time
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.evaluation.baseline_tiers import TIER1_PROMPT
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.usage import UsageStats


class DirectAgent:
    """Tier 1 / Direct: closed-book lower bound -- the model sees only the
    question, no repository/environment access of any kind. Reuses
    `TIER1_PROMPT` from the existing `ant.evaluation.baseline_tiers` module
    verbatim (not re-derived) so this stays byte-identical to ANT's own
    pre-existing Tier 1 baseline, just re-homed onto the generalized
    AgentAdapter interface. `environment_root` is accepted (to satisfy the
    protocol) but never read -- the whole point of this tier.
    """

    name = "direct"

    def __init__(self, model: str = "gpt-4.1") -> None:
        self.model = model

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        del environment_root  # deliberately unused -- Direct has no environment access
        provider = CountingOpenAIProvider(model=self.model)
        started = time.time()
        result = provider.responses_text(
            TIER1_PROMPT.format(question=example.question), max_output_tokens=1024
        )
        token_usage = provider.drain_usage()
        llm_calls = provider.drain_call_count()
        elapsed = time.time() - started
        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=result.text.strip(),
            trajectory=[{"prompt": TIER1_PROMPT.format(question=example.question)}],
            usage=UsageStats(
                llm_calls=llm_calls,
                input_tokens=token_usage.input_tokens,
                output_tokens=token_usage.output_tokens,
                total_tokens=token_usage.total_tokens,
                estimated_cost_usd=token_usage.estimated_cost_usd,
                wall_clock_seconds=elapsed,
            ),
            termination_reason="single_call_complete",
            metadata={"generation_model": self.model},
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(DirectAgent())
