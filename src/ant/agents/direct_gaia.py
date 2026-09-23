"""Direct (closed-book) baseline on GAIA -- `name = "direct_gaia"`.

Mirrors `ant.agents.direct.DirectAgent`'s own contract exactly: the model
sees only the question, no environment access of any kind -- no web
search, no attachment, no sandbox. `environment_root` is accepted (the
`AgentAdapter` protocol) but never read. This is GAIA's hardest possible
condition: every Level-2/3 task that needs an attachment or a live web
lookup is, by construction, unanswerable here -- that gap IS the
baseline's whole point, not a bug to work around.

Uses `gaia_shared.GAIA_ANSWER_FORMAT_INSTRUCTIONS` (the official
`FINAL ANSWER:` template) rather than `ant.evaluation.baseline_tiers.
TIER1_PROMPT` (which `DirectAgent` uses for repo-QA) -- GAIA's scorer
extracts an answer by looking for that literal template, so every one of
this suite's 7 GAIA methods threads it through its own answer-eliciting
prompt; see `gaia_shared`'s own module docstring.
"""

from __future__ import annotations

import time
from pathlib import Path

from ant.agents.base import AgentResult
from ant.agents.gaia_shared import build_answer_prompt
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.usage import UsageStats


class DirectGaiaAgent:
    name = "direct_gaia"

    def __init__(self, model: str = "gpt-4.1") -> None:
        self.model = model

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        del environment_root  # deliberately unused -- Direct has no environment access
        provider = CountingOpenAIProvider(model=self.model)
        prompt = build_answer_prompt(example.question, "")
        started = time.time()
        result = provider.responses_text(prompt, max_output_tokens=1024)
        token_usage = provider.drain_usage()
        llm_calls = provider.drain_call_count()
        elapsed = time.time() - started
        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=result.text.strip(),
            trajectory=[{"prompt": prompt}],
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

register_agent(DirectGaiaAgent())
