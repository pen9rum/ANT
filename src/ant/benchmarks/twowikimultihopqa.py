"""2WikiMultihopQA (default config, validation split). See
docs/long_context_dataset_audit.md for the full source audit: same
context/supporting_facts schema as HotpotQA (confirmed field-for-field
live, not assumed), validation used because this mirror's test split has
empty gold answers, and `framolfese/2WikiMultihopQA` used instead of
`xanhho/2WikiMultihopQA` (which has the same deprecated-loading-script
problem as LongBench) or LongBench itself.
"""
from __future__ import annotations

from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks._hotpot_style import (
    load_hotpot_style_examples,
    prepare_hotpot_style_environment,
    score_hotpot_style,
)
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.registry import register_benchmark
from ant.evaluation_suite.scoring import MetricResult

HF_PATH = "framolfese/2WikiMultihopQA"
HF_CONFIG = "default"
HF_SPLIT = "validation"


class TwoWikiMultihopQaAdapter:
    name = "2wikimultihopqa"

    def load_examples(
        self, limit: int | None = None, repo_filter: str | None = None
    ) -> list[TaskExample]:
        return load_hotpot_style_examples(
            benchmark_name=self.name,
            hf_path=HF_PATH,
            hf_config=HF_CONFIG,
            split=HF_SPLIT,
            limit=limit,
        )

    def prepare_environment(self, example: TaskExample) -> Path:
        return prepare_hotpot_style_environment(example, benchmark_name=self.name)

    def score(self, example: TaskExample, result: AgentResult) -> MetricResult:
        return score_hotpot_style(example, result, benchmark_name=self.name)


_ADAPTER = TwoWikiMultihopQaAdapter()
register_benchmark(_ADAPTER)
