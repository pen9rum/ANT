"""HotpotQA (distractor config, validation split). See
docs/long_context_dataset_audit.md for the full source audit: why
`distractor`/`validation` specifically (fullwiki's test split and
distractor's own test split both have hidden/blank answers), and why the
official `hotpotqa/hotpot_qa` release is used instead of LongBench's
flattened, boundary-losing, supporting-fact-free `context` string.
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

HF_PATH = "hotpotqa/hotpot_qa"
HF_CONFIG = "distractor"
HF_SPLIT = "validation"


class HotpotQaAdapter:
    name = "hotpotqa"

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


_ADAPTER = HotpotQaAdapter()
register_benchmark(_ADAPTER)
