from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    # Deferred (PEP 563, via `from __future__ import annotations`): these
    # names are never evaluated at runtime, only by static type checkers --
    # avoids a real import cycle (agents/base.py needs TaskExample from this
    # module; evaluation_suite/scoring.py needs both TaskExample and
    # AgentResult). benchmarks -> agents/evaluation_suite is a forward
    # reference only, never a runtime dependency.
    from ant.agents.base import AgentResult
    from ant.evaluation_suite.scoring import MetricResult


class TaskExample(BaseModel):
    """One benchmark-agnostic question, per the evaluation-suite audit's
    canonical schema. `metadata` is deliberately an open dict rather than a
    fixed set of fields -- a repo-QA benchmark's repo/commit/checklist and a
    web benchmark's gold_url/reasoning_type are both legitimately
    benchmark-specific, and forcing them into shared top-level fields would
    either bloat every other benchmark with irrelevant None-valued fields or
    require a new schema revision per benchmark added.
    """

    benchmark: str
    task_id: str
    question: str
    reference: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkAdapter(Protocol):
    """One implementation per benchmark (sweqa_pro.py, repoprobe.py, ...).

    Deliberately does NOT prescribe how `metadata` is shaped or how
    `prepare_environment` provisions an environment -- a repo-QA benchmark
    checks out a pinned git commit; a future non-repo benchmark might
    instead build a fixed retrieval index or do nothing at all. The
    contract is only: given a TaskExample this adapter itself produced,
    return something an agent can operate against, and given an
    AgentResult, return this benchmark's own native MetricResult without
    ANT ever second-guessing the rubric.
    """

    name: str

    def load_examples(
        self, limit: int | None = None, repo_filter: str | None = None
    ) -> list[TaskExample]: ...

    def prepare_environment(self, example: TaskExample) -> Path:
        """Ensures whatever this example needs to be answered against is
        locally available (e.g. a repo checked out at a pinned commit) and
        returns its root path. Idempotent -- safe to call once per example
        even across repeated runs against the same benchmark instance."""
        ...

    def score(self, example: TaskExample, result: AgentResult) -> MetricResult:
        """Invokes this benchmark's OWN native scorer -- never a rubric
        this module invents itself. See each concrete adapter's own
        docstring for exactly which official artifact (prompt file, scorer
        module, commit) this is ported from."""
        ...
