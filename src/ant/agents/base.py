from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.usage import UsageStats


class AgentResult(BaseModel):
    """Benchmark- and method-agnostic output of one agent run against one
    TaskExample, per the evaluation-suite audit's canonical schema.

    `trajectory`/`evidence` are deliberately untyped (`list[dict]`) rather
    than importing ANT's own `PlanningRound`/`Evidence` here -- forcing
    every method's trace (a single Direct-tier LLM call, a Retrieval-tier
    search loop, ANT's own multi-round Need Graph trajectory, a third-party
    agent's own trace shape) through one fixed structure would either lose
    fidelity or force ANT-specific concepts onto methods that don't have
    them. Each AgentAdapter is responsible for producing a faithful,
    JSON-serializable dump of whatever its own method actually did; nothing
    downstream (scoring, the fairness report) depends on trajectory/evidence
    having a specific shape, only on `final_answer` and `usage`.
    """

    benchmark: str
    task_id: str
    method: str
    final_answer: str
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    usage: UsageStats = Field(default_factory=UsageStats)
    termination_reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentAdapter(Protocol):
    """One implementation per method (direct.py, retrieval.py,
    matched_react.py, ant_adapter.py, external_wrappers/*.py). Each adapter
    owns its own model/tool/environment wiring -- the harness only ever
    calls `run` and reads back an AgentResult, never reaches into a
    method's internals.
    """

    name: str

    def run(self, example: TaskExample, environment_root: Any) -> AgentResult: ...
