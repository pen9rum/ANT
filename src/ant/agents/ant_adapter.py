from __future__ import annotations

from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.coordinator import LocalCoordinator
from ant.environment import RepoEnvironment
from ant.evaluation_suite.usage import UsageStats
from ant.indexing import build_worker_cards, discover_territories
from ant.memory import IndexStore
from ant.providers import OpenAIProvider


class AntAgent:
    """Thin adapter around the frozen, method-frozen clean ANT runtime
    (`LocalCoordinator.ask()`) -- per Phase B, this module must never
    reimplement, fork, or duplicate any ANT coordination logic. It only:
    (1) builds a static, evolution-free index for the example's repo if one
    doesn't already exist at `index_root / benchmark / repo_slug`
    (`discover_territories`/`build_worker_cards`, no LLM cards, matching
    exactly what `ant index` does by default -- no `evolve_workers` call
    anywhere in this file, ever); (2) constructs `LocalCoordinator` with
    `memory_routes`/`cross_repo_experience` both omitted, i.e. their
    default `[]` -- the same "ordinary clean runtime" shape validated in
    this session's own no-evolution boundary tests; (3) calls `.ask()`
    unmodified; (4) translates the resulting `EvidenceState` into the
    evaluation suite's generalized `AgentResult`, losing no information
    (the full round/graph trajectory is preserved in `trajectory`, not
    summarized away).
    """

    name = "ant"

    def __init__(
        self, model: str = "gpt-4.1", max_rounds: int = 6, index_root: Path | None = None
    ) -> None:
        self.model = model
        self.max_rounds = max_rounds
        self.index_root = index_root or Path(".ant/eval-suite")

    def _index_path_for(self, example: TaskExample, environment_root: Path) -> Path:
        repo_slug = environment_root.name
        return self.index_root / example.benchmark / repo_slug

    def _ensure_indexed(self, environment_root: Path, index_path: Path) -> None:
        if (index_path / "workers.json").exists():
            return
        environment = RepoEnvironment(environment_root)
        territories = discover_territories(environment)
        workers = build_worker_cards(environment.root, territories)
        IndexStore(index_path).save(territories, workers)

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        index_path = self._index_path_for(example, environment_root)
        self._ensure_indexed(environment_root, index_path)
        workers = IndexStore(index_path).load_workers()
        provider = OpenAIProvider(model=self.model)

        coordinator = LocalCoordinator(
            environment_root,
            workers,
            reasoner=provider,
            synthesizer=provider,
            index_path=index_path,
            # memory_routes / cross_repo_experience deliberately omitted --
            # default [] -- no cross-task memory, matching this session's
            # own no-evolution-runtime validation exactly.
        )
        state = coordinator.ask(example.question, max_rounds=self.max_rounds)

        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=state.answer,
            trajectory=[round_.model_dump() for round_ in state.rounds],
            evidence=[item.model_dump() for item in state.evidence],
            usage=UsageStats(
                input_tokens=state.usage.input_tokens,
                output_tokens=state.usage.output_tokens,
                total_tokens=state.usage.total_tokens,
                estimated_cost_usd=state.usage.estimated_cost_usd,
                wall_clock_seconds=state.usage.latency_ms / 1000.0,
                tool_calls=sum(
                    len(obs.actions)
                    for round_ in state.rounds
                    for ne in round_.node_executions
                    for obs in ne.observations
                ),
                unique_files_inspected=len({item.path for item in state.evidence}),
            ),
            termination_reason=(
                "unresolved_needs_remain" if state.unresolved_needs else "all_needs_resolved"
            ),
            metadata={
                "final_need_graph_size": len(state.final_need_graph),
                "facet_rescue": state.facet_rescue.model_dump() if state.facet_rescue else None,
                "generation_model": self.model,
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(AntAgent())
