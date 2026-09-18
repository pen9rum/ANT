from __future__ import annotations

from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.coordinator import LocalCoordinator
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.repo_scope import EvalRepoEnvironment
from ant.evaluation_suite.usage import UsageStats
from ant.evaluation_suite.vllm_provider import VLLMChatCompletionsProvider
from ant.indexing import build_worker_cards, discover_territories
from ant.memory import IndexStore


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
        self,
        model: str = "gpt-4.1",
        max_rounds: int = 6,
        index_root: Path | None = None,
        worker_model: str | None = None,
        worker_base_url: str | None = None,
        worker_max_context_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.max_rounds = max_rounds
        self.index_root = index_root or Path(".ant/eval-suite")
        # Coordination/execution model split (worker-model bake-off, see
        # ant.providers.vllm_provider's own docstring). Both left unset
        # (the default) reproduces the frozen single-GPT-4.1-provider
        # runtime exactly -- `worker_model`/`worker_base_url` only take
        # effect together; setting one without the other is a caller
        # error, asserted in run() rather than silently ignored.
        self.worker_model = worker_model
        self.worker_base_url = worker_base_url
        self.worker_max_context_tokens = worker_max_context_tokens

    def _index_path_for(self, example: TaskExample, environment_root: Path) -> Path:
        repo_slug = environment_root.name
        return self.index_root / example.benchmark / repo_slug

    def _ensure_indexed(self, environment_root: Path, index_path: Path) -> None:
        if (index_path / "workers.json").exists():
            return
        # EvalRepoEnvironment, not core's own RepoEnvironment directly --
        # supplies discover_territories the fairness-closure repo-scope
        # policy (content-based text detection, not TEXT_EXTENSIONS) so
        # ANT's own territory discovery sees the SAME expanded file
        # universe Matched ReAct/Retrieval do, before any territory
        # specialization happens. discover_territories/build_worker_cards
        # themselves are untouched, frozen core -- only the environment
        # object fed into them differs, and only in this evaluation-suite
        # call site (ant.cli/ant.evaluation.runner/ant.git_refresh all
        # still construct core's own RepoEnvironment, unaffected).
        environment = EvalRepoEnvironment(environment_root)
        territories = discover_territories(environment)
        workers = build_worker_cards(environment.root, territories)
        IndexStore(index_path).save(territories, workers)

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        index_path = self._index_path_for(example, environment_root)
        self._ensure_indexed(environment_root, index_path)
        workers = IndexStore(index_path).load_workers()
        # CountingOpenAIProvider, not core's own OpenAIProvider directly --
        # a plain subclass (frozen core untouched) that counts every
        # PHYSICAL responses_text() call. Passed as BOTH reasoner= and
        # synthesizer= exactly as before; LocalCoordinator/AutonomousWorker
        # only ever call methods on it (Protocol-typed parameters, no
        # isinstance(OpenAIProvider) check anywhere in local.py), so every
        # call any of ANT's own frozen coordination code makes through
        # this shared instance -- every worker's every reasoning call,
        # every plan_round, every consolidate_graph -- is counted too,
        # giving state.usage's own (core, TokenUsage-typed, call-count-less)
        # totals a real, accurate llm_calls figure for the first time this
        # adapter has reported one at all (previously unset, defaulting to
        # UsageStats' own 0).
        provider = CountingOpenAIProvider(model=self.model)

        assert (self.worker_model is None) == (self.worker_base_url is None), (
            "worker_model and worker_base_url must be set together (or both left unset)"
        )
        worker_provider = provider
        if self.worker_model is not None and self.worker_base_url is not None:
            # A SEPARATE provider instance, deliberately not shared with
            # `provider` above -- only AutonomousWorker's own local
            # tool-call loop (select_lookups/plan_worker_actions) is
            # routed to it, via LocalCoordinator's worker_reasoner= below.
            # Every orchestrator-level call (plan_round,
            # check_need_resolution, consolidate_graph,
            # verify_evidence_upgrade, select_evidence, observe, synthesis)
            # still goes through `provider` (GPT-4.1), unaffected.
            worker_provider = VLLMChatCompletionsProvider(
                model=self.worker_model,
                base_url=self.worker_base_url,
                max_context_tokens=self.worker_max_context_tokens,
            )

        coordinator = LocalCoordinator(
            environment_root,
            workers,
            reasoner=provider,
            synthesizer=provider,
            index_path=index_path,
            worker_reasoner=worker_provider,
            # memory_routes / cross_repo_experience deliberately omitted --
            # default [] -- no cross-task memory, matching this session's
            # own no-evolution-runtime validation exactly.
        )
        state = coordinator.ask(example.question, max_rounds=self.max_rounds)
        llm_calls = provider.drain_call_count()

        # `state.usage` only ever drains `self.synthesizer` (== `provider`,
        # the orchestrator's GPT-4.1 provider) -- see LocalCoordinator.ask's
        # final block. When a separate worker_provider is in play, its own
        # usage/call-count is otherwise never surfaced anywhere; drained
        # here and reported under metadata (not merged into `usage=` above,
        # which stays a pure orchestrator-cost figure comparable across
        # every other baseline that has no worker/orchestrator split at
        # all).
        worker_usage_metadata = None
        if worker_provider is not provider:
            worker_usage = worker_provider.drain_usage()
            worker_usage_metadata = {
                "model": self.worker_model,
                "base_url": self.worker_base_url,
                "max_context_tokens": self.worker_max_context_tokens,
                "llm_calls": worker_provider.drain_call_count(),
                "input_tokens": worker_usage.input_tokens,
                "output_tokens": worker_usage.output_tokens,
                "total_tokens": worker_usage.total_tokens,
                "wall_clock_seconds": worker_usage.latency_ms / 1000.0,
            }

        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=state.answer,
            trajectory=[round_.model_dump() for round_ in state.rounds],
            evidence=[item.model_dump() for item in state.evidence],
            usage=UsageStats(
                llm_calls=llm_calls,
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
                "worker_usage": worker_usage_metadata,
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(AntAgent())
