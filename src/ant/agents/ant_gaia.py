"""ANTMAN over GAIA -- `name = "ant_gaia"`. Also serves as ANTMAN-H
(`worker_model`/`worker_base_url` set, same seam `ant_adapter.AntAgent`
already uses for the repo-QA worker-model bake-off).

Wires the frozen `ant.coordinator.local.LocalCoordinator` onto GAIA via
`ant.agents.gaia_tools.GaiaCoordinatorSearchTool` -- a duck-typed
`LocalSearchTool` substitute BUILT FOR EXACTLY THIS PURPOSE (see that
class's own docstring: "injected into an UNMODIFIED LocalCoordinator
through its search_tool_factory constructor parameter... That is what
makes 'ANTMAN and Matched ReAct share one substrate' a structural fact
rather than a convention"). Nothing in `LocalCoordinator` is modified;
this file and `gaia_tools.py` only supply the substrate-specific tool
implementation and the initial territory roster, exactly the same shape
`ant_web.py` already established for the WebWalkerQA substrate.

WHY A SINGLE WORKER/TERRITORY (unlike `ant_web.py`'s multi-page
bootstrap): WebWalkerQA's territory roster mirrors first-level links
discovered on a root page, because navigation there is inherently
staged (you must reach a page before its content exists). GAIA has no
such staged discovery -- `GaiaCoordinatorSearchTool.search()` already
returns BOTH live web-search hits AND attachment keyword matches in one
call (see its own docstring), so there is exactly one bounded search
capability for the whole task, and therefore exactly one territory.
ANTMAN's own Need Graph still does real coordination work here -- need
decomposition, evidence tracking, resolution checking, recovery -- over
that one worker's repeated, need-conditioned invocations; a single
territory does not mean a single tool call.

OUTPUT-FORMAT HANDLING: see `gaia_shared.reformat_to_gaia_template`'s own
docstring for why `state.answer` is reformatted in a separate, final call
rather than baking `GAIA_ANSWER_FORMAT_INSTRUCTIONS` into the question
text handed to `coordinator.ask()`.

GOLD-LEAKAGE NOTE: this file and `GaiaCoordinatorSearchTool` read only
`example.question` and task-observable metadata (`file_name`,
`has_attachment`) -- never `example.reference` or
`example.metadata["_audit_only"]`.
"""

from __future__ import annotations

import time
from pathlib import Path

from ant.agents.base import AgentResult
from ant.agents.gaia_shared import build_registry, reformat_to_gaia_template
from ant.agents.gaia_tools import GaiaCoordinatorSearchTool
from ant.benchmarks.base import TaskExample
from ant.coordinator import LocalCoordinator
from ant.domain import WorkerCard
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.usage import UsageStats
from ant.evaluation_suite.vllm_provider import VLLMChatCompletionsProvider

DEFAULT_MAX_ROUNDS = 6


class AntGaiaAgent:
    name = "ant_gaia"

    def __init__(
        self,
        model: str = "gpt-4.1",
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        worker_model: str | None = None,
        worker_base_url: str | None = None,
        worker_max_context_tokens: int | None = None,
    ) -> None:
        self.model = model
        self.max_rounds = max_rounds
        # Same worker-model bake-off seam as ant_adapter.AntAgent -- both
        # left unset (default) reproduces the frozen single-GPT-4.1
        # runtime; setting both together is ANTMAN-H, this same class,
        # never a separate one.
        self.worker_model = worker_model
        self.worker_base_url = worker_base_url
        self.worker_max_context_tokens = worker_max_context_tokens

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        started = time.time()
        registry = build_registry(example, environment_root)

        worker = WorkerCard(
            id="worker-gaia",
            territory_id="gaia-task",
            name="GAIA task (web search, attachment, sandbox)",
            root=".",
            files=[],
            responsibilities=["Search the open web and this task's own attachment (if any)."],
        )

        provider = CountingOpenAIProvider(model=self.model)

        assert (self.worker_model is None) == (self.worker_base_url is None), (
            "worker_model and worker_base_url must be set together (or both left unset)"
        )
        worker_provider = provider
        if self.worker_model is not None and self.worker_base_url is not None:
            worker_provider = VLLMChatCompletionsProvider(
                model=self.worker_model,
                base_url=self.worker_base_url,
                max_context_tokens=self.worker_max_context_tokens,
            )

        # `reasoner=worker_provider` is the fix for the gap `gaia_tools.
        # GaiaCoordinatorSearchTool`'s own class docstring explains under
        # "THE `reasoner` PARAMETER": without it, ANTMAN-H's Qwen3-8B
        # worker_provider is fully wired below (same seam ant_adapter.
        # AntAgent uses) but architecturally inert on GAIA, because this
        # substrate's rank_symbols() always returns [] so the coordinator's
        # own worker_reasoner-gated call sites never fire. Passing it here
        # instead makes the worker's own local reasoning (query
        # refinement, follow-up fetch choice) actually run on
        # worker_provider -- Qwen3-8B for ant_gaia_h, GPT-4.1 otherwise.
        tool = GaiaCoordinatorSearchTool(registry, reasoner=worker_provider)

        coordinator = LocalCoordinator(
            environment_root,
            [worker],
            reasoner=provider,
            synthesizer=provider,
            index_path=None,
            worker_reasoner=worker_provider,
            search_tool_factory=lambda root, idx: tool,
        )
        state = coordinator.ask(example.question, max_rounds=self.max_rounds)
        final_answer = reformat_to_gaia_template(provider, example.question, state.answer)

        llm_calls = provider.drain_call_count()

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

        elapsed = time.time() - started
        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=final_answer,
            trajectory=[round_.model_dump() for round_ in state.rounds],
            evidence=[item.model_dump() for item in state.evidence],
            usage=UsageStats(
                llm_calls=llm_calls,
                tool_calls=registry.tool_call_count(),
                input_tokens=state.usage.input_tokens,
                output_tokens=state.usage.output_tokens,
                total_tokens=state.usage.total_tokens,
                estimated_cost_usd=state.usage.estimated_cost_usd,
                wall_clock_seconds=elapsed,
                unique_files_inspected=len({item.path for item in state.evidence}),
            ),
            termination_reason=(
                "unresolved_needs_remain" if state.unresolved_needs else "all_needs_resolved"
            ),
            metadata={
                "generation_model": self.model,
                "final_need_graph_size": len(state.final_need_graph),
                "gaia_call_log": registry.log_as_dicts(),
                "raw_answer_before_reformat": state.answer,
                "worker_usage": worker_usage_metadata,
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(AntGaiaAgent())
