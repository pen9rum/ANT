from __future__ import annotations

import time
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.environment import RepoEnvironment
from ant.evaluation.baseline_tiers import _TIER2_MAX_ROUNDS, _TIER2_QUERY_PROMPT
from ant.evaluation_suite.usage import UsageStats
from ant.providers import OpenAIProvider
from ant.providers.openai_provider import _loads_json_object
from ant.tools.local import LocalSearchTool


class RetrievalAgent:
    """Tier 2 / Retrieval: simple repository search access, no long-horizon
    autonomous exploration, no Need Graph, no multi-agent coordination --
    isolates the benefit of retrieval alone, decoupled from coordination
    architecture. Reuses `_TIER2_QUERY_PROMPT`/`_TIER2_MAX_ROUNDS` from
    `ant.evaluation.baseline_tiers` verbatim (not re-derived), so the exact
    retrieval budget (basic lexical search() only, up to 3 rounds of
    LLM-decided query refinement, no dense_search/navigate/callers/
    references/subclasses) matches ANT's own pre-existing Tier 2 baseline
    exactly, just re-homed onto the generalized AgentAdapter interface.

    `environment_root` here is a plain repository root path -- this tier's
    "retrieval budget" is exactly `_TIER2_MAX_ROUNDS` lexical search() calls
    plus one synthesis call; that is the documented, defensible, benchmark-
    neutral definition of "Retrieval" this evaluation suite uses everywhere
    a repository is the environment.
    """

    name = "retrieval"

    def __init__(self, model: str = "gpt-4.1") -> None:
        self.model = model

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        provider = OpenAIProvider(model=self.model)
        search_tool = LocalSearchTool(environment_root)
        # Same scope ANT's own territory discovery uses (RepoEnvironment's
        # IGNORED_DIRS + TEXT_EXTENSIONS allowlist), not a bespoke rglob --
        # see matched_react.py's own comment at the same call for why a
        # ".git"-only exclusion was an incomplete, unfair scope definition.
        environment = RepoEnvironment(environment_root)
        all_files = [str(path.relative_to(environment.root)) for path in environment.iter_files()]
        started = time.time()
        evidence = []
        trajectory: list[dict] = []
        query = example.question
        llm_calls = 0
        tool_calls = 0
        for round_index in range(_TIER2_MAX_ROUNDS):
            results = search_tool.search(query, all_files, limit=8)
            tool_calls += 1
            evidence.extend(results)
            evidence_text = "\n".join(
                f"[{item.path}:{item.line_start}-{item.line_end}] {item.quote[:300]}"
                for item in evidence[-8:]
            )
            decision_result = provider.responses_json(
                _TIER2_QUERY_PROMPT.format(
                    question=example.question,
                    round_index=round_index + 1,
                    evidence_text=evidence_text or "(no results yet)",
                ),
                max_output_tokens=256,
            )
            llm_calls += 1
            decision = _loads_json_object(decision_result.text)
            trajectory.append({"round": round_index, "query": query, "decision": decision})
            if decision.get("enough") is True:
                break
            next_query = decision.get("next_query")
            if not isinstance(next_query, str) or not next_query.strip():
                break
            query = next_query

        answer = provider.synthesize(question=example.question, evidence=evidence)
        llm_calls += 1
        token_usage = provider.drain_usage()
        elapsed = time.time() - started

        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=answer,
            trajectory=trajectory,
            evidence=[item.model_dump() for item in evidence],
            usage=UsageStats(
                llm_calls=llm_calls,
                tool_calls=tool_calls,
                input_tokens=token_usage.input_tokens,
                output_tokens=token_usage.output_tokens,
                total_tokens=token_usage.total_tokens,
                estimated_cost_usd=token_usage.estimated_cost_usd,
                wall_clock_seconds=elapsed,
                unique_files_inspected=len({item.path for item in evidence}),
            ),
            termination_reason="round_budget_exhausted_or_declared_enough",
            metadata={"generation_model": self.model},
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(RetrievalAgent())
