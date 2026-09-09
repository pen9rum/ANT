from __future__ import annotations

import time
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.domain import Evidence
from ant.evaluation_suite.usage import UsageStats
from ant.providers import OpenAIProvider
from ant.providers.openai_provider import _loads_json_object
from ant.tools.local import LocalSearchTool

# Fairness note (see matched_react_budget.md / the evaluation-suite audit's
# Phase F): ANT's own outer `max_rounds=6` is NOT a fair tool-call budget --
# one ANT "round" can fan out to several worker executions, each itself
# running up to 11 tool calls internally, so ANT's REAL per-question usage
# is nowhere near "6 of anything". Measured directly from 16 real ANT runs
# on these exact repos (qibo/seaborn/sphinx/yt-dlp/pennylane/sqlfluff/
# sanic/Pillow, this session's own no-evolution-runtime validation):
# mean 51.6 tool calls/question (range 19-124), mean >=41.4 LLM calls
# (lower-bound; undercounts true per-worker planning calls), mean ~303K
# tokens, mean $0.70. DEFAULT_TOOL_CALL_BUDGET below is that measured mean,
# rounded -- used only as a fallback. The preferred, more precise policy is
# per-question: when the exact same question has already been run through
# ANT and its real tool-call count is known, pass that count via
# `example.metadata["matched_react_tool_call_budget"]` (checked first in
# `run()` below) so this agent's budget matches THAT question's own real
# ANT opportunity, not a generic average -- see the smoke-test driver
# script, which sets this from the already-measured no-evolution-runtime
# validation traces for the exact 4 SWE-QA-Pro questions selected there.
DEFAULT_TOOL_CALL_BUDGET = 50

AVAILABLE_TOOLS = (
    "search",
    "dense_search",
    "navigate",
    "references",
    "callers",
    "callees",
    "assignments",
    "imports",
    "subclasses",
)

_SYSTEM_PROMPT = """You are a single autonomous agent answering a question about a \
software repository. You have direct access to the following tools over the \
repository (the same underlying navigation capability available to a \
specialized multi-agent system's own workers -- you are just one agent \
using it alone, with no team, no shared graph, no persistent memory):

- search(query): lexical/BM25 search across the repository
- dense_search(query): embedding-similarity search for paraphrase matches
- navigate(symbol): find a symbol's own definition
- references(symbol): find where a symbol is referenced
- callers(symbol): find callers of a symbol
- callees(symbol): find what a symbol calls
- assignments(symbol): find local assignments involving a symbol
- imports(symbol_or_module): resolve imports
- subclasses(symbol): find subclasses of a class

At each step, respond with ONLY a JSON object of one of these two shapes:
{{"thought": "<your reasoning>", "tool": "<tool name>", "query": "<argument>"}}
{{"thought": "<your reasoning>", "finish": "<your complete final answer, with evidence citations>"}}

You have a budget of {budget} tool calls for this question. Use them \
efficiently; finish as soon as you have enough evidence to answer well. \
No explanation outside the JSON object."""

_OBSERVATION_TEMPLATE = """Question: {question}

Tool call history so far:
{history}

You have used {used}/{budget} of your tool-call budget. Decide your next \
action (or finish)."""


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(none yet)"
    lines = []
    for entry in history:
        lines.append(f"[{entry['step']}] {entry['tool']}({entry['query']!r}):")
        for item in entry["results"][:5]:
            loc = f"{item['path']}:{item['line_start']}-{item['line_end']}"
            lines.append(f"    {loc} {item['quote'][:200]}")
        if not entry["results"]:
            lines.append("    (no results)")
    return "\n".join(lines)


class MatchedReActAgent:
    """Tier 3 / Matched ReAct: ANT's primary causal controlled baseline.

    One single agent, one continuous Thought-Action-Observation loop, no
    specialized WorkerCards, no Need Graph, no runtime graph revision, no
    multi-worker coordination, no explicit progress/recovery state, no
    local exhaustion, no cross-task memory -- exactly the negative-space
    Phase F specifies. The agent sees the WHOLE repository (every file
    ANT's own worker population collectively covers, not a
    territory-scoped slice) and has access to the SAME underlying tool
    implementations ANT's own AutonomousWorker uses (`ant.tools.local.
    LocalSearchTool`, the identical class, not a reimplementation) --
    matching "same underlying lexical/dense/symbol-navigation capability"
    as literally as this codebase allows.

    `tool_call_budget`, not `max_rounds`, is this agent's real compute
    ceiling -- see the module docstring above for why, and why its default
    is a measured mean from real ANT usage, not an arbitrary number.
    """

    name = "matched_react"

    def __init__(self, model: str = "gpt-4.1", tool_call_budget: int | None = None) -> None:
        self.model = model
        self.tool_call_budget = tool_call_budget or DEFAULT_TOOL_CALL_BUDGET

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        provider = OpenAIProvider(model=self.model)
        tools = LocalSearchTool(environment_root)
        all_files = [
            str(p.relative_to(environment_root))
            for p in environment_root.rglob("*")
            if p.is_file() and ".git" not in p.parts
        ]
        budget = example.metadata.get("matched_react_tool_call_budget") or self.tool_call_budget
        started = time.time()
        history: list[dict] = []
        llm_calls = 0
        final_answer = ""
        termination_reason = "budget_exhausted"

        system_prompt = _SYSTEM_PROMPT.format(budget=budget)
        for step in range(budget):
            observation = _OBSERVATION_TEMPLATE.format(
                question=example.question,
                history=_format_history(history),
                used=step,
                budget=budget,
            )
            response = provider.responses_json(
                system_prompt + "\n\n" + observation, max_output_tokens=512
            )
            llm_calls += 1
            decision = _loads_json_object(response.text)

            finish_text = decision.get("finish")
            if isinstance(finish_text, str) and finish_text.strip():
                final_answer = finish_text.strip()
                termination_reason = "agent_declared_finish"
                break

            tool_name = decision.get("tool")
            query = decision.get("query", "")
            if tool_name not in AVAILABLE_TOOLS or not isinstance(query, str) or not query.strip():
                # Malformed step -- counts against budget (a real ReAct
                # agent's own confusion is real, comparable overhead, not
                # something to silently retry for free) but doesn't crash
                # the run.
                history.append(
                    {"step": step, "tool": str(tool_name), "query": str(query), "results": []}
                )
                continue

            method = getattr(tools, tool_name)
            results = method(query, all_files, limit=6) if tool_name != "navigate" else (
                tools.resolve_symbol(query, all_files, limit=6, need=example.question)
                or tools.navigate(query, all_files, limit=6)
            )
            history.append(
                {
                    "step": step,
                    "tool": tool_name,
                    "query": query,
                    "results": [item.model_dump() for item in results],
                }
            )

        if not final_answer:
            # Ran out of budget without an explicit finish -- force one
            # final synthesis call from whatever evidence was gathered,
            # same "an explicit abstention beats silence" posture ANT's
            # own ask() applies (see local.py's own blank-answer guard).
            all_evidence = [item for entry in history for item in entry["results"]]
            final_answer = provider.synthesize(
                question=example.question,
                evidence=[Evidence.model_validate(e) for e in all_evidence],
            )
            llm_calls += 1

        token_usage = provider.drain_usage()
        elapsed = time.time() - started
        all_evidence = [item for entry in history for item in entry["results"]]

        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=final_answer,
            trajectory=history,
            evidence=all_evidence,
            usage=UsageStats(
                llm_calls=llm_calls,
                tool_calls=len(history),
                input_tokens=token_usage.input_tokens,
                output_tokens=token_usage.output_tokens,
                total_tokens=token_usage.total_tokens,
                estimated_cost_usd=token_usage.estimated_cost_usd,
                wall_clock_seconds=elapsed,
                unique_files_inspected=len({item["path"] for item in all_evidence}),
            ),
            termination_reason=termination_reason,
            metadata={"tool_call_budget": budget, "generation_model": self.model},
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(MatchedReActAgent())
