"""Matched ReAct baseline on GAIA -- `name = "matched_react_gaia"`.

Same Tier-3 shape as `ant.agents.matched_react.MatchedReActAgent`: one
single agent, one continuous Thought-Action-Observation loop, no
WorkerCards, no Need Graph, no multi-agent coordination -- driven here by
`GaiaToolRegistry.invoke()` (search/open_url/inspect_file/inspect_table/
run_python) instead of `LocalSearchTool`, since GAIA's tools are this
substrate's actual capability surface, the same way `LocalSearchTool` is
the repo-QA track's.

WHY THE DECISION SCHEMA DIFFERS FROM THE REPO-QA VERSION: every
`LocalSearchTool` method there takes one string `query` argument, so
`{"tool": ..., "query": ...}` is a uniform shape. GAIA's five tools do
not share a signature (`search(query, limit)`, `open_url(url)`,
`inspect_file(offset, limit)`, `inspect_table(max_rows)`,
`run_python(code)`), so the decision schema here is
`{"tool": ..., "args": {...}}` with a per-tool argument dict instead --
a disclosed, substrate-driven schema change, not a fairness-relevant one
(the agent still gets exactly one JSON decision per step, same budget
mechanics).

`tool_call_budget` defaults to 50, matching
`ant.agents.matched_react.DEFAULT_TOOL_CALL_BUDGET` for consistency
across tracks -- GAIA has no prior real-ANTMAN-usage measurement to base
a substrate-specific number on the way the repo-QA default was derived,
so the existing cross-track default is kept rather than invented fresh.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ant.agents.base import AgentResult
from ant.agents.gaia_shared import GAIA_ANSWER_FORMAT_INSTRUCTIONS, build_registry
from ant.agents.gaia_tools import GAIA_TOOL_NAMES, GaiaToolRegistry
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.usage import UsageStats
from ant.providers.openai_provider import _loads_json_object

DEFAULT_TOOL_CALL_BUDGET = 50

_SYSTEM_PROMPT = """You are a single autonomous agent answering a general-knowledge \
question that may require web search, reading a linked page, reading an \
attached file, or running a short Python program. Available tools:

- search: {{"query": "...", "limit": 8}} -- search the open web
- open_url: {{"url": "..."}} -- fetch one URL's text
- inspect_file: {{"offset": 0, "limit": 20000}} -- read this task's attached file as text
- inspect_table: {{"max_rows": 200}} -- read this task's attached file as table rows
- run_python: {{"code": "..."}} -- run a short Python program in an isolated sandbox

At each step, respond with ONLY a JSON object of one of these two shapes:
{{"thought": "<your reasoning>", "tool": "<tool name>", "args": {{...}}}}
{{"thought": "<your reasoning>", "finish": "<your complete final answer>"}}

You have a budget of {budget} tool calls for this question. Use them \
efficiently; finish as soon as you have enough evidence to answer well. \
No explanation outside the JSON object.

""" + GAIA_ANSWER_FORMAT_INSTRUCTIONS

_OBSERVATION_TEMPLATE = """Question: {question}

Tool call history so far:
{history}

You have used {used}/{budget} of your tool-call budget. Decide your next \
action (or finish, with the FINAL ANSWER: template)."""


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(none yet)"
    lines = []
    for entry in history:
        lines.append(f"[{entry['step']}] {entry['tool']}({entry['args']!r}):")
        summary = entry.get("summary", "")
        lines.append(f"    {summary[:800]}" if summary else "    (no result)")
    return "\n".join(lines)


def _invoke(registry: GaiaToolRegistry, tool: str, args: dict[str, Any]) -> str:
    result = registry.invoke(tool, **args)
    if tool == "search":
        lines = [f"- {h.title} ({h.url}): {h.snippet[:300]}" for h in result]
        return "\n".join(lines) or "(no results)"
    if tool == "inspect_table":
        return result.to_text()[:2000]
    return str(result)[:4000]


class MatchedReActGaiaAgent:
    name = "matched_react_gaia"

    def __init__(self, model: str = "gpt-4.1", tool_call_budget: int | None = None) -> None:
        self.model = model
        self.tool_call_budget = tool_call_budget or DEFAULT_TOOL_CALL_BUDGET

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        provider = CountingOpenAIProvider(model=self.model)
        registry = build_registry(example, environment_root)
        budget = self.tool_call_budget
        started = time.time()
        history: list[dict] = []
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
            decision = _loads_json_object(response.text)

            finish_text = decision.get("finish")
            if isinstance(finish_text, str) and finish_text.strip():
                final_answer = finish_text.strip()
                termination_reason = "agent_declared_finish"
                break

            tool_name = decision.get("tool")
            args = decision.get("args") if isinstance(decision.get("args"), dict) else {}
            if tool_name not in GAIA_TOOL_NAMES:
                history.append({"step": step, "tool": str(tool_name), "args": args, "summary": ""})
                continue
            try:
                summary = _invoke(registry, tool_name, args)
            except Exception as exc:  # noqa: BLE001 -- a bad tool call must not crash the loop
                summary = f"(tool error: {exc!r})"
            history.append({"step": step, "tool": tool_name, "args": args, "summary": summary})

        if not final_answer:
            fallback_prompt = (
                system_prompt
                + "\n\nTool call history:\n"
                + _format_history(history)
                + f"\n\nQuestion: {example.question}\n\nGive your best final answer now."
            )
            fallback_result = provider.responses_text(fallback_prompt, max_output_tokens=1024)
            final_answer = fallback_result.text.strip()
            termination_reason = "budget_exhausted_forced_answer"

        llm_calls = provider.drain_call_count()
        token_usage = provider.drain_usage()
        elapsed = time.time() - started

        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=final_answer,
            trajectory=history,
            evidence=[{"path": e["tool"], "quote": e["summary"][:500]} for e in history],
            usage=UsageStats(
                llm_calls=llm_calls,
                tool_calls=registry.tool_call_count(),
                input_tokens=token_usage.input_tokens,
                output_tokens=token_usage.output_tokens,
                total_tokens=token_usage.total_tokens,
                estimated_cost_usd=token_usage.estimated_cost_usd,
                wall_clock_seconds=elapsed,
            ),
            termination_reason=termination_reason,
            metadata={
                "tool_call_budget": budget,
                "generation_model": self.model,
                "call_log": registry.log_as_dicts(),
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(MatchedReActGaiaAgent())
