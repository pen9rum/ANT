from __future__ import annotations

import time
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.domain import Evidence
from ant.environment import RepoEnvironment
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
# rounded.
#
# PRIMARY protocol (this is the only one `run()` implements): frozen ANT
# config vs. ONE pre-specified global Matched-ReAct cap, fixed before
# looking at any individual question's ANT behavior and applied UNIFORMLY
# to every question in a benchmark run -- `tool_call_budget` is set once,
# at construction time, for the whole run (via DEFAULT_TOOL_CALL_BUDGET or
# an explicit override), never varied per example.
#
# An EARLIER version of this file also read a per-example
# `example.metadata["matched_react_tool_call_budget"]` override inside
# run() -- a retrospective policy that gave each question a budget equal to
# THAT question's own already-measured real ANT tool-call count. That is a
# genuinely different, invalid-as-a-primary-protocol experiment (it
# conditions the baseline's compute on the very ANT run it's being compared
# against, per-question, defeating "one pre-specified cap applied
# uniformly") and has been removed from this class. It was actually
# exercised by the smoke-test driver script that produced this session's
# existing 4-question SWE-QA-Pro smoke scores (tool_call_budget=31/19 for
# sqlfluff/Pillow, taken from those questions' own real ANT traces) -- so
# those specific matched_react rows were run under the retrospective
# policy, not this primary one; re-running under the uniform cap is
# required before those numbers can be compared as the primary protocol.
# Retrospective per-question compute matching may return later as an
# explicit, separately-labeled secondary analysis (e.g. a wrapper that
# constructs a distinct MatchedReActAgent per question) -- it must never
# again be a silent per-example lookup inside the shared agent's run().
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

At each step, respond with ONLY a JSON object of one of these two shapes.
The keys must be EXACTLY "tool" and "query" -- do NOT use the tool's own \
name as a JSON key (e.g. NEVER write {{"search": "some query"}}; always \
write {{"tool": "search", "query": "some query"}}):
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


def _normalize_decision(decision: dict) -> dict:
    """Defensive tolerance for a common, reproducible GPT-4.1 schema
    deviation: instead of the prescribed {"tool": "search", "query": ...}
    shape, the model frequently writes the tool's own name as the JSON key
    -- {"thought": ..., "search": "some query"} -- despite the system
    prompt explicitly forbidding this (see _SYSTEM_PROMPT). Confirmed via
    direct reproduction (this exact prompt scaffold, repeated live calls):
    GPT-4.1 produces this shorthand on a large fraction of steps,
    independent of question/repository. Without this normalization, every
    one of those steps silently burns one unit of tool-call budget for
    zero information gain, which is a harness parsing-robustness defect
    -- not a genuine reasoning/competence failure of the agent -- and was
    making Matched ReAct a systematically crippled baseline. This does not
    inspect or depend on the question's answer in any way.
    """
    if "tool" in decision or "finish" in decision:
        return decision
    for tool_name in AVAILABLE_TOOLS:
        if tool_name in decision and isinstance(decision[tool_name], str):
            normalized = dict(decision)
            normalized["tool"] = tool_name
            normalized["query"] = normalized.pop(tool_name)
            return normalized
    return decision


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
        # Reuses ANT's own RepoEnvironment.iter_files() (core, unmodified)
        # rather than a bespoke rglob -- this is the SAME definition of
        # "the repository" ANT's own territory discovery uses (IGNORED_DIRS
        # + a TEXT_EXTENSIONS allowlist, not just ".git"). A prior version
        # of this listing excluded only ".git", which left Matched ReAct
        # with a DIFFERENT (both noisier -- binary files like .png, whose
        # raw bytes could be handed to search() as evidence -- and, for
        # some extensions ANT's allowlist omits, e.g. .rst, broader) file
        # scope than what ANT's own worker population ever collectively
        # sees. Reusing the identical source of truth guarantees "whole
        # repository" means the same thing for both systems.
        environment = RepoEnvironment(environment_root)
        all_files = [str(path.relative_to(environment.root)) for path in environment.iter_files()]
        # Primary protocol: one budget, fixed at construction time, applied
        # identically to every example this agent instance runs -- never a
        # per-example override (see the module-level fairness note above).
        budget = self.tool_call_budget
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
            decision = _normalize_decision(_loads_json_object(response.text))

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

            # "navigate" and "callers" each have a more precise, index-backed
            # lookup that ANT's own AutonomousWorker always tries first,
            # falling back to the plain tool only when the index has
            # nothing (see AutonomousWorker._execute_tool) -- mirrored here
            # so Matched ReAct isn't handed a strictly weaker version of a
            # tool it nominally "has access to". Every other tool has no
            # such indexed variant, so this is not a general pattern to
            # extend beyond what ANT's own worker actually does.
            if tool_name == "navigate":
                results = tools.resolve_symbol(
                    query, all_files, limit=6, need=example.question
                ) or tools.navigate(query, all_files, limit=6)
            elif tool_name == "callers":
                results = tools.indexed_callers(query, all_files, limit=6) or tools.callers(
                    query, all_files, limit=6
                )
            else:
                method = getattr(tools, tool_name)
                results = method(query, all_files, limit=6)
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
