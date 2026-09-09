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

# Per-call result limits (the `limit=` argument each LocalSearchTool method
# receives). An earlier version of this file called every tool with a
# uniform limit=6 and rationalized it after the fact as "generous
# compensation" for being a single agent -- that justification was never
# actually specified anywhere before being written down, so it does not
# count as a "previously specified experimental reason" to keep an
# unequal primitive. The fairness-closure audit's own principle applies
# instead: expose comparable PRIMITIVE repository-information capabilities,
# and let ANT's advantage come only from its coordination mechanisms (Need
# Graph, multi-worker, recovery), not from a bigger single-call limit no
# one asked for. ANT_PARITY_TOOL_LIMITS is therefore the DEFAULT and is
# copied exactly from AutonomousWorker's own real, operative per-call
# limits -- specifically the reasoner-driven `_execute_tool` path
# (navigate/references/callers/callees/assignments/imports=2, subclasses=4)
# plus the two unconditional calls `AutonomousWorker.run()` always makes
# itself (search/dense_search=4) -- because that reasoner-driven path is
# the ONLY one the real AntAgent baseline ever exercises in this harness
# (AntAgent always constructs LocalCoordinator with a reasoner; the
# fixed/mechanical fallback pipeline in AutonomousWorker only runs in
# isolated unit tests that pass reasoner=None). ANT worker code itself is
# never modified to produce this table -- these numbers are read directly
# from ant.workers.autonomous.AutonomousWorker, not invented.
ANT_PARITY_TOOL_LIMITS: dict[str, int] = {
    "search": 4,
    "dense_search": 4,
    "navigate": 2,
    "references": 2,
    "callers": 2,
    "callees": 2,
    "assignments": 2,
    "imports": 2,
    "subclasses": 4,
}

# A deliberately more generous ALTERNATE configuration, kept only for a
# later, explicitly separate sensitivity experiment ("does Matched ReAct do
# better/worse with a bigger primitive budget than an ANT worker gets") --
# NOT the default, NOT run as part of this pass. Selecting it requires
# explicitly passing `tool_result_limits=GENEROUS_SENSITIVITY_TOOL_LIMITS`
# at construction time; nothing in this module does that itself.
GENEROUS_SENSITIVITY_TOOL_LIMITS: dict[str, int] = dict.fromkeys(AVAILABLE_TOOLS, 6)

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
    Phase F specifies. The agent sees every file ANT's own worker
    population collectively covers, not a territory-scoped slice -- NOT,
    despite how that might read, literally every file that exists in the
    git checkout. Both are scoped by ANT's own RepoEnvironment
    (IGNORED_DIRS + a TEXT_EXTENSIONS allowlist), which excludes binaries
    (correctly) but also some real text-bearing formats this allowlist
    happens to omit -- .rst and .sql being the two that matter most
    concretely: measured directly against 4 already-cloned benchmark
    repos, this excludes 400/1803 files from sphinx (mostly .rst -- sphinx
    is itself a documentation tool, so this is a meaningful chunk of its
    own repo) and 1473/3377 from sqlfluff (.sql fixtures -- sqlfluff is a
    SQL linter, so these are directly relevant to its own behavior). This
    is a real evaluation-infrastructure limitation shared identically by
    ANT and every baseline reusing RepoEnvironment (not a Matched-ReAct-
    specific gap, and not something this class can fix on its own --
    RepoEnvironment is frozen ANT core) -- never describe this as
    "unrestricted whole-repository access" in any report or paper text.
    The agent also has access to the SAME underlying tool implementations
    ANT's own AutonomousWorker uses (`ant.tools.local.LocalSearchTool`,
    the identical class, not a reimplementation) -- matching "same
    underlying lexical/dense/symbol-navigation capability" as literally as
    this codebase allows.

    `tool_call_budget`, not `max_rounds`, is this agent's real compute
    ceiling -- see the module docstring above for why, and why its default
    is a measured mean from real ANT usage, not an arbitrary number.

    `tool_result_limits` (per-call `limit=` values) default to
    ANT_PARITY_TOOL_LIMITS -- copied exactly from AutonomousWorker's own
    real per-call limits, not chosen independently -- so a single tool
    call here returns exactly as much as the same call would inside an
    ANT worker. See ANT_PARITY_TOOL_LIMITS's own module-level comment for
    why this is the default and GENEROUS_SENSITIVITY_TOOL_LIMITS is not.
    """

    name = "matched_react"

    def __init__(
        self,
        model: str = "gpt-4.1",
        tool_call_budget: int | None = None,
        tool_result_limits: dict[str, int] | None = None,
    ) -> None:
        self.model = model
        self.tool_call_budget = tool_call_budget or DEFAULT_TOOL_CALL_BUDGET
        # Default: exact ANT-worker per-call limits (see ANT_PARITY_TOOL_LIMITS
        # above). Pass GENEROUS_SENSITIVITY_TOOL_LIMITS explicitly to run the
        # deferred sensitivity configuration instead -- never the default.
        self.tool_result_limits = tool_result_limits or ANT_PARITY_TOOL_LIMITS

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
            # Every AVAILABLE_TOOLS entry is covered by both
            # ANT_PARITY_TOOL_LIMITS and GENEROUS_SENSITIVITY_TOOL_LIMITS;
            # the fallback only matters for a caller-supplied partial dict.
            tool_limit = self.tool_result_limits.get(tool_name, 4)
            if tool_name == "navigate":
                results = tools.resolve_symbol(
                    query, all_files, limit=tool_limit, need=example.question
                ) or tools.navigate(query, all_files, limit=tool_limit)
            elif tool_name == "callers":
                results = tools.indexed_callers(
                    query, all_files, limit=tool_limit
                ) or tools.callers(query, all_files, limit=tool_limit)
            else:
                method = getattr(tools, tool_name)
                results = method(query, all_files, limit=tool_limit)
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
