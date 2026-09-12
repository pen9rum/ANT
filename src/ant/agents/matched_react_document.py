"""Matched ReAct baseline for the long-document/multi-document evaluation
track (Section 7 of the long-context evaluation spec): ONE autonomous
GPT-4.1 agent with the SAME document primitives ANT's own worker path can
meaningfully use (`search`), plus the two additional primitives Section 3
specifies (`view`, `navigate`) that a single agent CAN be given without
touching frozen core -- no WorkerCards, no multiple workers, no Need
Graph, no runtime structural Need revision, no ANT progress controller, no
reroute/reframe/fallback, no cross-task memory. A fresh, dedicated class
(not a reuse of `ant.agents.matched_react.MatchedReActAgent`, whose fixed
`AVAILABLE_TOOLS` are eight CODE-SYMBOL tools -- navigate/references/
callers/callees/assignments/imports/subclasses -- that are simply the
wrong primitives for prose documents, even though they would not crash on
one) so this agent's tool set matches what Section 3 actually specifies
for this substrate, not a repurposed repository-QA tool list.

Tool-parity disclosure (see `ant.agents.ant_document_adapter`'s own module
docstring for the fuller version): this agent genuinely has `search` +
`view` + `navigate`; ANT's own document adapter, bound by frozen core's
`AutonomousWorker` tool loop, only meaningfully exercises `search`. That
asymmetry is real and is reported explicitly in this evaluation pass's
Section 14.B answer, not hidden by giving this agent fewer tools than
Section 3 calls for just to force a false symmetry.
"""
from __future__ import annotations

import time
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.domain import Evidence
from ant.evaluation_suite.answer_contract import condense_to_answer_span
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.document_scope import (
    DocumentRecord,
    EvalDocumentEnvironment,
    chunk_documents,
)
from ant.evaluation_suite.usage import UsageStats
from ant.providers.openai_provider import _loads_json_object
from ant.tools.document_tools import chunk_by_global_index, navigate_chunk, view_document
from ant.tools.local import LocalSearchTool

# Same measured-mean-based budget philosophy as the repository-QA Matched
# ReAct baseline (ant.agents.matched_react.DEFAULT_TOOL_CALL_BUDGET) --
# reused as a fixed, disclosed default here too rather than re-deriving a
# document-specific number from scratch, since no equivalent "real ANT
# document-adapter usage" measurement exists yet to base one on.
DEFAULT_TOOL_CALL_BUDGET = 50

AVAILABLE_TOOLS = ("search", "view", "navigate")

# Fixed, disclosed chunk size for this agent's OWN navigate() primitive --
# deliberately smaller than LongAgent's 2000-token member chunks (this is
# stepwise single-agent navigation granularity, not a static partition
# unit) and not tuned against any benchmark score.
NAVIGATE_CHUNK_SIZE_TOKENS = 300

_SYSTEM_PROMPT = """You are a single autonomous agent answering a question about a set of \
documents. You have direct access to the following tools (the same underlying document-search \
capability available to a specialized multi-agent system's own workers -- you are just one \
agent using it alone, with no team, no shared graph, no persistent memory):

- search(query): lexical/BM25 search across all documents, returns matching passages
- view(doc_id): read one entire document's full text, given its doc_id (e.g. "doc3")
- navigate(chunk_id direction): step to the next/previous fixed-size chunk from a chunk_id you \
already saw (e.g. "navigate" with query "4 next" or "4 previous")

At each step, respond with ONLY a JSON object of one of these two shapes.
The keys must be EXACTLY "tool" and "query" -- do NOT use the tool's own \
name as a JSON key:
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


class MatchedReActDocumentAgent:
    """Tier 3 / Matched ReAct for the document substrate. See module
    docstring for the tool set and the disclosed asymmetry with ANT.
    """

    name = "matched_react_document"

    def __init__(self, model: str = "gpt-4.1", tool_call_budget: int | None = None) -> None:
        self.model = model
        self.tool_call_budget = tool_call_budget or DEFAULT_TOOL_CALL_BUDGET

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        provider = CountingOpenAIProvider(model=self.model)
        documents = [DocumentRecord(**d) for d in example.metadata["documents"]]
        environment = EvalDocumentEnvironment(environment_root, documents)
        search_tool = LocalSearchTool(environment_root)
        all_files = [environment.relative_path_for(doc.doc_id) for doc in documents]
        chunks = chunk_documents(documents, NAVIGATE_CHUNK_SIZE_TOKENS)

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
            decision = _normalize_decision(_loads_json_object(response.text))

            finish_text = decision.get("finish")
            if isinstance(finish_text, str) and finish_text.strip():
                final_answer = finish_text.strip()
                termination_reason = "agent_declared_finish"
                break

            tool_name = decision.get("tool")
            query = decision.get("query", "")
            if tool_name not in AVAILABLE_TOOLS or not isinstance(query, str) or not query.strip():
                history.append(
                    {"step": step, "tool": str(tool_name), "query": str(query), "results": []}
                )
                continue

            results = self._execute_tool(
                tool_name, query, search_tool, all_files, environment, chunks
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
            all_evidence = [item for entry in history for item in entry["results"]]
            final_answer = provider.synthesize(
                question=example.question,
                evidence=[Evidence.model_validate(e) for e in all_evidence],
            )

        raw_answer = final_answer
        # Shared short-answer contract (Part A) -- post-hoc only, applied
        # after the agent's own tool-use loop has already finished (either
        # via its own "finish" decision or the budget-exhausted fallback
        # synthesis above); the tool-selection/reasoning loop itself is
        # completely unaffected.
        final_answer = condense_to_answer_span(provider, example.question, final_answer)
        llm_calls = provider.drain_call_count()
        token_usage = provider.drain_usage()
        elapsed = time.time() - started
        all_evidence = [item for entry in history for item in entry["results"]]
        tool_type_counts: dict[str, int] = {}
        for entry in history:
            tool_type_counts[entry["tool"]] = tool_type_counts.get(entry["tool"], 0) + 1

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
            metadata={
                "tool_call_budget": budget,
                "generation_model": self.model,
                "tool_type_counts": tool_type_counts,
                "steps_taken": len(history),
                "raw_answer_before_condensation": raw_answer,
            },
        )

    def _execute_tool(
        self,
        tool_name: str,
        query: str,
        search_tool: LocalSearchTool,
        all_files: list[str],
        environment: EvalDocumentEnvironment,
        chunks: list,
    ) -> list[Evidence]:
        if tool_name == "search":
            return search_tool.search(query, all_files, limit=8)
        if tool_name == "view":
            text = view_document(environment, query.strip())
            if not text:
                return []
            return [
                Evidence(
                    path=query.strip(),
                    line_start=0,
                    line_end=0,
                    quote=text[:4000],
                    reason=f"Full-document view of {query.strip()}.",
                )
            ]
        if tool_name == "navigate":
            parts = query.strip().split()
            if len(parts) != 2 or not parts[0].lstrip("-").isdigit():
                return []
            chunk_index, direction = int(parts[0]), parts[1]
            current = chunk_by_global_index(chunks, chunk_index)
            if current is None:
                return []
            target = navigate_chunk(chunks, chunk_index, direction)
            if target is None:
                return []
            return [
                Evidence(
                    path=target.document_id,
                    line_start=target.token_start,
                    line_end=target.token_end,
                    quote=target.text[:4000],
                    reason=(
                        f"Navigated {direction} from chunk {chunk_index} to chunk "
                        f"{target.global_chunk_index} (document {target.document_id})."
                    ),
                )
            ]
        return []


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(MatchedReActDocumentAgent())
