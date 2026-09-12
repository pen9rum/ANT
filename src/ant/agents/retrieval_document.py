"""Retrieval baseline for the long-document/multi-document evaluation
track (Section 6 of the long-context evaluation spec): GPT-4.1 generates a
search query, retrieves relevant document chunks via the shared `search`
primitive, optionally repeats under a frozen round budget, then
synthesizes an answer -- no Need Graph, no multi-agent coordination, no
ANT-specific recovery. Reuses `_TIER2_MAX_ROUNDS`/`_TIER2_QUERY_PROMPT`
from `ant.evaluation.baseline_tiers` verbatim, same as the repository-QA
`ant.agents.retrieval.RetrievalAgent`, so the retrieval BUDGET philosophy
(a fixed number of LLM-decided query-refinement rounds, then one synthesis
call) is identical across both substrates -- only the underlying corpus
`search()` runs against differs (documents instead of a repository).

Uses `ant.tools.local.LocalSearchTool` -- the exact same class/method the
ANT document adapter's own frozen `AutonomousWorker` calls for `search`,
and the same one `ant.agents.matched_react_document.MatchedReActDocumentAgent`
uses below -- so all three methods draw from an IDENTICAL document index,
per Section 10's "do not give ANT a stronger retrieval engine" requirement.
"""
from __future__ import annotations

import time
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.evaluation.baseline_tiers import _TIER2_MAX_ROUNDS, _TIER2_QUERY_PROMPT
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.document_scope import DocumentRecord, EvalDocumentEnvironment
from ant.evaluation_suite.usage import UsageStats
from ant.providers.openai_provider import _loads_json_object
from ant.tools.local import LocalSearchTool


class RetrievalDocumentAgent:
    """Tier 2 / Retrieval for the document substrate. See module docstring
    for why this is a distinct class/name (`retrieval_document`) from the
    repository-QA `RetrievalAgent`: same retrieval-budget philosophy and
    round-refinement loop, applied to a document corpus instead of a repo.
    """

    name = "retrieval_document"

    def __init__(self, model: str = "gpt-4.1") -> None:
        self.model = model

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        provider = CountingOpenAIProvider(model=self.model)
        documents = [DocumentRecord(**d) for d in example.metadata["documents"]]
        environment = EvalDocumentEnvironment(environment_root, documents)
        search_tool = LocalSearchTool(environment_root)
        all_files = [environment.relative_path_for(doc.doc_id) for doc in documents]

        started = time.time()
        evidence = []
        trajectory: list[dict] = []
        query = example.question
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
            decision = _loads_json_object(decision_result.text)
            trajectory.append({"round": round_index, "query": query, "decision": decision})
            if decision.get("enough") is True:
                break
            next_query = decision.get("next_query")
            if not isinstance(next_query, str) or not next_query.strip():
                break
            query = next_query

        answer = provider.synthesize(question=example.question, evidence=evidence)
        llm_calls = provider.drain_call_count()
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
            metadata={
                "generation_model": self.model,
                "retrieval_rounds": len(trajectory),
                "queries_issued": [entry["query"] for entry in trajectory],
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(RetrievalDocumentAgent())
