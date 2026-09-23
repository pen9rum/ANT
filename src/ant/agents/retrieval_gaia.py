"""Sparse Retrieval baseline on GAIA -- `name = "sparse_retrieval_gaia"`.

GAIA has no repository/document corpus to run BM25 over -- the open web
IS the corpus, and `GaiaToolRegistry.search()` (Tavily-backed, see
`gaia_web_backends.py`) is this substrate's one retrieval primitive, the
same way `LocalSearchTool.search()` is the repo-QA track's. This is
therefore a ONE-SHOT baseline (one search call, whatever the search
engine's own ranking returns, then answer) rather than
`ant.agents.retrieval.RetrievalAgent`'s up-to-3-round query-refinement
loop -- there is no repo-scoped "not enough results yet, narrow the
query" signal to iterate on here, and holding this baseline to a single
call keeps the Sparse/Dense contrast clean (see `dense_retrieval_gaia.py`'s
own docstring): Sparse takes the search engine's ranking and snippets
as-is; Dense fetches full pages and re-ranks locally by embedding
similarity. Disclosed here as a deviation from the repo-QA Sparse
baseline's own protocol, not silently matched.

If the task has an attachment, its text is folded into the same context
block as the search hits (best-effort, `read_attachment_text_safely` --
an unsupported-modality or missing attachment does not fail the run).
"""

from __future__ import annotations

import time
from pathlib import Path

from ant.agents.base import AgentResult
from ant.agents.gaia_shared import build_answer_prompt, build_registry, read_attachment_text_safely
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.usage import UsageStats

DEFAULT_SEARCH_LIMIT = 8


class SparseRetrievalGaiaAgent:
    name = "sparse_retrieval_gaia"

    def __init__(self, model: str = "gpt-4.1", search_limit: int = DEFAULT_SEARCH_LIMIT) -> None:
        self.model = model
        self.search_limit = search_limit

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        provider = CountingOpenAIProvider(model=self.model)
        registry = build_registry(example, environment_root)
        started = time.time()

        hits = registry.search(example.question, limit=self.search_limit)
        attachment_text = read_attachment_text_safely(registry)

        context_parts = [
            f"[{i + 1}] {h.title}\nURL: {h.url}\n{h.snippet}" for i, h in enumerate(hits)
        ]
        if attachment_text:
            file_name = registry.environment.file_name
            context_parts.append(f"[Attachment: {file_name}]\n{attachment_text}")
        context_block = "\n\n".join(context_parts)

        prompt = build_answer_prompt(example.question, context_block)
        result = provider.responses_text(prompt, max_output_tokens=1024)
        token_usage = provider.drain_usage()
        llm_calls = provider.drain_call_count()
        elapsed = time.time() - started

        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=result.text.strip(),
            trajectory=[{"prompt": prompt}],
            evidence=[{"path": h.url, "quote": h.snippet, "reason": h.title} for h in hits],
            usage=UsageStats(
                llm_calls=llm_calls,
                tool_calls=1,
                input_tokens=token_usage.input_tokens,
                output_tokens=token_usage.output_tokens,
                total_tokens=token_usage.total_tokens,
                estimated_cost_usd=token_usage.estimated_cost_usd,
                wall_clock_seconds=elapsed,
                unique_files_inspected=len({h.url for h in hits}),
            ),
            termination_reason="single_shot_search_complete",
            metadata={
                "generation_model": self.model,
                "n_search_hits": len(hits),
                "has_attachment": bool(attachment_text),
                "search_backend_last_used": getattr(
                    registry.search_backend, "last_used", "primary"
                ),
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(SparseRetrievalGaiaAgent())
