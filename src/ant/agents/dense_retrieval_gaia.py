"""Dense Retrieval baseline on GAIA -- `name = "dense_retrieval_gaia"`.

Same frozen embedding infrastructure every other Dense Retrieval variant
in this suite uses (`ant.retrieval.dense.DenseEmbedder`/`EmbeddingIndex`,
BAAI/bge-small-en-v1.5, local/free, unchanged) and the same paragraph-aware
chunk boundaries (`ant.tools.local._retrieval_regions`) as the document
track's own `dense_retrieval_document.py` -- imported, not reimplemented.

WHAT MAKES THIS "DENSE" RATHER THAN A COPY OF SPARSE: GAIA has no local
corpus to embed ahead of time, so this method searches the web (the same
`GaiaToolRegistry.search()` primitive Sparse uses) to get CANDIDATE URLs,
then does the substantive extra work Sparse deliberately skips -- it
FETCHES each candidate page's full text (`registry.open_url`), chunks it,
embeds every chunk, and re-ranks by cosine similarity against the
embedded question. Sparse answers from the search engine's own snippet
ranking; Dense answers from a local semantic re-ranking over full page
content. That is the real, disclosed algorithmic difference between the
two baselines on this substrate.

One-shot, no iterative retrieval -- same asymmetry with Sparse Retrieval
`dense_retrieval_document.py` already documents for the document track
(iterating would blur what this baseline isolates: lexical/engine ranking
vs. embedding ranking, holding the answer stage constant).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ant.agents.base import AgentResult
from ant.agents.gaia_shared import build_answer_prompt, build_registry, read_attachment_text_safely
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.usage import UsageStats
from ant.retrieval.dense import DenseEmbedder, EmbeddingEntry, EmbeddingIndex
from ant.tools.local import _retrieval_regions

DEFAULT_SEARCH_LIMIT = 8
DEFAULT_FETCH_LIMIT = 5
DEFAULT_TOP_K = 8
MAX_CHUNK_CHARS = 1200


class DenseRetrievalGaiaAgent:
    name = "dense_retrieval_gaia"

    def __init__(
        self,
        model: str = "gpt-4.1",
        search_limit: int = DEFAULT_SEARCH_LIMIT,
        fetch_limit: int = DEFAULT_FETCH_LIMIT,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.model = model
        self.search_limit = search_limit
        self.fetch_limit = fetch_limit
        self.top_k = top_k

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        provider = CountingOpenAIProvider(model=self.model)
        registry = build_registry(example, environment_root)
        started = time.time()

        hits = registry.search(example.question, limit=self.search_limit)
        attachment_text = read_attachment_text_safely(registry)

        entries: list[EmbeddingEntry] = []
        texts: list[str] = []
        n_fetched = 0
        n_fetch_errors = 0
        for hit in hits[: self.fetch_limit]:
            try:
                page_text = registry.open_url(hit.url)
            except Exception:  # noqa: BLE001 -- one dead link must not sink the task
                n_fetch_errors += 1
                continue
            n_fetched += 1
            for start_line, block in _retrieval_regions(page_text.splitlines()):
                text = "\n".join(block).strip()
                if not text:
                    continue
                entries.append(
                    EmbeddingEntry(
                        path=hit.url,
                        line_start=start_line,
                        line_end=start_line + len(block) - 1,
                        quote=text[:MAX_CHUNK_CHARS],
                    )
                )
                texts.append(text[:MAX_CHUNK_CHARS])

        if attachment_text:
            name = registry.environment.file_name or "attachment"
            for start_line, block in _retrieval_regions(attachment_text.splitlines()):
                text = "\n".join(block).strip()
                if not text:
                    continue
                entries.append(
                    EmbeddingEntry(
                        path=name,
                        line_start=start_line,
                        line_end=start_line + len(block) - 1,
                        quote=text[:MAX_CHUNK_CHARS],
                    )
                )
                texts.append(text[:MAX_CHUNK_CHARS])

        embedder = DenseEmbedder()  # local, frozen default model, zero API cost
        selected: list[tuple[float, EmbeddingEntry]] = []
        if entries:
            arr = np.asarray(embedder.embed(texts), dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            index = EmbeddingIndex(entries=entries, vectors=arr / norms)
            [query_vector] = embedder.embed([example.question])
            selected = index.search(query_vector, limit=self.top_k)

        context_parts = [f"[{i + 1}] {e.path}\n{e.quote}" for i, (_score, e) in enumerate(selected)]
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
            evidence=[
                {
                    "path": e.path,
                    "line_start": e.line_start,
                    "line_end": e.line_end,
                    "quote": e.quote,
                }
                for _score, e in selected
            ],
            usage=UsageStats(
                llm_calls=llm_calls,
                tool_calls=1 + n_fetched,
                input_tokens=token_usage.input_tokens,
                output_tokens=token_usage.output_tokens,
                total_tokens=token_usage.total_tokens,
                estimated_cost_usd=token_usage.estimated_cost_usd,
                wall_clock_seconds=elapsed,
                unique_files_inspected=len({e.path for _s, e in selected}),
            ),
            termination_reason="single_shot_dense_retrieval_complete",
            metadata={
                "generation_model": self.model,
                "embedding_model": embedder.model_name,
                "n_search_hits": len(hits),
                "n_pages_fetched": n_fetched,
                "n_fetch_errors": n_fetch_errors,
                "n_chunks_indexed": len(entries),
                "has_attachment": bool(attachment_text),
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(DenseRetrievalGaiaAgent())
