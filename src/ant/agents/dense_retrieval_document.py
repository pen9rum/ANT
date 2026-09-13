"""Dense Retrieval baseline for the document/long-context evaluation track
-- a standalone, evaluation-only adapter. See docs/
dense_retrieval_baseline_audit.md for the full audit performed before
writing this file.

AUDIT SUMMARY (why this is a NEW file, not a reuse of
`ant.tools.local.LocalSearchTool.dense_search`):

- `dense_search()`'s own embedding index is built by
  `ant.retrieval.dense.build_embedding_index`, which chunks at PYTHON
  SYMBOL granularity via `ant.tools.symbol_index.build_symbol_index` --
  and that function explicitly skips every file that doesn't end in
  `.py` (`if not relative.endswith(".py"): continue`). Document-QA
  environments are `doc_0000.txt`, `doc_0001.txt`, ... -- confirmed
  directly that `build_symbol_index` finds ZERO symbols in them, so
  `dense_search()` would return `[]` for every query on this substrate.
  It is not "already exposed as a standalone baseline" for documents; it
  is not usable here at all as-is.
- This file therefore reuses only the two REAL, substrate-agnostic
  pieces of the existing dense infrastructure: `DenseEmbedder` (the
  frozen local embedding model, `BAAI/bge-small-en-v1.5`, unchanged, zero
  API cost -- runs via `fastembed`/ONNX locally) and `EmbeddingIndex`
  (the frozen cosine-similarity math: L2-normalized rows, dot-product
  ranking) -- both imported unmodified from `ant.retrieval.dense`.
- For CHUNK BOUNDARIES, this file reuses `ant.tools.local.
  _retrieval_regions` -- the exact same paragraph-aware block splitter
  Sparse Retrieval's own BM25 index already uses for every file
  (code or plain text; it has no `.py`-only restriction, confirmed by
  reading it), giving Dense Retrieval byte-identical chunk boundaries to
  Sparse, not merely a "closest equivalent."
- No ANTMAN core file was modified. No persistent `.ant/dense/` cache is
  touched -- each call builds a small, in-memory-only index scoped to
  that one question's own document set, never shared/cached across
  examples, never read/written to disk.

DELIBERATE one-shot design (per the governing spec): question -> dense
top-k -> answer. No iterative retrieval, no query rewriting, no
second-hop retrieval, no reranking, no supporting-fact supervision. This
is an intentional asymmetry with Sparse Retrieval (which iterates up to
3 rounds of LLM-decided query refinement, `_TIER2_MAX_ROUNDS`) --
documented here, not silently matched, because forcing multi-round
behavior onto a "dense-only, no agentic retrieval" baseline would no
longer isolate what this baseline is meant to isolate (lexical vs.
embedding ranking, holding everything else about the answer-generation
stage constant).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.domain import Evidence
from ant.evaluation_suite.answer_contract import (
    apply_context_authoritative_regrounding,
    condense_to_answer_span,
)
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.document_scope import DocumentRecord, EvalDocumentEnvironment
from ant.evaluation_suite.usage import UsageStats
from ant.retrieval.dense import DenseEmbedder, EmbeddingEntry, EmbeddingIndex
from ant.tools.local import _retrieval_regions

# Same top-k Sparse Retrieval uses (`limit=8` in retrieval_document.py) --
# never widened just because embeddings are different.
TOP_K = 8


def _build_document_index(
    environment_root: Path, files: list[str], embedder: DenseEmbedder
) -> tuple[EmbeddingIndex, int]:
    """Builds a fresh, in-memory-only embedding index over `files`, using
    the SAME chunk boundaries Sparse Retrieval's own BM25 index uses
    (`_retrieval_regions`) -- never ANTMAN's persistent, symbol-only
    `.ant/dense/` cache. Returns (index, n_chunks_indexed).
    """
    entries: list[EmbeddingEntry] = []
    texts: list[str] = []
    for relative in files:
        path = environment_root / relative
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for start, block in _retrieval_regions(lines):
            text = "\n".join(block).strip()
            if not text:
                continue
            entries.append(
                EmbeddingEntry(
                    path=relative,
                    line_start=start,
                    line_end=start + len(block) - 1,
                    quote=text[:2400],
                )
            )
            texts.append(text)

    if not entries:
        return EmbeddingIndex(entries=[], vectors=np.zeros((0, 0), dtype=np.float32)), 0

    vectors = np.asarray(embedder.embed(texts), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return EmbeddingIndex(entries=entries, vectors=vectors / norms), len(entries)


class DenseRetrievalDocumentAgent:
    """Dense-ONLY retrieval for the document substrate: one embedding
    search (no BM25, no RRF fusion, no sparse fallback, no iterative
    refinement), then the SAME `synthesize()` answer-generation stage,
    Condition-B regrounding hook, and short-answer condensation contract
    every other document-track method already uses.
    """

    name = "dense_retrieval_document"

    def __init__(self, model: str = "gpt-4.1", top_k: int = TOP_K) -> None:
        self.model = model
        self.top_k = top_k

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        provider = CountingOpenAIProvider(model=self.model)
        documents = [DocumentRecord(**d) for d in example.metadata["documents"]]
        environment = EvalDocumentEnvironment(environment_root, documents)
        all_files = [environment.relative_path_for(doc.doc_id) for doc in documents]

        started = time.time()
        embedder = DenseEmbedder()  # local, frozen default model, zero API cost
        index, n_chunks_indexed = _build_document_index(environment_root, all_files, embedder)

        if index.entries:
            [query_vector] = embedder.embed([example.question])
            hits = index.search(query_vector, limit=self.top_k)
        else:
            hits = []

        evidence = [
            Evidence(
                path=entry.path,
                line_start=entry.line_start,
                line_end=entry.line_end,
                quote=entry.quote,
                reason=f"Dense semantic match (score={score:.3f}) for: {example.question[:80]}",
                dense_score=score,
            )
            for score, entry in hits
        ]

        raw_answer = provider.synthesize(question=example.question, evidence=evidence)
        # Single-Needle Contamination Study Condition B -- post-hoc, same
        # pattern every other document-track method uses.
        grounded_answer = raw_answer
        if example.metadata.get("answer_contract_condition") == "B":
            evidence_block = "\n".join(
                f"[{item.path}:{item.line_start}-{item.line_end}] {item.quote}" for item in evidence
            )
            grounded_answer = apply_context_authoritative_regrounding(
                provider, example.question, raw_answer, evidence_block
            )
        answer = condense_to_answer_span(provider, example.question, grounded_answer)
        llm_calls = provider.drain_call_count()
        token_usage = provider.drain_usage()
        elapsed = time.time() - started

        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=answer,
            trajectory=[
                {
                    "n_chunks_indexed": n_chunks_indexed,
                    "top_k": self.top_k,
                    "retrieved_chunk_ids": [
                        f"{e.path}:{e.line_start}-{e.line_end}" for e in evidence
                    ],
                    "retrieved_scores": [round(e.dense_score, 4) for e in evidence],
                }
            ],
            evidence=[item.model_dump() for item in evidence],
            usage=UsageStats(
                llm_calls=llm_calls,
                tool_calls=1,
                input_tokens=token_usage.input_tokens,
                output_tokens=token_usage.output_tokens,
                total_tokens=token_usage.total_tokens,
                estimated_cost_usd=token_usage.estimated_cost_usd,
                wall_clock_seconds=elapsed,
                unique_files_inspected=len({item.path for item in evidence}),
            ),
            termination_reason="single_shot_dense_retrieval_complete",
            metadata={
                "generation_model": self.model,
                "embedding_model": embedder.model_name,
                "embedding_dimension": index.vectors.shape[1] if index.vectors.size else 0,
                "n_chunks_indexed": n_chunks_indexed,
                "top_k": self.top_k,
                "retrieved_chunk_ids": [f"{e.path}:{e.line_start}-{e.line_end}" for e in evidence],
                "retrieved_scores": [round(e.dense_score, 4) for e in evidence],
                "raw_answer_before_condensation": raw_answer,
                "answer_contract_condition": example.metadata.get("answer_contract_condition"),
                "grounded_answer_after_regrounding": grounded_answer
                if grounded_answer != raw_answer
                else None,
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(DenseRetrievalDocumentAgent())
