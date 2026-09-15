"""Dense Retrieval baseline for the repo-QA evaluation track (RepoProbe,
SWE-QA-Pro) -- a standalone, evaluation-only adapter, sibling to
`ant.agents.dense_retrieval_document` (document/long-context track) and
modeled on `ant.agents.retrieval.RetrievalAgent` (this track's own Sparse
Retrieval baseline).

AUDIT SUMMARY (why this is a NEW file, not a reuse of
`ant.tools.local.LocalSearchTool.dense_search`):

- `dense_search()`'s own embedding index is built by
  `ant.retrieval.dense.build_embedding_index`, which chunks at PYTHON
  SYMBOL granularity via `ant.tools.symbol_index.build_symbol_index` --
  and that function explicitly skips every file that doesn't end in
  `.py` (`if not relative.endswith(".py"): continue`). RepoProbe-Python's
  and SWE-QA-Pro's own repos are real, multi-language codebases (sphinx's
  `.rst`, sqlfluff's `.sql`/config, etc.) -- using that path unmodified
  would silently give Dense Retrieval a narrower searchable corpus than
  Sparse Retrieval's already-frozen `retrieval` baseline on the exact same
  repos.
- This file therefore reuses only the two REAL, substrate-agnostic pieces
  of the existing dense infrastructure: `DenseEmbedder` (the frozen local
  embedding model, `BAAI/bge-small-en-v1.5`, unchanged, zero API cost --
  runs via `fastembed`/ONNX locally) and `EmbeddingIndex` (the frozen
  cosine-similarity math: L2-normalized rows, dot-product ranking) -- both
  imported unmodified from `ant.retrieval.dense`.
- The searchable file universe is built via `EvalRepoEnvironment.
  iter_files()` (`ant.evaluation_suite.repo_scope`) -- the SAME
  content-based, not-`.py`-only file discovery `RetrievalAgent` itself
  uses (see retrieval.py's own comment at the identical call).
- For CHUNK BOUNDARIES, this file reuses `ant.tools.local.
  _retrieval_regions` -- the exact same paragraph-aware block splitter
  Sparse Retrieval's own BM25 index already uses for every file (code or
  plain text; no `.py`-only restriction), giving Dense Retrieval
  byte-identical chunk boundaries to Sparse.
- Answer generation is exactly `RetrievalAgent`'s own contract: one
  `provider.synthesize(question=..., evidence=...)` call, nothing else --
  no condensation, no regrounding. RepoProbe/SWE-QA-Pro are LLM-judge
  scored over the full narrative answer, not EM/F1 against a short gold
  span, so the document track's `condense_to_answer_span` contract does
  not apply here and must not be borrowed just because it exists.

Disk-cached per repo checkout (`index_root / benchmark / repo_slug`),
unlike the document-track sibling: a repo-QA benchmark asks many
questions against the SAME pinned repo checkout (e.g. 30 RepoProbe
questions against one adk-python checkout), and that corpus is identical
across every one of them -- rebuilding the embedding index from scratch
per question would be pure wasted CPU, not a correctness concern (dense
embedding is local/free, but not instant). The cache is reused only when
its own file set exactly equals the current file universe (never merely
"a cache file exists at this path") -- the same stale-index guard already
found necessary and fixed in `AntDocumentAgent._ensure_indexed()`.

DELIBERATE one-shot design (per the governing spec): query -> dense top-k
evidence -> one answer-generation call. No ReAct, no graph navigation, no
iterative tool loop, no query rewriting -- an intentional asymmetry with
Sparse Retrieval (which iterates up to `_TIER2_MAX_ROUNDS` rounds of
LLM-decided query refinement), documented here, not silently matched, for
the same reason the document-track sibling documents it: forcing
multi-round behavior onto a "dense-only, no agentic retrieval" baseline
would no longer isolate what this baseline is meant to isolate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.domain import Evidence
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.repo_scope import EvalRepoEnvironment
from ant.evaluation_suite.usage import UsageStats
from ant.retrieval.dense import DenseEmbedder, EmbeddingEntry, EmbeddingIndex, _embed_entries
from ant.tools.local import _retrieval_regions

# Same top-k Sparse Retrieval uses (`limit=8` in retrieval.py) -- never
# widened just because embeddings are different.
TOP_K = 8

_CACHE_KEY = "repo"


def _build_repo_dense_index(
    environment_root: Path, files: list[str], embedder: DenseEmbedder
) -> tuple[EmbeddingIndex, int]:
    """Builds an embedding index over `files`, chunked with
    `_retrieval_regions` -- the exact same block splitter Sparse
    Retrieval's own territory-wide BM25 index uses. Returns
    (index, n_chunks_indexed).
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

    # A full repo checkout, chunked at paragraph granularity (not the much
    # sparser per-symbol granularity build_embedding_index uses), can
    # produce many thousands of chunks -- confirmed live that a single
    # unbatched embedder.embed(texts) call over ~4k texts did not return
    # within 10 minutes and drove memory to several GB, while the SAME
    # corpus embedded in _embed_entries' own 256-text batches (with visible
    # per-batch progress) completed normally. Reusing that helper, not
    # reimplementing batching here, keeps this file's batch size in sync
    # with DEFAULT_SCORING_CONFIG.dense.embed_batch_size everywhere else in
    # the codebase.
    index = _embed_entries(entries, texts, embedder, verbose=True)
    return index, len(entries)


def _source_files_path(index_dir: Path) -> Path:
    return index_dir / f"{_CACHE_KEY}.source_files.json"


def _ensure_repo_dense_index(
    environment_root: Path, files: list[str], embedder: DenseEmbedder, index_dir: Path
) -> tuple[EmbeddingIndex, int]:
    """Disk-cached per-repo dense index. Reuses the cached index ONLY when
    the exact `files` universe used to BUILD it (persisted separately, in
    `{key}.source_files.json`) equals the current one -- the same
    stale-index guard already found necessary in
    `AntDocumentAgent._ensure_indexed()`.

    Regression note: an earlier version compared `{entry.path for entry in
    cached.entries}` to `set(files)` instead. That is NOT the same set --
    `_build_repo_dense_index` only emits an entry for a file whose
    `_retrieval_regions` produced at least one non-blank chunk, so any
    file that is empty or whitespace-only (a common, unremarkable case --
    e.g. a package's `__init__.py`) never appears as an entry path even
    though it legitimately belongs to the file universe. That made the
    equality check fail on EVERY call, silently re-embedding the whole
    repo from scratch for every question against it -- confirmed live on
    RepoProbe-Python's FieldStation42 repo, which re-ran its full
    ~4,255-chunk embed multiple times before this was caught.
    """
    cached = EmbeddingIndex.load(index_dir, _CACHE_KEY)
    source_files_path = _source_files_path(index_dir)
    if cached is not None and source_files_path.exists():
        stored_files = json.loads(source_files_path.read_text(encoding="utf-8"))
        if set(stored_files) == set(files):
            return cached, len(cached.entries)
    index, n_chunks = _build_repo_dense_index(environment_root, files, embedder)
    index.save(index_dir, _CACHE_KEY)
    source_files_path.write_text(json.dumps(sorted(files)), encoding="utf-8")
    return index, n_chunks


class DenseRetrievalRepoAgent:
    """Dense-ONLY retrieval for the repo-QA substrate: one embedding search
    (no BM25, no RRF fusion, no sparse fallback, no iterative refinement),
    then the SAME `synthesize()` answer-generation stage Sparse Retrieval's
    own `RetrievalAgent` uses for this track -- no condensation contract,
    which belongs only to the document/EM-F1 track.
    """

    name = "dense_retrieval"

    def __init__(
        self,
        model: str = "gpt-4.1",
        top_k: int = TOP_K,
        index_root: Path | None = None,
    ) -> None:
        self.model = model
        self.top_k = top_k
        self.index_root = index_root or Path(".ant/eval-suite-dense")

    def _index_dir_for(self, example: TaskExample, environment_root: Path) -> Path:
        repo_slug = environment_root.name
        return self.index_root / example.benchmark / repo_slug

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        provider = CountingOpenAIProvider(model=self.model)
        # Same scope RetrievalAgent/AntAgent use -- EvalRepoEnvironment
        # (content-based text detection, not a closed extension allowlist).
        environment = EvalRepoEnvironment(environment_root)
        all_files = [str(path.relative_to(environment.root)) for path in environment.iter_files()]

        started = time.time()
        embedder = DenseEmbedder()  # local, frozen default model, zero API cost
        index_dir = self._index_dir_for(example, environment_root)
        index, n_chunks_indexed = _ensure_repo_dense_index(
            environment_root, all_files, embedder, index_dir
        )

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

        answer = provider.synthesize(question=example.question, evidence=evidence)
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
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(DenseRetrievalRepoAgent())
