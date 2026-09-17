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
import os
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


# Chunks buffered in memory before a save-and-release flush, not the whole
# repo at once. A full repo checkout chunked at paragraph granularity can
# produce many thousands of chunks (adk-python: ~34k) -- confirmed live
# that building+holding the whole corpus in memory before one final save
# both drove RSS to several GB (onnxruntime's own inference-time arena
# plus this process's own entries/texts/vectors buffers) and meant a crash
# partway through lost 100% of that repo's progress, forcing a full
# from-scratch retry every time. Flushing every ~2000 chunks bounds peak
# memory to roughly one flush's worth regardless of repo size, and makes
# progress resumable -- a retry after a crash only re-embeds the files
# that were never flushed, not the whole repo.
_FLUSH_CHUNK_BUDGET = 2000
# Recreate the embedder (and therefore its onnxruntime session/arena)
# periodically rather than reusing one session for the whole repo: an
# onnxruntime CPU arena does not shrink as inference proceeds, so a
# session that lives across many thousands of chunks can still accumulate
# memory across flushes even though each flush's own working set is
# bounded. A fresh session forces the old arena to actually be released.
_RECYCLE_EMBEDDER_EVERY_N_FLUSHES = 3

# Per-inference-call batch size for this large-repo path specifically --
# NOT the shared DEFAULT_SCORING_CONFIG.dense.embed_batch_size (256, still
# used everywhere else, e.g. per-worker territory embedding, which never
# OOM'd). A live isolated repro showed onnxruntime's own peak RSS for one
# call scales with batch size far more than the arena strategy setting
# alone controls: 256 texts/batch plateaued at ~7.8-9GB even with
# kSameAsRequested, while 32 texts/batch plateaued under 800MB on the
# same synthetic corpus.
_REPO_EMBED_BATCH_SIZE = int(os.getenv("ANT_REPO_EMBED_BATCH_SIZE", "32"))


def _covered_files_path(index_dir: Path) -> Path:
    return index_dir / f"{_CACHE_KEY}.covered_files.json"


def _load_covered_files(index_dir: Path) -> set[str]:
    path = _covered_files_path(index_dir)
    if not path.exists():
        return set()
    return set(json.loads(path.read_text(encoding="utf-8")))


def _iter_file_chunks(
    environment_root: Path, relative: str
) -> list[tuple[EmbeddingEntry, str]]:
    path = environment_root / relative
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out: list[tuple[EmbeddingEntry, str]] = []
    for start, block in _retrieval_regions(lines):
        text = "\n".join(block).strip()
        if not text:
            continue
        entry = EmbeddingEntry(
            path=relative, line_start=start, line_end=start + len(block) - 1, quote=text[:2400]
        )
        out.append((entry, text))
    return out


def _ensure_repo_dense_index(
    environment_root: Path, files: list[str], embedder: DenseEmbedder, index_dir: Path
) -> tuple[EmbeddingIndex, int]:
    """Disk-cached per-repo dense index, built and extended incrementally.

    Which files are already embedded and flushed to disk is tracked
    separately (`{key}.covered_files.json`), file-by-file -- not inferred
    from `{entry.path for entry in cached.entries}`, since a file that is
    empty or whitespace-only (e.g. a package's `__init__.py`) never
    produces a `_retrieval_regions` chunk and so never appears as an entry
    path even though it legitimately belongs to the file universe
    (regression found live on RepoProbe-Python's FieldStation42 repo,
    which silently re-embedded from scratch on every call until this was
    tracked file-by-file instead of re-derived from entries).

    Only files not yet in the covered set are embedded this call, in
    bounded-size flushes (see `_FLUSH_CHUNK_BUDGET`) -- a repo whose file
    universe grows (or whose previous embed run was interrupted partway
    through) picks up exactly where it left off, never re-embedding
    already-covered files.
    """
    covered = _load_covered_files(index_dir)
    remaining = [f for f in files if f not in covered]

    loaded = EmbeddingIndex.load(index_dir, _CACHE_KEY)
    index = loaded if loaded is not None else EmbeddingIndex(
        entries=[], vectors=np.zeros((0, 0), dtype=np.float32)
    )

    if not remaining:
        return index, len(index.entries)

    print(
        f"[dense] {len(covered)}/{len(files)} files already cached; "
        f"embedding the remaining {len(remaining)} incrementally...",
        flush=True,
    )

    pending_entries: list[EmbeddingEntry] = []
    pending_texts: list[str] = []
    pending_files: list[str] = []
    live_embedder = embedder
    flush_count = 0

    def _flush() -> None:
        nonlocal index, live_embedder, flush_count, pending_entries, pending_texts, pending_files
        if not pending_entries and not pending_files:
            return
        if pending_entries:
            # _embed_entries returns an EmbeddingIndex whose .entries IS the
            # `pending_entries` list object passed in (not a copy) -- so
            # `index`/`fresh` must never be built before pending_entries is
            # rebound to a fresh list below. Rebinding (not .clear()-ing)
            # pending_entries/pending_texts/pending_files is what keeps the
            # just-saved fresh.entries from being wiped out along with the
            # buffer once it's reset for the next group.
            fresh = _embed_entries(
                pending_entries,
                pending_texts,
                live_embedder,
                verbose=True,
                batch_size=_REPO_EMBED_BATCH_SIZE,
            )
            if index.entries:
                index = EmbeddingIndex(
                    entries=[*index.entries, *fresh.entries],
                    vectors=(
                        np.concatenate([index.vectors, fresh.vectors], axis=0)
                        if index.vectors.size
                        else fresh.vectors
                    ),
                )
            else:
                index = fresh
            index.save(index_dir, _CACHE_KEY)
        covered.update(pending_files)
        _covered_files_path(index_dir).write_text(json.dumps(sorted(covered)), encoding="utf-8")
        print(
            f"[dense] flushed {len(pending_files)} files ({len(pending_entries)} chunks); "
            f"{len(covered)}/{len(files)} files covered so far.",
            flush=True,
        )
        pending_entries = []
        pending_texts = []
        pending_files = []
        flush_count += 1
        if flush_count % _RECYCLE_EMBEDDER_EVERY_N_FLUSHES == 0:
            live_embedder = DenseEmbedder(embedder.model_name)

    for relative in remaining:
        for entry, text in _iter_file_chunks(environment_root, relative):
            pending_entries.append(entry)
            pending_texts.append(text)
        pending_files.append(relative)
        if len(pending_entries) >= _FLUSH_CHUNK_BUDGET:
            _flush()
    _flush()

    return index, len(index.entries)


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
