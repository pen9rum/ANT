"""Retrieval-based ranking of workers against a need's query text.

The Orchestrator (WorkerReasoner.plan_round) used to have exactly one
worker-selection signal: WorkerCard.routing_summary (a 6-term template
string) plus the first 12 of WorkerCard.searchable_terms shown in its
prompt. Both are built by indexing.cards._top_terms's round-robin sampling
(one symbol per file, then repeat) capped at a fixed length -- confirmed
directly on the real qibo index that this silently drops a worker's own
defining symbols (e.g. `FALQON`, `Circuit` at positions 13/15 of a 48-term
list; `draw` never made it into that list at all) even though they are
present, in full, on WorkerCard.symbols. This is the same failure class as
the search() bug fixed earlier tonight -- a positional/common signal
drowning a rare, discriminative one -- one level up, at worker-selection
instead of file-region-selection.

rank_workers() fixes this the same way: territory-wide BM25 + an
IDF-weighted exact-symbol/path channel + a dense embedding channel (over
WorkerCard.symbols, not the lossy searchable_terms projection of it),
combined by Reciprocal Rank Fusion. It never excludes a worker -- callers
(LocalCoordinator.ask() -> OpenAIProvider.plan_round) use the returned rank
only to reorder/annotate the existing full worker list, per this
codebase's established "every worker shown, no relevance-based
prefiltering" principle (see WorkerCard.routing_summary's own docstring).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ant.domain import WorkerCard
from ant.retrieval import extract_terms, is_stem_match
from ant.retrieval.bm25 import BM25Index
from ant.retrieval.dense import EmbeddingIndex, get_shared_embedder

_query_terms = extract_terms

# Reciprocal Rank Fusion's own standard constant (Cormack, Clarke &
# Buettcher 2009) -- not a value tuned for this codebase or dataset, same
# constant and same rationale as ant.tools.local's search() fusion.
_RRF_K = 60


@dataclass(frozen=True)
class WorkerIndex:
    """The BM25/exact-symbol corpus over one worker list, built once and
    reused across every round of a single LocalCoordinator.ask() call --
    `workers` doesn't change mid-task, so rebuilding this per round would
    just redo the same work for no benefit (same rationale as
    LocalSearchTool's own per-task caches).
    """

    bm25: BM25Index
    term_index: dict[str, set[str]]


def build_worker_index(workers: list[WorkerCard]) -> WorkerIndex:
    documents: list[tuple[str, str]] = []
    term_index: dict[str, set[str]] = defaultdict(set)
    for worker in workers:
        terms = _worker_document_terms(worker)
        documents.append((worker.id, " ".join(terms)))
        for term in set(terms):
            term_index[term].add(worker.id)
    return WorkerIndex(bm25=BM25Index(documents), term_index=dict(term_index))


def rank_workers(
    query: str,
    workers: list[WorkerCard],
    worker_index: WorkerIndex,
    embedding_index: EmbeddingIndex | None = None,
) -> dict[str, int]:
    """Ranks `workers` against `query` (1 = best match), fusing three
    channels by rank position alone (RRF), never by comparing raw scores
    across structurally different signals. A worker no channel found
    anything for simply has no entry in the returned dict -- callers treat
    that as "no retrieval signal", not "worst possible", and fall back to
    whatever order they already had for those workers.
    """
    terms = _query_terms(query)
    if not terms or not workers:
        return {}

    bm25_ranked = _bm25_channel_rank(worker_index.bm25, terms)
    exact_ranked = _exact_channel_rank(worker_index.term_index, terms)
    dense_ranked = _dense_channel_rank(query, workers, embedding_index)

    fused = _reciprocal_rank_fusion(bm25_ranked, exact_ranked, dense_ranked)
    return {worker_id: rank for rank, worker_id in enumerate(fused, start=1)}


def _worker_document_terms(worker: WorkerCard) -> list[str]:
    """The corpus text for one worker: symbol names/qualnames (the AST's
    real, deterministically-ordered, non-truncated definition list -- see
    WorkerCard.symbols), file stems/path components, and
    responsibilities (already includes the README summary text, see
    indexing.cards.build_worker_cards). Deliberately not
    searchable_terms -- that field is exactly the round-robin-truncated
    projection this module works around.
    """
    terms: list[str] = []
    for symbol in worker.symbols:
        terms.extend(_query_terms(symbol.name))
        terms.extend(_query_terms(symbol.qualname))
    for relative in worker.files:
        path = Path(relative)
        terms.extend(_query_terms(path.stem))
        for part in path.parts[:-1]:
            terms.extend(_query_terms(part))
    for text in worker.responsibilities:
        terms.extend(_query_terms(text))
    return terms


def _bm25_channel_rank(bm25: BM25Index, terms: list[str]) -> list[str]:
    return [worker_id for _score, worker_id in bm25.search(terms, limit=len(bm25.documents))]


def _exact_channel_rank(term_index: dict[str, set[str]], terms: list[str]) -> list[str]:
    """Same IDF-style weighting fix applied to ant.tools.local's
    _symbol_path_channel_rank tonight, for the identical reason: a term
    common across many workers' territories (e.g. a shared parent
    directory name) must not count the same as a term unique to one
    worker's symbol table. Weight = 1/(workers that term matches).
    """
    hits: dict[str, float] = defaultdict(float)
    for term in terms:
        matches = term_index.get(term)
        if not matches:
            matches = {
                worker_id
                for key, worker_ids in term_index.items()
                if is_stem_match(term, key)
                for worker_id in worker_ids
            }
        if not matches:
            continue
        weight = 1.0 / len(matches)
        for worker_id in matches:
            hits[worker_id] += weight
    return sorted(hits, key=lambda worker_id: (-hits[worker_id], worker_id))


def _dense_channel_rank(
    query: str, workers: list[WorkerCard], embedding_index: EmbeddingIndex | None
) -> list[str]:
    if embedding_index is None or not embedding_index.entries:
        return []
    embedder = get_shared_embedder()
    if embedder is None:
        return []
    vectors = embedder.embed([query])
    if not vectors:
        return []
    worker_ids = {worker.id for worker in workers}
    results = embedding_index.search(
        vectors[0], limit=len(embedding_index.entries), paths=worker_ids
    )
    return [entry.path for _score, entry in results]


def _reciprocal_rank_fusion(*ranked_lists: list[str]) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, worker_id in enumerate(ranked, start=1):
            scores[worker_id] += 1.0 / (_RRF_K + rank)
    return sorted(scores, key=lambda worker_id: -scores[worker_id])
