"""Optional dense/embedding retrieval, scoped to one worker's territory at a
time -- deliberately not a single global index over the whole repository.

This is a tool a *recruited* worker uses inside its own jurisdiction, exactly
like search()/navigate()/callers() (see LocalSearchTool.dense_search): it
augments what a worker can find inside territory it was already routed to,
it does not let a query skip recruitment/routing/coalition-formation and
pull evidence from anywhere in the repo. The worker-card index
(build_worker_card_index) is the one exception -- it is repo-wide, but it
only ever contributes a routing signal (which worker's card is semantically
close to this query), never evidence content itself. Keeping both pieces
scoped this way is what keeps the multi-agent architecture (need-conditioned
recruitment, temporary coalitions, absence proofs, Colony Memory routing) the
thing actually answering questions -- dense retrieval is one more signal an
agent can draw on, not a replacement for having agents at all.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

import numpy as np

from ant.domain import WorkerCard
from ant.scoring_config import DEFAULT_SCORING_CONFIG

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Worker-card entries reuse EmbeddingEntry's (path, quote) shape rather than
# a separate class: `path` holds the worker id and line_start/line_end are
# unused (0), since search()/save()/load() only ever touch `path` as an
# opaque grouping/identity key, never as a real file path.
WORKER_CARDS_KEY = "cards"

# Chunk-level embeddings are cached under this single repo-wide key, not one
# key per worker's exact file set. A worker's territory almost always shares
# files with another worker (a birth/coalition worker's territory is by
# definition the union of two existing ones) -- keying the cache per worker
# meant those shared files got re-embedded from scratch under every new
# territory hash, even though their vectors already existed on disk under a
# sibling worker's key. One shared index, filtered per query via
# EmbeddingIndex.search(paths=...), means a file's symbols are embedded once
# no matter how many workers' territories include it.
REPO_INDEX_KEY = "repo"


@dataclass(frozen=True)
class EmbeddingEntry:
    path: str
    line_start: int
    line_end: int
    quote: str


class DenseEmbedder:
    """Thin wrapper around a local text-embedding model.

    fastembed is imported lazily inside __init__, not at module load time, so
    importing ant.retrieval.dense never requires the optional 'dense' extra
    to be installed -- only actually constructing an embedder does. This
    keeps the core install and the default test suite exactly as
    fast/network-free as before dense retrieval existed.
    """

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            msg = (
                "Dense retrieval requires the optional 'dense' extra. "
                "Install it with: pip install 'ant-codebase[dense]'"
            )
            raise RuntimeError(msg) from exc
        self.model_name = model_name or os.getenv("ANT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self._model = TextEmbedding(model_name=self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [vector.tolist() for vector in self._model.embed(texts)]


# Module-level so the (potentially slow, model-loading) embedder is built at
# most once per process, shared across every caller (LocalSearchTool,
# LocalCoordinator routing, ...), not once per worker/round.
_shared_embedder: DenseEmbedder | None = None
_shared_embedder_attempted = False


def get_shared_embedder() -> DenseEmbedder | None:
    global _shared_embedder, _shared_embedder_attempted
    if not _shared_embedder_attempted:
        _shared_embedder_attempted = True
        try:
            _shared_embedder = DenseEmbedder()
        except RuntimeError:
            # Optional 'dense' extra not installed. An on-disk embedding
            # index without a usable embedder is just as unusable as no
            # index at all -- degrade to None (callers fall back to []/0
            # bonus) rather than crashing.
            _shared_embedder = None
    return _shared_embedder


@dataclass
class EmbeddingIndex:
    entries: list[EmbeddingEntry]
    vectors: np.ndarray  # shape (N, D), float32, L2-normalized rows

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        paths: set[str] | None = None,
    ) -> list[tuple[float, EmbeddingEntry]]:
        # The persisted index covers the whole repo (built once at `ant index`
        # time, like symbols.json), but a caller normally wants results
        # scoped to one worker's territory. `paths` restricts the search to
        # that subset before ranking, rather than ranking globally and hoping
        # enough in-territory hits survive a truncated top-`limit` cut.
        if not self.entries:
            return []
        indices = range(len(self.entries))
        if paths is not None:
            indices = [i for i in indices if self.entries[i].path in paths]
            if not indices:
                return []
        query = np.asarray(query_vector, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm
        pool = self.vectors[list(indices)]
        scores = pool @ query
        order = np.argsort(scores)[::-1][:limit]
        return [(float(scores[i]), self.entries[indices[i]]) for i in order]

    def save(self, index_dir: Path, key: str) -> None:
        # Written to temp names and atomically replaced into place, entries
        # before vectors: a background build (see build_and_cache_async)
        # can be saving this at the same moment another thread/process
        # calls load() for the same key. load() requires both final-named
        # files to exist, so a reader can only ever observe "not built yet"
        # or "fully built" -- never a half-written, corrupt-looking index.
        index_dir.mkdir(parents=True, exist_ok=True)
        manifest = [
            {
                "path": entry.path,
                "line_start": entry.line_start,
                "line_end": entry.line_end,
                "quote": entry.quote,
            }
            for entry in self.entries
        ]
        entries_tmp = index_dir / f"{key}.entries.json.tmp"
        # Must already end in .npy: np.save appends that suffix to any name
        # that doesn't already have it, so a ".vectors.npy.tmp" name would
        # silently become ".vectors.npy.tmp.npy" instead of the path below.
        vectors_tmp = index_dir / f"{key}.vectors.tmp.npy"
        entries_tmp.write_text(json.dumps(manifest), encoding="utf-8")
        np.save(vectors_tmp, self.vectors)
        entries_tmp.replace(index_dir / f"{key}.entries.json")
        vectors_tmp.replace(index_dir / f"{key}.vectors.npy")

    @classmethod
    def load(cls, index_dir: Path, key: str) -> EmbeddingIndex | None:
        entries_path = index_dir / f"{key}.entries.json"
        vectors_path = index_dir / f"{key}.vectors.npy"
        if not entries_path.exists() or not vectors_path.exists():
            return None
        manifest = json.loads(entries_path.read_text(encoding="utf-8"))
        entries = [EmbeddingEntry(**item) for item in manifest]
        vectors = np.load(vectors_path)
        return cls(entries=entries, vectors=vectors)


def _embed_entries(
    entries: list[EmbeddingEntry], texts: list[str], embedder: DenseEmbedder, *, verbose: bool
) -> EmbeddingIndex:
    if not entries:
        return EmbeddingIndex(entries=[], vectors=np.zeros((0, 0), dtype=np.float32))

    # Embedded in visible batches, not one call over the whole corpus: for a
    # large territory at this model's CPU throughput this step is the only
    # part of dense indexing that takes more than a few seconds, and a
    # silent multi-minute call with no output is indistinguishable from a
    # hang.
    batch_size = DEFAULT_SCORING_CONFIG.dense.embed_batch_size
    batches: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = embedder.embed(texts[start : start + batch_size])
        batches.extend(batch)
        if verbose:
            print(
                f"Embedded {min(start + batch_size, len(texts))}/{len(texts)} chunks.",
                flush=True,
            )

    vectors = np.asarray(batches, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return EmbeddingIndex(entries=entries, vectors=vectors / norms)


def build_embedding_index(
    repo_root: Path, files: list[str], embedder: DenseEmbedder, *, verbose: bool = True
) -> EmbeddingIndex:
    """One embedding per symbol (class/function), not per paragraph region.

    A worker's own territory can still have thousands of lines even though
    the worker's own file list is short -- chunking every file into
    paragraph/definition-sized regions (the same granularity BM25 uses)
    scaled with file *density*, not just file count, and a handful of large,
    dense source files could still produce thousands of chunks. Symbol
    count is a tighter, more natural bound: it grows with how much code
    there actually *is* to point to, not with an arbitrary line-window size,
    and BM25/navigate() already handle line-precise lexical matching within
    a symbol once dense_search has pointed at the right one.
    """
    from ant.tools.symbol_index import build_symbol_index

    index = build_symbol_index(repo_root, files)
    entries: list[EmbeddingEntry] = []
    texts: list[str] = []
    # Cache each file's lines once instead of re-reading+re-splitting it from
    # disk for every symbol defined in it -- index.definitions is one entry
    # per symbol, so a file with N functions was being read N times.
    lines_by_path: dict[str, list[str]] = {}
    for definition in index.definitions:
        if definition.path not in lines_by_path:
            path = repo_root / definition.path
            try:
                lines_by_path[definition.path] = path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
            except OSError:
                lines_by_path[definition.path] = []
        lines = lines_by_path[definition.path]
        if not lines:
            continue
        start = max(1, definition.line)
        max_lines = DEFAULT_SCORING_CONFIG.dense.symbol_snippet_max_lines
        end = min(len(lines), max(definition.end_line, definition.line), start + max_lines)
        text = "\n".join(lines[start - 1 : end]).strip()
        if not text:
            continue
        entries.append(
            EmbeddingEntry(path=definition.path, line_start=start, line_end=end, quote=text[:2400])
        )
        texts.append(f"{definition.qualname or definition.name}\n{text}")

    return _embed_entries(entries, texts, embedder, verbose=verbose)


def build_or_extend_repo_index(
    repo_root: Path,
    files: list[str],
    embedder: DenseEmbedder,
    existing: EmbeddingIndex | None,
    *,
    verbose: bool = True,
) -> EmbeddingIndex | None:
    """Return `existing` extended with embeddings for whichever of `files`
    it doesn't already cover, embedding only that delta -- not `files` in
    full, and not the rest of `existing`'s entries again.

    Returns `existing` unchanged (same object) when every file in `files` is
    already covered -- callers can compare by identity to skip a redundant
    save.
    """
    already = {entry.path for entry in existing.entries} if existing else set()
    missing = [f for f in files if f not in already]
    if not missing:
        return existing
    fresh = build_embedding_index(repo_root, missing, embedder, verbose=verbose)
    if not fresh.entries:
        return existing
    if existing is None or not existing.entries:
        return fresh
    return EmbeddingIndex(
        entries=[*existing.entries, *fresh.entries],
        vectors=np.concatenate([existing.vectors, fresh.vectors], axis=0),
    )


# One shared queue + one persistent worker thread for every background
# build in the process, not one thread per caller. Spawning a fresh thread
# per caller let N newly-created workers (e.g. right after evolve_workers
# specializes/births several at once) all embed concurrently -- but
# onnxruntime inference is CPU-bound, so N threads competing for the same
# cores made each individual build slower, not faster, sometimes badly
# enough that none of them finished inside a short eval pass (confirmed: a
# build measured at ~170s in isolation still hadn't produced a cache after
# several minutes of contention with other concurrent builds). A single
# queue processes one job at a time, so every build gets the full machine.
#
# Every job targets the same on-disk index (REPO_INDEX_KEY, almost always --
# WORKER_CARDS_KEY is the only other caller and that index is small enough
# to never go through this queue), so `existing` is deliberately *not*
# captured at enqueue time and threaded through the queue: two jobs enqueued
# close together would otherwise both build against the same stale snapshot,
# and whichever saves second would silently overwrite the first job's
# additions. The drain thread reloads from disk immediately before each
# build instead, so job 2 always sees job 1's already-saved output and only
# ever computes embeddings for whatever is *still* actually missing.
_BuildJob = tuple[Path, list[str], Path, str, DenseEmbedder, tuple[str, str, tuple[str, ...]]]
_build_queue: Queue[_BuildJob] = Queue()
_queue_worker_started = False
_queue_worker_lock = threading.Lock()
_inflight_builds: set[tuple[str, str, tuple[str, ...]]] = set()
_inflight_lock = threading.Lock()
# Guards the load-merge-save sequence for a repo-wide index against a race
# between the background drain thread and a synchronous warm_dense_cache()
# call happening in the same process at the same time.
_repo_index_lock = threading.Lock()


def _ensure_queue_worker_started() -> None:
    global _queue_worker_started
    with _queue_worker_lock:
        if _queue_worker_started:
            return
        _queue_worker_started = True

        def _drain() -> None:
            while True:
                repo_root, files, index_dir, key, embedder, identity = _build_queue.get()
                try:
                    with _repo_index_lock:
                        existing = EmbeddingIndex.load(index_dir, key)
                        updated = build_or_extend_repo_index(
                            repo_root, files, embedder, existing, verbose=False
                        )
                        if updated is not None and updated is not existing:
                            updated.save(index_dir, key)
                except Exception as exc:  # noqa: BLE001 - nothing else observes
                    # this thread; swallowing silently would make a failed
                    # build indistinguishable from "still queued behind
                    # others", which is exactly the failure mode this print
                    # exists to make visible.
                    print(f"[dense] background build failed for {key}: {exc!r}", flush=True)
                finally:
                    with _inflight_lock:
                        _inflight_builds.discard(identity)
                    _build_queue.task_done()

        threading.Thread(target=_drain, daemon=True).start()


def build_and_cache_in_background(
    repo_root: Path, files: list[str], index_dir: Path, key: str, embedder: DenseEmbedder
) -> None:
    """Queue an extend-in-place build of the shared index at `key` for the
    single background worker and return immediately, without blocking the
    caller.

    dense_search() calling this instead of building synchronously is the
    fix for a real failure mode: a worker's territory can contain enough
    code that even the (much cheaper, post-symbol-level-granularity, and
    now delta-only) build takes long enough that a query would otherwise
    sit waiting on it. Returning [] for the round that triggers the build,
    rather than making that round's question wait, means dense retrieval is
    never worse than "not available yet for these files" -- it can never
    turn into "this specific query now takes an extra N minutes."

    Deduping in-flight work by the caller's exact file list (not just
    index_dir/key) matters now that every caller shares one on-disk index:
    two different workers' territories legitimately need different files
    embedded, so both must be allowed to queue even while the other's job
    is still running -- only a genuinely repeated request for the same
    files should be dropped.
    """
    identity = (str(index_dir), key, tuple(sorted(files)))
    with _inflight_lock:
        if identity in _inflight_builds:
            return
        _inflight_builds.add(identity)
    _ensure_queue_worker_started()
    _build_queue.put((repo_root, files, index_dir, key, embedder, identity))


def warm_dense_cache(
    repo_root: Path,
    index_dir: Path,
    workers: list[WorkerCard],
    embedder: DenseEmbedder,
) -> list[str]:
    """Synchronously extend the shared repo-wide embedding index with
    whatever files any of `workers` still need, in the calling
    thread/process -- not queued to the background worker.

    For live/interactive use (`ant ask`), lazy background building is the
    right default: no single query should ever wait on it. For a
    controlled experiment (e.g. comparing colony state before/after
    `evolve_workers`), the opposite is true -- you want a *deterministic,
    fully-built* colony before measuring, not "however far the background
    queue happened to get before the eval pass ended". Call this right
    after evolve_workers (or before re-running an eval pass) to get that
    guarantee.

    All workers' missing files are embedded in one pass, not one call per
    worker: two workers sharing files (e.g. a birth worker and the parents
    it was coalesced from) would otherwise embed those shared files twice
    in the same warm-up. Returns the ids of the workers that had at least
    one file embedded for the first time by this call.
    """
    with _repo_index_lock:
        existing = EmbeddingIndex.load(index_dir, REPO_INDEX_KEY)
        already = {entry.path for entry in existing.entries} if existing else set()

        built: list[str] = []
        missing_files: list[str] = []
        seen: set[str] = set()
        for worker in workers:
            worker_missing = [f for f in worker.files if f not in already]
            if worker_missing:
                built.append(worker.id)
            for f in worker_missing:
                if f not in seen:
                    seen.add(f)
                    missing_files.append(f)

        if not missing_files:
            return built

        print(
            f"[dense] warming {len(missing_files)} files needed by "
            f"{len(built)} workers...",
            flush=True,
        )
        updated = build_or_extend_repo_index(
            repo_root, missing_files, embedder, existing, verbose=True
        )
        if updated is not None and updated is not existing:
            updated.save(index_dir, REPO_INDEX_KEY)
    return built


def build_worker_card_index(workers: list[WorkerCard], embedder: DenseEmbedder) -> EmbeddingIndex:
    """One embedding per worker card (its name/responsibilities/searchable
    terms), not per code chunk -- a repo with thousands of symbols still only
    has as many workers as it has territories (dozens, not thousands), so
    this is cheap enough to build eagerly at `ant index` time and use as a
    semantic routing signal, unlike full chunk-level embedding.
    """
    entries: list[EmbeddingEntry] = []
    texts: list[str] = []
    for worker in workers:
        text = " ".join(
            [worker.name, *worker.responsibilities, *worker.searchable_terms]
        ).strip()
        if not text:
            continue
        entries.append(EmbeddingEntry(path=worker.id, line_start=0, line_end=0, quote=text[:2400]))
        texts.append(text)

    return _embed_entries(entries, texts, embedder, verbose=False)
