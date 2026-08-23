from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ant.domain import WorkerCard

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Worker-card entries reuse EmbeddingEntry's (path, quote) shape rather than
# a separate class: `path` holds the worker id and line_start/line_end are
# unused (0), since search()/save()/load() only ever touch `path` as an
# opaque grouping/identity key, never as a real file path.
WORKER_CARDS_KEY = "cards"


def territory_key(files: list[str]) -> str:
    """Stable cache key for a worker's own file scope, used to name its
    lazily-built, on-disk-cached chunk embedding index. Two workers that
    happen to own the exact same file set intentionally share one cache
    entry -- there is nothing else that would distinguish them anyway.
    """
    joined = "|".join(sorted(files))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


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
        (index_dir / f"{key}.entries.json").write_text(json.dumps(manifest), encoding="utf-8")
        np.save(index_dir / f"{key}.vectors.npy", self.vectors)

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


def build_embedding_index(
    repo_root: Path, files: list[str], embedder: DenseEmbedder
) -> EmbeddingIndex:
    # Reuses the exact same chunking as BM25 (`_retrieval_regions`) so dense
    # and lexical hits are comparable units, not different-shaped candidates.
    from ant.tools.local import _retrieval_regions

    entries: list[EmbeddingEntry] = []
    texts: list[str] = []
    for relative in files:
        path = repo_root / relative
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for start, block in _retrieval_regions(lines):
            text = "\n".join(block).strip()
            if not text:
                continue
            end = start + len(block) - 1
            entries.append(
                EmbeddingEntry(path=relative, line_start=start, line_end=end, quote=text[:2400])
            )
            texts.append(text)

    if not entries:
        return EmbeddingIndex(entries=[], vectors=np.zeros((0, 0), dtype=np.float32))

    # Embedded in visible batches, not one call over the whole corpus: for a
    # few thousand chunks at this model's CPU throughput this step is the
    # only part of `ant index --dense` that takes more than a few seconds,
    # and a silent multi-minute call with no output is indistinguishable
    # from a hang.
    batch_size = 256
    batches: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = embedder.embed(texts[start : start + batch_size])
        batches.extend(batch)
        print(f"Embedded {min(start + batch_size, len(texts))}/{len(texts)} chunks.", flush=True)

    vectors = np.asarray(batches, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return EmbeddingIndex(entries=entries, vectors=vectors / norms)


def build_worker_card_index(workers: list[WorkerCard], embedder: DenseEmbedder) -> EmbeddingIndex:
    """One embedding per worker card (its name/responsibilities/searchable
    terms), not per code chunk -- a repo with thousands of chunks still only
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

    if not entries:
        return EmbeddingIndex(entries=[], vectors=np.zeros((0, 0), dtype=np.float32))

    vectors = np.asarray(embedder.embed(texts), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return EmbeddingIndex(entries=entries, vectors=vectors / norms)
