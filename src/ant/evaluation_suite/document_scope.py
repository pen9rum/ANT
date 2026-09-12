"""Evaluation-only long-document / multi-document substrate, analogous in
spirit to `repo_scope.py`'s `EvalRepoEnvironment` for repository QA.

Mechanical no-leakage guarantee: `DocumentRecord` (what every inference
method can see) carries ONLY `doc_id`/`title`/`text`. Supporting-fact /
`is_supporting` annotations live in a SEPARATE structure
(`SupportingFactIndex`) that benchmark adapters build for their own
scoring/diagnostics and the Lost-in-the-Middle perturbation utility below --
nothing in this module ever writes that information into a materialized
document file, a `TaskExample`, or anything an agent's `run()` receives.
"""
from __future__ import annotations

import re
from pathlib import Path

import tiktoken
from pydantic import BaseModel, Field

from ant.environment.repo import RepoEnvironment

_ENCODING_NAME = "cl100k_base"
_TITLE_SLUG_RE = re.compile(r"[^A-Za-z0-9]+")


class DocumentRecord(BaseModel):
    """One document as an inference method may legitimately see it. No
    supporting-fact/gold field exists on this model at all -- not merely
    hidden, structurally absent."""

    doc_id: str
    title: str
    text: str


class DocumentChunk(BaseModel):
    """One deterministic, fixed-size slice of one document's text."""

    document_id: str
    chunk_index: int  # index within this document
    global_chunk_index: int  # index across the whole ordered document list
    text: str
    token_start: int
    token_end: int


def _doc_filename(index: int) -> str:
    # Purely positional, index-based -- no title text in the filename, so
    # the filename itself never hints at relevance/content.
    return f"doc_{index:04d}.txt"


def materialize_documents(documents: list[DocumentRecord], root: Path) -> dict[str, Path]:
    """Writes each document to `root/doc_{index:04d}.txt` in the EXACT
    order given (original benchmark document order is preserved -- this
    function never sorts, shuffles, or reorders by relevance), with
    content `"Title: {title}\\n\\n{text}"` so titles remain visible to any
    method that just reads file text, without needing special-casing per
    method. Text is written verbatim -- no summarization. Returns
    {doc_id: absolute_path}.
    """
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for index, doc in enumerate(documents):
        path = root / _doc_filename(index)
        content = f"Title: {doc.title}\n\n{doc.text}" if doc.title else doc.text
        path.write_text(content, encoding="utf-8")
        paths[doc.doc_id] = path.resolve()
    return paths


class EvalDocumentEnvironment(RepoEnvironment):
    """Drop-in `RepoEnvironment` substitute (same Liskov-substitutable
    contract as `EvalRepoEnvironment` for repositories) over a directory of
    materialized document files. Adds a `doc_index` for O(1) doc_id/
    relative-path resolution, used by the document search/view/navigate
    primitives (`ant.tools.document_tools`) -- but `iter_files()`/`root`
    alone are already enough for anything that only expects a plain
    `RepoEnvironment` (LocalSearchTool, ANT's own AutonomousWorker path).
    """

    def __init__(self, root: Path, documents: list[DocumentRecord]) -> None:
        object.__setattr__(self, "root", root.resolve())
        ordered_ids = [doc.doc_id for doc in documents]
        object.__setattr__(self, "_ordered_doc_ids", ordered_ids)
        object.__setattr__(self, "_by_doc_id", {doc.doc_id: doc for doc in documents})
        object.__setattr__(
            self,
            "_filename_by_doc_id",
            {doc.doc_id: _doc_filename(i) for i, doc in enumerate(documents)},
        )

    def iter_files(self) -> list[Path]:
        return [self.root / self._filename_by_doc_id[doc_id] for doc_id in self._ordered_doc_ids]

    def relative_path_for(self, doc_id: str) -> str:
        return self._filename_by_doc_id[doc_id]

    def doc_id_for_relative_path(self, relative_path: str) -> str | None:
        for doc_id, filename in self._filename_by_doc_id.items():
            if filename == relative_path or filename == Path(relative_path).name:
                return doc_id
        return None

    def document(self, doc_id: str) -> DocumentRecord | None:
        return self._by_doc_id.get(doc_id)

    def ordered_documents(self) -> list[DocumentRecord]:
        return [self._by_doc_id[doc_id] for doc_id in self._ordered_doc_ids]


def chunk_documents(
    documents: list[DocumentRecord], chunk_size_tokens: int
) -> list[DocumentChunk]:
    """Deterministic, stable-tokenization, fixed-size chunking that
    preserves document identity and order: chunk boundaries never cross a
    document boundary (a document's own final chunk may be shorter than
    chunk_size_tokens; it is never merged with the next document's text).
    Chunk size is NOT chosen based on task performance -- see call sites
    for the fixed, disclosed value each one uses.
    """
    encoding = tiktoken.get_encoding(_ENCODING_NAME)
    chunks: list[DocumentChunk] = []
    global_index = 0
    for doc in documents:
        tokens = encoding.encode(doc.text, disallowed_special=())
        if not tokens:
            chunks.append(
                DocumentChunk(
                    document_id=doc.doc_id,
                    chunk_index=0,
                    global_chunk_index=global_index,
                    text="",
                    token_start=0,
                    token_end=0,
                )
            )
            global_index += 1
            continue
        for local_index, start in enumerate(range(0, len(tokens), chunk_size_tokens)):
            end = min(start + chunk_size_tokens, len(tokens))
            chunks.append(
                DocumentChunk(
                    document_id=doc.doc_id,
                    chunk_index=local_index,
                    global_chunk_index=global_index,
                    text=encoding.decode(tokens[start:end]),
                    token_start=start,
                    token_end=end,
                )
            )
            global_index += 1
    return chunks


# ---------------------------------------------------------------------------
# Lost-in-the-Middle positional-perturbation utility (Section 15: implement
# only, do not run a formal evaluation with it this pass).
# ---------------------------------------------------------------------------

Position = str  # "early" | "middle" | "late"


class PositionalVariant(BaseModel):
    """One EARLY/MIDDLE/LATE reordering of the SAME document set. Question,
    answer, distractor set, and every document's text are all unchanged --
    only the order of `documents` differs from the original."""

    position: Position
    documents: list[DocumentRecord]
    supporting_doc_ids: list[str] = Field(default_factory=list)  # construction-only, diagnostic


def build_positional_variants(
    documents: list[DocumentRecord], supporting_doc_ids: list[str]
) -> list[PositionalVariant]:
    """Given the full document set (in original order) and the ids of the
    supporting (gold-relevant) documents -- known ONLY here, at experiment-
    CONSTRUCTION time, never passed through to any inference method -- moves
    the same supporting documents to the beginning, middle, or end of the
    (otherwise-unchanged, same-relative-order) document list.

    Distractor documents keep their relative order among themselves in
    every variant; only where the supporting block is spliced in changes.
    This is pure list rearrangement -- no document's text is ever touched.
    """
    supporting = [d for d in documents if d.doc_id in set(supporting_doc_ids)]
    distractors = [d for d in documents if d.doc_id not in set(supporting_doc_ids)]
    if not supporting:
        msg = "build_positional_variants requires at least one supporting document"
        raise ValueError(msg)

    variants = []
    # EARLY: supporting block first, then all distractors.
    variants.append(
        PositionalVariant(
            position="early",
            documents=[*supporting, *distractors],
            supporting_doc_ids=list(supporting_doc_ids),
        )
    )
    # MIDDLE: supporting block spliced into the middle of the distractor list.
    mid = len(distractors) // 2
    variants.append(
        PositionalVariant(
            position="middle",
            documents=[*distractors[:mid], *supporting, *distractors[mid:]],
            supporting_doc_ids=list(supporting_doc_ids),
        )
    )
    # LATE: all distractors first, then the supporting block.
    variants.append(
        PositionalVariant(
            position="late",
            documents=[*distractors, *supporting],
            supporting_doc_ids=list(supporting_doc_ids),
        )
    )
    return variants
