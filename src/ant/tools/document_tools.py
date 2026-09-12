"""Shared long-document/multi-document information-access primitives
(Section 3 of the long-context evaluation spec): `search`, `view`,
`navigate`. Used identically by the document variants of Retrieval and
Matched ReAct wherever scientifically appropriate -- "no special tools
only to ANT" -- and independently by ANT's own frozen `AutonomousWorker`
tool loop for `search` (the one primitive that loop already exposes; see
`ant.agents.ant_document_adapter`'s own module docstring for the disclosed
`view`/`navigate` asymmetry this does NOT try to paper over).

`search(query)` is not reimplemented here: it is `ant.tools.local.
LocalSearchTool.search()`, the exact same class/method every repository-QA
method already shares, called with `files` set to every materialized
document's own relative path. That function's own BM25 + symbol/path
fusion degrades gracefully on prose (no AST symbols in plain text, so the
symbol-path channel simply contributes nothing and BM25 alone drives
ranking) -- confirmed by this module's own tests, not assumed.

`view`/`navigate` ARE new here, since no existing tool reads a whole
document or steps between adjacent chunks: both are pure, deterministic
reads over an `EvalDocumentEnvironment` and a precomputed chunk list --
no LLM calls, no question-aware preprocessing, no benchmark-specific
heuristics, no supporting-fact/gold access of any kind.
"""
from __future__ import annotations

from ant.evaluation_suite.document_scope import DocumentChunk, EvalDocumentEnvironment


def view_document(environment: EvalDocumentEnvironment, doc_id: str) -> str:
    """Returns one document's full text (title + body, exactly as
    materialized -- verbatim, no truncation, no summarization). Returns an
    empty string for an unknown doc_id rather than raising, matching this
    evaluation suite's existing convention of a graceful, empty result for
    a bad/hallucinated tool argument (see LocalSearchTool.search()'s own
    `if not terms: return []`).
    """
    document = environment.document(doc_id)
    if document is None:
        return ""
    return f"Title: {document.title}\n\n{document.text}" if document.title else document.text


def navigate_chunk(
    chunks: list[DocumentChunk], global_chunk_index: int, direction: str
) -> DocumentChunk | None:
    """Steps to the adjacent chunk in a precomputed, document-boundary-
    respecting chunk list (see `document_scope.chunk_documents`).
    `direction` is "next" or "previous"; any other value, or stepping past
    either end of the list, returns None rather than raising or wrapping
    around -- a caller (ReAct agent) sees this as "no further chunk in
    that direction", not a crash.
    """
    if not 0 <= global_chunk_index < len(chunks):
        return None
    if direction == "next":
        target = global_chunk_index + 1
    elif direction == "previous":
        target = global_chunk_index - 1
    else:
        return None
    if not 0 <= target < len(chunks):
        return None
    return chunks[target]


def chunk_by_global_index(
    chunks: list[DocumentChunk], global_chunk_index: int
) -> DocumentChunk | None:
    """Direct lookup by global_chunk_index, used to resolve a ReAct agent's
    own navigate(chunk_id) argument (a string it must parse to an int) back
    to the chunk object before navigate_chunk can step from it."""
    for chunk in chunks:
        if chunk.global_chunk_index == global_chunk_index:
            return chunk
    return None
