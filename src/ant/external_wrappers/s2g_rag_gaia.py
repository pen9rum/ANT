"""S2G-RAG on GAIA -- `name = "s2g_rag_gaia"`.

Same pattern `s2g_rag_repo.py` already established for the repository
substrate: subclass `S2GRAGAdapter` and override only the two substrate
hooks it exposes (`_build_retriever`, `ANSWER_SYSTEM_PROMPT`). The
S2G-Judge, gap-to-query mechanism, sentence-level Evidence Extractor,
`max_turns=4`, `top_docs=6`, append-only Evidence Context and termination
logic are inherited unchanged, byte-for-byte -- there is one algorithm in
this codebase.

THE RETRIEVAL UNIT IS A SEARCH-ENGINE SNIPPET, NOT A FETCHED PAGE
(disclosed deviation, matching this codebase's own precedent of
disclosing the repo-QA track's file-vs-chunk substitution): each S2G-RAG
turn can issue several gap-driven queries across up to 4 turns, and
fetching+chunking full pages for every one of them would multiply this
baseline's real-network cost several-fold over Sparse/Dense Retrieval's
own single search call for no clear fidelity gain -- upstream's own
corpus documents (short Wikipedia passages) are already snippet-sized,
so a search-engine snippet (title + URL + ~200-char excerpt from
`GaiaToolRegistry.search()`) is a closer analogue of upstream's own
retrieval unit than a multi-KB fetched page would be. `doc_id` is the
URL; the Evidence Extractor's sentence-level selection operates over the
snippet text.

NO GOLD LEAKAGE: `_build_retriever` reads only `environment_root` (this
task's `GaiaEnvironment`, itself built from task-observable metadata) --
never `example.reference` or `example.metadata["_audit_only"]`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ant.agents.gaia_shared import GAIA_ANSWER_FORMAT_INSTRUCTIONS, build_registry
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.gaia_scope import GaiaEnvironment
from ant.external_wrappers.s2g_rag import (
    DEVIATIONS,
    NO_RESULTS,
    REMOVE_REPEAT_DOCS,
    TOP_DOCS,
    S2GRAGAdapter,
    formatting2,
)
from ant.providers.openai_provider import SYNTHESIS_MAX_OUTPUT_TOKENS

GAIA_ANSWER_MAX_OUTPUT_TOKENS = SYNTHESIS_MAX_OUTPUT_TOKENS

gaia_force_answer_prompt = GAIA_ANSWER_FORMAT_INSTRUCTIONS + "\n\n" + formatting2

GAIA_DEVIATIONS: list[dict[str, str]] = [
    {
        "official_setting": (
            "Retrieval over a whole-Wikipedia Pyserini BM25 index; top_docs=6 counts "
            "corpus DOCUMENTS, which in that index are short Wikipedia passages."
        ),
        "adapted_setting": (
            "GaiaToolRegistry.search() (Tavily-backed, DuckDuckGo fallback) over the "
            "open web; top_docs=6 counts SEARCH-ENGINE SNIPPETS (title/URL/excerpt), "
            "not fetched pages -- a closer size-analogue of an upstream Wikipedia "
            "passage than a multi-KB fetched page, and consistent with Sparse "
            "Retrieval's own snippet-based candidate pool on this substrate."
        ),
        "reason": (
            "GAIA has no local corpus; the open web is the corpus, reached only "
            "through this substrate's one retrieval primitive. Fetching full pages "
            "for every one of S2G-RAG's up to 4 turns' worth of gap-driven queries "
            "would multiply real-network cost several-fold over the other retrieval "
            "baselines' own single search call, for no clear fidelity gain."
        ),
    },
    {
        "official_setting": (
            "force_answer_prompt instructs 'Prefer the shortest exact answer span that "
            "fully answers the question' and the answer call is capped at 128 new tokens."
        ),
        "adapted_setting": (
            "gaia_force_answer_prompt uses GAIA's own official FINAL ANSWER: template "
            "instructions (verbatim GAIA_SYSTEM_PROMPT) instead, since GAIA's official "
            "scorer extracts an answer by looking for that literal template -- every "
            "other GAIA method in this suite threads the same instructions through its "
            "own answer prompt. The answer call is capped at SYNTHESIS_MAX_OUTPUT_TOKENS."
        ),
        "reason": (
            "GAIA is scored by the official quasi-exact-match scorer against a "
            "template-extracted answer, not EM/F1 against a bare span and not an LLM "
            "rubric judge -- a different substrate-specific answer contract than either "
            "the document or repo-QA track, so (as upstream's own get_answer_system_prompt "
            "already dispatches per-dataset) this is a third, substrate-specific prompt."
        ),
    },
]


@dataclass(frozen=True)
class GaiaSnippetChunk:
    """One retrieval unit on the GAIA substrate -- a search-engine hit.
    Same `(doc_id, title, text)` duck-type as `DocumentRecord`/`RepoChunk`
    so the inherited loop needs no substrate branching."""

    doc_id: str  # the hit's URL
    title: str
    text: str


class GaiaSearchRetriever:
    """Search-snippet retrieval over the open web via `GaiaToolRegistry`.
    See this module's own docstring for why the unit is a snippet, not a
    fetched page. `corpus` is populated lazily from whatever has actually
    been retrieved, matching `RepoCorpusRetriever`'s own convention."""

    def __init__(
        self,
        environment: GaiaEnvironment,
        example: TaskExample,
        *,
        remove_repeat_docs: bool = REMOVE_REPEAT_DOCS,
        search_limit_headroom: int = 20,
    ) -> None:
        # A throwaway TaskExample-shaped registry build: build_registry
        # only reads environment_root/file_name, both already fixed by
        # `environment`, so a dedicated environment-only constructor path
        # isn't needed -- reuse the shared helper directly.
        self.registry = build_registry(example, environment.task_root)
        self.corpus: dict[str, GaiaSnippetChunk] = {}
        self.remove_repeat_docs = remove_repeat_docs
        self.search_limit_headroom = search_limit_headroom
        self.search_calls = 0

    def search(
        self, query: str, past_doc_ids: list[str], k: int = TOP_DOCS
    ) -> tuple[list[str], list[str], list[str]]:
        past = set(past_doc_ids) if self.remove_repeat_docs else set()
        self.search_calls += 1
        hits = self.registry.search(
            str(query or "").strip(), limit=max(k, self.search_limit_headroom)
        )

        titles: list[str] = []
        texts: list[str] = []
        doc_ids: list[str] = []
        seen: set[str] = set()
        for hit in hits:
            doc_id = hit.url
            if not doc_id or doc_id in seen:
                continue
            if self.remove_repeat_docs and doc_id in past:
                continue
            text = f"{hit.title}\n{hit.snippet}".strip()
            if not text:
                continue
            chunk = GaiaSnippetChunk(doc_id=doc_id, title=hit.title or doc_id, text=text)
            seen.add(doc_id)
            self.corpus[doc_id] = chunk
            titles.append(chunk.title)
            texts.append(chunk.text)
            doc_ids.append(doc_id)
            if len(doc_ids) >= k:
                break

        if not doc_ids:
            return [NO_RESULTS], [NO_RESULTS], [""]
        return titles, texts, doc_ids


class S2GRAGGaiaAdapter(S2GRAGAdapter):
    """S2G-RAG for GAIA. Registered under its own distinct agent name so
    it never collides with the document-track/repo-QA `s2g_rag`/
    `s2g_rag_repo` adapters."""

    name = "s2g_rag_gaia"
    corpus_substrate = "gaia_web_snippets"
    ANSWER_SYSTEM_PROMPT = gaia_force_answer_prompt
    answer_max_output_tokens = GAIA_ANSWER_MAX_OUTPUT_TOKENS

    def deviations(self) -> list[dict[str, str]]:
        carried = [
            entry
            for entry in DEVIATIONS
            if not entry["official_setting"].startswith("Retrieval over a whole-Wikipedia")
            and "force_answer_prompt instructs" not in entry["official_setting"]
        ]
        return [*carried, *GAIA_DEVIATIONS]

    def _build_retriever(self, example: TaskExample, environment_root: Path) -> Any:
        environment = GaiaEnvironment(environment_root, example.metadata.get("file_name"))
        return GaiaSearchRetriever(environment, example, remove_repeat_docs=self.remove_repeat_docs)

    def _assemble_final_answer(self, answer: str, rationale: str, raw: str) -> str:
        """The INHERITED `_answer()` still parses upstream's own
        `Answer:`/`Rationale:` template via `extract_final_answer_and_rationale`
        (frozen, not overridden here -- there is one algorithm in this
        codebase) -- but `ANSWER_SYSTEM_PROMPT` above asks for GAIA's
        `FINAL ANSWER:` template instead, so that parse will ALWAYS report
        "Answer not found" here. That is expected, not a bug: this branch
        then returns the raw response verbatim, which does contain
        `FINAL ANSWER: ...` -- `GaiaAdapter.score()` extracts it from
        there, downstream, via its own `extract_final_answer`. Kept as a
        fallback-through rather than forking `_answer()` to parse GAIA's
        template directly, to preserve "the base class's turn loop is
        never modified" exactly as `s2g_rag_repo.py` does."""
        if answer == "Answer not found":
            return (raw or "").strip()
        if rationale and rationale != "Rationale not found":
            return f"{answer}\n\n{rationale}"
        return answer


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(S2GRAGGaiaAdapter())
