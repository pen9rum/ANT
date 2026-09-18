"""S2G-RAG on the REPOSITORY-QA substrate (RepoProbe-Python, SWE-QA-Pro).

This is NOT a fork of the algorithm. `S2GRAGRepoAdapter` subclasses
`ant.external_wrappers.s2g_rag.S2GRAGAdapter` and overrides exactly the
three substrate hooks that class exposes -- which retriever wraps the
candidate pool, which answer prompt is dispatched, and how the parsed
Answer/Rationale pair becomes `final_answer`. The S2G-Judge (verbatim
`SUFF_SYSTEM_PROMPT` schema), the gap-to-query mechanism (target+slot ->
description fallback, K=3 phrases appended to the original question), the
sentence-level Evidence Extractor (verbatim `SELECTOR_SYSTEM_PROMPT`,
return_top_k=6, max_sents_per_doc=40), `max_turns=4`, `top_docs=6`, the
append-only Evidence Context, the first-turn forced-retrieval override and
the termination logic are all inherited unchanged, byte-for-byte, from the
document-track implementation. There is one algorithm in this codebase.

WHY A SEPARATE RETRIEVER (the fairness requirement)
---------------------------------------------------
The repo-QA track's already-frozen baselines source their candidate pool
identically, and this file reuses that exact pattern -- read from the real
code, not assumed:

- `ant.agents.retrieval.RetrievalAgent` (Sparse Retrieval, repo-QA):
  `EvalRepoEnvironment(environment_root)` -> `iter_files()` ->
  `LocalSearchTool.search(query, all_files, limit=8)`.
- `ant.agents.dense_retrieval_repo.DenseRetrievalRepoAgent`: the same
  `EvalRepoEnvironment.iter_files()` universe, chunked with the same
  `ant.tools.local._retrieval_regions` block splitter.

`RepoCorpusRetriever` below goes through `LocalSearchTool.search` over
that same `EvalRepoEnvironment.iter_files()` universe -- the identical
BM25 + symbol-path + Reciprocal-Rank-Fusion primitive, the identical file
universe, the identical `_retrieval_regions` chunk boundaries. No separate
index is built, no gold file list is consulted, and nothing here touches
or imports any GraphRAG/RepoDistill-specific index.

THE RETRIEVAL UNIT IS A CHUNK, NOT A FILE (disclosed, and deliberate)
--------------------------------------------------------------------
Upstream S2G-RAG's `top_docs=6` counts CORPUS DOCUMENTS, and its corpus is
a Pyserini Wikipedia index whose documents are short passages (~100-200
tokens) -- the same unit the document track's `DocumentRecord`s are. A
repository "document" (a whole source file) can be up to
`repo_scope.MAX_TEXT_FILE_BYTES` = 1 MiB, three to four orders of
magnitude larger, so treating a file as one retrieval unit would (a) be a
far poorer analogue of upstream's own unit and (b) make the Evidence
Extractor's numbered-sentence prompt enormous and silently truncated by
`max_sents_per_doc=40`. The retrieval unit here is therefore the
`_retrieval_regions` chunk `LocalSearchTool.search` already returns --
which is ALSO exactly the candidate unit Sparse and Dense Retrieval score
on this track, so the candidate pool is identical to theirs, not merely
similar.

SENTENCE SEGMENTATION ON CODE (verified, not assumed)
-----------------------------------------------------
`split_wiki_sentences` splits paragraphs on `(?<=[.!?])\\s+|[\\n]+`. On
source code the `[\\n]+` alternative dominates, so each LINE becomes one
"sentence" and the Evidence Extractor performs line-level selection over
the retrieved chunks. That is a sensible and well-defined behavior for
code, but it is an emergent property of upstream's regex rather than
something upstream designed for code, so it is called out explicitly here
and covered by its own test rather than left to be discovered later.

NO GOLD LEAKAGE. `run()` (inherited) and `_build_retriever` (here) read
only `example.question` and `example.metadata["repo"...]`-style corpus
locators. Neither ever reads `example.reference` (the gold answer for both
benchmarks) nor `example.metadata["checklist"]` (RepoProbe's gold scoring
rubric, which enumerates the facts a correct answer must contain and is by
far this track's sharpest leakage surface). Scoring reads those, after the
prediction is frozen -- `RepoProbeAdapter.score` / `SweQaProAdapter.score`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.repo_scope import EvalRepoEnvironment
from ant.external_wrappers.s2g_rag import (
    DEVIATIONS,
    NO_RESULTS,
    REMOVE_REPEAT_DOCS,
    TOP_DOCS,
    S2GRAGAdapter,
    formatting2,
)
from ant.providers.openai_provider import SYNTHESIS_MAX_OUTPUT_TOKENS
from ant.tools.local import LocalSearchTool

# The repo-QA track's own answer-generation budget. Every already-frozen
# repo-QA baseline (`RetrievalAgent`, `DenseRetrievalRepoAgent`, ANT) ends
# in `provider.synthesize(...)`, which uses
# `ant.providers.openai_provider.SYNTHESIS_MAX_OUTPUT_TOKENS`. Imported
# rather than re-typed so this can never silently drift away from the
# budget the other methods actually get.
REPO_ANSWER_MAX_OUTPUT_TOKENS = SYNTHESIS_MAX_OUTPUT_TOKENS


# ---------------------------------------------------------------------------
# The repo-QA answer prompt.
#
# Upstream S2G-RAG ALREADY dispatches this prompt per dataset -- see
# `inference/inference_bm25.py::get_answer_system_prompt`, which returns
# `TRIVIAQA_FORCE_ANSWER_PROMPT` for TriviaQA and `force_answer_prompt`
# otherwise, and likewise `get_selector_system_prompt`. A substrate-specific
# answer prompt is therefore upstream's own design pattern, not an invention
# here; this is the repo-QA member of that same dispatch.
#
# The single substantive change from `force_answer_prompt` is dropping
# "Prefer the shortest exact answer span that fully answers the question."
# That instruction exists because HotpotQA/2Wiki/MuSiQue are EM/F1-scored
# against a short gold span. RepoProbe and SWE-QA-Pro are instead scored by
# an LLM judge against a narrative rubric (RepoProbe: knowledge_score/9 +
# clarity_score/1 over a checklist; SWE-QA-Pro: a 5-axis
# correctness/completeness/relevance/clarity/reasoning rubric). Emitting a
# terse span into those judges would score near-floor for reasons that have
# nothing to do with whether the judge->gap->retrieve loop found the right
# code -- it would misrepresent the method, not measure it. The
# `Answer:`/`Rationale:` output contract is kept VERBATIM so upstream's own
# `extract_final_answer_and_rationale` parser applies unchanged.
# ---------------------------------------------------------------------------

repo_force_answer_prompt = (
    "You are a knowledgeable software repository question-answering assistant. "
    "Based on the context provided (if any), answer the following question about "
    "the codebase. Ground every claim in the retrieved evidence, and cite the "
    "file paths the evidence came from. Give a complete, well-organized "
    "explanation rather than a single short span. "
) + formatting2


REPO_DEVIATIONS: list[dict[str, str]] = [
    {
        "official_setting": (
            "Retrieval over a whole-Wikipedia Pyserini BM25 index; top_docs=6 counts "
            "corpus DOCUMENTS, which in that index are short Wikipedia passages."
        ),
        "adapted_setting": (
            "BM25 + symbol-path + RRF via ant.tools.local.LocalSearchTool over the "
            "EvalRepoEnvironment.iter_files() universe of this question's own pinned "
            "repository checkout. top_docs=6 counts _retrieval_regions CHUNKS -- the "
            "same candidate unit Sparse and Dense Retrieval already score on this track, "
            "and a far closer size-analogue of an upstream Wikipedia passage than a "
            "whole source file (which may be up to 1 MiB) would be."
        ),
        "reason": (
            "The fairness constraint requires the same question-visible candidate pool as "
            "every other method on this benchmark. A whole-file unit would additionally "
            "make the Evidence Extractor's numbered-sentence prompt enormous and would be "
            "silently truncated by upstream's own max_sents_per_doc=40 cap."
        ),
    },
    {
        "official_setting": (
            "force_answer_prompt instructs 'Prefer the shortest exact answer span that "
            "fully answers the question' and the answer call is capped at 128 new tokens."
        ),
        "adapted_setting": (
            "repo_force_answer_prompt drops the shortest-span instruction and asks for a "
            "complete, evidence-grounded, file-citing explanation; the answer call is "
            "capped at SYNTHESIS_MAX_OUTPUT_TOKENS (8192), the same budget every other "
            "repo-QA baseline's own answer call gets. The Answer:/Rationale: output "
            "contract and its parser are unchanged."
        ),
        "reason": (
            "RepoProbe and SWE-QA-Pro are LLM-judge-scored against narrative rubrics, not "
            "EM/F1-scored against a short gold span. Upstream ITSELF dispatches this "
            "prompt per dataset (get_answer_system_prompt returns a different prompt for "
            "TriviaQA), so a substrate-specific answer prompt is upstream's own pattern. "
            "Alternative considered and rejected: replacing the answer stage with the "
            "suite's shared provider.synthesize() -- that would hold answer generation "
            "constant across methods but would delete one of S2G-RAG's own components, a "
            "larger fidelity loss than swapping the prompt text within it."
        ),
    },
    {
        "official_setting": (
            "main_batch records only the parsed `Answer:` field as `Reasoner Answer`."
        ),
        "adapted_setting": (
            "final_answer is the parsed Answer text with the parsed Rationale appended "
            "when present (raw response as fallback if the format was not followed) -- "
            "nothing the method produced is discarded before the judge sees it."
        ),
        "reason": (
            "Upstream drops the rationale because EM/F1 scores a short span. A rubric "
            "judge scores reasoning and completeness explicitly, so discarding the "
            "rationale would throw away output the rubric is asking about."
        ),
    },
    {
        "official_setting": (
            "split_wiki_sentences is a Wikipedia prose sentence splitter."
        ),
        "adapted_setting": (
            "Applied unchanged to code chunks, where its `[\\n]+` alternative makes each "
            "LINE a selectable unit, so the Evidence Extractor performs line-level "
            "selection over retrieved code."
        ),
        "reason": (
            "Emergent from upstream's own regex rather than designed for code. Disclosed "
            "and test-covered rather than left implicit; no change was made to the "
            "splitter, since changing it would be a real algorithmic deviation."
        ),
    },
]


@dataclass(frozen=True)
class RepoChunk:
    """One retrieval unit on the repo substrate. Deliberately the same
    duck-type as `DocumentRecord` (`doc_id`/`title`/`text`) so the inherited
    loop, evidence list and metadata need no substrate branching -- and,
    like `DocumentRecord`, it structurally carries NO relevance/gold field
    of any kind."""

    doc_id: str  # "path:line_start-line_end"
    title: str  # "path:line_start-line_end", shown to the Evidence Extractor
    text: str


class RepoCorpusRetriever:
    """Chunk-level BM25 retrieval over exactly the repository this question
    makes visible, through the same `LocalSearchTool.search` primitive and
    the same `EvalRepoEnvironment.iter_files()` universe the repo-QA Sparse
    Retrieval baseline already uses.

    Same `(titles, texts, doc_ids)` triple, same `remove_repeat_docs`
    filtering against already-seen ids, and the same `"No results found."`
    sentinel as upstream's own `bm25_search_batch` and as the document
    track's `SharedCorpusRetriever` -- so the inherited loop cannot tell the
    two substrates apart.

    `corpus` is populated lazily from whatever has actually been retrieved
    (a repository has far too many chunks to enumerate eagerly, and nothing
    in the loop needs the full enumeration).
    """

    def __init__(
        self,
        environment_root: Path,
        *,
        remove_repeat_docs: bool = REMOVE_REPEAT_DOCS,
        search_limit_headroom: int = 50,
    ) -> None:
        self.environment = EvalRepoEnvironment(environment_root)
        self.search_tool = LocalSearchTool(environment_root)
        self.files = [
            str(path.relative_to(self.environment.root))
            for path in self.environment.iter_files()
        ]
        self.corpus: dict[str, RepoChunk] = {}
        self.remove_repeat_docs = remove_repeat_docs
        self.search_limit_headroom = search_limit_headroom
        self.search_calls = 0

    @staticmethod
    def chunk_id(path: str, line_start: int, line_end: int) -> str:
        return f"{path}:{line_start}-{line_end}"

    def search(
        self, query: str, past_doc_ids: list[str], k: int = TOP_DOCS
    ) -> tuple[list[str], list[str], list[str]]:
        past = set(past_doc_ids) if self.remove_repeat_docs else set()
        self.search_calls += 1
        # Upstream asks its searcher for max(k, 50) hits then filters down;
        # the same head-room is requested here.
        hits = self.search_tool.search(
            str(query or "").strip(), self.files, limit=max(k, self.search_limit_headroom)
        )

        titles: list[str] = []
        texts: list[str] = []
        doc_ids: list[str] = []
        seen: set[str] = set()
        for hit in hits:
            doc_id = self.chunk_id(hit.path, hit.line_start, hit.line_end)
            if doc_id in seen:
                continue
            if self.remove_repeat_docs and doc_id in past:
                continue
            chunk = RepoChunk(doc_id=doc_id, title=doc_id, text=hit.quote or "")
            if not chunk.text.strip():
                continue
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


class S2GRAGRepoAdapter(S2GRAGAdapter):
    """S2G-RAG for the repository substrate (RepoProbe, SWE-QA-Pro).

    Registered under its OWN distinct agent name so it can never collide
    with, or be mistaken for, the document-track `s2g_rag` adapter -- the
    two report separately and are configured separately.
    """

    name = "s2g_rag_repo"
    corpus_substrate = "repository_chunks"
    ANSWER_SYSTEM_PROMPT = repo_force_answer_prompt
    answer_max_output_tokens = REPO_ANSWER_MAX_OUTPUT_TOKENS

    def deviations(self) -> list[dict[str, str]]:
        # The judge substitution and the batching/pysbd/inline-gold-scoring
        # entries are substrate-independent and carry over verbatim; the
        # document track's own corpus entry is replaced by this track's.
        carried = [
            entry
            for entry in DEVIATIONS
            if not entry["official_setting"].startswith("Retrieval over a whole-Wikipedia")
        ]
        return [*carried, *REPO_DEVIATIONS]

    def _build_retriever(self, example: TaskExample, environment_root: Path) -> Any:
        # `example` is deliberately unused: on this substrate the entire
        # candidate pool is the pinned repository checkout at
        # `environment_root`, which the benchmark adapter's own
        # prepare_environment() produced. Reading anything off the example
        # here would be the leakage surface, so nothing is read.
        del example
        return RepoCorpusRetriever(
            environment_root, remove_repeat_docs=self.remove_repeat_docs
        )

    def _assemble_final_answer(self, answer: str, rationale: str, raw: str) -> str:
        """Rubric judges score reasoning/completeness explicitly, so the
        rationale is kept rather than dropped. Falls back to the raw
        response if the model did not follow the Answer:/Rationale: format
        at all (upstream's parser returns its own sentinel strings then)."""
        if answer == "Answer not found":
            return (raw or "").strip()
        if rationale and rationale != "Rationale not found":
            return f"{answer}\n\n{rationale}"
        return answer


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(S2GRAGRepoAdapter())
