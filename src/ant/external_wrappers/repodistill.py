"""RepoDistill (No-Training variant): paper-faithful reimplementation.

Paper: Xin Yin, Zixiang Ding, Yiang Zhang, Qiang Wang, Rui Wang, Chao Ni,
Zhe Cui. "RepoDistill: Distilling Repository Knowledge through
Compression-Aware Budget Allocation and Policy Optimization." Findings of
the Association for Computational Linguistics: ACL 2026, pp. 4425-4443.
https://aclanthology.org/2026.findings-acl.217/

REIMPLEMENTED FROM THE PUBLISHED PAPER, NOT PORTED FROM CODE. The paper
cites an Anonymous-GitHub mirror for its implementation; that link is
dead/unreachable, and no trained checkpoint is published anywhere. The
paper PDF was fetched from the ACL Anthology and read directly (sections
2.1, 2.2, 2.3, and Appendix A.1.2) before any of this was written. Every
hyperparameter here is the paper's own stated value, and the two prompt
templates below are reproduced VERBATIM from the paper's own Appendix
A.1.2 (Table 4) -- not paraphrased. Where the paper leaves something
unspecified or where this environment cannot supply the paper's exact
component, the deviation is stated explicitly in an "official setting ->
adapted setting -> reason" note, in this file or in the two component
modules.

THIS IS THE "No Training" VARIANT ONLY. The paper's headline system adds
SFT-with-knowledge-distillation (section 2.3.2) and multi-turn GRPO
(section 2.3.3) to produce a trained Qwen3 compressor. Neither the SFT
trajectory dataset (human-curated, 10 sampled trajectories per instance,
two annotators) nor the trained checkpoints were released, and the paper
itself defines "RepoDistill (No Training)" as instantiating the same
multi-turn decision procedure by prompting a strong LLM directly
(section 2.3.2: "we first instantiate RepoDistill (No Training) using
Qwen3-Max ... For each instance in the training set, it assigns a
retention budget to every code snippet and produces the corresponding
final answer"). That is exactly what this module does, with GPT-4.1 in
place of Qwen3-Max per this project's standard-baseline-model rule.

--------------------------------------------------------------------
THE THREE COMPONENTS AND WHERE THEY LIVE
--------------------------------------------------------------------
1. GraphRAG (section 2.1)         -> `repodistill_graph.py`
2. CABA    (section 2.2)          -> `repodistill_caba.py`
3. CAPO, No-Training (section 2.3
   + Appendix A.1.2)              -> this file

--------------------------------------------------------------------
COST STRUCTURE -- what is free and what is paid
--------------------------------------------------------------------
This mirrors how `ant.agents.dense_retrieval_repo` documents the same
distinction, because it is the load-bearing fact for budgeting a run.

ONE-TIME PER REPO CHECKOUT (local, free, disk-cached under
`index_root / benchmark / repo_slug`, reused by EVERY question asked
against that same checkout):
  - tree-sitter parse of every .py file + dependency-graph construction
  - embedding of every graph node (DenseEmbedder / fastembed / ONNX)
Neither ever calls a paid API. RepoProbe-Python has 8 repos and 108
questions, so this is paid ~8 times, not 108.

PER QUESTION, LOCAL AND FREE:
  - one query embedding
  - Node Localization + Reasoning Chain Mining + Re-Ranking (numpy)
  - all of CABA: per-line perplexity, AMI, Sim -- Qwen2.5-Coder-0.5B on
    CPU via transformers. Slow-ish in wall-clock, but zero dollars.

PER QUESTION, PAID (OpenAI, GPT-4.1) -- the ONLY paid surface:
  - `ceil(total_retrieved_tokens / CAPO_CHUNK_TOKENS)` context-compression
    turns, bounded above by `MAX_CAPO_TURNS`
  - exactly ONE final answer-generation call
So: paid calls per question = n_capo_turns + 1, with
1 <= n_capo_turns <= MAX_CAPO_TURNS.

--------------------------------------------------------------------
NO GOLD LEAKAGE
--------------------------------------------------------------------
Nothing in this pipeline reads `TaskExample.reference`,
`TaskExample.metadata['checklist']`, a gold relevant-file list, or a
reference trajectory. Retrieval sees `example.question` and repository
file CONTENT only; the two prompts below interpolate `example.question`
and retrieved/compressed code only. This is the same non-negotiable rule
every other baseline in this suite follows -- see
`tests/test_repodistill.py::test_no_gold_reference_or_checklist_reaches_the_pipeline`,
the direct analogue of Dense Retrieval's own
`test_no_gold_reference_reaches_generation`.

--------------------------------------------------------------------
DEVIATIONS FROM THE PAPER (official setting -> adapted setting -> reason)
--------------------------------------------------------------------
See also `repodistill_graph.py` (5 deviations, all in GraphRAG) and
`repodistill_caba.py` (4 deviations, none of them a model substitution).
The ones owned by this file:

A. TARGET / DECISION LLM.
   Official setting: for the No-Training variant the paper uses
   Qwen3-Max (section 2.3.2), and evaluates the method across six
   backbones (Qwen3-4B/30B/Coder-30B, GPT-5.2, Claude-Sonnet-4.5,
   Gemini-3-Pro).
   Adapted setting: GPT-4.1 for both the CAPO turns and the final answer.
   Reason: explicit instruction from the governing task, and GPT-4.1 is
   this project's frozen standard baseline model -- every other baseline
   in this suite (Direct, Sparse Retrieval, Dense Retrieval, Matched
   ReAct, ChainRAG, ANT) generates with GPT-4.1. Using the paper's own
   backbone instead would make RepoDistill's numbers incomparable with
   every system it is being compared against, which is the opposite of
   what this suite is for.

B. TRAINED POLICY.
   Official setting: budget allocation performed by a Qwen3 policy
   trained with SFT + multi-turn GRPO (sections 2.3.2-2.3.3).
   Adapted setting: not implemented at all; the No-Training variant's
   direct prompting is used instead.
   Reason: explicit scope decision in the governing task, and materially
   unreproducible -- no released checkpoint, no released SFT trajectory
   data, and the SFT corpus requires two human annotators selecting among
   10 sampled trajectories per instance (Appendix A.1.1).

C. BENCHMARK.
   Official setting: SWE-QA (Peng et al. 2025, arXiv:2509.14635),
   CoderEval, LongCodeU.
   Adapted setting: RepoProbe-Python (108 questions, 8 repos) and
   SWE-QA-Pro (80 questions), this suite's own frozen manifests.
   Reason: this is a deliberate CROSS-BENCHMARK ADAPTATION, not a
   reproduction, and is expected to be reported as such. NOTE the naming
   collision: RepoDistill's own "SWE-QA" (Peng et al.) and this project's
   "SWE-QA-Pro" (TIGER-AI-Lab) are DIFFERENT, differently-authored
   benchmarks that merely share a name prefix. No dataset, metadata
   schema, or file-structure assumption transfers between them, and none
   is assumed here -- this adapter consumes only the benchmark-agnostic
   `TaskExample` contract.

D. SCORING.
   Official setting: the paper's own LLM-as-judge (Appendix A.1.4,
   DeepSeek-V3.2 + Kimi-K2, 5 runs, 5 axes).
   Adapted setting: none of that is used. Output goes through each
   benchmark's OWN existing `bench.score(...)` in this suite, unchanged
   and unspecialized.
   Reason: a baseline that is scored differently from the systems it is
   compared against is not a baseline. This adapter therefore implements
   `run()` only and touches no scoring code whatsoever.

E. CAPO CHUNK PACKING GRANULARITY.
   Official setting: "We set the chunk size of CAPO to 10,000 tokens"
   (section 3, Implementation). How units are packed into a chunk is not
   described.
   Adapted setting: greedy in-rank-order packing -- retrieved units are
   appended to the current chunk until the next one would exceed 10,000
   tokens; a single unit larger than the chunk size becomes its own
   chunk rather than being split.
   Reason: splitting one function across two turns would ask the model to
   assign that function two different retention budgets, which the output
   schema (`{"Document_i": rate}`) cannot express. Rank order is
   preserved because the paper's multi-turn premise is that the memory
   summary accumulates over chunks, and seeing the most relevant context
   first is what makes an early summary useful.

F. TOKEN COUNTING FOR CHUNK PACKING.
   Official setting: unstated; "10,000 tokens" presumably in the target
   model's own tokenizer.
   Adapted setting: Qwen2.5-Coder-0.5B's tokenizer (already loaded for
   CABA) counts tokens for chunk packing, not GPT-4.1's.
   Reason: `tiktoken` is not a dependency of this project, and adding one
   to count tokens for a bound that is itself approximate is not worth a
   new dependency. Code tokenizers agree closely enough on code text that
   the difference cannot move the chunk count by more than a fraction of
   a chunk. Disclosed rather than silently assumed equal.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.domain import Evidence
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.repo_scope import EvalRepoEnvironment
from ant.evaluation_suite.usage import UsageStats
from ant.external_wrappers.repodistill_caba import (
    MMR_LAMBDA,
    PERPLEXITY_MODEL,
    SEGMENTATION_ALPHA,
    PerplexityScorer,
    compress_unit,
    get_shared_perplexity_scorer,
)
from ant.external_wrappers.repodistill_graph import (
    ANCHOR_TOP_K,
    FINAL_CANDIDATE_LIMIT,
    RERANK_LAMBDA,
    CodeUnit,
    RepoGraph,
    build_repository_graph,
    candidates_to_units,
    retrieve_candidates,
)
from ant.retrieval.dense import DenseEmbedder, EmbeddingEntry, EmbeddingIndex, _embed_entries

PAPER_CITATION = (
    'Yin, Ding, Zhang, Wang, Wang, Ni, Cui. "RepoDistill: Distilling Repository '
    "Knowledge through Compression-Aware Budget Allocation and Policy Optimization.\" "
    "Findings of ACL 2026, pp. 4425-4443. https://aclanthology.org/2026.findings-acl.217/"
)
IMPLEMENTATION_LABEL = "paper-faithful RepoDistill (No-Training) reimplementation"
UPSTREAM_CODE_NOTE = (
    "No code was ported. The paper's cited implementation link is an Anonymous-GitHub "
    "mirror that is dead/unreachable, and no trained checkpoint is published. This "
    "implementation is derived from the published paper text (sections 2.1/2.2/2.3 and "
    "Appendix A.1.2, read directly from the ACL Anthology PDF)."
)

# ===========================================================================
# VERBATIM PROMPTS -- Appendix A.1.2 / Table 4
# ===========================================================================
# These two strings are reproduced CHARACTER-FOR-CHARACTER from Table 4 of
# the paper ("Prompt of RepoDistill for context compression (top part) and
# final answer generation (bottom part)"), retrieved from the ACL
# Anthology PDF. Line breaks inside a paragraph are the PDF's own column
# wrapping and have been rejoined into flowing lines; no word, no symbol,
# and no punctuation mark has been changed, added, or removed.
#
# The paper's own placeholders are `{problem}`, `{memory}`, `{chunk}` and
# `{repository memory}`. Two mechanical adaptations were required for
# Python string interpolation, and NOTHING else:
#   - the literal JSON braces in the OUTPUT FORMAT block are doubled
#     (`{{`/`}}`) so `str.format` emits them literally;
#   - `{repository memory}` is rendered `{repository_memory}` because a
#     `str.format` field name cannot contain a space.
# The text the model actually receives is byte-identical to the paper's.
#
# VERIFIED MECHANICALLY, not by eye: both constants were checked to be
# exact whitespace-normalized substrings of the text extracted from Table 4
# of the published PDF (undoing the two mechanical adaptations above
# first). `test_prompts_are_the_verbatim_appendix_templates` pins the
# load-bearing spans so a later edit cannot quietly reword them.
#
# DO NOT "improve", reformat, or re-word these. Their exact wording IS the
# method's decision interface -- the same reason this repository vendors
# RepoProbe's and SWE-QA-Pro's judge prompts verbatim instead of
# paraphrasing them.

CONTEXT_COMPRESSION_PROMPT = """You are provided with a problem, a chunk of code context and a previous memory for previous chunks. Your task is to analyze each document in the provided chunk and determine the appropriate compression level needed to preserve the essential information for solving the given problem, while removing redundant content. For each document_id in the chunk, assign a compression level from the following options:
- 0% : Fully filtered (empty)
- 25% : Aggressive compression (essential information only)
- 50% : Balanced compression (core content retained)
- 75% : Light compression (key context preserved)
- 100% : Original text (no compression)
<problem> {problem} </problem>
<memory> {memory} </memory>
<chunk> {chunk} </chunk>
OUTPUT FORMAT REQUIREMENT:
You must output a valid JSON object with the following structure:
{{"compressed_documents": {{"Document_0": x, "Document_1": y, "Document_2": z, ...}}, "updated_summary": "A concise summary (max 200 words) that: (1) captures salient information from the current chunk, and (2) maintains coherence with previously processed chunks."}}"""

ANSWER_GENERATION_PROMPT = """You are presented with a problem, and a repository memory. Your task is to directly answer the problem based on the provided repository memory and problem statement.
<problem> {problem} </problem>
<repository memory> {repository_memory} </repository_memory>
OUTPUT FORMAT REQUIREMENT:
Provide the final answer concisely and directly, without code snippets, extra explanations or commentary."""

# ===========================================================================
# Frozen hyperparameters
# ===========================================================================

# Paper section 3, Implementation: "We set the chunk size of CAPO to
# 10,000 tokens".
CAPO_CHUNK_TOKENS = 10_000

# The five discrete retention budgets the compression prompt itself
# enumerates (section 2.3.1: "0%/25%/50%/75%/100%, where 0% indicates
# complete filtering and 100% indicates full retention").
VALID_BUDGET_RATES = (0.0, 0.25, 0.5, 0.75, 1.0)

# Iteration cap on the multi-turn decision process. NOT a paper
# parameter -- the paper's turn count is implicitly
# ceil(context_tokens / 10k) with no stated ceiling. This bound exists so
# a single pathological question can never issue an unbounded number of
# PAID calls. With FINAL_CANDIDATE_LIMIT = 24 retrieved units, real
# chunk counts are 1-3, so this cap is slack, not binding -- it is a
# blast-radius guard, not a behavioural parameter. Any question that hits
# it is flagged in `metadata["capo_turn_cap_hit"]` so a truncated run is
# never silently invisible.
MAX_CAPO_TURNS = int(os.getenv("ANT_REPODISTILL_MAX_CAPO_TURNS", "6"))

# Output-token ceilings. The compression turn emits a small JSON object
# (a handful of numeric budgets plus a <=200-word summary); the answer
# call gets the same generous ceiling this project's own synthesis path
# uses (SYNTHESIS_MAX_OUTPUT_TOKENS = 8192) rather than a tighter one
# that could truncate an answer mid-sentence and cost it judge points.
#
# KNOWN CROSS-BENCHMARK TENSION, DISCLOSED NOT PAPERED OVER: the verbatim
# answer-generation prompt ends "Provide the final answer concisely and
# directly, without code snippets, extra explanations or commentary."
# That instruction suits the paper's own SWE-QA/CoderEval/LongCodeU
# metrics, but BOTH benchmarks this adapter targets score with an
# LLM judge that explicitly rewards completeness and reasoning quality
# (SWE-QA-Pro's 5-axis rubric includes `completeness` and `reasoning`;
# RepoProbe's rubric scores against a multi-item checklist). A faithfully
# terse RepoDistill answer may therefore score lower here than the method
# would on its own paper's benchmarks -- a genuine property of the
# cross-benchmark adaptation, not an implementation defect. The prompt is
# NOT softened to chase judge points: doing so would silently replace the
# method's own published interface with a benchmark-tuned one, which is
# exactly the kind of quiet advantage this suite forbids. Practical
# consequence for cost estimation: real OUTPUT token counts on the answer
# call should land far below this 8192 ceiling.
COMPRESSION_MAX_OUTPUT_TOKENS = 1024
ANSWER_MAX_OUTPUT_TOKENS = 8192

_GRAPH_CACHE_KEY = "repodistill_graph"
_NODE_INDEX_KEY = "repodistill_nodes"

# Per-inference-call embedding batch size, matching the large-repo bound
# `dense_retrieval_repo` arrived at the hard way (onnxruntime's peak RSS
# scales with batch size far more than the arena strategy alone controls:
# 256 texts/batch plateaued at ~7.8-9GB, 32 texts/batch under 800MB).
_EMBED_BATCH_SIZE = int(os.getenv("ANT_REPO_EMBED_BATCH_SIZE", "32"))


# ===========================================================================
# CAPO: parsing the policy's output
# ===========================================================================


def _parse_budget_value(value: object) -> float | None:
    """Map one `compressed_documents` entry to a retention RATE in [0, 1].

    The paper's own prompt states the options as percentages ("- 75% :
    Light compression") while its own example output (Appendix A.1.3 /
    Table 5) uses fractions: `{"Document_0": 0.75, "Document_1": 0.25,
    "Document_2": 0, "Document_3": 1, ...}`. Both forms are therefore
    genuinely sanctioned by the paper and both are accepted here; a bare
    `1` is read as 100% retention, per that same example, and `75` is
    read as 75%. Anything unparseable returns None so the caller can
    apply an explicit, logged default rather than silently guessing.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip().rstrip("%")
        try:
            value = float(text)
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    rate = float(value)
    if rate > 1.0:
        rate = rate / 100.0
    if not (0.0 <= rate <= 1.0):
        return None
    # Snap to the five discrete levels the prompt enumerates -- the
    # decision space is discrete by construction (section 2.3.1), so a
    # model answering 0.6 is answering off-schema and the nearest legal
    # level is the faithful reading.
    return min(VALID_BUDGET_RATES, key=lambda level: abs(level - rate))


def _extract_json_object(text: str) -> dict:
    """Tolerant JSON extraction for the compression turn's response.

    Mirrors `repoprobe.py::_extract_json`'s own strategy (fenced block
    first, then a brace-span fallback) rather than inventing a third
    JSON-repair convention in this repository.
    """
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def pack_units_into_chunks(
    units: list[CodeUnit], scorer: PerplexityScorer, chunk_tokens: int | None = None
) -> list[list[CodeUnit]]:
    """Greedy in-rank-order packing into <= `chunk_tokens` chunks
    (deviation E). A single unit exceeding the budget becomes its own
    chunk rather than being split across turns.

    `chunk_tokens` defaults to None and resolves to the module-level
    `CAPO_CHUNK_TOKENS` at CALL time, not at definition time -- a default
    argument bound at import would silently ignore any later override of
    the module constant (which is exactly how the chunking tests, and any
    future sensitivity sweep over chunk size, need to drive it).
    """
    chunk_tokens = CAPO_CHUNK_TOKENS if chunk_tokens is None else chunk_tokens
    chunks: list[list[CodeUnit]] = []
    current: list[CodeUnit] = []
    current_tokens = 0
    for unit in units:
        n_tokens = scorer.count_tokens(unit.snippet)
        if current and current_tokens + n_tokens > chunk_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(unit)
        current_tokens += n_tokens
    if current:
        chunks.append(current)
    return chunks


def render_chunk(units: list[CodeUnit]) -> str:
    """The `{chunk}` the compression prompt interpolates.

    Document ids are `Document_0`, `Document_1`, ... exactly as the
    prompt's own output schema names them, numbered WITHIN the turn (each
    turn's chunk restarts at Document_0), because the prompt says "For
    each document_id in the chunk". File path and line range are included
    as document metadata -- section 2.1.1 lists them as recorded node
    metadata, and a budget decision about a code unit that does not say
    where the unit lives would be strictly less informed.
    """
    parts: list[str] = []
    for index, unit in enumerate(units):
        parts.append(
            f"Document_{index} ({unit.path}:{unit.line_start}-{unit.line_end}, "
            f"{unit.qualname}):\n{unit.snippet}"
        )
    return "\n\n".join(parts)


# ===========================================================================
# One-time per-repo preprocessing (local, free, disk-cached)
# ===========================================================================


def ensure_repo_graph_and_index(
    environment_root: Path,
    relative_paths: list[str],
    embedder: DenseEmbedder,
    index_dir: Path,
) -> tuple[RepoGraph, list[str], np.ndarray]:
    """Build (or load) this checkout's dependency graph and node-embedding
    index. ONE-TIME PER REPO, reused by every question against it.

    The cache is keyed by the repo checkout directory, and is only reused
    when the cached node-embedding index covers exactly the graph's own
    current node set -- never merely "a cache file exists here". That is
    the same stale-index guard `dense_retrieval_repo` and
    `AntDocumentAgent._ensure_indexed` both already apply, and for the
    same reason: a partially-built or out-of-date index silently degrades
    retrieval instead of failing loudly.
    """
    graph = RepoGraph.load(index_dir, _GRAPH_CACHE_KEY)
    if graph is None:
        graph = build_repository_graph(environment_root, relative_paths)
        graph.save(index_dir, _GRAPH_CACHE_KEY)

    node_ids = sorted(graph.units)
    if not node_ids:
        return graph, [], np.zeros((0, 0), dtype=np.float32)

    cached = EmbeddingIndex.load(index_dir, _NODE_INDEX_KEY)
    if cached is not None and [entry.path for entry in cached.entries] == node_ids:
        return graph, node_ids, cached.vectors

    entries = [
        EmbeddingEntry(
            path=node_id,
            line_start=graph.units[node_id].line_start,
            line_end=graph.units[node_id].line_end,
            quote=graph.units[node_id].snippet[:2400],
        )
        for node_id in node_ids
    ]
    texts = [graph.units[node_id].embedding_text() for node_id in node_ids]
    index = _embed_entries(entries, texts, embedder, verbose=True, batch_size=_EMBED_BATCH_SIZE)
    index.save(index_dir, _NODE_INDEX_KEY)
    return graph, node_ids, index.vectors


# ===========================================================================
# The adapter
# ===========================================================================


class RepoDistillAdapter:
    """RepoDistill (No Training) as a single `AgentAdapter`.

    Pipeline, once per question:
      1. GraphRAG retrieval over the (cached) repo dependency graph.
      2. CAPO turns: for each ~10k-token chunk of retrieved units, ONE
         GPT-4.1 call assigning a discrete retention budget per document
         and updating the running memory summary.
      3. CABA: compress each unit locally to its assigned budget using
         Qwen2.5-Coder-0.5B perplexity + MMR.
      4. ONE GPT-4.1 answer-generation call over the compressed
         repository memory.

    Steps 1 and 3 are entirely local and free; only steps 2 and 4 are
    paid.
    """

    name = "repodistill"

    def __init__(
        self,
        model: str = "gpt-4.1",
        top_k: int = ANCHOR_TOP_K,
        candidate_limit: int = FINAL_CANDIDATE_LIMIT,
        index_root: Path | None = None,
        perplexity_model: str = PERPLEXITY_MODEL,
    ) -> None:
        self.model = model
        self.top_k = top_k
        self.candidate_limit = candidate_limit
        self.index_root = index_root or Path(".ant/eval-suite-repodistill")
        self.perplexity_model = perplexity_model

    def _index_dir_for(self, example: TaskExample, environment_root: Path) -> Path:
        return self.index_root / example.benchmark / environment_root.name

    def _make_provider(self) -> CountingOpenAIProvider:
        return CountingOpenAIProvider(model=self.model)

    def _make_scorer(self) -> PerplexityScorer:
        return get_shared_perplexity_scorer(self.perplexity_model)

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        started = time.time()
        provider = self._make_provider()
        scorer = self._make_scorer()

        # --- file universe: the same content-based scope every other
        # baseline in this suite uses; GraphRAG then narrows to .py within
        # it (deviation 2 in repodistill_graph.py). ---
        environment = EvalRepoEnvironment(environment_root)
        relative_paths = [
            str(path.relative_to(environment.root)) for path in environment.iter_files()
        ]

        embedder = DenseEmbedder()  # local, free
        index_dir = self._index_dir_for(example, environment_root)
        graph, node_ids, node_vectors = ensure_repo_graph_and_index(
            environment_root, relative_paths, embedder, index_dir
        )

        # --- 1. GraphRAG (local, free). `example.question` is the ONLY
        # task-derived input that reaches this. ---
        if node_ids:
            [query_vector_list] = embedder.embed([example.question])
            query_vector = np.asarray(query_vector_list, dtype=np.float32)
            candidates, anchors = retrieve_candidates(
                graph,
                example.question,
                node_ids,
                node_vectors,
                query_vector,
                top_k=self.top_k,
                limit=self.candidate_limit,
            )
        else:
            candidates, anchors = [], []
        units = candidates_to_units(graph, candidates)

        # --- 2. CAPO turns (PAID) ---
        chunks = pack_units_into_chunks(units, scorer)
        turn_cap_hit = len(chunks) > MAX_CAPO_TURNS
        chunks = chunks[:MAX_CAPO_TURNS]
        # Units in the chunks the cap cut off were never shown to the
        # policy, so they are DROPPED, not carried forward. Keeping them
        # would mean falling back to 100% retention for context CAPO never
        # actually evaluated -- which would both defeat the cap (the
        # answer call's input would still grow without bound) and attribute
        # a retention decision to the model that it never made. Chunks are
        # packed in re-ranked relevance order, so what is dropped is always
        # the lowest-ranked tail.
        units = [unit for chunk in chunks for unit in chunk]

        memory = ""
        budgets: dict[str, float] = {}
        trajectory: list[dict] = []
        for turn_index, chunk_units in enumerate(chunks):
            prompt = CONTEXT_COMPRESSION_PROMPT.format(
                problem=example.question,
                memory=memory or "(no previous memory)",
                chunk=render_chunk(chunk_units),
            )
            response = provider.responses_text(
                prompt, max_output_tokens=COMPRESSION_MAX_OUTPUT_TOKENS
            )
            payload = _extract_json_object(response.text)
            raw_budgets = payload.get("compressed_documents")
            raw_budgets = raw_budgets if isinstance(raw_budgets, dict) else {}

            turn_budgets: dict[str, float] = {}
            for document_index, unit in enumerate(chunk_units):
                parsed = _parse_budget_value(raw_budgets.get(f"Document_{document_index}"))
                # An unparseable/missing budget falls back to 100%
                # retention -- i.e. the compression step declines to act
                # rather than silently DELETING a retrieved unit on the
                # strength of a malformed response. Failing open toward
                # more context is the conservative direction for an
                # answer-quality benchmark; failing closed would let one
                # bad JSON response quietly empty the evidence.
                rate = 1.0 if parsed is None else parsed
                budgets[unit.node_id] = rate
                turn_budgets[unit.node_id] = rate

            summary = payload.get("updated_summary")
            if isinstance(summary, str) and summary.strip():
                memory = summary.strip()

            trajectory.append(
                {
                    "capo_turn": turn_index,
                    "n_documents": len(chunk_units),
                    "document_ids": [unit.node_id for unit in chunk_units],
                    "assigned_budgets": turn_budgets,
                    "budget_parse_failures": sum(
                        1
                        for i in range(len(chunk_units))
                        if _parse_budget_value(raw_budgets.get(f"Document_{i}")) is None
                    ),
                    "memory_after_turn": memory,
                }
            )

        # --- 3. CABA compression (local, free) ---
        compressed: list[tuple[CodeUnit, float, str]] = []
        for unit in units:
            rate = budgets.get(unit.node_id, 1.0)
            text = compress_unit(unit.snippet, example.question, rate, scorer)
            compressed.append((unit, rate, text))

        repository_memory = self._render_repository_memory(memory, compressed)

        # --- 4. Answer generation (PAID, exactly one call) ---
        answer_prompt = ANSWER_GENERATION_PROMPT.format(
            problem=example.question, repository_memory=repository_memory
        )
        answer = provider.responses_text(
            answer_prompt, max_output_tokens=ANSWER_MAX_OUTPUT_TOKENS
        ).text.strip()

        llm_calls = provider.drain_call_count()
        token_usage = provider.drain_usage()
        elapsed = time.time() - started

        evidence = [
            Evidence(
                path=unit.path,
                line_start=unit.line_start,
                line_end=unit.line_end,
                quote=(text or unit.snippet)[:2400],
                reason=(
                    f"RepoDistill GraphRAG candidate ({unit.qualname}), "
                    f"CAPO retention budget {int(rate * 100)}%"
                ),
                symbols=[unit.qualname],
            )
            for unit, rate, text in compressed
            if rate > 0.0
        ]

        original_tokens = sum(scorer.count_tokens(unit.snippet) for unit in units)
        compressed_tokens = sum(scorer.count_tokens(text) for _, _, text in compressed)

        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=answer,
            trajectory=trajectory,
            evidence=[item.model_dump() for item in evidence],
            usage=UsageStats(
                llm_calls=llm_calls,
                # Local retrieval/compression work, counted as tool calls
                # so the fairness report can see the non-LLM effort: the
                # graph+embedding lookup, plus one CABA compression per
                # retrieved unit.
                tool_calls=1 + len(units),
                input_tokens=token_usage.input_tokens,
                output_tokens=token_usage.output_tokens,
                total_tokens=token_usage.total_tokens,
                estimated_cost_usd=token_usage.estimated_cost_usd,
                wall_clock_seconds=elapsed,
                unique_files_inspected=len({unit.path for unit in units}),
                unique_symbols_inspected=len({unit.qualname for unit in units}),
            ),
            termination_reason=(
                "capo_turn_cap_reached" if turn_cap_hit else "all_chunks_processed"
            ),
            metadata={
                "generation_model": self.model,
                "implementation": IMPLEMENTATION_LABEL,
                "paper_citation": PAPER_CITATION,
                "upstream_code_note": UPSTREAM_CODE_NOTE,
                "variant": "no_training",
                "embedding_model": embedder.model_name,
                "perplexity_model": self.perplexity_model,
                "graph_nodes": graph.n_nodes,
                "graph_edges": graph.n_edges,
                "anchor_node_ids": anchors,
                "top_k": self.top_k,
                "rerank_lambda": RERANK_LAMBDA,
                "mmr_lambda": MMR_LAMBDA,
                "segmentation_alpha": SEGMENTATION_ALPHA,
                "capo_chunk_tokens": CAPO_CHUNK_TOKENS,
                "n_candidates": len(candidates),
                "n_retrieved_units": len(units),
                "n_capo_turns": len(chunks),
                "capo_turn_cap_hit": turn_cap_hit,
                "retrieved_chunk_ids": [
                    f"{unit.path}:{unit.line_start}-{unit.line_end}" for unit in units
                ],
                "retention_budgets": {unit.node_id: rate for unit, rate, _ in compressed},
                "original_context_tokens": original_tokens,
                "compressed_context_tokens": compressed_tokens,
                "compression_ratio": (
                    compressed_tokens / original_tokens if original_tokens else 0.0
                ),
            },
        )

    @staticmethod
    def _render_repository_memory(
        memory: str, compressed: list[tuple[CodeUnit, float, str]]
    ) -> str:
        """The `{repository memory}` the answer-generation prompt consumes.

        Section 2.3.1: "After all chunks have been processed, RepoDistill
        applies the CABA to compress contexts according to its assigned
        budget. Finally, the LLM synthesizes the final answer based on the
        task query and the compressed contexts." The running memory
        summary is included ahead of the compressed contexts because it is
        the other half of what the multi-turn process produced -- dropping
        it would discard every cross-chunk observation the CAPO turns were
        spent generating.

        Units whose assigned budget was 0% are omitted entirely: the
        prompt defines 0% as "Fully filtered (empty)", so emitting an
        empty stub for them would contradict the model's own decision.
        """
        parts: list[str] = []
        if memory:
            parts.append(f"[memory summary]\n{memory}")
        for unit, rate, text in compressed:
            if rate <= 0.0 or not text.strip():
                continue
            parts.append(
                f"[{unit.path}:{unit.line_start}-{unit.line_end} {unit.qualname} "
                f"| retained {int(rate * 100)}%]\n{text}"
            )
        return "\n\n".join(parts) if parts else "(no context retained)"


def expected_paid_calls(n_retrieved_tokens: int) -> tuple[int, int]:
    """(capo_turns, total_paid_calls) for a question whose retrieved
    context is `n_retrieved_tokens` tokens.

    Exposed as a real function rather than left as prose so the cost model
    in this module's docstring is executable and testable, not an
    assertion nobody can check.
    """
    turns = max(1, math.ceil(n_retrieved_tokens / CAPO_CHUNK_TOKENS))
    turns = min(turns, MAX_CAPO_TURNS)
    return turns, turns + 1


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(RepoDistillAdapter())
