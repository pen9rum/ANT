"""Chain-of-Agents (CoA): paper-faithful adaptation. See
docs/chain_of_agents_fidelity_audit.md for the full audit (no official or
author-linked runnable implementation exists; reconstructed directly from
the paper's own Section 3).

PAPER_CITATION below documents the reference. This module implements ONLY
the paper's query-based (QA) worker/manager protocol -- sequential
workers, one per chunk, in original source order; each worker sees only
its own chunk, the previous worker's communication unit, and the
question; the manager sees only the LAST communication unit and the
question; every chunk is always processed (no retrieval, no relevance
skipping, no early stopping, no question-aware pruning).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.answer_contract import (
    apply_context_authoritative_regrounding,
    condense_to_answer_span,
)
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.usage import UsageStats
from ant.external_wrappers.longagent import serialize_repository

PAPER_CITATION = (
    'Zhang, Sun, Chen, Pfister, Zhang, Arık. "Chain of Agents: Large Language '
    'Models Collaborating on Long-Context Tasks." NeurIPS 2024. arXiv:2406.02818.'
)
IMPLEMENTATION_LABEL = "paper-faithful Chain-of-Agents adaptation"
UPSTREAM_COMMIT = None
UPSTREAM_REPO_NOTE = (
    "No official or author-linked runnable implementation exists. The paper's own "
    "project page (yszh8.github.io/chain-of-agents, first author's site) links only "
    "to the Penn State NLP Lab's general GitHub org (github.com/psunlpgroup), which "
    "has zero repositories matching 'chain' as of this audit. Three third-party, "
    "non-author-linked community repos exist but were not used as a reference. See "
    "docs/chain_of_agents_fidelity_audit.md."
)

_ENCODING_NAME = "cl100k_base"

# The paper's own reported 8K-agent-window setting (Section 4) -- used
# unchanged as the initial/smoke-test configuration, never tuned from
# results (see fidelity audit section 4).
DEFAULT_AGENT_WINDOW_TOKENS = 8192

# Disclosed engineering choices, NOT from the paper (which does not state
# an exact worker-output cap) -- see fidelity audit section 4. Fixed
# before any generation, never tuned per example/result.
MAX_CU_OUTPUT_TOKENS = 512
SAFETY_MARGIN_TOKENS = 200
MANAGER_MAX_OUTPUT_TOKENS = 1024

_WORKER_PROMPT = """{chunk_text}

Here is the summary of the previous source text:
{previous_cu}

Question:
{question}

You need to read the current source text and the summary of the previous source text, if \
any, and generate a summary that includes both. This summary will be used by later agents to \
answer the question. Preserve evidence useful for answering the question."""

_NO_PREVIOUS_CU = "(none -- this is the first chunk)"

_MANAGER_PROMPT = """Answer the question concisely and precisely, appropriate for a \
short-answer QA benchmark.

The source text was too long and has been summarized.
Answer based on the following summary:

{cu_l}

Question:
{question}

Answer:"""

# Sentence-aware chunking is a disclosed HEURISTIC (regex-based, not true
# NLP -- see fidelity audit section 4): split on sentence-ending
# punctuation followed by whitespace, or on any run of newlines
# (paragraph/file-marker boundaries in the serialized source document).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_into_sentences(text: str) -> list[str]:
    if not text.strip():
        return []
    return [s for s in (p.strip() for p in _SENTENCE_SPLIT_RE.split(text)) if s]


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _strip_all_whitespace(text: str) -> str:
    return "".join(text.split())


def verify_full_coverage(original_text: str, chunks: list[str]) -> bool:
    """True iff every non-whitespace character of `original_text` appears,
    in order, in the concatenation of `chunks` -- i.e. no content was
    dropped or duplicated beyond whitespace effects (exact whitespace is
    NOT preserved by design: sentence joins always insert a single space,
    and the oversized-sentence hard-split fallback can decode a token
    boundary that had no natural space in the original -- e.g. "juno."
    hard-split becomes "juno" + "."; joining chunks with a space then
    reads as "juno ." Comparing with ALL whitespace stripped, rather than
    merely collapsed, treats exactly this class of split-boundary spacing
    as the allowed normalization effect while still catching a genuinely
    dropped or duplicated WORD.
    """
    return _strip_all_whitespace(original_text) == _strip_all_whitespace(" ".join(chunks))


def _worker_instruction_overhead_tokens(encoding: tiktoken.Encoding) -> int:
    """Token count of the worker prompt template's own literal scaffolding
    text, with chunk/previous-CU/question all empty -- used to compute the
    EFFECTIVE per-chunk token budget below, not a hardcoded guess."""
    rendered = _WORKER_PROMPT.format(chunk_text="", previous_cu="", question="")
    return len(encoding.encode(rendered, disallowed_special=()))


def compute_chunk_token_budget(
    *, agent_window_tokens: int, question: str, encoding: tiktoken.Encoding | None = None
) -> int:
    """budget = k - tokens(question) - tokens(worker_instruction) -
    MAX_CU_OUTPUT_TOKENS (reserved worst-case space for CU_{i-1}) -
    SAFETY_MARGIN_TOKENS (prompt-formatting overhead). This is the max
    token count of the SOURCE CHUNK TEXT ITSELF (ci) that a worker call
    for this question can safely receive under `agent_window_tokens`, per
    the paper's own budget formula (Section 3.1). Never tuned from
    results -- a pure function of (agent_window_tokens, question, the
    frozen prompt template).
    """
    encoding = encoding or tiktoken.get_encoding(_ENCODING_NAME)
    question_tokens = len(encoding.encode(question, disallowed_special=()))
    instruction_tokens = _worker_instruction_overhead_tokens(encoding)
    budget = (
        agent_window_tokens
        - question_tokens
        - instruction_tokens
        - MAX_CU_OUTPUT_TOKENS
        - SAFETY_MARGIN_TOKENS
    )
    if budget <= 0:
        msg = (
            f"compute_chunk_token_budget produced a non-positive budget ({budget}) for "
            f"agent_window_tokens={agent_window_tokens} -- the question/instruction/CU "
            "overhead alone exceed the configured agent window. This is a configuration "
            "problem (increase agent_window_tokens), not something to silently clamp."
        )
        raise ValueError(msg)
    return budget


@dataclass
class ChunkInfo:
    text: str
    token_count: int


def build_chunks(
    source_text: str, budget_tokens: int, encoding: tiktoken.Encoding | None = None
) -> list[ChunkInfo]:
    """Greedily packs sentences (in ORIGINAL source order, no overlap) into
    chunks under `budget_tokens` each -- the paper's own Section 3.1
    chunking policy. No semantic retrieval, no gold information, no
    question-aware pruning of which sentences to include -- every sentence
    of `source_text` ends up in exactly one chunk. A single sentence
    exceeding `budget_tokens` on its own (rare) is hard-split at the token
    level as a disclosed fallback (see fidelity audit section 4) rather
    than silently dropped or truncated.
    """
    encoding = encoding or tiktoken.get_encoding(_ENCODING_NAME)
    sentences = split_into_sentences(source_text)
    chunks: list[ChunkInfo] = []
    current: list[str] = []
    current_tokens = 0

    def _flush() -> None:
        nonlocal current, current_tokens
        if current:
            text = " ".join(current)
            chunks.append(
                ChunkInfo(text=text, token_count=len(encoding.encode(text, disallowed_special=())))
            )
            current = []
            current_tokens = 0

    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence, disallowed_special=()))
        if sentence_tokens > budget_tokens:
            _flush()
            token_ids = encoding.encode(sentence, disallowed_special=())
            for start in range(0, len(token_ids), budget_tokens):
                piece_ids = token_ids[start : start + budget_tokens]
                chunks.append(
                    ChunkInfo(text=encoding.decode(piece_ids), token_count=len(piece_ids))
                )
            continue
        if current_tokens + sentence_tokens > budget_tokens and current:
            _flush()
        current.append(sentence)
        current_tokens += sentence_tokens
    _flush()
    return chunks


class ChainOfAgentsAdapter:
    """CoA baseline: sequential workers (one per chunk, original source
    order) each producing a communication unit from (own chunk, previous
    CU, question); a separate manager answers from (CU_l, question) only.
    See module docstring / fidelity audit for exactly what this does and
    does not implement.
    """

    name = "chain_of_agents"

    def __init__(
        self,
        model: str = "gpt-4.1",
        agent_window_tokens: int = DEFAULT_AGENT_WINDOW_TOKENS,
        apply_concise_answer_contract: bool = True,
    ) -> None:
        self.model = model
        self.agent_window_tokens = agent_window_tokens
        # Default True (unlike LongAgent's default False): this adapter is
        # being added ONLY for the document/long-context track (Part C of
        # its own governing spec), where every other method already
        # applies the shared short-answer contract as its last step -- see
        # ant.evaluation_suite.answer_contract's own module docstring.
        self.apply_concise_answer_contract = apply_concise_answer_contract

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        provider = CountingOpenAIProvider(model=self.model)
        started = time.time()
        encoding = tiktoken.get_encoding(_ENCODING_NAME)

        source_text, eligible_file_count = serialize_repository(environment_root)
        source_token_count = len(encoding.encode(source_text, disallowed_special=()))
        budget = compute_chunk_token_budget(
            agent_window_tokens=self.agent_window_tokens,
            question=example.question,
            encoding=encoding,
        )
        chunks = build_chunks(source_text, budget, encoding=encoding)
        assert verify_full_coverage(source_text, [c.text for c in chunks])
        n_chunks = len(chunks)

        trajectory: list[dict] = []
        cu = ""
        cu_token_counts: list[int] = []
        worker_calls = 0
        for index, chunk in enumerate(chunks):
            prompt = _WORKER_PROMPT.format(
                chunk_text=chunk.text,
                previous_cu=cu if cu else _NO_PREVIOUS_CU,
                question=example.question,
            )
            result = provider.responses_text(prompt, max_output_tokens=MAX_CU_OUTPUT_TOKENS)
            cu = result.text.strip()
            worker_calls += 1
            cu_tokens = len(encoding.encode(cu, disallowed_special=()))
            cu_token_counts.append(cu_tokens)
            trajectory.append(
                {
                    "step": index,
                    "role": "worker",
                    "chunk_index": index,
                    "chunk_token_count": chunk.token_count,
                    "cu_token_count": cu_tokens,
                    "communication_unit": cu,
                }
            )

        manager_prompt = _MANAGER_PROMPT.format(cu_l=cu, question=example.question)
        manager_result = provider.responses_text(
            manager_prompt, max_output_tokens=MANAGER_MAX_OUTPUT_TOKENS
        )
        manager_calls = 1
        raw_answer = manager_result.text.strip()
        trajectory.append({"step": n_chunks, "role": "manager", "raw_answer": raw_answer})

        grounded_answer = raw_answer
        if example.metadata.get("answer_contract_condition") == "B":
            grounded_answer = apply_context_authoritative_regrounding(
                provider, example.question, raw_answer, cu
            )
        if self.apply_concise_answer_contract:
            final_answer = condense_to_answer_span(provider, example.question, grounded_answer)
        else:
            final_answer = grounded_answer

        physical_llm_calls = provider.drain_call_count()
        token_usage = provider.drain_usage()
        retry_log = provider.drain_retry_log()
        elapsed = time.time() - started

        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=final_answer,
            trajectory=trajectory,
            usage=UsageStats(
                llm_calls=physical_llm_calls,
                input_tokens=token_usage.input_tokens,
                output_tokens=token_usage.output_tokens,
                total_tokens=token_usage.total_tokens,
                estimated_cost_usd=token_usage.estimated_cost_usd,
                wall_clock_seconds=elapsed,
                unique_files_inspected=eligible_file_count,
            ),
            termination_reason="manager_answered",
            metadata={
                "generation_model": self.model,
                "implementation_label": IMPLEMENTATION_LABEL,
                "upstream_commit": UPSTREAM_COMMIT,
                "upstream_repo_note": UPSTREAM_REPO_NOTE,
                "source_token_count": source_token_count,
                "agent_window_tokens": self.agent_window_tokens,
                "effective_chunk_source_token_budget": budget,
                "n_chunks": n_chunks,
                "n_workers": n_chunks,
                "worker_calls": worker_calls,
                "manager_calls": manager_calls,
                "total_physical_llm_calls": physical_llm_calls,
                "chunk_token_counts": [c.token_count for c in chunks],
                "cu_token_counts": cu_token_counts,
                "raw_answer_before_condensation": raw_answer,
                "answer_contract_condition": example.metadata.get("answer_contract_condition"),
                "grounded_answer_after_regrounding": grounded_answer
                if grounded_answer != raw_answer
                else None,
                "retry_total_physical_attempts": sum(e["attempt_count"] for e in retry_log),
                "retry_total_retries": sum(e["retry_count"] for e in retry_log),
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(ChainOfAgentsAdapter())
