"""Benchmark-scoped final-answer synthesis policy for ANTMAN's document
substrate, fixing a benchmark-policy leakage.

`ant.coordinator.local.LocalCoordinator.ask()` (frozen core, shared with
`ant.agents.ant_adapter.AntAgent`, ANTMAN's SWE-QA-Pro adapter) was
originally developed/tuned against SWE-QA-Pro, where an answer may
genuinely be unsupported by the repository and abstaining is a correct,
desired outcome. `ant.agents.ant_document_adapter.AntDocumentAgent`
previously used that same abstention-prone `state.answer` as the raw basis
for EVERY document benchmark it runs, including HotpotQA/2WikiMultihopQA/
MuSiQue -- standard multi-hop QA benchmarks where the expected answer is
routinely a compositional inference over several evidence sentences, not a
single literally-stated span. That produced spurious "not stated in the
provided documents"-style refusals on questions the retrieved evidence
could actually answer (see docs/antman_natural_qa_synthesis_fix.md for the
concrete traced cases this was diagnosed from).

This module is a SEPARATE, evidence-driven final-answer synthesis step,
used INSTEAD of (not chained after) `state.answer` +
`ant.evaluation_suite.answer_contract.condense_to_answer_span`, for
exactly the three benchmarks in `NATURAL_MULTIHOP_QA_BENCHMARKS`. It does
not read `LocalCoordinator.ask()`, does not touch worker routing, the Need
Graph, evidence selection/facet rescue, or retrieval in any way -- it only
consumes whatever evidence an (unmodified) coordination run already
produced and performs the final natural-language synthesis over it.

`ant.agents.ant_adapter.AntAgent` (SWE-QA-Pro) does not import this module
and was not modified by this change -- it continues to use `state.answer`
directly, completely unaffected.
"""
from __future__ import annotations

from typing import Any, Protocol

from ant.providers.openai_provider import ResponseResult


class _TextResponder(Protocol):
    """Same structural minimum as `ant.evaluation_suite.answer_contract`'s
    own `_TextResponder` -- duplicated rather than imported so this module
    has no dependency on `answer_contract.py` at all (the two final-answer
    policies are deliberately independent, not layered)."""

    def responses_text(self, prompt: str, max_output_tokens: int = 512) -> ResponseResult: ...


# The ONLY three benchmarks this policy applies to. Deliberately a
# hard-coded, benchmark-*identity* set (not a heuristic over question
# content, dataset size, or anything content-derived) -- the distinction
# this module exists to draw is "which benchmark is this", never anything
# about the specific question or its gold answer.
NATURAL_MULTIHOP_QA_BENCHMARKS = frozenset({"hotpotqa", "2wikimultihopqa", "musique"})

ABSTENTION_TEXT = "not stated in the provided evidence"

# Generic to "standard answerable multi-hop QA" as a task family -- no
# benchmark name, dataset-specific heuristic, entity name, or gold-derived
# rule appears anywhere in this prompt. It must not be tuned against any
# specific example's observed failure.
_NATURAL_QA_SYNTHESIS_PROMPT = """You are answering a standard multi-hop question-answering \
benchmark question using only the evidence supplied below.

Question: {question}

Evidence gathered for this question:
{evidence_block}

Instructions:
- Answer using only the evidence above; do not rely on outside/memorized knowledge that \
conflicts with or goes beyond it.
- The answer does NOT need to appear verbatim in a single evidence sentence. Combine evidence \
across multiple entries when necessary.
- Perform ordinary compositional inference when it is directly supported by the evidence: \
entity chaining, relation composition, comparisons, counting, simple arithmetic, resolving an \
entity across multiple evidence pieces, and deriving the requested relation from multiple \
stated facts are all expected and encouraged.
- Do not refuse to answer merely because the final relation or comparison is not explicitly \
written out as its own sentence -- derive it from the stated facts.
- However, if the evidence above genuinely does not contain enough information to derive an \
answer -- for example because a needed fact was never retrieved -- say so briefly (e.g. \
"{abstention_text}"). Do not invent or guess a fact the evidence does not support.
- Prefer the shortest direct answer span that answers the question. Do not add explanation or \
justification unless it is necessary to disambiguate between multiple plausible answers.

Answer:"""

# Larger than answer_contract.py's CONDENSE_MAX_OUTPUT_TOKENS (64) because
# this call does real synthesis/composition over evidence, not just
# shortening an already-produced answer -- but still small and fixed (not
# tuned per benchmark/example), matching that module's own philosophy.
NATURAL_QA_SYNTHESIS_MAX_OUTPUT_TOKENS = 150


def is_natural_multihop_qa_benchmark(benchmark: str) -> bool:
    return benchmark in NATURAL_MULTIHOP_QA_BENCHMARKS


def format_evidence_block(evidence: list[Any]) -> str:
    """Same `[path:start-end] quote` serialization already used at
    `ant_document_adapter.py`'s own `apply_context_authoritative_regrounding`
    call site -- reused here for consistency, not a new evidence format.
    Accepts either `Evidence` model instances (the live call site, from
    `state.evidence`) or plain dicts with the same keys (the offline
    re-answer path, reading an already-serialized trajectory JSON file) --
    both are the same information, just at different points in the
    load/dump cycle, so one function covers both without duplicating the
    formatting logic.
    """
    lines = []
    for item in evidence:
        if isinstance(item, dict):
            path, line_start, line_end, quote = (
                item["path"],
                item["line_start"],
                item["line_end"],
                item["quote"],
            )
        else:
            path, line_start, line_end, quote = (
                item.path,
                item.line_start,
                item.line_end,
                item.quote,
            )
        lines.append(f"[{path}:{line_start}-{line_end}] {quote}")
    return "\n".join(lines)


def synthesize_natural_multihop_answer(
    provider: _TextResponder, question: str, evidence_block: str
) -> str:
    """The complete final-answer synthesis step for HotpotQA/2WikiMultihopQA/
    MuSiQue in `AntDocumentAgent.run()` -- used INSTEAD of `state.answer` +
    `condense_to_answer_span` for exactly these three benchmarks. Receives
    only `question` and already-gathered `evidence_block` text: never a
    gold answer, supporting-fact annotation, or any other dataset-specific
    signal.
    """
    if not evidence_block.strip():
        return ABSTENTION_TEXT
    prompt = _NATURAL_QA_SYNTHESIS_PROMPT.format(
        question=question, evidence_block=evidence_block, abstention_text=ABSTENTION_TEXT
    )
    result = provider.responses_text(
        prompt, max_output_tokens=NATURAL_QA_SYNTHESIS_MAX_OUTPUT_TOKENS
    )
    answer = result.text.strip()
    return answer or ABSTENTION_TEXT
