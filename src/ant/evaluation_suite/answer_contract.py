"""Shared short-answer output contract for the long-context/multi-document
evaluation track (Part A of the follow-up spec). Fixes a real evaluation-
INTERFACE defect surfaced by the first 6-task smoke
(docs/long_context_smoke_report.md Section 5): Retrieval/ANT's shared,
repository-QA-designed `synthesize()` prompt produces long, discursive,
multi-paragraph answers, which HotpotQA/2Wiki/MuSiQue's word-overlap F1
scores near zero against official gold answers that are short phrases --
even when the underlying answer is substantively correct. This is a
metric/prompt mismatch, not a coordination-quality problem, and the fix
must not touch any method's actual reasoning/search/coordination behavior.

Design: ONE shared, purely POST-HOC condensation step
(`condense_to_answer_span`), applied identically as the LAST thing every
one of the five methods does to its own already-fully-computed raw answer,
using the SAME provider instance (and therefore the SAME accounting) each
method already has in scope -- never inserted earlier in any method's own
prompt chain. This is deliberately NOT implemented by appending the
instruction into a shared `question` string, because for ANT specifically
that string (`LocalCoordinator.ask()`'s own `question` parameter) is
reused throughout frozen `local.py` for root-Need text, worker
instructions, coverage-need normalization, and lexical term extraction
(`TOKEN_RE.findall(question)`, `_relevant_symbols(question)`) -- polluting
it with extra instruction text would risk measurably changing search-term
extraction and worker candidate ranking, i.e. a real (if inadvertent)
algorithmic behavior change disguised as "just data". Running condensation
strictly AFTER `coordinator.ask()` returns means ANT's routing, Need
Graph, and recovery logic see the EXACT same `question` string as before
this change, with zero exceptions -- verifiable directly from this
module's own call site in `ant_document_adapter.py` (one line, after
`state = coordinator.ask(...)`, before building `AgentResult`).

The condensation prompt is strictly EXTRACTIVE, not a second independent
answer attempt: it is explicitly instructed to use only the already-
produced answer's own content, never outside knowledge, so an already-
correct (or already-wrong) substantive answer is shortened, not
re-derived or "corrected". Same contract, same prompt, same model
call site for all five methods -- exactly the "no selective advantage"
requirement.
"""
from __future__ import annotations

from typing import Protocol

from ant.providers.openai_provider import ResponseResult


class _TextResponder(Protocol):
    """Structural minimum `condense_to_answer_span` needs -- deliberately
    NOT the concrete `CountingOpenAIProvider` class, so a test double only
    needs to implement this one method, not the whole provider surface."""

    def responses_text(self, prompt: str, max_output_tokens: int = 512) -> ResponseResult: ...

CONCISE_ANSWER_INSTRUCTION = (
    "Return only the minimal answer span required by the question. Do not provide "
    "explanation, justification, citations, or reasoning unless the question explicitly "
    "requires them."
)

_CONDENSE_PROMPT = """You are converting an already-produced answer into the short final-answer \
format a question-answering benchmark expects. Do not re-derive the answer from outside \
knowledge and do not change its substance -- extract or restate ONLY the minimal answer span \
already present in the ANSWER below.

Question: {question}

Answer (to be condensed, not replaced):
{raw_answer}

{instruction}
If the answer above does not state a clear answer, output exactly what it states is missing, \
as briefly as possible (e.g. "not stated in the provided documents")."""

# Short final answers do not need a large output budget; kept small and
# fixed (not tuned per method/benchmark) so every method's condensation
# call is directly comparable.
CONDENSE_MAX_OUTPUT_TOKENS = 64


CONTEXT_AUTHORITATIVE_INSTRUCTION = (
    "Answer strictly according to the provided context. Treat the provided context as "
    "authoritative even if it conflicts with your prior knowledge. Do not rely on memorized "
    "world knowledge when it conflicts with the supplied context."
)

_REGROUND_PROMPT = """Question: {question}

Context/evidence already gathered for this question:
{context_text}

An answer was already produced for this question: {raw_answer}

{instruction}

Re-state the answer, strictly grounded in the context/evidence above. If the context/evidence \
above directly states an answer that differs from what was previously produced, use the answer \
the context/evidence actually states instead."""

# Same fixed-budget philosophy as CONDENSE_MAX_OUTPUT_TOKENS -- not tuned
# per method/condition.
REGROUND_MAX_OUTPUT_TOKENS = 200

# A single-needle/multi-document context/evidence block can be very large
# (Direct's own full document context, in particular). Regrounding is meant
# to re-check the answer against material the method ALREADY saw, not to
# re-run a second full-context pass at unbounded cost -- so this caps how
# much of `context_text` the regrounding prompt actually includes. This is
# an engineering cost-control cap, not a relevance filter: it always takes
# a PREFIX (arbitrary, not chosen by relevance) of already-gathered
# material, never re-selects or re-ranks it.
MAX_REGROUND_CONTEXT_CHARS = 20_000


def apply_context_authoritative_regrounding(
    provider: _TextResponder, question: str, raw_answer: str, context_text: str
) -> str:
    """Single-Needle Contamination Study Condition B (see docs/niah_plus_
    fidelity_audit.md's sibling contamination-study note): a generic,
    method-neutral instruction telling the model to prefer the supplied
    context over memorized world knowledge -- applied the SAME way
    `condense_to_answer_span` is: strictly POST-HOC, after the calling
    method's own reasoning/search/coordination has already fully finished
    and already produced `raw_answer` and `context_text` (whatever
    evidence/context that method already gathered on its own). This is
    NOT injected into any question string used for routing/search/term
    extraction -- for ANT in particular, this must only ever be called
    after `LocalCoordinator.ask()` has already returned, exactly like
    `condense_to_answer_span`'s own module-docstring rationale.

    Applied BEFORE `condense_to_answer_span` when both are used together
    (a Condition-B run must reground first, then condense to the short
    final-answer format).
    """
    if not raw_answer.strip():
        return raw_answer
    prompt = _REGROUND_PROMPT.format(
        question=question,
        context_text=context_text[:MAX_REGROUND_CONTEXT_CHARS],
        raw_answer=raw_answer,
        instruction=CONTEXT_AUTHORITATIVE_INSTRUCTION,
    )
    result = provider.responses_text(prompt, max_output_tokens=REGROUND_MAX_OUTPUT_TOKENS)
    regrounded = result.text.strip()
    return regrounded or raw_answer


def condense_to_answer_span(provider: _TextResponder, question: str, raw_answer: str) -> str:
    """Applied identically, as the LAST step, by all five methods' own
    `run()` -- see module docstring for why this is a post-hoc wrapper
    rather than a prompt-injection change. Uses the SAME provider instance
    the calling method already constructed, so this call's own tokens/cost
    land in that method's ordinary `drain_usage()`/`drain_call_count()`
    totals -- callers must invoke this BEFORE draining, not after.

    A blank raw_answer (e.g. Direct's own context-window-exceeded early
    return, which made no LLM call at all) is returned unchanged -- there
    is nothing to condense, and manufacturing a call here would silently
    inflate that method's own llm_calls count for a task it never actually
    attempted.
    """
    if not raw_answer.strip():
        return raw_answer
    prompt = _CONDENSE_PROMPT.format(
        question=question, raw_answer=raw_answer, instruction=CONCISE_ANSWER_INSTRUCTION
    )
    result = provider.responses_text(prompt, max_output_tokens=CONDENSE_MAX_OUTPUT_TOKENS)
    condensed = result.text.strip()
    return condensed or raw_answer
