"""Direct baseline for the long-document/multi-document evaluation track
(Section 5 of the long-context evaluation spec): question + the FULL
benchmark context (every document, verbatim, in original order) -> GPT-4.1
-> answer. No tools, no retrieval, no iteration -- the same "single call,
whole information universe in the prompt" definition of Direct the spec
requires, which is deliberately NOT the same definition as this suite's
existing repository-QA `ant.agents.direct.DirectAgent` (a CLOSED-BOOK
lower bound there -- the question alone, no repository access at all).
Registered under a distinct name (`direct_document`) precisely because
these are two different, both-legitimate definitions of "Direct" for two
different substrates; neither name is reused for the other's behavior.
"""
from __future__ import annotations

import time
from pathlib import Path

import tiktoken

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.answer_contract import (
    apply_context_authoritative_regrounding,
    condense_to_answer_span,
)
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.document_scope import DocumentRecord
from ant.evaluation_suite.usage import UsageStats

_ENCODING_NAME = "cl100k_base"

# GPT-4.1's real context window is 1,047,576 tokens. This is set somewhat
# below that (not exactly at it) to leave headroom for the prompt's own
# instruction text, the question, and the model's own output allocation --
# an engineering safety margin, not a benchmark-tuned value. Exceeding this
# STOPS the task (see run() below) rather than silently truncating any
# document's text -- the policy Section 5 explicitly requires, deferring
# what to do about an over-limit task to a separate decision rather than
# quietly right-truncating context and reporting a real answer against a
# lossy input.
MAX_CONTEXT_TOKENS = 900_000

_DOCUMENT_DIRECT_PROMPT = """Answer the question using ONLY the documents provided below. If the \
documents do not contain enough information to answer, say so plainly rather than guessing.

# Documents:
{context}

# Question:
{question}

Respond with your final answer only, no explanation of your reasoning process."""


def _full_context(documents: list[DocumentRecord]) -> str:
    parts = []
    for document in documents:
        header = f"Title: {document.title}" if document.title else ""
        parts.append(f"{header}\n\n{document.text}".strip())
    return "\n\n---\n\n".join(parts)


class DirectDocumentAgent:
    """Tier 1 / Direct for the document substrate: full-context, single
    call, no environment access beyond what's in the prompt. See module
    docstring for why this is a distinct class/name from the repository-QA
    `DirectAgent`, not a shared/renamed one.
    """

    name = "direct_document"

    def __init__(self, model: str = "gpt-4.1") -> None:
        self.model = model

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        del environment_root  # documents come from example.metadata, not disk
        documents = [DocumentRecord(**d) for d in example.metadata["documents"]]
        context = _full_context(documents)
        prompt = _DOCUMENT_DIRECT_PROMPT.format(context=context, question=example.question)

        encoding = tiktoken.get_encoding(_ENCODING_NAME)
        prompt_tokens = len(encoding.encode(prompt, disallowed_special=()))
        if prompt_tokens > MAX_CONTEXT_TOKENS:
            return AgentResult(
                benchmark=example.benchmark,
                task_id=example.task_id,
                method=self.name,
                final_answer="",
                trajectory=[{"error": "context_window_exceeded", "prompt_tokens": prompt_tokens}],
                usage=UsageStats(llm_calls=0, wall_clock_seconds=0.0),
                termination_reason="context_window_exceeded",
                metadata={
                    "generation_model": self.model,
                    "prompt_tokens_estimate": prompt_tokens,
                    "max_context_tokens": MAX_CONTEXT_TOKENS,
                },
            )

        provider = CountingOpenAIProvider(model=self.model)
        started = time.time()
        result = provider.responses_text(prompt, max_output_tokens=1024)
        # Shared short-answer contract (Part A): applied as the LAST step,
        # after the method's own answer is fully computed -- see
        # ant.evaluation_suite.answer_contract's own module docstring for
        # why this is a post-hoc wrapper, never a prompt-injection change,
        # and identical across all five methods.
        raw_answer = result.text.strip()
        # Single-Needle Contamination Study Condition B (opt-in via
        # metadata, no-op for every other track/condition): a generic
        # context-authoritative instruction, applied post-hoc to the
        # already-produced answer against the SAME full context this
        # method already saw -- never re-injected into the original
        # answer-generation prompt itself.
        grounded_answer = raw_answer
        if example.metadata.get("answer_contract_condition") == "B":
            grounded_answer = apply_context_authoritative_regrounding(
                provider, example.question, raw_answer, context
            )
        final_answer = condense_to_answer_span(provider, example.question, grounded_answer)
        token_usage = provider.drain_usage()
        llm_calls = provider.drain_call_count()
        elapsed = time.time() - started

        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=final_answer,
            trajectory=[{"prompt_tokens_estimate": prompt_tokens}],
            usage=UsageStats(
                llm_calls=llm_calls,
                input_tokens=token_usage.input_tokens,
                output_tokens=token_usage.output_tokens,
                total_tokens=token_usage.total_tokens,
                estimated_cost_usd=token_usage.estimated_cost_usd,
                wall_clock_seconds=elapsed,
                unique_files_inspected=len(documents),
            ),
            termination_reason="single_call_complete",
            metadata={
                "generation_model": self.model,
                "prompt_tokens_estimate": prompt_tokens,
                "raw_answer_before_condensation": raw_answer,
                "answer_contract_condition": example.metadata.get("answer_contract_condition"),
                "grounded_answer_after_regrounding": grounded_answer
                if grounded_answer != raw_answer
                else None,
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(DirectDocumentAgent())
