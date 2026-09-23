"""Shared GAIA task wiring -- every one of this suite's 7 GAIA methods
(Direct, Sparse Retrieval, Dense Retrieval, Matched ReAct, S2G-RAG,
ANTMAN, ANTMAN-H) needs the identical `GaiaEnvironment` + `GaiaToolRegistry`
construction for a given task; this module is the one place that
wiring happens, so every method is provably looking at the same
substrate rather than seven independently-assembled near-duplicates.

`build_registry` reads only `environment_root` (from
`GaiaAdapter.prepare_environment`) and `example.metadata["file_name"]`
-- never `example.reference` or `example.metadata["_audit_only"]`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ant.agents.gaia_tools import GaiaToolRegistry
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.gaia_scope import GaiaEnvironment
from ant.evaluation_suite.gaia_scorer import GAIA_SYSTEM_PROMPT
from ant.evaluation_suite.gaia_web_backends import (
    build_default_fetch_backend,
    build_default_search_backend,
)

# Every GAIA method's own final-answer call must elicit the official
# `FINAL ANSWER: ...` template (see `gaia_scorer.py`'s own docstring) --
# the vendored official scorer's `question_scorer` is only ever run
# against an ALREADY-EXTRACTED answer, and `extract_final_answer` (this
# suite's own code) extracts by looking for that literal template. A
# method whose prompt never asks for it will still get scored (the
# extractor falls back to the raw text), but near-certainly wrong under
# quasi-exact-match, and `final_answer_template_found=False` in the
# result would flag it as a FORMAT failure rather than a content one --
# not a fair representation of the method's actual capability. This
# constant is threaded into every one of the 7 GAIA agents' own
# answer-eliciting prompt for exactly that reason.
GAIA_ANSWER_FORMAT_INSTRUCTIONS = GAIA_SYSTEM_PROMPT


def build_environment(example: TaskExample, environment_root: Path) -> GaiaEnvironment:
    return GaiaEnvironment(environment_root, example.metadata.get("file_name"))


def build_registry(
    example: TaskExample,
    environment_root: Path,
    *,
    max_search_results: int = 10,
) -> GaiaToolRegistry:
    """One registry per task call, real Tavily-primary/DuckDuckGo-fallback
    search and real fetch -- see `gaia_web_backends`'s own module docstring
    for why real network access is this substrate's whole point, unlike
    every other benchmark in this suite."""
    environment = build_environment(example, environment_root)
    return GaiaToolRegistry(
        environment=environment,
        search_backend=build_default_search_backend(),
        fetch_backend=build_default_fetch_backend(),
        max_search_results=max_search_results,
    )


def build_answer_prompt(question: str, context_block: str) -> str:
    """One shared prompt shape for the three non-agentic GAIA methods
    (Direct/Sparse/Dense) -- the official format instructions, the
    question, and whatever context that method gathered (empty string for
    Direct). Matched ReAct/S2G-RAG/ANTMAN each weave the same
    `GAIA_ANSWER_FORMAT_INSTRUCTIONS` into their own existing prompt shape
    instead of this one, since they already have a richer prompt
    structure of their own (see each file's own docstring)."""
    if context_block.strip():
        return (
            f"{GAIA_ANSWER_FORMAT_INSTRUCTIONS}\n\n"
            f"Question: {question}\n\nContext:\n{context_block}"
        )
    return f"{GAIA_ANSWER_FORMAT_INSTRUCTIONS}\n\nQuestion: {question}"


_REFORMAT_PROMPT = (
    GAIA_ANSWER_FORMAT_INSTRUCTIONS
    + "\n\nSomeone already worked out the answer to this question below. Do not "
    "re-derive it or change its content -- restate it, then finish with the "
    "required FINAL ANSWER: template exactly as instructed above.\n\n"
    "Question: {question}\n\nWorked-out answer: {raw_answer}"
)


def reformat_to_gaia_template(provider: Any, question: str, raw_answer: str) -> str:
    """For methods whose own answer-producing call/loop is FROZEN and
    reused as-is (ANTMAN/ANTMAN-H via `LocalCoordinator.ask()`) --
    baking `GAIA_ANSWER_FORMAT_INSTRUCTIONS` into the question text
    itself would risk perturbing ANTMAN's own Need Graph decomposition
    with unrelated formatting instructions, a confound this comparison
    cannot afford. Instead, ANTMAN's canonical `state.answer` is computed
    completely undisturbed, and ONE extra, clearly-labeled call restates
    it (content unchanged, per the prompt's own instruction) in GAIA's
    required template -- the same "keep the method's own reasoning
    untouched, adapt only what's needed to satisfy the substrate's output
    contract" principle every other GAIA adapter in this suite already
    follows for its own answer stage."""
    if not raw_answer.strip():
        return raw_answer
    prompt = _REFORMAT_PROMPT.format(question=question, raw_answer=raw_answer)
    result = provider.responses_text(prompt, max_output_tokens=512)
    return result.text.strip()


def read_attachment_text_safely(registry: GaiaToolRegistry, max_chars: int = 20_000) -> str:
    """Best-effort attachment text for methods that want to fold it
    directly into a prompt/evidence pool (Direct/Sparse/Dense) rather than
    treat it as a separately-invoked tool. Returns "" for no attachment or
    an unsupported/unreadable one -- never raises, since a Level-1
    text-only GAIA question has no attachment at all and that must not be
    an error."""
    if not registry.environment.has_attachment():
        return ""
    try:
        return registry.inspect_file(limit=max_chars)
    except Exception:  # noqa: BLE001 -- best-effort, see docstring
        return ""
