"""Strict, method-agnostic answer-span extraction, applied at SCORING
TIME only -- never during generation, never rerunning any agent. This is
a distinct, later layer from `ant.evaluation_suite.answer_contract`'s
`condense_to_answer_span` (see this module's own audit note below for
why that existing layer is insufficient on its own).

Audit of the existing answer contract (performed before writing this
module, not inferred from names):

- `condense_to_answer_span` IS already applied, identically, as the LAST
  step of all five document-track methods' own `run()` -- confirmed both
  by reading each agent's source and by inspecting a live trajectory
  (`output/runs/natural-multidoc-pilot/hotpotqa/trajectories/
  ant_document-5a8b57f25542995d1e6f1371.json`): `raw_answer_before_
  condensation` holds an ~1800-character multi-paragraph analysis;
  `final_answer` (post-condensation) holds "Yes, both were American." --
  condensation clearly ran and clearly shortened the answer dramatically.
- It is LLM-based (one `responses_text` call), not deterministic.
- Every benchmark adapter's own `score()` (`score_hotpot_style`,
  `NiahPlusAdapter.score`, `MuSiQueAdapter.score`) calls
  `score_qa(result.final_answer, ground_truths)` -- i.e. scoring already
  used the POST-condensation answer, not the raw pre-condensation text.
  Confirmed by direct code inspection, not assumed from the function name.
- Why the verbose examples survived: `condense_to_answer_span`'s own
  prompt instructs the model to "extract OR RESTATE" the minimal answer
  span -- the word "restate" explicitly permits a rewritten grammatical
  sentence rather than requiring a literal minimal substring. The model's
  own notion of "minimal" stopped at a complete, well-formed sentence
  ("Yes, both were American.") rather than the bare token ("yes")
  HotpotQA's own gold convention expects. This is the root cause this
  module fixes.

Design: this module does NOT replace or modify `condense_to_answer_span`
(frozen, per instruction) -- it is a SEPARATE, later, scoring-side-only
layer, applied by a rescoring script to already-saved `prediction`/
`final_answer` values, never during generation. It receives ONLY
`question` and `raw_model_answer` -- never gold, aliases, supporting
facts, needle locations, method identity, or score information.

Extraction is deterministic wherever possible (canonical yes/no
normalization, per the spec's own explicit allowance) and falls back to
one frozen, temperature=0 LLM call otherwise. In BOTH cases, the result
is validated against a hard safety rule: except for yes/no
normalization, the extracted text must be a verbatim (case-insensitive)
substring of the raw answer. Any violation -- including an LLM attempting
to "repair" a wrong answer, invent an entity, or use outside knowledge --
is rejected and the ORIGINAL raw answer is returned unchanged.

Revision (post first-rescoring audit, before any further rescoring or
the canonical rerun -- see docs/long_context_evaluation_fix_report.md
Section 4): the first frozen version optimized purely for "shortest
span," which over-shortened a class of already-correct, already-minimal
answers -- most visibly compound locations like "Greenwich Village, New
York City" cut down to just "Greenwich Village," turning an exact match
into a partial one. The goal is now "shortest span that FULLY answers
the question, preserving required qualifiers/components" (reworded in
the prompt below), backed by a deterministic, narrowly-targeted
safeguard (`_extend_over_trailing_qualifier`) that restores a dropped
trailing qualifier when the candidate is a strict prefix of the raw
answer and everything cut off is a single comma-attached, Title-Case
qualifying phrase running all the way to the end (the City/State/Country
pattern) -- never a general "always extend" rule, and it does not touch
list-style ambiguity (semicolon-separated candidates, or a dropped
clause not immediately comma-attached), which is a distinct,
already-documented limitation this revision does not attempt to fix.
This is the only revision made to this module after the first freeze,
made in response to the audited over-shortening pattern (a formatting
defect affecting all methods symmetrically), not in response to any
method's score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.providers.openai_provider import _loads_json_object

_YES_NO_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)

_EXTRACTION_PROMPT = """Question: {question}

Answer text (already produced by another system, to be shortened only -- not replaced):
{raw_answer}

Extract the SHORTEST span of text, copied VERBATIM character-for-character from the Answer text \
above, that FULLY and PRECISELY answers the Question. "Shortest" means: do not include \
surrounding sentences, filler phrases ("The answer is...", "Yes, ..."), or extra explanation. It \
does NOT mean dropping a qualifying component that is part of the answer itself -- if the answer \
is a compound name, a location with its city/state/country/region together (e.g. "Greenwich \
Village, New York City", "Paris, France"), or a date with day/month/year together, keep that \
whole component intact as one contiguous span. When in doubt between a shorter partial span and a \
slightly longer span that keeps the full precise answer, prefer the longer, complete one. Do not \
add, remove, correct, or change any words from the copied span. Do not use outside knowledge. Do \
not fix the answer if it looks wrong -- only copy what is already there. If the Answer text is \
already short and cannot be shortened further without losing part of the answer, copy it verbatim \
in full.

Respond with a JSON object of exactly this form, nothing else:
{{"extracted_span": "<verbatim substring copied from the Answer text above>"}}"""

# Deterministic safeguard for the confirmed over-shortening pattern: a
# candidate that is a strict prefix of the (period-stripped) raw answer,
# where everything cut off is a SINGLE comma-attached, Title-Case
# qualifying phrase running all the way to the end of the raw answer
# (the "City, State/Country"-style pattern). A small set of lowercase
# connector words is allowed inside that phrase (e.g. "United States of
# America") without weakening the overall Title-Case signal. Deliberately
# anchored and comma-specific -- it must NOT match a dropped clause that
# is merely SPACE-attached ("... by K. A. Applegate", "... while visiting
# her daughter") or semicolon-separated list items, both of which are
# legitimate shortening/selection decisions this safeguard leaves alone.
_QUALIFIER_WORD = r"(?:[A-Z][\w.&'-]*|of|the|de|la|and)"
_TRAILING_QUALIFIER_RE = re.compile(rf"^,\s+{_QUALIFIER_WORD}(?:\s+{_QUALIFIER_WORD})*$")


def _extend_over_trailing_qualifier(candidate: str, raw_answer: str) -> str:
    """See `_TRAILING_QUALIFIER_RE` above. Only ever *lengthens* `candidate`
    by pulling MORE verbatim text from `raw_answer` -- the result remains
    trivially a substring of `raw_answer` by construction (it is sliced
    directly from it), so no additional hallucination risk is introduced.
    """
    stripped_raw = raw_answer.strip().rstrip(".").rstrip()
    stripped_candidate = candidate.strip().rstrip(".").rstrip()
    if not stripped_candidate or not stripped_raw.lower().startswith(stripped_candidate.lower()):
        return candidate
    remainder = stripped_raw[len(stripped_candidate) :]
    if _TRAILING_QUALIFIER_RE.match(remainder):
        return stripped_raw
    return candidate


# Fixed, small output budget -- an extracted span is never long. Not
# tuned per benchmark/method.
EXTRACTION_MAX_OUTPUT_TOKENS = 128


class _ZeroTemperatureProvider(CountingOpenAIProvider):
    """Interface-only override (Category A): `_responses_kwargs` is the
    single choke point both of `OpenAIProvider.responses_text`'s own
    request-construction branches already funnel through, so overriding
    it here injects `temperature=0` into every physical call this
    provider instance makes, without touching
    `ant/providers/openai_provider.py` (frozen core) at all -- the exact
    same subclassing pattern `CountingOpenAIProvider` itself already
    uses one level up. gpt-4.1 (the model every document-track method
    uses) is not a reasoning model, so this never collides with the
    `reasoning` kwarg branch (`self.reasoning_effort` stays unset here).
    """

    def _responses_kwargs(self, prompt: str, max_output_tokens: int) -> dict:
        kwargs = super()._responses_kwargs(prompt, max_output_tokens)
        kwargs["temperature"] = 0
        return kwargs


@dataclass
class ExtractionResult:
    extracted_answer: str
    used_llm: bool
    llm_calls: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    rejected_hallucination: bool = False
    raw_llm_output: str | None = None


def _is_verbatim_substring(candidate: str, raw_answer: str) -> bool:
    """The hard safety rule: case-insensitive, whitespace-normalized
    substring check. Never allows semantic/fuzzy matching -- a candidate
    that is not a literal contiguous substring of `raw_answer` is
    rejected outright, regardless of how plausible it looks."""
    if not candidate.strip():
        return False
    normalized_candidate = " ".join(candidate.split()).lower()
    normalized_raw = " ".join(raw_answer.split()).lower()
    return normalized_candidate in normalized_raw


def _deterministic_yes_no(raw_answer: str) -> str | None:
    match = _YES_NO_RE.match(raw_answer)
    if not match:
        return None
    return match.group(1).lower()


def extract_answer_span(
    question: str, raw_model_answer: str, *, model: str = "gpt-4.1"
) -> ExtractionResult:
    """The one shared, method-agnostic extraction layer. Receives ONLY
    `question` and `raw_model_answer` -- see module docstring for the
    full no-leakage contract. Prefers deterministic yes/no normalization
    (no LLM call, no cost) and falls back to exactly one frozen,
    temperature=0 LLM call otherwise, with a hard verbatim-substring
    validation gate before ever trusting the LLM's own output.
    """
    if not raw_model_answer.strip():
        return ExtractionResult(extracted_answer=raw_model_answer, used_llm=False)

    deterministic = _deterministic_yes_no(raw_model_answer)
    if deterministic is not None:
        return ExtractionResult(extracted_answer=deterministic, used_llm=False)

    provider = _ZeroTemperatureProvider(model=model)
    prompt = _EXTRACTION_PROMPT.format(question=question, raw_answer=raw_model_answer)
    result = provider.responses_text(prompt, max_output_tokens=EXTRACTION_MAX_OUTPUT_TOKENS)
    llm_calls = provider.drain_call_count()
    usage = provider.drain_usage()

    try:
        parsed = _loads_json_object(result.text)
        candidate = str(parsed.get("extracted_span", "")).strip()
    except Exception:  # noqa: BLE001 -- malformed extractor output is a safe-fallback case, not a crash
        candidate = ""

    if candidate and _is_verbatim_substring(candidate, raw_model_answer):
        candidate = _extend_over_trailing_qualifier(candidate, raw_model_answer)
        return ExtractionResult(
            extracted_answer=candidate,
            used_llm=True,
            llm_calls=llm_calls,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=usage.estimated_cost_usd,
            raw_llm_output=result.text,
        )

    # Safety net: the proposed extraction was not a verbatim substring of
    # the raw answer (a hallucination/repair/invention attempt, or a
    # parse failure) -- reject it and fall back to the raw answer
    # completely unchanged. The physical call still happened (real cost),
    # so it is still logged, but its OUTPUT is never trusted.
    return ExtractionResult(
        extracted_answer=raw_model_answer,
        used_llm=True,
        llm_calls=llm_calls,
        total_tokens=usage.total_tokens,
        estimated_cost_usd=usage.estimated_cost_usd,
        rejected_hallucination=True,
        raw_llm_output=result.text,
    )
