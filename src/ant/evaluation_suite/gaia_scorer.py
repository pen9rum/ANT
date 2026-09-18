"""GAIA's OFFICIAL scorer, loaded verbatim -- plus the deterministic
`FINAL ANSWER:` extraction the official pipeline expects a *submitter* to
have already done.

WHY THIS IS A LOADER AND NOT AN IMPLEMENTATION: this suite's standing
rule (see `benchmarks/base.BenchmarkAdapter.score`'s own docstring --
"this benchmark's OWN native scorer, never a rubric this module invents
itself") means a re-typed-from-memory normalization scheme would be a
fidelity gap, not a convenience. GAIA's real scorer is published as a
single public file in the `gaia-benchmark/leaderboard` Hugging Face
Space -- which is NOT gated, unlike the dataset repo itself -- so this
pass vendored it byte-for-byte and executes that exact file. Nothing in
this module reimplements normalization; `question_scorer` below is the
official function object, reached through `importlib`.

The vendored bytes are SHA-256-checked at import (see
`_load_official_scorer`). A stray reformat/lint/edit of the vendored
file therefore fails loudly at import time rather than silently shifting
every GAIA score -- see `third_party/manifests/gaia/PROVENANCE.md` for
the pin, the license, and the upstream quirks deliberately preserved.

SCOPE BOUNDARY -- what is official here and what is ours:
  * OFFICIAL, vendored verbatim: `question_scorer` and its normalization
    helpers; `GAIA_SYSTEM_PROMPT` (copied verbatim from the leaderboard
    Space's own `content.py` `SUBMISSION_TEXT`).
  * OURS, disclosed as ours: `extract_final_answer`. GAIA's own scorer
    takes an ALREADY-EXTRACTED `model_answer` -- the official submission
    format is a JSONL of `{"task_id", "model_answer", ...}`, so the
    template-stripping step happens on the submitter's side and has no
    official reference implementation to vendor. This module's version is
    deliberately pure/deterministic (regex + string ops, zero LLM calls),
    unlike `ant.evaluation_suite.answer_extraction`, which is
    model-assisted and is NOT used for GAIA.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

# Pinned artifact identity -- kept in code (not only in PROVENANCE.md) so
# a run manifest can record exactly which scorer bytes produced a number.
OFFICIAL_SCORER_SPACE = "gaia-benchmark/leaderboard"
OFFICIAL_SCORER_REVISION = "9f133d71362e77b3539f1514f31b9c101a545fec"
OFFICIAL_SCORER_SHA256 = "0d44c07f3046eec521697c22e3eaca8719cc81e422a8eaf32695c5f22bdac6e2"

_VENDORED_SCORER_PATH = (
    Path(__file__).resolve().parents[3] / "third_party" / "manifests" / "gaia" / "scorer.py"
)

# Vendored verbatim from the same Space's `content.py` (`SUBMISSION_TEXT`),
# which introduces it as "we use a system prompt to instruct the model
# about the required format". Reproduced EXACTLY -- the trailing-format
# instructions are load-bearing for the scorer's own normalization
# branches (e.g. "don't use comma to write your number" is what keeps a
# numeric answer from being mis-split as a list).
GAIA_SYSTEM_PROMPT = (
    "You are a general AI assistant. I will ask you a question. Report your "
    "thoughts, and finish your answer with the following template: FINAL ANSWER: "
    "[YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as "
    "possible OR a comma separated list of numbers and/or strings. If you are asked "
    "for a number, don't use comma to write your number neither use units such as $ "
    "or percent sign unless specified otherwise. If you are asked for a string, "
    "don't use articles, neither abbreviations (e.g. for cities), and write the "
    "digits in plain text unless specified otherwise. If you are asked for a comma "
    "separated list, apply the above rules depending of whether the element to be "
    "put in the list is a number or a string."
)

_FINAL_ANSWER_RE = re.compile(r"FINAL\s+ANSWER\s*:", re.IGNORECASE)


def _load_official_scorer() -> ModuleType:
    """Executes the vendored official `scorer.py` as a module, after
    verifying its bytes still hash to the pinned SHA-256. Raises rather
    than falling back to anything local -- there is deliberately no
    "reimplemented backup" path, because a silent fallback to our own
    normalization is exactly the fidelity failure this module exists to
    prevent.
    """
    if not _VENDORED_SCORER_PATH.exists():
        raise RuntimeError(
            f"Official GAIA scorer missing at {_VENDORED_SCORER_PATH}. It is vendored "
            f"from the public (ungated) HF Space {OFFICIAL_SCORER_SPACE} at revision "
            f"{OFFICIAL_SCORER_REVISION}; re-download it rather than reimplementing it."
        )
    raw = _VENDORED_SCORER_PATH.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != OFFICIAL_SCORER_SHA256:
        raise RuntimeError(
            f"Vendored GAIA scorer at {_VENDORED_SCORER_PATH} has SHA-256 {actual}, "
            f"expected {OFFICIAL_SCORER_SHA256}. The official scorer must stay "
            "byte-for-byte verbatim -- if upstream genuinely changed, re-vendor it and "
            "update OFFICIAL_SCORER_REVISION/OFFICIAL_SCORER_SHA256 together, and "
            "re-state in PROVENANCE.md what changed."
        )
    spec = importlib.util.spec_from_file_location(
        "ant._vendored_gaia_scorer", _VENDORED_SCORER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not build an import spec for {_VENDORED_SCORER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_OFFICIAL = _load_official_scorer()

# Re-exported official callables. These ARE the upstream function objects
# -- not wrappers, not ports. Tests exercise them directly so the unit
# tests in tests/test_gaia_scorer.py are genuinely testing official
# behaviour, not our description of it.
normalize_number_str = _OFFICIAL.normalize_number_str
normalize_str = _OFFICIAL.normalize_str
split_string = _OFFICIAL.split_string
_official_question_scorer = _OFFICIAL.question_scorer


@dataclass(frozen=True)
class FinalAnswerExtraction:
    """Result of stripping GAIA's official response template.

    `template_found` is kept rather than thrown away because "the model
    never emitted FINAL ANSWER:" is a real, reportable failure mode
    (format non-compliance) that is worth separating from "the model
    emitted a wrong answer" when reading results -- the official pipeline
    can't distinguish them, since it only ever sees the extracted string.
    """

    answer: str
    template_found: bool
    raw: str


def extract_final_answer(raw_response: str | None) -> FinalAnswerExtraction:
    """Deterministically pull the answer out of GAIA's official
    `FINAL ANSWER: ...` template. THIS SUITE'S OWN CODE, not official --
    see the module docstring for why there is nothing official to vendor
    here.

    Rules, all chosen to be conservative (never "improve" the answer, only
    strip the template scaffolding):

    * The LAST `FINAL ANSWER:` occurrence wins (case-insensitive, tolerant
      of extra internal whitespace). A model that restates the template
      while reasoning and then answers should be read at its conclusion,
      not its first mention.
    * Only the remainder of THAT line is taken. GAIA answers are a number,
      a few words, or a comma-separated list -- always single-line -- so
      trailing prose after a newline is scaffolding, never answer content.
    * Markdown emphasis wrapping the whole answer (`**x**`, `*x*`, `_x_`)
      and surrounding brackets left over from the literal template
      placeholder (`[YOUR FINAL ANSWER]`) are stripped.
    * A single trailing sentence-final period is dropped. Nothing else is
      touched: no casing change, no internal punctuation removal, no
      unit stripping -- the OFFICIAL scorer owns all of that, and doing
      any of it here would double-normalize and could change outcomes.
    * No template at all -> the whole response, trimmed, with
      `template_found=False`. Failing closed to an empty string would
      silently convert a format miss into a guaranteed wrong answer and
      hide the distinction; the official scorer will judge it on merit.
    """
    raw = raw_response or ""
    matches = list(_FINAL_ANSWER_RE.finditer(raw))
    if not matches:
        return FinalAnswerExtraction(answer=raw.strip(), template_found=False, raw=raw)
    tail = raw[matches[-1].end() :]
    answer = tail.split("\n", 1)[0].strip()
    answer = _strip_wrappers(answer)
    return FinalAnswerExtraction(answer=answer, template_found=True, raw=raw)


def _strip_wrappers(text: str) -> str:
    """Peels template scaffolding off the answer span until it stops
    changing. Emphasis-stripping and sentence-period-stripping run in the
    SAME fixed-point loop, not in sequence, because they interleave in
    real responses: `**Paris**.` needs the period gone before the `**`
    pair becomes strippable, while `**Paris.**` needs the opposite order.
    One pass in either fixed order would leave one of those two cases
    half-stripped.
    """
    previous = None
    while previous != text:
        previous = text
        text = text.strip()
        for opener, closer in (("**", "**"), ("*", "*"), ("_", "_"), ("[", "]")):
            if (
                len(text) > len(opener) + len(closer)
                and text.startswith(opener)
                and text.endswith(closer)
            ):
                text = text[len(opener) : -len(closer)]
        text = _strip_sentence_period(text).strip()
    return text


def _strip_sentence_period(text: str) -> str:
    """Drops ONE trailing sentence-final period. Deliberately refuses to
    touch a period preceded by a digit: `3.` parses as a float upstream,
    and an answer like `2.` could be a genuinely truncated decimal -- so
    only a period after a non-digit is treated as prose punctuation.
    Ellipses are left entirely alone.
    """
    if not text.endswith(".") or text.endswith(".."):
        return text
    if len(text) >= 2 and text[-2].isdigit():
        return text
    return text[:-1]


def question_scorer(model_answer: str | None, ground_truth: str) -> bool:
    """The OFFICIAL `question_scorer`, called with its stdout chatter and
    `UserWarning` suppressed.

    Suppression is at the CALL boundary only -- the vendored file is
    untouched and its logic is unchanged. Upstream prints one line per
    call and warns on list-length mismatch, which is reasonable for a
    leaderboard script run by hand but would bury a 165-task x 4-method
    sweep's real output. The boolean this returns is bit-identical to
    calling upstream directly (asserted in tests/test_gaia_scorer.py).
    """
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return bool(_official_question_scorer(model_answer, ground_truth))


def score_response(
    raw_response: str | None, ground_truth: str
) -> tuple[bool, FinalAnswerExtraction]:
    """End-to-end convenience: template extraction (ours) followed by the
    official scorer (theirs). Returns both the verdict and the extraction
    record, so a caller can always report WHAT was scored, not just the
    outcome -- the adapter stores the extraction in MetricResult.metadata
    for exactly that reason.
    """
    extraction = extract_final_answer(raw_response)
    return question_scorer(extraction.answer, ground_truth), extraction
