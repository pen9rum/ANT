from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping

from pydantic import BaseModel, Field

from ant.domain import TokenUsage

_F1_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EvalScore(BaseModel):
    exact_match: bool
    contains_answer: bool
    # Token-overlap F1 between prediction and expected (SQuAD-style bag-of-
    # words precision/recall over _normalize()'d tokens, IDF-weighted
    # against the batch's own reference-answer corpus when one is supplied
    # -- see build_reference_idf) -- a softer signal than exact_match,
    # which is almost always False for a free-text multi-sentence answer
    # even when it's substantively correct. Plain (unweighted) token F1 was
    # confirmed to over-reward answers that share a question's own domain
    # vocabulary without actually citing anything specific -- e.g. a
    # closed-book answer that hedges in genre-appropriate jargon ("Bloch
    # vector", "matplotlib", "expectation values") scored a top-of-batch F1
    # despite admitting it didn't know the actual file/function names.
    # IDF-weighting down-weights exactly that shared, generic vocabulary
    # (common across most reference answers in the batch) relative to the
    # rare, specific tokens -- identifiers, numbers, file names -- that
    # actually distinguish a grounded citation from a fluent guess.
    f1: float = 0.0
    evidence_count: int
    unresolved_need_count: int
    correctness: int = 0
    completeness: int = 0
    relevance: int = 0
    clarity: int = 0
    reasoning: int = 0
    # The judge call's own usage/cost (judge_answer, judge="openai") --
    # deliberately separate from EvidenceState.usage/BatchResult.usage
    # (the main run's orchestrator/worker/synthesis cost), since the judge
    # runs on a different, hash-locked model (see
    # OFFICIAL_SWE_QA_PRO_JUDGE_MODEL) and mixing the two would make either
    # figure meaningless on its own. Zero for judge="heuristic".
    usage: TokenUsage = Field(default_factory=TokenUsage)


def evaluate_answer(
    *,
    prediction: str,
    expected: str,
    evidence_count: int,
    unresolved_need_count: int,
    idf: Mapping[str, float] | None = None,
) -> EvalScore:
    normalized_prediction = _normalize(prediction)
    normalized_expected = _normalize(expected)
    exact = bool(normalized_expected) and normalized_prediction == normalized_expected
    contains = bool(normalized_expected) and normalized_expected in normalized_prediction
    return EvalScore(
        exact_match=exact,
        contains_answer=contains,
        f1=_f1_score(normalized_prediction, normalized_expected, idf),
        evidence_count=evidence_count,
        unresolved_need_count=unresolved_need_count,
    )


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _f1_tokens(normalized_text: str) -> list[str]:
    # A plain .split() on _normalize()'s output leaves trailing/attached
    # punctuation glued to words ("module," / "implemented." / "qibo's"),
    # so the same word in two different sentence positions counts as two
    # different tokens -- confirmed this was silently fragmenting document
    # frequency counts for build_reference_idf specifically (the term that
    # matters most there is exactly "how many references mention this
    # word," which punctuation-splitting corrupts), and to a lesser extent
    # plain overlap counting too. _normalize's own output is untouched --
    # exact_match/contains_answer are legitimately string-level, where
    # punctuation is part of "exact."
    return _F1_TOKEN_RE.findall(normalized_text)


def build_reference_idf(references: Iterable[str]) -> dict[str, float]:
    """Smoothed IDF table (sklearn's convention: ln((N+1)/(df+1)) + 1, so
    every weight stays positive and a term appearing in every reference
    still counts for something) over a batch's own reference-answer
    corpus. Call once per batch/eval run -- over `examples` before scoring
    any of them, not per-question -- and thread the same table into every
    _f1_score call in that batch; a per-question corpus of one document
    would make every token maximally "rare" and defeat the point.

    Deliberately built from *reference* answers only, not predictions:
    the reference corpus is what defines "this batch's own generic,
    shared vocabulary" (question-domain words every honest answer would
    use) vs. the specific tokens -- identifiers, file names, numbers --
    that actually distinguish a grounded citation. Predictions vary in
    quality and would bias the weighting toward whatever a given
    condition's answers happen to repeat.
    """
    references = [ref for ref in references if ref]
    doc_count = len(references)
    if doc_count == 0:
        return {}
    document_frequency: Counter[str] = Counter()
    for reference in references:
        document_frequency.update(set(_f1_tokens(_normalize(reference))))
    return {
        term: math.log((doc_count + 1) / (freq + 1)) + 1
        for term, freq in document_frequency.items()
    }


def _f1_score(
    normalized_prediction: str,
    normalized_expected: str,
    idf: Mapping[str, float] | None = None,
) -> float:
    if not normalized_expected:
        return 0.0
    pred_tokens = _f1_tokens(normalized_prediction)
    gold_tokens = _f1_tokens(normalized_expected)
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    if idf is None:
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
    else:
        # A token absent from the reference corpus entirely (df=0 in
        # build_reference_idf's terms) gets the corpus's own maximum
        # possible weight, ln(doc_count+1)+1 -- the same value a term
        # with true df=0 would have received had it been in the table.
        # This intentionally does NOT reward hallucination: it means an
        # invented-sounding specific token is weighted as *rare*, exactly
        # like a real specific token would be -- _f1_score has no way to
        # know which is which, only that neither is generic filler.
        fallback = max(idf.values(), default=1.0) if idf else 1.0
        num_same = sum(count * idf.get(token, fallback) for token, count in common.items())
        if num_same == 0:
            return 0.0
        pred_weight = sum(idf.get(token, fallback) for token in pred_tokens)
        gold_weight = sum(idf.get(token, fallback) for token in gold_tokens)
        if pred_weight == 0 or gold_weight == 0:
            return 0.0
        precision = num_same / pred_weight
        recall = num_same / gold_weight
    return round(2 * precision * recall / (precision + recall), 6)
