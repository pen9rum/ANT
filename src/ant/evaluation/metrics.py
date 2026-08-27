from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from ant.domain import TokenUsage


class EvalScore(BaseModel):
    exact_match: bool
    contains_answer: bool
    # Token-overlap F1 between prediction and expected (SQuAD-style bag-of-
    # words precision/recall over _normalize()'d tokens) -- a softer signal
    # than exact_match, which is almost always False for a free-text
    # multi-sentence answer even when it's substantively correct.
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
) -> EvalScore:
    normalized_prediction = _normalize(prediction)
    normalized_expected = _normalize(expected)
    exact = bool(normalized_expected) and normalized_prediction == normalized_expected
    contains = bool(normalized_expected) and normalized_expected in normalized_prediction
    return EvalScore(
        exact_match=exact,
        contains_answer=contains,
        f1=_f1_score(normalized_prediction, normalized_expected),
        evidence_count=evidence_count,
        unresolved_need_count=unresolved_need_count,
    )


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _f1_score(normalized_prediction: str, normalized_expected: str) -> float:
    if not normalized_expected:
        return 0.0
    pred_tokens = normalized_prediction.split()
    gold_tokens = normalized_expected.split()
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return round(2 * precision * recall / (precision + recall), 6)
