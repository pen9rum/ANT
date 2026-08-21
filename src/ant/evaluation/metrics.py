from __future__ import annotations

from pydantic import BaseModel


class EvalScore(BaseModel):
    exact_match: bool
    contains_answer: bool
    evidence_count: int
    unresolved_need_count: int


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
        evidence_count=evidence_count,
        unresolved_need_count=unresolved_need_count,
    )


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
