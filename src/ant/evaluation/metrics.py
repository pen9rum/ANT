from __future__ import annotations

from pydantic import BaseModel, Field

from ant.domain import TokenUsage


class EvalScore(BaseModel):
    exact_match: bool
    contains_answer: bool
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
        evidence_count=evidence_count,
        unresolved_need_count=unresolved_need_count,
    )


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
