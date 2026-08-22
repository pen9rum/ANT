from __future__ import annotations

from ant.evaluation.metrics import EvalScore, evaluate_answer
from ant.providers import OpenAIProvider
from ant.providers.openai_provider import _loads_json_object

OFFICIAL_SWE_QA_PRO_JUDGE_MODEL = "gpt-5-2025-08-07"
OFFICIAL_SWE_QA_PRO_JUDGE_REASONING_EFFORT = "low"
OFFICIAL_SWE_QA_PRO_JUDGE_MAX_OUTPUT_TOKENS = 1024

JUDGE_PROMPT = """\
You are judging repository-level code question answering.
Score strictly from the provided reference answer and candidate answer.
Return ONLY one JSON object with integer fields from 1 to 10:
correctness, completeness, relevance, clarity, reasoning.

Question:
{question}

Reference answer:
{reference}

Candidate answer:
{candidate}
"""


def judge_answer(
    *,
    question: str,
    prediction: str,
    expected: str,
    evidence_count: int,
    unresolved_need_count: int,
    judge: str = "heuristic",
) -> EvalScore:
    score = evaluate_answer(
        prediction=prediction,
        expected=expected,
        evidence_count=evidence_count,
        unresolved_need_count=unresolved_need_count,
    )
    if judge != "openai" or not expected:
        return score

    provider = OpenAIProvider(
        model=OFFICIAL_SWE_QA_PRO_JUDGE_MODEL,
        reasoning_effort=OFFICIAL_SWE_QA_PRO_JUDGE_REASONING_EFFORT,
    )
    result = provider.responses_json(
        JUDGE_PROMPT.format(question=question, reference=expected, candidate=prediction),
        max_output_tokens=OFFICIAL_SWE_QA_PRO_JUDGE_MAX_OUTPUT_TOKENS,
    )
    data = _loads_json_object(result.text)
    return score.model_copy(
        update={
            "correctness": _score_int(data.get("correctness")),
            "completeness": _score_int(data.get("completeness")),
            "relevance": _score_int(data.get("relevance")),
            "clarity": _score_int(data.get("clarity")),
            "reasoning": _score_int(data.get("reasoning")),
        }
    )


def _score_int(value: object) -> int:
    if not isinstance(value, int | str | float):
        return 0
    try:
        return max(1, min(10, int(value)))
    except (TypeError, ValueError):
        return 0
