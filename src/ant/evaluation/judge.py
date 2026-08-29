from __future__ import annotations

from collections.abc import Mapping

from ant.evaluation.metrics import EvalScore, evaluate_answer
from ant.providers import OpenAIProvider
from ant.providers.openai_provider import _loads_json_object

OFFICIAL_SWE_QA_PRO_JUDGE_MODEL = "gpt-5-2025-08-07"
OFFICIAL_SWE_QA_PRO_JUDGE_REASONING_EFFORT = "low"
OFFICIAL_SWE_QA_PRO_JUDGE_MAX_OUTPUT_TOKENS = 1024

# LOCKED PROMPT -- do not edit the text below.
# This is the exact evaluator prompt the user specified verbatim (message
# timestamped in this session) and asked to be pinned so it cannot drift.
# `test_judge_prompt_is_pinned_and_unmodified` in tests/test_evaluation.py
# hashes this string and will fail the suite if a single character changes.
# If the prompt genuinely needs to change, get explicit confirmation first,
# then update both this constant and the pinned hash in that test together.
JUDGE_PROMPT = """\
You are a professional evaluator. Please rate the candidate answer against the reference answer based on five criteria.
Evaluation Criteria and Scoring Guidelines (each scored 1 to 10):
    1. Correctness:
        10 — Completely correct; core points and details are accurate with no ambiguity.
        8-9 — Mostly correct; only minor details are slightly inaccurate or loosely expressed.
        6-7 — Partially correct; some errors or omissions, but main points are generally accurate.
        4-5 — Several errors or ambiguities that affect understanding of the core information.
        2-3 — Many errors; misleading or fails to convey key information.
        1 — Serious errors; completely wrong or misleading.
    2. Completeness:
        10 — Covers all key points from the reference answer without omission.
        8-9 — Covers most key points; only minor non-critical information missing.
        6-7 — Missing several key points; content is somewhat incomplete.
        4-5 — Important information largely missing; content is one-sided.
        2-3 — Covers very little relevant information; seriously incomplete.
        1 — Covers almost no relevant information; completely incomplete.
    3. Relevance:
        10 — Content fully focused on the question topic; no irrelevant information.
        8-9 — Mostly focused; only minor irrelevant or peripheral information.
        6-7 — Generally on topic; some off-topic content but still relevant overall.
        4-5 — Topic not sufficiently focused; contains considerable off-topic content.
        2-3 — Content deviates from topic; includes excessive irrelevant information.
        1 — Majority of content irrelevant to the question.
    4. Clarity:
        10 — Fluent language; clear and precise expression; very easy to understand.
        8-9 — Mostly fluent; clear expression with minor unclear points.
        6-7 — Generally clear; some expressions slightly unclear or not concise.
        4-5 — Expression somewhat awkward; some ambiguity or lack of fluency.
        2-3 — Language obscure; sentences are not smooth; hinders understanding.
        1 — Expression confusing; very difficult to understand.
    5. Reasoning:
        10 — Reasoning is clear, logical, and well-structured; argumentation is excellent.
        8-9 — Reasoning is clear and logical; well-structured with solid argumentation.
        6-7 — Reasoning generally reasonable; mostly clear logic; minor jumps.
        4-5 — Reasoning is average; some logical jumps or organization issues.
        2-3 — Reasoning unclear; lacks logical order; difficult to follow.
        1 — No clear reasoning; logic is chaotic.

INPUT:
Question:{question}
Reference Answer:{reference}
Candidate Answer:{candidate}

OUTPUT:
Please output ONLY a JSON object with 5 integer fields in the range [1,10], corresponding
to the evaluation scores:
{{
"correctness": <1-10>,
"completeness": <1-10>,
"relevance": <1-10>,
"clarity": <1-10>,
"reasoning": <1-10>
}}

REQUIREMENT:
You should assume that a score of 5 represents an average but imperfect answer.
Scores above 7 should be reserved for answers that are clearly strong.
Do not infer or assume missing information. Score strictly based on what is explicitly stated.
No explanation, no extra text, no formatting other than valid JSON."""


def judge_answer(
    *,
    question: str,
    prediction: str,
    expected: str,
    evidence_count: int,
    unresolved_need_count: int,
    judge: str = "heuristic",
    idf: Mapping[str, float] | None = None,
) -> EvalScore:
    score = evaluate_answer(
        prediction=prediction,
        expected=expected,
        evidence_count=evidence_count,
        unresolved_need_count=unresolved_need_count,
        idf=idf,
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
            # provider is local to this call and never drained anywhere
            # else -- without this, its accumulated cost simply vanishes
            # when the function returns, and every judge="openai" run's
            # actual total spend has been under-reported by however much
            # the judge itself cost, silently, the whole time.
            "usage": provider.drain_usage(),
        }
    )


def _score_int(value: object) -> int:
    if not isinstance(value, int | str | float):
        return 0
    try:
        return max(1, min(10, int(value)))
    except (TypeError, ValueError):
        return 0
