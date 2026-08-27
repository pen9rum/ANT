import json
from pathlib import Path

from ant.evaluation.metrics import evaluate_answer
from ant.evaluation.report import build_report


def test_f1_is_1_for_identical_text() -> None:
    score = evaluate_answer(
        prediction="The cat sat on the mat.",
        expected="the cat sat on the mat.",
        evidence_count=1,
        unresolved_need_count=0,
    )
    assert score.exact_match is True
    assert score.f1 == 1.0


def test_f1_is_0_for_completely_disjoint_text() -> None:
    score = evaluate_answer(
        prediction="apples oranges bananas",
        expected="quantum circuit simulation",
        evidence_count=1,
        unresolved_need_count=0,
    )
    assert score.exact_match is False
    assert score.f1 == 0.0


def test_f1_reflects_partial_token_overlap() -> None:
    # pred tokens: the(x2) cat sat on mat -- 6 total
    # gold tokens: the cat sat on a mat -- 6 total
    # overlap (multiset min): the=1 cat=1 sat=1 on=1 mat=1 -> 5
    # precision = recall = 5/6 -> f1 = 5/6
    score = evaluate_answer(
        prediction="the cat sat on the mat",
        expected="the cat sat on a mat",
        evidence_count=1,
        unresolved_need_count=0,
    )
    assert score.exact_match is False
    assert score.f1 == round(5 / 6, 6)


def test_f1_is_0_when_expected_is_empty() -> None:
    score = evaluate_answer(
        prediction="anything", expected="", evidence_count=0, unresolved_need_count=0
    )
    assert score.f1 == 0.0


def test_build_report_aggregates_f1(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    rows = [
        {
            "example_id": "a",
            "question": "q",
            "prediction": "p",
            "score": {
                "exact_match": True,
                "contains_answer": True,
                "f1": 1.0,
                "evidence_count": 1,
                "unresolved_need_count": 0,
            },
        },
        {
            "example_id": "b",
            "question": "q",
            "prediction": "p",
            "score": {
                "exact_match": False,
                "contains_answer": False,
                "f1": 0.5,
                "evidence_count": 1,
                "unresolved_need_count": 0,
            },
        },
    ]
    results_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    report = build_report(results_path)

    assert report.avg_f1 == 0.75
