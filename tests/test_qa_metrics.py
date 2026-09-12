from __future__ import annotations

from ant.evaluation_suite.qa_metrics import (
    compute_exact,
    compute_f1,
    metric_max_over_ground_truths,
    normalize_answer,
    score_qa,
)


def test_normalize_answer_lowercases_strips_articles_and_punctuation() -> None:
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("A cat, a dog, and an owl.") == "cat dog and owl"


def test_normalize_answer_collapses_whitespace() -> None:
    assert normalize_answer("  too   many   spaces  ") == "too many spaces"


def test_compute_exact_matches_after_normalization_only() -> None:
    assert compute_exact("The Eiffel Tower", "eiffel tower") == 1.0
    assert compute_exact("Eiffel Tower", "Big Ben") == 0.0


def test_compute_f1_partial_overlap() -> None:
    score = compute_f1("the quick brown fox", "quick brown")
    assert 0.0 < score < 1.0


def test_compute_f1_no_overlap_is_zero() -> None:
    assert compute_f1("apple", "orange") == 0.0


def test_compute_f1_both_empty_after_normalization_is_one() -> None:
    assert compute_f1("the a an", "an the a") == 1.0


def test_metric_max_over_ground_truths_takes_the_best_alias() -> None:
    score = metric_max_over_ground_truths(compute_exact, "NYC", ["New York City", "NYC"])
    assert score == 1.0


def test_metric_max_over_ground_truths_empty_ground_truths_is_zero() -> None:
    assert metric_max_over_ground_truths(compute_exact, "anything", []) == 0.0


def test_score_qa_returns_both_em_and_f1() -> None:
    result = score_qa("Paris", ["Paris"])
    assert result == {"exact_match": 1.0, "f1": 1.0}
