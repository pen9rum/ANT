import json
from pathlib import Path

from ant.evaluation.metrics import build_reference_idf, evaluate_answer
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


def test_f1_treats_punctuation_attached_words_as_the_same_token() -> None:
    # Regression test: a plain .split() on normalized text used to leave
    # trailing punctuation glued to words ("module," / "implemented."),
    # so "module" in the prediction and "module," in the reference never
    # matched even though they're the same word. Two full-sentence
    # paragraphs differing only in punctuation should score close to 1.0.
    score = evaluate_answer(
        prediction="Qibo does not implement a centralized module, but rendering works.",
        expected="Qibo does not implement a centralized module but rendering works",
        evidence_count=0,
        unresolved_need_count=0,
    )
    assert score.f1 == 1.0


def test_build_reference_idf_does_not_fragment_document_frequency_on_punctuation() -> None:
    # The same word appearing with different trailing punctuation across
    # references must count as one term for document-frequency purposes,
    # not several near-unique ones.
    idf = build_reference_idf(
        [
            "quantum circuit basics.",
            "quantum, gate decomposition",
            "quantum bloch sphere",
        ]
    )
    assert "quantum" in idf
    assert "quantum," not in idf
    assert "quantum." not in idf


def test_build_reference_idf_weights_a_rare_term_higher_than_a_common_one() -> None:
    # "quantum" appears in every reference in this corpus (generic domain
    # word for a quantum-computing question set); "bloch" appears in only
    # one. IDF should rank the rare one strictly higher.
    idf = build_reference_idf(
        [
            "quantum circuit simulation basics",
            "quantum gate decomposition details",
            "quantum bloch sphere visualization",
        ]
    )
    assert idf["bloch"] > idf["quantum"]


def test_build_reference_idf_of_empty_corpus_is_empty() -> None:
    assert build_reference_idf([]) == {}
    assert build_reference_idf(["", "", ""]) == {}


def test_idf_weighted_f1_rewards_overlap_on_rare_terms_over_common_ones() -> None:
    # Same 4-token overlap count in both cases, but the corpus makes
    # "quantum"/"circuit" common (every reference has them) and
    # "bloch"/"sphere" rare (only the target reference has them) --
    # a prediction repeating the rare pair should score a higher weighted
    # F1 than one repeating the common pair, even though plain (unweighted)
    # F1 would treat them identically.
    corpus = [
        "quantum circuit gate basics",
        "quantum circuit fusion details",
        "quantum circuit bloch sphere visualization",
    ]
    idf = build_reference_idf(corpus)
    target = corpus[2]

    common_overlap_score = evaluate_answer(
        prediction="quantum circuit unrelated words here",
        expected=target,
        evidence_count=0,
        unresolved_need_count=0,
        idf=idf,
    )
    rare_overlap_score = evaluate_answer(
        prediction="bloch sphere unrelated words here",
        expected=target,
        evidence_count=0,
        unresolved_need_count=0,
        idf=idf,
    )
    assert rare_overlap_score.f1 > common_overlap_score.f1


def test_idf_weighted_f1_handles_a_prediction_token_absent_from_the_corpus() -> None:
    # A token in the prediction that never appears in any reference
    # (out-of-vocabulary for the idf table) must not crash or silently
    # drop to zero weight -- it falls back to the corpus's own maximum
    # idf weight (see _f1_score's docstring: treated as maximally rare,
    # not ignored).
    idf = build_reference_idf(["quantum circuit basics"])
    score = evaluate_answer(
        prediction="quantum circuit zzzznotinvocab",
        expected="quantum circuit basics",
        evidence_count=0,
        unresolved_need_count=0,
        idf=idf,
    )
    assert score.f1 > 0.0


def test_evaluate_answer_without_idf_matches_prior_unweighted_behavior() -> None:
    # Backward compatibility: omitting idf (every existing caller before
    # this change) must reproduce the exact plain-token-overlap F1.
    score = evaluate_answer(
        prediction="the cat sat on the mat",
        expected="the cat sat on a mat",
        evidence_count=1,
        unresolved_need_count=0,
    )
    assert score.f1 == round(5 / 6, 6)


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
