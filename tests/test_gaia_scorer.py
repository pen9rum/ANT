"""Tests for the vendored OFFICIAL GAIA scorer and this suite's own
deterministic `FINAL ANSWER:` extraction.

Every case here uses synthetic answer/ground-truth pairs invented for
this repository. No real GAIA question or answer appears anywhere, and no
LLM is called.
"""

from __future__ import annotations

import hashlib

import pytest

from ant.evaluation_suite.gaia_scorer import (
    _VENDORED_SCORER_PATH,
    GAIA_SYSTEM_PROMPT,
    OFFICIAL_SCORER_SHA256,
    extract_final_answer,
    normalize_str,
    question_scorer,
    score_response,
)


def test_vendored_scorer_bytes_match_the_pinned_hash():
    """The whole fidelity claim rests on these bytes being upstream's. If
    someone lints/reformats the vendored file, this fails before any
    score is ever computed."""
    actual = hashlib.sha256(_VENDORED_SCORER_PATH.read_bytes()).hexdigest()
    assert actual == OFFICIAL_SCORER_SHA256


def test_official_system_prompt_carries_the_required_template():
    assert "FINAL ANSWER:" in GAIA_SYSTEM_PROMPT
    assert "comma separated list" in GAIA_SYSTEM_PROMPT


# --------------------------------------------------------------------
# Numeric answers
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_answer", "ground_truth"),
    [
        ("42", "42"),  # exact
        ("42.0", "42"),  # float/int equivalence
        ("1,234", "1234"),  # comma grouping stripped
        ("$1500", "1500"),  # currency unit stripped
        ("17%", "17"),  # percent unit stripped
        ("  89  ", "89"),  # surrounding whitespace
    ],
)
def test_numeric_answers_match_after_official_normalization(model_answer, ground_truth):
    assert question_scorer(model_answer, ground_truth) is True


@pytest.mark.parametrize(
    ("model_answer", "ground_truth"),
    [
        ("43", "42"),  # plainly wrong
        ("4.2", "42"),  # decimal point matters
        ("forty two", "42"),  # words are not parsed into numbers
    ],
)
def test_numeric_answers_that_must_not_match(model_answer, ground_truth):
    assert question_scorer(model_answer, ground_truth) is False


def test_unparseable_answer_to_a_numeric_question_is_false_not_an_error():
    """Upstream returns inf rather than raising -- a documented quirk we
    preserve. A crash here would abort a whole sweep on one bad answer."""
    assert question_scorer("not a number at all", "7") is False


# --------------------------------------------------------------------
# String answers
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_answer", "ground_truth"),
    [
        ("Paris", "paris"),  # case-insensitive
        ("  Paris  ", "Paris"),  # leading/trailing whitespace
        ("sea gull", "seagull"),  # ALL whitespace removed, not just edges
        ("St. Petersburg", "St Petersburg"),  # punctuation removed
        ("Saint-Tropez", "Saint Tropez"),  # hyphen vs space
    ],
)
def test_string_answers_match_after_official_normalization(model_answer, ground_truth):
    assert question_scorer(model_answer, ground_truth) is True


def test_string_answer_that_should_fail_to_match():
    """The required negative case: normalization is forgiving about form,
    never about content."""
    assert question_scorer("Lyon", "Paris") is False


def test_normalize_str_removes_punctuation_only_when_asked():
    assert normalize_str("A.B, C") == "abc"
    assert normalize_str("A.B, C", remove_punct=False) == "a.b,c"


# --------------------------------------------------------------------
# List answers
# --------------------------------------------------------------------


def test_comma_separated_list_matches_with_whitespace_differences():
    assert question_scorer("apples,bananas", "apples, bananas") is True


def test_semicolon_is_also_a_list_separator():
    assert question_scorer("a; b", "a; b") is True


def test_mixed_numeric_and_string_list_elements_normalize_per_element():
    assert question_scorer("2000, apples", "2000, apples") is True
    # Per-element numeric normalization still applies inside a list: the
    # currency symbol is stripped by the official normalizer.
    assert question_scorer("$2000, apples", "2000, apples") is True


def test_comma_grouped_number_inside_a_list_is_split_and_therefore_fails():
    """A sharp edge of the official scorer, asserted so it is a KNOWN
    property rather than a mystery in a results table.

    `"1,000, apples"` splits on every comma into three elements
    (`1` / `000` / ` apples`) against the ground truth's two, so the
    length check fails before any value is compared. This is precisely
    why GAIA's own system prompt instructs "don't use comma to write your
    number" -- format compliance is load-bearing, not cosmetic, and a
    method that ignores it loses points for formatting alone.
    """
    assert question_scorer("1,000, apples", "1000, apples") is False


def test_list_answers_are_order_sensitive():
    """Upstream compares element-wise via zip, so order matters. Asserted
    so the behaviour is a known property rather than a surprise when a
    results table looks worse than expected."""
    assert question_scorer("bananas, apples", "apples, bananas") is False


def test_list_length_mismatch_fails():
    assert question_scorer("apples", "apples, bananas") is False
    assert question_scorer("apples, bananas, cherries", "apples, bananas") is False


# --------------------------------------------------------------------
# FINAL ANSWER: extraction (ours, not official)
# --------------------------------------------------------------------


def test_extracts_answer_from_official_template():
    result = extract_final_answer("Let me think about it.\nFINAL ANSWER: Paris")
    assert result.answer == "Paris"
    assert result.template_found is True


def test_extraction_is_case_insensitive_and_whitespace_tolerant():
    assert extract_final_answer("Final  Answer :  Paris").answer == "Paris"


def test_last_template_occurrence_wins():
    """A model that quotes the instructions while reasoning and then
    answers must be read at its conclusion."""
    raw = "I must end with FINAL ANSWER: <x>.\nWork...\nFINAL ANSWER: Lyon"
    assert extract_final_answer(raw).answer == "Lyon"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("FINAL ANSWER: **Paris**", "Paris"),
        ("FINAL ANSWER: **Paris**.", "Paris"),  # emphasis then period
        ("FINAL ANSWER: **Paris.**", "Paris"),  # period then emphasis
        ("FINAL ANSWER: [42]", "42"),  # leftover template brackets
        ("FINAL ANSWER: Paris.", "Paris"),  # sentence period
    ],
)
def test_template_scaffolding_is_stripped(raw, expected):
    assert extract_final_answer(raw).answer == expected


def test_a_decimal_point_is_never_mistaken_for_sentence_punctuation():
    assert extract_final_answer("FINAL ANSWER: 3.14").answer == "3.14"
    assert extract_final_answer("FINAL ANSWER: 2.").answer == "2."


def test_only_the_template_line_is_taken():
    raw = "FINAL ANSWER: Paris\n\nI hope that helps!"
    assert extract_final_answer(raw).answer == "Paris"


def test_list_answers_survive_extraction_unmangled():
    """Internal commas must NOT be touched -- the official scorer owns
    list splitting, and pre-mangling them here would change verdicts."""
    assert extract_final_answer("FINAL ANSWER: apples, bananas, cherries").answer == (
        "apples, bananas, cherries"
    )


def test_missing_template_falls_back_to_the_whole_response_and_is_flagged():
    result = extract_final_answer("Paris")
    assert result.answer == "Paris"
    assert result.template_found is False


def test_none_response_is_handled():
    result = extract_final_answer(None)
    assert result.answer == ""
    assert result.template_found is False


# --------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------


def test_score_response_combines_extraction_and_official_scoring():
    correct, extraction = score_response("Reasoning here.\nFINAL ANSWER: 1,234", "1234")
    assert correct is True
    assert extraction.answer == "1,234"
    assert extraction.template_found is True


def test_score_response_reports_a_wrong_answer_as_wrong():
    correct, extraction = score_response("FINAL ANSWER: Lyon", "Paris")
    assert correct is False
    assert extraction.answer == "Lyon"
