"""Tests for the Needle-in-a-Haystack PLUS reconstruction
(ant.evaluation_suite.niah_plus). See docs/niah_plus_fidelity_audit.md for
why this is a reconstruction, not the released benchmark. No real network
calls: `datasets.load_dataset` and `HotpotQaAdapter.load_examples` are
monkeypatched with small synthetic pools, the same mock-the-external-
boundary convention used throughout this evaluation suite.
"""

from __future__ import annotations

import pytest

from ant.evaluation_suite.document_scope import DocumentRecord
from ant.evaluation_suite.niah_plus import (
    MAX_OTHER_ENTITIES_PER_INSTANCE,
    MULTI_NEEDLE_DEPTHS,
    SINGLE_NEEDLE_DEPTHS,
    _fill_to_target_length,
    _insert_needles_by_token_depth,
    _salient_shared_entities,
    build_fully_counterfactualized_single_needle_instance,
    build_multi_needle_instance,
    build_single_needle_instance,
)


def _doc(doc_id: str, n_tokens_words: int) -> DocumentRecord:
    # "word " repeated n times is comfortably >= n tokens under cl100k_base.
    return DocumentRecord(doc_id=doc_id, title="", text=("word " * n_tokens_words).strip())


def test_insert_single_needle_at_depth_zero_lands_at_the_very_start() -> None:
    filler = [_doc(f"f{i}", 10) for i in range(10)]
    needle = _doc("needle", 5)
    result = _insert_needles_by_token_depth(filler, [(0.0, needle)])
    assert result[0].doc_id == "needle"
    assert len(result) == len(filler) + 1


def test_insert_single_needle_at_depth_one_lands_at_the_very_end() -> None:
    filler = [_doc(f"f{i}", 10) for i in range(10)]
    needle = _doc("needle", 5)
    result = _insert_needles_by_token_depth(filler, [(1.0, needle)])
    assert result[-1].doc_id == "needle"


def test_insert_single_needle_at_depth_half_lands_near_the_middle() -> None:
    filler = [_doc(f"f{i}", 10) for i in range(10)]  # 100 tokens total
    needle = _doc("needle", 5)
    result = _insert_needles_by_token_depth(filler, [(0.5, needle)])
    index = result.index(next(d for d in result if d.doc_id == "needle"))
    assert 3 <= index <= 7  # roughly the middle of 10 equal-sized filler docs


def test_insert_two_needles_preserves_both_and_respects_relative_order() -> None:
    # Regression test for an index-drift bug: inserting a SECOND needle
    # must account for the list having already grown by one from the
    # FIRST needle's own insertion, or the second needle lands one
    # position too early.
    filler = [_doc(f"f{i}", 10) for i in range(10)]
    early_needle = _doc("early_needle", 5)
    late_needle = _doc("late_needle", 5)
    result = _insert_needles_by_token_depth(filler, [(0.2, early_needle), (0.8, late_needle)])

    assert len(result) == len(filler) + 2
    early_index = result.index(early_needle)
    late_index = result.index(late_needle)
    assert early_index < late_index
    # Both needles present exactly once, nothing lost or duplicated.
    ids = [d.doc_id for d in result]
    assert ids.count("early_needle") == 1
    assert ids.count("late_needle") == 1


def test_insert_needles_order_independent_of_input_list_order() -> None:
    filler = [_doc(f"f{i}", 10) for i in range(10)]
    a, b = _doc("a", 5), _doc("b", 5)
    result_ab = _insert_needles_by_token_depth(filler, [(0.2, a), (0.8, b)])
    result_ba = _insert_needles_by_token_depth(filler, [(0.8, b), (0.2, a)])
    assert [d.doc_id for d in result_ab] == [d.doc_id for d in result_ba]


def test_fill_to_target_length_reaches_at_least_the_target_using_whole_documents() -> None:
    import random

    pool = [_doc(f"p{i}", 20) for i in range(5)]
    filler = _fill_to_target_length(pool, target_tokens=100, rng=random.Random(1))
    total = sum(len(d.text.split()) for d in filler)
    assert total >= 100
    # Whole documents only -- every filler doc's text is byte-identical to
    # some pool document's text (never truncated).
    pool_texts = {d.text for d in pool}
    assert all(d.text in pool_texts for d in filler)


def test_fill_to_target_length_can_exceed_pool_size_by_reshuffling() -> None:
    import random

    pool = [_doc("only", 10)]
    filler = _fill_to_target_length(pool, target_tokens=55, rng=random.Random(1))
    assert len(filler) >= 6  # must reuse the single pool doc repeatedly


def _fake_squad_rows(n: int = 30) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"squad{i}",
                "title": f"Topic{i}",
                "context": f"This is filler passage number {i}. " * 20
                + f"The answer entity is Person{i}.",
                "question": f"Who is mentioned in passage {i}?",
                "answers": {"text": [f"Person{i}"], "answer_start": [0]},
            }
        )
    return rows


def test_build_single_needle_instance_replaces_the_answer_entity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda path, split: _fake_squad_rows())

    instance = build_single_needle_instance(
        context_length_tokens=2000, position="early", question_index=0
    )

    assert instance.task_type == "single_needle"
    needle_doc = next(
        d for d in instance.documents if d.doc_id in instance.metadata["needle_doc_ids"]
    )
    assert "Person0" not in needle_doc.text  # original answer entity replaced
    assert instance.gold_answers[0] in needle_doc.text  # fictional entity present instead


def test_build_single_needle_instance_no_construction_metadata_on_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda path, split: _fake_squad_rows())
    instance = build_single_needle_instance(
        context_length_tokens=1500, position="middle", question_index=1
    )
    for doc in instance.documents:
        assert set(doc.model_dump()) == {"doc_id", "title", "text"}


def test_build_single_needle_instance_context_length_within_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda path, split: _fake_squad_rows())
    target = 5000
    instance = build_single_needle_instance(
        context_length_tokens=target, position="late", question_index=2
    )
    actual = instance.metadata["actual_token_count"]
    assert actual >= target
    assert actual <= target * 1.5  # whole-document granularity tolerance


def test_build_single_needle_instance_same_material_across_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda path, split: _fake_squad_rows())
    early = build_single_needle_instance(
        context_length_tokens=2000, position="early", question_index=3
    )
    late = build_single_needle_instance(
        context_length_tokens=2000, position="late", question_index=3
    )

    assert early.question == late.question
    assert early.gold_answers == late.gold_answers
    assert {d.text for d in early.documents} == {d.text for d in late.documents}
    # But the needle's own position differs.
    early_index = next(
        i for i, d in enumerate(early.documents) if d.doc_id in early.metadata["needle_doc_ids"]
    )
    late_index = next(
        i for i, d in enumerate(late.documents) if d.doc_id in late.metadata["needle_doc_ids"]
    )
    assert early_index < late_index


def test_build_single_needle_instance_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda path, split: _fake_squad_rows())
    first = build_single_needle_instance(
        context_length_tokens=2000, position="middle", question_index=0
    )
    second = build_single_needle_instance(
        context_length_tokens=2000, position="middle", question_index=0
    )
    assert first.model_dump() == second.model_dump()


def _fake_squad_rows_with_shared_entity(n: int = 30) -> list[dict]:
    """Like _fake_squad_rows, but the QUESTION also repeats a capitalized
    entity ("Central Subject N") that appears in the context -- needed to
    exercise Condition C's "replace additional entities shared between
    question and context" behavior, which _fake_squad_rows' own bland
    question ("Who is mentioned in passage N?") never triggers.
    """
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"squad{i}",
                "title": f"Topic{i}",
                "context": f"Central Subject {i} visited the town. " * 10
                + f"The answer entity is Person{i}.",
                "question": f"Who did Central Subject {i} visit?",
                "answers": {"text": [f"Person{i}"], "answer_start": [0]},
            }
        )
    return rows


def test_salient_shared_entities_finds_phrases_in_both_question_and_context() -> None:
    entities = _salient_shared_entities(
        question="Who did Central Subject 0 visit?",
        context="Central Subject 0 visited the town.",
        exclude=set(),
    )
    assert "Central Subject" in entities or "Central" in entities


def test_salient_shared_entities_excludes_the_answer_and_drops_overlapping_shorter_matches() -> (
    None
):
    entities = _salient_shared_entities(
        question="Where did Virgin Mary and Mary go?",
        context="Virgin Mary and Mary went to Lourdes.",
        exclude={"Lourdes"},
    )
    # "Virgin Mary" (longer) selected; the separately-matched shorter "Mary"
    # is dropped since it's contained in the already-selected longer phrase.
    assert "Virgin Mary" in entities
    assert "Mary" not in entities
    assert "Lourdes" not in entities  # excluded explicitly


def test_build_fully_counterfactualized_replaces_the_answer_entity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    monkeypatch.setattr(
        datasets, "load_dataset", lambda path, split: _fake_squad_rows_with_shared_entity()
    )
    instance = build_fully_counterfactualized_single_needle_instance(
        context_length_tokens=2000, position="early", question_index=0
    )
    needle_doc = next(
        d for d in instance.documents if d.doc_id in instance.metadata["needle_doc_ids"]
    )
    assert "Person0" not in needle_doc.text
    assert "Person0" not in instance.question
    assert instance.gold_answers[0] in needle_doc.text


def test_build_fully_counterfactualized_replaces_additional_shared_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    monkeypatch.setattr(
        datasets, "load_dataset", lambda path, split: _fake_squad_rows_with_shared_entity()
    )
    instance = build_fully_counterfactualized_single_needle_instance(
        context_length_tokens=2000, position="early", question_index=0
    )
    substitutions = instance.metadata["entity_substitutions"]
    # More than just the answer entity was substituted -- "Central Subject
    # 0" (shared between question and context) must also have been caught.
    assert len(substitutions) >= 2
    assert "Central Subject 0" not in instance.question
    needle_doc = next(
        d for d in instance.documents if d.doc_id in instance.metadata["needle_doc_ids"]
    )
    assert "Central Subject 0" not in needle_doc.text


def test_build_fully_counterfactualized_question_and_context_stay_mutually_consistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    monkeypatch.setattr(
        datasets, "load_dataset", lambda path, split: _fake_squad_rows_with_shared_entity()
    )
    instance = build_fully_counterfactualized_single_needle_instance(
        context_length_tokens=2000, position="early", question_index=0
    )
    needle_doc = next(
        d for d in instance.documents if d.doc_id in instance.metadata["needle_doc_ids"]
    )
    for original, fictional in instance.metadata["entity_substitutions"].items():
        if original == instance.metadata["original_answer_replaced"]:
            continue
        # Every non-answer substitution's fictional replacement appears in
        # BOTH the question and the needle context -- they were replaced
        # consistently, not independently.
        if fictional in instance.question:
            assert fictional in needle_doc.text


def test_build_fully_counterfactualized_never_states_the_original_answer_or_a_negative_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    monkeypatch.setattr(
        datasets, "load_dataset", lambda path, split: _fake_squad_rows_with_shared_entity()
    )
    instance = build_fully_counterfactualized_single_needle_instance(
        context_length_tokens=2000, position="early", question_index=0
    )
    full_text = instance.question + " " + " ".join(d.text for d in instance.documents)
    assert instance.metadata["original_answer_replaced"] not in full_text
    assert "do not answer" not in full_text.lower()
    assert "not the answer" not in full_text.lower()


def test_build_fully_counterfactualized_respects_the_max_other_entities_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    monkeypatch.setattr(
        datasets, "load_dataset", lambda path, split: _fake_squad_rows_with_shared_entity()
    )
    instance = build_fully_counterfactualized_single_needle_instance(
        context_length_tokens=2000, position="early", question_index=0
    )
    # answer + at most MAX_OTHER_ENTITIES_PER_INSTANCE others.
    assert len(instance.metadata["entity_substitutions"]) <= 1 + MAX_OTHER_ENTITIES_PER_INSTANCE


def test_build_fully_counterfactualized_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    monkeypatch.setattr(
        datasets, "load_dataset", lambda path, split: _fake_squad_rows_with_shared_entity()
    )
    first = build_fully_counterfactualized_single_needle_instance(
        context_length_tokens=2000, position="middle", question_index=1
    )
    second = build_fully_counterfactualized_single_needle_instance(
        context_length_tokens=2000, position="middle", question_index=1
    )
    assert first.model_dump() == second.model_dump()


def test_build_fully_counterfactualized_same_material_across_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    monkeypatch.setattr(
        datasets, "load_dataset", lambda path, split: _fake_squad_rows_with_shared_entity()
    )
    early = build_fully_counterfactualized_single_needle_instance(
        context_length_tokens=2000, position="early", question_index=2
    )
    late = build_fully_counterfactualized_single_needle_instance(
        context_length_tokens=2000, position="late", question_index=2
    )
    assert early.question == late.question
    assert early.gold_answers == late.gold_answers
    assert {d.text for d in early.documents} == {d.text for d in late.documents}


def _fake_hotpot_examples(n: int = 10):
    import json

    from ant.benchmarks.base import TaskExample

    examples = []
    for i in range(n):
        documents = [
            {
                "doc_id": f"doc{j}",
                "title": f"T{i}_{j}",
                "text": f"Question {i} filler content {j}. " * 10,
            }
            for j in range(10)
        ]
        examples.append(
            TaskExample(
                benchmark="hotpotqa",
                task_id=f"hotpot{i}",
                question=f"Question number {i}?",
                reference=json.dumps([f"Answer{i}"]),
                metadata={
                    "documents": documents,
                    "num_documents": 10,
                    "supporting_doc_ids": ["doc1", "doc4"],
                },
            )
        )
    return examples


def test_build_multi_needle_instance_has_exactly_two_needles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ant.benchmarks.hotpotqa import HotpotQaAdapter

    monkeypatch.setattr(
        HotpotQaAdapter, "load_examples", lambda self, limit=None: _fake_hotpot_examples()
    )
    instance = build_multi_needle_instance(
        context_length_tokens=3000, position="early", question_index=0
    )
    assert instance.task_type == "multi_needle"
    assert len(instance.metadata["needle_doc_ids"]) == 2


def test_build_multi_needle_instance_matches_the_source_question_and_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ant.benchmarks.hotpotqa import HotpotQaAdapter

    monkeypatch.setattr(
        HotpotQaAdapter, "load_examples", lambda self, limit=None: _fake_hotpot_examples()
    )
    instance = build_multi_needle_instance(
        context_length_tokens=3000, position="middle", question_index=2
    )
    assert instance.question == "Question number 2?"
    assert instance.gold_answers == ["Answer2"]


def test_build_multi_needle_instance_same_material_across_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ant.benchmarks.hotpotqa import HotpotQaAdapter

    monkeypatch.setattr(
        HotpotQaAdapter, "load_examples", lambda self, limit=None: _fake_hotpot_examples()
    )
    early = build_multi_needle_instance(
        context_length_tokens=3000, position="early", question_index=1
    )
    late = build_multi_needle_instance(
        context_length_tokens=3000, position="late", question_index=1
    )

    assert early.question == late.question
    assert early.gold_answers == late.gold_answers
    assert {d.text for d in early.documents} == {d.text for d in late.documents}


def test_depth_mappings_use_only_paper_disclosed_values() -> None:
    assert SINGLE_NEEDLE_DEPTHS == {"early": 0.0, "middle": 0.5, "late": 1.0}
    assert MULTI_NEEDLE_DEPTHS == {
        "early": (0.0, 0.33),
        "middle": (0.33, 0.66),
        "late": (0.66, 1.0),
    }
