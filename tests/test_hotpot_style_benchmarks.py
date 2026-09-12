"""Tests for HotpotQA/2WikiMultihopQA (identical schema, shared logic in
ant.benchmarks._hotpot_style -- see that module's own docstring and
docs/long_context_dataset_audit.md for why they share one implementation).
No real network calls: `datasets.load_dataset` is monkeypatched with a
synthetic in-memory row set matching the real schema field-for-field, the
same "mock the one external boundary" convention used throughout this
evaluation suite (e.g. tests/test_longagent.py's own module docstring).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ant.agents.base import AgentResult
from ant.benchmarks._hotpot_style import (
    load_hotpot_style_examples,
    prepare_hotpot_style_environment,
    score_hotpot_style,
)
from ant.evaluation_suite.usage import UsageStats


def _fake_rows() -> list[dict]:
    return [
        {
            "id": "task-1",
            "question": "Were Alpha and Beta from the same country?",
            "answer": "Yes",
            "context": {
                "title": ["Alpha Doc", "Beta Doc", "Gamma Doc"],
                "sentences": [
                    ["Alpha was born in France.", " Alpha is a director."],
                    ["Beta was born in France.", " Beta is an actor."],
                    ["Gamma is unrelated filler content."],
                ],
            },
            "supporting_facts": {"title": ["Alpha Doc", "Beta Doc"], "sent_id": [0, 0]},
            "type": "comparison",
            "level": "easy",
        },
        {
            "id": "task-2",
            "question": "Second question?",
            "answer": "Second answer",
            "context": {
                "title": ["Solo Doc"],
                "sentences": [["Solo content."]],
            },
            "supporting_facts": {"title": ["Solo Doc"], "sent_id": [0]},
            "type": "bridge",
            "level": "hard",
        },
    ]


class _FakeDataset(list):
    pass


def _fake_load_dataset(path: str, config: str, split: str) -> _FakeDataset:
    return _FakeDataset(_fake_rows())


def test_load_hotpot_style_examples_builds_documents_in_original_order_with_positional_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)

    examples = load_hotpot_style_examples(
        benchmark_name="hotpotqa", hf_path="x", hf_config="y", split="validation", limit=None
    )

    assert len(examples) == 2
    first = examples[0]
    assert first.task_id == "task-1"
    documents = first.metadata["documents"]
    assert [d["doc_id"] for d in documents] == ["doc0", "doc1", "doc2"]
    assert documents[0]["title"] == "Alpha Doc"
    assert documents[0]["text"] == "Alpha was born in France. Alpha is a director."
    assert first.metadata["num_documents"] == 3


def test_load_hotpot_style_examples_maps_supporting_titles_to_doc_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)
    examples = load_hotpot_style_examples(
        benchmark_name="hotpotqa", hf_path="x", hf_config="y", split="validation", limit=None
    )
    first = examples[0]
    # "Alpha Doc" -> doc0, "Beta Doc" -> doc1 (Gamma Doc, doc2, is NOT supporting).
    assert first.metadata["supporting_doc_ids"] == ["doc0", "doc1"]


def test_load_hotpot_style_examples_reference_never_exposes_supporting_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)
    examples = load_hotpot_style_examples(
        benchmark_name="hotpotqa", hf_path="x", hf_config="y", split="validation", limit=None
    )
    for example in examples:
        assert "supporting" not in example.reference.lower()
        for document in example.metadata["documents"]:
            # DocumentRecord shape only -- doc_id/title/text, nothing else.
            assert set(document) == {"doc_id", "title", "text"}


def test_load_hotpot_style_examples_respects_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)
    examples = load_hotpot_style_examples(
        benchmark_name="hotpotqa", hf_path="x", hf_config="y", split="validation", limit=1
    )
    assert len(examples) == 1


def test_prepare_hotpot_style_environment_materializes_documents_in_order_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import datasets

    from ant.benchmarks import _hotpot_style as hotpot_style_module

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(hotpot_style_module, "DOCUMENT_ENV_ROOT", tmp_path)

    examples = load_hotpot_style_examples(
        benchmark_name="hotpotqa", hf_path="x", hf_config="y", split="validation", limit=1
    )
    example = examples[0]

    root1 = prepare_hotpot_style_environment(example, benchmark_name="hotpotqa")
    files = sorted(p.name for p in root1.iterdir() if p.suffix == ".txt")
    assert files == ["doc_0000.txt", "doc_0001.txt", "doc_0002.txt"]
    assert "Alpha was born in France" in (root1 / "doc_0000.txt").read_text(encoding="utf-8")

    # Second call: idempotent (marker file present), no re-materialization
    # error, same root returned.
    root2 = prepare_hotpot_style_environment(example, benchmark_name="hotpotqa")
    assert root1 == root2


def test_score_hotpot_style_uses_official_em_f1_against_the_gold_answer() -> None:
    import json

    from ant.benchmarks.base import TaskExample

    example = TaskExample(
        benchmark="hotpotqa",
        task_id="t1",
        question="q",
        reference=json.dumps(["Paris"]),
        metadata={},
    )
    result = AgentResult(
        benchmark="hotpotqa",
        task_id="t1",
        method="direct_document",
        final_answer="Paris",
        usage=UsageStats(),
    )
    metric = score_hotpot_style(example, result, benchmark_name="hotpotqa")
    assert metric.submetrics["exact_match"] == 1.0
    assert metric.native_score == metric.submetrics["f1"]
    assert metric.metadata["scoring_method"] == "official_em_f1"
