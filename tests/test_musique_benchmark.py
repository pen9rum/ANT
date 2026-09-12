"""Tests for the MuSiQue benchmark adapter -- its own distinct schema
(flat `paragraphs` list with `is_supporting` flags, `answer_aliases`), not
shared with _hotpot_style.py (see musique.py's own module docstring and
docs/long_context_dataset_audit.md). No real network calls:
`datasets.load_dataset` is monkeypatched with a synthetic row set matching
the real schema field-for-field.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ant.agents.base import AgentResult
from ant.benchmarks import musique as musique_module
from ant.benchmarks.musique import MuSiQueAdapter
from ant.evaluation_suite.usage import UsageStats


def _fake_rows() -> list[dict]:
    return [
        {
            "id": "2hop__1_2",
            "question": "Who is the spouse of the performer?",
            "answer": "Jane Doe",
            "answer_aliases": ["Jane R. Doe"],
            "answerable": True,
            "paragraphs": [
                {
                    "idx": 0,
                    "title": "Performer Bio",
                    "paragraph_text": "The performer married Jane Doe.",
                    "is_supporting": True,
                },
                {
                    "idx": 1,
                    "title": "Unrelated",
                    "paragraph_text": "Filler content about weather.",
                    "is_supporting": False,
                },
                {
                    "idx": 2,
                    "title": "Another Support",
                    "paragraph_text": "More detail on the marriage.",
                    "is_supporting": True,
                },
            ],
        }
    ]


def _fake_load_dataset(path: str, config: str, split: str) -> list[dict]:
    return _fake_rows()


def test_load_examples_builds_documents_in_original_paragraph_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)

    adapter = MuSiQueAdapter()
    examples = adapter.load_examples()

    assert len(examples) == 1
    example = examples[0]
    assert example.task_id == "2hop__1_2"
    documents = example.metadata["documents"]
    assert [d["doc_id"] for d in documents] == ["doc0", "doc1", "doc2"]
    assert documents[0]["title"] == "Performer Bio"
    assert example.metadata["num_documents"] == 3


def test_load_examples_supporting_doc_ids_come_from_is_supporting_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)
    adapter = MuSiQueAdapter()
    example = adapter.load_examples()[0]
    assert example.metadata["supporting_doc_ids"] == ["doc0", "doc2"]


def test_load_examples_reference_includes_answer_and_all_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)
    adapter = MuSiQueAdapter()
    example = adapter.load_examples()[0]
    ground_truths = json.loads(example.reference)
    assert ground_truths == ["Jane Doe", "Jane R. Doe"]


def test_load_examples_respects_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    def _two_rows(path: str, config: str, split: str) -> list[dict]:
        return _fake_rows() * 2

    monkeypatch.setattr(datasets, "load_dataset", _two_rows)
    adapter = MuSiQueAdapter()
    assert len(adapter.load_examples(limit=1)) == 1


def test_prepare_environment_materializes_documents_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)
    monkeypatch.setattr(musique_module, "DOCUMENT_ENV_ROOT", tmp_path)

    adapter = MuSiQueAdapter()
    example = adapter.load_examples()[0]

    root1 = adapter.prepare_environment(example)
    files = sorted(p.name for p in root1.iterdir() if p.suffix == ".txt")
    assert files == ["doc_0000.txt", "doc_0001.txt", "doc_0002.txt"]

    root2 = adapter.prepare_environment(example)
    assert root1 == root2


def test_score_uses_max_over_answer_and_aliases() -> None:
    from ant.benchmarks.base import TaskExample

    example = TaskExample(
        benchmark="musique",
        task_id="t1",
        question="q",
        reference=json.dumps(["Jane Doe", "Jane R. Doe"]),
        metadata={},
    )
    # Prediction matches the ALIAS, not the primary answer -- score_qa must
    # take the max over all ground truths, not just the first.
    result = AgentResult(
        benchmark="musique",
        task_id="t1",
        method="direct_document",
        final_answer="Jane R. Doe",
        usage=UsageStats(),
    )
    adapter = MuSiQueAdapter()
    metric = adapter.score(example, result)
    assert metric.submetrics["exact_match"] == 1.0
    assert metric.metadata["scoring_method"] == "official_em_f1"
