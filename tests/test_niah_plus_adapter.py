"""Tests for the NIAH+ BenchmarkAdapter wrapper (no leakage, deterministic
generation of the frozen 12-condition grid, materialization, scoring).
Uses small monkeypatched context lengths so the full grid generates fast
without needing real 32K/128K-token network-backed instances -- the
generation logic itself (chunking/depth placement) is already covered by
tests/test_niah_plus.py; this file only covers the adapter's own wiring
(TaskExample shape, no-leakage, prepare_environment, score).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ant.agents.base import AgentResult
from ant.benchmarks import niah_plus_adapter as niah_plus_adapter_module
from ant.benchmarks.niah_plus_adapter import NiahPlusAdapter
from ant.evaluation_suite.usage import UsageStats


def _fake_squad_rows(n: int = 20) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"squad{i}",
                "title": f"Topic{i}",
                "context": f"Filler passage {i}. " * 10 + f"The answer entity is Person{i}.",
                "question": f"Who is in passage {i}?",
                "answers": {"text": [f"Person{i}"], "answer_start": [0]},
            }
        )
    return rows


def _fake_hotpot_examples(n: int = 10):
    from ant.benchmarks.base import TaskExample

    examples = []
    for i in range(n):
        documents = [
            {"doc_id": f"doc{j}", "title": f"T{i}_{j}", "text": f"Q{i} filler {j}. " * 10}
            for j in range(10)
        ]
        examples.append(
            TaskExample(
                benchmark="hotpotqa",
                task_id=f"hotpot{i}",
                question=f"Question {i}?",
                reference=json.dumps([f"Answer{i}"]),
                metadata={
                    "documents": documents,
                    "num_documents": 10,
                    "supporting_doc_ids": ["doc1", "doc4"],
                },
            )
        )
    return examples


@pytest.fixture(autouse=True)
def _small_grid_and_mocked_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    from ant.benchmarks.hotpotqa import HotpotQaAdapter

    monkeypatch.setattr(datasets, "load_dataset", lambda path, split: _fake_squad_rows())
    monkeypatch.setattr(
        HotpotQaAdapter, "load_examples", lambda self, limit=None: _fake_hotpot_examples()
    )
    # Small lengths so the 12-condition grid generates fast in tests --
    # only the adapter's own wiring is under test here, not the generator
    # internals (covered separately by test_niah_plus.py).
    monkeypatch.setattr(niah_plus_adapter_module, "CONTEXT_LENGTHS", (500, 1000))


def test_load_examples_generates_all_twelve_conditions() -> None:
    adapter = NiahPlusAdapter()
    examples = adapter.load_examples()
    assert len(examples) == 12
    keys = {
        (e.metadata["task_type"], e.metadata["context_length_tokens"], e.metadata["position"])
        for e in examples
    }
    assert len(keys) == 12  # every condition distinct


def test_load_examples_respects_limit() -> None:
    adapter = NiahPlusAdapter()
    examples = adapter.load_examples(limit=3)
    assert len(examples) == 3


def test_task_examples_carry_only_plain_document_records_no_construction_metadata_on_docs() -> (
    None
):
    adapter = NiahPlusAdapter()
    for example in adapter.load_examples(limit=4):
        for doc in example.metadata["documents"]:
            assert set(doc) == {"doc_id", "title", "text"}
        # Construction-only diagnostics live in a clearly separate,
        # never-read-by-agents key.
        assert "niah_metadata" in example.metadata
        assert "needle_doc_ids" in example.metadata["niah_metadata"]


def test_prepare_environment_materializes_and_is_idempotent(tmp_path: Path) -> None:
    import ant.benchmarks._hotpot_style as hotpot_style_module

    original_root = hotpot_style_module.DOCUMENT_ENV_ROOT
    hotpot_style_module.DOCUMENT_ENV_ROOT = tmp_path
    niah_plus_adapter_module.DOCUMENT_ENV_ROOT = tmp_path
    try:
        adapter = NiahPlusAdapter()
        example = adapter.load_examples(limit=1)[0]
        root1 = adapter.prepare_environment(example)
        files = list(root1.glob("*.txt"))
        assert len(files) == example.metadata["num_documents"]
        root2 = adapter.prepare_environment(example)
        assert root1 == root2
    finally:
        hotpot_style_module.DOCUMENT_ENV_ROOT = original_root
        niah_plus_adapter_module.DOCUMENT_ENV_ROOT = original_root


def test_score_uses_official_em_f1_and_records_condition_metadata() -> None:
    adapter = NiahPlusAdapter()
    example = adapter.load_examples(limit=1)[0]
    gold = json.loads(example.reference)
    result = AgentResult(
        benchmark="niah_plus",
        task_id=example.task_id,
        method="direct_document",
        final_answer=gold[0],
        usage=UsageStats(),
    )
    metric = adapter.score(example, result)
    assert metric.submetrics["exact_match"] == 1.0
    assert metric.metadata["scoring_method"] == "official_em_f1_reconstruction"
    assert metric.metadata["task_type"] == example.metadata["task_type"]
    assert metric.metadata["position"] == example.metadata["position"]
