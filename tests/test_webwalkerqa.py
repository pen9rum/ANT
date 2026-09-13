"""Tests for the WebWalkerQA preparation infrastructure
(ant.evaluation_suite.webwalkerqa). No real network calls:
`datasets.load_dataset` is monkeypatched with a small synthetic pool
matching the dataset's own documented schema, the same
mock-the-external-boundary convention used throughout this suite (e.g.
test_niah_plus.py). This module never crawls URLs and never calls an LLM
regardless -- only load_webwalkerqa_records touches network at all (a
Hugging Face Hub dataset fetch), and that is exactly what is mocked here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ant.evaluation_suite.webwalkerqa.loader import (
    WebWalkerQaRecord,
    filter_by_difficulty,
    filter_by_domain,
    filter_by_hop_type,
    filter_english,
    inference_view,
    load_webwalkerqa_records,
    sample_deterministic,
)
from ant.evaluation_suite.webwalkerqa.manifest import build_prep_manifest, save_prep_manifest


def _fake_rows(n: int = 10) -> list[dict]:
    # Matches the REAL live dataset schema (verified by actually loading
    # it -- see loader.py's own schema-correction docstring note):
    # lowercase top-level keys, and info.lang/type/difficulty_level use
    # short codes ("en"/"zh", "single_source"/"multi_source"), not full
    # names.
    rows = []
    for i in range(n):
        rows.append(
            {
                "question": f"What is fact {i} on the site?",
                "answer": f"Fact{i}",
                "root_url": f"https://example{i}.com",
                "info": {
                    "type": "single_source" if i % 2 == 0 else "multi_source",
                    "domain": "conference" if i % 3 == 0 else "education",
                    "lang": "en" if i % 4 != 0 else "zh",
                    "difficulty_level": "easy" if i % 2 == 0 else "hard",
                    "source_website": [f"https://example{i}.com/page{j}" for j in range(2)],
                    "golden_path": [f"https://example{i}.com", f"https://example{i}.com/page0"],
                },
            }
        )
    return rows


def test_load_webwalkerqa_records_normalizes_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda path, split: _fake_rows())
    records = load_webwalkerqa_records(split="main")

    assert len(records) == 10
    for record in records:
        assert isinstance(record, WebWalkerQaRecord)
        assert record.example_id.startswith("webwalkerqa-")
        assert record.question
        assert record.gold_answer
        assert record.root_url
        assert record.language in {"en", "zh"}
        assert record.domain in {"conference", "education"}
        assert record.hop_type in {"single_source", "multi_source"}
        assert record.difficulty in {"easy", "hard"}
        assert len(record.source_websites) == 2
        assert len(record.golden_path) == 2


def test_load_webwalkerqa_records_respects_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda path, split: _fake_rows())
    records = load_webwalkerqa_records(split="main", limit=3)
    assert len(records) == 3


def test_example_id_is_stable_and_content_derived(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda path, split: _fake_rows())
    first = load_webwalkerqa_records(split="main")
    second = load_webwalkerqa_records(split="main")
    assert [r.example_id for r in first] == [r.example_id for r in second]
    # distinct rows must never collide
    assert len({r.example_id for r in first}) == len(first)


def test_filter_english_keeps_only_english_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda path, split: _fake_rows())
    records = load_webwalkerqa_records(split="main")
    english = filter_english(records)
    assert english
    assert all(r.language == "en" for r in english)
    assert len(english) < len(records)


def test_filter_by_domain_difficulty_and_hop_type(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda path, split: _fake_rows())
    records = load_webwalkerqa_records(split="main")

    conference_only = filter_by_domain(records, "Conference")  # case-insensitive
    assert conference_only and all(r.domain == "conference" for r in conference_only)

    hard_only = filter_by_difficulty(records, "Hard")
    assert hard_only and all(r.difficulty == "hard" for r in hard_only)

    multi_source_only = filter_by_hop_type(records, "multi_source")
    assert multi_source_only and all(r.hop_type == "multi_source" for r in multi_source_only)


def test_sample_deterministic_first_n_without_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda path, split: _fake_rows())
    records = load_webwalkerqa_records(split="main")
    sampled = sample_deterministic(records, 3)
    assert [r.example_id for r in sampled] == [r.example_id for r in records[:3]]


def test_sample_deterministic_seeded_is_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda path, split: _fake_rows())
    records = load_webwalkerqa_records(split="main")
    a = sample_deterministic(records, 4, seed=42)
    b = sample_deterministic(records, 4, seed=42)
    assert [r.example_id for r in a] == [r.example_id for r in b]


def test_inference_view_never_exposes_gold_or_path_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda path, split: _fake_rows())
    record = load_webwalkerqa_records(split="main")[0]
    view = inference_view(record)

    assert set(view.keys()) == {
        "example_id",
        "question",
        "root_url",
        "language",
        "domain",
        "hop_type",
        "difficulty",
    }
    assert "gold_answer" not in view
    assert "source_websites" not in view
    assert "golden_path" not in view
    # The gold answer's own string content must never appear anywhere in
    # the sanitized view's values either (not just as a missing key).
    assert record.gold_answer not in json_dump_values(view)
    # root_url legitimately appears in the inference view (it's the
    # navigation starting point, not secret) -- and golden_path naturally
    # starts there too, so only the REST of the path (beyond root_url) is
    # checked for leakage here.
    for path_url in record.golden_path:
        if path_url == record.root_url:
            continue
        assert path_url not in json_dump_values(view)


def json_dump_values(d: dict) -> str:
    return " ".join(str(v) for v in d.values())


def test_prep_manifest_freezes_filters_seed_and_selected_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda path, split: _fake_rows())
    records = load_webwalkerqa_records(split="main")
    english = filter_english(records)
    selected = sample_deterministic(english, 2, seed=7)

    manifest = build_prep_manifest(
        split="main",
        filters={"language": "English"},
        seed=7,
        total_rows_before_filter=len(records),
        selected_records=selected,
    )
    out_path = tmp_path / "manifest.json"
    save_prep_manifest(manifest, out_path)

    assert manifest.total_rows_before_filter == 10
    assert manifest.total_rows_after_filter == 2
    assert manifest.filters == {"language": "English"}
    assert manifest.seed == 7
    assert manifest.selected_example_ids == [r.example_id for r in selected]
    assert out_path.exists()
    assert manifest.dataset_split == "main"
