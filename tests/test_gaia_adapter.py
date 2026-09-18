"""Tests for `GaiaAdapter` -- loading, environment preparation, scoring,
and above all GOLD-LEAKAGE PREVENTION.

The leakage tests are assertion-based rather than eyeball checks, and
they hunt for distinctive sentinel strings planted in the synthetic
fixtures' annotator metadata, so a regression that starts copying
annotator metadata onto an agent-visible surface fails loudly.

No network, no LLM, no real GAIA content.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ant.agents.base import AgentResult
from ant.benchmarks.gaia import GaiaAdapter
from ant.evaluation_suite.registry import get_benchmark
from ant.evaluation_suite.scoring import MetricResult

FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "manifests"
    / "gaia"
    / "synthetic_fixtures.json"
)

# Planted in the fixtures specifically so leakage has a fingerprint.
STEPS_SENTINEL = "SENTINEL_STEPS_"
TOOLS_SENTINEL = "SENTINEL_TOOLS_"


@pytest.fixture
def adapter(tmp_path: Path) -> GaiaAdapter:
    return GaiaAdapter(source="synthetic", data_root=tmp_path, fixtures_path=FIXTURES)


@pytest.fixture
def examples(adapter: GaiaAdapter):
    return adapter.load_examples()


# --------------------------------------------------------------------
# Registration and loading
# --------------------------------------------------------------------


def test_adapter_registers_itself_under_the_gaia_name():
    assert get_benchmark("gaia").name == "gaia"


def test_loads_all_synthetic_examples(examples):
    # 10 since the .docx and .pptx fixtures were added alongside
    # OFFICE_DOC support; the count is asserted rather than derived so a
    # fixture accidentally dropped from the spec fails loudly.
    assert len(examples) == 10
    assert all(example.benchmark == "gaia" for example in examples)


def test_every_supported_modality_has_a_synthetic_fixture(examples):
    """Each modality this substrate DECLARES supported must have an
    offline fixture, or its reader is only covered in theory."""
    modalities = {example.metadata["attachment_modality"] for example in examples}
    assert {"text", "tabular", "archive", "pdf", "office_doc"} <= modalities


def test_limit_is_honored(adapter: GaiaAdapter):
    assert len(adapter.load_examples(limit=3)) == 3


def test_repo_filter_is_reinterpreted_as_a_level_filter(adapter: GaiaAdapter):
    level_one = adapter.load_examples(repo_filter="1")
    assert level_one
    assert {example.metadata["level"] for example in level_one} == {"1"}


def test_resolved_source_is_recorded_on_every_example(adapter: GaiaAdapter, examples):
    """A results file must always be able to prove which corpus produced
    it -- synthetic data must never be mistakable for a real run."""
    assert adapter.resolved_source() == "synthetic"
    assert all(example.metadata["source"] == "synthetic" for example in examples)


def test_auto_source_falls_back_to_synthetic_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(GaiaAdapter, "_token_visible", staticmethod(lambda: False))
    adapter = GaiaAdapter(source="auto", data_root=tmp_path, fixtures_path=FIXTURES)
    adapter.load_examples(limit=1)
    assert adapter.resolved_source() == "synthetic"


def test_invalid_source_is_rejected():
    with pytest.raises(ValueError):
        GaiaAdapter(source="mirror")


# --------------------------------------------------------------------
# REQUIRED: the loader never exposes gold to agent-visible surfaces
# --------------------------------------------------------------------


def test_question_text_never_contains_the_gold_answer(examples):
    for example in examples:
        assert example.reference
        assert example.reference.lower() not in example.question.lower()


def test_annotator_metadata_never_appears_outside_audit_only(examples):
    """The strongest leakage assertion: serialize every agent-visible
    surface and prove the sentinels are absent from all of it."""
    for example in examples:
        visible = {key: value for key, value in example.metadata.items() if key != "_audit_only"}
        blob = json.dumps({"question": example.question, "task_id": example.task_id, **visible})
        assert STEPS_SENTINEL not in blob
        assert TOOLS_SENTINEL not in blob


def test_gold_answer_never_appears_in_agent_visible_metadata(examples):
    for example in examples:
        visible = {key: value for key, value in example.metadata.items() if key != "_audit_only"}
        blob = json.dumps(visible).lower()
        assert example.reference.lower() not in blob


def test_audit_only_key_does_hold_the_evaluator_fields(examples):
    """The other half of the contract: the data is preserved for offline
    audit, just quarantined -- not silently discarded."""
    for example in examples:
        audit = example.metadata["_audit_only"]
        assert audit["final_answer"] == example.reference
        assert STEPS_SENTINEL in audit["annotator_metadata"]["Steps"]


def test_agent_visible_metadata_keys_are_exactly_the_expected_allowlist(examples):
    """Pins the agent-visible surface so a newly added field has to be a
    deliberate decision rather than an accidental widening."""
    expected = {
        "level",
        "file_name",
        "has_attachment",
        "attachment_modality",
        "attachment_supported",
        "territories",
        "source",
        "_audit_only",
    }
    for example in examples:
        assert set(example.metadata) == expected


def test_territories_carry_no_gold(examples):
    for example in examples:
        blob = json.dumps(example.metadata["territories"])
        assert STEPS_SENTINEL not in blob
        assert TOOLS_SENTINEL not in blob


def test_environment_object_holds_no_question_or_answer(adapter: GaiaAdapter, examples):
    """An agent is handed a GaiaEnvironment; it must be impossible to
    read gold back off it."""
    example = examples[1]
    environment = adapter.environment_for(example)
    assert not hasattr(environment, "reference")
    assert not hasattr(environment, "question")
    assert vars(environment).keys() == {"task_root", "file_name"}


# --------------------------------------------------------------------
# REQUIRED: attachment paths resolve
# --------------------------------------------------------------------


def test_prepare_environment_materializes_the_attachment(adapter: GaiaAdapter, examples):
    example = next(e for e in examples if e.metadata["file_name"] == "quarterly_widgets.csv")
    root = adapter.prepare_environment(example)
    assert (root / "quarterly_widgets.csv").exists()


def test_prepare_environment_is_idempotent(adapter: GaiaAdapter, examples):
    example = next(e for e in examples if e.metadata["file_name"] == "inventory.xlsx")
    first = adapter.prepare_environment(example)
    before = (first / "inventory.xlsx").read_bytes()
    second = adapter.prepare_environment(example)
    assert first == second
    assert (second / "inventory.xlsx").read_bytes() == before


def test_tasks_get_isolated_directories(adapter: GaiaAdapter, examples):
    roots = {adapter.prepare_environment(example) for example in examples}
    assert len(roots) == len(examples)


def test_prepare_environment_works_for_a_task_without_an_attachment(adapter, examples):
    example = next(e for e in examples if not e.metadata["file_name"])
    root = adapter.prepare_environment(example)
    assert root.is_dir()


def test_environment_for_reads_a_materialized_attachment(adapter: GaiaAdapter, examples):
    example = next(e for e in examples if e.metadata["file_name"] == "field_notes.txt")
    assert "Redwood Bluff" in adapter.environment_for(example).read_text()


# --------------------------------------------------------------------
# Scoring (deterministic -- no judge, no cost)
# --------------------------------------------------------------------


def result_for(example, answer: str) -> AgentResult:
    return AgentResult(
        benchmark="gaia", task_id=example.task_id, method="stub", final_answer=answer
    )


def test_correct_answer_scores_one(adapter: GaiaAdapter, examples):
    example = examples[0]
    metric = adapter.score(example, result_for(example, f"FINAL ANSWER: {example.reference}"))
    assert isinstance(metric, MetricResult)
    assert metric.native_score == 1.0
    assert metric.normalized_score == 100.0


def test_wrong_answer_scores_zero(adapter: GaiaAdapter, examples):
    example = examples[0]
    metric = adapter.score(example, result_for(example, "FINAL ANSWER: definitely not it"))
    assert metric.native_score == 0.0


def test_scoring_makes_exactly_one_grader_run_and_costs_nothing(adapter, examples):
    example = examples[0]
    metric = adapter.score(example, result_for(example, "FINAL ANSWER: x"))
    assert len(metric.grader_runs) == 1
    assert metric.metadata["judge_cost_usd"] == 0.0
    assert metric.metadata["judge_model"] is None
    assert metric.metadata["n_judge_calls"] == 0


def test_scoring_records_the_pinned_official_scorer_identity(adapter, examples):
    example = examples[0]
    metric = adapter.score(example, result_for(example, "FINAL ANSWER: x"))
    assert "official GAIA question_scorer" in metric.metadata["scorer"]
    assert len(metric.metadata["scorer_sha256"]) == 64


def test_format_noncompliance_is_reported_separately_from_wrongness(adapter, examples):
    """A missing FINAL ANSWER: template is a FORMAT failure. The official
    pipeline cannot see the difference; this suite records it."""
    example = examples[0]
    compliant = adapter.score(example, result_for(example, f"FINAL ANSWER: {example.reference}"))
    bare = adapter.score(example, result_for(example, example.reference))
    assert compliant.metadata["final_answer_template_found"] is True
    assert bare.metadata["final_answer_template_found"] is False
    # Both are still scored on merit -- a format miss is not auto-zeroed.
    assert bare.native_score == 1.0


def test_scoring_is_deterministic_across_repeat_calls(adapter, examples):
    example = examples[0]
    answer = result_for(example, f"FINAL ANSWER: {example.reference}")
    first = adapter.score(example, answer)
    second = adapter.score(example, answer)
    assert first.native_score == second.native_score
    assert first.metadata["extracted_answer"] == second.metadata["extracted_answer"]


def test_scoring_reads_reference_and_never_audit_metadata(adapter, examples):
    """Removing _audit_only entirely must not change any score -- proving
    the scorer depends only on `reference`."""
    example = examples[0]
    answer = result_for(example, f"FINAL ANSWER: {example.reference}")
    baseline = adapter.score(example, answer).native_score
    example.metadata.pop("_audit_only")
    assert adapter.score(example, answer).native_score == baseline


def test_list_answer_task_scores_through_the_official_list_branch(adapter, examples):
    example = next(e for e in examples if "," in e.reference)
    metric = adapter.score(example, result_for(example, f"FINAL ANSWER: {example.reference}"))
    assert metric.native_score == 1.0
