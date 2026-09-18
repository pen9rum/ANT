"""Tests for the capability-covered GAIA subset and its frozen manifest.

The load-bearing tests here are the LEAKAGE ones. A subset chosen with
any sight of gold answers, annotator metadata, or model performance is
not a predeclared subset -- it is selection on the outcome, which is the
classic way an evaluation result becomes meaningless. Those guarantees
are asserted structurally (on signatures and dataclass fields), not by
reading the code and trusting it.

No network, no LLM, no real GAIA content.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from ant.evaluation_suite.gaia_scope import MODALITY_SUPPORT, Modality, SupportStatus
from ant.evaluation_suite.gaia_subset import (
    FROZEN_STATUS,
    MANIFEST_PATH,
    SELECTION_RULE_VERSION,
    ExclusionRecord,
    ManifestNotFrozenError,
    SubsetCandidate,
    build_manifest,
    candidates_from_examples,
    require_frozen_manifest,
    select_subset,
    supported_modalities,
    write_manifest,
)

SUPPORTED_FILES = [
    "notes.txt",
    "data.csv",
    "book.xlsx",
    "bundle.zip",
    "paper.pdf",
    "memo.docx",
    "deck.pptx",
    "structure.pdb",
    "meta.jsonld",
    "script.py",
]
UNSUPPORTED_FILES = ["photo.png", "scan.jpg", "clip.mp3", "movie.mov", "thing.qqq"]


def candidate(task_id: str, file_name: str | None, level: str = "1") -> SubsetCandidate:
    return SubsetCandidate(task_id=task_id, level=level, file_name=file_name)


# --------------------------------------------------------------------
# The selection rule
# --------------------------------------------------------------------


def test_a_task_with_no_attachment_is_always_retained():
    selection = select_subset([candidate("t1", None)])
    assert selection.retained_ids() == ["t1"]
    assert not selection.excluded


@pytest.mark.parametrize("file_name", SUPPORTED_FILES)
def test_every_supported_modality_is_retained(file_name):
    selection = select_subset([candidate("t1", file_name)])
    assert selection.retained_ids() == ["t1"], file_name


@pytest.mark.parametrize("file_name", UNSUPPORTED_FILES)
def test_every_unsupported_modality_is_excluded(file_name):
    selection = select_subset([candidate("t1", file_name)])
    assert not selection.retained
    assert len(selection.excluded) == 1


def test_pdf_docx_and_pptx_are_retained_after_this_pass():
    """These three moved from conditional/not-implemented to SUPPORTED,
    which is the whole reason the subset is bigger than §G projected."""
    selection = select_subset(
        [candidate("a", "paper.pdf"), candidate("b", "memo.docx"), candidate("c", "deck.pptx")]
    )
    assert selection.retained_ids() == ["a", "b", "c"]


def test_image_audio_and_video_remain_the_only_exclusion_reasons():
    candidates = [candidate(f"t{i}", name) for i, name in enumerate(UNSUPPORTED_FILES)]
    selection = select_subset(candidates)
    assert set(selection.exclusions_by_modality()) == {"image", "audio", "video", "unknown"}


def test_the_exclusion_ledger_records_a_reason_per_task():
    selection = select_subset([candidate("t1", "photo.png"), candidate("t2", "clip.mp3")])
    assert [record.task_id for record in selection.excluded] == ["t1", "t2"]
    for record in selection.excluded:
        assert record.reason
        assert record.file_name
        assert record.modality in ("image", "audio")


def test_selection_preserves_input_order_and_is_deterministic():
    candidates = [candidate(f"t{i}", name) for i, name in enumerate(SUPPORTED_FILES)]
    first, second = select_subset(candidates), select_subset(candidates)
    assert first == second
    assert first.retained_ids() == [f"t{i}" for i in range(len(SUPPORTED_FILES))]


def test_totals_add_up():
    candidates = [candidate(f"s{i}", n) for i, n in enumerate(SUPPORTED_FILES)] + [
        candidate(f"u{i}", n) for i, n in enumerate(UNSUPPORTED_FILES)
    ]
    selection = select_subset(candidates)
    assert selection.total == len(candidates)
    assert len(selection.retained) + len(selection.excluded) == len(candidates)


def test_supported_modalities_is_read_from_the_single_source_of_truth():
    declared = {
        modality.value
        for modality, status in MODALITY_SUPPORT.items()
        if status is SupportStatus.SUPPORTED
    }
    assert set(supported_modalities()) == declared
    assert Modality.IMAGE.value not in supported_modalities()


# --------------------------------------------------------------------
# REQUIRED: the rule cannot see gold, annotator metadata, or performance
# --------------------------------------------------------------------


def test_subset_candidate_has_no_field_for_gold_or_annotator_metadata():
    """Structural, mirroring `derive_territories`'s own signature test:
    there is no slot through which an answer could reach the selector,
    so selection CANNOT be contaminated even by a careless caller."""
    fields = set(SubsetCandidate.__dataclass_fields__)
    assert fields == {"task_id", "level", "file_name", "question_chars"}
    forbidden = {
        "reference",
        "answer",
        "final_answer",
        "gold",
        "annotator_metadata",
        "metadata",
        "steps",
        "tools",
        "score",
        "correct",
        "accuracy",
    }
    assert not (fields & forbidden)


def test_select_subset_takes_only_candidates():
    parameters = set(inspect.signature(select_subset).parameters)
    assert parameters == {"candidates"}


def test_the_partition_is_identical_when_every_level_is_blanked():
    """`level` is carried for REPORTING only. If it ever leaked into the
    include/exclude decision this test fails -- which is a stronger
    guarantee than a comment saying it does not."""
    with_levels = [
        candidate("a", "notes.txt", "1"),
        candidate("b", "photo.png", "3"),
        candidate("c", None, "2"),
    ]
    blanked = [
        SubsetCandidate(task_id=c.task_id, level="", file_name=c.file_name) for c in with_levels
    ]
    assert select_subset(with_levels).retained_ids() == select_subset(blanked).retained_ids()
    assert [r.task_id for r in select_subset(with_levels).excluded] == [
        r.task_id for r in select_subset(blanked).excluded
    ]


def test_the_partition_is_identical_when_question_length_varies():
    """Nothing about the question may influence selection either."""
    short = [SubsetCandidate("a", "1", "notes.txt", question_chars=5)]
    long = [SubsetCandidate("a", "1", "notes.txt", question_chars=5000)]
    assert select_subset(short).retained_ids() == select_subset(long).retained_ids()


def test_candidates_from_examples_drops_reference_and_audit_only():
    """The projection is where the leakage boundary is crossed in the
    SAFE direction: a TaskExample carries gold, a SubsetCandidate
    cannot."""
    from ant.benchmarks.base import TaskExample

    example = TaskExample(
        benchmark="gaia",
        task_id="t1",
        question="how many?",
        reference="SENTINEL_GOLD_ANSWER",
        metadata={
            "level": "2",
            "file_name": "data.csv",
            "_audit_only": {"annotator_metadata": {"Steps": "SENTINEL_STEPS"}},
        },
    )
    [projected] = candidates_from_examples([example])
    serialized = json.dumps(projected.__dict__)
    assert "SENTINEL_GOLD_ANSWER" not in serialized
    assert "SENTINEL_STEPS" not in serialized
    assert projected.task_id == "t1"
    assert projected.file_name == "data.csv"


# --------------------------------------------------------------------
# The manifest document
# --------------------------------------------------------------------


def test_a_frozen_manifest_carries_ids_counts_and_the_ledger():
    selection = select_subset(
        [candidate("keep1", "notes.txt"), candidate("keep2", None), candidate("drop", "photo.png")]
    )
    manifest = build_manifest(selection)
    assert manifest["status"] == FROZEN_STATUS
    assert manifest["task_ids"] == ["keep1", "keep2"]
    assert manifest["counts"]["validation_total"] == 3
    assert manifest["counts"]["retained"] == 2
    assert manifest["counts"]["excluded"] == 1
    assert manifest["counts"]["excluded_by_modality"] == {"image": 1}
    assert manifest["exclusion_ledger"][0]["task_id"] == "drop"
    assert manifest["selection_rule_version"] == SELECTION_RULE_VERSION


def test_a_manifest_with_no_selection_uses_null_not_zero():
    """"Not computable yet" and "computed, found none" must not share a
    representation -- a zero would read like a completed audit."""
    manifest = build_manifest(None, status="blocked_on_gated_access", blocked_reason="gate")
    assert manifest["counts"] is None
    assert manifest["task_ids"] is None
    assert manifest["exclusion_ledger"] is None
    assert manifest["blocked_reason"] == "gate"


def test_a_manifest_with_no_selection_cannot_claim_to_be_frozen():
    with pytest.raises(ValueError, match="cannot be marked frozen"):
        build_manifest(None)


def test_require_frozen_manifest_refuses_a_blocked_manifest():
    manifest = build_manifest(None, status="blocked_on_gated_access", blocked_reason="gate")
    with pytest.raises(ManifestNotFrozenError, match="blocked_on_gated_access"):
        require_frozen_manifest(manifest)


def test_require_frozen_manifest_refuses_an_empty_frozen_manifest():
    """An empty subset would produce a results file with N=0 that reads
    like a completed sweep."""
    manifest = build_manifest(select_subset([candidate("t", "photo.png")]))
    assert manifest["task_ids"] == []
    with pytest.raises(ManifestNotFrozenError, match="no task ids"):
        require_frozen_manifest(manifest)


def test_require_frozen_manifest_accepts_a_real_one():
    manifest = build_manifest(select_subset([candidate("t", "notes.txt")]))
    assert require_frozen_manifest(manifest) is manifest


def test_the_manifest_never_contains_dataset_content():
    """GAIA's gate forbids resharing. The manifest must be task IDs and
    counts only -- no question, answer or attachment byte."""
    selection = select_subset([candidate("t1", "notes.txt"), candidate("t2", "photo.png")])
    manifest = build_manifest(selection)
    # Only the DATA-carrying sections are checked. The prose fields name
    # GAIA's column headers on purpose -- `selection_inputs_forbidden`
    # exists precisely to say "the gold `Final answer` was not consulted"
    # -- so scanning the whole document for those strings would forbid
    # the manifest from documenting its own leakage boundary.
    data_sections = json.dumps(
        {
            key: manifest[key]
            for key in ("task_ids", "exclusion_ledger", "counts",
                        "retained_level_distribution", "retained_extension_distribution")
        }
    )
    for header in ("Question", "Final answer", "Annotator Metadata", "Steps"):
        assert header not in data_sections


def test_the_manifest_records_the_substrate_dependency_versions():
    manifest = build_manifest(select_subset([candidate("t", "paper.pdf")]))
    assert manifest["substrate_dependencies"]
    assert "pymupdf" in manifest["substrate_dependencies"]


def test_write_manifest_round_trips(tmp_path: Path):
    manifest = build_manifest(select_subset([candidate("t", "notes.txt")]))
    target = write_manifest(manifest, tmp_path / "manifest.json")
    assert json.loads(target.read_text(encoding="utf-8")) == manifest
    assert target.read_text(encoding="utf-8").endswith("\n")


# --------------------------------------------------------------------
# The committed manifest file
# --------------------------------------------------------------------


def test_the_committed_manifest_exists_and_is_valid_json():
    assert MANIFEST_PATH.is_file()
    json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_the_committed_manifest_declares_its_own_status_honestly():
    """It is currently NOT frozen (the HF gate is closed). If it is ever
    promoted to frozen with no ids, this fails."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["status"] == FROZEN_STATUS:
        require_frozen_manifest(manifest)
    else:
        assert manifest["blocked_reason"]
        assert manifest["task_ids"] is None
        with pytest.raises(ManifestNotFrozenError):
            require_frozen_manifest(manifest)


def test_the_committed_manifest_records_the_current_modality_policy():
    """If someone changes MODALITY_SUPPORT without re-freezing, the
    manifest and the code disagree and this catches it."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert set(manifest["supported_modalities"]) == set(supported_modalities())


def test_exclusion_record_serializes_every_field():
    record = ExclusionRecord("t", "1", "a.png", "image", "unsupported", "because")
    assert set(record.to_dict()) == {
        "task_id",
        "level",
        "file_name",
        "modality",
        "support_status",
        "reason",
    }


# --------------------------------------------------------------------
# The FROZEN manifest -- pins the real, live-computed numbers
#
# These assert the committed artefact, not the algorithm. If someone
# re-freezes and the numbers move, that is either a real dataset change
# or a rule change, and either way it must be a deliberate, visible edit
# rather than a silent drift under a results table that cites this file.
# --------------------------------------------------------------------


def _frozen() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_the_frozen_manifest_pins_the_live_validation_total():
    """N = 165, measured live. This resolves the discrepancy the pass-1
    report left open: the paper's prose says "166 annotated questions",
    the HF validation split actually ships 165 rows."""
    assert _frozen()["counts"]["validation_total"] == 165


def test_the_frozen_manifest_pins_the_retained_and_excluded_counts():
    counts = _frozen()["counts"]
    assert counts["retained"] == 152
    assert counts["excluded"] == 13
    assert counts["retained"] + counts["excluded"] == counts["validation_total"]
    assert len(_frozen()["task_ids"]) == 152
    assert len(_frozen()["exclusion_ledger"]) == 13


def test_every_exclusion_is_a_perception_modality():
    """The only reason any task is dropped must be a modality withheld
    from EVERY method equally. Anything else in this ledger would mean
    the subset had been selected on something other than capability."""
    assert _frozen()["counts"]["excluded_by_modality"] == {"image": 10, "audio": 3}
    for record in _frozen()["exclusion_ledger"]:
        assert record["modality"] in ("image", "audio")
        assert record["support_status"] == "unsupported"


def test_the_frozen_manifest_pins_the_retained_level_distribution():
    assert _frozen()["retained_level_distribution"] == {"1": 49, "2": 78, "3": 25}


def test_the_retained_subset_still_spans_every_supported_modality():
    """The point of adding PDF/DOCX/PPTX was to retain those tasks. If a
    future change silently dropped one, the headline count would barely
    move but the capability coverage claim would be false."""
    distribution = _frozen()["retained_extension_distribution"]
    for extension in (".xlsx", ".pdf", ".docx", ".pptx", ".csv", ".zip", ".txt", ".py"):
        assert distribution.get(extension), extension
    assert distribution[".pdf"] == 3
    assert distribution[".docx"] == 1
    assert distribution[".pptx"] == 1


def test_no_task_id_appears_in_both_the_subset_and_the_ledger():
    manifest = _frozen()
    assert not set(manifest["task_ids"]) & {r["task_id"] for r in manifest["exclusion_ledger"]}


def test_the_frozen_manifest_carries_no_question_or_answer_text():
    """GAIA's gate forbids resharing. Task IDs and counts only.

    Checked on the DATA-carrying sections: the prose fields deliberately
    name GAIA's column headers in order to document the leakage
    boundary, so scanning the whole document would forbid the manifest
    from describing itself."""
    manifest = _frozen()
    data = json.dumps(
        {k: manifest[k] for k in ("task_ids", "exclusion_ledger", "counts",
                                  "retained_level_distribution",
                                  "retained_extension_distribution")}
    )
    for header in ("Question", "Final answer", "Annotator Metadata", "Steps"):
        assert header not in data
    # Ledger file names are `<task_id>.<ext>`, which carries no content.
    for record in manifest["exclusion_ledger"]:
        assert record["file_name"].startswith(record["task_id"])
