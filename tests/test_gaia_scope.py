"""Tests for the GAIA substrate: attachment resolution, the declared
modality policy, deterministic parsers, and territory derivation.

All fixtures are hand-authored synthetic files built by
`ant.evaluation_suite.gaia_fixtures`. No real GAIA content, no network,
no LLM.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from ant.evaluation_suite.gaia_fixtures import build_fixture_attachment, materialize_all
from ant.evaluation_suite.gaia_scope import (
    AttachmentMissingError,
    GaiaEnvironment,
    Modality,
    NoAttachmentError,
    SupportStatus,
    TerritoryKind,
    UnsupportedModalityError,
    derive_territories,
    modality_for,
    support_status_for,
)

FIXTURES = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "manifests"
    / "gaia"
    / "synthetic_fixtures.json"
)


@pytest.fixture
def attachments(tmp_path: Path) -> Path:
    materialize_all(tmp_path, FIXTURES)
    return tmp_path


def env_for(root: Path, file_name: str | None) -> GaiaEnvironment:
    return GaiaEnvironment(root, file_name)


# --------------------------------------------------------------------
# Modality classification and support policy
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [
        ("a.txt", Modality.TEXT),
        ("a.json", Modality.TEXT),
        ("a.pdb", Modality.TEXT),
        ("a.py", Modality.TEXT),
        ("a.csv", Modality.TABULAR),
        ("a.xlsx", Modality.TABULAR),
        ("a.pdf", Modality.PDF),
        ("a.docx", Modality.OFFICE_DOC),
        ("a.png", Modality.IMAGE),
        ("a.JPG", Modality.IMAGE),  # case-insensitive
        ("a.mp3", Modality.AUDIO),
        ("a.mov", Modality.VIDEO),
        ("a.zip", Modality.ARCHIVE),
        ("a.qqq", Modality.UNKNOWN),
    ],
)
def test_modality_classification(file_name, expected):
    assert modality_for(file_name) is expected


def test_image_audio_and_video_are_declared_unsupported():
    """The fairness line: these would require a perception pipeline that
    cannot be offered identically to every method in this pass."""
    for modality in (Modality.IMAGE, Modality.AUDIO, Modality.VIDEO, Modality.UNKNOWN):
        assert support_status_for(modality) is SupportStatus.UNSUPPORTED


def test_text_tabular_and_archive_are_supported():
    for modality in (Modality.TEXT, Modality.TABULAR, Modality.ARCHIVE):
        assert support_status_for(modality) is SupportStatus.SUPPORTED


# --------------------------------------------------------------------
# REQUIRED: an unsupported modality fails EXPLICITLY
# --------------------------------------------------------------------


def test_image_attachment_raises_instead_of_degrading(attachments: Path):
    environment = env_for(attachments, "diagram.png")
    with pytest.raises(UnsupportedModalityError) as excinfo:
        environment.read_text()
    assert excinfo.value.modality is Modality.IMAGE
    assert excinfo.value.status is SupportStatus.UNSUPPORTED


def test_audio_attachment_raises_instead_of_degrading(attachments: Path):
    environment = env_for(attachments, "briefing.mp3")
    with pytest.raises(UnsupportedModalityError) as excinfo:
        environment.read_text()
    assert excinfo.value.modality is Modality.AUDIO


def test_unsupported_modality_never_returns_empty_or_partial_content(attachments: Path):
    """The failure mode this test guards against is the dangerous one: a
    substrate that returns "" for an image would look like a reasoning
    failure in the results table instead of a capability gap."""
    for file_name in ("diagram.png", "briefing.mp3"):
        environment = env_for(attachments, file_name)
        with pytest.raises(UnsupportedModalityError):
            environment.read_text()


def test_unsupported_error_message_names_the_modality_and_status(attachments: Path):
    environment = env_for(attachments, "diagram.png")
    with pytest.raises(UnsupportedModalityError, match="image"):
        environment.read_text()


def test_read_table_on_a_non_tabular_attachment_fails_explicitly(attachments: Path):
    environment = env_for(attachments, "field_notes.txt")
    with pytest.raises(UnsupportedModalityError):
        environment.read_table()


# --------------------------------------------------------------------
# REQUIRED: attachment paths resolve correctly
# --------------------------------------------------------------------


def test_attachment_path_resolves_inside_the_task_root(attachments: Path):
    environment = env_for(attachments, "field_notes.txt")
    resolved = environment.attachment_path()
    assert resolved.exists()
    assert resolved.parent == attachments.resolve()
    assert resolved.name == "field_notes.txt"


def test_missing_attachment_raises_a_typed_error(tmp_path: Path):
    environment = env_for(tmp_path, "absent.txt")
    with pytest.raises(AttachmentMissingError):
        environment.attachment_path()


def test_attachment_primitive_on_a_task_without_one_raises(tmp_path: Path):
    environment = env_for(tmp_path, None)
    assert environment.has_attachment() is False
    with pytest.raises(NoAttachmentError):
        environment.attachment_path()


def test_path_traversal_outside_the_task_root_is_refused(tmp_path: Path):
    outside = tmp_path / "secret.txt"
    outside.write_text("should never be readable", encoding="utf-8")
    task_root = tmp_path / "task"
    task_root.mkdir()
    environment = env_for(task_root, "../secret.txt")
    with pytest.raises(AttachmentMissingError, match="outside"):
        environment.attachment_path()


# --------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------


def test_csv_attachment_parses_into_rows(attachments: Path):
    view = env_for(attachments, "quarterly_widgets.csv").read_table()
    assert view.rows[0] == ["Quarter", "Units", "Region"]
    assert view.rows[1] == ["Q1", "300", "North"]
    assert len(view.rows) == 5


def test_xlsx_attachment_parses_with_stdlib_only(attachments: Path):
    """Exercises both cell encodings: shared-string text cells and raw
    numeric cells."""
    view = env_for(attachments, "inventory.xlsx").read_table()
    assert view.rows[0] == ["Item", "Stock"]
    assert view.rows[1] == ["apples", "52"]
    assert view.rows[4] == ["dates", "8"]


def test_xlsx_row_limit_is_honored_and_flagged(attachments: Path):
    view = env_for(attachments, "inventory.xlsx").read_table(max_rows=2)
    assert len(view.rows) == 2
    assert view.truncated_rows is True


def test_text_attachment_reads_through(attachments: Path):
    text = env_for(attachments, "field_notes.txt").read_text()
    assert "Redwood Bluff" in text


def test_archive_listing_is_sorted_and_excludes_directories(attachments: Path):
    members = env_for(attachments, "bundle.zip").list_archive()
    assert members == ["data/values.txt", "readme.txt"]


def test_tabular_read_text_redirects_to_the_table_reader(attachments: Path):
    """A .xlsx read as raw text would be ZIP mojibake; the substrate
    redirects rather than handing back garbage."""
    text = env_for(attachments, "inventory.xlsx").read_text()
    assert "apples" in text


def test_fixture_builder_is_byte_deterministic(tmp_path: Path):
    """Tool-call-log determinism downstream is meaningless if the fixture
    inputs themselves drift between builds."""
    first = build_fixture_attachment("inventory.xlsx", tmp_path / "a.xlsx", FIXTURES)
    second = build_fixture_attachment("inventory.xlsx", tmp_path / "b.xlsx", FIXTURES)
    assert first.read_bytes() == second.read_bytes()


# --------------------------------------------------------------------
# Territories
# --------------------------------------------------------------------


def test_every_task_gets_web_and_computation_territories():
    kinds = {t.kind for t in derive_territories(file_name=None)}
    assert kinds == {TerritoryKind.WEB, TerritoryKind.COMPUTATION}


def test_a_tabular_attachment_adds_attachment_and_tabular_territories():
    kinds = {t.kind for t in derive_territories(file_name="data.xlsx")}
    assert TerritoryKind.ATTACHMENT in kinds
    assert TerritoryKind.TABULAR in kinds


def test_a_text_attachment_adds_a_document_territory():
    kinds = {t.kind for t in derive_territories(file_name="notes.txt")}
    assert TerritoryKind.DOCUMENT in kinds


def test_an_unsupported_attachment_territory_is_marked_unsupported():
    """A capability gap must be visible to a routing layer as a fact,
    not discovered as an exception several rounds later."""
    territories = derive_territories(file_name="photo.png")
    attachment = next(t for t in territories if t.kind is TerritoryKind.ATTACHMENT)
    assert attachment.supported is False


def test_derive_territories_has_no_parameter_that_could_carry_gold():
    """Structural leakage guarantee, mirroring
    `web_scope.build_territory_from_discovered`'s own signature test:
    there is no slot through which an answer or annotator metadata could
    reach territory derivation, so routing CANNOT see gold even by a
    careless caller."""
    parameters = set(inspect.signature(derive_territories).parameters)
    assert parameters == {"file_name", "question_length"}
    forbidden = {"answer", "final_answer", "reference", "annotator_metadata", "metadata", "gold"}
    assert not (parameters & forbidden)


def test_gaia_environment_has_no_slot_for_question_or_answer():
    """The environment is built from a directory and a filename only."""
    parameters = set(inspect.signature(GaiaEnvironment.__init__).parameters)
    assert parameters == {"self", "task_root", "file_name"}
