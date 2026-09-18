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
    MODALITY_DEPENDENCIES,
    AttachmentMissingError,
    GaiaEnvironment,
    MissingSubstrateDependencyError,
    Modality,
    NoAttachmentError,
    SupportStatus,
    TerritoryKind,
    UnsupportedModalityError,
    check_substrate_dependencies,
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


def test_text_tabular_archive_pdf_and_office_doc_are_supported():
    for modality in (
        Modality.TEXT,
        Modality.TABULAR,
        Modality.ARCHIVE,
        Modality.PDF,
        Modality.OFFICE_DOC,
    ):
        assert support_status_for(modality) is SupportStatus.SUPPORTED


def test_support_status_is_a_property_of_the_code_not_of_the_environment(monkeypatch):
    """PDF support used to be resolved by ATTEMPTING an import, which
    made the declared capability set vary by machine and therefore made a
    frozen subset non-reproducible. Pin the new contract: hiding the
    dependency changes nothing about what is DECLARED supported."""
    import builtins

    real_import = builtins.__import__

    def refuse_pymupdf(name, *args, **kwargs):
        if name == "pymupdf":
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_pymupdf)
    assert support_status_for(Modality.PDF) is SupportStatus.SUPPORTED


def test_a_missing_pinned_dependency_is_a_provisioning_error_not_a_capability_gap(
    attachments: Path, monkeypatch
):
    """`MissingSubstrateDependencyError` must NOT be an
    `UnsupportedModalityError`. Conflating them would let a forgotten
    `pip install` show up in a results table as a principled coverage
    ceiling."""
    import builtins

    real_import = builtins.__import__

    def refuse_pymupdf(name, *args, **kwargs):
        if name == "pymupdf":
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_pymupdf)
    with pytest.raises(MissingSubstrateDependencyError) as excinfo:
        env_for(attachments, "report.pdf").read_text()
    assert not isinstance(excinfo.value, UnsupportedModalityError)
    assert "pymupdf" in str(excinfo.value)
    with pytest.raises(MissingSubstrateDependencyError):
        check_substrate_dependencies()


def test_check_substrate_dependencies_reports_resolved_versions():
    found = check_substrate_dependencies()
    assert "pymupdf" in found
    assert found["pymupdf"]


def test_only_pdf_needs_a_third_party_reader():
    """.xlsx/.csv/.docx/.pptx are stdlib-only by design, so the declared
    dependency map must stay a single entry -- if it grows, the `gaia`
    extra in pyproject.toml has silently drifted."""
    assert set(MODALITY_DEPENDENCIES) == {Modality.PDF}


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


# --------------------------------------------------------------------
# PDF / DOCX / PPTX -- the modalities promoted from conditional/not
# implemented to SUPPORTED in this pass
# --------------------------------------------------------------------


def test_pdf_attachment_extracts_real_text(attachments: Path):
    text = env_for(attachments, "report.pdf").read_text()
    assert "Quorvian Survey Report" in text


def test_pdf_pages_are_separated_by_a_form_feed(tmp_path: Path):
    """GAIA PDF questions are frequently "on page N ...", so the page
    boundary is signal and must survive extraction."""
    import pymupdf

    path = tmp_path / "two_pages.pdf"
    document = pymupdf.open()
    for body in ("ALPHA PAGE", "BETA PAGE"):
        page = document.new_page()
        page.insert_text((72, 720), body)
    document.save(path)
    document.close()

    text = env_for(tmp_path, "two_pages.pdf").read_text()
    assert "\f" in text
    assert text.split("\f")[0].strip().startswith("ALPHA")
    assert text.split("\f")[1].strip().startswith("BETA")


def test_pdf_is_read_through_the_modern_import_name(attachments: Path, monkeypatch):
    """`import fitz` is deprecated upstream and slated for removal. If
    this substrate ever falls back to it, this test fails rather than a
    future PyMuPDF bump failing a whole sweep."""
    import builtins

    real_import = builtins.__import__

    def refuse_fitz(name, *args, **kwargs):
        if name == "fitz":
            raise AssertionError("gaia_scope must import `pymupdf`, not the legacy `fitz` alias")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse_fitz)
    assert env_for(attachments, "report.pdf").read_text()


def test_docx_attachment_extracts_paragraphs_in_order(attachments: Path):
    text = env_for(attachments, "charter.docx").read_text()
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines[0].strip() == "Quorvian Cartographers Guild Charter"
    assert "Article One" in lines[1]
    assert "Article Two" in lines[2]


def test_docx_reassembles_text_split_across_runs(attachments: Path):
    """Real Word files fragment a sentence across `<w:r>` runs at every
    formatting change; the fixture does the same. A reader that took
    only the first `<w:t>` per paragraph would truncate silently."""
    text = env_for(attachments, "charter.docx").read_text()
    assert "the Guild shall survey every province of Quorvia once each decade" in text


def test_docx_includes_table_cell_text(attachments: Path):
    assert "Survey cadence: 10 years" in env_for(attachments, "charter.docx").read_text()


def test_docx_reading_needs_no_third_party_library(attachments: Path, monkeypatch):
    """Same stdlib-only bar `_read_xlsx` already meets: `python-docx`
    must never become a hidden requirement."""
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name in ("docx", "pptx", "lxml"):
            raise AssertionError(f"docx/pptx reading must be stdlib-only, imported {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert env_for(attachments, "charter.docx").read_text()
    assert env_for(attachments, "roadmap.pptx").read_text()


def test_pptx_slides_are_ordered_numerically_not_lexicographically(attachments: Path):
    """The bug this pins: `slide10.xml` sorts BEFORE `slide9.xml` as a
    string. "What is on the third slide" is a real GAIA question shape,
    so a string sort silently renumbers the deck."""
    text = env_for(attachments, "roadmap.pptx").read_text()
    order = [line for line in text.splitlines() if line.startswith("--- slide ")]
    numbers = [int(line.split()[2]) for line in order]
    assert numbers == sorted(numbers)
    assert numbers == list(range(1, len(numbers) + 1))
    assert text.index("--- slide 9 ---") < text.index("--- slide 10 ---")


def test_pptx_includes_speaker_notes(attachments: Path):
    """Notes live in `ppt/notesSlides/notesSlideN.xml` -- note the
    CAPITAL S, which a case-sensitive part matcher misses entirely."""
    text = env_for(attachments, "roadmap.pptx").read_text()
    assert "[notes] Opening slide; keep to thirty seconds." in text
    assert "[notes] Closing slide; the sealing ceremony is in Vandermeer." in text


def test_pptx_notes_attach_to_the_right_slide(attachments: Path):
    text = env_for(attachments, "roadmap.pptx").read_text()
    first_block = text.split("--- slide 2 ---")[0]
    assert "Opening slide" in first_block
    assert "Closing slide" not in first_block


def test_legacy_binary_office_extensions_are_not_claimed_as_supported():
    """.doc/.ppt are OLE compound files, not OOXML. They are deliberately
    absent from `EXTENSION_MODALITY`, so they classify UNKNOWN and are
    refused -- rather than being mapped to OFFICE_DOC and then failing
    deep inside a ZIP reader, which would also wrongly retain them in a
    capability-covered subset."""
    for file_name in ("old.doc", "old.ppt"):
        assert modality_for(file_name) is Modality.UNKNOWN
        assert support_status_for(Modality.UNKNOWN) is SupportStatus.UNSUPPORTED


def test_an_office_doc_with_an_unhandled_suffix_fails_explicitly(tmp_path: Path, monkeypatch):
    """Defence in depth for the case above: if a future edit DID map
    `.doc` to OFFICE_DOC, the reader must still refuse it by name rather
    than handing an OLE file to a ZIP parser."""
    from ant.evaluation_suite import gaia_scope

    monkeypatch.setitem(gaia_scope.EXTENSION_MODALITY, ".doc", Modality.OFFICE_DOC)
    (tmp_path / "old.doc").write_bytes(b"\xd0\xcf\x11\xe0not really an OLE file")
    with pytest.raises(UnsupportedModalityError, match="legacy OLE"):
        env_for(tmp_path, "old.doc").read_text()


def test_a_docx_without_its_document_part_fails_explicitly(tmp_path: Path):
    import zipfile

    path = tmp_path / "broken.docx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("not/the/right/part.xml", "<a/>")
    with pytest.raises(UnsupportedModalityError, match="not a .docx"):
        env_for(tmp_path, "broken.docx").read_text()


def test_a_pptx_without_slide_parts_fails_explicitly(tmp_path: Path):
    import zipfile

    path = tmp_path / "broken.pptx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ppt/presentation.xml", "<a/>")
    with pytest.raises(UnsupportedModalityError, match="not a .pptx"):
        env_for(tmp_path, "broken.pptx").read_text()


def test_office_and_pdf_attachments_get_a_supported_document_territory():
    for file_name in ("paper.pdf", "notes.docx", "deck.pptx"):
        territories = derive_territories(file_name=file_name)
        document = next(t for t in territories if t.kind is TerritoryKind.DOCUMENT)
        assert document.supported is True


def test_image_audio_and_video_remain_unsupported_after_this_pass(attachments: Path):
    """Explicitly pinned: extending the substrate to PDF/DOCX/PPTX must
    NOT have loosened the vision/ASR fairness line anywhere."""
    for modality in (Modality.IMAGE, Modality.AUDIO, Modality.VIDEO):
        assert support_status_for(modality) is SupportStatus.UNSUPPORTED
    for file_name in ("diagram.png", "briefing.mp3"):
        with pytest.raises(UnsupportedModalityError):
            env_for(attachments, file_name).read_text()


def test_fixture_builders_are_byte_deterministic(tmp_path: Path):
    """The docx/pptx writers join the existing determinism guarantee --
    otherwise the byte-stable tool-call log assertions elsewhere would
    be resting on drifting inputs."""
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    materialize_all(first, FIXTURES)
    materialize_all(second, FIXTURES)
    for name in ("charter.docx", "roadmap.pptx"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
