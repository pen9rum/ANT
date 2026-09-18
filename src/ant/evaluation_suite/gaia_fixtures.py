"""Deterministic builder for the GAIA synthetic fixture attachments.

WHY A BUILDER INSTEAD OF COMMITTED BINARIES: the fixture spec
(`third_party/manifests/gaia/synthetic_fixtures.json`) is human-readable
and reviewable; a committed `.xlsx`/`.png`/`.zip` is an opaque blob that
nobody can diff and that quietly bloats the repository. Generating them
from the spec keeps every fixture byte accounted for in plain text, and
makes "what exactly is in the test spreadsheet?" answerable by reading
one JSON file.

Every writer here is byte-deterministic -- fixed ZIP timestamps, no
compression nondeterminism, no randomness -- so a fixture built today
and one built next month are identical. That matters because
`tests/test_gaia_tools.py` asserts byte-stable tool-call logs, which
would be meaningless if the inputs themselves drifted.

Contains NO GAIA material; see the spec file's own `_README`.
"""

from __future__ import annotations

import json
import struct
import zipfile
import zlib
from pathlib import Path
from typing import Any

# Fixed timestamp for every ZIP member (1980-01-01, the ZIP epoch) so
# archive bytes never depend on when the fixture was built.
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

_SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_PRESENTATION_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


def load_fixture_spec(fixtures_path: Path) -> dict[str, Any]:
    return json.loads(Path(fixtures_path).read_text(encoding="utf-8"))


def build_fixture_attachment(file_name: str, destination: Path, fixtures_path: Path) -> Path:
    """Writes the fixture attachment named `file_name` to `destination`.

    Raises `KeyError` for an unknown name rather than writing a
    placeholder -- a test that silently got an empty file instead of the
    spreadsheet it asked for would pass for the wrong reason.
    """
    spec = load_fixture_spec(fixtures_path)
    attachments = spec.get("attachments", {})
    if file_name not in attachments:
        raise KeyError(
            f"No synthetic fixture attachment named {file_name!r}. "
            f"Known: {sorted(attachments)}"
        )
    entry = attachments[file_name]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    kind = entry.get("kind")
    writer = _WRITERS.get(kind)
    if writer is None:
        raise KeyError(f"Unknown fixture attachment kind {kind!r} for {file_name!r}.")
    writer(entry, destination)
    return destination


def materialize_all(destination_dir: Path, fixtures_path: Path) -> list[Path]:
    """Builds every fixture attachment into one directory. Convenience for
    tests that want the whole corpus present at once."""
    spec = load_fixture_spec(fixtures_path)
    return [
        build_fixture_attachment(name, Path(destination_dir) / name, fixtures_path)
        for name in sorted(spec.get("attachments", {}))
    ]


def _write_csv(entry: dict[str, Any], destination: Path) -> None:
    lines = [",".join(str(cell) for cell in row) for row in entry["rows"]]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_text(entry: dict[str, Any], destination: Path) -> None:
    destination.write_text(entry["text"], encoding="utf-8")


def _write_binary_stub(entry: dict[str, Any], destination: Path) -> None:
    """Deliberately NOT a valid file of its extension. These fixtures back
    modalities this substrate refuses to parse, so the bytes are never
    read -- writing a real MP3 would imply a decoding capability that does
    not exist and is not claimed."""
    del entry
    destination.write_bytes(b"SYNTHETIC-FIXTURE-NOT-REAL-MEDIA\n")


def _write_png(entry: dict[str, Any], destination: Path) -> None:
    """A genuinely valid 1x1 greyscale PNG, assembled chunk by chunk.

    Real rather than stubbed because an image fixture should be
    well-formed enough that the substrate's refusal to read it is
    demonstrably a POLICY decision, not an accidental parse failure on
    corrupt bytes.
    """
    del entry

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    destination.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00\x00"))
        + chunk(b"IEND", b"")
    )


def _write_pdf(entry: dict[str, Any], destination: Path) -> None:
    """A minimal but structurally valid single-page PDF containing the
    spec's text, built by hand with correct xref offsets so a real PDF
    reader can open it."""
    text = entry.get("text", "")
    escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1", "replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode()
    destination.write_bytes(bytes(out))


def _write_zip(entry: dict[str, Any], destination: Path) -> None:
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for member_name in sorted(entry["members"]):
            info = zipfile.ZipInfo(member_name, date_time=_ZIP_EPOCH)
            archive.writestr(info, entry["members"][member_name])


def _write_xlsx(entry: dict[str, Any], destination: Path) -> None:
    """A minimal OOXML workbook.

    Text cells go through the shared-strings table and numeric-looking
    cells are written as raw numbers -- i.e. this fixture deliberately
    exercises BOTH branches of `gaia_scope._cell_value`, so the reader is
    tested against the same two cell encodings real spreadsheets use
    rather than only the easy one.
    """
    rows: list[list[str]] = [[str(cell) for cell in row] for row in entry["rows"]]

    shared: list[str] = []
    shared_index: dict[str, int] = {}
    for row in rows:
        for cell in row:
            if not _looks_numeric(cell) and cell not in shared_index:
                shared_index[cell] = len(shared)
                shared.append(cell)

    cells_xml: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        parts = []
        for column_number, cell in enumerate(row):
            ref = f"{_column_letters(column_number)}{row_number}"
            if _looks_numeric(cell):
                parts.append(f'<c r="{ref}"><v>{cell}</v></c>')
            else:
                parts.append(f'<c r="{ref}" t="s"><v>{shared_index[cell]}</v></c>')
        cells_xml.append(f'<row r="{row_number}">{"".join(parts)}</row>')

    sheet = (
        f'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="{_SHEET_NS}">'
        f'<sheetData>{"".join(cells_xml)}</sheetData></worksheet>'
    )
    shared_xml = (
        f'<?xml version="1.0" encoding="UTF-8"?><sst xmlns="{_SHEET_NS}" '
        f'count="{len(shared)}" uniqueCount="{len(shared)}">'
        + "".join(f"<si><t>{_escape(value)}</t></si>" for value in shared)
        + "</sst>"
    )
    workbook = (
        f'<?xml version="1.0" encoding="UTF-8"?><workbook xmlns="{_SHEET_NS}">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>'
        "</sheets></workbook>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-'
        'officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.'
        'openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.'
        'openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/></Types>'
    )
    members = {
        "[Content_Types].xml": content_types,
        "xl/workbook.xml": workbook,
        "xl/sharedStrings.xml": shared_xml,
        "xl/worksheets/sheet1.xml": sheet,
    }
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(members):
            archive.writestr(zipfile.ZipInfo(name, date_time=_ZIP_EPOCH), members[name])


def _write_docx(entry: dict[str, Any], destination: Path) -> None:
    """A minimal WordprocessingML document.

    Deliberately exercises MORE than the easy path, mirroring
    `_write_xlsx`'s reasoning: each spec paragraph is split into several
    `<w:r>` RUNS rather than one, because real Word files fragment a
    sentence across runs at every formatting change and a reader that
    only handled one-run paragraphs would pass a naive fixture and then
    silently truncate real text. A `<w:tab>` and a `<w:br>` are included
    for the same reason, and one paragraph is wrapped in a `<w:tbl>` so
    table-cell text is proven to be reachable.
    """
    paragraphs: list[str] = []
    for index, text in enumerate(entry["paragraphs"]):
        # Split into two runs at the midpoint: same visible characters,
        # fragmented exactly as Word would fragment them.
        middle = len(text) // 2
        runs = "".join(
            f"<w:r><w:t xml:space=\"preserve\">{_escape(part)}</w:t></w:r>"
            for part in (text[:middle], text[middle:])
            if part
        )
        if index == 1:
            runs = f'<w:r><w:tab/></w:r>{runs}<w:r><w:br/></w:r>'
        paragraphs.append(f"<w:p>{runs}</w:p>")

    table_text = entry.get("table_cell", "")
    table = (
        f"<w:tbl><w:tr><w:tc><w:p><w:r><w:t>{_escape(table_text)}</w:t></w:r>"
        f"</w:p></w:tc></w:tr></w:tbl>"
        if table_text
        else ""
    )
    document = (
        f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{_WORD_NS}">'
        f'<w:body>{"".join(paragraphs)}{table}</w:body></w:document>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.'
        'openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
    )
    _write_ooxml_package(
        destination,
        {"[Content_Types].xml": content_types, "word/document.xml": document},
    )


def _write_pptx(entry: dict[str, Any], destination: Path) -> None:
    """A minimal PresentationML deck.

    The spec lists slides in order; this writer numbers the parts
    `slide1..slideN`. Fixtures with TEN OR MORE slides are the point of
    the `slides` spec allowing that many: `slide10.xml` sorts before
    `slide9.xml` lexicographically, so a reader that sorts by filename
    string silently reorders the deck. `_read_pptx_text` sorts by the
    numeric suffix, and the fixture exists to prove it.
    """
    members: dict[str, str] = {}
    overrides: list[str] = []
    for number, slide in enumerate(entry["slides"], start=1):
        shapes = "".join(
            f'<p:sp><p:txBody><a:p><a:r><a:t>{_escape(line)}</a:t></a:r></a:p>'
            f"</p:txBody></p:sp>"
            for line in slide["lines"]
        )
        members[f"ppt/slides/slide{number}.xml"] = (
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<p:sld xmlns:p="{_PRESENTATION_NS}" xmlns:a="{_DRAWING_NS}">'
            f"<p:cSld><p:spTree>{shapes}</p:spTree></p:cSld></p:sld>"
        )
        overrides.append(
            f'<Override PartName="/ppt/slides/slide{number}.xml" ContentType="application/'
            'vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        )
        note = slide.get("notes")
        if note:
            members[f"ppt/notesSlides/notesSlide{number}.xml"] = (
                f'<?xml version="1.0" encoding="UTF-8"?>'
                f'<p:notes xmlns:p="{_PRESENTATION_NS}" xmlns:a="{_DRAWING_NS}">'
                f"<p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r>"
                f"<a:t>{_escape(note)}</a:t></a:r></a:p></p:txBody></p:sp>"
                f"</p:spTree></p:cSld></p:notes>"
            )
    members["[Content_Types].xml"] = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(overrides)
        + "</Types>"
    )
    _write_ooxml_package(destination, members)


def _write_ooxml_package(destination: Path, members: dict[str, str]) -> None:
    """Shared ZIP writer for the OOXML fixtures. Fixed member timestamps
    and sorted member order, so the package bytes are identical every
    time -- the same determinism guarantee `_write_xlsx` relies on."""
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(members):
            archive.writestr(zipfile.ZipInfo(name, date_time=_ZIP_EPOCH), members[name])


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _column_letters(index: int) -> str:
    letters = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_WRITERS = {
    "csv": _write_csv,
    "text": _write_text,
    "binary_stub": _write_binary_stub,
    "png": _write_png,
    "pdf": _write_pdf,
    "zip": _write_zip,
    "xlsx": _write_xlsx,
    "docx": _write_docx,
    "pptx": _write_pptx,
}
