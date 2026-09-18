"""ANTMAN's GAIA substrate: attachment resolution, a declared modality
support policy, deterministic parsers, and information "territories".

WHY GAIA IS NOT MODELLED LIKE WebWalkerQA (this is the central design
decision of this module; see `docs/gaia_environment_report.md` section E
for the full argument):

`web_scope.py` models a territory as *a region of one website*, because
WebWalkerQA is a single-root-site traversal task -- every territory there
is the same KIND of thing (a page subtree), reached the same way
(follow a discovered link), and the only question is which subtree.
GAIA is the opposite shape. A GAIA task is not "navigate one site"; it is
"this question needs a spreadsheet read AND a web lookup AND an
arithmetic step". The heterogeneity is across INFORMATION KINDS, not
across regions of one homogeneous space.

So a `GaiaTerritory` here is **a kind of information source with its own
access primitive**, not a region of a single space:

  * `ATTACHMENT`   -- the one file this task ships with, if any.
  * `WEB`          -- open-web search/fetch. Unlike WebWalkerQA there is
                      no root URL and no "discovered links only" rule;
                      GAIA questions genuinely require open search.
  * `DOCUMENT`     -- prose content extracted from the attachment.
  * `TABULAR`      -- row/column content extracted from the attachment.
  * `COMPUTATION`  -- deterministic arithmetic over facts already found.

Two consequences matter for fairness and are enforced structurally, not
just documented:

1. Territories are derived from **task-observable** signals ONLY -- the
   presence of an attachment and its file extension. Nothing in this
   module reads a question's `Final answer`, `Annotator Metadata`, or any
   other evaluator-only field; `derive_territories()` has no parameter
   through which any of those could be threaded in (asserted by a
   signature test, matching `web_scope.build_territory_from_discovered`'s
   own precedent).
2. Territory derivation is **not an answer heuristic**. It says "a
   spreadsheet is present, so a tabular-reading capability is in scope",
   never "this question is of type X, so do Y". The coordinator's job
   stays the generic unresolved-Need -> territory -> evidence -> Need
   revision loop.

MODALITY POLICY (see `MODALITY_SUPPORT`): every attachment type GAIA
ships is classified explicitly as supported or unsupported. An
unsupported one raises `UnsupportedModalityError` at the point of access.
It is never silently skipped, never degraded to "here is the filename,
guess", and never quietly returns empty content -- because all three of
those turn a capability gap into an invisible accuracy loss that looks
like a reasoning failure in the results table. Crucially the policy is a
property of the SUBSTRATE, so it binds every method identically: there is
no code path by which ANTMAN could read a modality a Matched-ReAct
baseline over the same registry could not.
"""

from __future__ import annotations

import csv
import re
import zipfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from xml.etree import ElementTree

# Bounded reads everywhere -- a GAIA attachment is a hand-authored
# artifact, never a multi-gigabyte dump, so these ceilings exist to keep
# one pathological file from exhausting a context window or wedging a
# run, not to implement a retrieval policy.
MAX_TEXT_CHARS = 200_000
MAX_TABLE_ROWS = 2_000
MAX_TABLE_COLS = 256


class Modality(StrEnum):
    """What KIND of thing an attachment is, decided by extension alone.

    Extension-based rather than content-sniffed on purpose: GAIA's own
    `file_name` field is the only attachment signal a task exposes, the
    corpus is small and hand-curated with accurate extensions, and a
    content sniff would make the support decision depend on bytes the
    agent may not be allowed to read yet.
    """

    TEXT = "text"
    TABULAR = "tabular"
    PDF = "pdf"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    ARCHIVE = "archive"
    OFFICE_DOC = "office_doc"
    UNKNOWN = "unknown"


class SupportStatus(StrEnum):
    """`UNSUPPORTED` and `NOT_IMPLEMENTED` are kept distinct because they
    mean different things to a reader of the results table.
    `NOT_IMPLEMENTED` is a scope decision this pass made and could revisit
    cheaply; `UNSUPPORTED` is a capability this substrate has decided it
    cannot offer FAIRLY to every method (vision, ASR), where "just add it
    for ANTMAN" would be precisely the privileged-capability failure the
    evaluation design forbids. Both fail loudly; only the remediation
    differs.
    """

    SUPPORTED = "supported"
    NOT_IMPLEMENTED = "not_implemented"
    UNSUPPORTED = "unsupported"


EXTENSION_MODALITY: dict[str, Modality] = {
    # Plain text / structured text -- all readable as characters.
    ".txt": Modality.TEXT,
    ".md": Modality.TEXT,
    ".json": Modality.TEXT,
    ".jsonld": Modality.TEXT,
    ".xml": Modality.TEXT,
    ".py": Modality.TEXT,
    ".pdb": Modality.TEXT,  # Protein Data Bank: fixed-column ASCII.
    # Tabular.
    ".csv": Modality.TABULAR,
    ".tsv": Modality.TABULAR,
    ".xlsx": Modality.TABULAR,
    ".xls": Modality.TABULAR,
    # Documents.
    ".pdf": Modality.PDF,
    ".docx": Modality.OFFICE_DOC,
    ".pptx": Modality.OFFICE_DOC,
    # Non-text media.
    ".png": Modality.IMAGE,
    ".jpg": Modality.IMAGE,
    ".jpeg": Modality.IMAGE,
    ".gif": Modality.IMAGE,
    ".mp3": Modality.AUDIO,
    ".m4a": Modality.AUDIO,
    ".wav": Modality.AUDIO,
    ".mov": Modality.VIDEO,
    ".mp4": Modality.VIDEO,
    ".zip": Modality.ARCHIVE,
}

MODALITY_SUPPORT: dict[Modality, SupportStatus] = {
    Modality.TEXT: SupportStatus.SUPPORTED,
    Modality.TABULAR: SupportStatus.SUPPORTED,
    Modality.ARCHIVE: SupportStatus.SUPPORTED,
    # PDF is now a DECLARED capability of this substrate, not a lucky
    # import. PyMuPDF is pinned in `pyproject.toml`'s `gaia` extra, so
    # "PDF works" is a property of the declared environment rather than
    # of whichever machine happened to run the sweep. A missing PyMuPDF
    # is therefore an ENVIRONMENT-PROVISIONING failure
    # (`MissingSubstrateDependencyError`), categorically different from
    # a capability gap: the remediation is `pip install`, not a design
    # decision. `check_substrate_dependencies()` surfaces it up front so
    # a sweep cannot start half-capable.
    Modality.PDF: SupportStatus.SUPPORTED,
    # .docx/.pptx are OOXML (ZIP + XML) and are read here with the
    # stdlib only, exactly as `_read_xlsx` already reads .xlsx -- no new
    # dependency, fully deterministic, no LLM anywhere. Scope and known
    # limits are documented on `_read_docx_text`/`_read_pptx_text`.
    Modality.OFFICE_DOC: SupportStatus.SUPPORTED,
    # Vision and ASR are the fairness line. Supporting either would mean
    # handing some method a perception pipeline; unless it is wired
    # identically into the shared registry for every method, it is a
    # privileged capability. Declared UNSUPPORTED so these tasks fail
    # visibly and can be reported as a known coverage gap.
    Modality.IMAGE: SupportStatus.UNSUPPORTED,
    Modality.AUDIO: SupportStatus.UNSUPPORTED,
    Modality.VIDEO: SupportStatus.UNSUPPORTED,
    Modality.UNKNOWN: SupportStatus.UNSUPPORTED,
}


class GaiaSubstrateError(RuntimeError):
    """Base for every explicit GAIA substrate failure. Exists so a runner
    can catch the whole family and record a typed failure reason instead
    of letting a capability gap surface as a generic exception that reads
    like a crash."""


class UnsupportedModalityError(GaiaSubstrateError):
    """Raised when an attachment's modality is not fairly supported. See
    MODALITY_SUPPORT's own comment: raising is the POINT, not a defect."""

    def __init__(self, modality: Modality, status: SupportStatus, detail: str = "") -> None:
        self.modality = modality
        self.status = status
        message = (
            f"GAIA attachment modality {modality.value!r} is {status.value} in this "
            f"substrate and is deliberately not degraded silently."
        )
        if detail:
            message = f"{message} {detail}"
        super().__init__(message)


class MissingSubstrateDependencyError(GaiaSubstrateError):
    """A modality this substrate DECLARES as supported could not be read
    because its pinned third-party dependency is not installed.

    Deliberately NOT an `UnsupportedModalityError`. That distinction is
    the whole point: `UnsupportedModalityError` says "this substrate has
    decided, for fairness, that nobody gets this capability";
    this says "the capability exists and is declared, and this machine is
    provisioned wrongly". Conflating them would let a `pip install` that
    someone forgot show up in a results table as a principled coverage
    ceiling.
    """

    def __init__(self, modality: Modality, package: str, extra: str = "gaia") -> None:
        self.modality = modality
        self.package = package
        super().__init__(
            f"GAIA modality {modality.value!r} is SUPPORTED by this substrate but its "
            f"pinned dependency {package!r} is not installed in this environment. "
            f"Install it with `pip install 'ant-codebase[{extra}]'`. This is an "
            f"environment-provisioning failure, not a capability gap."
        )


class AttachmentMissingError(GaiaSubstrateError):
    """The task declares a `file_name` but the file is not on disk."""


class NoAttachmentError(GaiaSubstrateError):
    """An attachment primitive was called on a task that has none."""


def modality_for(file_name: str) -> Modality:
    return EXTENSION_MODALITY.get(Path(file_name).suffix.lower(), Modality.UNKNOWN)


#: Modality -> the pinned distribution that implements it. Only PDF has
#: one: text, tabular, archive and office-doc reading are all stdlib.
#: Kept as data so `check_substrate_dependencies()` and the manifest
#: builder agree by construction rather than by two parallel lists.
MODALITY_DEPENDENCIES: dict[Modality, tuple[str, str]] = {
    # modality -> (import name, distribution name)
    Modality.PDF: ("pymupdf", "pymupdf"),
}


def support_status_for(modality: Modality) -> SupportStatus:
    """The DECLARED support status. Note what this deliberately does NOT
    do any more: it no longer probes the environment.

    The previous version resolved PDF by attempting an import, which
    made the substrate's declared capability set vary by machine -- and
    therefore made a frozen subset non-reproducible, since a manifest
    built on a machine without PyMuPDF would silently exclude different
    tasks. Support is now a fixed property of the code; whether this
    machine can *deliver* it is a separate question answered by
    `check_substrate_dependencies()`.
    """
    return MODALITY_SUPPORT.get(modality, SupportStatus.UNSUPPORTED)


def check_substrate_dependencies() -> dict[str, str]:
    """Preflight: for every modality declared SUPPORTED, is its pinned
    dependency actually importable here?

    Returns `{distribution: version}` for everything present and raises
    `MissingSubstrateDependencyError` on the first thing that is not.
    Call this before freezing a manifest or starting a sweep, so a
    provisioning gap fails at minute zero instead of surfacing as a
    handful of mysteriously wrong answers.
    """
    found: dict[str, str] = {}
    for modality, (import_name, distribution) in MODALITY_DEPENDENCIES.items():
        if MODALITY_SUPPORT.get(modality) is not SupportStatus.SUPPORTED:
            continue
        try:
            __import__(import_name)
        except ImportError as exc:
            raise MissingSubstrateDependencyError(modality, distribution) from exc
        found[distribution] = _installed_version(distribution)
    return found


def _installed_version(distribution: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:  # pragma: no cover - defensive
        return "unknown"


def require_supported(modality: Modality, detail: str = "") -> None:
    status = support_status_for(modality)
    if status is not SupportStatus.SUPPORTED:
        raise UnsupportedModalityError(modality, status, detail)


class TerritoryKind(StrEnum):
    ATTACHMENT = "attachment"
    WEB = "web"
    DOCUMENT = "document"
    TABULAR = "tabular"
    COMPUTATION = "computation"


@dataclass(frozen=True)
class GaiaTerritory:
    """One information source with its own access primitive. Plain data --
    exactly like `web_scope.WebTerritory`, this class fetches nothing and
    decides nothing on its own; a coordinator layer consumes it.

    `supported` is carried on the territory itself so a routing layer can
    see a capability gap as a first-class fact ("this task has an audio
    territory this substrate cannot enter") instead of discovering it as
    an exception several rounds later.
    """

    id: str
    kind: TerritoryKind
    description: str
    attachment_ref: str | None = None
    modality: Modality | None = None
    supported: bool = True


def derive_territories(
    *, file_name: str | None, question_length: int = 0
) -> list[GaiaTerritory]:
    """Derives this task's information territories from TASK-OBSERVABLE
    signals only: whether an attachment exists and what extension it has.

    There is deliberately NO parameter for the reference answer, annotator
    metadata, level, or any other evaluator-only field -- the guarantee is
    structural, the same way `web_scope.build_territory_from_discovered`
    has no slot for `golden_path`. `tests/test_gaia_scope.py` asserts this
    function's own signature to keep it that way.

    `question_length` is accepted but intentionally does not gate anything;
    it is recorded on the WEB territory's description purely as a
    descriptive breadcrumb. Every GAIA task gets a WEB and a COMPUTATION
    territory unconditionally, because deciding from the question text
    which tasks "need" the web would be exactly the answer heuristic this
    design forbids.
    """
    territories: list[GaiaTerritory] = [
        GaiaTerritory(
            id="web",
            kind=TerritoryKind.WEB,
            description=(
                "Open web search and page retrieval. Granted unconditionally to every "
                f"task (question length {question_length} chars is recorded, not used "
                "as a gate)."
            ),
        ),
        GaiaTerritory(
            id="computation",
            kind=TerritoryKind.COMPUTATION,
            description="Deterministic arithmetic over facts gathered from other territories.",
        ),
    ]
    if not file_name:
        return territories

    modality = modality_for(file_name)
    supported = support_status_for(modality) is SupportStatus.SUPPORTED
    territories.append(
        GaiaTerritory(
            id="attachment",
            kind=TerritoryKind.ATTACHMENT,
            description=f"The file shipped with this task ({file_name}).",
            attachment_ref=file_name,
            modality=modality,
            supported=supported,
        )
    )
    if modality is Modality.TABULAR:
        territories.append(
            GaiaTerritory(
                id="attachment_tabular",
                kind=TerritoryKind.TABULAR,
                description=f"Row/column contents of {file_name}.",
                attachment_ref=file_name,
                modality=modality,
                supported=supported,
            )
        )
    elif modality in (Modality.TEXT, Modality.PDF, Modality.OFFICE_DOC):
        territories.append(
            GaiaTerritory(
                id="attachment_document",
                kind=TerritoryKind.DOCUMENT,
                description=f"Prose contents of {file_name}.",
                attachment_ref=file_name,
                modality=modality,
                supported=supported,
            )
        )
    return territories


@dataclass
class TableView:
    """A bounded, deterministic row/column read of a tabular attachment."""

    rows: list[list[str]] = field(default_factory=list)
    sheet_name: str | None = None
    truncated_rows: bool = False
    truncated_cols: bool = False

    def to_text(self) -> str:
        return "\n".join("\t".join(cell for cell in row) for row in self.rows)


class GaiaEnvironment:
    """Per-task substrate root: resolves this task's attachment and reads
    it under the declared modality policy.

    Mirrors the role `EvalRepoEnvironment` plays for repo tasks and
    `EvalWebEnvironment` plays for web tasks -- the thing an agent is
    handed after `prepare_environment()`. Holds no question text and no
    reference answer by construction: it is built from a directory and a
    filename, and there is no constructor slot for anything else.
    """

    def __init__(self, task_root: Path, file_name: str | None = None) -> None:
        self.task_root = Path(task_root)
        self.file_name = file_name or None

    def has_attachment(self) -> bool:
        return self.file_name is not None

    def attachment_modality(self) -> Modality | None:
        return modality_for(self.file_name) if self.file_name else None

    def attachment_path(self) -> Path:
        """Resolves the attachment, refusing any path that escapes
        `task_root`. GAIA's `file_name` comes from dataset metadata rather
        than from a model, so this is defence in depth rather than an
        expected attack path -- but an adapter that ever passed a
        model-proposed filename through here would otherwise have an
        arbitrary-read primitive.
        """
        if not self.file_name:
            raise NoAttachmentError("This GAIA task declares no attachment.")
        root = self.task_root.resolve()
        candidate = (root / self.file_name).resolve()
        if not candidate.is_relative_to(root):
            raise AttachmentMissingError(
                f"Attachment {self.file_name!r} resolves outside the task root {root}."
            )
        if not candidate.exists():
            raise AttachmentMissingError(
                f"Attachment {self.file_name!r} declared by this task is not present at "
                f"{candidate}. Run the adapter's prepare_environment() first."
            )
        return candidate

    def read_text(self, max_chars: int = MAX_TEXT_CHARS) -> str:
        """Text/PDF/office attachment -> characters. Raises
        `UnsupportedModalityError` for anything this substrate does not
        fairly support."""
        path = self.attachment_path()
        modality = modality_for(self.file_name or "")
        if modality is Modality.TABULAR:
            # Tabular files have a dedicated primitive; routing them
            # through read_text would hand back either mojibake (.xlsx is
            # a ZIP) or an unlabelled CSV blob. Redirect explicitly.
            return self.read_table().to_text()[:max_chars]
        require_supported(modality)
        if modality is Modality.PDF:
            return _read_pdf_text(path)[:max_chars]
        if modality is Modality.OFFICE_DOC:
            suffix = path.suffix.lower()
            if suffix == ".docx":
                return _read_docx_text(path)[:max_chars]
            if suffix == ".pptx":
                return _read_pptx_text(path)[:max_chars]
            # .doc/.ppt are the pre-2007 OLE binary containers -- a
            # different format entirely, not a variant of OOXML. Declared
            # rather than guessed at, exactly as .xls is in read_table().
            raise UnsupportedModalityError(
                modality,
                SupportStatus.NOT_IMPLEMENTED,
                f"{suffix!r} is a legacy OLE binary Office container; only the OOXML "
                "formats .docx and .pptx are read here.",
            )
        if modality is Modality.ARCHIVE:
            return "\n".join(self.list_archive())
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]

    def read_table(
        self, max_rows: int = MAX_TABLE_ROWS, max_cols: int = MAX_TABLE_COLS
    ) -> TableView:
        path = self.attachment_path()
        modality = modality_for(self.file_name or "")
        if modality is not Modality.TABULAR:
            raise UnsupportedModalityError(
                modality,
                support_status_for(modality),
                "read_table() requires a tabular attachment (.csv/.tsv/.xlsx).",
            )
        suffix = path.suffix.lower()
        if suffix in (".csv", ".tsv"):
            return _read_delimited(path, suffix, max_rows, max_cols)
        if suffix == ".xlsx":
            return _read_xlsx(path, max_rows, max_cols)
        # .xls is the pre-2007 binary format -- a genuinely different
        # container that the stdlib cannot open. Declared, not guessed at.
        raise UnsupportedModalityError(
            Modality.TABULAR,
            SupportStatus.NOT_IMPLEMENTED,
            f"{suffix!r} is the legacy binary Excel container; only .csv/.tsv/.xlsx are read here.",
        )

    def list_archive(self) -> list[str]:
        path = self.attachment_path()
        require_supported(modality_for(self.file_name or ""))
        with zipfile.ZipFile(path) as archive:
            return sorted(info.filename for info in archive.infolist() if not info.is_dir())


def _read_pdf_text(path: Path) -> str:
    """PDF -> text, via PyMuPDF (pinned in the `gaia` extra).

    Imported as `pymupdf`, not the legacy `fitz` alias: PyMuPDF >= 1.24
    emits a deprecation warning for `import fitz` and the alias is slated
    for removal, so pinning the modern name keeps this from breaking on a
    future bump.

    Page text is joined with a form feed so a downstream reader can tell
    where a page boundary was -- GAIA PDF questions are frequently "on
    page N ...", and collapsing the boundary throws that signal away.
    Extraction is PyMuPDF's default reading order; no OCR is attempted,
    so a SCANNED (image-only) PDF yields empty or near-empty text rather
    than silently wrong text. That is the honest failure: OCR is a vision
    capability and vision is UNSUPPORTED in this substrate for every
    method equally.
    """
    try:
        import pymupdf
    except ImportError as exc:
        raise MissingSubstrateDependencyError(Modality.PDF, "pymupdf") from exc

    with pymupdf.open(path) as document:
        return "\f".join(page.get_text() for page in document)


# OOXML namespaces. `w:` is WordprocessingML, `a:`/`p:` are
# DrawingML/PresentationML -- .pptx keeps its text inside DrawingML
# shapes, which is why the text element there is `a:t` and not `p:t`.
_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_DRAWING_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def _read_docx_text(path: Path) -> str:
    """Minimal, dependency-free .docx reader over the OOXML container.

    Same bar and same technique as `_read_xlsx`: `zipfile` +
    `xml.etree`, no `python-docx`, fully deterministic, zero LLM. A
    .docx is a ZIP whose `word/document.xml` holds a flat sequence of
    `<w:p>` paragraphs; the visible characters of a paragraph are its
    descendant `<w:t>` runs concatenated in document order.

    SCOPE, stated rather than discovered later:
      * The main document body only. Headers, footers, footnotes,
        endnotes and comments live in separate parts and are NOT read.
      * Table cells ARE included -- a `<w:tbl>` contains ordinary
        `<w:p>` paragraphs, so they fall out of the same traversal -- but
        row/column GEOMETRY is lost; a table reads as consecutive lines.
      * `<w:tab>` becomes a tab and `<w:br>` a newline, so simple
        layout survives. Numbering, styles, revision marks and field
        codes are not interpreted.
      * Tracked-change deletions (`<w:delText>`) are NOT included, so
        the text read is the document as it currently stands.
    """
    with zipfile.ZipFile(path) as archive:
        if "word/document.xml" not in set(archive.namelist()):
            raise UnsupportedModalityError(
                Modality.OFFICE_DOC,
                SupportStatus.NOT_IMPLEMENTED,
                f"No `word/document.xml` part inside {path.name}; this is not a .docx.",
            )
        root = ElementTree.fromstring(archive.read("word/document.xml"))

    lines: list[str] = []
    for paragraph in root.iter(f"{_WORD_NS}p"):
        lines.append(_ooxml_paragraph_text(paragraph, _WORD_NS))
    return "\n".join(lines)


def _ooxml_paragraph_text(paragraph: ElementTree.Element, namespace: str) -> str:
    pieces: list[str] = []
    for node in paragraph.iter():
        tag = node.tag
        if tag == f"{namespace}t":
            pieces.append(node.text or "")
        elif tag == f"{namespace}tab":
            pieces.append("\t")
        elif tag == f"{namespace}br" or tag == f"{namespace}cr":
            pieces.append("\n")
    return "".join(pieces)


#: Case-insensitive because the two part families spell the stem
#: differently: `ppt/slides/slide7.xml` but `ppt/notesSlides/notesSlide7.xml`.
#: A case-sensitive pattern silently matched slides and missed every set
#: of speaker notes -- caught by a fixture whose first slide has notes.
_SLIDE_NUMBER_RE = re.compile(r"slide(\d+)\.xml$", re.IGNORECASE)


def _read_pptx_text(path: Path) -> str:
    """Minimal, dependency-free .pptx reader over the OOXML container.

    A .pptx keeps one XML part per slide at `ppt/slides/slideN.xml`, and
    each slide's visible text lives in DrawingML `<a:p>` paragraphs
    inside shapes. Speaker notes live in parallel parts at
    `ppt/notesSlides/notesSlideN.xml` and ARE read, because a GAIA
    question about a deck can just as easily be about the notes.

    SLIDE ORDER: parts are sorted by the NUMERIC suffix of their
    filename, not lexicographically -- `slide10.xml` sorts after
    `slide9.xml`, which a plain string sort gets wrong. That matters:
    "what is on the third slide" is a real GAIA question shape.

    KNOWN LIMITS: the presentation's own `presentation.xml` slide-id
    order is not consulted, so a deck whose slides were reordered
    without renaming its parts would be read in file order. Shape
    geometry, z-order, grouping, charts, SmartArt and embedded media are
    not interpreted; a table reads as consecutive lines.
    """
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        slides = sorted(
            (name for name in names if _SLIDE_NUMBER_RE.search(name) and "/slides/" in name),
            key=lambda name: int(_SLIDE_NUMBER_RE.search(name).group(1)),  # type: ignore[union-attr]
        )
        notes = {
            int(_SLIDE_NUMBER_RE.search(name).group(1)): name  # type: ignore[union-attr]
            for name in names
            if _SLIDE_NUMBER_RE.search(name) and "/notesSlides/" in name
        }
        if not slides:
            raise UnsupportedModalityError(
                Modality.OFFICE_DOC,
                SupportStatus.NOT_IMPLEMENTED,
                f"No `ppt/slides/slideN.xml` parts inside {path.name}; this is not a .pptx.",
            )

        blocks: list[str] = []
        for name in slides:
            number = int(_SLIDE_NUMBER_RE.search(name).group(1))  # type: ignore[union-attr]
            body = _drawingml_text(archive.read(name))
            block = f"--- slide {number} ---\n{body}"
            if number in notes:
                note_text = _drawingml_text(archive.read(notes[number]))
                if note_text.strip():
                    block = f"{block}\n[notes] {note_text}"
            blocks.append(block)
    return "\n".join(blocks)


def _drawingml_text(payload: bytes) -> str:
    root = ElementTree.fromstring(payload)
    return "\n".join(
        _ooxml_paragraph_text(paragraph, _DRAWING_NS)
        for paragraph in root.iter(f"{_DRAWING_NS}p")
    )


def _read_delimited(path: Path, suffix: str, max_rows: int, max_cols: int) -> TableView:
    delimiter = "\t" if suffix == ".tsv" else ","
    view = TableView(sheet_name=path.name)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for index, row in enumerate(csv.reader(handle, delimiter=delimiter)):
            if index >= max_rows:
                view.truncated_rows = True
                break
            if len(row) > max_cols:
                view.truncated_cols = True
                row = row[:max_cols]
            view.rows.append([cell.strip() for cell in row])
    return view


_SHEET_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_CELL_REF_RE = re.compile(r"^([A-Z]+)")


def _column_index(cell_ref: str) -> int:
    """`"BC12"` -> 54 (0-based). Needed because .xlsx omits empty cells
    entirely, so row reconstruction has to be by column reference, not by
    counting the `<c>` elements present."""
    match = _CELL_REF_RE.match(cell_ref or "")
    if not match:
        return 0
    index = 0
    for char in match.group(1):
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def _read_xlsx(path: Path, max_rows: int, max_cols: int) -> TableView:
    """Minimal, dependency-free .xlsx reader over the OOXML container.

    Uses only `zipfile` + `xml.etree`, matching `web_fetch.py`'s
    stdlib-only precedent, rather than adding `openpyxl` as a dependency.
    Scope is deliberate and declared: the FIRST worksheet, cell VALUES
    only. Formulas are read as their cached computed value (what a reader
    of the file would see); styles, number formats, merged-cell geometry,
    and date serial-number conversion are NOT interpreted -- a date will
    surface as its underlying serial number. That is a real limitation,
    stated here rather than discovered later in a results table.

    .xlsx is the single largest attachment category in GAIA's 2023
    validation split (13 of 38 attachment files), which is why this is
    built rather than declared NOT_IMPLEMENTED.
    """
    view = TableView()
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{_SHEET_NS}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{_SHEET_NS}t")))

        sheet_names = [name for name in sorted(names) if name.startswith("xl/worksheets/sheet")]
        if not sheet_names:
            raise UnsupportedModalityError(
                Modality.TABULAR,
                SupportStatus.NOT_IMPLEMENTED,
                f"No worksheet part found inside {path.name}.",
            )
        target = sheet_names[0]
        view.sheet_name = Path(target).stem
        sheet = ElementTree.fromstring(archive.read(target))
        for row_element in sheet.iter(f"{_SHEET_NS}row"):
            if len(view.rows) >= max_rows:
                view.truncated_rows = True
                break
            cells: dict[int, str] = {}
            for cell in row_element.findall(f"{_SHEET_NS}c"):
                column = _column_index(cell.get("r", ""))
                if column >= max_cols:
                    view.truncated_cols = True
                    continue
                cells[column] = _cell_value(cell, shared)
            width = max(cells) + 1 if cells else 0
            view.rows.append([cells.get(index, "") for index in range(width)])
    return view


def _cell_value(cell: ElementTree.Element, shared: list[str]) -> str:
    cell_type = cell.get("t")
    if cell_type == "s":
        value = cell.findtext(f"{_SHEET_NS}v")
        try:
            return shared[int(value)] if value is not None else ""
        except (ValueError, IndexError):
            return ""
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{_SHEET_NS}t"))
    return cell.findtext(f"{_SHEET_NS}v") or ""
