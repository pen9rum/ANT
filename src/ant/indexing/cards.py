from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ant.domain import CodeSymbol, Territory, WorkerCard
from ant.retrieval import STOP_WORDS
from ant.tools.symbol_index import build_symbol_index

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)")
DEF_RE = re.compile(
    r"^\s*(class|def|async\s+def)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
README_NAME_RE = re.compile(r"^readme", re.IGNORECASE)


def build_worker_cards(repo_root: Path, territories: list[Territory]) -> list[WorkerCard]:
    readme_document_frequency = _readme_document_frequency(repo_root, territories)
    cards: list[WorkerCard] = []
    for territory in territories:
        terms = _top_terms(repo_root, territory.files, readme_document_frequency)
        symbols = _owned_symbols(repo_root, territory.files)
        responsibilities = [territory.summary]
        readme_summary = _readme_summary(repo_root, territory.files)
        if readme_summary:
            responsibilities.append(readme_summary)
        card = WorkerCard(
            id=f"worker-{territory.id}",
            territory_id=territory.id,
            name=f"{territory.root or 'root'} worker",
            root=territory.root,
            responsibilities=responsibilities,
            searchable_terms=terms,
            files=territory.files,
            symbols=symbols,
        )
        cards.append(card.model_copy(update={"routing_summary": template_routing_summary(card)}))
    return cards


def template_routing_summary(card: WorkerCard) -> str:
    """Deterministic, zero-cost routing_summary: territory + core
    capability + typical needs handled, built from a card's own fields
    with no LLM call. Used whenever no LLM is available (`ant index`
    without `--llm-cards`, or an evolution call with no reasoner) and for
    the ephemeral temporary-bridge worker (LocalCoordinator's
    _build_temporary_bridge -- never worth an LLM call for something built
    and thrown away within a single task).
    """
    territory = card.territory_id or card.root or "root"
    capability = "; ".join(card.responsibilities[:2]) or card.name
    terms = ", ".join(card.searchable_terms[:6])
    return f"territory: {territory} | capability: {capability} | typical needs: {terms}"


def _top_terms(
    repo_root: Path,
    files: list[str],
    readme_document_frequency: Counter[str] | None = None,
    limit: int = 48,
) -> list[str]:
    lexical: Counter[str] = Counter()
    symbols: Counter[str] = Counter()
    classes_by_file: list[list[str]] = []
    functions_by_file: list[list[str]] = []
    for relative in files[:80]:
        text = _read_text(repo_root, relative)
        lexical.update(token.lower() for token in TOKEN_RE.findall(text))
        lexical.update(_split_terms(relative))
    for relative in files[:400]:
        text = _read_text(repo_root, relative)
        file_classes = []
        file_functions = []
        for kind, symbol in DEF_RE.findall(text):
            if kind == "class":
                file_classes.append(symbol)
            else:
                file_functions.append(symbol)
            symbols[symbol.lower()] += 1
            symbols.update(_split_terms(symbol))
        if file_classes:
            classes_by_file.append(file_classes)
        if file_functions:
            functions_by_file.append(file_functions)
    stop = {"from", "import", "return", "class", "function", "const", "with", "this", "that"}
    raw_symbols = [
        term
        for term in [*_round_robin(classes_by_file), *_round_robin(functions_by_file)]
        if term.lower() not in stop
    ]
    definition_terms = [term for term, _ in symbols.most_common() if term not in stop]
    lexical_terms = [term for term, _ in lexical.most_common() if term not in stop]
    # READMEs are written to describe what a territory is for in plain
    # language -- unlike lexical/symbol terms above, they are read in full
    # (no files[:N] cap) so a README deep in a large territory's file list
    # (sorted alphabetically) can't get silently dropped just because there
    # are hundreds of other files ahead of it. They get priority placement
    # since that natural-language vocabulary is exactly what a question is
    # likely to use, and identifier names often aren't.
    readme_terms = _readme_terms(repo_root, files, stop, readme_document_frequency or Counter())
    ordered = list(
        dict.fromkeys([*readme_terms, *raw_symbols, *definition_terms, *lexical_terms])
    )
    return ordered[:limit]


def _readme_files(files: list[str]) -> list[str]:
    return [file for file in files if README_NAME_RE.match(Path(file).name)]


def _strip_markup(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def _readme_document_frequency(repo_root: Path, territories: list[Territory]) -> Counter[str]:
    """How many territories' READMEs each word appears in.

    Plain term frequency favors whichever word repeats most *within* one
    README, which is usually a generic domain word ("quantum", "circuit")
    that appears in nearly every territory's README too -- not the word that
    actually distinguishes this territory from its siblings. Down-weighting
    by how many territories a word shows up in (document frequency) lets a
    rare-but-defining word like "bloch", mentioned twice in one README and
    nowhere else, outrank a word mentioned four times here but present in
    every other example's README as well.
    """
    document_frequency: Counter[str] = Counter()
    for territory in territories:
        terms_in_territory: set[str] = set()
        for relative in _readme_files(territory.files):
            text = _strip_markup(_read_text(repo_root, relative))
            terms_in_territory.update(
                token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOP_WORDS
            )
        document_frequency.update(terms_in_territory)
    return document_frequency


def _readme_terms(
    repo_root: Path,
    files: list[str],
    stop: set[str],
    document_frequency: Counter[str],
    limit: int = 24,
) -> list[str]:
    term_frequency: Counter[str] = Counter()
    for relative in _readme_files(files):
        text = _strip_markup(_read_text(repo_root, relative))
        term_frequency.update(
            token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOP_WORDS
        )
    scored = sorted(
        term_frequency.items(),
        key=lambda item: item[1] / document_frequency.get(item[0], 1),
        reverse=True,
    )
    return [term for term, _ in scored[:limit] if term not in stop]


def _readme_summary(repo_root: Path, files: list[str], max_length: int = 240) -> str:
    readmes = _readme_files(files)
    if not readmes:
        return ""
    text = _read_text(repo_root, readmes[0])
    title = ""
    paragraph: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        if stripped.startswith("#"):
            if not title:
                title = stripped.lstrip("#").strip()
            continue
        if stripped.startswith(("![", "[!")):
            continue
        paragraph.append(stripped)
    summary = ": ".join(part for part in [title, " ".join(paragraph)] if part).strip()
    if not summary:
        return ""
    if len(summary) > max_length:
        summary = summary[: max_length - 1].rstrip() + "…"
    return f"README ({readmes[0]}): {summary}"


def _read_text(repo_root: Path, relative: str) -> str:
    return (repo_root / relative).read_text(encoding="utf-8", errors="replace")


def _round_robin(groups: list[list[str]]) -> list[str]:
    ordered = []
    max_length = max((len(group) for group in groups), default=0)
    for index in range(max_length):
        for group in groups:
            if index < len(group):
                ordered.append(group[index])
    return ordered


def _split_terms(relative: str) -> list[str]:
    terms = []
    for token in TOKEN_RE.findall(relative):
        terms.append(token.lower())
        terms.extend(part.lower() for part in CAMEL_RE.findall(token))
    return terms


def _owned_symbols(repo_root: Path, files: list[str], limit: int = 160) -> list[CodeSymbol]:
    index = build_symbol_index(repo_root, files)
    return [
        CodeSymbol(
            name=definition.name,
            kind=definition.kind,
            path=definition.path,
            line=definition.line,
            qualname=definition.qualname,
            bases=list(definition.bases),
        )
        for definition in sorted(index.definitions, key=lambda item: (item.path, item.line))[:limit]
    ]
