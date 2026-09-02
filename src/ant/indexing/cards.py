from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path

from ant.domain import CodeSymbol, Territory, WorkerCard
from ant.retrieval import STOP_WORDS
from ant.tools.symbol_index import SymbolDefinition, build_symbol_index

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)")
README_NAME_RE = re.compile(r"^readme", re.IGNORECASE)


def build_worker_cards(repo_root: Path, territories: list[Territory]) -> list[WorkerCard]:
    readme_document_frequency = _readme_document_frequency(repo_root, territories)
    cards: list[WorkerCard] = []
    for territory in territories:
        symbols = _owned_symbols(repo_root, territory.files)
        terms = _top_terms(repo_root, territory.files, readme_document_frequency, symbols=symbols)
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
    symbols: list[CodeSymbol] | None = None,
) -> list[str]:
    """`symbols` should be `_owned_symbols(repo_root, files)`'s own result
    (build_worker_cards passes it through so both fields are derived from
    one AST scan, not two) -- its ordering is already breadth-first and
    file-representative (see _owned_symbols' own docstring), which this
    function relies on directly: grouping it back by path and doing
    round-robin over those groups preserves that same guarantee here,
    rather than re-deriving symbols from a second, independently-capped
    regex scan the way this function used to (`for relative in
    files[:400]`, plus a plain sort-then-slice -- the exact same failure
    class as _owned_symbols' old bug, confirmed separately: yt-dlp's
    1010-file extractor/ territory's searchable_terms still had zero
    site-specific terms even after _owned_symbols was fixed, because this
    function was still doing its own, still-capped extraction).
    """
    if symbols is None:
        symbols = _owned_symbols(repo_root, files)
    lexical: Counter[str] = Counter()
    symbol_counts: Counter[str] = Counter()
    for relative in files[:80]:
        text = _read_text(repo_root, relative)
        lexical.update(token.lower() for token in TOKEN_RE.findall(text))
        lexical.update(_split_terms(relative))
    by_file: dict[str, list[CodeSymbol]] = defaultdict(list)
    for symbol in symbols:
        by_file[symbol.path].append(symbol)
        symbol_counts[symbol.name.lower()] += 1
        symbol_counts.update(_split_terms(symbol.name))
    classes_by_file = [
        [item.name for item in defs if item.kind == "class"] for defs in by_file.values()
    ]
    functions_by_file = [
        [item.name for item in defs if item.kind != "class"] for defs in by_file.values()
    ]
    classes_by_file = [group for group in classes_by_file if group]
    functions_by_file = [group for group in functions_by_file if group]
    stop = {"from", "import", "return", "class", "function", "const", "with", "this", "that"}
    # Evenly sample each of the class/function round-robin sequences on
    # its own -- classes must still win over functions when both compete
    # for the same final `limit` slots (see
    # test_worker_card_keeps_class_names_before_function_names), so
    # sampling only kicks in on whichever side actually exceeds `limit`,
    # not on the two already-concatenated together.
    raw_symbols = [
        term
        for term in [
            *_evenly_sampled(_round_robin(classes_by_file), limit),
            *_evenly_sampled(_round_robin(functions_by_file), limit),
        ]
        if term.lower() not in stop
    ]
    definition_terms = [term for term, _ in symbol_counts.most_common() if term not in stop]
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


def _evenly_sampled(items: list[str], limit: int) -> list[str]:
    """Down-samples `items` to at most `limit` entries spread evenly
    across the whole list (preserving relative order), not a prefix --
    see _top_terms' own call site for why a plain [:limit] on a
    file-ordered list systematically favors whichever files sort first.
    """
    if len(items) <= limit or limit <= 0:
        return items
    step = len(items) / limit
    return [items[int(index * step)] for index in range(limit)]


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


def _owned_symbols(repo_root: Path, files: list[str], limit: int = 400) -> list[CodeSymbol]:
    """`limit` bounds DEPTH, never BREADTH: every file that defines
    anything always gets at least its first definition represented here,
    however many files the territory has, before `limit` is even
    consulted -- only additional passes going deeper into files already
    represented can be capped.

    A plain sort-then-slice[:limit] (the old behavior) silently dropped
    every file past whichever one the cap landed on, in (path, line)
    order -- confirmed directly on yt-dlp's single 1010-file `extractor/`
    territory: only the ~23 alphabetically-earliest files (all starting
    with 'a') ended up represented in WorkerCard.symbols at all.
    `teachable.py`, `youtube.py`, `common.py` (InfoExtractor's own file)
    had zero symbols on the card, making this worker invisible to every
    downstream symbol-based routing signal -- BM25/exact-match/dense
    worker ranking (worker_retrieval.py, retrieval/dense.py's
    build_worker_card_index) all read WorkerCard.symbols, not a fresh
    per-query file scan -- regardless of anything done to fix truncation
    further downstream in a prompt or a ranking channel.

    Within one file, definitions are reordered so round 0 (one definition
    per file, see above) lands on the file's most *representative* one,
    not just whichever comes first by line number: a top-level definition
    (`parent == ""` -- a module-level class/function, not a method or a
    closure nested inside one) outranks a nested one, and among top-level
    definitions, one whose name shares a term with the file's own
    filename (e.g. `YoutubeIE` in `youtube.py`) outranks one that
    doesn't. Confirmed directly both tiebreaks matter on real yt-dlp
    files: `youtube.py`'s first-by-line definition is `BadgeType`, an
    unrelated top-level helper enum -- `YoutubeIE` is defined 1000+ lines
    later in the same file (stem tiebreak fixes this). `common.py`'s
    first-by-line-among-stem-matches was `extract_common`, a function
    nested 3 levels deep inside `InfoExtractor._parse_mpd_periods` that
    merely happens to share the token "common" with the filename --
    `InfoExtractor` itself, the file's actual top-level class, sorts
    before it once nesting is checked first.
    """
    index = build_symbol_index(repo_root, files)
    by_file: dict[str, list[SymbolDefinition]] = defaultdict(list)
    for definition in sorted(index.definitions, key=lambda item: (item.path, item.line)):
        by_file[definition.path].append(definition)
    for path, defs in by_file.items():
        stem_terms = set(_split_terms(Path(path).stem))
        by_file[path] = sorted(
            defs,
            key=lambda item: (
                0 if not item.parent else 1,
                0 if set(_split_terms(item.name)) & stem_terms else 1,
                item.line,
            ),
        )
    file_order = sorted(by_file)

    ordered_definitions: list[SymbolDefinition] = []
    round_index = 0
    while True:
        added_this_round = False
        for path in file_order:
            defs = by_file[path]
            if round_index < len(defs):
                ordered_definitions.append(defs[round_index])
                added_this_round = True
        if not added_this_round:
            break
        round_index += 1
        if len(ordered_definitions) >= limit:
            break

    return [
        CodeSymbol(
            name=definition.name,
            kind=definition.kind,
            path=definition.path,
            line=definition.line,
            qualname=definition.qualname,
            bases=list(definition.bases),
        )
        for definition in ordered_definitions
    ]
