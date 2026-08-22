from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ant.domain import Territory, WorkerCard

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)")
DEF_RE = re.compile(
    r"^\s*(class|def|async\s+def)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)


def build_worker_cards(repo_root: Path, territories: list[Territory]) -> list[WorkerCard]:
    cards: list[WorkerCard] = []
    for territory in territories:
        terms = _top_terms(repo_root, territory.files)
        cards.append(
            WorkerCard(
                id=f"worker-{territory.id}",
                territory_id=territory.id,
                name=f"{territory.root or 'root'} worker",
                root=territory.root,
                responsibilities=[territory.summary],
                searchable_terms=terms,
                files=territory.files,
            )
        )
    return cards


def _top_terms(repo_root: Path, files: list[str], limit: int = 48) -> list[str]:
    lexical: Counter[str] = Counter()
    symbols: Counter[str] = Counter()
    classes_by_file: list[list[str]] = []
    functions_by_file: list[list[str]] = []
    for relative in files[:80]:
        text = (repo_root / relative).read_text(encoding="utf-8", errors="replace")
        lexical.update(token.lower() for token in TOKEN_RE.findall(text))
        lexical.update(_split_terms(relative))
    for relative in files[:400]:
        text = (repo_root / relative).read_text(encoding="utf-8", errors="replace")
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
    ordered = list(dict.fromkeys([*raw_symbols, *definition_terms, *lexical_terms]))
    return ordered[:limit]


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
