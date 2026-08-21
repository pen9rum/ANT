from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from ant.domain import Territory, WorkerCard

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)")
DEF_RE = re.compile(r"^\s*(?:class|def|async\s+def)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


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


def _top_terms(repo_root: Path, files: list[str], limit: int = 16) -> list[str]:
    counter: Counter[str] = Counter()
    for relative in files[:80]:
        text = (repo_root / relative).read_text(encoding="utf-8", errors="replace")
        counter.update(token.lower() for token in TOKEN_RE.findall(text))
        counter.update(_split_path_terms(relative))
    for relative in files[:400]:
        text = (repo_root / relative).read_text(encoding="utf-8", errors="replace")
        for symbol in DEF_RE.findall(text):
            counter[symbol.lower()] += 12
            counter.update(_split_path_terms(symbol))
    stop = {"from", "import", "return", "class", "function", "const", "with", "this", "that"}
    return [term for term, _ in counter.most_common() if term not in stop][:limit]


def _split_path_terms(relative: str) -> list[str]:
    terms = []
    for token in TOKEN_RE.findall(relative):
        terms.append(token.lower())
        terms.extend(part.lower() for part in CAMEL_RE.findall(token))
    return terms
