from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ant.domain import Evidence

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
STOP_WORDS = {
    "are",
    "for",
    "handled",
    "how",
    "the",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class LocalSearchTool:
    repo_root: Path

    def search(self, query: str, files: list[str], limit: int = 8) -> list[Evidence]:
        terms = [
            term.lower()
            for term in TOKEN_RE.findall(query)
            if len(term) > 2 and term.lower() not in STOP_WORDS
        ]
        if not terms:
            return []

        matches: list[Evidence] = []
        for relative in files:
            path = self.repo_root / relative
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for index, line in enumerate(lines, start=1):
                lowered = line.lower()
                if any(term in lowered for term in terms):
                    matches.append(
                        Evidence(
                            path=relative,
                            line_start=index,
                            line_end=index,
                            quote=line.strip()[:500],
                            reason="Matched local query term.",
                        )
                    )
                    if len(matches) >= limit:
                        return matches
        return matches
