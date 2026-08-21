from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ant.domain import Evidence

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)")
STOP_WORDS = {
    "are",
    "codebase",
    "for",
    "handled",
    "how",
    "this",
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

    def search(
        self,
        query: str,
        files: list[str],
        limit: int = 8,
        context_lines: int = 6,
    ) -> list[Evidence]:
        terms = _query_terms(query)
        if not terms:
            return []

        candidates: list[tuple[int, str, int, list[str], str]] = []
        for relative in files:
            path = self.repo_root / relative
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for index, line in enumerate(lines, start=1):
                score = _line_score(line, terms)
                if score <= 0:
                    continue
                start = max(1, index - context_lines)
                end = min(len(lines), index + context_lines)
                candidates.append((score, relative, start, lines[start - 1 : end], line.strip()))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return _merge_windows(candidates[: limit * 2], limit=limit)


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for token in TOKEN_RE.findall(query):
        token = token.lower()
        if len(token) > 2 and token not in STOP_WORDS:
            terms.append(token)
        terms.extend(
            part.lower()
            for part in CAMEL_RE.findall(token)
            if len(part) > 2 and part.lower() not in STOP_WORDS
        )
    return sorted(set(terms))


def _line_score(line: str, terms: list[str]) -> int:
    line_terms = set(_query_terms(line))
    lowered = line.lower()
    score = 0
    for term in terms:
        if term in line_terms:
            score += 3
        elif term in lowered:
            score += 1
    return score


def _merge_windows(
    candidates: list[tuple[int, str, int, list[str], str]],
    limit: int,
) -> list[Evidence]:
    evidence: list[Evidence] = []
    occupied: dict[str, list[tuple[int, int]]] = {}
    for score, relative, start, lines, matched_line in candidates:
        end = start + len(lines) - 1
        ranges = occupied.setdefault(relative, [])
        if any(not (end < used_start or start > used_end) for used_start, used_end in ranges):
            continue
        ranges.append((start, end))
        evidence.append(
            Evidence(
                path=relative,
                line_start=start,
                line_end=end,
                quote="\n".join(lines).strip()[:1600],
                reason=f"Scored local retrieval match ({score}) around: {matched_line[:120]}",
            )
        )
        if len(evidence) >= limit:
            break
    return evidence
