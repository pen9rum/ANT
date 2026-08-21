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

    def navigate(self, symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        terms = _query_terms(symbol)
        if not terms:
            return []
        candidates: list[tuple[int, str, int, list[str], str]] = []
        for relative in files:
            path = self.repo_root / relative
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for index, line in enumerate(lines, start=1):
                stripped = line.strip()
                if not _is_definition_line(stripped):
                    continue
                score = _line_score(stripped, terms)
                if score <= 0:
                    continue
                end = _block_end(lines, index)
                candidates.append((score + 4, relative, index, lines[index - 1 : end], stripped))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return _merge_windows(candidates, limit=limit)

    def read_region(self, path: str, line: int, context_lines: int = 12) -> Evidence:
        lines = (self.repo_root / path).read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, line - context_lines)
        end = min(len(lines), line + context_lines)
        return Evidence(
            path=path,
            line_start=start,
            line_end=end,
            quote="\n".join(lines[start - 1 : end]).strip()[:2400],
            reason=f"Read bounded region around {path}:{line}.",
        )


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


def _is_definition_line(line: str) -> bool:
    return (
        line.startswith("def ")
        or line.startswith("class ")
        or line.startswith("async def ")
        or line.startswith("function ")
        or line.startswith("export function ")
        or line.startswith("const ")
    )


def _block_end(lines: list[str], start_line: int) -> int:
    base_line = lines[start_line - 1]
    base_indent = len(base_line) - len(base_line.lstrip())
    end = start_line
    for index in range(start_line, len(lines)):
        line = lines[index]
        if not line.strip():
            end = index + 1
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= base_indent and index + 1 > start_line:
            break
        end = index + 1
    return end
