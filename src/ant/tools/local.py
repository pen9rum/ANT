from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ant.domain import Evidence
from ant.retrieval import BM25Index

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
        symbols = _query_symbols(query)
        if not terms:
            return []

        candidates: list[tuple[int, str, int, list[str], str]] = []
        for relative in files:
            path = self.repo_root / relative
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            bm25_line_ids = set(_bm25_line_ids(relative, lines, terms, limit=80))
            for index, line in enumerate(lines, start=1):
                line_score = _line_score(line, terms, symbols=symbols)
                bm25_hit = f"{relative}:{index}" in bm25_line_ids
                if line_score <= 0 and not bm25_hit:
                    continue
                score = line_score + _path_score(relative, symbols)
                if bm25_hit:
                    score += 6
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

    def references(self, symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        results = self.search(symbol, files, limit=limit, context_lines=3)
        return [
            item.model_copy(update={"reason": f"Reference search for symbol {symbol}."})
            for item in results
        ]

    def imports(self, module_or_symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        terms = _query_terms(module_or_symbol)
        candidates: list[tuple[int, str, int, list[str], str]] = []
        for relative in files:
            lines = (self.repo_root / relative).read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
            for index, line in enumerate(lines, start=1):
                stripped = line.strip()
                if not (stripped.startswith("import ") or stripped.startswith("from ")):
                    continue
                score = _line_score(stripped, terms)
                if score > 0:
                    candidates.append((score + 3, relative, index, [line], stripped))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return _merge_windows(candidates, limit=limit)

    def callers(self, symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        terms = _query_terms(symbol)
        candidates: list[tuple[int, str, int, list[str], str]] = []
        for relative in files:
            lines = (self.repo_root / relative).read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
            for index, line in enumerate(lines, start=1):
                stripped = line.strip()
                if not _is_definition_line(stripped):
                    continue
                end = _block_end(lines, index)
                block = "\n".join(lines[index - 1 : end])
                if f"{symbol}(" not in block or _definition_name(stripped) == symbol:
                    continue
                candidates.append(
                    (
                        10 + _line_score(block, terms),
                        relative,
                        index,
                        lines[index - 1 : end],
                        stripped,
                    )
                )
        candidates.sort(key=lambda item: item[0], reverse=True)
        return _merge_windows(candidates, limit=limit)

    def callees(self, symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        definitions = self.navigate(symbol, files, limit=2)
        call_names: list[str] = []
        for item in definitions:
            for call in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", item.quote):
                if call not in {symbol, "if", "for", "while", "return"} and call not in call_names:
                    call_names.append(call)
        evidence: list[Evidence] = []
        for call in call_names[:limit]:
            evidence.extend(self.navigate(call, files, limit=1))
            if len(evidence) >= limit:
                break
        return [
            item.model_copy(update={"reason": f"Callee navigation from symbol {symbol}."})
            for item in evidence[:limit]
        ]

    def assignments(self, symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        pattern = re.compile(rf"(^|\W)(self\.)?{re.escape(symbol)}\s*=")
        candidates: list[tuple[int, str, int, list[str], str]] = []
        for relative in files:
            lines = (self.repo_root / relative).read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
            for index, line in enumerate(lines, start=1):
                if not pattern.search(line):
                    continue
                start = max(1, index - 3)
                end = min(len(lines), index + 3)
                candidates.append((10, relative, start, lines[start - 1 : end], line.strip()))
        return _merge_windows(candidates, limit=limit)


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


def _query_symbols(query: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(query)
        if "_" in token or any(character.isupper() for character in token)
    ]


def _line_score(line: str, terms: list[str], symbols: list[str] | None = None) -> int:
    line_terms = set(_query_terms(line))
    lowered = line.lower()
    score = 0
    for term in terms:
        if term in line_terms:
            score += 3
        elif term in lowered:
            score += 1
    for symbol in symbols or []:
        if symbol in line:
            score += 8
        if _is_definition_line(line.strip()) and symbol in line:
            score += 8
    return score


def _path_score(relative: str, symbols: list[str]) -> int:
    path = relative.replace("\\", "/")
    score = 0
    if path.endswith(".py"):
        score += 2
    if "/README" in path or path.endswith("README.md"):
        score -= 4
    if path.startswith("examples/"):
        score -= 2
    lowered = path.lower()
    for symbol in symbols:
        if symbol.lower() in lowered:
            score += 4
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
        if any(_is_contained(start, end, used_start, used_end) for used_start, used_end in ranges):
            continue
        ranges.append((start, end))
        evidence.append(
            Evidence(
                path=relative,
                line_start=start,
                line_end=end,
                quote="\n".join(lines).strip()[:4000],
                reason=f"Scored local retrieval match ({score}) around: {matched_line[:120]}",
            )
        )
        if len(evidence) >= limit:
            break
    return evidence


def _bm25_line_ids(relative: str, lines: list[str], terms: list[str], limit: int) -> list[str]:
    documents = [
        (f"{relative}:{index}", " ".join(_query_terms(line)))
        for index, line in enumerate(lines, start=1)
        if line.strip()
    ]
    return [doc_id for _, doc_id in BM25Index(documents).search(terms, limit=limit)]


def _is_contained(start: int, end: int, used_start: int, used_end: int) -> bool:
    return start >= used_start and end <= used_end


def _is_definition_line(line: str) -> bool:
    return (
        line.startswith("def ")
        or line.startswith("class ")
        or line.startswith("async def ")
        or line.startswith("function ")
        or line.startswith("export function ")
        or line.startswith("const ")
    )


def _definition_name(line: str) -> str:
    match = re.match(r"(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
    return match.group(1) if match else ""


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
