from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ant.domain import Evidence

# Imported from the submodules directly, not the `ant.retrieval` package
# aggregator: this module is itself reachable from inside that package's
# __init__ (ant.retrieval.relevance -> ant.tools.path_prior -> ant.tools's
# own __init__ -> this module), so if `ant.retrieval` happens to be the
# first of the two packages a process imports, its __init__ is still
# mid-execution at that point and hasn't bound these names onto the package
# object yet -- but the submodules themselves have, since Python binds a
# module's own top-level names as it executes, not only once the importing
# package's __init__ finishes.
from ant.retrieval.bm25 import BM25Index
from ant.retrieval.dense import (
    REPO_INDEX_KEY,
    EmbeddingIndex,
    build_and_cache_in_background,
    get_shared_embedder,
    shared_repo_dense_dir,
)
from ant.retrieval.relevance import extract_terms, is_stem_match, score_evidence
from ant.tools.symbol_index import SymbolDefinition, SymbolIndex, build_symbol_index

# `_query_terms` used to be its own local implementation; it is now a thin
# alias so every one of this file's many call sites keeps working unchanged
# while actually running the single canonical extractor everything else in
# the ranking pipeline uses too (see ant.retrieval.relevance).
_query_terms = extract_terms



@dataclass(frozen=True)
class LocalSearchTool:
    repo_root: Path
    index_path: Path | None = None
    _symbol_indexes: dict[tuple[str, ...], SymbolIndex] = field(
        default_factory=dict,
        init=False,
        compare=False,
        repr=False,
    )
    # Keyed by REPO_INDEX_KEY only -- one shared, repo-wide embedding index,
    # not one per worker file scope (see ant.retrieval.dense's module
    # docstring). In-process cache only; the disk-backed cache under
    # shared_repo_dense_dir(repo_root) is what survives across process runs
    # (and across however many worker-card index directories analyze this
    # same repo checkout).
    _embedding_index_cache: dict[str, EmbeddingIndex | None] = field(
        default_factory=dict,
        init=False,
        compare=False,
        repr=False,
    )
    # Territory-wide retrieval index, keyed by tuple(sorted(files)) same as
    # _symbol_indexes above -- built once per distinct file scope, reused
    # across a worker's several search() calls in one task. See
    # _territory_index's own docstring for what's cached here and why.
    _territory_cache: dict[
        tuple[str, ...],
        tuple[BM25Index, list[tuple[str, int, list[str]]], dict[str, set[int]]],
    ] = field(default_factory=dict, init=False, compare=False, repr=False)

    def search(
        self,
        query: str,
        files: list[str],
        limit: int = 8,
        context_lines: int = 6,
    ) -> list[Evidence]:
        # context_lines is accepted for backward compatibility with every
        # existing caller (references() passes it explicitly) but no
        # longer changes region size: a region's boundaries now come from
        # _retrieval_regions' own definition/paragraph-aware block
        # splitting (shared with the territory-wide BM25 corpus below),
        # not a fixed line radius around a single matched line.
        del context_lines
        terms = _query_terms(query)
        if not terms:
            return []

        bm25, regions, symbol_term_index = self._territory_index(tuple(sorted(files)))
        if not regions:
            return []
        # Gather more than the caller's own `limit` from each channel
        # before fusing -- same head-room principle the old code already
        # used (its own candidates[: limit * 2] cut before _merge_windows),
        # sized a bit deeper here since Reciprocal Rank Fusion benefits
        # from seeing further down each channel's own ranking, not just
        # its very top.
        channel_limit = limit * 6
        bm25_ranked = _bm25_channel_rank(bm25, regions, terms, channel_limit)
        symbol_ranked = _symbol_path_channel_rank(regions, symbol_term_index, terms, channel_limit)
        fused = _reciprocal_rank_fusion(bm25_ranked, symbol_ranked)
        candidates = [
            (round(score * 1000), path, start, block, matched)
            for score, path, start, block, matched in fused
        ]
        return _merge_windows(candidates[: limit * 4], limit=limit)

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
                end = _capped_block_end(lines, index)
                candidates.append((score + 4, relative, index, lines[index - 1 : end], stripped))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return _merge_windows(candidates, limit=limit)

    def resolve_symbol(
        self, symbol: str, files: list[str], limit: int = 6, need: str = ""
    ) -> list[Evidence]:
        index = self.symbol_index(files)
        definitions = index.by_name.get(symbol, [])
        if not definitions and "." in symbol:
            definitions = index.by_name.get(symbol.rsplit(".", 1)[-1], [])
        matched = sorted(definitions, key=lambda item: (item.path, item.line))[:limit]
        expanded = _expand_large_classes(self.repo_root, index, matched, need)
        return _definitions_to_evidence(
            self.repo_root,
            expanded,
            reason=f"Resolved symbol definition for {symbol}.",
        )

    def resolve_import(
        self, symbol: str, from_path: str, files: list[str], limit: int = 6
    ) -> list[Evidence]:
        definitions = self.symbol_index(files).resolve_import(symbol, from_path)
        return _definitions_to_evidence(
            self.repo_root,
            sorted(definitions, key=lambda item: (item.path, item.line))[:limit],
            reason=f"Resolved import binding for {symbol} from {from_path}.",
        )

    def subclasses(self, symbol: str, files: list[str], limit: int = 8) -> list[Evidence]:
        index = self.symbol_index(files)
        names = {symbol, symbol.rsplit(".", 1)[-1]}
        definitions = []
        seen = set()
        for name in names:
            for definition in index.subclasses.get(name, []):
                key = (definition.path, definition.line, definition.qualname)
                if key in seen:
                    continue
                seen.add(key)
                definitions.append(definition)
        definitions.sort(key=lambda item: (item.path, item.line))
        return _definitions_to_evidence(
            self.repo_root,
            definitions[:limit],
            reason=f"Subclass lookup for base symbol {symbol}.",
        )

    def indexed_callers(self, symbol: str, files: list[str], limit: int = 6) -> list[Evidence]:
        index = self.symbol_index(files)
        definitions = index.callers.get(symbol, [])
        definitions.sort(key=lambda item: (item.path, item.line, item.qualname))
        return _definitions_to_evidence(
            self.repo_root,
            definitions[:limit],
            reason=f"Indexed caller lookup for symbol {symbol}.",
        )

    def symbol_index(self, files: list[str]) -> SymbolIndex:
        key = tuple(sorted(files))
        if key not in self._symbol_indexes:
            self._symbol_indexes[key] = build_symbol_index(self.repo_root, list(key))
        return self._symbol_indexes[key]

    def _territory_index(
        self, sorted_files: tuple[str, ...]
    ) -> tuple[BM25Index, list[tuple[str, int, list[str]]], dict[str, set[int]]]:
        """Cached (per sorted file scope, same key as symbol_index) territory-
        wide retrieval index for search(): reads every file in the scope
        once (no is_low_value_path/has_low_value_part filtering here -- an
        earlier version excluded files under directory names like
        "examples"/"test"/"doc", which silently zeroed out the entire
        corpus for any worker whose whole assigned territory happens to
        live under one of those names, e.g. worker-examples-*; a worker's
        `files` list is already that worker's deliberately assigned scope,
        not a place to second-guess with a generic noise filter), splits
        each into _retrieval_regions' own definition/paragraph-aware
        blocks, and builds

        - one BM25Index over *every* region across the whole scope (not one
          per file -- this is the actual fix for search()'s old per-file-
          scoped BM25: BM25Index's own IDF now sees document frequency
          across the real territory, so a term common across most files in
          it is correctly weighted lower than one rare across the territory,
          regardless of how common or rare it is within any single file);
        - a term -> region-index lookup for exact symbol/filename-stem/path-
          component matches, each normalized through the same _query_terms
          extractor search()'s own query goes through -- a lowercase query
          term matches a symbol's real (possibly capitalized) name exactly
          the way it matches the query's own tokenization, with no
          dependency on the query string happening to preserve the source's
          original capitalization.

        A definition's own region is looked up by (path, its AST line
        number) against the regions _retrieval_regions produced for that
        file. This lookup can miss for a *decorated* definition specifically
        -- _retrieval_regions treats a non-blank, non-definition-prefixed
        decorator line as the start of its own short region, so the block
        starting exactly at the decorated def/class's own line number may
        never get created. A decorated symbol is not lost in that case (it
        is still just as findable through the BM25 channel, which doesn't
        depend on this alignment at all), it only misses the exact-match
        channel's extra boost -- a known, accepted limitation, not a crash
        or a silent wrong answer.
        """
        if sorted_files in self._territory_cache:
            return self._territory_cache[sorted_files]

        regions: list[tuple[str, int, list[str]]] = []
        symbol_term_index: dict[str, set[int]] = defaultdict(set)
        region_by_start: dict[tuple[str, int], int] = {}

        for relative in sorted_files:
            path = self.repo_root / relative
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            stem_terms = _query_terms(Path(relative).stem)
            part_terms = [
                term for part in Path(relative).parts[:-1] for term in _query_terms(part)
            ]
            for start, block in _retrieval_regions(lines):
                region_index = len(regions)
                regions.append((relative, start, block))
                region_by_start[(relative, start)] = region_index
                for term in (*stem_terms, *part_terms):
                    symbol_term_index[term].add(region_index)

        symbol_idx = self.symbol_index(list(sorted_files))
        for definition in symbol_idx.definitions:
            region_index = region_by_start.get((definition.path, definition.line))
            if region_index is None:
                continue
            for term in {*_query_terms(definition.name), *_query_terms(definition.qualname)}:
                symbol_term_index[term].add(region_index)

        documents = [
            (str(index), " ".join(_query_terms("\n".join(block))))
            for index, (_, _, block) in enumerate(regions)
        ]
        result = (BM25Index(documents), regions, dict(symbol_term_index))
        self._territory_cache[sorted_files] = result
        return result

    def dense_search(self, query: str, files: list[str], limit: int = 4) -> list[Evidence]:
        """Paraphrase-robust retrieval: finds candidates whose wording has no
        lexical overlap with `query` at all, by embedding similarity instead
        of term matching.

        Chunk embeddings are built lazily, shared across every worker's
        territory (one repo-wide index, see ant.retrieval.dense's module
        docstring), at symbol-level granularity rather than one embedding
        per paragraph region (see build_embedding_index) -- but embedding a
        worker's files for the first time can still take a while. Rather
        than block this round's query on it, an uncached file returns []
        immediately (or whatever's already cached, filtered to `files`) and
        the build runs on a background thread, cached to disk under
        shared_repo_dense_dir(repo_root) for every later call (from any
        process, and any worker-card index directory over the same repo
        checkout) to pick up once it lands. A query is never worse off than
        "dense retrieval wasn't available yet for these files" -- it can
        never become "this query now waits N minutes."

        Results are restricted to `files` via EmbeddingIndex.search's
        `paths` filter even though the underlying index spans the whole
        repo: a worker must never receive evidence from outside its own
        territory just because the shared index happens to contain it.

        Returns [] whenever no embedder is available (fastembed not
        installed) rather than erroring, so every existing caller keeps
        working unchanged when dense retrieval isn't in use.
        """
        if not self.index_path:
            return []
        embedder = get_shared_embedder()
        if embedder is None:
            return []

        dense_dir = shared_repo_dense_dir(self.repo_root)
        if REPO_INDEX_KEY not in self._embedding_index_cache:
            loaded = EmbeddingIndex.load(dense_dir, REPO_INDEX_KEY)
            self._embedding_index_cache[REPO_INDEX_KEY] = loaded
        index = self._embedding_index_cache[REPO_INDEX_KEY]

        covered = {entry.path for entry in index.entries} if index else set()
        if any(f not in covered for f in files):
            build_and_cache_in_background(
                self.repo_root, files, dense_dir, REPO_INDEX_KEY, embedder
            )
        if index is None:
            return []

        [query_vector] = embedder.embed([query])
        hits = index.search(query_vector, limit=limit, paths=set(files))
        return [
            Evidence(
                path=entry.path,
                line_start=entry.line_start,
                line_end=entry.line_end,
                quote=entry.quote,
                reason=f"Dense semantic match (score={score:.2f}) for: {query[:80]}",
                dense_score=score,
            )
            for score, entry in hits
        ]

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
            lines = (
                (self.repo_root / relative)
                .read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                .splitlines()
            )
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
            lines = (
                (self.repo_root / relative)
                .read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                .splitlines()
            )
            for index, line in enumerate(lines, start=1):
                stripped = line.strip()
                if not _is_definition_line(stripped):
                    continue
                end = _capped_block_end(lines, index)
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
        occurrence = re.compile(rf"\b(?:self\.)?{re.escape(symbol)}\b")
        assignment = re.compile(rf"(^|\W)(?:self\.)?{re.escape(symbol)}\s*=")
        candidates: list[tuple[int, str, int, list[str], str]] = []
        for relative in files:
            lines = (
                (self.repo_root / relative)
                .read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                .splitlines()
            )
            for index, line in enumerate(lines, start=1):
                if not occurrence.search(line):
                    continue
                start = max(1, index - 3)
                end = min(len(lines), index + 3)
                score = 12 if assignment.search(line) else 6
                if "return " in line or "(" in line:
                    score += 2
                candidates.append((score, relative, start, lines[start - 1 : end], line.strip()))
        results = _merge_windows(sorted(candidates, reverse=True), limit=limit)
        return [
            item.model_copy(update={"reason": f"Local data-flow use of symbol {symbol}."})
            for item in results
        ]

    def rank_symbols(
        self,
        symbols: list[str],
        files: list[str],
        limit: int = 8,
        need: str = "",
    ) -> list[str]:
        if not symbols:
            return []
        definitions = _definition_names(self.repo_root, files)
        imports = _imported_names(self.repo_root, files)
        paths = " ".join(files).lower()
        need_terms = set(_query_terms(need))
        scored: list[tuple[int, int, str]] = []
        seen = set()
        for index, symbol in enumerate(symbols):
            if symbol in seen:
                continue
            seen.add(symbol)
            lowered = symbol.lower()
            score = 0
            if symbol in definitions:
                score += 40
            if symbol in imports:
                score += 50
            if "_" in symbol:
                score += 20
            if any(character.isupper() for character in symbol):
                score += 15
            if lowered in paths:
                score += 10
            symbol_terms = set(_query_terms(symbol))
            score += 30 * len(symbol_terms & need_terms)
            if lowered in need.lower():
                score += 40
            if score <= 0:
                continue
            scored.append((score, -index, symbol))
        scored.sort(reverse=True)
        return [symbol for _, _, symbol in scored[:limit]]


def _line_score(line: str, terms: list[str]) -> int:
    # Only used by navigate()/callers()/imports() now -- ranking candidate
    # *definition*/*import*/*caller-usage* lines within an already-narrow,
    # structurally-filtered candidate set, not search()'s open-ended
    # cross-file territory ranking (see search()'s own territory-wide
    # BM25 + symbol/path channels for that -- a flat per-term score is
    # exactly what search() moved away from, since it has no notion of
    # how common or rare a term is across the files being searched; these
    # three callers' candidate sets are already narrowed to a specific
    # structural role first, where that doesn't matter the same way).
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


LARGE_CLASS_LINE_THRESHOLD = 60
# A relevance filter, not just a safety cap. The first version of this
# expansion returned every member (or the first N by line position), and
# that traded one bug for another: with a large class often having dozens
# of methods, the correctly-found-but-irrelevant ones flooded the worker's
# small evidence_limit budget and pushed out a genuinely better match a
# plain search() call had already found elsewhere. Ranking by relevance to
# the actual need and keeping only the top few avoids both failure modes.
MAX_EXPANDED_MEMBERS = 12


def _expand_large_classes(
    repo_root: Path,
    index: SymbolIndex,
    definitions: list[SymbolDefinition],
    need: str,
) -> list[SymbolDefinition]:
    """Swap an oversized class definition for its own most relevant member
    definitions too, not just the class itself.

    `_definitions_to_evidence` truncates each definition's joined text to a
    flat character cap. For a class spanning hundreds of lines (e.g. a
    ~1400-line `Circuit`), that cap is reached long before a method deep
    inside it -- so resolving "Circuit" to answer a question about one of
    its methods (`draw`, say) returns a blob whose visible text never
    actually contains that method, even though the class was correctly
    found. The class's own members are already tracked individually in the
    symbol index (`SymbolDefinition.parent`); returning the ones that best
    match the current need means that method gets its own small, complete
    evidence item instead of being buried past the truncation point of one
    giant one -- or lost among dozens of unranked siblings.
    """
    terms = _query_terms(need) if need else []
    expanded: list[SymbolDefinition] = []
    for definition in definitions:
        expanded.append(definition)
        span = max(1, definition.end_line - definition.line + 1)
        if definition.kind != "class" or span <= LARGE_CLASS_LINE_THRESHOLD:
            continue
        members = [member for member in index.definitions if member.parent == definition.qualname]
        if not members:
            continue
        if terms:
            text = (repo_root / definition.path).read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            scored = sorted(
                members,
                key=lambda member: _member_relevance(member, lines, terms),
                reverse=True,
            )
            expanded.extend(scored[:MAX_EXPANDED_MEMBERS])
        else:
            expanded.extend(sorted(members, key=lambda item: item.line)[:MAX_EXPANDED_MEMBERS])
    return expanded


def _member_relevance(member: SymbolDefinition, lines: list[str], terms: list[str]) -> int:
    start = max(1, member.line)
    end = min(len(lines), max(member.end_line, member.line))
    snippet = "\n".join(lines[start - 1 : min(end, start + 20)])
    return score_evidence(quote=snippet, path=member.path, terms=terms, symbol_name=member.name)


def _definitions_to_evidence(
    repo_root: Path,
    definitions: list[SymbolDefinition],
    reason: str,
) -> list[Evidence]:
    evidence = []
    for definition in definitions:
        lines = (
            (repo_root / definition.path)
            .read_text(encoding="utf-8", errors="replace")
            .splitlines()
        )
        start = max(1, definition.line)
        end = min(len(lines), max(definition.end_line, definition.line))
        evidence.append(
            Evidence(
                path=definition.path,
                line_start=start,
                line_end=end,
                quote="\n".join(lines[start - 1 : end]).strip()[:4000],
                reason=reason,
                claim=_definition_claim(definition),
                symbols=[definition.name, definition.qualname, *definition.bases],
            )
        )
    return evidence


def _definition_claim(definition: SymbolDefinition) -> str:
    if definition.kind == "class" and definition.bases:
        return (
            f"Defines class {definition.qualname or definition.name} "
            f"inheriting from {', '.join(definition.bases)}."
        )
    return f"Defines {definition.kind} {definition.qualname or definition.name}."


_RegionRank = tuple[str, int, list[str], str]  # (path, start_line, block_lines, matched_line)


def _bm25_channel_rank(
    bm25: BM25Index,
    regions: list[tuple[str, int, list[str]]],
    terms: list[str],
    limit: int,
) -> list[_RegionRank]:
    """search()'s BM25 channel: regions already come from the territory-
    wide index built once by LocalSearchTool._territory_index, so this is
    just running the query against it and translating doc_id back to the
    region it names -- the actual cross-file IDF fix lives in that index's
    construction (one BM25Index over the whole territory), not here.
    """
    ranked: list[_RegionRank] = []
    for _score, doc_id in bm25.search(terms, limit=limit):
        path, start, block = regions[int(doc_id)]
        matched = next((line.strip() for line in block if line.strip()), "")
        ranked.append((path, start, block, matched))
    return ranked


def _symbol_path_channel_rank(
    regions: list[tuple[str, int, list[str]]],
    symbol_term_index: dict[str, set[int]],
    terms: list[str],
    limit: int,
) -> list[_RegionRank]:
    """search()'s exact-match channel: how many distinct query terms hit a
    region through symbol_term_index (symbol name/qualname, filename stem,
    or path component -- see _territory_index) decides its rank here. This
    produces a rank ordering for Reciprocal Rank Fusion, not a score
    comparable to BM25's -- RRF only ever looks at rank position within
    each channel, never at the two channels' raw scores side by side,
    which is exactly why RRF is the right way to combine two structurally
    different signals without inventing a new weight to balance them.

    A hit is weighted by 1/(number of regions that term maps to), not
    counted flatly: a shared parent directory (e.g. every file living
    under "extractor/") makes that path component match *every* region in
    the territory, and a flat +1 per matched term would let it swamp a
    genuinely rare symbol/filename match one term at a time -- the exact
    same "common term outweighs the one rare discriminative term" failure
    this whole retrieval rewrite exists to remove, just relocated into
    this channel instead of _line_score. Weighting by inverse match-count
    is the same "rare signals count for more" principle BM25's own IDF
    already applies, not a new hand-tuned constant.
    """
    region_hits: dict[int, float] = defaultdict(float)
    for term in terms:
        matches = symbol_term_index.get(term)
        if not matches:
            # No exact token match -- fall back to the same >=4-char-
            # prefix stem heuristic already used elsewhere in this
            # codebase (ant.retrieval.relevance.is_stem_match) for a
            # different grammatical form of the same word: a query asking
            # about "worker selection" should still match a symbol/file
            # actually spelled "select_workers" or "workers", without
            # this channel silently missing it just because the exact
            # token strings differ.
            matches = {
                region_index
                for key, region_indices in symbol_term_index.items()
                if is_stem_match(term, key)
                for region_index in region_indices
            }
        if not matches:
            continue
        weight = 1.0 / len(matches)
        for region_index in matches:
            region_hits[region_index] += weight
    ranked_indices = sorted(
        region_hits, key=lambda index: (-region_hits[index], regions[index][0], regions[index][1])
    )
    ranked: list[_RegionRank] = []
    for index in ranked_indices[:limit]:
        path, start, block = regions[index]
        matched = next((line.strip() for line in block if line.strip()), "")
        ranked.append((path, start, block, matched))
    return ranked


# Reciprocal Rank Fusion's own standard constant (Cormack, Clarke & Buettcher
# 2009) -- not a value tuned for this codebase or dataset, the one deliberate
# exception to not adding new hand-tuned constants to this ranking path.
_RRF_K = 60


def _reciprocal_rank_fusion(
    *ranked_lists: list[_RegionRank],
) -> list[tuple[float, str, int, list[str], str]]:
    """Combines any number of independently-ranked candidate lists into one
    fused ranking by rank position alone (score(region) = sum over lists of
    1/(_RRF_K + rank_in_that_list)) -- deliberately never looks at either
    channel's own raw score, since BM25 scores and "how many exact terms
    matched" aren't on a comparable scale and blending them directly would
    just be a new hand-tuned weight in disguise. A region appearing highly
    in even one channel, or moderately in both, naturally floats up.
    """
    scores: dict[tuple[str, int], float] = defaultdict(float)
    payload: dict[tuple[str, int], _RegionRank] = {}
    for ranked in ranked_lists:
        for rank, region in enumerate(ranked, start=1):
            key = (region[0], region[1])
            scores[key] += 1.0 / (_RRF_K + rank)
            payload.setdefault(key, region)
    fused = [(scores[key], *payload[key]) for key in scores]
    fused.sort(key=lambda item: item[0], reverse=True)
    return fused


def _retrieval_regions(lines: list[str]) -> list[tuple[int, list[str]]]:
    regions: list[tuple[int, list[str]]] = []
    index = 1
    while index <= len(lines):
        if not lines[index - 1].strip():
            index += 1
            continue
        if _is_definition_line(lines[index - 1].strip()):
            end = _capped_block_end(lines, index)
        else:
            end = min(len(lines), index + 7)
            for cursor in range(index, end):
                if not lines[cursor].strip():
                    end = cursor
                    break
        regions.append((index, lines[index - 1 : end]))
        index = max(index + 1, end + 1)
    return regions


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


def _definition_names(repo_root: Path, files: list[str]) -> set[str]:
    names = set()
    for relative in files:
        lines = (repo_root / relative).read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines:
            name = _definition_name(line.strip())
            if name:
                names.add(name)
    return names


def _imported_names(repo_root: Path, files: list[str]) -> set[str]:
    names = set()
    for relative in files:
        lines = (repo_root / relative).read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("from ") and " import " in stripped:
                imported = stripped.split(" import ", 1)[1]
                names.update(_names_from_import_clause(imported))
            elif stripped.startswith("import "):
                names.update(_names_from_import_clause(stripped.removeprefix("import ")))
    return names


def _names_from_import_clause(clause: str) -> set[str]:
    names = set()
    for part in clause.split(","):
        token = part.strip().split(" as ", 1)[-1].strip()
        token = token.split(".", 1)[0]
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", token):
            names.add(token)
    return names


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


def _capped_block_end(lines: list[str], start_line: int, class_header_lines: int = 11) -> int:
    """`_block_end` walks a definition block until the next dedent, which for
    a class means walking to the end of its ENTIRE body -- every method --
    so for a large class that can span hundreds of lines. A caller that
    then truncates the joined text to a flat character limit
    (`_merge_windows`/`_definitions_to_evidence`) silently loses whatever
    methods live past that many characters in, even though the returned
    line_start/line_end metadata claims full coverage of the class. Cap a
    class's own region at its first nested definition (or a small fallback
    window) instead, so each method remains reachable as its own, separate,
    un-truncated match.
    """
    end = _block_end(lines, start_line)
    if not lines[start_line - 1].lstrip().startswith("class "):
        return end
    nested_definition = next(
        (
            cursor
            for cursor in range(start_line + 1, end + 1)
            if _is_definition_line(lines[cursor - 1].strip())
        ),
        end + 1,
    )
    return min(end, nested_definition - 1, start_line + class_header_lines)
