from __future__ import annotations

import re

from ant.scoring_config import DEFAULT_SCORING_CONFIG, EvidenceScoringConfig

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)")
STOP_WORDS = {
    "and",
    "are",
    "codebase",
    "does",
    "for",
    "handled",
    "how",
    "into",
    "not",
    "return",
    "returns",
    "this",
    "the",
    "through",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def extract_terms(text: str) -> list[str]:
    """Tokenize `text` into lowercase, stopword-filtered terms, also splitting
    camelCase/PascalCase identifiers into their component words. This is the
    one canonical term extractor -- every place that used to build its own
    ad hoc word list (a plain `.split()`, a bare regex findall with no
    stopword filtering, ...) is a place where two supposedly-equivalent
    relevance judgments could silently disagree.
    """
    terms: list[str] = []
    for token in TOKEN_RE.findall(text):
        lowered = token.lower()
        if len(lowered) > 2 and lowered not in STOP_WORDS:
            terms.append(lowered)
        terms.extend(
            part.lower()
            for part in CAMEL_RE.findall(token)
            if len(part) > 2 and part.lower() not in STOP_WORDS
        )
    return sorted(set(terms))


def is_stem_match(a: str, b: str) -> bool:
    """A query word and a candidate word that share a >=4-character common
    stem, one a prefix of the other, are almost always the same underlying
    concept in a different grammatical form -- "drawing" a question asks
    about vs. a `draw` method it should match, "backends" vs. "backend". A
    plain exact/substring match misses this whenever the wording doesn't
    happen to reuse the exact inflection. Prefix-only (not arbitrary
    substring) keeps this from firing on unrelated words that merely
    contain one another, e.g. "gate" must not match "fusedgate" (starts
    with "fused", not "gate").
    """
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return len(shorter) >= 4 and longer.startswith(shorter)


def score_evidence(
    *,
    quote: str,
    path: str,
    reason: str = "",
    terms: list[str],
    symbol_name: str = "",
    dense_score: float = 0.0,
    config: EvidenceScoringConfig = DEFAULT_SCORING_CONFIG.evidence,
) -> int:
    """The one canonical relevance score for a piece of candidate evidence
    against a set of query terms.

    Before this, three call sites (a worker's own evidence_limit cut, the
    coordinator's final evidence_limit cut, and a large-class member
    ranking) each hand-rolled their own scoring function with different
    weights and different term-matching rules -- none of them stemming-aware,
    one of them not even stopword-filtered. That divergence is not
    cosmetic: a candidate could rank #1 in whichever pass decided it was
    worth considering at all, then get silently dropped by a later pass
    that scored the exact same text differently. Every stage must use this
    same function so "is this evidence relevant" means the same thing
    everywhere in the pipeline.
    """
    # Imported here, not at module top: ant.tools.local depends on this
    # module's names (TOKEN_RE/extract_terms/score_evidence), and ant.tools's
    # package __init__ runs as soon as any ant.tools submodule is touched --
    # including this one, right here. A module-level import of path_prior
    # would make that reverse edge observable mid-way through this module's
    # own initialization whenever `ant.retrieval` happens to be the first of
    # the two packages a process imports. Deferring to call time sidesteps
    # the ordering question entirely: by the time anything actually calls
    # score_evidence(), both packages have long finished importing.
    from ant.tools.path_prior import has_low_value_part, has_source_part, is_low_value_path

    lowered_quote = quote.lower()
    normalized_path = path.replace("\\", "/")
    lowered_reason = reason.lower()
    quote_terms = set(extract_terms(quote))

    value = 0
    if "class " in lowered_quote:
        value += config.class_definition_bonus
    if "def " in lowered_quote:
        value += config.function_definition_bonus
    if "definition" in lowered_reason or "implementation" in lowered_reason:
        value += config.definition_reason_bonus
    if has_source_part(normalized_path):
        value += config.source_path_bonus
    if normalized_path.endswith(".py"):
        value += config.python_file_bonus
    if is_low_value_path(normalized_path):
        value -= config.low_value_path_penalty
    if has_low_value_part(normalized_path):
        value -= config.low_value_part_penalty

    if symbol_name:
        lowered_name = symbol_name.lower()
        if any(is_stem_match(term, lowered_name) for term in terms):
            value += config.symbol_stem_match_bonus

    # Stemming is deliberately NOT used for this generic quote-term overlap
    # tally, only for the precise symbol_name match above. A query term is
    # free text here, not a specific identifier -- against a whole quote's
    # worth of words, a >=4-char prefix match throws false positives (e.g.
    # the query term "qibocal" stem-matching "qibo", which is the host
    # repo's own name and appears in nearly every file, would inflate
    # every candidate's score regardless of actual relevance).
    for term in terms:
        if term in quote_terms:
            value += config.quote_term_overlap_bonus
        elif term in lowered_quote:
            value += config.quote_substring_bonus

    # dense_score is a 0..1 cosine similarity from the optional embedding index
    # (0.0 when no dense index exists, or a candidate was never a dense hit).
    # This is the one place a paraphrase-only match -- one sharing no lexical
    # terms at all with the query, so every bonus above is zero -- can still
    # win a competitive score, instead of being invisible to a purely
    # lexical reranker.
    value += round(dense_score * config.dense_score_weight)
    return value
