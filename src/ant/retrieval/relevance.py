from __future__ import annotations

import re

from ant.tools.path_prior import has_low_value_part, has_source_part, is_low_value_path

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
    lowered_quote = quote.lower()
    normalized_path = path.replace("\\", "/")
    lowered_reason = reason.lower()
    quote_terms = set(extract_terms(quote))

    value = 0
    if "class " in lowered_quote:
        value += 10
    if "def " in lowered_quote:
        value += 8
    if "definition" in lowered_reason or "implementation" in lowered_reason:
        value += 6
    if has_source_part(normalized_path):
        value += 5
    if normalized_path.endswith(".py"):
        value += 3
    if is_low_value_path(normalized_path):
        value -= 10
    if has_low_value_part(normalized_path):
        value -= 10

    if symbol_name:
        lowered_name = symbol_name.lower()
        if any(is_stem_match(term, lowered_name) for term in terms):
            value += 6

    # Stemming is deliberately NOT used for this generic quote-term overlap
    # tally, only for the precise symbol_name match above. A query term is
    # free text here, not a specific identifier -- against a whole quote's
    # worth of words, a >=4-char prefix match throws false positives (e.g.
    # the query term "qibocal" stem-matching "qibo", which is the host
    # repo's own name and appears in nearly every file, would inflate
    # every candidate's score regardless of actual relevance).
    for term in terms:
        if term in quote_terms:
            value += 3
        elif term in lowered_quote:
            value += 1
    return value
