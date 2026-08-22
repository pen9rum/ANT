from __future__ import annotations

import math
from collections import Counter


class BM25Index:
    def __init__(
        self,
        documents: list[tuple[str, str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(text.split()) for _, text in documents]
        self.doc_lengths = [sum(counter.values()) for counter in self.term_frequencies]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.document_frequency = Counter(
            term for counter in self.term_frequencies for term in counter.keys()
        )

    def search(self, query_terms: list[str], limit: int = 10) -> list[tuple[float, str]]:
        scored = []
        for index, (doc_id, _) in enumerate(self.documents):
            score = self._score_doc(index, query_terms)
            if score > 0:
                scored.append((score, doc_id))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:limit]

    def _score_doc(self, index: int, query_terms: list[str]) -> float:
        if not self.documents:
            return 0.0
        score = 0.0
        frequencies = self.term_frequencies[index]
        doc_length = self.doc_lengths[index]
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            score += self._idf(term) * (
                frequency
                * (self.k1 + 1)
                / (frequency + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length))
            )
        return score

    def _idf(self, term: str) -> float:
        n_docs = len(self.documents)
        doc_freq = self.document_frequency.get(term, 0)
        return math.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))
