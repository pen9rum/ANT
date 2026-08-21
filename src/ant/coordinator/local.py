from __future__ import annotations

import re
from pathlib import Path

from ant.domain import (
    Evidence,
    EvidenceState,
    RecruitmentRound,
    TokenUsage,
    UnresolvedNeed,
    WorkerCard,
)
from ant.providers import AnswerSynthesizer, MockLLMProvider, UsageReporter, WorkerReasoner
from ant.tools import LocalSearchTool

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


class LocalCoordinator:
    def __init__(
        self,
        repo_root: Path,
        workers: list[WorkerCard],
        reasoner: WorkerReasoner | None = None,
        synthesizer: AnswerSynthesizer | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.workers = workers
        self.reasoner = reasoner or MockLLMProvider()
        self.synthesizer = synthesizer

    def ask(self, question: str, max_rounds: int = 2) -> EvidenceState:
        evidence: list[Evidence] = []
        unresolved_needs: list[UnresolvedNeed] = []
        rounds: list[RecruitmentRound] = []
        seen_worker_ids: set[str] = set()
        query = question
        search = LocalSearchTool(self.repo_root)

        for round_index in range(max_rounds):
            selected = self._select_workers(query, seen_worker_ids=seen_worker_ids)
            if not selected:
                break

            observations = []
            for worker in selected:
                worker_evidence = search.search(query, worker.files, limit=4)
                navigation_evidence = []
                for symbol in _candidate_symbols(query, worker_evidence)[:4]:
                    navigation_evidence.extend(search.navigate(symbol, worker.files, limit=2))
                    if len(navigation_evidence) >= 4:
                        break
                worker_evidence = _dedupe_evidence([*navigation_evidence, *worker_evidence])[:8]
                evidence.extend(worker_evidence)
                observation = self.reasoner.observe(
                    question=query,
                    worker_id=worker.id,
                    territory_id=worker.territory_id,
                    evidence_count=len(worker_evidence),
                )
                observation.evidence = worker_evidence
                observations.append(observation)
                unresolved_needs.extend(observation.unresolved_needs)
                seen_worker_ids.add(worker.id)

            rounds.append(
                RecruitmentRound(
                    round_index=round_index,
                    query=query,
                    selected_worker_ids=[worker.id for worker in selected],
                    rationale="Selected workers by overlap with query terms and worker card terms.",
                    observations=observations,
                )
            )

            if evidence:
                break
            query = self._query_from_needs(question, unresolved_needs)

        if not evidence and not unresolved_needs:
            unresolved_needs.append(
                UnresolvedNeed(
                    description="No local evidence matched the question.",
                    suggested_terms=question.split()[:6],
                    suggested_territories=[worker.territory_id for worker in self.workers[:5]],
                )
            )

        answer = ""
        if self.synthesizer and evidence:
            coalition_workers = _last_coalition_workers(rounds)
            if len(coalition_workers) > 1:
                answer = self.synthesizer.synthesize_coalition(
                    question=question,
                    worker_ids=coalition_workers,
                    evidence=evidence[:12],
                )
            else:
                answer = self.synthesizer.synthesize(question=question, evidence=evidence[:12])
        usage = (
            self.synthesizer.drain_usage()
            if isinstance(self.synthesizer, UsageReporter)
            else None
        )

        return EvidenceState(
            question=question,
            answer=answer,
            evidence=evidence[:12],
            unresolved_needs=unresolved_needs,
            rounds=rounds,
            usage=usage if isinstance(usage, TokenUsage) else TokenUsage(),
        )

    def _select_workers(
        self,
        question: str,
        limit: int = 3,
        seen_worker_ids: set[str] | None = None,
    ) -> list[WorkerCard]:
        seen_worker_ids = seen_worker_ids or set()
        query_terms = {term.lower() for term in TOKEN_RE.findall(question) if len(term) > 2}
        scored: list[tuple[int, WorkerCard]] = []
        for worker in self.workers:
            if worker.id in seen_worker_ids:
                continue
            terms = set(worker.searchable_terms) | {worker.root.lower(), worker.name.lower()}
            path_terms = {
                token.lower()
                for file in worker.files
                for token in TOKEN_RE.findall(file)
                if len(token) > 2
            }
            terms |= path_terms
            score = sum(1 for query_term in query_terms if _matches_term(query_term, terms))
            if any(token[:1].isupper() for token in TOKEN_RE.findall(question)) and worker.root in {
                "src",
                "lib",
                "ant",
            }:
                score += 2
            scored.append((score, worker))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [worker for score, worker in scored[:limit] if score > 0]
        fallback = [worker for worker in self.workers if worker.id not in seen_worker_ids]
        return selected or fallback[:limit]

    @staticmethod
    def _query_from_needs(question: str, needs: list[UnresolvedNeed]) -> str:
        terms = []
        for need in needs:
            terms.extend(need.suggested_terms)
        return " ".join([question, *terms])


def _matches_term(query_term: str, terms: set[str]) -> bool:
    return any(query_term in term or term in query_term for term in terms)


def _candidate_symbols(query: str, evidence: list[Evidence]) -> list[str]:
    ordered = []
    seen = set()
    for symbol in TOKEN_RE.findall(query):
        if symbol not in seen:
            ordered.append(symbol)
            seen.add(symbol)
    for item in evidence:
        for token in TOKEN_RE.findall(item.quote):
            if token in seen or len(token) <= 3:
                continue
            if "_" in token or token[:1].isupper():
                ordered.append(token)
                seen.add(token)
    return [symbol for symbol in ordered if len(symbol) > 3]


def _dedupe_evidence(evidence: list[Evidence]) -> list[Evidence]:
    deduped = []
    for item in evidence:
        if _overlaps_existing(item, deduped):
            continue
        deduped.append(item)
    return deduped


def _overlaps_existing(item: Evidence, existing: list[Evidence]) -> bool:
    for other in existing:
        if item.path != other.path:
            continue
        if item.line_start >= other.line_start and item.line_end <= other.line_end:
            return True
    return False


def _last_coalition_workers(rounds: list[RecruitmentRound]) -> list[str]:
    if not rounds:
        return []
    return rounds[-1].selected_worker_ids
