from __future__ import annotations

import re
from pathlib import Path

from ant.domain import Evidence, EvidenceState, RecruitmentRound, UnresolvedNeed, WorkerCard
from ant.providers import MockLLMProvider, WorkerReasoner
from ant.tools import LocalSearchTool

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")


class LocalCoordinator:
    def __init__(
        self,
        repo_root: Path,
        workers: list[WorkerCard],
        reasoner: WorkerReasoner | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.workers = workers
        self.reasoner = reasoner or MockLLMProvider()

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

        return EvidenceState(
            question=question,
            evidence=evidence[:12],
            unresolved_needs=unresolved_needs,
            rounds=rounds,
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
            score = len(query_terms & terms)
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
