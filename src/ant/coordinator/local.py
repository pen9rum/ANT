from __future__ import annotations

from pathlib import Path

from ant.domain import EvidenceState, UnresolvedNeed, WorkerCard
from ant.tools import LocalSearchTool


class LocalCoordinator:
    def __init__(self, repo_root: Path, workers: list[WorkerCard]) -> None:
        self.repo_root = repo_root
        self.workers = workers

    def ask(self, question: str) -> EvidenceState:
        selected = self._select_workers(question)
        evidence = []
        search = LocalSearchTool(self.repo_root)
        for worker in selected:
            evidence.extend(search.search(question, worker.files, limit=4))

        needs = []
        if not evidence:
            needs.append(
                UnresolvedNeed(
                    description="No local evidence matched the question.",
                    suggested_terms=question.split()[:6],
                    suggested_territories=[worker.territory_id for worker in self.workers[:5]],
                )
            )

        return EvidenceState(question=question, evidence=evidence[:12], unresolved_needs=needs)

    def _select_workers(self, question: str, limit: int = 3) -> list[WorkerCard]:
        query_terms = {term.lower() for term in question.split() if len(term) > 2}
        scored: list[tuple[int, WorkerCard]] = []
        for worker in self.workers:
            terms = set(worker.searchable_terms) | {worker.root.lower(), worker.name.lower()}
            score = len(query_terms & terms)
            scored.append((score, worker))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [worker for score, worker in scored[:limit] if score > 0]
        return selected or self.workers[:limit]
