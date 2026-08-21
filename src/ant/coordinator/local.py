from __future__ import annotations

import re
from pathlib import Path

from ant.domain import (
    Evidence,
    EvidenceState,
    RecruitmentRound,
    TokenUsage,
    UnresolvedNeed,
    WorkerAction,
    WorkerCard,
)
from ant.providers import AnswerSynthesizer, MockLLMProvider, UsageReporter, WorkerReasoner
from ant.tools import LocalSearchTool
from ant.workers import AutonomousWorker, WorkerRunConfig

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
        worker_config = WorkerRunConfig(max_tool_calls=10, evidence_limit=8)

        for round_index in range(max_rounds):
            selected = self._select_workers(query, seen_worker_ids=seen_worker_ids)
            if not selected:
                break

            observations = []
            for worker in selected:
                observation = AutonomousWorker(self.repo_root, worker, search).run(
                    query,
                    config=worker_config,
                )
                worker_evidence = observation.evidence
                evidence.extend(worker_evidence)
                reasoner_observation = self.reasoner.observe(
                    question=query,
                    worker_id=worker.id,
                    territory_id=worker.territory_id,
                    evidence_count=len(worker_evidence),
                )
                observation.unresolved_needs.extend(reasoner_observation.unresolved_needs)
                observations.append(observation)
                unresolved_needs.extend(observation.unresolved_needs)
                seen_worker_ids.add(worker.id)

            if len(observations) > 1:
                _add_coalition_cross_checks(observations)

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
        evidence = _rank_global_evidence(evidence, question)[:12]
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
            evidence=evidence,
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


def _rank_global_evidence(evidence: list[Evidence], question: str) -> list[Evidence]:
    terms = {term.lower() for term in TOKEN_RE.findall(question) if len(term) > 2}

    def score(item: Evidence) -> int:
        quote = item.quote.lower()
        path = item.path.replace("\\", "/")
        value = 0
        if "class " in quote:
            value += 10
        if "def " in quote:
            value += 8
        if path.startswith("src/"):
            value += 5
        if path.endswith(".py"):
            value += 3
        if path.endswith("setup.py") or path.endswith("README.md") or path.startswith("examples/"):
            value -= 10
        value += sum(1 for term in terms if term in quote)
        return value

    return sorted(evidence, key=score, reverse=True)


def _last_coalition_workers(rounds: list[RecruitmentRound]) -> list[str]:
    if not rounds:
        return []
    return rounds[-1].selected_worker_ids


def _add_coalition_cross_checks(observations) -> None:
    for observation in observations:
        peer_paths = sorted(
            {
                evidence.path
                for peer in observations
                if peer.worker_id != observation.worker_id
                for evidence in peer.evidence
            }
        )
        observation.actions.append(
            WorkerAction(
                tool="cross_check",
                query=", ".join(peer_paths[:6]),
                result_count=len(peer_paths),
                rationale="One-pass coalition cross-check against peer evidence paths.",
            )
        )
