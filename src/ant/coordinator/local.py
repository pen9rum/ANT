from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from ant.domain import (
    Evidence,
    EvidenceState,
    RecruitmentRound,
    TokenUsage,
    UnresolvedNeed,
    WorkerAction,
    WorkerCard,
    WorkerRoutingScore,
)
from ant.providers import AnswerSynthesizer, MockLLMProvider, UsageReporter, WorkerReasoner
from ant.tools import LocalSearchTool
from ant.tools.path_prior import has_low_value_part, has_source_part, is_low_value_path
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
        self.reasoner = reasoner or (
            cast(WorkerReasoner, synthesizer) if synthesizer is not None else MockLLMProvider()
        )
        self.synthesizer = synthesizer

    def ask(self, question: str, max_rounds: int = 2) -> EvidenceState:
        evidence: list[Evidence] = []
        unresolved_needs: list[UnresolvedNeed] = []
        rounds: list[RecruitmentRound] = []
        seen_worker_ids: set[str] = set()
        query = question
        active_need: UnresolvedNeed | None = None
        search = LocalSearchTool(self.repo_root)
        worker_config = WorkerRunConfig(max_tool_calls=10, evidence_limit=8)

        for round_index in range(max_rounds):
            ranked = self._rank_worker_scores(query, seen_worker_ids, active_need)
            candidates = [worker for _, worker in ranked]
            routing_scores = [score for score, _ in ranked[:10]]
            selected = (
                self._initial_workers(candidates, question)
                if active_need is None
                else candidates[:1]
            )
            if not selected:
                break

            observations = []
            round_needs: list[UnresolvedNeed] = []
            for worker in selected:
                observation = AutonomousWorker(self.repo_root, worker, search).run(
                    query,
                    config=worker_config,
                )
                worker_evidence = [
                    item.model_copy(update={"worker_id": worker.id})
                    for item in observation.evidence
                ]
                observation.evidence = worker_evidence
                evidence.extend(worker_evidence)
                reasoner_observation = self.reasoner.observe(
                    question=question,
                    worker_id=worker.id,
                    territory_id=worker.territory_id,
                    evidence=worker_evidence,
                )
                observation.unresolved_needs.extend(reasoner_observation.unresolved_needs)
                observations.append(observation)
                round_needs.extend(observation.unresolved_needs)
                seen_worker_ids.add(worker.id)

            coalition_formed = bool(
                active_need
                and active_need.scope == "cross_territory"
                and any(worker.id != active_need.source_worker_id for worker in selected)
                and any(item.evidence for item in observations)
            )
            if coalition_formed:
                _add_coalition_cross_checks(observations, evidence)

            rounds.append(
                RecruitmentRound(
                    round_index=round_index,
                    query=query,
                    input_need=active_need.description if active_need else "",
                    candidate_worker_ids=[worker.id for worker in candidates],
                    routing_scores=routing_scores,
                    selected_worker_ids=[worker.id for worker in selected],
                    rationale="Selected workers by overlap with query terms and worker card terms.",
                    selection_reason=(
                        (
                            "Continued the best local worker for a semantic need."
                            if active_need and selected[0].id == active_need.source_worker_id
                            else "Recruited the best worker for a semantic need."
                        )
                        if active_need
                        else "Initial focused recruitment from the user question."
                    ),
                    coalition_formed=coalition_formed,
                    coalition_reason=(
                        "Cross-territory follow-up evidence was related to prior evidence."
                        if coalition_formed
                        else ""
                    ),
                    observations=observations,
                )
            )

            unresolved_needs = round_needs
            pending = [need for need in round_needs if need.suggested_terms or need.description]
            if not pending:
                break
            active_need = pending[0]
            query = self._query_from_needs(question, [active_need])

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
            if coalition_workers:
                answer = self.synthesizer.synthesize_coalition(
                    question=question,
                    worker_ids=coalition_workers,
                    evidence=evidence[:12],
                )
            else:
                answer = self.synthesizer.synthesize(question=question, evidence=evidence[:12])
        usage = (
            self.synthesizer.drain_usage() if isinstance(self.synthesizer, UsageReporter) else None
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
        return self._rank_workers(question, seen_worker_ids or set(), None)[:limit]

    def _rank_workers(
        self,
        question: str,
        seen_worker_ids: set[str],
        need: UnresolvedNeed | None,
    ) -> list[WorkerCard]:
        return [worker for _, worker in self._rank_worker_scores(question, seen_worker_ids, need)]

    def _rank_worker_scores(
        self,
        question: str,
        seen_worker_ids: set[str],
        need: UnresolvedNeed | None,
    ) -> list[tuple[WorkerRoutingScore, WorkerCard]]:
        query_text = _need_query_text(need) if need else question
        query_terms = _term_set(query_text)
        suggested_terms = _term_set(" ".join(need.suggested_terms)) if need else set()
        scored: list[tuple[WorkerRoutingScore, WorkerCard]] = []
        for worker in self.workers:
            terms = _worker_terms(worker)
            query_hits = sorted(
                query_term for query_term in query_terms if _matches_term(query_term, terms)
            )
            suggested_term_hits = sorted(
                term for term in suggested_terms if _matches_term(term, terms)
            )
            score = len(query_hits)
            source_worker_bonus = 0
            if need and worker.id == need.source_worker_id:
                local_overlap = len(suggested_term_hits) or len(query_hits)
                source_worker_bonus = 8 if need.scope == "local" else 1
                source_worker_bonus += min(local_overlap, 4)
                score += source_worker_bonus
            territory_hint_score = 0
            test_path_penalty = 0
            if need:
                territory_hint_score = _territory_hint_score(worker, need.suggested_territories)
                score += territory_hint_score
                if need.scope == "cross_territory" and _asks_for_source_implementation(need):
                    test_path_penalty = _test_path_penalty(worker)
                    score -= test_path_penalty
            seen_worker_penalty = 0
            if worker.id in seen_worker_ids:
                if need and worker.id == need.source_worker_id:
                    seen_worker_penalty = 2 if need.scope == "local" else 8
                elif need is None:
                    seen_worker_penalty = 4
                score -= seen_worker_penalty
            source_path_bonus = 0
            if score > 0:
                source_path_bonus = _source_path_bonus(worker)
                score += source_path_bonus
            scored.append(
                (
                    WorkerRoutingScore(
                        worker_id=worker.id,
                        territory_id=worker.territory_id,
                        final_score=score,
                        query_hits=query_hits,
                        suggested_term_hits=suggested_term_hits,
                        territory_hint_score=territory_hint_score,
                        source_worker_bonus=source_worker_bonus,
                        source_path_bonus=source_path_bonus,
                        test_path_penalty=test_path_penalty,
                        seen_worker_penalty=seen_worker_penalty,
                    ),
                    worker,
                )
            )
        scored.sort(key=lambda item: item[0].final_score, reverse=True)
        selected = [(score, worker) for score, worker in scored if score.final_score > 0]
        fallback = list(self.workers)
        return selected or [
            (
                WorkerRoutingScore(
                    worker_id=worker.id,
                    territory_id=worker.territory_id,
                    final_score=0,
                ),
                worker,
            )
            for worker in fallback
        ]

    def _initial_workers(self, candidates: list[WorkerCard], question: str) -> list[WorkerCard]:
        if len(candidates) < 2:
            return candidates[:1]
        first, second = candidates[:2]
        first_score = self._worker_score(first, question)
        second_score = self._worker_score(second, question)
        genuinely_close = first_score > 0 and second_score >= first_score * 0.9
        complementary = first.territory_id != second.territory_id
        return candidates[:2] if genuinely_close and complementary else candidates[:1]

    @staticmethod
    def _worker_score(worker: WorkerCard, question: str) -> int:
        query_terms = {term.lower() for term in TOKEN_RE.findall(question) if len(term) > 2}
        terms = set(worker.searchable_terms) | {worker.root.lower(), worker.name.lower()}
        terms |= {
            token.lower()
            for file in worker.files
            for token in TOKEN_RE.findall(file)
            if len(token) > 2
        }
        return sum(1 for term in query_terms if _matches_term(term, terms))

    @staticmethod
    def _query_from_needs(question: str, needs: list[UnresolvedNeed]) -> str:
        parts = []
        for need in needs:
            parts.append(_need_query_text(need))
        query = " ".join(part for part in parts if part).strip()
        return query or question


def _matches_term(query_term: str, terms: set[str]) -> bool:
    return any(query_term in term or term in query_term for term in terms)


def _need_query_text(need: UnresolvedNeed | None) -> str:
    if need is None:
        return ""
    return " ".join(
        part
        for part in [
            need.missing,
            need.description,
            " ".join(need.suggested_terms),
            " ".join(need.suggested_territories),
        ]
        if part
    )


def _term_set(text: str) -> set[str]:
    return {term.lower() for term in TOKEN_RE.findall(text) if len(term) > 2}


def _worker_terms(worker: WorkerCard) -> set[str]:
    terms = _term_set(" ".join([*worker.searchable_terms, worker.root, worker.name]))
    terms |= {
        token.lower()
        for file in worker.files
        for token in TOKEN_RE.findall(file)
        if len(token) > 2
    }
    return terms


def _territory_hint_score(worker: WorkerCard, hints: list[str]) -> int:
    if not hints:
        return 0
    worker_terms = _worker_terms(worker)
    territory_terms = _term_set(" ".join([worker.territory_id, worker.root, worker.name]))
    score = 0
    for hint in hints:
        hint_terms = _term_set(hint)
        if not hint_terms:
            continue
        overlap = sum(
            1
            for hint_term in hint_terms
            if _matches_term(hint_term, territory_terms)
            or _matches_term(hint_term, worker_terms)
        )
        if overlap == len(hint_terms):
            score += 14
        elif overlap:
            score += 6 + (overlap * 2)
    return score


def _asks_for_source_implementation(need: UnresolvedNeed) -> bool:
    text = _need_query_text(need).lower()
    indicators = [
        "implementation",
        "code path",
        "source code",
        "code location",
        "definition",
        "defines",
        "function",
        "class",
        "module",
    ]
    return any(indicator in text for indicator in indicators)


def _test_path_penalty(worker: WorkerCard) -> int:
    files = worker.files or [worker.root]
    test_like = sum(1 for file in files if "test" in file.replace("\\", "/").lower().split("/"))
    if not files:
        return 0
    return 20 if test_like / len(files) >= 0.5 else 0


def _source_path_bonus(worker: WorkerCard) -> int:
    files = worker.files or [worker.root]
    source_like = sum(1 for file in files if has_source_part(file))
    if not files:
        return 0
    return 2 if source_like / len(files) >= 0.5 else 0


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
        if is_low_value_path(path):
            value -= 10
        if has_low_value_part(path):
            value -= 10
        value += sum(1 for term in terms if term in quote)
        return value

    return sorted(evidence, key=score, reverse=True)


def _last_coalition_workers(rounds: list[RecruitmentRound]) -> list[str]:
    for round_ in reversed(rounds):
        if round_.coalition_formed:
            prior = [
                worker_id
                for earlier in rounds[: round_.round_index]
                for worker_id in earlier.selected_worker_ids
            ]
            return list(dict.fromkeys([*prior, *round_.selected_worker_ids]))
    return []


def _add_coalition_cross_checks(observations, prior_evidence: list[Evidence]) -> None:
    for observation in observations:
        peer_paths = sorted(
            {
                evidence.path
                for evidence in prior_evidence
                if evidence.worker_id != observation.worker_id
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
