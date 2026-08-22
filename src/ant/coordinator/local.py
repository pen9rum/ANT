from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from ant.domain import (
    AbsenceProof,
    Evidence,
    EvidenceState,
    RecruitmentRound,
    TokenUsage,
    UnresolvedNeed,
    WorkerAction,
    WorkerCard,
    WorkerRoutingScore,
)
from ant.memory import MemoryRoute
from ant.providers import AnswerSynthesizer, MockLLMProvider, UsageReporter, WorkerReasoner
from ant.tools import LocalSearchTool
from ant.tools.local import STOP_WORDS
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
        memory_routes: list[MemoryRoute] | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.workers = workers
        self.reasoner = reasoner or (
            cast(WorkerReasoner, synthesizer) if synthesizer is not None else MockLLMProvider()
        )
        self.synthesizer = synthesizer
        self.memory_routes = memory_routes or []

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

            unresolved_needs = _normalize_coverage_needs(question, round_needs, self.workers)
            pending = [need for need in round_needs if need.suggested_terms or need.description]
            if not pending:
                break
            active_need = pending[0]
            query = self._query_from_needs(question, [active_need])

        unresolved_needs.extend(_coverage_needs(question, evidence, unresolved_needs, self.workers))

        if not evidence and not unresolved_needs:
            unresolved_needs.append(
                UnresolvedNeed(
                    description="No local evidence matched the question in the selected workers.",
                    kind="coverage_gap",
                    need_type=(
                        "implementation_location"
                        if _asks_for_source_implementation_text(question)
                        else "negative_presence"
                    ),
                    missing="Grounded evidence for the requested symbol, behavior, or absence.",
                    scope="unknown",
                    relevant_symbols=sorted(_relevant_symbols(question)),
                    suggested_terms=[
                        term
                        for term in TOKEN_RE.findall(question)
                        if term.lower() not in STOP_WORDS
                    ][:8],
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
            absence_proofs=_absence_proofs(question, rounds, unresolved_needs, self.workers),
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
        relevant_symbols = _relevant_symbols(query_text, need)
        implementation_intent = _asks_for_source_implementation_text(query_text)
        scored: list[tuple[WorkerRoutingScore, WorkerCard]] = []
        for worker in self.workers:
            terms = _worker_terms(worker)
            query_hits = sorted(
                query_term for query_term in query_terms if _matches_term(query_term, terms)
            )
            suggested_term_hits = sorted(
                term for term in suggested_terms if _matches_term(term, terms)
            )
            relevant_symbol_hits = sorted(
                symbol for symbol in relevant_symbols if _matches_symbol(symbol, worker)
            )
            score = len(query_hits) + len(suggested_term_hits) + (len(relevant_symbol_hits) * 6)
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
            if implementation_intent:
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
            if score > 0 or implementation_intent:
                source_path_bonus = _source_path_bonus(worker)
                score += source_path_bonus
            memory_route_bonus = _memory_route_bonus(worker, query_terms, self.memory_routes)
            score += memory_route_bonus
            scored.append(
                (
                    WorkerRoutingScore(
                        worker_id=worker.id,
                        territory_id=worker.territory_id,
                        final_score=score,
                        query_hits=query_hits,
                        suggested_term_hits=suggested_term_hits,
                        relevant_symbol_hits=relevant_symbol_hits,
                        territory_hint_score=territory_hint_score,
                        source_worker_bonus=source_worker_bonus,
                        source_path_bonus=source_path_bonus,
                        memory_route_bonus=memory_route_bonus,
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
    for term in terms:
        if query_term == term:
            return True
        if query_term in _compound_parts(term):
            return True
    return False


def _compound_parts(term: str) -> set[str]:
    if "_" not in term:
        return set()
    return {part for part in term.split("_") if len(part) > 2}


def _need_query_text(need: UnresolvedNeed | None) -> str:
    if need is None:
        return ""
    return " ".join(
        part
        for part in [
            need.missing,
            need.description,
            "" if need.need_type == "unknown" else need.need_type,
            " ".join(need.relevant_symbols),
            " ".join(need.suggested_terms),
            " ".join(need.suggested_territories),
        ]
        if part
    )


def _term_set(text: str) -> set[str]:
    return {
        term.lower()
        for term in TOKEN_RE.findall(text)
        if len(term) > 2 and term.lower() not in STOP_WORDS
    }


def _relevant_symbols(text: str, need: UnresolvedNeed | None = None) -> set[str]:
    symbols = set(need.relevant_symbols if need else [])
    for token in TOKEN_RE.findall(text):
        if "_" in token or any(character.isupper() for character in token):
            symbols.add(token)
    return {symbol for symbol in symbols if len(symbol) > 2}


def _matches_symbol(symbol: str, worker: WorkerCard) -> bool:
    lowered = symbol.lower()
    worker_terms = {term.lower() for term in worker.searchable_terms}
    if symbol in worker.searchable_terms or lowered in worker_terms:
        return True
    for owned_symbol in worker.symbols:
        names = {
            owned_symbol.name,
            owned_symbol.qualname,
            *owned_symbol.bases,
        }
        if symbol in names or lowered in {name.lower() for name in names if name}:
            return True
    return any(lowered in file.replace("\\", "/").lower() for file in worker.files)


def _worker_terms(worker: WorkerCard) -> set[str]:
    terms = _term_set(" ".join([*worker.searchable_terms, worker.root, worker.name]))
    terms |= _term_set(
        " ".join(
            [
                term
                for symbol in worker.symbols
                for term in [symbol.name, symbol.qualname, symbol.kind, *symbol.bases]
                if term
            ]
        )
    )
    terms |= {
        token.lower()
        for file in worker.files
        for token in TOKEN_RE.findall(file)
        if len(token) > 2
    }
    return terms


def _memory_route_bonus(
    worker: WorkerCard,
    query_terms: set[str],
    routes: list[MemoryRoute],
) -> int:
    bonus = 0
    for route in routes:
        route_terms = {term.lower() for term in route.need_terms}
        if worker.id not in route.worker_ids or not (query_terms & route_terms):
            continue
        bonus = max(bonus, min(12, round(route.weight * 4) + len(query_terms & route_terms)))
    return bonus


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
    return _asks_for_source_implementation_text(_need_query_text(need))


def _asks_for_source_implementation_text(text: str) -> bool:
    text = text.lower()
    indicators = [
        "implementation",
        "implemented",
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
    test_like = sum(
        1
        for file in files
        if {"test", "tests"} & set(file.replace("\\", "/").lower().split("/"))
    )
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


def _coverage_needs(
    question: str,
    evidence: list[Evidence],
    existing_needs: list[UnresolvedNeed],
    workers: list[WorkerCard],
) -> list[UnresolvedNeed]:
    needs = []
    existing_types = {need.need_type for need in existing_needs}
    question_symbols = sorted(_relevant_symbols(question))
    evidence_text = "\n".join(item.quote for item in evidence)
    if (
        "subclass_lookup" not in existing_types
        and _asks_for_inheritance_text(question)
        and not _has_subclass_evidence(evidence_text)
    ):
        needs.append(
            UnresolvedNeed(
                description=(
                    "Need subclass definitions that inherit from the requested base symbol."
                ),
                kind="coverage_gap",
                need_type="subclass_lookup",
                missing="Subclass definitions and their base-class relationship.",
                scope="unknown",
                relevant_symbols=question_symbols,
                suggested_terms=[*question_symbols, "subclass", "inherit"],
                suggested_territories=[worker.territory_id for worker in workers[:5]],
            )
        )
    if (
        "implementation_location" not in existing_types
        and _asks_for_source_implementation_text(question)
        and not _has_source_definition_evidence(evidence)
    ):
        needs.append(
            UnresolvedNeed(
                description="Need source implementation definitions, not only references or tests.",
                kind="coverage_gap",
                need_type="implementation_location",
                missing="Source code definitions that implement the requested behavior.",
                scope="unknown",
                relevant_symbols=question_symbols,
                suggested_terms=[*question_symbols, "implementation", "definition"],
                suggested_territories=[
                    worker.territory_id
                    for worker in workers
                    if any(has_source_part(file) for file in worker.files)
                ][:5],
            )
        )
    if _asks_for_source_test_relationship(question):
        has_source = _has_source_definition_evidence(evidence)
        has_test = any(_is_test_evidence(item) for item in evidence)
        if not (has_source and has_test) and "source_test_coalition" not in existing_types:
            missing_side = "source implementation" if not has_source else "test coverage"
            needs.append(
                UnresolvedNeed(
                    description=(
                        "Need evidence from both source and tests before making a coverage claim."
                    ),
                    kind="coverage_gap",
                    need_type="source_test_coalition",
                    missing=missing_side,
                    scope="cross_territory",
                    relevant_symbols=question_symbols,
                    suggested_terms=[*question_symbols, "test", "implementation"],
                    suggested_territories=[worker.territory_id for worker in workers[:5]],
                )
            )
    return needs


def _asks_for_source_test_relationship(question: str) -> bool:
    lowered = question.lower()
    mentions_test = any(word in lowered for word in ["test", "tests", "tested", "coverage"])
    return mentions_test and _asks_for_source_implementation_text(question)


def _is_test_evidence(item: Evidence) -> bool:
    parts = set(item.path.replace("\\", "/").lower().split("/"))
    return bool({"test", "tests"} & parts) or Path(item.path).name.startswith("test_")


def _normalize_coverage_needs(
    question: str,
    needs: list[UnresolvedNeed],
    workers: list[WorkerCard],
) -> list[UnresolvedNeed]:
    symbols = sorted(_relevant_symbols(question))
    territories = [worker.territory_id for worker in workers[:5]]
    normalized = []
    for need in needs:
        if need.kind != "missing_evidence":
            normalized.append(need)
            continue
        need_type = need.need_type
        if need_type == "unknown":
            need_type = (
                "implementation_location"
                if _asks_for_source_implementation_text(question)
                else "negative_presence"
            )
        normalized.append(
            need.model_copy(
                update={
                    "kind": "coverage_gap",
                    "need_type": need_type,
                    "relevant_symbols": need.relevant_symbols or symbols,
                    "suggested_territories": need.suggested_territories or territories,
                }
            )
        )
    return normalized


def _absence_proofs(
    question: str,
    rounds: list[RecruitmentRound],
    needs: list[UnresolvedNeed],
    workers: list[WorkerCard],
) -> list[AbsenceProof]:
    negative_needs = [
        need
        for need in needs
        if need.kind == "coverage_gap"
        and need.need_type in {"negative_presence", "implementation_location"}
    ]
    if not negative_needs:
        return []
    searched_workers = list(
        dict.fromkeys(worker_id for round_ in rounds for worker_id in round_.selected_worker_ids)
    )
    workers_by_id = {worker.id: worker for worker in workers}
    searched_paths = sorted(
        {
            path
            for worker_id in searched_workers
            if worker_id in workers_by_id
            for path in workers_by_id[worker_id].files
        }
    )
    tools = sorted(
        {
            action.tool
            for round_ in rounds
            for observation in round_.observations
            for action in observation.actions
        }
    )
    return [
        AbsenceProof(
            query=question,
            relevant_symbols=sorted(
                {symbol for need in negative_needs for symbol in need.relevant_symbols}
            ),
            searched_worker_ids=searched_workers,
            searched_territories=sorted(
                {
                    workers_by_id[item].territory_id
                    for item in searched_workers
                    if item in workers_by_id
                }
            ),
            searched_paths=searched_paths,
            tools=tools,
            exhaustive=bool(workers) and set(searched_workers) == set(workers_by_id),
            conclusion="not_found" if searched_workers else "inconclusive",
        )
    ]


def _asks_for_inheritance_text(text: str) -> bool:
    lowered = text.lower()
    return any(
        indicator in lowered
        for indicator in [
            "subclass",
            "subclasses",
            "inherit",
            "inherits",
            "inherited",
            "base class",
            "derived",
        ]
    )


def _has_subclass_evidence(text: str) -> bool:
    return bool(re.search(r"\bclass\s+[A-Za-z_][A-Za-z0-9_]*\([^)]*[A-Za-z_]", text))


def _has_source_definition_evidence(evidence: list[Evidence]) -> bool:
    return any(
        has_source_part(item.path)
        and not has_low_value_part(item.path)
        and ("class " in item.quote or "def " in item.quote)
        for item in evidence
    )


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
        peer_claims = sorted(
            {
                evidence.claim
                for evidence in prior_evidence
                if evidence.worker_id != observation.worker_id and evidence.claim
            }
        )
        peer_paths = sorted(
            {
                evidence.path
                for evidence in prior_evidence
                if evidence.worker_id != observation.worker_id
            }
        )
        query = ", ".join(peer_claims[:6]) if peer_claims else ", ".join(peer_paths[:6])
        observation.actions.append(
            WorkerAction(
                tool="cross_check",
                query=query,
                result_count=len(peer_claims) or len(peer_paths),
                rationale="One-pass coalition cross-check against peer evidence claims.",
            )
        )
