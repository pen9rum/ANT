from __future__ import annotations

import re
from pathlib import Path
from typing import cast

from ant.domain import (
    AbsenceProof,
    Evidence,
    EvidenceState,
    NeedResolution,
    RecruitmentRound,
    TokenUsage,
    UnresolvedNeed,
    WorkerAction,
    WorkerCard,
    WorkerObservation,
    WorkerRoutingScore,
)
from ant.memory import MemoryRoute
from ant.providers import AnswerSynthesizer, MockLLMProvider, UsageReporter, WorkerReasoner
from ant.retrieval import STOP_WORDS, TOKEN_RE, extract_terms, is_stem_match, score_evidence
from ant.retrieval.dense import WORKER_CARDS_KEY, EmbeddingIndex, get_shared_embedder
from ant.scoring_config import DEFAULT_SCORING_CONFIG
from ant.tools import LocalSearchTool
from ant.tools.path_prior import has_low_value_part, has_source_part
from ant.workers import AutonomousWorker, WorkerRunConfig

BASE_CLASS_RE = re.compile(r"\bclass\s+[A-Za-z_][A-Za-z0-9_]*\(([^)]*)\)")

# need_type values that a heuristic evidence match is allowed to close outright.
# Absence-shaped needs (negative_presence, unknown) are deliberately excluded:
# not finding a symbol is never proof it does not exist, so closing those on a
# heuristic match would let synthesis assert a false absence.
_CLOSABLE_BY_EVIDENCE = {"subclass_lookup", "implementation_location", "source_test_coalition"}

# Consecutive rounds an active need may go with no new evidence and no
# resolution progress before escalation (a qualitatively different tactic,
# not just "pick another worker again") kicks in. 2 rather than 1: a single
# quiet round is common and not yet evidence of being stuck, only a repeat
# of it is.
_STUCK_THRESHOLD = 2


class LocalCoordinator:
    def __init__(
        self,
        repo_root: Path,
        workers: list[WorkerCard],
        reasoner: WorkerReasoner | None = None,
        synthesizer: AnswerSynthesizer | None = None,
        memory_routes: list[MemoryRoute] | None = None,
        index_path: Path | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.workers = workers
        self.reasoner = reasoner or (
            cast(WorkerReasoner, synthesizer) if synthesizer is not None else MockLLMProvider()
        )
        self.synthesizer = synthesizer
        self.memory_routes = memory_routes or []
        self.index_path = index_path
        self._worker_card_index: EmbeddingIndex | None = None
        self._worker_card_index_loaded = False

    def ask(self, question: str, max_rounds: int = 6) -> EvidenceState:
        # Hard safety ceiling only, not the intended stopping point: whether
        # to actually use another round is now the reasoner's
        # should_continue_recruiting() call each iteration (see below),
        # made fresh from the current need/evidence rather than a fixed
        # round-count cutoff. Raised from the old 2 (which used to *be* the
        # real stopping rule) since it now only guards against a
        # pathological "always says continue" loop.
        evidence: list[Evidence] = []
        accumulated_needs: list[UnresolvedNeed] = []
        rounds: list[RecruitmentRound] = []
        seen_worker_ids: set[str] = set()
        query = question
        active_need: UnresolvedNeed | None = None
        search = LocalSearchTool(self.repo_root, index_path=self.index_path)
        worker_config = WorkerRunConfig(max_tool_calls=11)
        # Per-need-identity count of consecutive rounds with neither new
        # evidence nor resolution progress; a need whose streak reaches
        # _STUCK_THRESHOLD is escalated (see _escalate_stuck_need) instead
        # of retried the normal way. Cleared whenever a need makes
        # progress, so a need that stalls, gets set aside for a different
        # one, and later becomes active again starts its streak fresh
        # rather than inheriting stale history.
        no_progress_streak: dict[tuple[str, str], int] = {}
        # Needs every escalation tactic already failed on: kept in
        # accumulated_needs (so they still surface in the final
        # unresolved_needs report) but never picked as active_need again --
        # retrying forever is exactly the stuck loop this exists to avoid.
        abandoned_need_keys: set[tuple[str, str]] = set()
        # Which worker last attempted a given local-scope need, and whether
        # that attempt made progress: lets a local need's *second and
        # later* attempt go through decide_local_action (continue this
        # worker / handoff / coalition / escalate) instead of the routing
        # score alone incidentally re-picking (or not re-picking) the same
        # worker. A need's first attempt has no prior progress to judge, so
        # it always goes through the normal path below.
        local_last_worker: dict[tuple[str, str], WorkerCard] = {}
        local_last_progress: dict[tuple[str, str], bool] = {}

        for round_index in range(max_rounds):
            # Snapshot before this round's workers run: a worker re-run on
            # an unchanged query deterministically re-finds the same
            # evidence it found last round, so "evidence.extend() grew"
            # cannot be the progress signal -- only content genuinely new
            # to the pool counts. Compared against the *evidence* list's
            # own dedupe key (see _dedupe_evidence) after the round, not a
            # bare count.
            pre_round_evidence_keys = {_evidence_key(item) for item in evidence}
            ranked = self._rank_worker_scores(query, seen_worker_ids, active_need)
            candidates = [worker for _, worker in ranked]
            routing_scores = [score for score, _ in ranked[:10]]
            escalation_used = ""
            local_action = ""
            if active_need is None:
                selected = self._initial_workers(candidates, question)
                observations, round_needs = self._run_selected_workers(
                    selected, query, question, search, worker_config, evidence, seen_worker_ids
                )
            elif no_progress_streak.get(_need_identity(active_need), 0) >= _STUCK_THRESHOLD:
                escalation_used, selected, observations, round_needs = self._escalate_stuck_need(
                    question=question,
                    active_need=active_need,
                    seen_worker_ids=seen_worker_ids,
                    evidence=evidence,
                    search=search,
                    worker_config=worker_config,
                )
            elif active_need.scope == "local" and _need_identity(active_need) in local_last_worker:
                need_key = _need_identity(active_need)
                (
                    local_action,
                    escalation_used,
                    selected,
                    observations,
                    round_needs,
                ) = self._decide_and_run_local_action(
                    active_need=active_need,
                    last_worker=local_last_worker[need_key],
                    made_progress_last_time=local_last_progress.get(need_key, False),
                    candidates=candidates,
                    query=query,
                    question=question,
                    search=search,
                    worker_config=worker_config,
                    evidence=evidence,
                    seen_worker_ids=seen_worker_ids,
                )
            else:
                selected = _select_for_active_need(candidates, active_need, seen_worker_ids)
                observations, round_needs = self._run_selected_workers(
                    selected, query, question, search, worker_config, evidence, seen_worker_ids
                )
            if not selected and not escalation_used:
                break

            coalition_formed = (
                escalation_used in ("third_worker", "temporary_bridge")
                or local_action == "coalition"
                or bool(
                    active_need
                    and not escalation_used
                    and not local_action
                    and active_need.scope == "cross_territory"
                    and any(worker.id != active_need.source_worker_id for worker in selected)
                    and any(item.evidence for item in observations)
                )
            )
            if coalition_formed and not escalation_used:
                _add_coalition_cross_checks(observations, evidence)
                joint_observation = _run_coalition_cross_check(
                    reasoner=self.reasoner,
                    question=question,
                    selected=selected,
                    evidence=evidence,
                    search=search,
                )
                if joint_observation is not None:
                    observations.append(joint_observation)
                    evidence.extend(joint_observation.evidence)
                    round_needs.extend(joint_observation.unresolved_needs)

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
                        f"Escalated a stuck need via {escalation_used}."
                        if escalation_used
                        else (
                            f"Local continuation decision: {local_action}."
                            if local_action
                            else (
                                (
                                    "Continued the best local worker for a semantic need."
                                    if active_need
                                    and selected[0].id == active_need.source_worker_id
                                    else "Recruited the best worker for a semantic need."
                                )
                                if active_need
                                else "Initial focused recruitment from the user question."
                            )
                        )
                    ),
                    coalition_formed=coalition_formed,
                    coalition_reason=(
                        "Cross-territory follow-up evidence was related to prior evidence."
                        if coalition_formed
                        else ""
                    ),
                    observations=observations,
                    escalation_used=escalation_used,
                )
            )

            # Resolution check + progress tracking for the need that drove
            # this round, generalizing the old 3-need_type-only heuristic
            # closure (_close_resolved_needs below, still run first as a
            # cheap bulk pass) to every need_type via real judgment instead
            # of silently never closing -- or re-raising verbatim forever --
            # the rest.
            resolution: NeedResolution | None = None
            active_key: tuple[str, str] | None = None
            if active_need is not None:
                active_key = _need_identity(active_need)
                new_round_evidence = [
                    item
                    for obs in observations
                    for item in obs.evidence
                    if _evidence_key(item) not in pre_round_evidence_keys
                ]
                resolution = self.reasoner.check_need_resolution(
                    need=active_need, new_evidence=new_round_evidence, question=question
                )
                made_progress = bool(new_round_evidence) or resolution.status != "unresolved"
                if made_progress:
                    no_progress_streak.pop(active_key, None)
                elif escalation_used:
                    # Every escalation tactic ran and none surfaced anything --
                    # this need is genuinely stuck, not just slow.
                    abandoned_need_keys.add(active_key)
                else:
                    no_progress_streak[active_key] = no_progress_streak.get(active_key, 0) + 1
                if active_need.scope == "local" and selected:
                    # Feeds decide_local_action's *next* call for this need:
                    # which worker just attempted it, and whether that
                    # attempt actually went anywhere.
                    local_last_worker[active_key] = selected[0]
                    local_last_progress[active_key] = made_progress

            round_normalized = _normalize_coverage_needs(question, round_needs, self.workers)
            accumulated_needs = _merge_needs(accumulated_needs, round_normalized)
            # Re-check every open need against the evidence accumulated so far
            # (across all rounds, not just this one) so a need raised early is
            # dropped as soon as later evidence actually resolves it.
            accumulated_needs = _close_resolved_needs(accumulated_needs, evidence, question)
            if resolution is not None and active_key is not None:
                if resolution.status == "resolved":
                    accumulated_needs = [
                        need for need in accumulated_needs if _need_identity(need) != active_key
                    ]
                elif resolution.status == "partial" and resolution.refined_need is not None:
                    accumulated_needs = [
                        need for need in accumulated_needs if _need_identity(need) != active_key
                    ]
                    accumulated_needs.append(resolution.refined_need)

            pending = [
                need
                for need in accumulated_needs
                if (need.suggested_terms or need.description)
                and _need_identity(need) not in abandoned_need_keys
            ]
            if not pending:
                break
            active_need = pending[0]
            # max_rounds is a safety ceiling, not the intended stopping
            # point (see ask()'s default): whether another round is worth
            # it -- vs. the colony genuinely having nothing better left to
            # try for this need -- is the reasoner's call, made fresh each
            # round rather than fixed at a round-count cutoff.
            if not self.reasoner.should_continue_recruiting(
                question=question,
                need=active_need,
                evidence=evidence,
                rounds_completed=round_index + 1,
            ):
                break
            query = self._query_from_needs(question, [active_need])

        # Inheritance is a global structural fact, not a territory-scoped one:
        # recruitment routing only ever proves "this worker's own files have
        # no more subclasses", never "the repository has no more subclasses
        # elsewhere". Run one definitive scan across every indexed file so
        # completeness claims are actually true instead of hedged.
        inheritance_evidence, inheritance_proof = _verify_inheritance_completeness(
            question, self.workers, search
        )
        if inheritance_evidence:
            evidence = _dedupe_evidence([*evidence, *inheritance_evidence])

        unresolved_needs = accumulated_needs
        unresolved_needs = unresolved_needs + _coverage_needs(
            question, evidence, unresolved_needs, self.workers
        )
        unresolved_needs = _close_resolved_needs(unresolved_needs, evidence, question)

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

        absence_proofs = _absence_proofs(question, rounds, unresolved_needs, self.workers)
        if inheritance_proof is not None:
            absence_proofs.append(inheritance_proof)

        answer = ""
        ranked_evidence = _rank_global_evidence(evidence, question)
        evidence = _select_evidence(self.reasoner, question, ranked_evidence, search)
        if self.synthesizer and evidence:
            coalition_workers = _last_coalition_workers(rounds)
            if coalition_workers:
                answer = self.synthesizer.synthesize_coalition(
                    question=question,
                    worker_ids=coalition_workers,
                    evidence=evidence,
                    absence_proofs=absence_proofs,
                )
            else:
                answer = self.synthesizer.synthesize(
                    question=question, evidence=evidence, absence_proofs=absence_proofs
                )
        usage = (
            self.synthesizer.drain_usage() if isinstance(self.synthesizer, UsageReporter) else None
        )

        return EvidenceState(
            question=question,
            answer=answer,
            evidence=evidence,
            unresolved_needs=unresolved_needs,
            rounds=rounds,
            absence_proofs=absence_proofs,
            usage=usage if isinstance(usage, TokenUsage) else TokenUsage(),
        )

    def _run_selected_workers(
        self,
        selected: list[WorkerCard],
        query: str,
        question: str,
        search: LocalSearchTool,
        worker_config: WorkerRunConfig,
        evidence: list[Evidence],
        seen_worker_ids: set[str],
    ) -> tuple[list[WorkerObservation], list[UnresolvedNeed]]:
        """Runs each selected worker in turn, mutating `evidence` and
        `seen_worker_ids` in place as it goes rather than batching updates
        until the end, so each subsequent worker's own observe() call --
        and an escalation tactic reusing this helper later in the same
        task -- sees what earlier workers already found this round, exactly
        as the original inline round loop did.
        """
        observations: list[WorkerObservation] = []
        round_needs: list[UnresolvedNeed] = []
        for worker in selected:
            observation = AutonomousWorker(
                self.repo_root, worker, search, reasoner=self.reasoner
            ).run(
                query,
                config=worker_config,
            )
            worker_evidence = [
                item.model_copy(update={"worker_id": worker.id}) for item in observation.evidence
            ]
            observation.evidence = worker_evidence
            evidence.extend(worker_evidence)
            # Give the reasoner the full shared evidence state, not just this
            # worker's own findings, so it does not re-raise a need another
            # worker already grounded (e.g. worker B finding the subclass
            # worker A's need was about).
            reasoner_context = _dedupe_evidence([*worker_evidence, *evidence])
            reasoner_observation = self.reasoner.observe(
                question=question,
                worker_id=worker.id,
                territory_id=worker.territory_id,
                evidence=reasoner_context,
            )
            observation.unresolved_needs.extend(reasoner_observation.unresolved_needs)
            observations.append(observation)
            round_needs.extend(observation.unresolved_needs)
            seen_worker_ids.add(worker.id)
        return observations, round_needs

    def _escalate_stuck_need(
        self,
        *,
        question: str,
        active_need: UnresolvedNeed,
        seen_worker_ids: set[str],
        evidence: list[Evidence],
        search: LocalSearchTool,
        worker_config: WorkerRunConfig,
    ) -> tuple[str, list[WorkerCard], list[WorkerObservation], list[UnresolvedNeed]]:
        """Tactics that are qualitatively different from normal recruitment
        (not just "pick another worker and hope"), tried in order, stopping
        at the first one that surfaces new evidence:

        1. Widen the candidate pool past the LLM-routing prefilter -- a
           worker with a real answer but a low lexical/dense score never
           even enters the top-K pool select_workers() judges from during
           normal recruitment.
        2. Recruit a third complementary worker and cross-check jointly
           with everyone already tried for this need -- a real 3-way
           coalition, not another 1:1 handoff.
        3. An ephemeral, in-task-only "bridge" worker whose territory is
           the union of every worker already tried, framed specifically
           around the interface between them -- never persisted to colony
           state (that is evolve_workers' job, at a slow cross-task
           timescale, not a single stuck task's).
        4. Last resort: search the whole indexed repository, unscoped by
           any single worker's territory.

        Returns (tactic name, or "" if every tactic failed; the worker(s)
        actually involved; their observations; any needs they raised).
        """
        query = self._query_from_needs(question, [active_need])

        unseen = [worker for worker in self.workers if worker.id not in seen_worker_ids]
        if unseen:
            picked_ids = self.reasoner.select_workers(
                query=query, need=active_need, candidates=unseen, limit=len(unseen), memory_hints={}
            )
            by_id = {worker.id: worker for worker in unseen}
            candidate = next((by_id[wid] for wid in picked_ids if wid in by_id), unseen[0])
            observations, round_needs = self._run_selected_workers(
                [candidate], query, question, search, worker_config, evidence, seen_worker_ids
            )
            if any(observation.evidence for observation in observations):
                return "expand_pool", [candidate], observations, round_needs

        unseen_after = [worker for worker in self.workers if worker.id not in seen_worker_ids]
        if unseen_after:
            third = unseen_after[0]
            observations, round_needs = self._run_selected_workers(
                [third], query, question, search, worker_config, evidence, seen_worker_ids
            )
            tried_workers = [worker for worker in self.workers if worker.id in seen_worker_ids]
            joint_observation = _run_coalition_cross_check(
                reasoner=self.reasoner,
                question=question,
                selected=tried_workers,
                evidence=evidence,
                search=search,
            )
            if joint_observation is not None:
                observations.append(joint_observation)
                evidence.extend(joint_observation.evidence)
                round_needs.extend(joint_observation.unresolved_needs)
            if any(observation.evidence for observation in observations):
                return "third_worker", [third], observations, round_needs

        tried_workers = [worker for worker in self.workers if worker.id in seen_worker_ids]
        if len(tried_workers) >= 2:
            bridge = _build_temporary_bridge(tried_workers)
            bridge_query = (
                f"{query} -- focus specifically on the interface/connection "
                "between these territories."
            )
            observations, round_needs = self._run_selected_workers(
                [bridge], bridge_query, question, search, worker_config, evidence, seen_worker_ids
            )
            if any(observation.evidence for observation in observations):
                return "temporary_bridge", [bridge], observations, round_needs

        all_files = sorted({file for worker in self.workers for file in worker.files})
        global_hits = search.search(query, all_files, limit=8)
        if global_hits:
            evidence.extend(global_hits)
            observation = WorkerObservation(
                worker_id="global-fallback",
                territory_id="global",
                evidence=global_hits,
                stop_reason="global_fallback",
            )
            return "global_fallback", [], [observation], []

        return "", [], [], []

    def _decide_and_run_local_action(
        self,
        *,
        active_need: UnresolvedNeed,
        last_worker: WorkerCard,
        made_progress_last_time: bool,
        candidates: list[WorkerCard],
        query: str,
        question: str,
        search: LocalSearchTool,
        worker_config: WorkerRunConfig,
        evidence: list[Evidence],
        seen_worker_ids: set[str],
    ) -> tuple[str, str, list[WorkerCard], list[WorkerObservation], list[UnresolvedNeed]]:
        """Makes "continue with the current worker" a deliberate action for
        a local-scope need's second-and-later attempt (see
        WorkerReasoner.decide_local_action), instead of an emergent side
        effect of whichever worker the routing score happens to rank
        highest. Returns (local_action, escalation_used, selected,
        observations, round_needs) -- escalation_used is set only when the
        decision itself is "escalate", so the caller can treat that branch
        identically to the streak-triggered escalation path.
        """
        progress_text = (
            "The current worker found new evidence in its last attempt."
            if made_progress_last_time
            else "The current worker's last attempt found nothing new."
        )
        action = self.reasoner.decide_local_action(
            need=active_need,
            evidence=evidence,
            worker_progress=progress_text,
            worker=last_worker,
        )
        if action not in ("continue", "handoff", "coalition", "escalate"):
            action = "continue"

        if action == "escalate":
            escalation_used, selected, observations, round_needs = self._escalate_stuck_need(
                question=question,
                active_need=active_need,
                seen_worker_ids=seen_worker_ids,
                evidence=evidence,
                search=search,
                worker_config=worker_config,
            )
            return action, escalation_used, selected, observations, round_needs

        if action == "continue":
            selected = [last_worker]
        elif action == "handoff":
            other = [worker for worker in candidates if worker.id != last_worker.id]
            selected = other[:1] or candidates[:1] or [last_worker]
        else:  # "coalition"
            other = [worker for worker in candidates if worker.id != last_worker.id]
            partner = other[:1] or candidates[:1]
            selected = [last_worker, *[worker for worker in partner if worker.id != last_worker.id]]
            if not selected:
                selected = [last_worker]

        observations, round_needs = self._run_selected_workers(
            selected, query, question, search, worker_config, evidence, seen_worker_ids
        )
        return action, "", selected, observations, round_needs

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

    def _dense_routing_scores(self, query_text: str) -> dict[str, float]:
        """worker_id -> cosine similarity against the (cheap, eagerly-built
        at `ant index --dense` time) worker-card index. Returns {} whenever
        no card index exists or no embedder is available -- routing then
        falls back to the lexical signals alone, exactly as before dense
        retrieval existed.
        """
        if not self.index_path:
            return {}
        if not self._worker_card_index_loaded:
            self._worker_card_index = EmbeddingIndex.load(
                self.index_path / "dense", WORKER_CARDS_KEY
            )
            self._worker_card_index_loaded = True
        index = self._worker_card_index
        if index is None or not index.entries:
            return {}
        embedder = get_shared_embedder()
        if embedder is None:
            return {}
        [query_vector] = embedder.embed([query_text])
        hits = index.search(query_vector, limit=len(index.entries))
        return {entry.path: score for score, entry in hits}

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
        symbol_weights = _relevant_symbol_weights(relevant_symbols, self.workers)
        implementation_intent = _asks_for_source_implementation_text(query_text)
        dense_scores = self._dense_routing_scores(query_text)
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
            relevant_symbol_score = sum(symbol_weights[symbol] for symbol in relevant_symbol_hits)
            score = len(query_hits) + len(suggested_term_hits) + relevant_symbol_score
            routing_config = DEFAULT_SCORING_CONFIG.routing
            source_worker_bonus = 0
            if need and worker.id == need.source_worker_id:
                local_overlap = len(suggested_term_hits) or len(query_hits)
                source_worker_bonus = (
                    routing_config.source_worker_local_bonus
                    if need.scope == "local"
                    else routing_config.source_worker_global_bonus
                )
                source_worker_bonus += min(local_overlap, routing_config.source_worker_overlap_cap)
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
                    seen_worker_penalty = (
                        routing_config.seen_worker_local_penalty
                        if need.scope == "local"
                        else routing_config.seen_worker_global_penalty
                    )
                elif need is None:
                    seen_worker_penalty = routing_config.seen_worker_no_need_penalty
                score -= seen_worker_penalty
            source_path_bonus = 0
            if score > 0 or implementation_intent:
                source_path_bonus = _source_path_bonus(worker)
                score += source_path_bonus
            memory_route_bonus = _memory_route_bonus(worker, query_terms, self.memory_routes)
            score += memory_route_bonus
            # Coarse semantic signal alongside the lexical ones above: a
            # worker card whose responsibilities/terms are semantically
            # close to the query, even with no shared vocabulary at all,
            # still gets a real (if modest) push -- catches the case where
            # every lexical signal here is zero purely because the query's
            # wording happens not to overlap this worker's card.
            dense_routing_bonus = round(
                dense_scores.get(worker.id, 0.0) * routing_config.dense_routing_weight
            )
            score += dense_routing_bonus
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
                        dense_routing_bonus=dense_routing_bonus,
                        test_path_penalty=test_path_penalty,
                        seen_worker_penalty=seen_worker_penalty,
                    ),
                    worker,
                )
            )
        scored.sort(key=lambda item: item[0].final_score, reverse=True)
        scored = _llm_reorder(self.reasoner, query_text, need, scored)
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
        genuinely_close = (
            first_score > 0
            and second_score
            >= first_score * DEFAULT_SCORING_CONFIG.routing.initial_worker_closeness_ratio
        )
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
        # Keep the original question as a stable lexical anchor in every round's
        # query, not just as a fallback for when the need text is empty. Round 2+
        # queries are built from an LLM-generated UnresolvedNeed, and that call has
        # no temperature/seed pinning -- its wording varies between otherwise
        # identical runs. Dropping the original question's terms each round made
        # routing and retrieval fully dependent on that one call's phrasing; keeping
        # them present means a lucky/unlucky word choice can only add or subtract
        # recall, not erase the anchor entirely.
        parts = [question]
        for need in needs:
            parts.append(_need_query_text(need))
        query = " ".join(part for part in parts if part).strip()
        return query or question


def _select_for_active_need(
    candidates: list[WorkerCard],
    active_need: UnresolvedNeed,
    seen_worker_ids: set[str],
) -> list[WorkerCard]:
    if active_need.scope != "cross_territory":
        return candidates[:1]
    # A cross_territory need means the source worker's own territory could
    # not resolve this on its own -- the seen-worker penalty nudges scoring
    # away from re-picking it but does not forbid it, and a still-relevant-
    # looking card can out-score that penalty. Confirmed directly on two
    # real seaborn traces: round 1 re-selected the exact same source worker
    # for a need it had just flagged cross_territory, asked it the same
    # question again, got nothing new, and coalition_formed never fired (it
    # requires the round's pick to differ from source_worker_id) --
    # silently skipping the joint cross-check this need type exists to
    # trigger. Force genuine diversification instead of hoping the score
    # avoids this on its own; only re-recruit a previously-seen worker if
    # nothing else is left.
    #
    # Exclude every worker seen so far this task, not just
    # active_need.source_worker_id: once a coalition cross-check runs, its
    # own re-raised need carries a synthetic "coalition:a-b" source id that
    # never string-matches any real WorkerCard.id, silently turning this
    # exclusion into a no-op. Confirmed directly on a real yt-dlp trace:
    # rounds 2-5 all re-picked the exact same worker (already proven unable
    # to answer in round 1) because the active need's source was
    # "coalition:worker-yt-dlp-utils", not "worker-yt-dlp-utils" -- five
    # rounds spent re-asking one worker the same question with cosmetically
    # reworded need text each time, producing 11 near-duplicate unresolved
    # needs instead of ever trying a different territory.
    other_candidates = [worker for worker in candidates if worker.id not in seen_worker_ids]
    return other_candidates[:1] or candidates[:1]


def _build_temporary_bridge(tried_workers: list[WorkerCard]) -> WorkerCard:
    """An ephemeral, in-task-only worker for _escalate_stuck_need's third
    tactic: the union of every territory already tried for a stuck need,
    so a single search pass can see across their boundary instead of two
    separate territory-scoped workers each missing the other's half. Never
    saved to IndexStore -- persistent reorganization is evolve_workers'
    job, at a slow cross-task timescale earned by *repeated* need, not
    granted to a single task's stuck attempt.
    """
    files = sorted({file for worker in tried_workers for file in worker.files})
    terms = sorted({term for worker in tried_workers for term in worker.searchable_terms})[:32]
    ids = sorted(worker.id.removeprefix("worker-") for worker in tried_workers)
    bridge_id = f"worker-temp-bridge-{'-'.join(ids)}"[:96]
    names = " + ".join(worker.name for worker in tried_workers[:3])
    return WorkerCard(
        id=bridge_id,
        territory_id="temp-bridge",
        name=f"{names} temporary bridge",
        root="",
        responsibilities=[
            "Ephemeral, task-scoped interface specialist -- not part of persistent colony state.",
            f"Combines territories: {', '.join(worker.id for worker in tried_workers)}.",
        ],
        searchable_terms=terms,
        files=files,
    )


def _llm_reorder(
    reasoner: WorkerReasoner,
    query_text: str,
    need: UnresolvedNeed | None,
    scored: list[tuple[WorkerRoutingScore, WorkerCard]],
) -> list[tuple[WorkerRoutingScore, WorkerCard]]:
    """Let the reasoner make the actual recruitment call among the top
    candidates by lexical+dense score, instead of that composite score
    deciding outright. The composite score still bounds *which* workers are
    even considered (cost control on a large evolved population) and is the
    fallback order if the reasoner call degrades -- it just no longer picks
    the winner by itself.
    """
    if not scored:
        return scored
    pool_size = DEFAULT_SCORING_CONFIG.routing.llm_routing_candidate_pool_size
    pool = scored[:pool_size]
    tail = scored[pool_size:]
    # Surface the memory-route replay signal as text the reasoner can
    # actually read, instead of only as an invisible number folded into the
    # pool-ranking score it no longer decides the winner with.
    memory_hints = {
        score.worker_id: f"resolved a similar past need (route bonus +{score.memory_route_bonus})"
        for score, _ in pool
        if score.memory_route_bonus > 0
    }
    selected_ids = reasoner.select_workers(
        query=query_text,
        need=need,
        candidates=[worker for _, worker in pool],
        limit=len(pool),
        memory_hints=memory_hints,
    )
    if not selected_ids:
        return scored
    by_id = {worker.id: (score, worker) for score, worker in pool}
    reordered = [by_id[worker_id] for worker_id in selected_ids if worker_id in by_id]
    reordered_ids = {worker_id for worker_id in selected_ids if worker_id in by_id}
    remaining = [item for item in pool if item[1].id not in reordered_ids]
    return reordered + remaining + tail


def _select_evidence(
    reasoner: WorkerReasoner,
    question: str,
    ranked_evidence: list[Evidence],
    search: LocalSearchTool,
) -> list[Evidence]:
    """Same pool-then-LLM-decides pattern as _llm_reorder, applied to the
    final evidence cut before synthesis: the score-ranked order still
    bounds the candidate pool for prompt-size/cost control, but the
    reasoner -- not a fixed top-N slice -- decides what actually reaches
    the synthesizer. Per-worker collection deliberately does no equivalent
    filtering of its own anymore (AutonomousWorker's evidence_limit is a
    generous safety cap, not a relevance decision), so this single pass
    over the fully pooled evidence is the one place spent on real judgment.
    The same call also flags kept items whose quote is too narrow to answer
    from; those get their full source region reopened here (Shared
    Evidence State: "evidence compression is reversible") rather than only
    at coalition cross-check time.
    """
    if not ranked_evidence:
        return []
    pool_size = DEFAULT_SCORING_CONFIG.routing.llm_evidence_pool_size
    keep_limit = DEFAULT_SCORING_CONFIG.routing.llm_evidence_keep_limit
    pool = ranked_evidence[:pool_size]
    keep_ids, expand_ids = reasoner.select_evidence(
        question=question, evidence=pool, limit=keep_limit
    )
    kept_indices: list[int] = []
    for raw_index in keep_ids:
        try:
            index = int(raw_index)
        except ValueError:
            continue
        if 0 <= index < len(pool):
            kept_indices.append(index)
    if not kept_indices:
        kept_indices = list(range(min(keep_limit, len(pool))))
    reopened_map = _reopen_evidence_by_index(expand_ids, pool, search) if expand_ids else {}
    return [reopened_map.get(index, pool[index]) for index in kept_indices[:keep_limit]]


def _matches_term(query_term: str, terms: set[str]) -> bool:
    for term in terms:
        if query_term == term:
            return True
        parts = _compound_parts(term)
        if query_term in parts:
            return True
        if is_stem_match(query_term, term):
            return True
        # Confirmed on a real seaborn trace: a symbol named
        # `_assign_variables_wideform` compound-splits to {"assign",
        # "variables", "wideform"} -- "wideform" stays one un-split token
        # because the source didn't put an underscore between "wide" and
        # "form". A query term "wide" then matched neither the full term
        # (stem-match checked only the whole "assign_variables_wideform",
        # which doesn't start with "wide") nor the compound-part set
        # (exact membership only, "wide" != "wideform"). Meanwhile a
        # tutorial file literally named `wide_form_violinplot.py` split
        # into {"wide", "form", "violinplot"} and got credit "wide" never
        # could. Stem-matching against each compound part too closes that
        # gap: "wide" is a >=4-char prefix of "wideform".
        if any(is_stem_match(query_term, part) for part in parts):
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


def _relevant_symbol_weights(symbols: set[str], workers: list[WorkerCard]) -> dict[str, int]:
    """Weight a relevant-symbol match inversely to how many workers it
    matches, instead of a flat bonus per hit.

    A symbol unique (or nearly so) to one worker -- a domain-specific
    concept mentioned in exactly one README, say -- is a strong routing
    signal and keeps close to full weight. A symbol that matches most of
    the colony -- most often the repository's own package name, present in
    literally every file path under a standard src/<pkg>/ layout via
    `_matches_symbol`'s path-substring fallback -- carries almost no
    discriminating power: it would otherwise inflate every such worker's
    score by the same flat amount, systematically favoring whichever
    workers also happen to pick up an unrelated bonus (like the source-path
    bonus a src/ worker gets that an examples/ worker never does) purely
    because their path contains the repo's own name, not because they are
    actually relevant to the question. A fixed match-count cutoff would
    need a different threshold per repository's directory layout; scaling
    the weight continuously does not.
    """
    base_weight = DEFAULT_SCORING_CONFIG.routing.relevant_symbol_base_weight
    weights: dict[str, int] = {}
    for symbol in symbols:
        match_count = sum(1 for worker in workers if _matches_symbol(symbol, worker))
        weights[symbol] = max(1, round(base_weight / max(1, match_count)))
    return weights


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
    """A route recorded for one question can get replayed for a completely
    different one that happens to share a single generic word (e.g. both
    mention "quantum" in a quantum-computing repo). Requiring the overlap to
    cover a real fraction of the route's own vocabulary -- not just be
    non-empty -- is what tells "this new question is basically the same
    need as before" apart from "these two unrelated questions both used one
    common word". A route recorded from a narrow, single-term need (ratio
    1.0 on an exact rematch) still gets full credit; a route whose need
    covered a dozen words and shares only one with this query does not.
    """
    config = DEFAULT_SCORING_CONFIG.routing
    bonus = 0
    for route in routes:
        route_terms = {term.lower() for term in route.need_terms}
        if worker.id not in route.worker_ids or not route_terms:
            continue
        overlap = query_terms & route_terms
        if not overlap or len(overlap) / len(route_terms) < config.min_memory_route_overlap_ratio:
            continue
        candidate = round(route.weight * config.memory_route_weight_multiplier) + len(overlap)
        bonus = max(bonus, min(config.memory_route_bonus_cap, candidate))
    return bonus


def _territory_hint_score(worker: WorkerCard, hints: list[str]) -> int:
    if not hints:
        return 0
    worker_terms = _worker_terms(worker)
    territory_terms = _term_set(" ".join([worker.territory_id, worker.root, worker.name]))
    config = DEFAULT_SCORING_CONFIG.routing
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
            score += config.territory_hint_full_match_bonus
        elif overlap:
            score += config.territory_hint_partial_base_bonus + (
                overlap * config.territory_hint_partial_per_overlap_bonus
            )
    return score


def _asks_for_source_implementation(need: UnresolvedNeed) -> bool:
    return _asks_for_source_implementation_text(_need_query_text(need))


def _asks_for_source_implementation_text(text: str) -> bool:
    text = text.lower()
    indicators = [
        # "implement" as a stem catches implement/implements/implementing/
        # implementation/implemented -- "How does X implement Y" (no -ation
        # or -ed suffix) previously matched none of the older, longer-form
        # indicators and so never triggered the coverage-gap safety net.
        "implement",
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
    config = DEFAULT_SCORING_CONFIG.routing
    files = worker.files or [worker.root]
    test_like = sum(
        1
        for file in files
        if {"test", "tests"} & set(file.replace("\\", "/").lower().split("/"))
    )
    if not files:
        return 0
    if test_like / len(files) >= config.test_path_ratio_threshold:
        return config.test_path_penalty
    return 0


def _source_path_bonus(worker: WorkerCard) -> int:
    config = DEFAULT_SCORING_CONFIG.routing
    files = worker.files or [worker.root]
    source_like = sum(1 for file in files if has_source_part(file))
    if not files:
        return 0
    if source_like / len(files) >= config.source_path_ratio_threshold:
        return config.source_path_bonus
    return 0


def _rank_global_evidence(evidence: list[Evidence], question: str) -> list[Evidence]:
    terms = extract_terms(question)

    def score(item: Evidence) -> int:
        return score_evidence(
            quote=item.quote,
            path=item.path,
            reason=item.reason,
            terms=terms,
            dense_score=item.dense_score,
            symbol_name=item.symbols[0] if item.symbols else "",
        )

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
    if (
        "subclass_lookup" not in existing_types
        and _asks_for_inheritance_text(question)
        and not _has_subclass_evidence_for(set(question_symbols), evidence)
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


def _verify_inheritance_completeness(
    question: str,
    workers: list[WorkerCard],
    search: LocalSearchTool,
) -> tuple[list[Evidence], AbsenceProof | None]:
    """Run one definitive, repository-wide subclass lookup for an inheritance
    question, instead of relying on whichever worker recruitment happened to
    reach. `subclasses()` is an AST-index lookup, not a heuristic search, so
    scanning the union of every worker's files (not just the routed worker's
    territory) makes "these are all the subclasses" an actually-true claim,
    and the resulting AbsenceProof gives the synthesizer real grounds to
    state that instead of hedging about territories it never visited.
    """
    if not _asks_for_inheritance_text(question):
        return [], None
    symbols = sorted(_relevant_symbols(question))
    if not symbols:
        return [], None
    all_files = sorted({file for worker in workers for file in worker.files})
    if not all_files:
        return [], None

    found: list[Evidence] = []
    for symbol in symbols:
        found.extend(search.subclasses(symbol, all_files, limit=50))
    found = _dedupe_evidence(found)

    proof = AbsenceProof(
        query=question,
        relevant_symbols=symbols,
        searched_worker_ids=sorted(worker.id for worker in workers),
        searched_territories=sorted({worker.territory_id for worker in workers}),
        searched_paths=all_files,
        tools=["subclasses"],
        exhaustive=True,
        conclusion=(
            f"found_{len(found)}_subclass{'es' if len(found) != 1 else ''}"
            if found
            else "not_found"
        ),
    )
    return found, proof


def _has_subclass_evidence(text: str) -> bool:
    return bool(re.search(r"\bclass\s+[A-Za-z_][A-Za-z0-9_]*\([^)]*[A-Za-z_]", text))


def _has_subclass_evidence_for(candidates: set[str], evidence: list[Evidence]) -> bool:
    """True if evidence contains a class definition whose base matches a candidate.

    Scoped matching (base name equals one of `candidates`) is preferred over the
    generic "some class has some base" check so that evidence for an unrelated
    base class cannot be mistaken for coverage of the base the need is about.
    Falls back to the generic check only when no specific base symbol is known.
    """
    if not candidates:
        return any(_has_subclass_evidence(item.quote) for item in evidence)
    lowered = {candidate.lower() for candidate in candidates}
    for item in evidence:
        for bases in BASE_CLASS_RE.findall(item.quote):
            for base in bases.split(","):
                base_name = base.strip().split(".")[-1]
                if base_name.lower() in lowered:
                    return True
    return False


def _has_source_definition_evidence(evidence: list[Evidence]) -> bool:
    return any(
        has_source_part(item.path)
        and not has_low_value_part(item.path)
        and ("class " in item.quote or "def " in item.quote)
        for item in evidence
    )


def _has_source_definition_evidence_for(candidates: set[str], evidence: list[Evidence]) -> bool:
    """Scoped variant of `_has_source_definition_evidence`: the definition must
    actually define one of `candidates`, not merely sit near source-looking code."""
    if not candidates:
        return _has_source_definition_evidence(evidence)
    for item in evidence:
        if not has_source_part(item.path) or has_low_value_part(item.path):
            continue
        quote = item.quote
        if "class " not in quote and "def " not in quote:
            continue
        if any(_symbol_defined_in_quote(candidate, quote) for candidate in candidates):
            return True
    return False


def _symbol_defined_in_quote(symbol: str, quote: str) -> bool:
    return bool(re.search(rf"\b(?:class|def)\s+{re.escape(symbol)}\b", quote))


def _need_symbol_candidates(need: UnresolvedNeed, question: str) -> set[str]:
    return _relevant_symbols(f"{_need_query_text(need)} {question}", need)


def _is_need_satisfied(need: UnresolvedNeed, evidence: list[Evidence], question: str) -> bool:
    if need.need_type not in _CLOSABLE_BY_EVIDENCE:
        return False
    candidates = _need_symbol_candidates(need, question)
    if need.need_type == "subclass_lookup":
        return _has_subclass_evidence_for(candidates, evidence)
    if need.need_type == "implementation_location":
        return _has_source_definition_evidence_for(candidates, evidence)
    if need.need_type == "source_test_coalition":
        return _has_source_definition_evidence(evidence) and any(
            _is_test_evidence(item) for item in evidence
        )
    return False


def _close_resolved_needs(
    needs: list[UnresolvedNeed],
    evidence: list[Evidence],
    question: str,
) -> list[UnresolvedNeed]:
    """Drop needs whose target evidence is now present in the cumulative evidence
    state. Re-run after every round so a need raised while evidence was sparse
    does not linger once a later round (possibly from a different worker)
    actually grounds it."""
    return [need for need in needs if not _is_need_satisfied(need, evidence, question)]


def _need_identity(need: UnresolvedNeed) -> tuple[str, str]:
    text = (need.missing or need.description or "").strip().lower()
    return (need.need_type, text)


def _merge_needs(
    existing: list[UnresolvedNeed],
    new_needs: list[UnresolvedNeed],
) -> list[UnresolvedNeed]:
    """Accumulate needs across rounds instead of overwriting. A need that keeps
    getting re-raised for the same reason is merged (union of symbols/terms)
    rather than duplicated, so a stale reasoner that never self-closes a need
    does not flood the final list with repeats of the same gap."""
    merged = list(existing)
    index_by_key = {_need_identity(need): index for index, need in enumerate(merged)}
    for need in new_needs:
        key = _need_identity(need)
        if key in index_by_key:
            merged[index_by_key[key]] = _merge_need_pair(merged[index_by_key[key]], need)
        else:
            index_by_key[key] = len(merged)
            merged.append(need)
    return merged


def _merge_need_pair(old: UnresolvedNeed, new: UnresolvedNeed) -> UnresolvedNeed:
    return new.model_copy(
        update={
            "relevant_symbols": sorted(set(old.relevant_symbols) | set(new.relevant_symbols)),
            "suggested_terms": sorted(set(old.suggested_terms) | set(new.suggested_terms)),
            "suggested_territories": sorted(
                set(old.suggested_territories) | set(new.suggested_territories)
            ),
            "evidence_ids": sorted(set(old.evidence_ids) | set(new.evidence_ids)),
        }
    )


def _evidence_key(item: Evidence) -> tuple[str, int, int, str]:
    return (item.path, item.line_start, item.line_end, item.quote)


def _dedupe_evidence(evidence: list[Evidence]) -> list[Evidence]:
    # See the matching note on autonomous._dedupe: keep the first-seen copy
    # of a (path, lines, quote) duplicate, but merge a later duplicate's
    # dense_score forward rather than silently dropping it, so a chunk two
    # different workers/rounds both surfaced -- one lexically, one via
    # dense_search -- doesn't lose the dense signal to whichever copy
    # happened to arrive first.
    index_by_key: dict[tuple[str, int, int, str], int] = {}
    deduped: list[Evidence] = []
    for item in evidence:
        key = _evidence_key(item)
        if key in index_by_key:
            existing = deduped[index_by_key[key]]
            if item.dense_score > existing.dense_score:
                deduped[index_by_key[key]] = existing.model_copy(
                    update={"dense_score": item.dense_score}
                )
            continue
        index_by_key[key] = len(deduped)
        deduped.append(item)
    return deduped


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


def _run_coalition_cross_check(
    *,
    reasoner: WorkerReasoner,
    question: str,
    selected: list[WorkerCard],
    evidence: list[Evidence],
    search: LocalSearchTool,
) -> WorkerObservation | None:
    """Coalitions exist to reason jointly across territories, not just to log
    that a cross-check happened. Run one real reasoning pass over the pooled
    coalition evidence so the reasoner can catch a conflict or an
    under-specified claim between peers' evidence that no single worker would
    see on its own; if it flags a gap tied to a specific piece of evidence,
    reopen that evidence's full source region (Evidence Compression Is
    Reversible) instead of only trusting the compressed quote already held.
    """
    joint_evidence = _dedupe_evidence(evidence)
    coalition_worker_id = "coalition:" + "-".join(sorted(worker.id for worker in selected))
    joint_observation = reasoner.observe(
        question=question,
        worker_id=coalition_worker_id,
        territory_id="coalition",
        evidence=joint_evidence,
    )
    if not joint_observation.unresolved_needs:
        return None
    reopened = _reopen_referenced_evidence(
        joint_observation.unresolved_needs, joint_evidence, search
    )
    return WorkerObservation(
        worker_id=coalition_worker_id,
        territory_id="coalition",
        evidence=reopened,
        unresolved_needs=joint_observation.unresolved_needs,
        stop_reason="coalition_cross_check",
    )


def _reopen_referenced_evidence(
    needs: list[UnresolvedNeed],
    evidence_pool: list[Evidence],
    search: LocalSearchTool,
    context_lines: int = 30,
) -> list[Evidence]:
    reopened: list[Evidence] = []
    seen: set[tuple[str, int]] = set()
    for need in needs:
        for raw_index in need.evidence_ids:
            try:
                index = int(raw_index)
            except ValueError:
                continue
            if not (0 <= index < len(evidence_pool)):
                continue
            source = evidence_pool[index]
            key = (source.path, source.line_start)
            if key in seen:
                continue
            seen.add(key)
            try:
                region = search.read_region(source.path, source.line_start, context_lines)
            except OSError:
                continue
            reopened.append(
                region.model_copy(
                    update={
                        "worker_id": source.worker_id,
                        "reason": f"Reopened for coalition cross-check: {source.reason}".strip(),
                    }
                )
            )
    return reopened


def _reopen_evidence_by_index(
    indices: list[str],
    pool: list[Evidence],
    search: LocalSearchTool,
    context_lines: int = 30,
) -> dict[int, Evidence]:
    """Maps pool index -> reopened Evidence (larger source region), for
    _select_evidence's own expand requests. Kept separate from
    _reopen_referenced_evidence above (which dedupes by source location
    across several needs' evidence_ids) since this one resolves a single
    selection call's own flagged indices.
    """
    reopened: dict[int, Evidence] = {}
    for raw_index in indices:
        try:
            index = int(raw_index)
        except ValueError:
            continue
        if not (0 <= index < len(pool)):
            continue
        source = pool[index]
        try:
            region = search.read_region(source.path, source.line_start, context_lines)
        except OSError:
            continue
        reopened[index] = region.model_copy(
            update={
                "worker_id": source.worker_id,
                "reason": f"Reopened for deeper context: {source.reason}".strip(),
            }
        )
    return reopened
