"""Read-only, post-hoc telemetry over already-completed traces: which
evolved/overlay workers (WorkerCard.parent_worker_ids non-empty -- see its
own docstring) were even considered, actually recruited, and what they
produced. Nothing here is consulted by routing, candidate selection,
evolution decisions, resolution, synthesis, or scoring -- every function in
this module takes an already-finished EvidenceState (or a list of them) and
a worker population as plain input and returns plain data, computed purely
by reading fields that already exist on NodeExecutionTrace/Evidence/
WorkerCard. Nothing here is called from inside ask()/evolve_workers()
themselves; ant.evaluation.gen_compare calls it once per stage, after that
stage's own run_batch has already finished and saved its trace files.
"""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field

from ant.domain import Evidence, EvidenceState, WorkerCard


def _evidence_key(item: Evidence) -> tuple[str, int, int, str]:
    return (item.path, item.line_start, item.line_end, item.quote)


class EvolvedWorkerNeedAttribution(BaseModel):
    """One evolved/overlay worker's involvement in one need's execution
    within one round of one question, derived read-only from an
    already-completed NodeExecutionTrace."""

    need_id: str
    worker_id: str
    structural_action: str
    parent_worker_ids: list[str] = Field(default_factory=list)
    generation_created: int = 0
    # Present in this round's candidate_worker_ids -- retrieval considered
    # it, whether or not the Orchestrator actually picked it.
    matched: bool = False
    # Present in this round's worker_ids -- actually assigned/ran.
    recruited: bool = False
    evidence_count: int = 0
    # Of this worker's own evidence_count, how many items no OTHER worker
    # (evolved or base) also contributed anywhere else in this same
    # question -- see compute_question_attribution's own docstring for the
    # exact scope.
    unique_evidence_count: int = 0
    resolution_before: str = "unresolved"
    resolution_after: str = "unresolved"
    need_reduction: int = 0
    # True if any of this worker's own parent_worker_ids were ALSO
    # recruited for this same need_id this same round -- the Orchestrator
    # falling back to (or supplementing with) the base/source worker
    # alongside its structural descendant, not instead of it.
    base_fallback_recruited: bool = False


class QuestionEvolvedWorkerAttribution(BaseModel):
    example_id: str
    entries: list[EvolvedWorkerNeedAttribution] = Field(default_factory=list)


class GenerationWorkerUsageSummary(BaseModel):
    """One evolved/overlay worker's usage aggregated across every question
    in one generation's own stage -- what GenerationSnapshot's own
    docstring calls "created ≠ actually useful"."""

    worker_id: str
    structural_action: str
    parent_worker_ids: list[str] = Field(default_factory=list)
    generation_created: int = 0
    matched_task_count: int = 0
    recruited_task_count: int = 0
    rounds_used: int = 0
    evidence_contributed: int = 0
    unique_evidence_contributed: int = 0
    tasks_with_progress: int = 0
    tasks_with_resolved_need: int = 0
    base_fallback_task_count: int = 0


def compute_question_attribution(
    example_id: str, state: EvidenceState, workers_by_id: dict[str, WorkerCard]
) -> QuestionEvolvedWorkerAttribution:
    """Pure function: reads `state`/`workers_by_id`, mutates neither,
    produces one EvolvedWorkerNeedAttribution entry per (round, need,
    evolved worker) that either matched or was recruited that need this
    round. A base worker (WorkerCard.parent_worker_ids empty -- an
    original, non-structural worker) never gets an entry here; this module
    only ever reports on evolved/overlay workers.

    `unique_evidence_count`'s scope is per-QUESTION (this whole EvidenceState,
    not just this one round/need): an evidence item this worker contributed
    anywhere in this task that no OTHER worker_id's own contribution
    anywhere else in this same task also produced (same (path, line_start,
    line_end, quote) key). Deliberately task-wide, not per-round-need --
    the same worker re-finding its own earlier item on a later round for
    the SAME need is not "non-unique" just because it repeats itself; what
    matters here is whether some OTHER worker's own contribution duplicates
    it.
    """
    evidence_keys_by_worker: dict[str, set[tuple[str, int, int, str]]] = defaultdict(set)
    for round_state in state.rounds:
        for trace in round_state.node_executions:
            for observation in trace.observations:
                for item in observation.evidence:
                    if item.worker_id:
                        evidence_keys_by_worker[item.worker_id].add(_evidence_key(item))

    key_owner_count: dict[tuple[str, int, int, str], int] = defaultdict(int)
    for keys in evidence_keys_by_worker.values():
        for key in keys:
            key_owner_count[key] += 1

    entries: list[EvolvedWorkerNeedAttribution] = []
    for round_state in state.rounds:
        for trace in round_state.node_executions:
            candidate_ids = set(trace.candidate_worker_ids)
            recruited_ids = set(trace.worker_ids)
            involved_ids = candidate_ids | recruited_ids
            for worker_id in sorted(involved_ids):
                worker = workers_by_id.get(worker_id)
                if worker is None or not worker.parent_worker_ids:
                    continue
                own_evidence = [
                    item
                    for observation in trace.observations
                    for item in observation.evidence
                    if item.worker_id == worker_id
                ]
                unique_count = sum(
                    1 for item in own_evidence if key_owner_count.get(_evidence_key(item), 0) == 1
                )
                entries.append(
                    EvolvedWorkerNeedAttribution(
                        need_id=trace.need_id,
                        worker_id=worker_id,
                        structural_action=worker.structural_action,
                        parent_worker_ids=worker.parent_worker_ids,
                        generation_created=worker.generation_created,
                        matched=worker_id in candidate_ids,
                        recruited=worker_id in recruited_ids,
                        evidence_count=len(own_evidence),
                        unique_evidence_count=unique_count,
                        resolution_before=trace.resolution_before,
                        resolution_after=trace.resolution_after,
                        need_reduction=trace.need_reduction,
                        base_fallback_recruited=bool(
                            set(worker.parent_worker_ids) & recruited_ids
                        ),
                    )
                )
    return QuestionEvolvedWorkerAttribution(example_id=example_id, entries=entries)


def aggregate_generation_worker_usage(
    attributions: list[QuestionEvolvedWorkerAttribution],
) -> list[GenerationWorkerUsageSummary]:
    """Pure function: folds every question's own QuestionEvolvedWorkerAttribution
    (already computed by compute_question_attribution) into one
    GenerationWorkerUsageSummary per evolved worker, for a generation's own
    GenerationSnapshot. Task counts are DISTINCT example_ids, not raw entry
    counts (the same worker can have multiple entries in one question --
    once per round/need it was involved in)."""
    matched_tasks: dict[str, set[str]] = defaultdict(set)
    recruited_tasks: dict[str, set[str]] = defaultdict(set)
    progress_tasks: dict[str, set[str]] = defaultdict(set)
    resolved_tasks: dict[str, set[str]] = defaultdict(set)
    fallback_tasks: dict[str, set[str]] = defaultdict(set)
    rounds_used: dict[str, int] = defaultdict(int)
    evidence_contributed: dict[str, int] = defaultdict(int)
    unique_evidence_contributed: dict[str, int] = defaultdict(int)
    meta: dict[str, tuple[str, list[str], int]] = {}

    for question in attributions:
        for entry in question.entries:
            meta[entry.worker_id] = (
                entry.structural_action,
                entry.parent_worker_ids,
                entry.generation_created,
            )
            if entry.matched:
                matched_tasks[entry.worker_id].add(question.example_id)
            if entry.recruited:
                recruited_tasks[entry.worker_id].add(question.example_id)
                rounds_used[entry.worker_id] += 1
                evidence_contributed[entry.worker_id] += entry.evidence_count
                unique_evidence_contributed[entry.worker_id] += entry.unique_evidence_count
                if entry.need_reduction > 0:
                    resolved_tasks[entry.worker_id].add(question.example_id)
                if entry.resolution_after != entry.resolution_before:
                    progress_tasks[entry.worker_id].add(question.example_id)
                if entry.base_fallback_recruited:
                    fallback_tasks[entry.worker_id].add(question.example_id)

    all_worker_ids = set(matched_tasks) | set(recruited_tasks)
    summaries = []
    for worker_id in sorted(all_worker_ids):
        structural_action, parent_worker_ids, generation_created = meta[worker_id]
        summaries.append(
            GenerationWorkerUsageSummary(
                worker_id=worker_id,
                structural_action=structural_action,
                parent_worker_ids=parent_worker_ids,
                generation_created=generation_created,
                matched_task_count=len(matched_tasks.get(worker_id, ())),
                recruited_task_count=len(recruited_tasks.get(worker_id, ())),
                rounds_used=rounds_used.get(worker_id, 0),
                evidence_contributed=evidence_contributed.get(worker_id, 0),
                unique_evidence_contributed=unique_evidence_contributed.get(worker_id, 0),
                tasks_with_progress=len(progress_tasks.get(worker_id, ())),
                tasks_with_resolved_need=len(resolved_tasks.get(worker_id, ())),
                base_fallback_task_count=len(fallback_tasks.get(worker_id, ())),
            )
        )
    return summaries
