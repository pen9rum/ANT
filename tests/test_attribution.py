import copy

from ant.domain import (
    Evidence,
    EvidenceState,
    NodeExecutionTrace,
    PlanningRound,
    WorkerCard,
    WorkerObservation,
)
from ant.evaluation.attribution import (
    aggregate_generation_worker_usage,
    compute_question_attribution,
)


def _evidence(path: str, quote: str, worker_id: str) -> Evidence:
    return Evidence(
        path=path, line_start=1, line_end=2, quote=quote, reason="r", worker_id=worker_id
    )


def _fixture_state() -> EvidenceState:
    return EvidenceState(
        question="q",
        rounds=[
            PlanningRound(
                round_index=0,
                node_executions=[
                    NodeExecutionTrace(
                        need_id="root",
                        worker_ids=["worker-bridge"],
                        candidate_worker_ids=["worker-base", "worker-bridge"],
                        resolution="partial",
                        resolution_before="unresolved",
                        resolution_after="partial",
                        need_reduction=0,
                        observations=[
                            WorkerObservation(
                                worker_id="worker-bridge",
                                territory_id="bridge",
                                evidence=[_evidence("a.py", "q1", "worker-bridge")],
                            )
                        ],
                    )
                ],
            ),
            PlanningRound(
                round_index=1,
                node_executions=[
                    NodeExecutionTrace(
                        need_id="root",
                        worker_ids=["worker-bridge", "worker-base-a"],
                        candidate_worker_ids=["worker-bridge"],
                        resolution="resolved",
                        resolution_before="partial",
                        resolution_after="resolved",
                        need_reduction=1,
                        observations=[
                            WorkerObservation(
                                worker_id="worker-bridge",
                                territory_id="bridge",
                                evidence=[_evidence("b.py", "q2", "worker-bridge")],
                            ),
                            WorkerObservation(
                                worker_id="worker-base-a",
                                territory_id="a",
                                # Same (path, lines, quote) as round 0's own
                                # worker-bridge item -- a different worker
                                # also producing this exact evidence makes
                                # it non-unique to worker-bridge task-wide.
                                evidence=[_evidence("a.py", "q1", "worker-base-a")],
                            ),
                        ],
                    )
                ],
            ),
        ],
    )


def _fixture_workers() -> dict[str, WorkerCard]:
    workers = [
        WorkerCard(id="worker-base", territory_id="base", name="base", root="base", files=[]),
        WorkerCard(id="worker-base-a", territory_id="a", name="a", root="a", files=["a.py"]),
        WorkerCard(id="worker-base-b", territory_id="b", name="b", root="b", files=["b.py"]),
        WorkerCard(
            id="worker-bridge",
            territory_id="bridge",
            name="bridge",
            root="",
            files=["a.py", "b.py"],
            parent_worker_ids=["worker-base-a", "worker-base-b"],
            structural_action="birth",
            generation_created=2,
            lifecycle_state="probationary",
        ),
    ]
    return {worker.id: worker for worker in workers}


def test_compute_question_attribution_reports_matched_recruited_evidence_and_fallback() -> None:
    state = _fixture_state()
    result = compute_question_attribution("q1", state, _fixture_workers())

    # worker-bridge appears in both rounds; worker-base-a is also recruited
    # in round 1 but is a base worker (no parent_worker_ids), so it never
    # gets its own entry (see the dedicated test for that).
    assert len(result.entries) == 2

    round0 = result.entries[0]
    assert round0.worker_id == "worker-bridge"
    assert round0.matched is True
    assert round0.recruited is True
    assert round0.evidence_count == 1
    assert round0.unique_evidence_count == 0  # a.py/q1 is later duplicated by worker-base-a
    assert round0.resolution_before == "unresolved"
    assert round0.resolution_after == "partial"
    assert round0.need_reduction == 0
    assert round0.base_fallback_recruited is False  # neither parent recruited this round
    assert round0.structural_action == "birth"
    assert sorted(round0.parent_worker_ids) == ["worker-base-a", "worker-base-b"]
    assert round0.generation_created == 2

    round1 = result.entries[1]
    assert round1.worker_id == "worker-bridge"
    assert round1.matched is True
    assert round1.recruited is True
    assert round1.evidence_count == 1
    assert round1.unique_evidence_count == 1  # b.py/q2 is genuinely unique to worker-bridge
    assert round1.resolution_after == "resolved"
    assert round1.need_reduction == 1
    # worker-base-a (one of worker-bridge's own parents) was ALSO recruited
    # this same round.
    assert round1.base_fallback_recruited is True


def test_compute_question_attribution_never_reports_a_base_worker() -> None:
    state = _fixture_state()
    result = compute_question_attribution("q1", state, _fixture_workers())

    assert all(entry.worker_id != "worker-base" for entry in result.entries)
    assert all(entry.worker_id != "worker-base-a" for entry in result.entries)


def test_compute_question_attribution_does_not_mutate_its_input_state() -> None:
    state = _fixture_state()
    before = copy.deepcopy(state)

    compute_question_attribution("q1", state, _fixture_workers())

    assert state == before


def test_aggregate_generation_worker_usage_uses_distinct_task_counts_not_raw_entries() -> None:
    state = _fixture_state()
    workers = _fixture_workers()
    q1 = compute_question_attribution("q1", state, workers)
    # Same worker recruited again in a second question -- distinct task
    # counting must not just sum raw entry counts across questions.
    q2 = compute_question_attribution("q2", state, workers)

    summaries = aggregate_generation_worker_usage([q1, q2])
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.worker_id == "worker-bridge"
    assert summary.matched_task_count == 2
    assert summary.recruited_task_count == 2
    assert summary.rounds_used == 4  # 2 rounds recruited per question x 2 questions
    assert summary.evidence_contributed == 4  # 2 items per question x 2 questions
    assert summary.unique_evidence_contributed == 2  # 1 unique item per question x 2
    assert summary.tasks_with_progress == 2
    assert summary.tasks_with_resolved_need == 2
    assert summary.base_fallback_task_count == 2
