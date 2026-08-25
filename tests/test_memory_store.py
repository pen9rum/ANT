from pathlib import Path

from typer.testing import CliRunner

from ant.cli import app
from ant.domain import (
    CodeSymbol,
    Evidence,
    EvidenceState,
    RecruitmentRound,
    Territory,
    UnresolvedNeed,
    WorkerCard,
)
from ant.memory import ColonyMemoryStore, IndexStore, MemoryRoute
from ant.memory.colony import CoalitionRecord, record_task_memory


def test_index_store_persists_workers_and_traces(tmp_path: Path) -> None:
    territory = Territory(id="src", root="src", files=["src/app.py"], summary="Owns source.")
    worker = WorkerCard(
        id="worker-src",
        territory_id="src",
        name="src worker",
        root="src",
        files=["src/app.py"],
        symbols=[
            CodeSymbol(
                name="App",
                kind="class",
                path="src/app.py",
                line=1,
                qualname="App",
            )
        ],
    )
    store = IndexStore(tmp_path / ".ant")

    store.save([territory], [worker])
    trace_id = store.save_trace(EvidenceState(question="Where is app?"))

    assert store.load_workers() == [worker]
    assert trace_id == 1
    assert (tmp_path / ".ant" / "ant.sqlite3").exists()
    assert (tmp_path / ".ant" / "symbols.json").exists()
    assert store.load_symbols()[0].name == "App"

    result = CliRunner().invoke(
        app, ["symbols", "--index", str(tmp_path / ".ant"), "--query", "App"]
    )
    assert result.exit_code == 0
    assert '"worker_id": "worker-src"' in result.stdout


def test_colony_memory_returns_matching_routes(tmp_path: Path) -> None:
    memory = ColonyMemoryStore(tmp_path)
    memory.save_route(
        MemoryRoute(
            need_terms=["measurement", "sample_shots"],
            worker_ids=["worker-backends"],
            weight=2.5,
        )
    )
    memory.save_route(
        MemoryRoute(
            need_terms=["draw"],
            worker_ids=["worker-models"],
            weight=3.0,
        )
    )

    routes = memory.matching_routes(["How", "measurement", "works"])

    assert len(routes) == 1
    assert routes[0].worker_ids == ["worker-backends"]


def test_mark_stale_hides_route_until_revalidated(tmp_path: Path) -> None:
    memory = ColonyMemoryStore(tmp_path)
    memory.save_route(
        MemoryRoute(need_terms=["auth"], worker_ids=["worker-auth"], weight=2.0)
    )

    marked = memory.mark_stale(["worker-auth"])

    assert marked == 1
    assert memory.matching_routes(["auth"]) == []

    outcome = memory.revalidate_stale({"worker-auth"})

    assert outcome == {"refreshed": 1, "repaired": 0, "discarded": 0}
    assert memory.matching_routes(["auth"])[0].worker_ids == ["worker-auth"]


def test_revalidate_repairs_multi_worker_route_when_one_worker_survives(
    tmp_path: Path,
) -> None:
    memory = ColonyMemoryStore(tmp_path)
    memory.save_route(
        MemoryRoute(
            need_terms=["auth"],
            worker_ids=["worker-auth", "worker-session"],
            weight=2.0,
        )
    )
    memory.mark_stale(["worker-session"])

    outcome = memory.revalidate_stale({"worker-auth"})

    assert outcome == {"refreshed": 0, "repaired": 1, "discarded": 0}
    assert memory.matching_routes(["auth"])[0].worker_ids == ["worker-auth"]


def test_revalidate_discards_route_when_no_workers_survive(tmp_path: Path) -> None:
    memory = ColonyMemoryStore(tmp_path)
    memory.save_route(
        MemoryRoute(need_terms=["auth"], worker_ids=["worker-auth"], weight=2.0)
    )
    memory.mark_stale(["worker-auth"])

    outcome = memory.revalidate_stale(set())

    assert outcome == {"refreshed": 0, "repaired": 0, "discarded": 1}
    assert memory.matching_routes(["auth"]) == []


def test_recurring_coalitions_excludes_stale_entries_until_revalidated(
    tmp_path: Path,
) -> None:
    memory = ColonyMemoryStore(tmp_path)
    for _ in range(2):
        memory.record_coalition(
            CoalitionRecord(
                worker_ids=["worker-auth", "worker-session"],
                question="How does auth use sessions?",
                evidence_count=4,
                unresolved_need_count=0,
            )
        )

    assert memory.recurring_coalitions(min_count=2) == [
        (["worker-auth", "worker-session"], 2)
    ]

    memory.mark_stale(["worker-session"])

    assert memory.recurring_coalitions(min_count=2) == []

    memory.revalidate_stale({"worker-auth", "worker-session"})

    assert memory.recurring_coalitions(min_count=2) == [
        (["worker-auth", "worker-session"], 2)
    ]


def _round(
    *, selected: list[str], coalition_formed: bool, round_index: int = 0
) -> RecruitmentRound:
    return RecruitmentRound(
        round_index=round_index,
        query="q",
        selected_worker_ids=selected,
        rationale="test round",
        coalition_formed=coalition_formed,
    )


def test_record_task_memory_records_coalition_and_high_quality_route(
    tmp_path: Path,
) -> None:
    memory = ColonyMemoryStore(tmp_path)
    state = EvidenceState(
        question="How does auth use sessions?",
        evidence=[
            Evidence(
                path="src/auth.py",
                line_start=1,
                line_end=2,
                quote="def authenticate(): ...",
                reason="definition",
            )
        ],
        rounds=[
            _round(selected=["worker-auth"], coalition_formed=False, round_index=0),
            _round(selected=["worker-session"], coalition_formed=True, round_index=1),
        ],
    )

    record_task_memory(memory, state.question, state)

    # The coalition record reflects the FULL membership -- the worker
    # selected this round plus every worker selected in earlier rounds of
    # the same task -- not just this round's own single new recruit, since
    # evolve_workers' recurring-coalition birth requires >=2 members and
    # would never fire on a single-worker "coalition". The route separately
    # reflects the full accumulated path across every round of the task.
    assert memory.recurring_coalitions(min_count=1) == [
        (["worker-auth", "worker-session"], 1)
    ]
    routes = memory.matching_routes(["auth", "sessions"])
    assert routes and routes[0].worker_ids == ["worker-auth", "worker-session"]


def test_coalition_membership_is_order_independent_across_tasks(tmp_path: Path) -> None:
    # Task 1 recruits worker-session first, then worker-auth joins. Task 2
    # recruits the same real pair in the opposite order. Both are the same
    # underlying coalition and must count as one recurring pattern, not two
    # separate one-off patterns that individually never reach the
    # recurrence threshold evolve_workers checks against.
    memory = ColonyMemoryStore(tmp_path)
    task_one = EvidenceState(
        question="q1",
        evidence=[Evidence(path="a.py", line_start=1, line_end=1, quote="x", reason="r")],
        rounds=[
            _round(selected=["worker-session"], coalition_formed=False, round_index=0),
            _round(selected=["worker-auth"], coalition_formed=True, round_index=1),
        ],
    )
    task_two = EvidenceState(
        question="q2",
        evidence=[Evidence(path="a.py", line_start=1, line_end=1, quote="x", reason="r")],
        rounds=[
            _round(selected=["worker-auth"], coalition_formed=False, round_index=0),
            _round(selected=["worker-session"], coalition_formed=True, round_index=1),
        ],
    )

    record_task_memory(memory, task_one.question, task_one)
    record_task_memory(memory, task_two.question, task_two)

    assert memory.recurring_coalitions(min_count=2) == [
        (["worker-auth", "worker-session"], 2)
    ]


def test_record_task_memory_skips_route_when_unresolved_needs_remain(
    tmp_path: Path,
) -> None:
    memory = ColonyMemoryStore(tmp_path)
    state = EvidenceState(
        question="How does auth use sessions?",
        evidence=[
            Evidence(
                path="src/auth.py",
                line_start=1,
                line_end=2,
                quote="def authenticate(): ...",
                reason="definition",
            )
        ],
        unresolved_needs=[
            UnresolvedNeed(description="Still missing the session refresh path.")
        ],
        rounds=[_round(selected=["worker-auth"], coalition_formed=False)],
    )

    record_task_memory(memory, state.question, state)

    assert memory.matching_routes(["auth", "sessions"]) == []


def test_low_quality_route_is_recorded_for_specialization_but_hidden_from_routing(
    tmp_path: Path,
) -> None:
    # Regression test: routes used to be dropped entirely when the task
    # wasn't high quality, which meant a colony that was consistently
    # struggling on a territory -- precisely the case specialization exists
    # to fix -- accumulated no evidence for evolve_workers to specialize
    # from at all. The task's low quality must still keep it out of the
    # query-time routing bonus (matching_routes), but it must now be visible
    # to all_routes() (what _specialize_overloaded_workers reads).
    memory = ColonyMemoryStore(tmp_path)
    state = EvidenceState(
        question="How does auth use sessions?",
        evidence=[
            Evidence(
                path="src/auth.py",
                line_start=1,
                line_end=2,
                quote="def authenticate(): ...",
                reason="definition",
            )
        ],
        unresolved_needs=[
            UnresolvedNeed(description="Still missing the session refresh path.")
        ],
        rounds=[_round(selected=["worker-auth"], coalition_formed=False)],
    )

    record_task_memory(memory, state.question, state, is_high_quality=False)

    assert memory.matching_routes(["auth", "sessions"]) == []
    all_routes = memory.all_routes()
    assert len(all_routes) == 1
    assert all_routes[0].worker_ids == ["worker-auth"]
    assert all_routes[0].is_high_quality is False


def test_record_task_memory_saves_a_per_need_route_carrying_need_type_and_scope(
    tmp_path: Path,
) -> None:
    # Regression test: the task-level aggregate route only ever carried the
    # original question's generic terms and a single pass/fail flag --
    # evolve_workers had no way to see WHAT KIND of gap a worker struggled
    # with (missing an implementation vs. an unanswerable absence question
    # vs. needing another territory), only THAT it struggled at all. A need
    # still open when the task ends now gets its own route, tagged with its
    # own need_type/scope and attributed to the specific worker that raised
    # it, so evolve_workers' input actually distinguishes failure modes.
    memory = ColonyMemoryStore(tmp_path)
    state = EvidenceState(
        question="How does auth use sessions?",
        evidence=[
            Evidence(
                path="src/auth.py",
                line_start=1,
                line_end=2,
                quote="def authenticate(): ...",
                reason="definition",
            )
        ],
        unresolved_needs=[
            UnresolvedNeed(
                description="Still missing the session refresh path.",
                need_type="call_path",
                scope="cross_territory",
                missing="How the refresh token flows into re-authentication.",
                suggested_terms=["refresh_token", "reauthenticate"],
                source_worker_id="worker-auth",
            )
        ],
        rounds=[_round(selected=["worker-auth"], coalition_formed=False)],
    )

    record_task_memory(memory, state.question, state, is_high_quality=False)

    all_routes = memory.all_routes()
    # The task-level aggregate route (need_type/scope both "") plus the new
    # per-need route (need_type/scope populated, single-worker-attributed).
    assert len(all_routes) == 2
    per_need = [route for route in all_routes if route.need_type]
    assert len(per_need) == 1
    assert per_need[0].need_type == "call_path"
    assert per_need[0].scope == "cross_territory"
    assert per_need[0].worker_ids == ["worker-auth"]
    assert per_need[0].is_high_quality is False
    assert "refresh_token" in per_need[0].need_terms
    # Never leaks into the query-time routing bonus -- it's always
    # is_high_quality=False by construction.
    assert memory.matching_routes(["refresh_token", "reauthenticate"]) == []


def test_record_task_memory_does_not_save_a_per_need_route_for_a_resolved_need(
    tmp_path: Path,
) -> None:
    # A need that was raised mid-task but is no longer in state.unresolved_needs
    # by the time the task ends was resolved (by a later round or a
    # coalition) -- that is the mechanism working, not a gap, so it must not
    # get recorded as a struggle signal.
    memory = ColonyMemoryStore(tmp_path)
    state = EvidenceState(
        question="How does auth use sessions?",
        evidence=[
            Evidence(
                path="src/auth.py",
                line_start=1,
                line_end=2,
                quote="def authenticate(): ...",
                reason="definition",
            )
        ],
        unresolved_needs=[],
        rounds=[_round(selected=["worker-auth"], coalition_formed=False)],
    )

    record_task_memory(memory, state.question, state, is_high_quality=True)

    all_routes = memory.all_routes()
    assert len(all_routes) == 1
    assert all_routes[0].need_type == ""
