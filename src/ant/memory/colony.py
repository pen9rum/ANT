from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field

from ant.domain import EvidenceState, PlanningRound, UnresolvedNeed
from ant.retrieval.relevance import extract_terms


class CoalitionRecord(BaseModel):
    worker_ids: list[str]
    question: str
    evidence_count: int
    unresolved_need_count: int


class CollaborationEpisode(BaseModel):
    """One row per need-driven round of a finished task: what was needed,
    who was recruited, what strategy handled it (a plain follow-up, a
    coalition, or which escalation tactic), and whether it actually
    produced anything. This is what evolve_workers reads (via
    ColonyMemoryStore.aggregate_episodes) to notice "escalation tactic X
    keeps being the one that works for this kind of need", instead of only
    seeing raw worker co-occurrence the way recurring_coalitions() does.
    """

    need: str
    workers: list[str]
    strategy: str
    outcome: str
    evidence_gain: int
    # Nodes this execution itself closed directly (0 or 1 -- see
    # NodeExecutionTrace.need_reduction's docstring on why closure-derived
    # parent resolutions are deliberately NOT folded in here: attributing
    # a round-level closure to one specific collaboration would be
    # arbitrary). Lets evolve_workers see "this strategy actually closes
    # needs", not just "this strategy finds evidence" -- a strategy can
    # gather plenty of evidence_gain while never actually resolving
    # anything.
    need_reduction: int = 0
    # One value per real record_task_memory() call (i.e. per finished
    # ask()/retry_from_trajectory task), shared by every episode that call
    # records -- lets aggregate_episodes tell "this pattern recurred across
    # N separate tasks" apart from "this task got stuck and recruited the
    # same pattern N times across its own rounds". Defaulted (not required)
    # so existing call sites/tests that construct one CollaborationEpisode
    # per intended-distinct-task keep working unchanged -- each gets its
    # own random id, which is exactly the semantics they already relied on.
    task_id: str = Field(default_factory=lambda: uuid4().hex)


class EpisodeAggregate(BaseModel):
    """A collaboration pattern aggregated across every recorded episode row
    for one (strategy, exact worker set) group -- `occurrences` counts
    EVERY such row, which is one per round this pattern was recruited, so a
    single task stuck on the same need for 6 rounds contributes 6 all by
    itself (confirmed live: a 25-occurrence pattern traced back to only 4
    distinct tasks). `unique_task_count`/`tasks_with_progress`/
    `tasks_with_need_reduction` (see CollaborationEpisode.task_id) are the
    task-level counterpart: how many *separate* tasks this pattern actually
    recurred across, which is the stronger recurrence signal -- within-task
    repetition from one stuck task is much weaker structural evidence than
    the same pattern independently recurring across several different
    tasks. need_terms is a supplementary union of vocabulary seen across
    those occurrences, not part of the grouping key -- the (strategy,
    workers) pair is a far more stable identifier than independently
    LLM-worded need text ever is.
    """

    strategy: str
    workers: list[str]
    need_terms: list[str]
    occurrences: int
    successes: int
    total_evidence_gain: int
    total_need_reduction: int = 0
    unique_task_count: int = 0
    tasks_with_progress: int = 0
    tasks_with_need_reduction: int = 0


class MemoryRoute(BaseModel):
    need_terms: list[str] = Field(default_factory=list)
    worker_ids: list[str] = Field(default_factory=list)
    weight: float = 1.0
    # True: this recruitment also produced a good final answer, so it's
    # trustworthy as a *routing* precedent (matching_routes/_memory_route_bonus
    # use only these). False: the recruitment still happened -- this worker
    # really was asked about this need -- but the answer wasn't good enough
    # to trust for future routing. Specialization only needs "was this
    # worker recruited for this sub-topic", not "did the answer score well",
    # so evolve_workers reads all_routes() unfiltered by this flag.
    is_high_quality: bool = True
    # need_type/scope: carried over from the UnresolvedNeed that produced
    # this route, when there was one (see record_task_memory's per-need
    # routes below) -- "" for the task-level aggregate route, which isn't
    # tied to any single need. Without these, evolve_workers could see THAT
    # a worker struggled but never WHAT KIND of gap it struggled with (a
    # missing implementation vs. an absence question vs. a cross-territory
    # dependency), collapsing every distinct failure mode into the same
    # generic "low quality" signal.
    need_type: str = ""
    scope: str = ""


class ColonyMemoryStore:
    def __init__(self, index_path: Path) -> None:
        self.db_path = index_path / "ant.sqlite3"

    def record_coalition(self, record: CoalitionRecord) -> None:
        with sqlite3.connect(self.db_path) as connection:
            _create_schema(connection)
            connection.execute(
                "insert into coalitions(worker_ids, question, payload) values (?, ?, ?)",
                (
                    ",".join(record.worker_ids),
                    record.question,
                    record.model_dump_json(),
                ),
            )

    def record_episode(self, episode: CollaborationEpisode) -> None:
        with sqlite3.connect(self.db_path) as connection:
            _create_schema(connection)
            connection.execute(
                "insert into episodes(need, strategy, outcome, evidence_gain, payload) "
                "values (?, ?, ?, ?, ?)",
                (
                    episode.need,
                    episode.strategy,
                    episode.outcome,
                    episode.evidence_gain,
                    episode.model_dump_json(),
                ),
            )

    def aggregate_episodes(self, min_count: int = 2) -> list[EpisodeAggregate]:
        """Groups recorded episodes by (strategy, exact worker set) across
        every task recorded so far -- deliberately not by need text, which
        is independently LLM-worded per task and rarely byte-identical
        even for the same underlying gap. (strategy, workers) is the far
        more stable identifier: it directly answers "did this specific
        temporary adaptation keep working for this specific pairing".
        need_terms is a supplementary union of vocabulary across the
        group's occurrences, not part of the grouping key.
        """
        with sqlite3.connect(self.db_path) as connection:
            _create_schema(connection)
            rows = connection.execute(
                "select id, need, strategy, outcome, evidence_gain, payload from episodes"
            ).fetchall()
        groups: dict[tuple[str, tuple[str, ...]], list[tuple[str, int, int, list[str], str]]] = (
            defaultdict(list)
        )
        for row_id, need, strategy, outcome, evidence_gain, payload in rows:
            payload_data = json.loads(payload)
            workers = tuple(sorted(payload_data.get("workers", [])))
            if not workers:
                continue
            need_reduction = int(payload_data.get("need_reduction", 0))
            # Rows recorded before task_id existed have none in their own
            # payload -- fall back to this row's own db id so each still
            # counts as its own distinct pseudo-task (the honest reading of
            # data where real task grouping was never recorded), rather
            # than silently collapsing them all into one false "task".
            task_id = payload_data.get("task_id") or f"__legacy_row_{row_id}"
            groups[(strategy, workers)].append(
                (outcome, evidence_gain, need_reduction, extract_terms(need)[:6], task_id)
            )
        aggregates: list[EpisodeAggregate] = []
        for (strategy, workers), items in groups.items():
            if len(items) < min_count:
                continue
            successes = sum(1 for outcome, _, _, _, _ in items if outcome == "progress")
            total_gain = sum(gain for _, gain, _, _, _ in items)
            total_reduction = sum(reduction for _, _, reduction, _, _ in items)
            need_terms = sorted({term for _, _, _, terms, _ in items for term in terms})[:12]
            task_ids = {task_id for _, _, _, _, task_id in items}
            tasks_with_progress = {
                task_id for outcome, _, _, _, task_id in items if outcome == "progress"
            }
            tasks_with_need_reduction = {
                task_id for _, _, reduction, _, task_id in items if reduction > 0
            }
            aggregates.append(
                EpisodeAggregate(
                    strategy=strategy,
                    workers=list(workers),
                    need_terms=need_terms,
                    occurrences=len(items),
                    successes=successes,
                    total_evidence_gain=total_gain,
                    total_need_reduction=total_reduction,
                    unique_task_count=len(task_ids),
                    tasks_with_progress=len(tasks_with_progress),
                    tasks_with_need_reduction=len(tasks_with_need_reduction),
                )
            )
        aggregates.sort(key=lambda item: item.occurrences, reverse=True)
        return aggregates

    def distinct_task_count(self) -> int:
        """How many distinct real tasks (record_task_memory calls) have
        ever contributed an episode to this colony's memory -- the same
        task_id de-duplication aggregate_episodes' unique_task_count relies
        on, exposed on its own as a per-generation audit metric (how much
        real experience has this colony accumulated by the time a given
        evolve_workers() call runs). Rows from before task_id existed each
        count as their own distinct pseudo-task, the same conservative
        fallback aggregate_episodes uses.
        """
        with sqlite3.connect(self.db_path) as connection:
            _create_schema(connection)
            rows = connection.execute("select id, payload from episodes").fetchall()
        task_ids = {
            json.loads(payload).get("task_id") or f"__legacy_row_{row_id}"
            for row_id, payload in rows
        }
        return len(task_ids)

    def recurring_coalitions(self, min_count: int = 2) -> list[tuple[list[str], int]]:
        with sqlite3.connect(self.db_path) as connection:
            _create_schema(connection)
            rows = connection.execute(
                """
                select worker_ids, count(*) as n
                from coalitions
                where stale = 0
                group by worker_ids
                having n >= ?
                order by n desc
                """,
                (min_count,),
            ).fetchall()
        return [(row[0].split(","), int(row[1])) for row in rows if row[0]]

    def save_route(self, route: MemoryRoute) -> None:
        with sqlite3.connect(self.db_path) as connection:
            _create_schema(connection)
            connection.execute(
                "insert into routes(need_terms, worker_ids, weight, is_high_quality, "
                "need_type, scope) values (?, ?, ?, ?, ?, ?)",
                (
                    json.dumps(route.need_terms),
                    json.dumps(route.worker_ids),
                    route.weight,
                    int(route.is_high_quality),
                    route.need_type,
                    route.scope,
                ),
            )

    def all_routes(self, include_stale: bool = False) -> list[MemoryRoute]:
        # Deliberately unfiltered by is_high_quality: this is the accessor
        # evolve_workers()/_specialize_overloaded_workers uses, and
        # specialization only needs "was this worker recruited for this
        # sub-topic", not "did the final answer score well" -- see
        # MemoryRoute.is_high_quality.
        query = (
            "select need_terms, worker_ids, weight, is_high_quality, need_type, scope from routes"
        )
        if not include_stale:
            query += " where stale = 0"
        with sqlite3.connect(self.db_path) as connection:
            _create_schema(connection)
            rows = connection.execute(query).fetchall()
        return [
            MemoryRoute(
                need_terms=[str(term) for term in json.loads(need_terms_json)],
                worker_ids=[str(worker_id) for worker_id in json.loads(worker_ids_json)],
                weight=float(weight),
                is_high_quality=bool(is_high_quality),
                need_type=need_type or "",
                scope=scope or "",
            )
            for need_terms_json, worker_ids_json, weight, is_high_quality, need_type, scope in rows
        ]

    def matching_routes(self, terms: list[str], limit: int = 5) -> list[MemoryRoute]:
        # Filtered to is_high_quality: this is the accessor the query-time
        # routing bonus (_memory_route_bonus) uses, and a route from a task
        # that didn't actually produce a good answer would actively steer
        # future routing astray if trusted as a precedent.
        if not terms:
            return []
        query_terms = {term.lower() for term in terms}
        routes = []
        with sqlite3.connect(self.db_path) as connection:
            _create_schema(connection)
            rows = connection.execute(
                """
                select need_terms, worker_ids, weight, need_type, scope from routes
                where stale = 0 and is_high_quality = 1
                order by weight desc, id desc
                """
            ).fetchall()
        for need_terms_json, worker_ids_json, weight, need_type, scope in rows:
            need_terms = [str(term) for term in json.loads(need_terms_json)]
            overlap = query_terms & {term.lower() for term in need_terms}
            if not overlap:
                continue
            routes.append(
                (
                    len(overlap),
                    float(weight),
                    MemoryRoute(
                        need_terms=need_terms,
                        worker_ids=[str(worker_id) for worker_id in json.loads(worker_ids_json)],
                        weight=float(weight),
                        need_type=need_type or "",
                        scope=scope or "",
                    ),
                )
            )
        routes.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [route for _, _, route in routes[:limit]]

    def mark_stale(
        self,
        affected_worker_ids: list[str],
        reason: str = "Territory changed; dependent memory requires revalidation.",
    ) -> int:
        """Mark every route/coalition that depends on any of `affected_worker_ids`
        as stale, so `matching_routes`/`recurring_coalitions` stop serving it
        until `revalidate_stale` has had a chance to refresh, repair, or
        discard it. Staleness is tracked by worker id (the unit workers.json
        actually evolves at), not raw file path, so it stays meaningful across
        card regeneration and worker birth/merge/retire."""
        affected = {worker_id for worker_id in affected_worker_ids if worker_id}
        if not affected:
            return 0
        count = 0
        with sqlite3.connect(self.db_path) as connection:
            _create_schema(connection)
            rows = connection.execute(
                "select id, worker_ids from routes where stale = 0"
            ).fetchall()
            for row_id, worker_ids_json in rows:
                worker_ids = set(json.loads(worker_ids_json))
                if worker_ids & affected:
                    connection.execute("update routes set stale = 1 where id = ?", (row_id,))
                    count += 1
            rows = connection.execute(
                "select id, worker_ids from coalitions where stale = 0"
            ).fetchall()
            for row_id, worker_ids_csv in rows:
                worker_ids = set(worker_ids_csv.split(",")) if worker_ids_csv else set()
                if worker_ids & affected:
                    connection.execute("update coalitions set stale = 1 where id = ?", (row_id,))
                    count += 1
            for worker_id in sorted(affected):
                connection.execute(
                    "insert into stale_memory(path, worker_id, reason) values ('', ?, ?)",
                    (worker_id, reason),
                )
        return count

    def revalidate_stale(self, current_worker_ids: set[str]) -> dict[str, int]:
        """Resolve every stale route/coalition against the colony's current
        worker ids: still fully supported -> refresh (clear stale, keep as
        is); a route with some but not all of its workers gone -> repair
        (narrow to the survivors, since a partial fallback path is still
        useful); nothing left -> discard (delete). A coalition is an exact
        collaboration pattern, not a fallback list, so it only refreshes if
        every member survives and is discarded otherwise."""
        refreshed = 0
        repaired = 0
        discarded = 0
        with sqlite3.connect(self.db_path) as connection:
            _create_schema(connection)
            rows = connection.execute(
                "select id, worker_ids from routes where stale = 1"
            ).fetchall()
            for row_id, worker_ids_json in rows:
                worker_ids = json.loads(worker_ids_json)
                surviving = [
                    worker_id for worker_id in worker_ids if worker_id in current_worker_ids
                ]
                if not surviving:
                    connection.execute("delete from routes where id = ?", (row_id,))
                    discarded += 1
                elif len(surviving) == len(worker_ids):
                    connection.execute("update routes set stale = 0 where id = ?", (row_id,))
                    refreshed += 1
                else:
                    connection.execute(
                        "update routes set worker_ids = ?, stale = 0 where id = ?",
                        (json.dumps(surviving), row_id),
                    )
                    repaired += 1

            rows = connection.execute(
                "select id, worker_ids from coalitions where stale = 1"
            ).fetchall()
            for row_id, worker_ids_csv in rows:
                worker_ids = worker_ids_csv.split(",") if worker_ids_csv else []
                if worker_ids and all(worker_id in current_worker_ids for worker_id in worker_ids):
                    connection.execute("update coalitions set stale = 0 where id = ?", (row_id,))
                    refreshed += 1
                else:
                    connection.execute("delete from coalitions where id = ?", (row_id,))
                    discarded += 1

            connection.execute(
                "update stale_memory set resolved_at = current_timestamp where resolved_at is null"
            )
        return {"refreshed": refreshed, "repaired": repaired, "discarded": discarded}


def record_task_memory(
    colony_memory: ColonyMemoryStore,
    question: str,
    state: EvidenceState,
    *,
    is_high_quality: bool | None = None,
    route_weight: float | None = None,
) -> None:
    """Persist what a finished task learned into Colony Memory.

    This must run after every `ask()` -- not only inside a batch eval run --
    otherwise the colony never actually learns from real usage and the
    "repeated collaboration becomes reorganization evidence" mechanism the
    whole design is built around silently never fires.

    Coalition occurrences and routes are now both recorded unconditionally
    (`is_high_quality` tags the route rather than gating whether it gets
    saved at all): a low-scoring task still proves "this worker was
    recruited for this need", which is exactly the signal
    `evolve_workers`/`_specialize_overloaded_workers` mines to detect a
    territory covering genuinely different sub-areas. Gating route recording
    on answer quality used to mean a colony that was consistently *struggling*
    on a territory -- precisely the case specialization exists to fix --
    accumulated no evidence to specialize from at all. The quality gate still
    matters for the query-time routing bonus (`matching_routes`, used by
    `_memory_route_bonus`): a bad route actively steers future routing
    astray if trusted as a precedent, so that accessor filters to
    `is_high_quality` routes only, while `all_routes()` (evolve_workers'
    input) does not. Callers with a judge score should pass
    `is_high_quality`/`route_weight` explicitly; callers without one (e.g.
    interactive `ask`) fall back to a judge-free signal: the task ended with
    grounded evidence and no unresolved needs left.
    """
    # Dedupe by membership within this task before writing: a need that
    # stayed stuck across several rounds can form a coalition with the
    # exact same worker pair every one of those rounds (confirmed directly
    # on a real yt-dlp trace, pre-graph-pipeline -- 5 rounds stuck on one
    # need inserted 5 identical coalition rows). recurring_coalitions()
    # counts raw rows, so one buggy or merely slow task could masquerade
    # as several independent instances of genuine recurring collaboration
    # and misfire a birth. One row per unique membership per task keeps
    # that count meaning what it says: how many separate TASKS this
    # pairing recurred in. A coalition's full membership is available
    # directly from a single NodeExecutionTrace.worker_ids now (unlike the
    # old RecruitmentRound-per-round shape, which only ever recruited one
    # new worker per round and needed reconstructing membership from prior
    # rounds -- see the removed _coalition_membership).
    seen_memberships: set[tuple[str, ...]] = set()
    for round_state in state.rounds:
        for trace in round_state.node_executions:
            if not trace.coalition_formed:
                continue
            membership = tuple(sorted(trace.worker_ids))
            if not membership or membership in seen_memberships:
                continue
            seen_memberships.add(membership)
            colony_memory.record_coalition(
                CoalitionRecord(
                    worker_ids=list(membership),
                    question=question,
                    evidence_count=len(state.evidence),
                    unresolved_need_count=len(state.unresolved_needs),
                )
            )

    # One episode per node execution: what was needed, who was recruited,
    # what strategy handled it (a plain follow-up, a coalition, or which
    # special tactic), whether it actually produced anything, and how many
    # needs it directly closed. Closure-derived parent resolutions
    # (PlanningRound.derived_resolved_nodes) are deliberately NOT recorded
    # as episodes here -- a closure check isn't a collaboration (no
    # strategy/workers to attribute it to), it's verification that a
    # decomposition already covered its parent.
    # One task_id shared by every episode this call records -- lets
    # aggregate_episodes distinguish "this pattern recurred across several
    # separate tasks" (strong recurrence signal) from "this one task got
    # stuck and recruited the same pattern across many of its own rounds"
    # (weak: it's the same task's own struggle counted multiple times). See
    # CollaborationEpisode.task_id's own docstring.
    task_id = uuid4().hex
    for round_state in state.rounds:
        for trace in round_state.node_executions:
            if not trace.need:
                continue
            colony_memory.record_episode(
                CollaborationEpisode(
                    need=trace.need,
                    workers=trace.worker_ids,
                    strategy=(
                        trace.special_tactic
                        or ("coalition" if trace.coalition_formed else "normal")
                    ),
                    outcome=(
                        "progress"
                        if trace.evidence_gain > 0 or trace.need_reduction > 0
                        else "no_progress"
                    ),
                    evidence_gain=trace.evidence_gain,
                    need_reduction=trace.need_reduction,
                    task_id=task_id,
                )
            )

    worker_ids = _selected_route_workers(state.rounds)
    if not worker_ids:
        return
    high_quality = is_high_quality if is_high_quality is not None else _task_fully_resolved(state)
    colony_memory.save_route(
        MemoryRoute(
            need_terms=_route_terms(question),
            worker_ids=worker_ids,
            weight=route_weight if route_weight is not None else 2.0,
            is_high_quality=high_quality,
        )
    )

    # One additional route per need that was STILL open when the task ended
    # (not every need that was ever raised -- one resolved by a later round
    # or a coalition isn't a real gap, it's the mechanism working). Unlike
    # the aggregate route above, this carries the need's own need_type and
    # scope, tagged to the specific worker that raised it and always
    # is_high_quality=False by construction (a need that's still open at
    # task end is, by definition, a case that worker didn't resolve) -- so
    # it never leaks into matching_routes()'s query-time routing bonus, only
    # into evolve_workers' all_routes(), where it's the signal this whole
    # mechanism exists to supply: not just THAT a worker struggled, but
    # WHAT KIND of gap it struggled with.
    for need in state.unresolved_needs:
        if not need.source_worker_id:
            continue
        colony_memory.save_route(
            MemoryRoute(
                need_terms=_need_route_terms(need),
                worker_ids=[need.source_worker_id],
                weight=1.0,
                is_high_quality=False,
                need_type=need.need_type,
                scope=need.scope,
            )
        )


def _task_fully_resolved(state: EvidenceState) -> bool:
    return state.has_evidence() and not state.unresolved_needs


def _selected_route_workers(rounds: list[PlanningRound]) -> list[str]:
    worker_ids = [
        worker_id
        for round_state in rounds
        for trace in round_state.node_executions
        for worker_id in trace.worker_ids
    ]
    return list(dict.fromkeys(worker_ids))


def _need_route_terms(need: UnresolvedNeed) -> list[str]:
    # suggested_terms first (the reasoner's own pick of what's relevant),
    # topped up with terms extracted from `missing`/`description` so a need
    # whose reasoner call didn't populate suggested_terms still yields
    # something to group by -- extract_terms (stopword-filtered, camelCase
    # aware) rather than the coarser _route_terms tokenizer below, since
    # this text is about a specific technical gap, not a free-form question.
    terms = list(dict.fromkeys(term.lower() for term in need.suggested_terms if term))
    if len(terms) < 6:
        extra = extract_terms(need.missing or need.description)
        terms.extend(term for term in extra if term not in terms)
    return terms[:16]


def _route_terms(question: str) -> list[str]:
    return sorted(
        {
            token.lower()
            for token in question.replace("_", " ").split()
            if len(token) > 2 and token.isascii()
        }
    )[:16]


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        create table if not exists coalitions (
            id integer primary key autoincrement,
            worker_ids text not null,
            question text not null,
            payload text not null,
            created_at text not null default current_timestamp
        );
        create table if not exists routes (
            id integer primary key autoincrement,
            need_terms text not null,
            worker_ids text not null,
            weight real not null,
            created_at text not null default current_timestamp
        );
        create table if not exists stale_memory (
            id integer primary key autoincrement,
            path text,
            reason text not null,
            created_at text not null default current_timestamp,
            resolved_at text
        );
        create table if not exists episodes (
            id integer primary key autoincrement,
            need text not null,
            strategy text not null,
            outcome text not null,
            evidence_gain integer not null,
            payload text not null,
            created_at text not null default current_timestamp
        );
        """
    )
    _ensure_column(connection, "coalitions", "stale", "integer not null default 0")
    _ensure_column(connection, "routes", "stale", "integer not null default 0")
    # Default 1 (True): every route saved before this column existed was, by
    # construction, one that already passed the old quality gate at save
    # time -- see record_task_memory / matching_routes vs. all_routes below.
    _ensure_column(connection, "routes", "is_high_quality", "integer not null default 1")
    # "" default: routes saved before these columns existed (and the
    # task-level aggregate route record_task_memory still saves today) carry
    # no single need_type/scope -- only the new per-need routes below do.
    _ensure_column(connection, "routes", "need_type", "text not null default ''")
    _ensure_column(connection, "routes", "scope", "text not null default ''")
    _ensure_column(connection, "stale_memory", "worker_id", "text")


def _ensure_column(
    connection: sqlite3.Connection, table: str, column: str, declaration: str
) -> None:
    existing = {row[1] for row in connection.execute(f"pragma table_info({table})")}
    if column not in existing:
        connection.execute(f"alter table {table} add column {column} {declaration}")
