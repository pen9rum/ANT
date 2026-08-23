from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from ant.domain import EvidenceState, RecruitmentRound


class CoalitionRecord(BaseModel):
    worker_ids: list[str]
    question: str
    evidence_count: int
    unresolved_need_count: int


class MemoryRoute(BaseModel):
    need_terms: list[str] = Field(default_factory=list)
    worker_ids: list[str] = Field(default_factory=list)
    weight: float = 1.0


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
                "insert into routes(need_terms, worker_ids, weight) values (?, ?, ?)",
                (json.dumps(route.need_terms), json.dumps(route.worker_ids), route.weight),
            )

    def all_routes(self, include_stale: bool = False) -> list[MemoryRoute]:
        query = "select need_terms, worker_ids, weight from routes"
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
            )
            for need_terms_json, worker_ids_json, weight in rows
        ]

    def matching_routes(self, terms: list[str], limit: int = 5) -> list[MemoryRoute]:
        if not terms:
            return []
        query_terms = {term.lower() for term in terms}
        routes = []
        with sqlite3.connect(self.db_path) as connection:
            _create_schema(connection)
            rows = connection.execute(
                """
                select need_terms, worker_ids, weight from routes
                where stale = 0
                order by weight desc, id desc
                """
            ).fetchall()
        for need_terms_json, worker_ids_json, weight in rows:
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

    Coalition occurrences are recorded unconditionally: `evolve_workers`'
    recurring-coalition mining is a statistical signal that benefits from
    every occurrence, not only from ones judged "good". Route memory (which
    worker(s) answered a need well) is gated by `is_high_quality`, since a
    bad route actively steers future routing astray. Callers with a judge
    score should pass `is_high_quality`/`route_weight` explicitly; callers
    without one (e.g. interactive `ask`) fall back to a judge-free signal:
    the task ended with grounded evidence and no unresolved needs left.
    """
    for round_state in state.rounds:
        if round_state.coalition_formed:
            colony_memory.record_coalition(
                CoalitionRecord(
                    worker_ids=_coalition_membership(state.rounds, round_state.round_index),
                    question=question,
                    evidence_count=len(state.evidence),
                    unresolved_need_count=len(state.unresolved_needs),
                )
            )

    high_quality = is_high_quality if is_high_quality is not None else _task_fully_resolved(state)
    if not high_quality:
        return
    worker_ids = _selected_route_workers(state.rounds)
    if not worker_ids:
        return
    colony_memory.save_route(
        MemoryRoute(
            need_terms=_route_terms(question),
            worker_ids=worker_ids,
            weight=route_weight if route_weight is not None else 2.0,
        )
    )


def _coalition_membership(rounds: list[RecruitmentRound], round_index: int) -> list[str]:
    """A coalition-forming round only ever selects the ONE new worker being
    recruited that round (`candidates[:1]` in the coordinator's round loop);
    the workers it is joining forces WITH came from earlier rounds. Using
    just `round.selected_worker_ids` therefore records a single-worker
    "coalition" every time -- `evolve_workers`'s recurring-coalition birth
    requires >=2 members, so no coalition could ever actually trigger
    reorganization, no matter how many times the same real, multi-worker
    group recurred. Reconstruct the full membership the same way
    `coordinator.local._last_coalition_workers` does for answer synthesis:
    every worker selected in this round plus every prior round of the task.
    """
    prior = [
        worker_id
        for earlier in rounds[:round_index]
        for worker_id in earlier.selected_worker_ids
    ]
    current = rounds[round_index].selected_worker_ids
    return list(dict.fromkeys([*prior, *current]))


def _task_fully_resolved(state: EvidenceState) -> bool:
    return state.has_evidence() and not state.unresolved_needs


def _selected_route_workers(rounds) -> list[str]:
    worker_ids = [
        worker_id for round_state in rounds for worker_id in round_state.selected_worker_ids
    ]
    return list(dict.fromkeys(worker_ids))


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
        """
    )
    _ensure_column(connection, "coalitions", "stale", "integer not null default 0")
    _ensure_column(connection, "routes", "stale", "integer not null default 0")
    _ensure_column(connection, "stale_memory", "worker_id", "text")


def _ensure_column(
    connection: sqlite3.Connection, table: str, column: str, declaration: str
) -> None:
    existing = {row[1] for row in connection.execute(f"pragma table_info({table})")}
    if column not in existing:
        connection.execute(f"alter table {table} add column {column} {declaration}")
