from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field


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

    def mark_stale(self, changed_files: list[str]) -> int:
        if not changed_files:
            return 0
        count = 0
        with sqlite3.connect(self.db_path) as connection:
            _create_schema(connection)
            for file in changed_files:
                cursor = connection.execute(
                    "insert into stale_memory(path, reason) values (?, ?)",
                    (file, "Changed by git diff; dependent memories require revalidation."),
                )
                if cursor.rowcount:
                    count += cursor.rowcount
        return count

    def revalidate_stale(self, repo_root: Path) -> dict[str, int]:
        repaired = 0
        discarded = 0
        with sqlite3.connect(self.db_path) as connection:
            _create_schema(connection)
            rows = connection.execute(
                "select id, path from stale_memory where resolved_at is null"
            ).fetchall()
            for row_id, path in rows:
                status = "revalidated" if (repo_root / path).exists() else "discarded"
                if status == "revalidated":
                    repaired += 1
                else:
                    discarded += 1
                connection.execute(
                    """
                    update stale_memory
                    set resolved_at = current_timestamp, reason = reason || ? 
                    where id = ?
                    """,
                    (f" Revalidation status: {status}.", row_id),
                )
        return {"revalidated": repaired, "discarded": discarded}


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
            path text not null,
            reason text not null,
            created_at text not null default current_timestamp,
            resolved_at text
        );
        """
    )
