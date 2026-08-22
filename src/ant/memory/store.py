from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ant.domain import CodeSymbol, EvidenceState, Territory, WorkerCard


class IndexStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.db_path = path / "ant.sqlite3"

    def save(self, territories: list[Territory], workers: list[WorkerCard]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "territories.json").write_text(
            json.dumps([item.model_dump() for item in territories], indent=2),
            encoding="utf-8",
        )
        (self.path / "workers.json").write_text(
            json.dumps([item.model_dump() for item in workers], indent=2),
            encoding="utf-8",
        )
        (self.path / "symbols.json").write_text(
            json.dumps(_symbol_manifest(workers), indent=2),
            encoding="utf-8",
        )
        with self._connect() as connection:
            self._create_schema(connection)
            connection.execute("delete from territories")
            connection.execute("delete from workers")
            connection.executemany(
                "insert into territories(id, root, payload) values (?, ?, ?)",
                [(item.id, item.root, item.model_dump_json()) for item in territories],
            )
            connection.executemany(
                "insert into workers(id, territory_id, root, payload) values (?, ?, ?, ?)",
                [
                    (item.id, item.territory_id, item.root, item.model_dump_json())
                    for item in workers
                ],
            )

    def load_workers(self) -> list[WorkerCard]:
        if self.db_path.exists():
            with self._connect() as connection:
                self._create_schema(connection)
                rows = connection.execute("select payload from workers order by id").fetchall()
            if rows:
                return [WorkerCard.model_validate_json(row[0]) for row in rows]

        data = json.loads((self.path / "workers.json").read_text(encoding="utf-8"))
        return [WorkerCard.model_validate(item) for item in data]

    def load_symbols(self) -> list[CodeSymbol]:
        return [
            CodeSymbol.model_validate(item["symbol"])
            for item in self.load_symbol_manifest()
            if isinstance(item.get("symbol"), dict)
        ]

    def load_symbol_manifest(self) -> list[dict[str, object]]:
        path = self.path / "symbols.json"
        if path.exists():
            return list(json.loads(path.read_text(encoding="utf-8")))
        manifest = []
        for worker in self.load_workers():
            manifest.extend(_symbol_manifest([worker]))
        return manifest

    def save_trace(self, state: EvidenceState) -> int:
        self.path.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            self._create_schema(connection)
            cursor = connection.execute(
                "insert into traces(question, payload) values (?, ?)",
                (state.question, state.model_dump_json()),
            )
            if cursor.lastrowid is None:
                msg = "SQLite did not return a trace id."
                raise RuntimeError(msg)
            return cursor.lastrowid

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            create table if not exists territories (
                id text primary key,
                root text not null,
                payload text not null
            );
            create table if not exists workers (
                id text primary key,
                territory_id text not null,
                root text not null,
                payload text not null
            );
            create table if not exists traces (
                id integer primary key autoincrement,
                question text not null,
                payload text not null,
                created_at text not null default current_timestamp
            );
            """
        )


def _symbol_manifest(workers: list[WorkerCard]) -> list[dict[str, object]]:
    manifest = []
    for worker in workers:
        for symbol in worker.symbols:
            manifest.append(
                {
                    "worker_id": worker.id,
                    "territory_id": worker.territory_id,
                    "symbol": symbol.model_dump(),
                }
            )
    return sorted(
        manifest,
        key=lambda item: (
            str(item["territory_id"]),
            str(item["worker_id"]),
            str(cast_symbol(item["symbol"]).get("path", "")),
            int(cast_symbol(item["symbol"]).get("line", 0)),
        ),
    )


def cast_symbol(value: object) -> dict:
    return value if isinstance(value, dict) else {}
