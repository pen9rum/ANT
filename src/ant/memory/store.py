from __future__ import annotations

import json
from pathlib import Path

from ant.domain import Territory, WorkerCard


class IndexStore:
    def __init__(self, path: Path) -> None:
        self.path = path

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

    def load_workers(self) -> list[WorkerCard]:
        data = json.loads((self.path / "workers.json").read_text(encoding="utf-8"))
        return [WorkerCard.model_validate(item) for item in data]
