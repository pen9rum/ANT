from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from ant.domain import Territory
from ant.environment import RepoEnvironment
from ant.generation import generate_worker_cards
from ant.indexing import discover_territories
from ant.memory import IndexStore
from ant.providers import CardGenerator


class RefreshResult(BaseModel):
    changed_files: list[str] = Field(default_factory=list)
    affected_territories: list[str] = Field(default_factory=list)
    worker_count: int = 0


def changed_files(repo_root: Path, base: str = "HEAD") -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", base],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def refresh_changed_workers(
    *,
    repo_root: Path,
    index_path: Path,
    base: str = "HEAD",
    generator: CardGenerator | None = None,
) -> RefreshResult:
    changed = changed_files(repo_root, base=base)
    environment = RepoEnvironment(repo_root)
    territories = discover_territories(environment)
    affected = _affected_territories(changed, territories)
    if not affected:
        return RefreshResult(changed_files=changed)

    store = IndexStore(index_path)
    existing = {worker.territory_id: worker for worker in store.load_workers()}
    selected_territories = [territory for territory in territories if territory.id in affected]
    refreshed = generate_worker_cards(repo_root, selected_territories, generator=generator)
    for worker in refreshed:
        existing[worker.territory_id] = worker
    store.save(territories, list(existing.values()))
    return RefreshResult(
        changed_files=changed,
        affected_territories=sorted(affected),
        worker_count=len(existing),
    )


def _affected_territories(changed: list[str], territories: list[Territory]) -> set[str]:
    affected: set[str] = set()
    for path in changed:
        for territory in territories:
            if path in territory.files:
                affected.add(territory.id)
    return affected
