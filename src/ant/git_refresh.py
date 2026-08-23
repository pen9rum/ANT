from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from ant.domain import Territory, WorkerCard
from ant.environment import RepoEnvironment
from ant.generation import generate_worker_cards
from ant.indexing import discover_territories
from ant.memory import ColonyMemoryStore, IndexStore
from ant.providers import CardGenerator
from ant.tools.symbol_index import build_symbol_index, module_name, resolve_import_module


class RefreshResult(BaseModel):
    changed_files: list[str] = Field(default_factory=list)
    affected_territories: list[str] = Field(default_factory=list)
    neighbor_territories: list[str] = Field(default_factory=list)
    retired_worker_ids: list[str] = Field(default_factory=list)
    worker_count: int = 0
    stale_memory_count: int = 0


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
    current_territory_ids = {territory.id for territory in territories}

    store = IndexStore(index_path)
    existing = {worker.territory_id: worker for worker in store.load_workers()}

    # A territory that no longer exists on disk (directory deleted or fully
    # renamed away) has no representation in a fresh discovery pass, so it
    # can never show up in `affected` below (that only matches territories
    # that still exist). Left alone, its worker would linger forever
    # pointing at dead files. Retire it explicitly here instead.
    retired = [
        worker
        for territory_id, worker in existing.items()
        if territory_id not in current_territory_ids
    ]
    for worker in retired:
        del existing[worker.territory_id]

    affected = _affected_territories(changed, territories)
    neighbors = _neighboring_territories(repo_root, changed, territories, exclude=affected)
    all_affected = affected | neighbors
    changed_worker_ids = sorted(_worker_ids_for_territories(all_affected, existing))
    retired_worker_ids = sorted(worker.id for worker in retired)

    colony_memory = ColonyMemoryStore(index_path)
    stale_count = colony_memory.mark_stale(
        changed_worker_ids,
        reason="Territory files (or a dependency's public interface) changed; "
        "dependent memory requires revalidation.",
    ) + colony_memory.mark_stale(
        retired_worker_ids,
        reason="Territory no longer exists; dependent memory should be discarded.",
    )

    if not all_affected and not retired:
        return RefreshResult(changed_files=changed, stale_memory_count=stale_count)

    selected_territories = [territory for territory in territories if territory.id in all_affected]
    refreshed = generate_worker_cards(repo_root, selected_territories, generator=generator)
    for worker in refreshed:
        existing[worker.territory_id] = worker
    store.save(territories, list(existing.values()))
    return RefreshResult(
        changed_files=changed,
        affected_territories=sorted(affected),
        neighbor_territories=sorted(neighbors),
        retired_worker_ids=retired_worker_ids,
        worker_count=len(existing),
        stale_memory_count=stale_count,
    )


def _neighboring_territories(
    repo_root: Path,
    changed: list[str],
    territories: list[Territory],
    exclude: set[str],
) -> set[str]:
    """Territories that don't directly own a changed file, but import from it
    or call one of its top-level symbols -- i.e. territories whose assumptions
    about a changed public interface may now be wrong. This is the "if the
    public interface or relationship changed, extend refresh to directly
    affected neighboring territories" step: a change is not only local to the
    file it touched.
    """
    changed_py = [path for path in changed if path.endswith(".py")]
    if not changed_py:
        return set()

    all_files = [file for territory in territories for file in territory.files]
    index = build_symbol_index(repo_root, all_files)
    changed_modules = {module_name(path) for path in changed_py}
    changed_symbol_names = {
        definition.name
        for path in changed_py
        for definition in index.definitions_by_module.get(module_name(path), [])
    }

    dependent_files: set[str] = set()
    for binding in index.imports:
        if resolve_import_module(binding.module, binding.path) in changed_modules:
            dependent_files.add(binding.path)
    for symbol in changed_symbol_names:
        for caller in index.callers.get(symbol, []):
            dependent_files.add(caller.path)

    neighbors: set[str] = set()
    for territory in territories:
        if territory.id in exclude:
            continue
        if any(file in dependent_files for file in territory.files):
            neighbors.add(territory.id)
    return neighbors


def _affected_territories(changed: list[str], territories: list[Territory]) -> set[str]:
    affected: set[str] = set()
    for path in changed:
        for territory in territories:
            if path in territory.files:
                affected.add(territory.id)
    return affected


def _worker_ids_for_territories(
    territory_ids: set[str],
    existing: dict[str, WorkerCard],
) -> set[str]:
    return {
        existing[territory_id].id if territory_id in existing else f"worker-{territory_id}"
        for territory_id in territory_ids
    }
