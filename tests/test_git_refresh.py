import subprocess
from pathlib import Path

from ant.environment import RepoEnvironment
from ant.git_refresh import refresh_changed_workers
from ant.indexing import build_worker_cards, discover_territories
from ant.memory import ColonyMemoryStore, IndexStore, MemoryRoute


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)


def test_refresh_changed_workers_updates_affected_territory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('one')\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    environment = RepoEnvironment(repo)
    territories = discover_territories(environment)
    workers = build_worker_cards(environment.root, territories)
    index_path = tmp_path / ".ant"
    IndexStore(index_path).save(territories, workers)
    colony_memory = ColonyMemoryStore(index_path)
    colony_memory.save_route(
        MemoryRoute(need_terms=["app"], worker_ids=["worker-src"], weight=2.0)
    )
    (repo / "src" / "app.py").write_text("print('two')\n", encoding="utf-8")

    result = refresh_changed_workers(repo_root=repo, index_path=index_path, base="HEAD")

    assert result.changed_files == ["src/app.py"]
    assert result.affected_territories == ["src"]
    assert result.retired_worker_ids == []
    assert result.stale_memory_count == 1

    # The affected route is stale, so live routing should not serve it yet.
    assert colony_memory.matching_routes(["app"]) == []

    # worker-src still exists (the territory only changed, it didn't
    # disappear), so revalidation should refresh it rather than repair or
    # discard it.
    current_worker_ids = {worker.id for worker in IndexStore(index_path).load_workers()}
    outcome = colony_memory.revalidate_stale(current_worker_ids)
    assert outcome == {"refreshed": 1, "repaired": 0, "discarded": 0}
    assert colony_memory.matching_routes(["app"])[0].worker_ids == ["worker-src"]


def test_refresh_extends_to_neighbor_territory_that_imports_changed_symbol(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "base").mkdir()
    (repo / "consumer").mkdir()
    (repo / "base" / "mod.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (repo / "consumer" / "mod.py").write_text(
        "from base.mod import helper\n\n\ndef use():\n    return helper()\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    environment = RepoEnvironment(repo)
    territories = discover_territories(environment)
    workers = build_worker_cards(environment.root, territories)
    index_path = tmp_path / ".ant"
    IndexStore(index_path).save(territories, workers)
    colony_memory = ColonyMemoryStore(index_path)
    colony_memory.save_route(
        MemoryRoute(need_terms=["use"], worker_ids=["worker-consumer"], weight=2.0)
    )

    (repo / "base" / "mod.py").write_text("def helper():\n    return 2\n", encoding="utf-8")

    result = refresh_changed_workers(repo_root=repo, index_path=index_path, base="HEAD")

    assert result.affected_territories == ["base"]
    assert result.neighbor_territories == ["consumer"]
    # consumer's own files did not change, but its route depends on a symbol
    # whose public interface just did -- it should be stale too.
    assert colony_memory.matching_routes(["use"]) == []


def test_refresh_retires_worker_for_deleted_territory_and_discards_its_memory(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "src").mkdir()
    (repo / "legacy").mkdir()
    (repo / "src" / "app.py").write_text("print('app')\n", encoding="utf-8")
    (repo / "legacy" / "old.py").write_text("print('old')\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    environment = RepoEnvironment(repo)
    territories = discover_territories(environment)
    workers = build_worker_cards(environment.root, territories)
    index_path = tmp_path / ".ant"
    IndexStore(index_path).save(territories, workers)
    colony_memory = ColonyMemoryStore(index_path)
    colony_memory.save_route(
        MemoryRoute(need_terms=["old"], worker_ids=["worker-legacy"], weight=2.0)
    )

    (repo / "legacy" / "old.py").unlink()
    (repo / "legacy").rmdir()
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)

    result = refresh_changed_workers(repo_root=repo, index_path=index_path, base="HEAD")

    assert result.retired_worker_ids == ["worker-legacy"]
    remaining_worker_ids = {worker.id for worker in IndexStore(index_path).load_workers()}
    assert "worker-legacy" not in remaining_worker_ids

    current_worker_ids = {worker.id for worker in IndexStore(index_path).load_workers()}
    outcome = colony_memory.revalidate_stale(current_worker_ids)
    assert outcome["discarded"] == 1
    assert colony_memory.matching_routes(["old"]) == []
