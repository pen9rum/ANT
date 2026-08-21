import subprocess
from pathlib import Path

from ant.environment import RepoEnvironment
from ant.git_refresh import refresh_changed_workers
from ant.indexing import build_worker_cards, discover_territories
from ant.memory import ColonyMemoryStore, IndexStore


def test_refresh_changed_workers_updates_affected_territory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('one')\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    environment = RepoEnvironment(repo)
    territories = discover_territories(environment)
    workers = build_worker_cards(environment.root, territories)
    index_path = tmp_path / ".ant"
    IndexStore(index_path).save(territories, workers)
    (repo / "src" / "app.py").write_text("print('two')\n", encoding="utf-8")

    result = refresh_changed_workers(repo_root=repo, index_path=index_path, base="HEAD")

    assert result.changed_files == ["src/app.py"]
    assert result.affected_territories == ["src"]
    assert result.stale_memory_count == 1
    assert ColonyMemoryStore(index_path).revalidate_stale(repo)["revalidated"] == 1
