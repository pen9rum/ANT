from pathlib import Path

from ant.environment import RepoEnvironment
from ant.indexing import build_worker_cards, discover_territories


def test_discovers_territories_and_worker_cards(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth.py").write_text("class AuthService:\n    pass\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Demo\nAuthentication notes\n", encoding="utf-8")

    repo = RepoEnvironment(tmp_path)
    territories = discover_territories(repo)
    workers = build_worker_cards(repo.root, territories)

    assert {territory.root for territory in territories} == {"", "src"}
    assert len(workers) == len(territories)
    assert any("authservice" in worker.searchable_terms for worker in workers)
