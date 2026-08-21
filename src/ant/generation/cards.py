from __future__ import annotations

from pathlib import Path

from ant.domain import Territory, WorkerCard
from ant.indexing.cards import build_worker_cards
from ant.providers import CardGenerator


class HeuristicCardGenerator:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def generate_card(self, *, repo_root: str, territory_root: str, files: list[str]) -> WorkerCard:
        territory = Territory(
            id=territory_root or "root",
            root=territory_root,
            files=files,
            summary=f"Owns {len(files)} files under {territory_root or 'repository root'}.",
        )
        return build_worker_cards(self.repo_root, [territory])[0]


def generate_worker_cards(
    repo_root: Path,
    territories: list[Territory],
    generator: CardGenerator | None = None,
) -> list[WorkerCard]:
    if generator is None:
        return build_worker_cards(repo_root, territories)
    return [
        generator.generate_card(
            repo_root=str(repo_root),
            territory_root=territory.root,
            files=territory.files,
        )
        for territory in territories
    ]
