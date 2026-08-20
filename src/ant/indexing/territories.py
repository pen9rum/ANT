from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ant.domain import Territory, as_posix
from ant.environment import RepoEnvironment


def discover_territories(repo: RepoEnvironment) -> list[Territory]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in repo.iter_files():
        relative = path.relative_to(repo.root)
        root = relative.parts[0] if len(relative.parts) > 1 else ""
        grouped[root].append(as_posix(relative))

    territories: list[Territory] = []
    for index, (root, files) in enumerate(sorted(grouped.items()), start=1):
        territory_id = _territory_id(root, index)
        territories.append(
            Territory(
                id=territory_id,
                root=root,
                files=files,
                summary=_summarize(root, files),
            )
        )
    return territories


def _territory_id(root: str, index: int) -> str:
    if not root:
        return "root"
    normalized = "".join(ch if ch.isalnum() else "-" for ch in root.lower()).strip("-")
    return normalized or f"territory-{index}"


def _summarize(root: str, files: list[str]) -> str:
    label = root or "repository root"
    suffixes = sorted({Path(file).suffix for file in files if Path(file).suffix})
    suffix_text = ", ".join(suffixes[:8]) if suffixes else "mixed files"
    return f"Owns {len(files)} files under {label}; common file types: {suffix_text}."
