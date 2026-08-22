from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from ant.domain import Territory, as_posix
from ant.environment import RepoEnvironment

GENERIC_CONTAINERS = {"src", "lib", "libs", "packages", "pkg"}


def discover_territories(repo: RepoEnvironment) -> list[Territory]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in repo.iter_files():
        relative = path.relative_to(repo.root)
        root = _natural_root(relative)
        grouped[root].append(as_posix(relative))

    grouped = _merge_tiny_groups(grouped)

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


def _natural_root(relative: Path) -> str:
    parts = relative.parts
    if len(parts) <= 1:
        return ""
    if parts[0] not in GENERIC_CONTAINERS:
        return parts[0]
    if len(parts) <= 3:
        return "/".join(parts[:-1])
    # Keep the import package and its first meaningful subsystem together.
    return "/".join(parts[:3])


def _merge_tiny_groups(grouped: dict[str, list[str]]) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = defaultdict(list)
    for root, files in grouped.items():
        if len(files) == 1 and "/" in root:
            root = root.rsplit("/", 1)[0]
        merged[root].extend(files)
    return merged


def _summarize(root: str, files: list[str]) -> str:
    label = root or "repository root"
    suffixes = sorted({Path(file).suffix for file in files if Path(file).suffix})
    suffix_text = ", ".join(suffixes[:8]) if suffixes else "mixed files"
    return f"Owns {len(files)} files under {label}; common file types: {suffix_text}."
