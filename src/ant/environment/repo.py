from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IGNORED_DIRS = {
    ".git",
    ".ant",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
    ".ruff_cache",
}

TEXT_EXTENSIONS = {
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".css",
    ".go",
    ".h",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class RepoEnvironment:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.resolve())

    def iter_files(self) -> list[Path]:
        files: list[Path] = []
        for path in self.root.rglob("*"):
            if any(part in IGNORED_DIRS for part in path.parts):
                continue
            if path.is_file() and self.is_text_file(path):
                files.append(path)
        return sorted(files)

    def read_text(self, relative_path: str) -> str:
        path = (self.root / relative_path).resolve()
        path.relative_to(self.root)
        return path.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def is_text_file(path: Path) -> bool:
        return path.suffix.lower() in TEXT_EXTENSIONS or path.name in {
            "Dockerfile",
            "Makefile",
            "README",
        }
