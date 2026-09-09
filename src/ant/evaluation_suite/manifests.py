from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RunManifest(BaseModel):
    """Same convention as cli.py's own `eval`/`gen-compare` commands
    (a manifest.json alongside results), generalized: what benchmark, what
    method, which ANT commit produced this run, and when. Written once per
    (benchmark, method) run directory -- this is what makes a smoke-test
    result later traceable back to an exact code state, independent of
    whatever this session's own conversational record says.
    """

    benchmark: str
    method: str
    ant_commit: str
    started_at: str
    finished_at: str = ""
    example_count: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


def _current_ant_commit(repo_root: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def start_manifest(
    *, benchmark: str, method: str, example_count: int, repo_root: Path | None = None
) -> RunManifest:
    return RunManifest(
        benchmark=benchmark,
        method=method,
        ant_commit=_current_ant_commit(repo_root),
        started_at=datetime.now(UTC).isoformat(),
        example_count=example_count,
    )


def finish_and_save(manifest: RunManifest, out_path: Path) -> None:
    manifest.finished_at = datetime.now(UTC).isoformat()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")
