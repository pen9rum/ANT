"""Freezes a WebWalkerQA dataset-preparation manifest: which split, which
filters, which seed, which example IDs were selected, and which ANT code
state produced it -- the same "traceable back to an exact state"
convention as `ant.evaluation_suite.manifests.RunManifest`, but for a
dataset-PREPARATION step rather than a run.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ant.evaluation_suite.manifests import _current_ant_commit
from ant.evaluation_suite.webwalkerqa.loader import DEFAULT_SPLIT, HF_PATH, WebWalkerQaRecord


class WebWalkerQaPrepManifest(BaseModel):
    dataset_path: str = HF_PATH
    dataset_split: str = DEFAULT_SPLIT
    dataset_revision: str = "unknown"
    ant_commit: str
    created_at: str
    filters: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None
    total_rows_before_filter: int
    total_rows_after_filter: int
    selected_example_ids: list[str] = Field(default_factory=list)


def dataset_revision_best_effort() -> str:
    """Best-effort dataset revision capture -- "if available" per the
    governing spec. The `datasets` library does not guarantee a git-like
    commit SHA is exposed for a Hub dataset loaded without an explicit
    `revision=` pin; this reports the library version actually used
    (a real, verifiable piece of reproducibility context) rather than
    fabricating a SHA that was never actually pinned.
    """
    try:
        import datasets

        return f"datasets=={datasets.__version__}"
    except Exception:
        return "unknown"


def build_prep_manifest(
    *,
    split: str,
    filters: dict[str, Any],
    seed: int | None,
    total_rows_before_filter: int,
    selected_records: list[WebWalkerQaRecord],
    repo_root: Path | None = None,
) -> WebWalkerQaPrepManifest:
    return WebWalkerQaPrepManifest(
        dataset_split=split,
        dataset_revision=dataset_revision_best_effort(),
        ant_commit=_current_ant_commit(repo_root),
        created_at=datetime.now(UTC).isoformat(),
        filters=filters,
        seed=seed,
        total_rows_before_filter=total_rows_before_filter,
        total_rows_after_filter=len(selected_records),
        selected_example_ids=[r.example_id for r in selected_records],
    )


def save_prep_manifest(manifest: WebWalkerQaPrepManifest, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")
