from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ant.benchmarks.base import TaskExample

# A manifest's `status` field is the load-bearing guard against a "dev
# demonstration" sample silently becoming "the" main-evaluation sample by
# accident. "development" is the ONLY status build_sample_manifest ever
# produces -- promotion to "formal" is a deliberate, out-of-band, human
# decision (edit the saved JSON's `status` field, or construct a
# SampleManifest with status="formal" explicitly by hand) that this module
# never performs on its own, and specifically never as a function of
# sample size, per_stratum, or any other numeric knob. This module does
# not choose a main-evaluation sample size -- that is a separate decision,
# to be made explicitly elsewhere, not inferred from whatever a dev
# demonstration happened to use.
ManifestStatus = Literal["development", "formal"]


class SampleManifest(BaseModel):
    """A pre-specified, saved main-sample selection: which exact task_ids
    were chosen, from which benchmark, under what stratification/seed --
    written BEFORE any agent or judge ever runs on these examples, so a
    later reviewer (or a later run of this same code) can verify the
    sample was fixed ahead of time and never touched by observed
    performance. `task_ids` is the ONLY thing that matters for
    reproducing the exact sample; `stratum_key`/`per_stratum`/`seed` are
    recorded purely so the selection process itself is auditable, not
    re-derived from them (apply_sample_manifest always filters by the
    saved task_ids list, never by re-running the sampler).

    `status` defaults to "development" and MUST be "formal" before a
    manifest may back a main-evaluation run -- see `require_formal_manifest`
    below, which every formal-run driver script must call before using a
    manifest's task_ids for real agent/judge execution. A manifest built
    by `build_sample_manifest` is always "development"; nothing in this
    module ever produces "formal" on its own.
    """

    benchmark: str
    status: ManifestStatus = "development"
    stratum_key: str
    per_stratum: int
    seed: int
    total_examples_considered: int
    strata: dict[str, int]
    task_ids: list[str]
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    note: str = ""


def stratified_sample(
    examples: list[TaskExample],
    *,
    per_stratum: int,
    stratum_key: Callable[[TaskExample], str] = lambda e: str(e.metadata.get("repo", "")),
    seed: int,
) -> list[TaskExample]:
    """Deterministic, repository-stratified sampling: groups `examples` by
    `stratum_key` (default: repo), takes up to `per_stratum` from EACH
    stratum, choosing which ones via a seeded shuffle within that stratum
    (not upstream dataset row order, which may itself carry an unknown,
    unaudited grouping/ordering bias). Strata are visited in sorted key
    order and each stratum's own examples are first sorted by task_id
    before shuffling, so the result depends only on (examples, per_stratum,
    stratum_key, seed) -- never on the order `examples` happened to arrive
    in from a live fetch. A stratum with fewer than `per_stratum` examples
    contributes all of it (never pads, never errors) -- flagged via the
    returned manifest's own `strata` counts, which record what was
    actually available, not just what was requested.
    """
    groups: dict[str, list[TaskExample]] = defaultdict(list)
    for example in examples:
        groups[stratum_key(example)].append(example)

    selected: list[TaskExample] = []
    for key in sorted(groups):
        bucket = sorted(groups[key], key=lambda e: e.task_id)
        rng = random.Random(f"{seed}:{key}")
        rng.shuffle(bucket)
        selected.extend(bucket[:per_stratum])
    return selected


def build_sample_manifest(
    examples: list[TaskExample],
    *,
    benchmark: str,
    per_stratum: int,
    stratum_key: Callable[[TaskExample], str] = lambda e: str(e.metadata.get("repo", "")),
    stratum_key_name: str = "repo",
    seed: int,
    note: str = "",
) -> SampleManifest:
    """Always produces a status="development" manifest -- this function
    has no way to build a "formal" one. `per_stratum`/`seed` here are
    whatever the caller passes; this function does not choose or default
    a main-evaluation sample size, and a caller demonstrating the
    mechanism (rather than pre-specifying a real main sample) should pass
    a `note` saying so explicitly (see the two `sample_manifest_dev.json`
    files under third_party/manifests/ for the existing example)."""
    groups: dict[str, list[TaskExample]] = defaultdict(list)
    for example in examples:
        groups[stratum_key(example)].append(example)
    sampled = stratified_sample(
        examples, per_stratum=per_stratum, stratum_key=stratum_key, seed=seed
    )
    return SampleManifest(
        benchmark=benchmark,
        status="development",
        stratum_key=stratum_key_name,
        per_stratum=per_stratum,
        seed=seed,
        total_examples_considered=len(examples),
        strata={key: len(groups[key]) for key in sorted(groups)},
        task_ids=[example.task_id for example in sampled],
        note=note,
    )


def require_formal_manifest(manifest: SampleManifest) -> SampleManifest:
    """The one required gate every formal-run driver script must call
    before using a manifest's task_ids to run real agents/judges. Raises
    on any manifest whose status is not exactly "formal" -- a
    status="development" manifest (everything build_sample_manifest
    produces, including both existing sample_manifest_dev.json files) is
    refused, so a dev demonstration manifest can never be silently used to
    back a real evaluation run just because a driver script happened to
    point at its file path."""
    if manifest.status != "formal":
        raise ValueError(
            f"Refusing to use a status={manifest.status!r} sample manifest "
            f"(benchmark={manifest.benchmark!r}, {len(manifest.task_ids)} task_ids) to drive a "
            "formal evaluation run. Only a manifest whose status is explicitly 'formal' -- a "
            "deliberate, out-of-band decision, never automatic -- may be used this way. If this "
            "manifest was meant to be the real main sample, promote it explicitly (edit its "
            "saved JSON's `status` field) rather than routing around this check."
        )
    return manifest


def save_sample_manifest(manifest: SampleManifest, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest.model_dump(), indent=2), encoding="utf-8")


def load_sample_manifest(path: Path) -> SampleManifest:
    return SampleManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def apply_sample_manifest(
    examples: list[TaskExample], manifest: SampleManifest
) -> list[TaskExample]:
    """Filters `examples` down to exactly the pre-specified `task_ids`, in
    the manifest's own saved order -- never re-runs the sampler. Raises if
    any saved task_id can no longer be found, since that means the
    underlying benchmark data has drifted since the sample was
    pre-specified (a live-fetched dataset changing upstream, a CSV losing
    a row, ...) -- silently dropping a pre-specified id would be a
    quiet, undisclosed change to a sample that was supposed to be fixed.
    """
    by_id = {example.task_id: example for example in examples}
    missing = [task_id for task_id in manifest.task_ids if task_id not in by_id]
    if missing:
        raise KeyError(
            f"{len(missing)} task_id(s) from the pre-specified sample manifest are no longer "
            f"present in the live benchmark data (benchmark={manifest.benchmark!r}): {missing}. "
            "The manifest was supposed to fix the sample ahead of time -- investigate whether "
            "the upstream dataset changed rather than silently re-sampling."
        )
    return [by_id[task_id] for task_id in manifest.task_ids]
