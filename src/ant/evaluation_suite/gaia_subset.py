"""The frozen, capability-covered GAIA validation subset.

WHAT THIS IS. `docs/gaia_environment_report.md` §G recommended running a
predeclared subset of GAIA's 2023 validation split rather than the whole
split, because a handful of tasks require modalities this substrate
declares UNSUPPORTED for *every* method (image, audio, video). Including
them would add a constant floor of guaranteed failures shared by every
system in the roster, compressing the whole results table toward zero
and measuring this substrate's OCR/ASR gap instead of the coordination
question under study. This module implements that recommendation.

THE SELECTION RULE, in full, and it is the whole rule:

    Include a validation task IFF every modality it requires is
    SUPPORTED in `gaia_scope.MODALITY_SUPPORT`.
    Concretely: include it iff it declares no attachment, OR its
    attachment's extension maps to a modality whose declared status is
    SUPPORTED (currently TEXT, TABULAR, ARCHIVE, PDF, OFFICE_DOC).
    Exclude it iff that modality is UNSUPPORTED or NOT_IMPLEMENTED
    (currently IMAGE, AUDIO, VIDEO, UNKNOWN).

WHAT THE RULE MAY NOT LOOK AT -- enforced structurally, not by review:

  * NOT the gold answer. `select_subset()` takes `SubsetCandidate`s, and
    that dataclass has NO field for a reference answer. There is no
    parameter through which one could be threaded, mirroring
    `derive_territories()`'s own signature guarantee.
  * NOT `Annotator Metadata` (`metadata["_audit_only"]`). Same
    structural argument: the candidate carries level, file name and
    nothing else from the row.
  * NOT model performance. No run has happened; nothing in this module
    can read a score. Selecting on observed difficulty would be the
    classic subset-selection fraud and it is excluded by construction,
    not by promise.

`level` IS read, but only to REPORT the retained distribution -- it never
appears in an include/exclude decision. `select_subset()` would produce
exactly the same partition if every level were blanked, which
`tests/test_gaia_subset.py` asserts directly rather than asking a reader
to trust it.

REPRODUCIBILITY. Support status is now a fixed property of the code
(`support_status_for` no longer probes the environment -- see its own
docstring), so two machines produce the same partition. The manifest
nonetheless records the resolved dependency versions at freeze time, so
a later "why does this PDF not parse" has a starting point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ant.evaluation_suite.gaia_scope import (
    MODALITY_SUPPORT,
    Modality,
    SupportStatus,
    check_substrate_dependencies,
    modality_for,
    support_status_for,
)

#: Where the frozen manifest lives, matching every other benchmark's
#: `third_party/manifests/<benchmark>/` convention.
MANIFEST_PATH = (
    Path(__file__).resolve().parents[3] / "third_party" / "manifests" / "gaia" / "manifest.json"
)

#: Bumped whenever the SELECTION RULE changes. A results file that cites
#: a manifest with a different rule version is not comparable to one
#: that cites this version, and saying so in a single integer is cheaper
#: than reconstructing it from a diff later.
SELECTION_RULE_VERSION = 1


@dataclass(frozen=True)
class SubsetCandidate:
    """One validation task, reduced to exactly the TASK-OBSERVABLE
    signals the selection rule is permitted to see.

    The absence of a `reference` / `annotator_metadata` field is the
    leakage guarantee. `level` is present for reporting only -- see the
    module docstring and the test that blanks it.
    """

    task_id: str
    level: str
    file_name: str | None
    question_chars: int = 0


@dataclass(frozen=True)
class ExclusionRecord:
    """Why one task was dropped. Recorded per task rather than as an
    aggregate count so the exclusion ledger is auditable: a reader can
    re-derive the decision from the file name alone."""

    task_id: str
    level: str
    file_name: str
    modality: str
    support_status: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "level": self.level,
            "file_name": self.file_name,
            "modality": self.modality,
            "support_status": self.support_status,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SubsetSelection:
    retained: tuple[SubsetCandidate, ...]
    excluded: tuple[ExclusionRecord, ...]

    @property
    def total(self) -> int:
        return len(self.retained) + len(self.excluded)

    def retained_ids(self) -> list[str]:
        return [candidate.task_id for candidate in self.retained]

    def level_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in self.retained:
            counts[candidate.level or "unknown"] = counts.get(candidate.level or "unknown", 0) + 1
        return dict(sorted(counts.items()))

    def extension_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for candidate in self.retained:
            key = Path(candidate.file_name).suffix.lower() if candidate.file_name else "(none)"
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def exclusions_by_modality(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.excluded:
            counts[record.modality] = counts.get(record.modality, 0) + 1
        return dict(sorted(counts.items()))


def supported_modalities() -> tuple[str, ...]:
    """The declared SUPPORTED set, read from the single source of truth
    rather than re-typed here -- so adding a modality to
    `MODALITY_SUPPORT` cannot leave this module disagreeing with it."""
    return tuple(
        sorted(
            modality.value
            for modality, status in MODALITY_SUPPORT.items()
            if status is SupportStatus.SUPPORTED
        )
    )


def select_subset(candidates: list[SubsetCandidate]) -> SubsetSelection:
    """Applies the selection rule. Pure: no I/O, no environment probe, no
    randomness, no ordering dependence beyond preserving input order."""
    retained: list[SubsetCandidate] = []
    excluded: list[ExclusionRecord] = []
    for candidate in candidates:
        if not candidate.file_name:
            # No attachment -> no modality requirement beyond web and
            # computation, which every task gets unconditionally.
            retained.append(candidate)
            continue
        modality = modality_for(candidate.file_name)
        status = support_status_for(modality)
        if status is SupportStatus.SUPPORTED:
            retained.append(candidate)
            continue
        excluded.append(
            ExclusionRecord(
                task_id=candidate.task_id,
                level=candidate.level,
                file_name=candidate.file_name,
                modality=modality.value,
                support_status=status.value,
                reason=_exclusion_reason(modality, status),
            )
        )
    return SubsetSelection(tuple(retained), tuple(excluded))


def _exclusion_reason(modality: Modality, status: SupportStatus) -> str:
    if status is SupportStatus.UNSUPPORTED:
        return (
            f"attachment modality {modality.value!r} is declared UNSUPPORTED at the "
            "substrate level (perception capability withheld from EVERY method equally, "
            "so no method in the roster can fairly attempt this task)"
        )
    return (
        f"attachment modality {modality.value!r} is declared {status.value} -- a scope "
        "decision, not a fairness judgement; it is cheap to revisit and would move this "
        "task back in"
    )


#: The one status value that authorises a manifest to back a real
#: evaluation run. Anything else -- `blocked_on_gated_access`, a draft,
#: a development sample -- must be refused, exactly as SWE-QA-Pro's own
#: sample manifests are refused until promoted.
FROZEN_STATUS = "frozen"


class ManifestNotFrozenError(RuntimeError):
    """A manifest was asked to back an evaluation before it was frozen."""


def require_frozen_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Gate. A manifest with no task ids, or one whose status is not
    `frozen`, must never silently back a run -- an empty subset would
    produce a results file with N=0 that reads like a completed sweep."""
    status = manifest.get("status")
    if status != FROZEN_STATUS:
        raise ManifestNotFrozenError(
            f"GAIA manifest status is {status!r}, not {FROZEN_STATUS!r}. "
            f"Blocked reason: {manifest.get('blocked_reason') or 'none recorded'}."
        )
    if not manifest.get("task_ids"):
        raise ManifestNotFrozenError("GAIA manifest is frozen but carries no task ids.")
    return manifest


def build_manifest(
    selection: SubsetSelection | None,
    *,
    dataset_revision: str | None = None,
    note: str = "",
    status: str = FROZEN_STATUS,
    blocked_reason: str | None = None,
) -> dict[str, Any]:
    """The manifest document. Mirrors the shape of the other frozen
    manifests under `third_party/manifests/` -- a provenance block, the
    rule that produced the selection, the selected ids, and a `status`
    field that a runner checks before letting the file back an
    evaluation.

    `selection=None` writes the manifest in its NOT-YET-FROZEN form:
    every count and id is `null` rather than `0` or `[]`, because
    "nobody has been able to compute this yet" and "the computation ran
    and found nothing" must not be representable by the same bytes. The
    rule, the declared modality policy and the resolved dependency
    versions are all still recorded, since those are real now and are
    what the freeze will be judged against later.
    """
    if selection is None and status == FROZEN_STATUS:
        raise ValueError(
            "A manifest with no selection cannot be marked frozen; pass an explicit "
            "status such as 'blocked_on_gated_access'."
        )
    try:
        dependencies: dict[str, str] | None = check_substrate_dependencies()
        dependency_error: str | None = None
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
        dependencies, dependency_error = None, str(exc)

    return {
        "name": "GAIA",
        "role": "capability-covered evaluation subset of the 2023 validation split",
        "benchmark": "gaia",
        "status": status,
        "paper": "arXiv:2311.12983 (ICLR 2024)",
        "dataset": "https://huggingface.co/datasets/gaia-benchmark/GAIA",
        "dataset_config": "2023_all",
        "split": "validation",
        "dataset_revision": dataset_revision,
        "license_data": (
            "gated; redistribution prohibited by the dataset's own gate terms. This "
            "manifest therefore lists TASK IDS ONLY -- no question, answer or "
            "attachment byte appears here or anywhere else in this repository."
        ),
        "benchmark_adapter": "src/ant/benchmarks/gaia.py",
        "selection_rule_version": SELECTION_RULE_VERSION,
        "selection_rule": (
            "Include a validation task IFF every modality it requires is SUPPORTED in "
            "ant.evaluation_suite.gaia_scope.MODALITY_SUPPORT. Concretely: include iff "
            "the task declares no attachment, OR its attachment's file extension maps "
            "to a SUPPORTED modality. Exclude otherwise. Computed from TASK-OBSERVABLE "
            "signals only (declared attachment presence and file extension)."
        ),
        "selection_inputs_forbidden": [
            "the gold `Final answer` (TaskExample.reference)",
            "`Annotator Metadata` (metadata['_audit_only'], i.e. Steps / Tools)",
            "any model's performance on the task",
            "the question text",
        ],
        "selection_leakage_guarantee": (
            "Structural, not conventional: gaia_subset.SubsetCandidate has no field for "
            "a reference answer or annotator metadata, so select_subset() has no "
            "parameter through which either could be threaded. `level` is carried for "
            "REPORTING only; tests/test_gaia_subset.py asserts the partition is "
            "identical when every level is blanked."
        ),
        "supported_modalities": list(supported_modalities()),
        "unsupported_modalities": [
            modality.value
            for modality, support in sorted(MODALITY_SUPPORT.items())
            if support is not SupportStatus.SUPPORTED
        ],
        "substrate_dependencies": dependencies,
        "substrate_dependency_error": dependency_error,
        "counts": (
            {
                "validation_total": selection.total,
                "retained": len(selection.retained),
                "excluded": len(selection.excluded),
                "excluded_by_modality": selection.exclusions_by_modality(),
            }
            if selection is not None
            else None
        ),
        "retained_level_distribution": (
            selection.level_distribution() if selection is not None else None
        ),
        "retained_extension_distribution": (
            selection.extension_distribution() if selection is not None else None
        ),
        "task_ids": selection.retained_ids() if selection is not None else None,
        "exclusion_ledger": (
            [record.to_dict() for record in selection.excluded]
            if selection is not None
            else None
        ),
        "blocked_reason": blocked_reason,
        "freeze_command": "python scripts/freeze_gaia_manifest.py",
        "created_at": datetime.now(UTC).isoformat(),
        "note": note,
    }


def write_manifest(manifest: dict[str, Any], path: Path | None = None) -> Path:
    """Writes the manifest with stable formatting so a re-freeze produces
    a reviewable diff rather than a reformat."""
    target = Path(path) if path is not None else MANIFEST_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target


def candidates_from_examples(examples: list[Any]) -> list[SubsetCandidate]:
    """Projects `TaskExample`s down to the observable signals.

    This projection is where the leakage boundary is actually crossed
    in the safe direction: a `TaskExample` carries `reference` and
    `_audit_only`, a `SubsetCandidate` cannot, and everything downstream
    of here sees only the latter.
    """
    return [
        SubsetCandidate(
            task_id=example.task_id,
            level=str(example.metadata.get("level", "")),
            file_name=example.metadata.get("file_name") or None,
            question_chars=len(example.question),
        )
        for example in examples
    ]
