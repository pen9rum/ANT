"""GAIA (ICLR 2024, arXiv:2311.12983) benchmark adapter.

SCOPE: this is the ORIGINAL GAIA -- the `gaia-benchmark/GAIA` dataset,
2023 validation split. Not GAIA2, not any community re-release or
re-annotation.

WHY VALIDATION AND NOT TEST: GAIA's test split ships questions without
answers; its gold labels are held privately for the leaderboard. Scoring
test locally is impossible, so an offline comparison of several methods
has to run on validation. This is stated up front because "we evaluated
on GAIA" is ambiguous in the literature and the two splits are not
interchangeable.

DATA IS GATED AND IS NEVER COMMITTED. `gaia-benchmark/GAIA` is a gated HF
dataset whose own gate text is "you agree to not reshare this dataset
outside of a gated or private repository on the HF hub". `load_examples`
therefore reads live from HF using the caller's own credentials and
caches under a GITIGNORED directory; no GAIA question, answer, or
attachment byte ever enters this repository. See
`third_party/manifests/gaia/PROVENANCE.md`.

GOLD-LEAKAGE STRUCTURE (the `_audit_only` pattern, matching
`benchmarks/webwalkerqa.py`):
  * Agent-visible: `question`, `task_id`, and the metadata an agent
    legitimately needs to act -- `file_name`, `has_attachment`,
    `attachment_modality`, `attachment_supported`, `territories`, `level`.
  * `reference`: the gold `Final answer`. Read ONLY by `score()`. The
    shared `TaskExample` schema puts it at the top level for every
    benchmark in this suite, so GAIA follows suit rather than inventing a
    different shape; every agent adapter in the suite already treats it as
    off-limits.
  * `metadata["_audit_only"]`: GAIA's `Annotator Metadata` -- the
    human solver's `Steps`, `Number of steps`, `Tools`, `Number of tools`,
    `How long did this take?`. This is the highest-value leak in the whole
    dataset: `Steps` is a literal worked solution and `Tools` is the exact
    capability list needed. Nothing on any generation path reads this key;
    it exists so an OFFLINE audit can characterise capability requirements
    after the fact.

`level` is deliberately NOT audit-only. It is a difficulty label, not
solution content (it says "this is hard", not "here is how"), it is
needed to stratify and report results, and `benchmarks/repoprobe.py`
already keeps its own `difficulty` field agent-visible. Documented so the
choice is a decision on the record rather than an oversight.

SCORING IS DETERMINISTIC -- no judge, no LLM, no cost. GAIA is
quasi-exact-match by construction ("evaluation is done via quasi exact
match between a model's answer and the ground truth", leaderboard
`content.py`), so `JudgeType.DETERMINISTIC` is correct here and
`run_judge_noise_protocol` calls the scorer exactly once. That makes GAIA
the only substrate in this suite whose scoring is free, which is worth
knowing when budgeting a sweep.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.gaia_scope import (
    GaiaEnvironment,
    derive_territories,
    modality_for,
    support_status_for,
)
from ant.evaluation_suite.gaia_scorer import (
    OFFICIAL_SCORER_REVISION,
    OFFICIAL_SCORER_SHA256,
    score_response,
)
from ant.evaluation_suite.registry import register_benchmark
from ant.evaluation_suite.scoring import (
    JudgeType,
    MetricResult,
    normalize_0_100,
    run_judge_noise_protocol,
)

GAIA_HF_DATASET = "gaia-benchmark/GAIA"
GAIA_HF_CONFIG = "2023_all"
GAIA_SPLIT = "validation"

#: Gitignored. Raw gated material lands here and never in git.
DEFAULT_DATA_ROOT = Path(".gaia-data")

_FIXTURES_PATH = (
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "manifests"
    / "gaia"
    / "synthetic_fixtures.json"
)

_GATED_ACCESS_HELP = (
    f"Could not read the gated dataset {GAIA_HF_DATASET!r}. GAIA is gated on Hugging "
    "Face: you must (1) be signed in to a Hugging Face account that has accepted the "
    f"dataset's terms on its own page for {GAIA_HF_DATASET} (the gate is configured as "
    "auto-approval, so accepting the terms grants access immediately -- there is no "
    "manual review queue), and (2) expose that account's token to this process, either "
    "by running `huggingface-cli login` or by setting the HF_TOKEN environment "
    "variable. Do not work around the gate with a mirror or scraped copy. To develop "
    "without access, construct this adapter with source='synthetic'."
)


class GaiaAdapter:
    """`name = "gaia"`.

    `source` selects where examples come from:
      * `"live"`      -- the real gated HF dataset. Raises with actionable
                         instructions if credentials/access are missing.
      * `"synthetic"` -- hand-authored fixtures containing no GAIA
                         material, for tests and offline development.
      * `"auto"`      -- live when a token is visible in the environment,
                         synthetic otherwise. `auto` NEVER silently
                         substitutes synthetic data for a live run that
                         was expected to work: `load_examples` stamps
                         `metadata["source"]` on every example and
                         `resolved_source()` reports it, so a results file
                         can always prove which corpus produced it.

    `repo_filter` (the shared `BenchmarkAdapter` parameter name) is
    reinterpreted here as a LEVEL filter -- `"1"`, `"2"`, `"3"`. GAIA has
    no repository concept; the parameter name is kept only for structural
    interface compatibility, exactly as `WebWalkerQaAdapter` reinterprets
    it as a domain filter.
    """

    name = "gaia"

    def __init__(
        self,
        source: str = "auto",
        data_root: Path | None = None,
        fixtures_path: Path | None = None,
    ) -> None:
        if source not in ("auto", "live", "synthetic"):
            raise ValueError(f"source must be 'auto', 'live' or 'synthetic', got {source!r}")
        self.source = source
        self.data_root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
        self.fixtures_path = Path(fixtures_path) if fixtures_path is not None else _FIXTURES_PATH
        self._resolved_source: str | None = None

    # ---- source resolution ---------------------------------------------

    def resolved_source(self) -> str:
        """Which corpus `load_examples` actually used. Never guesses --
        returns None-safe only after a load has happened."""
        return self._resolved_source or self.source

    @staticmethod
    def _token_visible() -> bool:
        if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            return True
        try:
            from huggingface_hub import get_token
        except ImportError:
            return False
        try:
            return bool(get_token())
        except Exception:
            return False

    # ---- loading --------------------------------------------------------

    def load_examples(
        self, limit: int | None = None, repo_filter: str | None = None
    ) -> list[TaskExample]:
        source = self.source
        if source == "auto":
            source = "live" if self._token_visible() else "synthetic"
        self._resolved_source = source
        rows = self._live_rows() if source == "live" else self._synthetic_rows()

        out: list[TaskExample] = []
        for row in rows:
            if repo_filter is not None:
                if str(row.get("Level", "")).strip() != str(repo_filter).strip():
                    continue
            example = self._to_example(row, source)
            if example is not None:
                out.append(example)
            if limit is not None and len(out) >= limit:
                break
        return out

    def _live_rows(self) -> list[dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "The 'datasets' package is required to read the live GAIA dataset "
                "(`pip install 'ant-codebase[bench]'`). " + _GATED_ACCESS_HELP
            ) from exc
        try:
            dataset = load_dataset(GAIA_HF_DATASET, GAIA_HF_CONFIG, split=GAIA_SPLIT)
        except Exception as exc:
            raise RuntimeError(f"{_GATED_ACCESS_HELP} Underlying error: {exc}") from exc
        return [dict(row) for row in dataset]

    def _synthetic_rows(self) -> list[dict[str, Any]]:
        payload = json.loads(self.fixtures_path.read_text(encoding="utf-8"))
        return list(payload["examples"])

    def _to_example(self, row: dict[str, Any], source: str) -> TaskExample | None:
        question = str(row.get("Question") or "").strip()
        task_id = str(row.get("task_id") or "").strip()
        if not question or not task_id:
            return None
        file_name = str(row.get("file_name") or "").strip() or None
        modality = modality_for(file_name) if file_name else None
        supported = support_status_for(modality).value if modality else None
        territories = derive_territories(file_name=file_name, question_length=len(question))

        return TaskExample(
            benchmark=self.name,
            task_id=task_id,
            question=question,
            # Gold. Read ONLY by score(); every agent adapter in this suite
            # already treats TaskExample.reference as off-limits.
            reference=str(row.get("Final answer") or "").strip(),
            metadata={
                # --- agent-visible: what a solver legitimately needs ---
                "level": str(row.get("Level", "")).strip(),
                "file_name": file_name,
                "has_attachment": file_name is not None,
                "attachment_modality": modality.value if modality else None,
                "attachment_supported": supported,
                "territories": [
                    {
                        "id": territory.id,
                        "kind": territory.kind.value,
                        "description": territory.description,
                        "supported": territory.supported,
                    }
                    for territory in territories
                ],
                "source": source,
                # --- evaluator/audit only ---
                # GAIA's Annotator Metadata is a literal worked solution
                # (`Steps`) plus the exact capability list (`Tools`). No
                # generation code path reads this key; see this module's
                # own docstring and tests/test_gaia_adapter.py's
                # leakage-prevention tests.
                "_audit_only": {
                    "annotator_metadata": row.get("Annotator Metadata") or {},
                    "final_answer": str(row.get("Final answer") or "").strip(),
                },
            },
        )

    # ---- environment ----------------------------------------------------

    def prepare_environment(self, example: TaskExample) -> Path:
        """Materialises this task's own directory and places its
        attachment (if any) inside it. Returns that directory, which is
        what a `GaiaEnvironment` is constructed from -- the same shape as
        every other adapter in this suite (a Path an agent operates
        against). Idempotent: an already-present attachment is left alone.
        """
        task_dir = self.data_root / self.resolved_source() / example.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        file_name = example.metadata.get("file_name")
        if not file_name:
            return task_dir.resolve()
        destination = task_dir / file_name
        if destination.exists():
            return task_dir.resolve()
        if self.resolved_source() == "synthetic":
            self._materialize_synthetic_attachment(file_name, destination)
        else:
            self._download_live_attachment(example, file_name, destination)
        return task_dir.resolve()

    def _materialize_synthetic_attachment(self, file_name: str, destination: Path) -> None:
        from ant.evaluation_suite.gaia_fixtures import build_fixture_attachment

        build_fixture_attachment(file_name, destination, self.fixtures_path)

    def _download_live_attachment(
        self, example: TaskExample, file_name: str, destination: Path
    ) -> None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(_GATED_ACCESS_HELP) from exc
        try:
            cached = hf_hub_download(
                repo_id=GAIA_HF_DATASET,
                repo_type="dataset",
                filename=f"2023/{GAIA_SPLIT}/{file_name}",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not download attachment {file_name!r} for task "
                f"{example.task_id!r}. {_GATED_ACCESS_HELP} Underlying error: {exc}"
            ) from exc
        shutil.copyfile(cached, destination)

    def environment_for(self, example: TaskExample) -> GaiaEnvironment:
        """Convenience: `prepare_environment` + the `GaiaEnvironment` an
        agent actually operates against. Kept separate from
        `prepare_environment` so the `BenchmarkAdapter` protocol's own
        Path-returning contract stays exactly as every other adapter
        implements it."""
        return GaiaEnvironment(self.prepare_environment(example), example.metadata.get("file_name"))

    # ---- scoring --------------------------------------------------------

    def score(self, example: TaskExample, result: AgentResult) -> MetricResult:
        """GAIA's OFFICIAL scorer, run once. Deterministic and free.

        `run_judge_noise_protocol(DETERMINISTIC)` is used rather than
        calling the scorer inline so GAIA's `grader_runs` has the same
        shape every other benchmark in this suite produces -- a downstream
        reader should not need a GAIA special case.
        """

        def single_call() -> dict[str, Any]:
            correct, extraction = score_response(result.final_answer, example.reference)
            return {
                "correct": correct,
                "extracted_answer": extraction.answer,
                "final_answer_template_found": extraction.template_found,
                "judge_cost_usd": 0.0,
            }

        native_score, grader_runs = run_judge_noise_protocol(
            judge_type=JudgeType.DETERMINISTIC,
            single_call=single_call,
            native_score_from_call=lambda record: 1.0 if record["correct"] else 0.0,
        )
        first = grader_runs[0]
        return MetricResult(
            benchmark=self.name,
            task_id=example.task_id,
            native_score=native_score,
            normalized_score=normalize_0_100(native_score, native_min=0.0, native_max=1.0),
            grader_runs=grader_runs,
            metadata={
                "level": example.metadata.get("level", ""),
                "scorer": "official GAIA question_scorer (vendored verbatim)",
                "scorer_revision": OFFICIAL_SCORER_REVISION,
                "scorer_sha256": OFFICIAL_SCORER_SHA256,
                "judge_model": None,
                "n_judge_calls": 0,
                "judge_cost_usd": 0.0,
                "extracted_answer": first["extracted_answer"],
                # A False here means the model never emitted GAIA's own
                # `FINAL ANSWER:` template -- a FORMAT failure, which is
                # worth separating from a content failure when reading
                # results. The official pipeline cannot distinguish them
                # because it only ever receives the extracted string.
                "final_answer_template_found": first["final_answer_template_found"],
                "generation_model": result.metadata.get("generation_model", "unknown"),
            },
        )


_ADAPTER = GaiaAdapter()
register_benchmark(_ADAPTER)
