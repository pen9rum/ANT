from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import urllib.request
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.directory_tree import build_directory_structure
from ant.evaluation_suite.judge import DEFAULT_JUDGE_MODEL, call_judge
from ant.evaluation_suite.registry import register_benchmark
from ant.evaluation_suite.scoring import (
    JudgeType,
    MetricResult,
    normalize_0_100,
    run_judge_noise_protocol,
)

# Pinned per the evaluation-suite audit: official repo
# Tencent-Hunyuan/RepoProbe, commit 5ac4212 ("Pin upstream repositories to
# their snapshot commits and link the arXiv preprint" -- the latest commit
# at audit time; the repo is very new, first published Aug 5-6 2026, no
# release tags exist).
REPOPROBE_COMMIT = "5ac4212"
REPOPROBE_RAW_BASE = f"https://raw.githubusercontent.com/Tencent-Hunyuan/RepoProbe/{REPOPROBE_COMMIT}"

# Vendored verbatim from
# https://raw.githubusercontent.com/Tencent-Hunyuan/RepoProbe/5ac4212/prompt_templates_en.py
# (PromptTemplateManager.DIRECT_SCORING_WITH_REPO_TEMPLATE) -- see
# sweqa_pro.py's own docstring for why the exact wording is vendored
# rather than paraphrased.
_SCORING_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "manifests"
    / "repoprobe"
    / "prompt_templates_en.py"
)

# Judge model note: RepoProbe's own released code defaults its judge to
# "gpt-5.2"; its paper's own headline/leaderboard number was produced with
# Claude Sonnet 4.5 instead (held fixed "to control for judge variability",
# arxiv.org/html/2608.04783v2). This adapter uses neither -- per this
# evaluation suite's own fairness philosophy (see
# ant.evaluation_suite.judge's own docstring), the load-bearing condition
# is "same judge model across every system we compare in this suite", not
# "match whichever judge each paper's own authors happened to pick". A
# judge-model difference from RepoProbe's own paper is a disclosed,
# deliberate choice, not a fidelity gap to work around.


def _fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def _load_scoring_template() -> str:
    text = _SCORING_TEMPLATE_PATH.read_text(encoding="utf-8")
    match = re.search(
        r'DIRECT_SCORING_WITH_REPO_TEMPLATE = """(.*?)"""', text, re.DOTALL
    )
    if match is None:
        raise RuntimeError(
            f"Could not extract DIRECT_SCORING_WITH_REPO_TEMPLATE from {_SCORING_TEMPLATE_PATH} "
            "-- the vendored file may be stale/corrupted."
        )
    return match.group(1)


def _repos_info() -> list[dict]:
    return json.loads(_fetch_text(f"{REPOPROBE_RAW_BASE}/repos_info.json"))


class RepoProbeAdapter:
    """RepoProbe benchmark adapter. CSV parsing uses the standard-library
    `csv` module (not `pandas`, which the official loader uses but which
    is not an ANT dependency) over the exact same column mapping the
    official `dataset_loader.py` uses (`taxonomy`/`difficulty`/`question`/
    `answer`/`checklist`), read directly from the pinned commit's raw CSV
    files -- never a locally re-typed copy.

    Directory-structure fidelity note (CORRECTED -- see the RepoProbe-Python
    calibration audit): the official scoring prompt's `repo_info_section`
    is populated from `repomix-output.md`'s own "Directory Structure"
    section when the official evaluator has actually run repomix (a
    Node.js tool, not vendored here to avoid a new non-Python toolchain
    dependency), and falls back to an EMPTY string when it has not (see
    evaluator.py's own `_parse_repomix_repo_info`) -- exactly the state
    this adapter previously shipped in unconditionally. That empty state
    was confirmed live to cause the judge to flag REAL, verified-existing
    file citations as "likely fabricated" (100% false-positive
    hallucination rate on an 8-case controlled ablation), which is why
    `score()` now calls `build_directory_structure()` (see
    evaluation_suite/directory_tree.py) for a real, deterministic,
    dependency-free directory listing of the task's own pinned repository
    instead. This is a real fix to the scoring INPUT, not the checklist
    itself (`question.scoring_criteria` is unaffected) -- flagged in
    MetricResult.metadata so the correction is never silently invisible.
    """

    name = "repoprobe"

    def __init__(
        self, repo_root: Path | None = None, judge_model: str = DEFAULT_JUDGE_MODEL
    ) -> None:
        self.repo_root = repo_root or Path("repos-repoprobe")
        self.judge_model = judge_model
        self._repos_info: list[dict] | None = None

    def _repo_info_by_short_name(self) -> dict[str, dict]:
        if self._repos_info is None:
            self._repos_info = _repos_info()
        return {entry["name"].split("/")[-1]: entry for entry in self._repos_info}

    def load_examples(
        self, limit: int | None = None, repo_filter: str | None = None
    ) -> list[TaskExample]:
        by_short_name = self._repo_info_by_short_name()
        repo_names = [repo_filter] if repo_filter else sorted(by_short_name)
        out: list[TaskExample] = []
        for repo_short_name in repo_names:
            if repo_short_name not in by_short_name:
                continue
            csv_text = _fetch_text(f"{REPOPROBE_RAW_BASE}/dataset/{repo_short_name}.csv")
            reader = csv.DictReader(io.StringIO(csv_text))
            for row in reader:
                question = (row.get("question") or "").strip()
                if not question:
                    continue
                out.append(
                    TaskExample(
                        benchmark=self.name,
                        task_id=row.get("question_id") or f"{repo_short_name}-{len(out)}",
                        question=question,
                        reference=(row.get("answer") or "").strip(),
                        metadata={
                            "repo": by_short_name[repo_short_name]["name"],
                            "repo_short_name": repo_short_name,
                            "commit": by_short_name[repo_short_name]["snapshot"]["git"]["head"],
                            "taxonomy": row.get("taxonomy", ""),
                            "difficulty": row.get("difficulty", ""),
                            "checklist": row.get("checklist", ""),
                            "primary_language": by_short_name[repo_short_name].get(
                                "primary_language"
                            ),
                        },
                    )
                )
                if limit is not None and len(out) >= limit:
                    return out
        return out

    def prepare_environment(self, example: TaskExample) -> Path:
        repo_full_name = example.metadata["repo"]
        commit = example.metadata["commit"]
        repo_short_name = example.metadata["repo_short_name"]
        repo_path = self.repo_root / repo_short_name
        if repo_path.exists():
            current = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo_path, capture_output=True, text=True
            ).stdout.strip()
            if current == commit:
                return repo_path.resolve()
        else:
            self.repo_root.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", f"https://github.com/{repo_full_name}", str(repo_path)],
                check=True,
            )
        subprocess.run(["git", "fetch", "--all"], cwd=repo_path, check=True)
        subprocess.run(["git", "checkout", commit], cwd=repo_path, check=True)
        return repo_path.resolve()

    def score(self, example: TaskExample, result: AgentResult) -> MetricResult:
        template = _load_scoring_template()
        # Real, deterministic, gold-answer-independent directory listing of
        # the task's own pinned repository -- identical for every method's
        # answer to this same task_id (build_directory_structure takes only
        # a repo path, never a model name or answer). prepare_environment
        # is idempotent (returns immediately if already checked out to the
        # pinned commit), so this is safe to call even if score() is ever
        # invoked before the caller's own prepare_environment step.
        repo_dir = self.prepare_environment(example)
        directory_structure = build_directory_structure(repo_dir)
        prompt = template.format(
            description=example.question,
            repo_info_section=f"**Directory Structure**:\n```\n{directory_structure}\n```",
            reference_text=example.reference,
            scoring_criteria=example.metadata.get("checklist", ""),
            model_name=result.method,
            model_answer=result.final_answer,
        )
        system = "You are a professional code understanding evaluation expert."

        def single_call() -> dict:
            judge_result = call_judge(system=system, user=prompt, max_output_tokens=1024)
            payload = _extract_json(judge_result.text)
            return {
                "raw_text": judge_result.text,
                "payload": payload,
                "judge_cost_usd": judge_result.estimated_cost_usd,
            }

        def native_score_from_call(record: dict) -> float:
            payload = record["payload"]
            if not isinstance(payload, dict):
                return 0.0
            return float(payload.get("total_score", 0.0))

        native_score, grader_runs = run_judge_noise_protocol(
            judge_type=JudgeType.LLM_SCALAR,
            single_call=single_call,
            native_score_from_call=native_score_from_call,
            # This suite's own internal stabilization policy (3 independent
            # calls, mean) applied uniformly to every LLM-scalar benchmark
            # -- NOT an attempt to match RepoProbe's own main-table protocol
            # (which is a single call; its own 5-run averaging is a
            # separate, RQ2-only stability-analysis experiment). Consistent
            # internal comparability across the suite matters more here
            # than mirroring each benchmark's own idiosyncratic repeat count.
            n_scalar_calls=3,
        )
        first_payload = (
            grader_runs[0]["payload"]
            if grader_runs and isinstance(grader_runs[0]["payload"], dict)
            else {}
        )
        return MetricResult(
            benchmark=self.name,
            task_id=example.task_id,
            native_score=native_score,
            normalized_score=normalize_0_100(native_score, native_min=0.0, native_max=10.0),
            submetrics={
                "knowledge_score": float(first_payload.get("knowledge_score", 0.0)),
                "knowledge_max": float(first_payload.get("knowledge_max", 9.0)),
                "clarity_score": float(first_payload.get("clarity_score", 0.0)),
                "clarity_max": float(first_payload.get("clarity_max", 1.0)),
            },
            grader_runs=grader_runs,
            metadata={
                "judge_model": self.judge_model,
                "n_judge_calls": len(grader_runs),
                "hallucination": bool(first_payload.get("hallucination", False)),
                "directory_structure_fidelity_gap": False,
                "directory_structure_source": "build_directory_structure (git ls-files)",
                "generation_model": result.metadata.get("generation_model", "unknown"),
                "judge_cost_usd": sum(run.get("judge_cost_usd", 0.0) for run in grader_runs),
            },
        )


def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None


_ADAPTER = RepoProbeAdapter()
register_benchmark(_ADAPTER)
