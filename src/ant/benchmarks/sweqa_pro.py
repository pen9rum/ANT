from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.evaluation.datasets import load_examples
from ant.evaluation.repos import RepoSpec, fetch_repositories, load_repo_specs
from ant.evaluation_suite.judge import DEFAULT_JUDGE_MODEL, call_judge
from ant.evaluation_suite.registry import register_benchmark
from ant.evaluation_suite.scoring import (
    JudgeType,
    MetricResult,
    normalize_0_100,
    run_judge_noise_protocol,
)

# Pinned per the evaluation-suite audit: official repo TIGER-AI-Lab/SWE-QA-Pro,
# commit 93ac6a4 ("update LICENSE", the latest commit on `main` at audit
# time -- no release tags exist). The repos.txt URL below is read live from
# that pinned ref's raw content, not a moving `main` pointer, so the exact
# 26-repo/commit list this benchmark evaluates against cannot silently
# drift out from under a later re-run.
SWEQA_PRO_COMMIT = "93ac6a4"
SWEQA_PRO_REPOS_TXT = (
    f"https://raw.githubusercontent.com/TIGER-AI-Lab/SWE-QA-Pro/{SWEQA_PRO_COMMIT}/eval/repos.txt"
)

# Vendored verbatim from
# https://raw.githubusercontent.com/TIGER-AI-Lab/SWE-QA-Pro/93ac6a4/eval/prompts/judge_prompt.txt
# -- the exact, unmodified 5-axis rubric text SWE-QA-Pro's own judge.py
# loads and formats. See PART D/scoring's own docstring below for why this
# is vendored rather than "rewritten": the rubric's exact wording is the
# evaluation semantics; paraphrasing it would not be "using the official
# protocol exactly".
_JUDGE_PROMPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "manifests"
    / "sweqa_pro"
    / "judge_prompt.txt"
)
_SCORE_KEYS = ("correctness", "completeness", "relevance", "clarity", "reasoning")


def _load_prompt_template() -> str:
    return _JUDGE_PROMPT_PATH.read_text(encoding="utf-8")


def _parse_judge_scores(text: str) -> dict[str, int] | None:
    """Byte-faithful port of SWE-QA-Pro's own judge.py:_parse_scores --
    strips markdown code fences, parses JSON, validates each of the 5 keys
    is an int in [1, 10]. Deliberately identical logic, not a
    reinterpretation, since a looser/stricter parser would itself be a
    (small) deviation from "the official protocol exactly"."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    for key in _SCORE_KEYS:
        if key not in data or not isinstance(data[key], int) or not (1 <= data[key] <= 10):
            return None
    return {key: int(data[key]) for key in _SCORE_KEYS}


class SweQaProAdapter:
    """SWE-QA-Pro benchmark adapter -- dataset loading reuses the existing
    `ant.evaluation.datasets`/`ant.evaluation.repos` modules (not
    duplicated). Scoring preserves SWE-QA-Pro's own 5-axis rubric text
    verbatim (vendored, see module docstring above), but judges it with
    this evaluation suite's own centralized GPT-5 judge
    (`ant.evaluation_suite.judge.call_judge`) rather than treating "match
    the paper's exact judge model" as a requirement -- per this suite's
    own fairness philosophy, the load-bearing condition is "same
    benchmark + same benchmark-specific rubric + same judge model across
    every system we compare", not "reproduce each paper's own,
    mutually-inconsistent judge choice". It happens that SWE-QA-Pro's own
    paper judge (gpt-5-2025-08-07) IS this suite's chosen default -- see
    judge.py's own docstring for why that coincidence doesn't make this
    "the official protocol", just a convenient overlap. 3x-averaging is
    this suite's own internal stabilization policy (see
    run_judge_noise_protocol), applied uniformly to every LLM-scalar
    benchmark, not specifically because SWE-QA-Pro's paper happens to
    document the same number.
    """

    name = "sweqa_pro"

    def __init__(self, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or Path("repos")
        self._repo_specs: dict[str, RepoSpec] | None = None

    def load_examples(
        self, limit: int | None = None, repo_filter: str | None = None
    ) -> list[TaskExample]:
        examples = load_examples("hf://swe-qa-pro", split="test", limit=limit)
        out = []
        for example in examples:
            if repo_filter is not None and repo_filter not in example.repo:
                continue
            out.append(
                TaskExample(
                    benchmark=self.name,
                    task_id=example.id,
                    question=example.question,
                    reference=example.answer,
                    metadata={
                        "repo": example.repo,
                        "commit_id": example.metadata.get("commit_id", ""),
                        "cluster": example.metadata.get("cluster"),
                        "qa_type": example.metadata.get("qa_type"),
                    },
                )
            )
        return out

    def _repo_specs_by_full_name(self) -> dict[str, RepoSpec]:
        if self._repo_specs is None:
            specs = load_repo_specs(SWEQA_PRO_REPOS_TXT)
            self._repo_specs = {spec.full_name: spec for spec in specs}
        return self._repo_specs

    def prepare_environment(self, example: TaskExample) -> Path:
        repo_full_name = example.metadata["repo"]
        specs_by_name = self._repo_specs_by_full_name()
        spec = specs_by_name.get(repo_full_name)
        if spec is None:
            raise KeyError(
                f"{repo_full_name!r} not found in SWE-QA-Pro's own pinned repos.txt "
                f"(commit {SWEQA_PRO_COMMIT}) -- {sorted(specs_by_name)}"
            )
        repo_path = self.repo_root / spec.name
        if repo_path.exists():
            current = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo_path, capture_output=True, text=True
            ).stdout.strip()
            if current == spec.commit:
                return repo_path.resolve()
        fetch_repositories(target_dir=self.repo_root, specs=[spec])
        return repo_path.resolve()

    def score(self, example: TaskExample, result: AgentResult) -> MetricResult:
        template = _load_prompt_template()
        prompt = template.format(
            question=example.question, reference=example.reference, candidate=result.final_answer
        )

        def single_call() -> dict:
            judge_result = call_judge(
                system="You are a helpful assistant.", user=prompt, max_output_tokens=1024
            )
            scores = _parse_judge_scores(judge_result.text)
            return {
                "raw_text": judge_result.text,
                "scores": scores,
                "total_score": sum(scores.values()) if scores else None,
                "judge_cost_usd": judge_result.estimated_cost_usd,
            }

        def native_score_from_call(record: dict) -> float:
            return float(record["total_score"]) if record["total_score"] is not None else 0.0

        native_score, grader_runs = run_judge_noise_protocol(
            judge_type=JudgeType.LLM_SCALAR,
            single_call=single_call,
            native_score_from_call=native_score_from_call,
            n_scalar_calls=3,
        )
        submetrics: dict[str, float] = {}
        for key in _SCORE_KEYS:
            values = [run["scores"][key] for run in grader_runs if run["scores"]]
            if values:
                submetrics[key] = sum(values) / len(values)

        return MetricResult(
            benchmark=self.name,
            task_id=example.task_id,
            native_score=native_score,
            normalized_score=normalize_0_100(native_score, native_min=5.0, native_max=50.0),
            submetrics=submetrics,
            grader_runs=grader_runs,
            metadata={
                "judge_model": DEFAULT_JUDGE_MODEL,
                "n_judge_calls": len(grader_runs),
                "generation_model": result.metadata.get("generation_model", "unknown"),
                "judge_cost_usd": sum(run.get("judge_cost_usd", 0.0) for run in grader_runs),
            },
        )


_ADAPTER = SweQaProAdapter()
register_benchmark(_ADAPTER)
