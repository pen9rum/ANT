"""WebWalkerQA benchmark adapter -- the `TaskExample`/`BenchmarkAdapter`
layer over `ant.evaluation_suite.webwalkerqa`'s existing dataset-loading
infrastructure (see that package's own docstring: it deliberately stops
at loading/filtering/manifesting and does not crawl URLs or call an LLM).
This module adds the missing pieces: a `TaskExample` producer with
structural gold-field isolation, environment preparation (a per-task page-
cache directory -- no live crawl happens here, per the governing spec's
"environment/setup only" scope), and a scorer.

SCORING PROTOCOL NOTE: `ant.evaluation_suite.scoring.JudgeType` already
classifies WebWalkerQA as `LLM_BINARY` (correct/incorrect majority vote
over `n_binary_calls=3`), based on a prior audit of the paper (its own
scorer: "exact-match rejected as infeasible given variable answer
lengths... CoT-prompted comparison against ground truth instead" --
`docs/webwalkerqa_ant_mapping.md` section 3). The paper does NOT publish a
fixed, machine-readable judge prompt template the way RepoProbe/SWE-QA-Pro
do (their official repos ship one; WebWalkerQA's own `evaluate.py`
describes the comparison narratively) -- so `_JUDGE_PROMPT` below is THIS
SUITE'S OWN construction, not a vendored official template. This is
disclosed explicitly, not silently presented as "the official protocol":
the SCORING SEMANTICS (binary correctness judged against a reference
answer) are preserved; the exact prompt wording is not paper-official.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.judge import DEFAULT_JUDGE_MODEL, call_judge
from ant.evaluation_suite.registry import register_benchmark
from ant.evaluation_suite.scoring import (
    JudgeType,
    MetricResult,
    normalize_0_100,
    run_judge_noise_protocol,
)
from ant.evaluation_suite.webwalkerqa.loader import (
    filter_english,
    inference_view,
    load_webwalkerqa_records,
)

_JUDGE_PROMPT = """You are judging whether a candidate answer to a web-navigation \
question is correct, compared against a reference answer.

Question: {question}

Reference answer: {reference}

Candidate answer: {candidate}

The candidate need not match the reference's exact wording -- judge whether it \
conveys the same factual answer. Respond with ONLY a JSON object of the form \
{{"correct": true or false, "reasoning": "one short sentence"}}."""


class WebWalkerQaAdapter:
    """`name = "webwalkerqa"`. `repo_filter` (per the shared
    `BenchmarkAdapter` protocol's parameter name) is reinterpreted here as
    a DOMAIN filter (education/conference/game/organization) -- there is
    no "repo" concept for a web-navigation benchmark; the parameter name
    is kept only for structural interface compatibility with every other
    adapter in this suite, not because "repo" is semantically meaningful
    here.
    """

    name = "webwalkerqa"

    def __init__(self, cache_root: Path | None = None) -> None:
        self.cache_root = cache_root or Path(".ant/eval-suite-web") / self.name

    def load_examples(
        self, limit: int | None = None, repo_filter: str | None = None
    ) -> list[TaskExample]:
        records = load_webwalkerqa_records()
        records = filter_english(records)
        if repo_filter is not None:
            target = repo_filter.strip().lower()
            records = [r for r in records if r.domain.strip().lower() == target]
        if limit is not None:
            records = records[:limit]

        out: list[TaskExample] = []
        for record in records:
            # structurally omits gold_answer/source_websites/golden_path
            view = inference_view(record)
            question = view.pop("question")
            example_id = view.pop("example_id")
            out.append(
                TaskExample(
                    benchmark=self.name,
                    task_id=example_id,
                    question=question,
                    # reference is read ONLY by .score(), never by a generation agent
                    reference=record.gold_answer,
                    metadata={
                        **view,
                        # AUDIT/DIAGNOSTIC ONLY -- see this module's own docstring and
                        # tests/test_webwalkerqa_adapter.py's leakage-prevention tests.
                        # No generation agent code path reads this key; it exists so an
                        # OFFLINE validation script (accessibility checks, eligibility
                        # classification) can inspect gold source pages without those
                        # pages ever reaching inference.
                        "_audit_only": {
                            "source_websites": list(record.source_websites),
                            "golden_path": list(record.golden_path),
                        },
                    },
                )
            )
        return out

    def prepare_environment(self, example: TaskExample) -> Path:
        """Returns this task's local page-cache directory -- creates it if
        absent, but performs NO live crawl (no HTTP request happens here).
        The root URL to navigate from lives at `example.metadata["root_url"]`
        (populated by `inference_view()`); an agent constructs its own
        `EvalWebEnvironment(root_url=..., cache=PageCache(cache_dir=<this
        path>, ...))` from these two pieces, matching how a repo-QA agent
        constructs `EvalRepoEnvironment(environment_root)` from the Path
        `prepare_environment()` returns there.
        """
        cache_dir = self.cache_root / example.task_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def score(self, example: TaskExample, result: AgentResult) -> MetricResult:
        def single_call() -> dict:
            prompt = _JUDGE_PROMPT.format(
                question=example.question,
                reference=example.reference,
                candidate=result.final_answer,
            )
            judge_result = call_judge(
                system="You are a careful, literal fact-checking judge.", user=prompt
            )
            payload = _extract_json(judge_result.text)
            correct = bool(payload.get("correct")) if isinstance(payload, dict) else False
            return {
                "raw_text": judge_result.text,
                "correct": correct,
                "reasoning": payload.get("reasoning", "") if isinstance(payload, dict) else "",
                "judge_cost_usd": judge_result.estimated_cost_usd,
            }

        def native_score_from_call(record: dict) -> float:
            return 1.0 if record["correct"] else 0.0

        native_score, grader_runs = run_judge_noise_protocol(
            judge_type=JudgeType.LLM_BINARY,
            single_call=single_call,
            native_score_from_call=native_score_from_call,
            n_binary_calls=3,
        )

        return MetricResult(
            benchmark=self.name,
            task_id=example.task_id,
            native_score=native_score,
            normalized_score=normalize_0_100(native_score, native_min=0.0, native_max=1.0),
            grader_runs=grader_runs,
            metadata={
                "judge_model": DEFAULT_JUDGE_MODEL,
                "n_judge_calls": len(grader_runs),
                "generation_model": result.metadata.get("generation_model", "unknown"),
                "judge_cost_usd": sum(run.get("judge_cost_usd", 0.0) for run in grader_runs),
                "judge_prompt_is_official": False,
                "judge_prompt_note": (
                    "WebWalkerQA's own evaluate.py describes its judge narratively, with "
                    "no published fixed prompt template to vendor -- this is this suite's "
                    "own binary-correctness judge prompt, not an official one. The scoring "
                    "SEMANTICS (LLM-judged correctness vs. a reference answer) match the "
                    "paper; the exact wording does not."
                ),
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


_ADAPTER = WebWalkerQaAdapter()
register_benchmark(_ADAPTER)
