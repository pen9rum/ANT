from __future__ import annotations

import json
import shutil
from pathlib import Path

from pydantic import BaseModel, Field

from ant.coordinator import LocalCoordinator
from ant.domain import EvidenceState
from ant.evaluation.datasets import EvalExample
from ant.evaluation.judge import judge_answer
from ant.evaluation.metrics import EvalScore, build_reference_idf
from ant.evaluation.runner import BatchResult, _fallback_prediction, run_batch
from ant.evolution import EvolutionEvent, evolve_workers
from ant.memory import IndexStore
from ant.providers import OpenAIProvider


class GenCompareResult(BaseModel):
    """One dataset's gen0 -> slow-gen1 -> fast-gen1 comparison, each stage
    keyed example_id -> EvalScore -- the same shape every ad-hoc gen0/gen1
    comparison run this project has done by hand already wrote to its own
    summary.json, now produced by one reusable function instead of being
    re-derived by a throwaway script each time.
    """

    gen0: dict[str, EvalScore] = Field(default_factory=dict)
    slow_gen1: dict[str, EvalScore] = Field(default_factory=dict)
    fast_gen1: dict[str, EvalScore] = Field(default_factory=dict)
    evolution_events: list[EvolutionEvent] = Field(default_factory=list)
    gen0_worker_count: int = 0
    slow_worker_count: int = 0


def run_gen_compare(
    *,
    examples: list[EvalExample],
    repo_root: Path,
    index_path: Path,
    run_dir: Path,
    max_rounds: int = 6,
    fast_max_rounds: int | None = None,
    judge: str = "heuristic",
    run_gen0: bool = True,
    run_slow_gen1: bool = True,
    run_fast_gen1: bool = True,
) -> GenCompareResult:
    """Runs some or all of gen0 -> slow-gen1 (colony evolution) -> fast-gen1
    (task-conditioned retry) for `examples` against one repo -- any of the
    three stages can be switched off (all four combinations that make sense
    are supported: everything, gen0 only, slow-gen1 only, fast-gen1 only),
    so a stage nothing changed for doesn't have to be re-run every time.
    Saves every stage's per-question EvidenceState JSON under `run_dir`
    (`gen0-<id>.json`, `slow-gen1-<id>.json`, `fast-gen1-<id>.json`) plus a
    `*-results.jsonl` per stage -- both always written when that stage
    actually runs, `run_dir` is the single source of truth for everything
    any call against it has ever produced. A skipped stage's own numbers,
    if a `*-results.jsonl` from an earlier call against this same `run_dir`
    is still there, are read back into this call's own GenCompareResult
    too, so a partial (e.g. fast-gen1-only) re-run's summary still reports
    every stage, not just the one that ran this time. Synthesis (gen0/
    slow-gen1's `ask()`, fast-gen1's `retry_from_trajectory`) and colony
    evolution all require a real reasoner; this always uses OpenAIProvider
    (mirroring `ant retry`'s own `--synthesize openai` requirement -- there
    is no heuristic/mock fallback for any of these).

    gen0 and slow-gen1 both run against `index_path` directly (colony memory
    accumulates there across both stages, matching every prior run's own
    methodology). fast-gen1 instead runs against a frozen copy of
    `index_path` taken right after gen0 finishes, before evolve_workers
    mutates it in place -- a task-conditioned retry repairs gen0's own
    trajectory with gen0's own worker set, it must never silently benefit
    from evolution it never actually saw (see
    LocalCoordinator.retry_from_trajectory's own "task-conditioned, not
    colony reorganization" docstring). A fast-gen1-only call reuses
    whichever snapshot an earlier gen0 call against this `run_dir` already
    froze; if none exists (e.g. this `run_dir` predates that snapshot
    mechanism), it falls back to `index_path` as-is, since that is exactly
    what every fast-gen1 run before this mechanism existed already did.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_fast_max_rounds = fast_max_rounds or max_rounds
    gen0_results_path = run_dir / "gen0-results.jsonl"
    slow_gen1_results_path = run_dir / "slow-gen1-results.jsonl"
    fast_gen1_results_path = run_dir / "fast-gen1-results.jsonl"
    gen0_index_snapshot = run_dir / "_gen0_index_snapshot"

    gen0_worker_count = len(IndexStore(index_path).load_workers())
    if run_gen0:
        run_batch(
            examples=examples,
            repo_root=repo_root,
            index_path=index_path,
            out_path=gen0_results_path,
            max_rounds=max_rounds,
            synthesize="openai",
            judge=judge,
            state_dump_dir=run_dir,
            state_dump_prefix="gen0-",
        )
        if gen0_index_snapshot.exists():
            shutil.rmtree(gen0_index_snapshot)
        shutil.copytree(index_path, gen0_index_snapshot)
    gen0_scores = _load_or_rebuild_scores(
        results_path=gen0_results_path,
        run_dir=run_dir,
        stage_prefix="gen0-",
        examples=examples,
        judge=judge,
    )

    evolution_events: list[EvolutionEvent] = []
    slow_worker_count = gen0_worker_count
    if run_slow_gen1:
        evolution_result = evolve_workers(
            index_path, repo_root=repo_root, reasoner=OpenAIProvider()
        )
        evolution_events = evolution_result.events
        slow_worker_count = evolution_result.worker_count
        run_batch(
            examples=examples,
            repo_root=repo_root,
            index_path=index_path,
            out_path=slow_gen1_results_path,
            max_rounds=max_rounds,
            synthesize="openai",
            judge=judge,
            state_dump_dir=run_dir,
            state_dump_prefix="slow-gen1-",
        )
    slow_gen1_scores = _load_or_rebuild_scores(
        results_path=slow_gen1_results_path,
        run_dir=run_dir,
        stage_prefix="slow-gen1-",
        examples=examples,
        judge=judge,
    )

    if run_fast_gen1:
        missing = [
            example.id
            for example in examples
            if not (run_dir / f"gen0-{example.id}.json").exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"fast-gen1 retries each example's own saved gen0 EvidenceState, but "
                f"{run_dir} has none for: {missing}. Run the gen0 stage against this "
                "run_dir at least once first."
            )
        fast_gen1_index = gen0_index_snapshot if gen0_index_snapshot.exists() else index_path
        _run_fast_gen1(
            examples=examples,
            repo_root=repo_root,
            gen0_index=fast_gen1_index,
            run_dir=run_dir,
            max_rounds=resolved_fast_max_rounds,
            judge=judge,
        )
    fast_gen1_scores = _load_or_rebuild_scores(
        results_path=fast_gen1_results_path,
        run_dir=run_dir,
        stage_prefix="fast-gen1-",
        examples=examples,
        judge=judge,
    )

    return GenCompareResult(
        gen0=gen0_scores,
        slow_gen1=slow_gen1_scores,
        fast_gen1=fast_gen1_scores,
        evolution_events=evolution_events,
        gen0_worker_count=gen0_worker_count,
        slow_worker_count=slow_worker_count,
    )


def _run_fast_gen1(
    *,
    examples: list[EvalExample],
    repo_root: Path,
    gen0_index: Path,
    run_dir: Path,
    max_rounds: int,
    judge: str,
) -> dict[str, EvalScore]:
    provider = OpenAIProvider()
    workers = IndexStore(gen0_index).load_workers()
    idf = build_reference_idf(example.answer for example in examples)
    scores: dict[str, EvalScore] = {}
    out_path = run_dir / "fast-gen1-results.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            gen0_trace_path = run_dir / f"gen0-{example.id}.json"
            prior_state = EvidenceState.model_validate_json(
                gen0_trace_path.read_text(encoding="utf-8")
            )
            coordinator = LocalCoordinator(
                repo_root, workers, synthesizer=provider, index_path=gen0_index
            )
            state = coordinator.retry_from_trajectory(
                prior_state, fast_reasoner=provider, max_rounds=max_rounds
            )
            (run_dir / f"fast-gen1-{example.id}.json").write_text(
                json.dumps(state.model_dump(), indent=2), encoding="utf-8"
            )
            prediction = state.answer or _fallback_prediction(state.evidence)
            score = judge_answer(
                question=example.question,
                prediction=prediction,
                expected=example.answer,
                evidence_count=len(state.evidence),
                unresolved_need_count=len(state.unresolved_needs),
                judge=judge,
                idf=idf,
            )
            scores[example.id] = score
            result = BatchResult(
                example_id=example.id,
                question=example.question,
                prediction=prediction,
                score=score,
                usage=state.usage,
                metadata={"repo": example.repo},
            )
            handle.write(json.dumps(result.model_dump(), ensure_ascii=True) + "\n")
    return scores


def _scores_by_id(results_path: Path) -> dict[str, EvalScore]:
    scores: dict[str, EvalScore] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scores[row["example_id"]] = EvalScore.model_validate(row["score"])
    return scores


def _load_or_rebuild_scores(
    *,
    results_path: Path,
    run_dir: Path,
    stage_prefix: str,
    examples: list[EvalExample],
    judge: str,
) -> dict[str, EvalScore]:
    """A skipped stage's `*-results.jsonl` is the fast path, but it does not
    exist for every run_dir -- notably one written by an ad-hoc pre-
    run_gen_compare script, which saved each stage's per-question
    EvidenceState JSON (this project's own long-standing convention) but
    never a `*-results.jsonl` alongside it. Silently returning {} in that
    case would make a partial re-run's summary.json overwrite a
    previously-complete stage's numbers with nothing, discarding real
    already-computed results still sitting right there on disk (confirmed
    live: this exact bug erased gen0/slow-gen1 aggregates for two run_dirs
    on a fast-gen1-only re-run before this fallback existed). Rebuilds by
    re-judging each stage-<id>.json trace's own saved answer instead --
    the real prediction is preserved verbatim, only the judge call is
    repeated -- and persists the result as this stage's own
    `*-results.jsonl` so this rebuild only ever happens once per run_dir.
    """
    if results_path.exists():
        return _scores_by_id(results_path)
    idf = build_reference_idf(example.answer for example in examples)
    scores: dict[str, EvalScore] = {}
    rows: list[dict] = []
    for example in examples:
        trace_path = run_dir / f"{stage_prefix}{example.id}.json"
        if not trace_path.exists():
            continue
        state = EvidenceState.model_validate_json(trace_path.read_text(encoding="utf-8"))
        prediction = state.answer or _fallback_prediction(state.evidence)
        score = judge_answer(
            question=example.question,
            prediction=prediction,
            expected=example.answer,
            evidence_count=len(state.evidence),
            unresolved_need_count=len(state.unresolved_needs),
            judge=judge,
            idf=idf,
        )
        scores[example.id] = score
        rows.append(
            BatchResult(
                example_id=example.id,
                question=example.question,
                prediction=prediction,
                score=score,
                usage=state.usage,
                metadata={"repo": example.repo},
            ).model_dump()
        )
    if rows:
        with results_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return scores
