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
from ant.evaluation.runner import (
    BatchResult,
    _fallback_prediction,
    _repo_basename,
    _resolve_repo,
    run_batch,
)
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

    gen0 and slow-gen1 both run against `index_path`'s real per-repo index
    (colony memory accumulates there across both stages, matching every
    prior run's own methodology) -- see `resolved_index_path` below for
    exactly which directory that is. fast-gen1 instead runs against a
    frozen copy of that same real index taken right after gen0 finishes,
    before evolve_workers mutates it in place -- a task-conditioned retry
    repairs gen0's own trajectory with gen0's own worker set, it must never
    silently benefit from evolution it never actually saw (see
    LocalCoordinator.retry_from_trajectory's own "task-conditioned, not
    colony reorganization" docstring). A fast-gen1-only call reuses
    whichever snapshot an earlier gen0 call against this `run_dir` already
    froze; if none exists (e.g. this `run_dir` predates that snapshot
    mechanism), it falls back to the real index as-is.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_fast_max_rounds = fast_max_rounds or max_rounds
    gen0_results_path = run_dir / "gen0-results.jsonl"
    slow_gen1_results_path = run_dir / "slow-gen1-results.jsonl"
    fast_gen1_results_path = run_dir / "fast-gen1-results.jsonl"
    gen0_index_snapshot = run_dir / "_gen0_index_snapshot"
    evolution_metadata_path = run_dir / "slow-gen1-evolution.json"

    # run_batch's own _run_example resolves its REAL per-repo index to
    # index_path / _repo_basename(example.repo) whenever a dataset row's
    # repo isn't "." (every "owner/repo"-formatted dataset row this project
    # uses) -- `index_path` itself is left untouched, never read or written
    # by any actual question in that case. Everything below that reads or
    # snapshots "the index gen0 actually used" (worker count, the pre-
    # evolution snapshot fast-gen1 replays against, evolve_workers itself)
    # must use this SAME resolved path, not the raw one -- passing the raw
    # top-level index_path to any of them silently operated on a
    # permanently-empty directory instead. Confirmed live: pennylane's real
    # per-question index had 45 recorded routes; the top-level index
    # evolve_workers was previously called with had 0 -- specialize/birth/
    # merge found nothing to act on, silently, for every prior gen0/
    # slow-gen1 run against an "owner/repo"-formatted dataset this project
    # has ever done. `run_batch` itself keeps receiving the raw `index_path`
    # below (unchanged) -- it does this same resolution per example
    # internally, so pre-resolving it here would double-nest.
    resolved_index_path = (
        index_path if examples[0].repo == "." else index_path / _repo_basename(examples[0].repo)
    )

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
        shutil.copytree(resolved_index_path, gen0_index_snapshot)
    # Read after run_gen0 (if it ran), so a first-ever call against a fresh
    # repo doesn't try to read resolved_index_path before run_batch's own
    # _build_index has had a chance to create it.
    try:
        gen0_worker_count = len(IndexStore(resolved_index_path).load_workers())
    except FileNotFoundError:
        gen0_worker_count = 0
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
        # `repo_root` here is run_gen_compare's own raw --repo value, which
        # for an "owner/repo"-formatted dataset (see _resolve_repo) must be
        # the checkout's PARENT directory so run_batch's own per-example
        # resolution (repo_root / basename fallback) lands on the actual
        # repo -- not the repo root itself. evolve_workers, by contrast,
        # needs the actual repo root directly (its specialize path reads a
        # worker's own files straight off `repo_root / file`, both via
        # _child_worker's card rebuild and _semantic_groups' live
        # retrieval). Passing the raw parent-dir repo_root straight through
        # here silently pointed both at the wrong files -- caught before it
        # ever ran live, since specialization/semantic-clustering had never
        # actually fired against an owner/repo-formatted dataset before.
        # Resolve once, the same way run_batch resolves it per example.
        # `repo_root` here is run_gen_compare's own raw --repo value, which
        # for an "owner/repo"-formatted dataset (see _resolve_repo) must be
        # the checkout's PARENT directory so run_batch's own per-example
        # resolution (repo_root / basename fallback) lands on the actual
        # repo -- not the repo root itself. evolve_workers, by contrast,
        # needs the actual repo root directly (its specialize path reads a
        # worker's own files straight off `repo_root / file`, both via
        # _child_worker's card rebuild and _semantic_groups' live
        # retrieval). Resolve once, the same way run_batch resolves it per
        # example.
        worker_repo_root = _resolve_repo(repo_root, examples[0].repo) or repo_root
        evolution_result = evolve_workers(
            resolved_index_path, repo_root=worker_repo_root, reasoner=OpenAIProvider()
        )
        evolution_events = evolution_result.events
        slow_worker_count = evolution_result.worker_count
        # Persist alongside gen0/slow-gen1's own *-results.jsonl so a LATER
        # call against this same run_dir that skips slow-gen1 (e.g. this
        # project's own two-call pattern for "owner/repo"-formatted
        # datasets: a separate --no-slow-gen1 --fast-gen1 call) can still
        # report it -- see the reload branch below. Regression fix for a
        # real bug: without this, that later call re-initialized
        # evolution_events=[]/slow_worker_count=gen0_worker_count above and
        # its own summary.json write silently overwrote this call's real,
        # non-empty telemetry with those defaults.
        evolution_metadata_path.write_text(
            json.dumps(
                {
                    "events": [event.model_dump() for event in evolution_events],
                    "worker_count": slow_worker_count,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
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
    elif evolution_metadata_path.exists():
        metadata = json.loads(evolution_metadata_path.read_text(encoding="utf-8"))
        evolution_events = [EvolutionEvent.model_validate(event) for event in metadata["events"]]
        slow_worker_count = metadata["worker_count"]
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
        fast_gen1_index = (
            gen0_index_snapshot if gen0_index_snapshot.exists() else resolved_index_path
        )
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
    # Read once, not per-example: when the monotonic gate (see
    # LocalCoordinator.ask's own docstring) leaves prior_state.answer
    # completely untouched, re-judging byte-identical text is pure
    # judge-sampling noise, not signal -- confirmed live on a real
    # yt-dlp run, 6/10 questions hit this gate, and the SAME text scored
    # up to +-4 points apart across the two independent judge calls (f1,
    # a deterministic metric, was identical both times; only the LLM
    # judge's own rubric scores moved). Reusing gen0's own already-judged
    # score for these removes that noise from the fast-gen1 column
    # entirely, at zero extra judge cost.
    gen0_results_path = run_dir / "gen0-results.jsonl"
    gen0_scores = _scores_by_id(gen0_results_path) if gen0_results_path.exists() else {}
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
            reused_gen0_score = (
                gen0_scores.get(example.id)
                if state.answer and state.answer == prior_state.answer
                else None
            )
            score = reused_gen0_score or judge_answer(
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
