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
from ant.memory import ColonyMemoryStore, IndexStore
from ant.providers import OpenAIProvider


class GenerationSnapshot(BaseModel):
    """Population/evolution audit trail for one evolve_workers() call that
    produced slow-gen{generation} -- everything needed to answer "what
    actually changed this cycle, and how much experience had the colony
    accumulated when it decided that" without re-deriving it from raw
    trace files. Persisted alongside that generation's own *-results.jsonl
    (slow-gen{generation}-evolution.json) so a later call against the same
    run_dir that doesn't recompute this generation can still report it --
    same "skipped stage's own numbers are still read back" convention
    every score dict here already follows.
    """

    generation: int
    worker_count: int = 0
    added_worker_ids: list[str] = Field(default_factory=list)
    removed_worker_ids: list[str] = Field(default_factory=list)
    events: list[EvolutionEvent] = Field(default_factory=list)
    active_high_quality_route_count: int = 0
    total_route_count: int = 0
    accumulated_task_count: int = 0
    # Population/lifecycle (Phase 8 instrumentation, multi-generation
    # organizational evolution redesign): overlay_worker_ids is every
    # worker with real structural lineage (WorkerCard.parent_worker_ids
    # non-empty) as of THIS generation -- the complement of "base" -- so
    # base-vs-overlay population size is directly readable without
    # re-deriving it from the raw worker list. lifecycle_counts /
    # structural_action_counts are the same population/events grouped by
    # WorkerCard.lifecycle_state and EvolutionEvent.kind respectively.
    overlay_worker_ids: list[str] = Field(default_factory=list)
    lifecycle_counts: dict[str, int] = Field(default_factory=dict)
    structural_action_counts: dict[str, int] = Field(default_factory=dict)
    # Route memory (Phase 6/8): raw_route_proposals is what total_route_count
    # would be WITHOUT consolidation (sum of every route's occurrence_count
    # -- see ColonyMemoryStore.route_stats) -- the two together are the
    # direct audit of whether consolidation is actually preventing
    # unbounded cardinality growth (the qibo 49->80->117 pattern this whole
    # phase exists to address) while confidence/support keeps accumulating.
    raw_route_proposals: int = 0


class GenCompareResult(BaseModel):
    """gen0 -> N cumulative slow generations (gen0's own memory, evolved
    and re-evolved in place, each generation's own run adding more
    experience for the next) -> an optional per-generation fast repair
    pass, each stage keyed example_id -> EvalScore. slow_generations/
    fast_generations/generation_snapshots are keyed by generation number:
    slow_generations[1] is slow-gen1 (the first evolved generation),
    fast_generations[0] is fast-gen0 (a task-conditioned repair of gen0's
    own trajectory -- see run_gen_compare's own docstring for why fast
    never stacks across generations the way slow does).
    """

    gen0: dict[str, EvalScore] = Field(default_factory=dict)
    slow_generations: dict[int, dict[str, EvalScore]] = Field(default_factory=dict)
    fast_generations: dict[int, dict[str, EvalScore]] = Field(default_factory=dict)
    generation_snapshots: dict[int, GenerationSnapshot] = Field(default_factory=dict)
    gen0_worker_count: int = 0


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
    slow_generations: int = 1,
    fast_generations: int = 0,
    start_generation: int = 1,
) -> GenCompareResult:
    """Runs gen0 -> up to `slow_generations` CUMULATIVE colony-evolution
    generations -> an optional fast (task-conditioned retry) repair pass
    per generation, for `examples` against one repo.

    SLOW is genuinely cumulative: slow-gen1 evolves gen0's own memory and
    runs on the result; slow-gen2 evolves slow-gen1's memory (gen0's
    experience PLUS everything slow-gen1's own run just added) and runs on
    THAT result; slow-genN keeps compounding the same way. Every
    generation's evolve_workers() call and run_batch() call operate on the
    SAME real, persistent index at `resolved_index_path` (see below) --
    there is no separate "per-generation" index, the whole point is that
    later generations see everything earlier ones learned.

    FAST is deliberately NOT cumulative -- fast-gen{k} is always a single
    task-conditioned repair of generation k's OWN trajectory (gen0-<id>.json
    for k=0, slow-gen{k}-<id>.json for k>=1), run against a FROZEN snapshot
    of the index taken right after generation k's own run finished (before
    any LATER generation's evolve_workers() mutates it further) -- never
    fast-of-fast. Stacking repairs across generations would blur "ephemeral,
    task-conditioned repair" and "persistent, colony-wide evolution" into
    one concept; keeping fast anchored per-generation and always exactly
    one hop from that generation's own real trajectory keeps the two
    orthogonal, matching retry_from_trajectory's own "task-conditioned, not
    colony reorganization" design.

    `run_gen0` toggles whether THIS call executes gen0's own run (default
    True); `start_generation` (default 1) is the first slow generation THIS
    call actually computes -- every generation below it is assumed to
    already exist on disk from an earlier call against this same run_dir
    (its own *-results.jsonl/evolution.json/index snapshot reloaded, not
    recomputed) and every generation from `start_generation` through
    `slow_generations` (inclusive) is computed fresh. This is the resume
    path: a call with `run_gen0=False, start_generation=3, slow_generations=3`
    against a run_dir that already has gen0 through slow-gen2 on disk from
    an earlier call picks up exactly at slow-gen3, without re-evolving or
    re-running anything already done. fast_generations (default 0) is how
    many population states starting at gen0 (0, 1, 2, ... fast_generations-1)
    get a fast pass THIS call -- always recomputed fresh regardless of
    start_generation (a fast pass is one repair call per question, no
    evolution, cheap enough that resuming it isn't worth the extra
    bookkeeping the way skipping a re-evolution is).

    Saves every stage's per-question EvidenceState JSON under `run_dir`
    (`gen0-<id>.json`, `slow-gen{k}-<id>.json`, `fast-gen{k}-<id>.json`)
    plus a `*-results.jsonl` per stage. Synthesis (`ask()`,
    `retry_from_trajectory`) and colony evolution all require a real
    reasoner; this always uses OpenAIProvider (mirroring `ant retry`'s own
    `--synthesize openai` requirement -- there is no heuristic/mock
    fallback for any of these).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_fast_max_rounds = fast_max_rounds or max_rounds
    gen0_results_path = run_dir / "gen0-results.jsonl"
    gen0_index_snapshot = run_dir / "_gen0_index_snapshot"

    # run_batch's own _run_example resolves its REAL per-repo index to
    # index_path / _repo_basename(example.repo) whenever a dataset row's
    # repo isn't "." (every "owner/repo"-formatted dataset row this project
    # uses) -- `index_path` itself is left untouched, never read or written
    # by any actual question in that case. Everything below that reads or
    # snapshots "the index gen0 actually used" (worker count, the
    # per-generation evolution snapshots, evolve_workers itself) must use
    # this SAME resolved path, not the raw one -- passing the raw top-level
    # index_path to any of them silently operated on a permanently-empty
    # directory instead. Confirmed live: pennylane's real per-question
    # index had 45 recorded routes; the top-level index evolve_workers was
    # previously called with had 0.
    resolved_index_path = (
        index_path if examples[0].repo == "." else index_path / _repo_basename(examples[0].repo)
    )
    worker_repo_root = _resolve_repo(repo_root, examples[0].repo) or repo_root

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
        _snapshot_index(resolved_index_path, gen0_index_snapshot)
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

    slow_generation_scores: dict[int, dict[str, EvalScore]] = {}
    generation_snapshots: dict[int, GenerationSnapshot] = {}
    for generation in range(1, slow_generations + 1):
        results_path = run_dir / f"slow-gen{generation}-results.jsonl"
        index_snapshot_path = run_dir / f"_gen{generation}_index_snapshot"
        metadata_path = run_dir / f"slow-gen{generation}-evolution.json"
        if generation >= start_generation:
            memory = ColonyMemoryStore(resolved_index_path)
            before_worker_ids = {w.id for w in IndexStore(resolved_index_path).load_workers()}
            evolution_result = evolve_workers(
                resolved_index_path,
                repo_root=worker_repo_root,
                reasoner=OpenAIProvider(),
                generation=generation,
            )
            after_worker_ids = {w.id for w in IndexStore(resolved_index_path).load_workers()}
            run_batch(
                examples=examples,
                repo_root=repo_root,
                index_path=index_path,
                out_path=results_path,
                max_rounds=max_rounds,
                synthesize="openai",
                judge=judge,
                state_dump_dir=run_dir,
                state_dump_prefix=f"slow-gen{generation}-",
            )
            _snapshot_index(resolved_index_path, index_snapshot_path)
            current_workers = IndexStore(resolved_index_path).load_workers()
            structural_action_counts: dict[str, int] = {}
            for event in evolution_result.events:
                structural_action_counts[event.kind] = (
                    structural_action_counts.get(event.kind, 0) + 1
                )
            lifecycle_counts: dict[str, int] = {}
            for worker in current_workers:
                lifecycle_counts[worker.lifecycle_state] = (
                    lifecycle_counts.get(worker.lifecycle_state, 0) + 1
                )
            route_stats = memory.route_stats(include_stale=True)
            snapshot = GenerationSnapshot(
                generation=generation,
                worker_count=evolution_result.worker_count,
                added_worker_ids=sorted(after_worker_ids - before_worker_ids),
                removed_worker_ids=sorted(before_worker_ids - after_worker_ids),
                events=evolution_result.events,
                active_high_quality_route_count=len(
                    [route for route in memory.all_routes() if route.is_high_quality]
                ),
                total_route_count=len(memory.all_routes(include_stale=True)),
                accumulated_task_count=memory.distinct_task_count(),
                overlay_worker_ids=sorted(
                    worker.id for worker in current_workers if worker.parent_worker_ids
                ),
                lifecycle_counts=lifecycle_counts,
                structural_action_counts=structural_action_counts,
                raw_route_proposals=route_stats["raw_route_proposals"],
            )
            metadata_path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
            generation_snapshots[generation] = snapshot
        elif metadata_path.exists():
            generation_snapshots[generation] = GenerationSnapshot.model_validate_json(
                metadata_path.read_text(encoding="utf-8")
            )
        slow_generation_scores[generation] = _load_or_rebuild_scores(
            results_path=results_path,
            run_dir=run_dir,
            stage_prefix=f"slow-gen{generation}-",
            examples=examples,
            judge=judge,
        )

    fast_generation_scores: dict[int, dict[str, EvalScore]] = {}
    for generation in range(fast_generations):
        trace_prefix = "gen0-" if generation == 0 else f"slow-gen{generation}-"
        anchor_snapshot = (
            gen0_index_snapshot
            if generation == 0
            else run_dir / f"_gen{generation}_index_snapshot"
        )
        missing = [
            example.id
            for example in examples
            if not (run_dir / f"{trace_prefix}{example.id}.json").exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"fast-gen{generation} retries generation {generation}'s own saved "
                f"trajectory, but {run_dir} has none for: {missing}. Run that "
                "generation against this run_dir at least once first."
            )
        fast_index = anchor_snapshot if anchor_snapshot.exists() else resolved_index_path
        fast_generation_scores[generation] = _run_fast_generation(
            examples=examples,
            repo_root=repo_root,
            anchor_index=fast_index,
            run_dir=run_dir,
            trace_prefix=trace_prefix,
            out_prefix=f"fast-gen{generation}-",
            max_rounds=resolved_fast_max_rounds,
            judge=judge,
        )

    return GenCompareResult(
        gen0=gen0_scores,
        slow_generations=slow_generation_scores,
        fast_generations=fast_generation_scores,
        generation_snapshots=generation_snapshots,
        gen0_worker_count=gen0_worker_count,
    )


def _snapshot_index(resolved_index_path: Path, snapshot_path: Path) -> None:
    if snapshot_path.exists():
        shutil.rmtree(snapshot_path)
    shutil.copytree(resolved_index_path, snapshot_path)


def _run_fast_generation(
    *,
    examples: list[EvalExample],
    repo_root: Path,
    anchor_index: Path,
    run_dir: Path,
    trace_prefix: str,
    out_prefix: str,
    max_rounds: int,
    judge: str,
) -> dict[str, EvalScore]:
    provider = OpenAIProvider()
    workers = IndexStore(anchor_index).load_workers()
    idf = build_reference_idf(example.answer for example in examples)
    scores: dict[str, EvalScore] = {}
    # Read once, not per-example: when the monotonic gate (see
    # LocalCoordinator.ask's own docstring) leaves prior_state.answer
    # completely untouched, re-judging byte-identical text is pure
    # judge-sampling noise, not signal -- confirmed live on a real
    # yt-dlp run, 6/10 questions hit this gate, and the SAME text scored
    # up to +-4 points apart across two independent judge calls (f1, a
    # deterministic metric, was identical both times; only the LLM judge's
    # own rubric scores moved). Reusing the anchor generation's own
    # already-judged score for these removes that noise entirely, at zero
    # extra judge cost.
    prior_results_path = run_dir / f"{trace_prefix}results.jsonl"
    prior_scores = _scores_by_id(prior_results_path) if prior_results_path.exists() else {}
    out_path = run_dir / f"{out_prefix}results.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            prior_trace_path = run_dir / f"{trace_prefix}{example.id}.json"
            prior_state = EvidenceState.model_validate_json(
                prior_trace_path.read_text(encoding="utf-8")
            )
            coordinator = LocalCoordinator(
                repo_root, workers, synthesizer=provider, index_path=anchor_index
            )
            state = coordinator.retry_from_trajectory(
                prior_state, fast_reasoner=provider, max_rounds=max_rounds
            )
            (run_dir / f"{out_prefix}{example.id}.json").write_text(
                json.dumps(state.model_dump(), indent=2), encoding="utf-8"
            )
            prediction = state.answer or _fallback_prediction(state.evidence)
            reused_score = (
                prior_scores.get(example.id)
                if state.answer and state.answer == prior_state.answer
                else None
            )
            score = reused_score or judge_answer(
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
