from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ant.coordinator import LocalCoordinator
from ant.evaluation.datasets import EvalExample
from ant.evaluation.judge import judge_answer
from ant.evaluation.metrics import EvalScore
from ant.memory import IndexStore
from ant.memory.colony import ColonyMemoryStore, record_task_memory
from ant.providers import OpenAIProvider


class BatchResult(BaseModel):
    example_id: str
    question: str
    prediction: str
    score: EvalScore
    status: str = "completed"
    trace_id: int | None = None
    metadata: dict = Field(default_factory=dict)


def run_batch(
    *,
    examples: list[EvalExample],
    repo_root: Path,
    index_path: Path,
    out_path: Path,
    max_rounds: int = 2,
    synthesize: str = "none",
    judge: str = "heuristic",
) -> list[BatchResult]:
    provider = OpenAIProvider() if synthesize == "openai" else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[BatchResult] = []
    with out_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            example_repo = _resolve_repo(repo_root, example.repo)
            if example_repo is None:
                result = BatchResult(
                    example_id=example.id,
                    question=example.question,
                    prediction="",
                    score=EvalScore(
                        exact_match=False,
                        contains_answer=False,
                        evidence_count=0,
                        unresolved_need_count=0,
                    ),
                    status="skipped_missing_repo",
                    metadata={"repo": example.repo},
                )
                results.append(result)
                handle.write(json.dumps(result.model_dump(), ensure_ascii=True) + "\n")
                continue

            try:
                result = _run_example(
                    example=example,
                    example_repo=example_repo,
                    index_path=index_path,
                    provider=provider,
                    max_rounds=max_rounds,
                    judge=judge,
                )
            except Exception as exc:  # noqa: BLE001 - one bad example must not sink the batch
                # A single malformed model response (or any other failure)
                # here used to propagate all the way out of run_batch and
                # abort every remaining example in the batch. Isolate it per
                # example instead: record what happened and keep going.
                result = BatchResult(
                    example_id=example.id,
                    question=example.question,
                    prediction="",
                    score=EvalScore(
                        exact_match=False,
                        contains_answer=False,
                        evidence_count=0,
                        unresolved_need_count=0,
                    ),
                    status=f"error: {type(exc).__name__}: {exc}",
                    metadata={"repo": example.repo},
                )
            results.append(result)
            handle.write(json.dumps(result.model_dump(), ensure_ascii=True) + "\n")
    return results


def _run_example(
    *,
    example: EvalExample,
    example_repo: Path,
    index_path: Path,
    provider: OpenAIProvider | None,
    max_rounds: int,
    judge: str,
) -> BatchResult:
    example_index = (
        index_path if example.repo == "." else index_path / _repo_basename(example.repo)
    )
    if (
        not (example_index / "workers.json").exists()
        and not (example_index / "ant.sqlite3").exists()
    ):
        _build_index(example_repo, example_index)
    store = IndexStore(example_index)
    colony_memory = ColonyMemoryStore(example_index)
    workers = store.load_workers()
    memory_routes = colony_memory.matching_routes(example.question.split())
    coordinator = LocalCoordinator(
        example_repo,
        workers,
        synthesizer=provider,
        memory_routes=memory_routes,
        index_path=example_index,
    )
    state = coordinator.ask(example.question, max_rounds=max_rounds)
    trace_id = store.save_trace(state)
    prediction = state.answer or _fallback_prediction(state.evidence)
    score = judge_answer(
        question=example.question,
        prediction=prediction,
        expected=example.answer,
        evidence_count=len(state.evidence),
        unresolved_need_count=len(state.unresolved_needs),
        judge=judge,
    )
    record_task_memory(
        colony_memory,
        example.question,
        state,
        is_high_quality=_is_high_quality_route(score),
        route_weight=_route_weight(score),
    )
    return BatchResult(
        example_id=example.id,
        question=example.question,
        prediction=prediction,
        score=score,
        trace_id=trace_id,
        metadata={"repo": example.repo},
    )


def _fallback_prediction(evidence) -> str:
    return "\n".join(f"{item.path}:{item.line_start}: {item.quote}" for item in evidence[:4])


def _is_high_quality_route(score: EvalScore) -> bool:
    if score.exact_match or score.contains_answer:
        return True
    if score.correctness is None or score.completeness is None:
        return False
    return score.correctness >= 8 and score.completeness >= 8


def _route_weight(score: EvalScore) -> float:
    if score.correctness is not None and score.completeness is not None:
        return round((score.correctness + score.completeness) / 4, 2)
    return 2.0


def _resolve_repo(repo_root: Path, repo: str) -> Path | None:
    if repo == ".":
        return repo_root
    candidates = [
        repo_root / repo,
        repo_root / _repo_basename(repo),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _repo_basename(repo: str) -> str:
    return repo.rstrip("/").removesuffix(".git").split("/")[-1]


def _build_index(repo: Path, index_path: Path) -> None:
    from ant.environment import RepoEnvironment
    from ant.generation import generate_worker_cards
    from ant.indexing import discover_territories

    environment = RepoEnvironment(repo)
    territories = discover_territories(environment)
    workers = generate_worker_cards(environment.root, territories)
    IndexStore(index_path).save(territories, workers)
