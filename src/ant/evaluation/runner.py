from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ant.coordinator import LocalCoordinator
from ant.evaluation.datasets import EvalExample
from ant.evaluation.judge import judge_answer
from ant.evaluation.metrics import EvalScore
from ant.memory import IndexStore
from ant.memory.colony import CoalitionRecord, ColonyMemoryStore
from ant.providers import OpenAIProvider


class BatchResult(BaseModel):
    example_id: str
    question: str
    prediction: str
    score: EvalScore
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
    store = IndexStore(index_path)
    colony_memory = ColonyMemoryStore(index_path)
    workers = store.load_workers()
    provider = OpenAIProvider() if synthesize == "openai" else None
    coordinator = LocalCoordinator(repo_root, workers, synthesizer=provider)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[BatchResult] = []
    with out_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            state = coordinator.ask(example.question, max_rounds=max_rounds)
            trace_id = store.save_trace(state)
            for round_state in state.rounds:
                if len(round_state.selected_worker_ids) > 1:
                    colony_memory.record_coalition(
                        CoalitionRecord(
                            worker_ids=round_state.selected_worker_ids,
                            question=example.question,
                            evidence_count=len(state.evidence),
                            unresolved_need_count=len(state.unresolved_needs),
                        )
                    )
            prediction = state.answer or _fallback_prediction(state.evidence)
            score = judge_answer(
                question=example.question,
                prediction=prediction,
                expected=example.answer,
                evidence_count=len(state.evidence),
                unresolved_need_count=len(state.unresolved_needs),
                judge=judge,
            )
            result = BatchResult(
                example_id=example.id,
                question=example.question,
                prediction=prediction,
                score=score,
                trace_id=trace_id,
                metadata={"repo": example.repo},
            )
            results.append(result)
            handle.write(json.dumps(result.model_dump(), ensure_ascii=True) + "\n")
    return results


def _fallback_prediction(evidence) -> str:
    return "\n".join(f"{item.path}:{item.line_start}: {item.quote}" for item in evidence[:4])
