from __future__ import annotations

import json
import time
from pathlib import Path

from ant.domain import WorkerCard
from ant.evaluation.datasets import EvalExample
from ant.evaluation.judge import judge_answer
from ant.evaluation.metrics import build_reference_idf
from ant.memory import IndexStore
from ant.providers import OpenAIProvider
from ant.providers.openai_provider import _loads_json_object
from ant.tools.local import LocalSearchTool
from ant.workers.autonomous import AutonomousWorker, WorkerRunConfig

TIER1_PROMPT = (
    "You are answering a question about a software repository from memory "
    "only -- you have NO access to its source code, only what you already "
    "know about this project in general. Answer as best you can, being "
    "honest about uncertainty rather than inventing specifics you cannot "
    "verify.\n\nQuestion: {question}"
)


def _all_files(index_path: Path) -> list[str]:
    workers = IndexStore(index_path).load_workers()
    files: set[str] = set()
    for worker in workers:
        files.update(worker.files)
    return sorted(files)


def run_tier1_closed_book(
    examples: list[EvalExample], out_path: Path, judge: str = "openai"
) -> list[dict]:
    """Tier 1: closed-book / long-context LM, no repo access at all."""
    provider = OpenAIProvider()
    idf = build_reference_idf(example.answer for example in examples)
    rows: list[dict] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            started = time.time()
            result = provider.responses_text(
                TIER1_PROMPT.format(question=example.question), max_output_tokens=1024
            )
            prediction = result.text.strip()
            run_cost = provider.drain_usage().estimated_cost_usd
            score = judge_answer(
                question=example.question,
                prediction=prediction,
                expected=example.answer,
                evidence_count=0,
                unresolved_need_count=0,
                judge=judge,
                idf=idf,
            )
            row = {
                "id": example.id,
                "question": example.question,
                "prediction": prediction,
                "score": score.model_dump(),
                "run_cost_usd": run_cost,
                "judge_cost_usd": score.usage.estimated_cost_usd,
                "elapsed_seconds": round(time.time() - started, 2),
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    (out_path.parent / (out_path.stem + ".json")).write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return rows


_TIER2_QUERY_PROMPT = (
    "You are a simple code-search agent with only ONE tool: basic keyword "
    "search over a repository. You do not have symbol navigation, "
    "call-graph, or embedding search -- only this.\n\n"
    "Question: {question}\n"
    "Search results so far ({round_index} round(s) of basic search):\n"
    "{evidence_text}\n\n"
    "Return JSON with key \"enough\" (bool: do you have enough to answer "
    "the question?) and, only when not enough, key \"next_query\" (a "
    "short keyword search query to try next, different from previous "
    "queries)."
)

_TIER2_MAX_ROUNDS = 3


def run_tier2_simple_agent(
    examples: list[EvalExample],
    repo_root: Path,
    index_path: Path,
    out_path: Path,
    judge: str = "openai",
) -> list[dict]:
    """Tier 2: simple single-agent -- only basic search(), no dense_search/
    navigate/callers/references/subclasses, iterative via an observe()-style
    feedback loop deciding whether to search again or answer."""
    provider = OpenAIProvider()
    search_tool = LocalSearchTool(repo_root)
    all_files = _all_files(index_path)
    idf = build_reference_idf(example.answer for example in examples)
    rows: list[dict] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            started = time.time()
            evidence = []
            query = example.question
            for round_index in range(_TIER2_MAX_ROUNDS):
                results = search_tool.search(query, all_files, limit=8)
                evidence.extend(results)
                evidence_text = "\n".join(
                    f"[{item.path}:{item.line_start}-{item.line_end}] {item.quote[:300]}"
                    for item in evidence[-8:]
                )
                decision_result = provider.responses_json(
                    _TIER2_QUERY_PROMPT.format(
                        question=example.question,
                        round_index=round_index + 1,
                        evidence_text=evidence_text or "(no results yet)",
                    ),
                    max_output_tokens=256,
                )
                decision = _loads_json_object(decision_result.text)
                if decision.get("enough") is True:
                    break
                next_query = decision.get("next_query")
                if not isinstance(next_query, str) or not next_query.strip():
                    break
                query = next_query
            prediction = provider.synthesize(question=example.question, evidence=evidence)
            usage = provider.drain_usage()
            score = judge_answer(
                question=example.question,
                prediction=prediction,
                expected=example.answer,
                evidence_count=len(evidence),
                unresolved_need_count=0,
                judge=judge,
                idf=idf,
            )
            row = {
                "id": example.id,
                "question": example.question,
                "prediction": prediction,
                "score": score.model_dump(),
                "run_cost_usd": usage.estimated_cost_usd,
                "judge_cost_usd": score.usage.estimated_cost_usd,
                "elapsed_seconds": round(time.time() - started, 2),
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    (out_path.parent / (out_path.stem + ".json")).write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return rows


def run_tier3_monolithic_agent(
    examples: list[EvalExample],
    repo_root: Path,
    index_path: Path,
    out_path: Path,
    judge: str = "openai",
) -> list[dict]:
    """Tier 3: monolithic tool-using agent -- ANT's full AutonomousWorker
    toolset, but one whole-repo WorkerCard, no routing/Need-Graph/
    coalition/evolution."""
    provider = OpenAIProvider()
    all_files = _all_files(index_path)
    card = WorkerCard(
        id="worker-monolithic",
        territory_id="monolithic",
        name="whole repo",
        root="",
        files=all_files,
    )
    search_tool = LocalSearchTool(repo_root)
    idf = build_reference_idf(example.answer for example in examples)
    rows: list[dict] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            started = time.time()
            worker = AutonomousWorker(repo_root, card, search_tool, reasoner=provider)
            observation = worker.run(example.question, config=WorkerRunConfig(max_tool_calls=30))
            prediction = provider.synthesize(
                question=example.question, evidence=observation.evidence
            )
            usage = provider.drain_usage()
            score = judge_answer(
                question=example.question,
                prediction=prediction,
                expected=example.answer,
                evidence_count=len(observation.evidence),
                unresolved_need_count=len(observation.unresolved_needs),
                judge=judge,
                idf=idf,
            )
            row = {
                "id": example.id,
                "question": example.question,
                "prediction": prediction,
                "score": score.model_dump(),
                "run_cost_usd": usage.estimated_cost_usd,
                "judge_cost_usd": score.usage.estimated_cost_usd,
                "elapsed_seconds": round(time.time() - started, 2),
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    (out_path.parent / (out_path.stem + ".json")).write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return rows
