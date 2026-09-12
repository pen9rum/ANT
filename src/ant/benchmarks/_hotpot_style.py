"""Shared loading/scoring logic for the two benchmarks that ship the
IDENTICAL context/supporting_facts schema -- HotpotQA (distractor config)
and 2WikiMultihopQA -- confirmed field-for-field identical by loading real
rows from both live (see docs/long_context_dataset_audit.md). Not a public
adapter itself; `hotpotqa.py` and `twowikimultihopqa.py` each import this
and register their own distinctly-named BenchmarkAdapter.
"""
from __future__ import annotations

import json
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.document_scope import DocumentRecord, materialize_documents
from ant.evaluation_suite.qa_metrics import score_qa
from ant.evaluation_suite.scoring import MetricResult

DOCUMENT_ENV_ROOT = Path("document-envs")


def load_hotpot_style_examples(
    *, benchmark_name: str, hf_path: str, hf_config: str, split: str, limit: int | None
) -> list[TaskExample]:
    from datasets import load_dataset

    rows = load_dataset(hf_path, hf_config, split=split)
    examples: list[TaskExample] = []
    for row in rows:
        titles = row["context"]["title"]
        sentence_lists = row["context"]["sentences"]
        documents = [
            {"doc_id": f"doc{i}", "title": title, "text": "".join(sentences)}
            for i, (title, sentences) in enumerate(zip(titles, sentence_lists, strict=True))
        ]
        title_to_doc_id = {d["title"]: d["doc_id"] for d in documents}
        supporting_titles = row["supporting_facts"]["title"]
        # Construction/diagnostic-only: which documents are gold-relevant.
        # NEVER read by any agent's run() -- only by score() diagnostics
        # and the (not-run-this-pass) Lost-in-the-Middle perturbation
        # utility. See document_scope.py's own module docstring.
        supporting_doc_ids = sorted(
            {title_to_doc_id[t] for t in supporting_titles if t in title_to_doc_id}
        )
        examples.append(
            TaskExample(
                benchmark=benchmark_name,
                task_id=str(row["id"]),
                question=row["question"],
                # Gold answer, JSON-encoded as a list of ground truths for a
                # uniform score() code path across all 3 QA benchmarks (only
                # MuSiQue has real aliases; here it's always a 1-element
                # list). Agents never read `reference` -- same convention
                # every other benchmark adapter in this suite already uses.
                reference=json.dumps([row["answer"]]),
                metadata={
                    "documents": documents,
                    "num_documents": len(documents),
                    "supporting_doc_ids": supporting_doc_ids,
                    "question_type": row.get("type"),
                    "level": row.get("level"),
                },
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


def prepare_hotpot_style_environment(example: TaskExample, *, benchmark_name: str) -> Path:
    root = DOCUMENT_ENV_ROOT / benchmark_name / example.task_id
    marker = root / ".materialized"
    if marker.exists():
        return root.resolve()
    documents = [DocumentRecord(**d) for d in example.metadata["documents"]]
    materialize_documents(documents, root)
    marker.write_text("ok", encoding="utf-8")
    return root.resolve()


def score_hotpot_style(example: TaskExample, result: AgentResult, *, benchmark_name: str) -> MetricResult:
    ground_truths = json.loads(example.reference)
    metrics = score_qa(result.final_answer, ground_truths)
    native_score = metrics["f1"]  # F1 is the primary reported metric for these benchmarks
    return MetricResult(
        benchmark=benchmark_name,
        task_id=example.task_id,
        native_score=native_score,
        normalized_score=native_score * 100.0,
        submetrics=metrics,
        grader_runs=[],
        metadata={
            "generation_model": result.metadata.get("generation_model", "unknown"),
            "prediction": result.final_answer,
            "ground_truths": ground_truths,
            "scoring_method": "official_em_f1",
        },
    )
