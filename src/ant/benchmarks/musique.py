"""MuSiQue (default config, validation split, the "answerable" variant --
`answerable=True` on every row observed). See
docs/long_context_dataset_audit.md for the full source audit. Distinct
schema from HotpotQA/2WikiMultihopQA: documents come as a flat
`paragraphs` list (`{idx, title, paragraph_text, is_supporting}`), and gold
answers may have `answer_aliases` -- both handled here, not shared with
`_hotpot_style.py`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from ant.agents.base import AgentResult
from ant.benchmarks._hotpot_style import DOCUMENT_ENV_ROOT
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.document_scope import DocumentRecord, materialize_documents
from ant.evaluation_suite.qa_metrics import score_qa
from ant.evaluation_suite.registry import register_benchmark
from ant.evaluation_suite.scoring import MetricResult

HF_PATH = "dgslibisey/MuSiQue"
HF_CONFIG = "default"
HF_SPLIT = "validation"


class MuSiQueAdapter:
    name = "musique"

    def load_examples(
        self, limit: int | None = None, repo_filter: str | None = None
    ) -> list[TaskExample]:
        from datasets import load_dataset

        # cast(Any, ...): see _hotpot_style.py's own identical comment --
        # same overloaded-return-type static-analysis limitation, same
        # actual runtime shape (a dict-row-yielding Dataset).
        rows = cast(Any, load_dataset(HF_PATH, HF_CONFIG, split=HF_SPLIT))
        examples: list[TaskExample] = []
        for row in rows:
            documents = [
                {"doc_id": f"doc{p['idx']}", "title": p["title"], "text": p["paragraph_text"]}
                for p in row["paragraphs"]
            ]
            supporting_doc_ids = sorted(
                f"doc{p['idx']}" for p in row["paragraphs"] if p["is_supporting"]
            )
            ground_truths = [row["answer"], *row["answer_aliases"]]
            examples.append(
                TaskExample(
                    benchmark=self.name,
                    task_id=str(row["id"]),
                    question=row["question"],
                    reference=json.dumps(ground_truths),
                    metadata={
                        "documents": documents,
                        "num_documents": len(documents),
                        "supporting_doc_ids": supporting_doc_ids,
                        "answerable": row.get("answerable"),
                    },
                )
            )
            if limit is not None and len(examples) >= limit:
                break
        return examples

    def prepare_environment(self, example: TaskExample) -> Path:
        root = DOCUMENT_ENV_ROOT / self.name / example.task_id
        marker = root / ".materialized"
        if marker.exists():
            return root.resolve()
        documents = [DocumentRecord(**d) for d in example.metadata["documents"]]
        materialize_documents(documents, root)
        marker.write_text("ok", encoding="utf-8")
        return root.resolve()

    def score(self, example: TaskExample, result: AgentResult) -> MetricResult:
        ground_truths = json.loads(example.reference)
        metrics = score_qa(result.final_answer, ground_truths)
        native_score = metrics["f1"]
        return MetricResult(
            benchmark=self.name,
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


_ADAPTER = MuSiQueAdapter()
register_benchmark(_ADAPTER)
