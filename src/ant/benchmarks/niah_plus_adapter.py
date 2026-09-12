"""BenchmarkAdapter wrapper around the NIAH+ reconstruction
(ant.evaluation_suite.niah_plus) -- makes each generated instance consumable
by every existing document-track method (Direct/Retrieval/Matched ReAct/
LongAgent/ANT) through the exact same `TaskExample`/`environment_root:
Path` interface every other benchmark in this suite already uses, with
ZERO changes to any of those five agent classes.

Generates exactly the 12 frozen conditions this pass's manifest specifies
(2 task types x 2 context lengths x 3 positions), each with a single fixed
`question_index=0` -- deterministic, gold-independent, chosen before any
inference. See docs/niah_plus_fidelity_audit.md for the full construction
audit and docs/long_context_dataset_audit.md for the sibling audit of the
original 3-benchmark track.

No leakage by construction: `TaskExample.metadata["documents"]` is a list
of plain `DocumentRecord` dicts (doc_id/title/text only); needle_doc_ids/
depth_percents/actual_token_count live under
`TaskExample.metadata["niah_metadata"]`, read only by this adapter's own
`score()`/diagnostics -- never by any agent's `run()`, the same discipline
`supporting_doc_ids` already follows elsewhere in this suite.
"""
from __future__ import annotations

import json
from pathlib import Path

from ant.agents.base import AgentResult
from ant.benchmarks._hotpot_style import DOCUMENT_ENV_ROOT
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.document_scope import DocumentRecord, materialize_documents
from ant.evaluation_suite.niah_plus import (
    NiahPlusExample,
    Position,
    TaskType,
    build_multi_needle_instance,
    build_single_needle_instance,
)
from ant.evaluation_suite.qa_metrics import score_qa
from ant.evaluation_suite.registry import register_benchmark
from ant.evaluation_suite.scoring import MetricResult

# The frozen 12-condition smoke grid (Section 14 of the governing spec):
# 2 task types x 2 context lengths x 3 positions, ONE instance each
# (question_index=0, fixed before any generation/inference -- never
# selected based on gold answers or scores). "Do NOT increase sample
# count yet."
CONTEXT_LENGTHS = (32_000, 128_000)
POSITIONS: tuple[Position, ...] = ("early", "middle", "late")
TASK_TYPES: tuple[TaskType, ...] = ("single_needle", "multi_needle")


def _to_task_example(instance: NiahPlusExample) -> TaskExample:
    return TaskExample(
        benchmark="niah_plus",
        task_id=instance.task_id,
        question=instance.question,
        reference=json.dumps(instance.gold_answers),
        metadata={
            "documents": [d.model_dump() for d in instance.documents],
            "num_documents": len(instance.documents),
            "task_type": instance.task_type,
            "context_length_tokens": instance.context_length_tokens,
            "position": instance.position,
            # Construction-only diagnostics (needle_doc_ids, depth_percents,
            # actual_token_count, source dataset) -- never read by any
            # agent's run().
            "niah_metadata": instance.metadata,
        },
    )


class NiahPlusAdapter:
    name = "niah_plus"

    def load_examples(
        self, limit: int | None = None, repo_filter: str | None = None
    ) -> list[TaskExample]:
        del repo_filter
        examples: list[TaskExample] = []
        for task_type in TASK_TYPES:
            builder = (
                build_single_needle_instance
                if task_type == "single_needle"
                else build_multi_needle_instance
            )
            for length in CONTEXT_LENGTHS:
                for position in POSITIONS:
                    instance = builder(
                        context_length_tokens=length, position=position, question_index=0
                    )
                    examples.append(_to_task_example(instance))
                    if limit is not None and len(examples) >= limit:
                        return examples
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
                "scoring_method": "official_em_f1_reconstruction",
                "task_type": example.metadata["task_type"],
                "context_length_tokens": example.metadata["context_length_tokens"],
                "position": example.metadata["position"],
            },
        )


_ADAPTER = NiahPlusAdapter()
register_benchmark(_ADAPTER)
