from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class EvalReport(BaseModel):
    count: int
    exact_match_rate: float
    contains_answer_rate: float
    avg_evidence_count: float
    avg_unresolved_need_count: float
    avg_correctness: float
    avg_completeness: float
    avg_relevance: float
    avg_clarity: float
    avg_reasoning: float


def build_report(results_path: Path, out_path: Path | None = None) -> EvalReport:
    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    count = len(rows)
    report = EvalReport(
        count=count,
        exact_match_rate=_mean([row["score"]["exact_match"] for row in rows]),
        contains_answer_rate=_mean([row["score"]["contains_answer"] for row in rows]),
        avg_evidence_count=_mean([row["score"]["evidence_count"] for row in rows]),
        avg_unresolved_need_count=_mean(
            [row["score"]["unresolved_need_count"] for row in rows]
        ),
        avg_correctness=_mean([row["score"].get("correctness", 0) for row in rows]),
        avg_completeness=_mean([row["score"].get("completeness", 0) for row in rows]),
        avg_relevance=_mean([row["score"].get("relevance", 0) for row in rows]),
        avg_clarity=_mean([row["score"].get("clarity", 0) for row in rows]),
        avg_reasoning=_mean([row["score"].get("reasoning", 0) for row in rows]),
    )
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def _mean(values: list) -> float:
    if not values:
        return 0.0
    numeric = [float(value) for value in values]
    return round(sum(numeric) / len(numeric), 4)
