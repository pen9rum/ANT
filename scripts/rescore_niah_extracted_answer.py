"""Apply the shared, frozen answer-extraction layer to already-saved niah
QA-style results, adding extracted_answer/extracted_em/extracted_f1
alongside the existing raw em/f1 -- scoring-side only, no agent rerun.

Method-agnostic: reads only task_id/question/prediction/gold_answers from
an existing results.jsonl (run_suite / run_adaptive_ablations.py output
shape) and writes a sibling *.rescored.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ant.evaluation_suite.answer_extraction import extract_answer_span  # noqa: E402
from ant.evaluation_suite.qa_metrics import score_qa  # noqa: E402


def _question_for(task_id: str, manifest_conditions: dict[str, str]) -> str:
    question = manifest_conditions.get(task_id)
    if question is None:
        raise KeyError(f"No question text found for task_id={task_id!r} in manifest")
    return question


def rescore_file(in_path: Path, out_path: Path, manifest_conditions: dict[str, str]) -> None:
    total_cost = 0.0
    total_calls = 0
    with in_path.open(encoding="utf-8") as fh, out_path.open("w", encoding="utf-8") as out:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            task_id = row.get("task_id")
            prediction = row.get("final_answer") or row.get("metric", {}).get(
                "metadata", {}
            ).get("prediction", "")
            gold = row.get("metric", {}).get("metadata", {}).get("ground_truths")
            if task_id is None or gold is None or row.get("status") != "completed":
                out.write(json.dumps(row) + "\n")
                continue
            question = _question_for(task_id, manifest_conditions)
            extraction = extract_answer_span(question, prediction)
            extracted_metrics = score_qa(extraction.extracted_answer, gold)
            row["metric"]["extracted_answer"] = extraction.extracted_answer
            row["metric"]["extracted_em"] = extracted_metrics["exact_match"]
            row["metric"]["extracted_f1"] = extracted_metrics["f1"]
            row["metric"]["extraction_used_llm"] = extraction.used_llm
            row["metric"]["extraction_cost_usd"] = extraction.estimated_cost_usd
            total_cost += extraction.estimated_cost_usd
            total_calls += extraction.llm_calls
            out.write(json.dumps(row) + "\n")
    print(f"{in_path} -> {out_path} (extraction cost=${total_cost:.4f}, llm_calls={total_calls})")


def _load_manifest_questions(manifest_path: Path) -> dict[str, str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {c["task_id"]: c["question"] for c in payload["conditions"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_jsonl", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT
        / "third_party"
        / "manifests"
        / "long_context"
        / "multineedle_scaling_manifest_512k.json",
    )
    args = parser.parse_args()
    manifest_conditions = _load_manifest_questions(args.manifest)
    out_path = args.results_jsonl.with_suffix(".rescored.jsonl")
    rescore_file(args.results_jsonl, out_path, manifest_conditions)


if __name__ == "__main__":
    main()
