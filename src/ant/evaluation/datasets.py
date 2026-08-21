from __future__ import annotations

import json
from hashlib import sha1
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EvalExample(BaseModel):
    id: str
    question: str
    answer: str = ""
    repo: str = "."
    metadata: dict[str, Any] = Field(default_factory=dict)


def load_examples(path: str, split: str = "test", limit: int | None = None) -> list[EvalExample]:
    if path.startswith("hf://"):
        dataset_name = _dataset_alias(path.removeprefix("hf://"))
        return _load_huggingface_examples(dataset_name, split=split, limit=limit)
    return _load_jsonl_examples(Path(path), limit=limit)


def _load_jsonl_examples(path: Path, limit: int | None) -> list[EvalExample]:
    examples: list[EvalExample] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        examples.append(_coerce_example(json.loads(line)))
        if limit is not None and len(examples) >= limit:
            break
    return examples


def _load_huggingface_examples(
    dataset_name: str,
    split: str,
    limit: int | None,
) -> list[EvalExample]:
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        msg = "Install the optional 'datasets' package to load Hugging Face benchmarks."
        raise RuntimeError(msg) from exc

    rows = load_dataset(dataset_name, split=split, streaming=True)
    examples: list[EvalExample] = []
    for row in rows:
        examples.append(_coerce_example(dict(row)))
        if limit is not None and len(examples) >= limit:
            break
    return examples


def _coerce_example(row: dict[str, Any]) -> EvalExample:
    question = (
        row.get("question")
        or row.get("problem_statement")
        or row.get("query")
        or row.get("prompt")
        or ""
    )
    answer = (
        row.get("answer")
        or row.get("reference_answer")
        or row.get("gold")
        or row.get("expected_answer")
        or row.get("ground_truth")
        or ""
    )
    example_id = row.get("id") or row.get("instance_id") or row.get("qid") or _stable_id(question)
    repo = row.get("repo") or row.get("repository") or row.get("repo_name") or "."
    return EvalExample(
        id=str(example_id),
        question=str(question),
        answer=str(answer),
        repo=str(repo),
        metadata=row,
    )


def _dataset_alias(name: str) -> str:
    aliases = {
        "swe-qa-pro": "TIGER-Lab/SWE-QA-Pro-Bench",
        "SWE-QA-Pro": "TIGER-Lab/SWE-QA-Pro-Bench",
        "swe-qa": "swe-qa/SWE-QA-Benchmark",
    }
    return aliases.get(name, name)


def _stable_id(text: str) -> str:
    return sha1(text.encode("utf-8")).hexdigest()[:16]
