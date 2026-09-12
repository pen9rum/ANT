"""Official exact-match / F1 scoring for extractive multi-hop QA
(MuSiQue, HotpotQA, 2WikiMultihopQA). Byte-faithful port of the
normalization these three benchmarks' own official evaluation scripts all
use identically -- verified live against all three (not assumed):
hotpotqa/hotpot/hotpot_evaluate_v1.py, StonyBrookNLP/musique/metrics/
answer.py, and Alab-NII/2wikimultihop/2wikimultihop_evaluate_v1.1.py all
define the exact same `normalize_answer` (lowercase -> remove punctuation
-> remove articles a/an/the -> collapse whitespace) before computing EM/F1.
See docs/long_context_dataset_audit.md's own "Official scoring metric"
section for the audit trail. No GPT-5 judge is used for these tasks.
"""
from __future__ import annotations

import re
import string
from collections import Counter

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return _ARTICLES_RE.sub(" ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    def lower(s: str) -> str:
        return s.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def compute_exact(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        # Official convention (matches all three benchmarks' scripts):
        # if either is empty, EM-style equality is the only signal.
        return 1.0 if pred_tokens == gold_tokens else 0.0
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def metric_max_over_ground_truths(
    metric_fn, prediction: str, ground_truths: list[str]
) -> float:
    """Byte-faithful port of MuSiQue's own metrics/answer.py helper --
    max score over every ground truth (the primary answer plus any
    answer_aliases, for MuSiQue; just [answer] for HotpotQA/2Wiki, which
    have no aliases field)."""
    if not ground_truths:
        return 0.0
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def score_qa(prediction: str, ground_truths: list[str]) -> dict[str, float]:
    """Single entry point every document benchmark adapter's own `score()`
    calls: returns {"exact_match": ..., "f1": ...} in [0, 1], each the max
    over all provided ground truths."""
    em = metric_max_over_ground_truths(compute_exact, prediction, ground_truths)
    f1 = metric_max_over_ground_truths(compute_f1, prediction, ground_truths)
    return {"exact_match": em, "f1": f1}
