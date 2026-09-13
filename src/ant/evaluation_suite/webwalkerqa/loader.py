"""WebWalkerQA (ACL 2025, "WebWalker: Benchmarking LLMs in Web Traversal")
dataset loader and normalized record schema.

Official dataset: `datasets.load_dataset("callanwu/WebWalkerQA", split="main")`
(680 human-verified queries -- confirmed by actually loading it, not
assumed).

SCHEMA CORRECTION (verified against the live dataset before writing this
normalizer, not assumed from documentation): the actual schema uses
LOWERCASE top-level keys `question`/`answer`/`root_url`/`info`, and
`info` itself has keys `domain`/`source_website`/`golden_path`/`type`/
`difficulty_level`/`lang` -- not the `Question`/`Answer`/`Root_Url`/
`Info.{Hop,Domain,Language,Difficulty_Level,Source_Website,Golden_Path}`
schema the governing spec described as "typical." `lang` holds an ISO
code (`"en"`/`"zh"`/`""`), not a full language name, and `type` (values
`"single_source"`/`"multi_source"`) is what the spec called "hop type."
Verified value distributions across all 680 rows: `lang` -- zh 375, en
247, empty 58; `domain` -- education 322, conference 156, game 136,
organization 66; `type` -- single_source 340, multi_source 340;
`difficulty_level` -- medium 280, hard 240, easy 160. This module
normalizes from the REAL schema; `WebWalkerQaRecord`'s own field names
below still follow the spec's own naming (`language`, `hop_type`) for
its internal/external contract, populated FROM the real `lang`/`type`
values.

CRITICAL separation (per the governing spec): `source_websites` and
`golden_path` are EVALUATION/AUDIT metadata only -- they identify which
real websites and navigation path the gold answer came from, and handing
them to an inference method would trivially leak the answer path. They
are present on `WebWalkerQaRecord` for audit/manifest purposes ONLY;
`inference_view()` below is the one function any future inference method
should ever be given a record through, and it structurally omits both
fields (and `gold_answer`), the same "gold never reaches inference"
discipline every other benchmark adapter in this suite already follows.

This module does NOT crawl URLs and does NOT call an LLM -- only
`datasets.load_dataset` (a Hugging Face Hub dataset fetch, not a website
crawl) is a real network call here.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from pydantic import BaseModel, Field

HF_PATH = "callanwu/WebWalkerQA"
DEFAULT_SPLIT = "main"


class WebWalkerQaRecord(BaseModel):
    """Normalized internal record for one WebWalkerQA example. `metadata`
    is intentionally omitted (unlike `TaskExample`) -- this is a
    dataset-preparation-stage record, not yet a `TaskExample`; a future
    inference-track adapter is expected to convert this into a
    `TaskExample` at that time.
    """

    example_id: str
    question: str
    gold_answer: str
    root_url: str
    language: str
    domain: str
    hop_type: str
    difficulty: str
    # AUDIT/EVALUATION METADATA ONLY -- see module docstring. Never read by
    # inference_view() or by any future inference method.
    source_websites: list[str] = Field(default_factory=list)
    golden_path: list[str] = Field(default_factory=list)


def _stable_example_id(question: str, root_url: str) -> str:
    """The raw dataset has no natural stable ID field (see the schema in
    the module docstring) -- a content hash of (question, root_url) is
    used instead of the row's own position/index, so an ID stays stable
    even if the underlying dataset is re-shuffled or re-fetched, and two
    genuinely different rows never collide by construction (a hash
    collision here is astronomically less likely than 680 rows sharing an
    identical question+root_url pair, which would itself indicate a
    duplicate row in the source data, not an ID scheme bug).
    """
    digest = hashlib.sha256(f"{question}|{root_url}".encode()).hexdigest()
    return f"webwalkerqa-{digest[:16]}"


def _normalize_row(row: Any) -> WebWalkerQaRecord:
    info = row.get("info") or {}
    question = str(row.get("question", ""))
    root_url = str(row.get("root_url", ""))
    return WebWalkerQaRecord(
        example_id=_stable_example_id(question, root_url),
        question=question,
        gold_answer=str(row.get("answer", "")),
        root_url=root_url,
        language=str(info.get("lang", "")),
        domain=str(info.get("domain", "")),
        hop_type=str(info.get("type", "")),
        difficulty=str(info.get("difficulty_level", "")),
        source_websites=list(info.get("source_website") or []),
        golden_path=list(info.get("golden_path") or []),
    )


def load_webwalkerqa_records(
    split: str = DEFAULT_SPLIT, limit: int | None = None
) -> list[WebWalkerQaRecord]:
    """Loads and normalizes WebWalkerQA rows. No LLM call, no URL crawl --
    only a Hugging Face Hub dataset fetch. `limit`, when given, truncates
    to the first `limit` rows in the dataset's OWN native order (a
    zero-cost, deterministic operation) -- use `sample_deterministic` for
    seeded sampling instead.
    """
    from datasets import load_dataset

    rows = load_dataset(HF_PATH, split=split)
    records = []
    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        records.append(_normalize_row(row))
    return records


def filter_english(records: list[WebWalkerQaRecord]) -> list[WebWalkerQaRecord]:
    # Real data uses ISO-ish codes ("en"/"zh"), not full names -- see
    # module docstring's schema correction. Matches both "en" (the real
    # value) and "english" defensively in case a differently-sourced
    # split ever spells it out in full.
    return [r for r in records if r.language.strip().lower() in {"en", "english"}]


def filter_by_domain(records: list[WebWalkerQaRecord], domain: str) -> list[WebWalkerQaRecord]:
    target = domain.strip().lower()
    return [r for r in records if r.domain.strip().lower() == target]


def filter_by_difficulty(
    records: list[WebWalkerQaRecord], difficulty: str
) -> list[WebWalkerQaRecord]:
    target = difficulty.strip().lower()
    return [r for r in records if r.difficulty.strip().lower() == target]


def filter_by_hop_type(records: list[WebWalkerQaRecord], hop_type: str) -> list[WebWalkerQaRecord]:
    target = hop_type.strip().lower()
    return [r for r in records if r.hop_type.strip().lower() == target]


def sample_deterministic(
    records: list[WebWalkerQaRecord], n: int, seed: int | None = None
) -> list[WebWalkerQaRecord]:
    """`seed=None` (default): deterministic first-N in the given list's own
    order -- no randomness at all. `seed=<int>`: a deterministic seeded
    shuffle (via `random.Random(seed)`, never the global `random` module)
    followed by taking the first N -- same seed always yields the same
    sample, independent of process/run order. Never re-rolled based on
    what the sample contains.
    """
    if n >= len(records):
        return list(records)
    if seed is None:
        return list(records[:n])
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    return shuffled[:n]


def inference_view(record: WebWalkerQaRecord) -> dict:
    """The ONLY view of a record any future inference method should ever
    receive. Structurally omits `gold_answer`, `source_websites`, and
    `golden_path` -- not merely convention, this function's return value
    contains no key through which any of the three could reach a prompt.
    """
    return {
        "example_id": record.example_id,
        "question": record.question,
        "root_url": record.root_url,
        "language": record.language,
        "domain": record.domain,
        "hop_type": record.hop_type,
        "difficulty": record.difficulty,
    }
