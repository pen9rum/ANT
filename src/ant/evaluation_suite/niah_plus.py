"""Needle-in-a-Haystack PLUS: a faithful RECONSTRUCTION, not the exact
released benchmark. See docs/niah_plus_fidelity_audit.md (written before
this module) for the full audit -- the released benchmark's own repository
(zuucan/NeedleInAHaystack-PLUS) contains no generation code, only a README
and two figures, and its one data file is an unverifiable opaque download
with no accompanying construction script. Every construction decision
below is disclosed there as this pass's own reconstruction choice, built
from the paper's own disclosed protocol (arXiv:2402.11550, Section 3.1)
and disclosed source datasets (SQuAD for single-needle, this suite's own
already-audited HotpotQA adapter for multi-needle) -- never presented as
the literal released benchmark's own instances.

Reuses the existing document substrate (`DocumentRecord`,
`materialize_documents`, `EvalDocumentEnvironment`) unchanged: a NIAH+
instance IS a document list like any other benchmark's, so every existing
method (Direct/Retrieval/Matched ReAct/LongAgent/ANT) consumes it through
the exact same `environment_root: Path` interface with zero changes
anywhere outside this module.

No leakage by construction: `NiahPlusExample.documents` are plain
`DocumentRecord`s (doc_id/title/text only); needle doc_ids and depth
percentages live in `NiahPlusExample.metadata`, read only by this module's
own scoring/diagnostics -- never exposed to any inference method, same
discipline as every other benchmark adapter in this suite.
"""

from __future__ import annotations

import random
import re
from typing import Literal

import tiktoken
from pydantic import BaseModel, Field

from ant.evaluation_suite.document_scope import DocumentRecord

_ENCODING_NAME = "cl100k_base"

# Fixed, disclosed seed for every random choice this module makes (which
# SQuAD/HotpotQA rows are sampled as filler, which needle is picked for a
# given cell) -- chosen once, arbitrarily, and never varied based on what
# it produces. See fidelity audit Section 6.
NIAH_PLUS_SEED = 20260912

# A small, fixed pool of clearly-fictional entity names used to replace a
# SQuAD needle's own answer entity (see fidelity audit Section 6/5: the
# paper does this specifically to defeat world-knowledge shortcuts, not to
# obscure the task). Picked deterministically by index, never re-rolled
# based on how a run scores.
_FICTIONAL_ENTITIES = [
    "Zorvath Quennelin",
    "Bregdil Thanewick",
    "Ossenfer Malquorin",
    "Tindrel Voskamp",
    "Halcyneth Drurow",
    "Farnwick Ostrelle",
    "Grendleth Ashvarr",
    "Quillmark Sennador",
    "Wrenthold Kavastri",
    "Milgrave Ondrethin",
]

# Single-Needle Contamination Study Condition C (fully counterfactualized):
# a SEPARATE fictional-name pool from _FICTIONAL_ENTITIES (which is
# reserved for the answer entity) used for OTHER salient entities replaced
# in both question and context -- deliberately distinct so a reader/audit
# can immediately tell "answer substitute" from "other-entity substitute"
# apart in any logged instance. Picked deterministically by index, never
# re-rolled based on output.
_FICTIONAL_OTHER_ENTITIES = [
    "Ashcombe Vale",
    "Kellmoor",
    "Draventhorpe",
    "Northwick Fen",
    "Belmara",
    "Thistledown Reach",
    "Vosmere",
    "Calderfen",
    "Ormsgate",
    "Brindlewood",
]

# How many additional salient entities (beyond the answer itself) Condition
# C replaces -- fixed before any generation, not tuned per instance.
MAX_OTHER_ENTITIES_PER_INSTANCE = 2

# Heuristic (capitalization-pattern-based) proper-noun-phrase matcher --
# NOT true NER. Disclosed limitation (see docs/niah_plus_fidelity_audit.md's
# contamination-study note): may over- or under-identify entities on
# unusual phrasing. Chosen because SQuAD/Wikipedia prose reliably
# capitalizes proper nouns, and adding a real NER dependency for this small
# diagnostic study was judged not worth the added complexity/risk.
_PROPER_PHRASE_RE = re.compile(r"\b[A-Z][a-zA-Z']*(?:\s+[A-Z][a-zA-Z']*)*\b")
_STOPWORD_SINGLE_WORDS = {
    "The",
    "A",
    "An",
    "In",
    "On",
    "At",
    "To",
    "For",
    "Of",
    "And",
    "Who",
    "What",
    "When",
    "Where",
    "Which",
    "How",
    "Why",
    "Is",
    "Was",
    "Were",
    "Are",
    "Did",
    "Does",
    "Do",
    "By",
    "With",
    "From",
}


def _salient_shared_entities(question: str, context: str, exclude: set[str]) -> list[str]:
    """Proper-noun-like phrases appearing in BOTH `question` and `context`,
    longest match first, with shorter overlapping matches dropped (e.g. if
    "Virgin Mary" is selected, a separately-matched "Mary" is dropped
    rather than substituted a second time, which would otherwise corrupt
    the already-substituted phrase). See module-level comment above this
    function for the disclosed heuristic-vs-true-NER limitation.
    """
    candidates: list[str] = []
    seen: set[str] = set()
    for match in _PROPER_PHRASE_RE.finditer(question):
        phrase = match.group(0)
        if phrase in exclude or phrase in seen:
            continue
        words = phrase.split()
        if len(words) == 1 and words[0] in _STOPWORD_SINGLE_WORDS:
            continue
        if phrase in context:
            candidates.append(phrase)
            seen.add(phrase)
    candidates.sort(key=len, reverse=True)
    selected: list[str] = []
    for phrase in candidates:
        if any(phrase in longer for longer in selected):
            continue
        selected.append(phrase)
    return selected


TaskType = Literal["single_needle", "multi_needle"]
Position = Literal["early", "middle", "late"]

# Paper-consistent depth mapping (fidelity audit Section 6) -- fixed
# before any generation or inference, independent of scores.
SINGLE_NEEDLE_DEPTHS: dict[Position, float] = {"early": 0.0, "middle": 0.5, "late": 1.0}
MULTI_NEEDLE_DEPTHS: dict[Position, tuple[float, float]] = {
    "early": (0.0, 0.33),
    "middle": (0.33, 0.66),
    "late": (0.66, 1.0),
}


class NiahPlusExample(BaseModel):
    task_id: str
    task_type: TaskType
    context_length_tokens: int  # TARGET length
    position: Position
    question: str
    gold_answers: list[str]
    documents: list[DocumentRecord]
    # Construction-only diagnostics -- needle_doc_ids/depth_percents/
    # actual_token_count. Never read by any inference method; only by this
    # module's own scoring and by report-generation tooling.
    metadata: dict = Field(default_factory=dict)


def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(_ENCODING_NAME)


def _token_count(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def _insert_needles_by_token_depth(
    filler: list[DocumentRecord],
    needles: list[tuple[float, DocumentRecord]],
) -> list[DocumentRecord]:
    """Inserts each (depth_fraction, needle_document) into `filler` at the
    list position whose cumulative token count (over `filler` alone, BEFORE
    any insertion) is closest to `depth_fraction * total_filler_tokens` --
    i.e. depth is measured in TOKEN space, matching the paper's own
    percentage-of-haystack framing, not mere list-index position. Needles
    are inserted in depth order so an earlier needle's own insertion does
    not shift a later needle's target index.
    """
    encoding = _encoding()
    filler_tokens = [len(encoding.encode(doc.text, disallowed_special=())) for doc in filler]
    total_filler_tokens = sum(filler_tokens)
    cumulative = [0]
    for count in filler_tokens:
        cumulative.append(cumulative[-1] + count)

    result = list(filler)
    # Needles are processed lowest-depth first, and each one's target
    # index is computed against the ORIGINAL filler-only cumulative counts
    # (matching the paper's "position within the haystack", i.e. the
    # filler alone -- not perturbed by any other needle's own length).
    # `inserted_so_far` corrects the actual list-insertion index for the
    # fact that every earlier (lower-depth) needle has already grown
    # `result` by one position at or before this one's own target --
    # sorted-ascending processing guarantees that offset is always exactly
    # `inserted_so_far`, never more or less.
    inserted_so_far = 0
    for depth, needle in sorted(needles, key=lambda item: item[0]):
        target_tokens = depth * total_filler_tokens
        # Find the filler index whose cumulative token count (up to and
        # including it) is closest to target_tokens.
        original_index = min(
            range(len(cumulative)), key=lambda i: abs(cumulative[i] - target_tokens)
        )
        result.insert(original_index + inserted_so_far, needle)
        inserted_so_far += 1

    return result


def _fill_to_target_length(
    pool: list[DocumentRecord], target_tokens: int, rng: random.Random
) -> list[DocumentRecord]:
    """Deterministically samples (with replacement, reshuffled each pass)
    from `pool` until the cumulative token count reaches at least
    `target_tokens`. Whole documents only -- no mid-document truncation --
    preserving this suite's "no summarization, faithful document text"
    principle even for synthetic filler; the resulting total is therefore
    always >= target_tokens, within one document's length of it (checked
    against a documented tolerance by this module's own tests).
    """
    filler: list[DocumentRecord] = []
    total = 0
    order = list(pool)
    rng.shuffle(order)
    cursor = 0
    while total < target_tokens:
        if cursor >= len(order):
            rng.shuffle(order)
            cursor = 0
        doc = order[cursor]
        cursor += 1
        filler.append(doc)
        total += _token_count(doc.text)
    return filler


def _load_squad_pool(limit: int = 400) -> list[dict]:
    from datasets import load_dataset

    rows = load_dataset("rajpurkar/squad", split="train")
    pool = []
    for i, row in enumerate(rows):
        if i >= limit:
            break
        pool.append(row)
    return pool


def build_single_needle_instance(
    *, context_length_tokens: int, position: Position, question_index: int = 0
) -> NiahPlusExample:
    """One single-document QA instance: a SQuAD question whose own context
    passage (with its answer entity replaced by a fictional one) is the
    needle, inserted at the depth SINGLE_NEEDLE_DEPTHS[position] within a
    haystack of OTHER SQuAD training-set passages -- see fidelity audit
    Section 6 for why SQuAD is the filler source too (paper-disclosed).
    """
    # Seeded WITHOUT `position`: Section 15 of the governing spec requires
    # the same underlying question/needle/distractor material across
    # EARLY/MIDDLE/LATE for a given (task_type, context_length,
    # question_index) -- only the needle's insertion depth may vary. The
    # filler pool's own sampled order therefore must not depend on
    # `position` at all; only `_insert_needles_by_token_depth`'s `depth`
    # argument below does.
    # random.Random() only accepts None/int/float/str/bytes/bytearray --
    # str(...) of the tuple is used instead of the tuple itself, and
    # (per Python's own seed() implementation) a str seed is hashed via
    # sha512 deterministically regardless of PYTHONHASHSEED, so this
    # remains fully reproducible across processes/runs.
    rng = random.Random(str((NIAH_PLUS_SEED, "single", question_index)))
    pool = _load_squad_pool()

    needle_row = None
    for row in pool[question_index:] + pool[:question_index]:
        if row["answers"]["text"] and row["answers"]["text"][0] in row["context"]:
            needle_row = row
            break
    if needle_row is None:
        msg = "no SQuAD row in the sampled pool has its answer verbatim in its own context"
        raise ValueError(msg)

    original_answer = needle_row["answers"]["text"][0]
    fictional_entity = _FICTIONAL_ENTITIES[question_index % len(_FICTIONAL_ENTITIES)]
    needle_text = needle_row["context"].replace(original_answer, fictional_entity)
    needle_doc = DocumentRecord(doc_id="needle0", title=needle_row["title"], text=needle_text)

    filler_candidates = [
        DocumentRecord(doc_id=f"filler{i}", title=row["title"], text=row["context"])
        for i, row in enumerate(pool)
        if row["id"] != needle_row["id"]
    ]
    needle_tokens = _token_count(needle_text)
    filler = _fill_to_target_length(
        filler_candidates, max(0, context_length_tokens - needle_tokens), rng
    )
    depth = SINGLE_NEEDLE_DEPTHS[position]
    documents = _insert_needles_by_token_depth(filler, [(depth, needle_doc)])
    documents = [d.model_copy(update={"doc_id": f"doc{i}"}) for i, d in enumerate(documents)]
    needle_final_id = next(d.doc_id for d in documents if d.text == needle_text)

    actual_tokens = sum(_token_count(d.text) for d in documents)
    return NiahPlusExample(
        task_id=f"niah_single_{context_length_tokens}_{position}_{question_index}",
        task_type="single_needle",
        context_length_tokens=context_length_tokens,
        position=position,
        question=needle_row["question"],
        gold_answers=[fictional_entity],
        documents=documents,
        metadata={
            "needle_doc_ids": [needle_final_id],
            "depth_percents": [depth],
            "actual_token_count": actual_tokens,
            "source_dataset": "squad",
            "original_answer_replaced": original_answer,
        },
    )


def build_fully_counterfactualized_single_needle_instance(
    *, context_length_tokens: int, position: Position, question_index: int = 0
) -> NiahPlusExample:
    """Single-Needle Contamination Study Condition C: replaces not just the
    answer entity but up to MAX_OTHER_ENTITIES_PER_INSTANCE additional
    salient entities shared between the question and its own context
    (e.g. the central subject, a location) with deterministic fictional
    substitutes, applied identically to BOTH question and context so they
    remain mutually consistent -- so the memorized SQuAD question->answer
    mapping no longer directly applies, without changing the underlying
    semantic relation/QA structure or reasoning complexity, and without
    ever stating the original answer or an explicit negative instruction
    anywhere in the constructed instance (the substitution is invisible at
    inference time -- only this function's own returned `metadata` records
    what was replaced, for diagnostics only).
    """
    # Independent seed namespace ("single_c") from Condition A/B's "single"
    # -- guarantees this condition's own filler sampling never accidentally
    # collides with build_single_needle_instance's, even for the same
    # question_index.
    rng = random.Random(str((NIAH_PLUS_SEED, "single_c", question_index)))
    pool = _load_squad_pool()

    needle_row = None
    for row in pool[question_index:] + pool[:question_index]:
        if row["answers"]["text"] and row["answers"]["text"][0] in row["context"]:
            needle_row = row
            break
    if needle_row is None:
        msg = "no SQuAD row in the sampled pool has its answer verbatim in its own context"
        raise ValueError(msg)

    original_answer = needle_row["answers"]["text"][0]
    fictional_answer = _FICTIONAL_ENTITIES[question_index % len(_FICTIONAL_ENTITIES)]

    other_entities = _salient_shared_entities(
        needle_row["question"], needle_row["context"], exclude={original_answer}
    )[:MAX_OTHER_ENTITIES_PER_INSTANCE]
    substitutions: dict[str, str] = {original_answer: fictional_answer}
    offset = question_index * MAX_OTHER_ENTITIES_PER_INSTANCE
    for i, entity in enumerate(other_entities):
        substitutions[entity] = _FICTIONAL_OTHER_ENTITIES[
            (offset + i) % len(_FICTIONAL_OTHER_ENTITIES)
        ]

    # Longest original strings substituted first so a shorter substitution
    # target that happens to be a substring of a longer one (unlikely given
    # these are real distinct entities, but not assumed) never partially
    # corrupts the longer one before it's replaced.
    counterfactual_question = needle_row["question"]
    counterfactual_context = needle_row["context"]
    for original in sorted(substitutions, key=len, reverse=True):
        fictional = substitutions[original]
        counterfactual_question = counterfactual_question.replace(original, fictional)
        counterfactual_context = counterfactual_context.replace(original, fictional)

    needle_doc = DocumentRecord(
        doc_id="needle0", title=needle_row["title"], text=counterfactual_context
    )
    filler_candidates = [
        DocumentRecord(doc_id=f"filler{i}", title=row["title"], text=row["context"])
        for i, row in enumerate(pool)
        if row["id"] != needle_row["id"]
    ]
    needle_tokens = _token_count(counterfactual_context)
    filler = _fill_to_target_length(
        filler_candidates, max(0, context_length_tokens - needle_tokens), rng
    )
    depth = SINGLE_NEEDLE_DEPTHS[position]
    documents = _insert_needles_by_token_depth(filler, [(depth, needle_doc)])
    documents = [d.model_copy(update={"doc_id": f"doc{i}"}) for i, d in enumerate(documents)]
    needle_final_id = next(d.doc_id for d in documents if d.text == counterfactual_context)

    actual_tokens = sum(_token_count(d.text) for d in documents)
    return NiahPlusExample(
        task_id=f"niah_single_cf_{context_length_tokens}_{position}_{question_index}",
        task_type="single_needle",
        context_length_tokens=context_length_tokens,
        position=position,
        question=counterfactual_question,
        gold_answers=[fictional_answer],
        documents=documents,
        metadata={
            "needle_doc_ids": [needle_final_id],
            "depth_percents": [depth],
            "actual_token_count": actual_tokens,
            "source_dataset": "squad",
            "original_answer_replaced": original_answer,
            "entity_substitutions": substitutions,
            "condition": "C_fully_counterfactualized",
        },
    )


def build_multi_needle_instance(
    *, context_length_tokens: int, position: Position, question_index: int = 0
) -> NiahPlusExample:
    """One multi-document QA instance: a HotpotQA validation question's own
    2 supporting documents are the needles, inserted at
    MULTI_NEEDLE_DEPTHS[position] within a haystack built from OTHER,
    unrelated HotpotQA validation questions' own documents (this pass's
    own filler-source decision -- see fidelity audit Section 5/6, the
    paper does not specify multi-document filler composition).
    """
    from ant.benchmarks.hotpotqa import HotpotQaAdapter

    # Seeded WITHOUT `position` -- see build_single_needle_instance's own
    # comment for why (Section 15: same material across EARLY/MIDDLE/LATE).
    # str(...) -- see that function's own comment on random.Random's
    # accepted seed types.
    rng = random.Random(str((NIAH_PLUS_SEED, "multi", question_index)))
    adapter = HotpotQaAdapter()
    examples = adapter.load_examples(limit=max(20, question_index + 5))
    target_example = examples[question_index]
    all_docs = [DocumentRecord(**d) for d in target_example.metadata["documents"]]
    supporting_ids = set(target_example.metadata["supporting_doc_ids"])
    needle_docs = [d for d in all_docs if d.doc_id in supporting_ids][:2]
    if len(needle_docs) != 2:
        msg = f"expected exactly 2 supporting documents, found {len(needle_docs)}"
        raise ValueError(msg)

    filler_candidates: list[DocumentRecord] = []
    for other_index, other_example in enumerate(examples):
        if other_index == question_index:
            continue
        for d in other_example.metadata["documents"]:
            filler_candidates.append(
                DocumentRecord(
                    doc_id=f"filler{other_index}_{d['doc_id']}", title=d["title"], text=d["text"]
                )
            )
    needle_tokens = sum(_token_count(d.text) for d in needle_docs)
    filler = _fill_to_target_length(
        filler_candidates, max(0, context_length_tokens - needle_tokens), rng
    )
    depth1, depth2 = MULTI_NEEDLE_DEPTHS[position]
    documents = _insert_needles_by_token_depth(
        filler, [(depth1, needle_docs[0]), (depth2, needle_docs[1])]
    )
    needle_texts = {d.text for d in needle_docs}
    documents = [d.model_copy(update={"doc_id": f"doc{i}"}) for i, d in enumerate(documents)]
    needle_final_ids = [d.doc_id for d in documents if d.text in needle_texts]

    actual_tokens = sum(_token_count(d.text) for d in documents)
    return NiahPlusExample(
        task_id=f"niah_multi_{context_length_tokens}_{position}_{question_index}",
        task_type="multi_needle",
        context_length_tokens=context_length_tokens,
        position=position,
        question=target_example.question,
        gold_answers=_ground_truths_from_reference(target_example.reference),
        documents=documents,
        metadata={
            "needle_doc_ids": needle_final_ids,
            "depth_percents": [depth1, depth2],
            "actual_token_count": actual_tokens,
            "source_dataset": "hotpotqa",
            "source_task_id": target_example.task_id,
        },
    )


def _ground_truths_from_reference(reference: str) -> list[str]:
    import json

    return json.loads(reference)
