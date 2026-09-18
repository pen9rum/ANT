"""S2G-RAG ("Structured Sufficiency and Gap Judging RAG"): paper-faithful
adaptation with ONE disclosed, deliberate substitution -- GPT-4.1 plays the
S2G-Judge role by prompting, in place of the paper's LoRA-fine-tuned
Llama-3.2-3B-Instruct judge, for which no trained checkpoint is released.

Paper: Li, Zou, Lv, Zhang, Zhou, ACL 2026 Long Papers.
Official-looking repo: https://github.com/nianaaa/S2G-RAG (provenance is
self-asserted in that README; the algorithm details below are directly
verifiable from its source and were read before writing this file --
`inference/inference_bm25.py`, `utils/prompt_template.py`,
`utils/text_processing.py`, `run_S2G-RAG.py`).

WHAT IS REPRODUCED VERBATIM FROM UPSTREAM
-----------------------------------------
- `SUFF_SYSTEM_PROMPT` (the judge's structured sufficiency/gap schema:
  `{"sufficient": bool, "gap_items": [{category, target, slot,
  description}]}`), `SELECTOR_SYSTEM_PROMPT` (the sentence-level Evidence
  Extractor) and `force_answer_prompt` -- character-for-character from
  `utils/prompt_template.py`.
- `build_suff_user_prompt`, `build_query_from_missing`,
  `should_force_first_retrieval`, `format_missing_facts_for_selector`,
  `merge_evidence_only`, `append_evidence_context`,
  `update_task_with_evidence`, `split_wiki_sentences`, `safe_json_load`,
  `concat_raw_retrieved_docs` -- ported line-for-line from
  `inference/inference_bm25.py`.
- `extract_gap_items`, `extract_evidence_global_ids`, `get_json_value`,
  `extract_final_answer_and_rationale` -- ported from
  `utils/text_processing.py`.
- The turn loop's own control flow: `while turn <= max_turns`, the
  first-turn forced-retrieval override, `done` on (verdict OR
  turn == max_turns), retrieval only while `turn < max_turns`, and the
  append-only (never overwriting) Evidence Context.
- Upstream CLI defaults, read from `inference/inference_bm25.py`'s own
  `parse_args()`: `--max_turns 4`, `--top_docs 6`, and the hardcoded
  `return_top_k=6` / `max_sents_per_doc=40` at the `main_batch` call site.
  The gap-phrase cap is upstream's own non-TriviaQA default of 3
  (`build_query_from_missing`'s `max_facts = 1 if dataset == "triviaqa"
  else 3`) -- HotpotQA/2Wiki/MuSiQue all take the 3 branch.

DISCLOSED DEVIATIONS (see this module's DEVIATIONS constant, which is also
emitted into every AgentResult's metadata)
------------------------------------------
1. S2G-Judge model substitution -- the headline, pre-approved one.
2. Retrieval backend: upstream ranks over a whole-Wikipedia Pyserini BM25
   index (or e5-base-v2 + FAISS). This suite's fairness constraint
   requires every method to see the SAME per-question candidate document
   collection, so retrieval here runs `ant.tools.local.LocalSearchTool`
   -- the identical BM25+RRF primitive `retrieval_document.py` (Sparse
   Retrieval) already uses -- over the identical materialized
   `DocumentRecord` set. Same family of ranker (BM25), same corpus every
   other method sees, no hidden/oracle index.
3. Batching: upstream batches 8 questions through one HF `generate()`
   call purely for local-GPU throughput. This suite's `AgentAdapter`
   contract is one example per `run()`, and batching is not part of the
   algorithm (it changes no decision), so it is dropped.
4. Sentence segmentation: upstream prefers `pysbd` and already ships its
   own regex fallback in a `try/except ImportError`. That exact
   try/except is ported; `pysbd` is not installed in this environment, so
   the (upstream's own) regex fallback is what runs.

NO GOLD INFORMATION IS READ. `run()` touches only `example.question` and
`example.metadata["documents"]`. `example.reference`,
`metadata["supporting_doc_ids"]` and every other gold/diagnostic field are
never read on the inference path -- upstream's own `_load_gold_answers` /
`_load_gold_documents` / `Correct Retrieval` bookkeeping is deliberately
NOT ported, because in this suite scoring is the benchmark adapter's job
(`score_hotpot_style` / `MuSiQueAdapter.score`), not the method's.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ant.agents.base import AgentResult
from ant.benchmarks.base import TaskExample
from ant.evaluation_suite.counting_provider import CountingOpenAIProvider
from ant.evaluation_suite.document_scope import DocumentRecord, EvalDocumentEnvironment
from ant.evaluation_suite.usage import UsageStats
from ant.tools.local import LocalSearchTool

PAPER_CITATION = (
    'Li, Zou, Lv, Zhang, Zhou. "S2G-RAG: Structured Sufficiency and Gap Judging for '
    "Retrieval-Augmented Multi-Hop Question Answering.\" ACL 2026 Long Papers. "
    "arXiv:2604.23783."
)
IMPLEMENTATION_LABEL = "paper-faithful S2G-RAG adaptation, GPT-4.1-substituted S2G-Judge"
UPSTREAM_COMMIT = "5d842a67a0a99a7b545bbad0dc402ceaae0e5eff"
UPSTREAM_REPO_NOTE = (
    "https://github.com/nianaaa/S2G-RAG, pinned at the commit above (2026-07-25, the latest "
    "at audit time; the repository ships no LICENSE file). Prompts and control flow ported "
    "from inference/inference_bm25.py, utils/prompt_template.py and utils/text_processing.py. "
    "Provenance of that repository is self-asserted in its README; the algorithm details "
    "ported here were verified against its source directly."
)
DEFAULT_MODEL = "gpt-4.1"

# --- Frozen upstream hyperparameters. Every value below was read from the
# upstream source named in the comment; none was tuned for this suite. ---
MAX_TURNS = 4  # inference_bm25.py parse_args(): --max_turns default 4
TOP_DOCS = 6  # inference_bm25.py parse_args(): --top_docs default 6
EVIDENCE_TOP_K = 6  # main_batch's own concat_and_pick_sentences_batch(return_top_k=6)
MAX_SENTS_PER_DOC = 40  # main_batch's own concat_and_pick_sentences_batch(max_sents_per_doc=40)
# build_query_from_missing: `1 if dataset == "triviaqa" else 3`. All three
# benchmarks here are multi-hop, never TriviaQA, so 3 is the live value.
MAX_GAP_FACTS_FOR_QUERY = 3
MAX_GAP_FACTS_FOR_SELECTOR = 3  # format_missing_facts_for_selector(max_facts=3)
# run_S2G-RAG.py's own interactive default is "remove repeated retrieved
# docs? (yes/no, default: yes)", so every script it generates passes
# --remove_repeat_docs. Upstream's bare argparse default is False; the
# shipped runner's default (True) is what a user actually runs.
REMOVE_REPEAT_DOCS = True
EVIDENCE_SEPARATOR = "\n\n---\n\n"  # append_evidence_context's own `sep` default
QUESTION_TYPE = "OEQ"  # parse_args(): --question_type default "OEQ"
NO_RESULTS = "No results found."  # bm25_search_batch's own sentinel

DEVIATIONS: list[dict[str, str]] = [
    {
        "official_setting": (
            "S2G-Judge is Llama-3.2-3B-Instruct LoRA-fine-tuned through the paper's "
            "6-stage pipeline (GPT-4o-mini teacher distillation -> supervision "
            "filtering -> LoRA SFT)."
        ),
        "adapted_setting": (
            "GPT-4.1 at temperature 0 is prompted with upstream's own verbatim "
            "SUFF_SYSTEM_PROMPT and emits the identical JSON schema."
        ),
        "reason": (
            "No trained checkpoint is released with the paper or repository, and the "
            "user explicitly decided not to run the training pipeline. Every other part "
            "of the algorithm -- schema, gap-to-query mechanism, evidence extractor, "
            "turn loop, caps -- is unchanged."
        ),
    },
    {
        "official_setting": (
            "Retrieval over a whole-Wikipedia Pyserini BM25 index (or "
            "intfloat/e5-base-v2 + FAISS), top_docs=6."
        ),
        "adapted_setting": (
            "BM25+RRF over this question's own materialized DocumentRecord set via "
            "ant.tools.local.LocalSearchTool -- the same primitive and the same "
            "candidate collection Sparse Retrieval already uses. top_docs=6 unchanged."
        ),
        "reason": (
            "The evaluation's fairness constraint requires every method to retrieve "
            "from the SAME question-visible candidate document collection. Giving "
            "S2G-RAG a whole-Wikipedia index would make it the only method with a "
            "larger corpus."
        ),
    },
    {
        "official_setting": "Questions processed in batches of 8 through one HF generate() call.",
        "adapted_setting": "One question per run(), sequential calls.",
        "reason": (
            "Batching is a local-GPU throughput device that changes no decision; this "
            "suite's AgentAdapter contract is one example per run()."
        ),
    },
    {
        "official_setting": (
            "split_wiki_sentences prefers pysbd, with upstream's own regex fallback "
            "inside a try/except."
        ),
        "adapted_setting": "Same try/except ported; pysbd is absent here, so the fallback runs.",
        "reason": "Upstream's own documented fallback path, not a new behavior.",
    },
    {
        "official_setting": (
            "Upstream's main_batch also computes Correct Answer / Correct Retrieval "
            "against gold answers and gold document titles inline."
        ),
        "adapted_setting": "Not ported; scoring is the benchmark adapter's job in this suite.",
        "reason": (
            "Porting it would require reading the gold answer and gold supporting-document "
            "fields of the TaskExample inside the method, which the no-leakage constraint "
            "forbids. score_hotpot_style / MuSiQueAdapter.score do it instead, after the "
            "prediction is frozen."
        ),
    },
]


# ===========================================================================
# Verbatim ports: utils/prompt_template.py
# ===========================================================================

formatting2 = """Respond only with the following format, nothing else:
Answer: [Provide the answer here]
Rationale: [Provide the rationale here]

Do not include any additional text, headers, or explanations outside this format.
"""

force_answer_prompt = (
    "You are a knowledgeable question-answering assistant. "
    "Based on the context provided (if any), answer the following "
    "multihop question. Provide a brief multihop explanation. "
    "Prefer the shortest exact answer span that fully answers the question. "
) + formatting2

SUFF_SYSTEM_PROMPT = """You are a QA/RAG sufficiency judge.
Given a QUESTION and a CONTEXT (documents retrieved so far),
decide whether the CONTEXT alone contains enough information to reliably answer the QUESTION.
If not, list the gap items that describe what information is still missing.

You MUST respond with a single JSON object with the following shape:

{
  "sufficient": true/false,
  "gap_items": [
    {
      "category": "bridge_entity | attribute | relation | evidence_span | other",
      "target": "string",
      "slot": "string",
      "description": "string"
    },
    ...
  ]
}

If the information is sufficient, "gap_items" MUST be an empty list [].
"""

SELECTOR_SYSTEM_PROMPT = """
You are a sentence-level evidence selector for a multi-hop RAG system.

You will receive:
1. an ORIGINAL QUESTION,
2. MISSING FACTS that describe what information is still missing,
3. a numbered list of SENTENCES from retrieved documents.

Your task is to select the sentence ids that maximize answerability for the ORIGINAL QUESTION.

Selection policy:
1. First prioritize sentences that fill the MISSING FACTS, especially bridge entities, attributes, relations, and evidence spans needed for the next hop.
2. Then prioritize sentences that directly support the final answer to the ORIGINAL QUESTION.
3. Prefer sentences that are self-contained and explicit:
   - they mention the key entity, relation, attribute, date, number, or answer-bearing fact;
   - they remain understandable when extracted alone.
4. If a selected sentence depends on nearby context to be understandable or useful, include the minimal additional sentence(s) needed to preserve that context.
5. Do not infer, rewrite, paraphrase, or generate evidence text. Only return ids from the provided list.
6. If no sentence is useful, return an empty list.

Output format (strict):
Return exactly one JSON object and nothing else:
{"evidence_global_ids": [1, 5, 7]}

Constraints:
- "evidence_global_ids" must be a JSON array of integers.
- Select at most K sentences, where K is given in the user message.
- Only use ids that appear in the numbered sentence list.
- Do not repeat ids.
""".strip()


# ===========================================================================
# Verbatim ports: utils/text_processing.py
# ===========================================================================

FINAL_ANSWER_AND_RATIONALE_RE = re.compile(r"Answer:\s*(.*?)\s*Rationale:\s*(.*)", re.DOTALL)


def _normalize_json_key(key: Any) -> str:
    return re.sub(r"[\s_]+", "", str(key).strip().lower())


def get_json_value(payload: Any, *candidates: str, default: Any = None) -> Any:
    if not isinstance(payload, dict):
        return default
    for candidate in candidates:
        if candidate in payload:
            return payload[candidate]
    normalized_map = {_normalize_json_key(key): value for key, value in payload.items()}
    for candidate in candidates:
        normalized_candidate = _normalize_json_key(candidate)
        if normalized_candidate in normalized_map:
            return normalized_map[normalized_candidate]
    return default


def extract_gap_items(payload: Any) -> Any:
    return get_json_value(payload, "gap_items", "gap items", "missing_facts", default=[])


def extract_evidence_global_ids(payload: Any) -> Any:
    return get_json_value(payload, "evidence_global_ids", "evidence global ids", default=[])


def extract_final_answer_and_rationale(
    text: str, question_type: str = "OEQ"
) -> tuple[str, str]:
    match = FINAL_ANSWER_AND_RATIONALE_RE.search(text or "")
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "Answer not found", "Rationale not found"


# ===========================================================================
# Verbatim ports: inference/inference_bm25.py
# ===========================================================================

try:  # upstream's own try/except -- see DEVIATIONS entry 4
    import pysbd

    _PYSBD_OK = True
    _PYSBD_SEGMENTER = pysbd.Segmenter(language="en", clean=False)
except Exception:  # noqa: BLE001 - upstream catches bare Exception here too
    pysbd = None  # type: ignore[assignment]
    _PYSBD_OK = False
    _PYSBD_SEGMENTER = None

_CJK_RE = re.compile(r"[一-鿿]")
_JSON_RE = re.compile(r"\{.*\}", re.S)


def safe_json_load(text: str) -> Any:
    """Parse JSON robustly, including fenced code blocks."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        match = _JSON_RE.search(text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:  # noqa: BLE001
            return None


def split_wiki_sentences(text: str) -> list[str]:
    """Split Wikipedia-like text into sentences."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    paragraphs = re.split(r"\n\s*\n+", text)
    sentences: list[str] = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        cjk_count = len(_CJK_RE.findall(paragraph))
        if cjk_count > 0 and (cjk_count / max(1, len(paragraph))) > 0.02:
            parts = re.split(r"(?<=[。！？!?])", paragraph)
            sentences.extend(x.strip() for x in parts if x and x.strip())
            continue
        if _PYSBD_OK:
            parts = _PYSBD_SEGMENTER.segment(paragraph)  # type: ignore[union-attr]
            sentences.extend(x.strip() for x in parts if x and x.strip())
        else:
            parts = re.split(r"(?<=[.!?])\s+|[\n]+", paragraph)
            sentences.extend(x.strip() for x in parts if x and x.strip())
    return sentences


def concat_raw_retrieved_docs(titles: list[str], texts: list[str]) -> str:
    blocks = []
    for title, raw_text in zip(titles or [], texts or [], strict=False):
        if not raw_text:
            continue
        blocks.append(f"[{title or 'N/A'}]\n{raw_text.strip()}")
    return EVIDENCE_SEPARATOR.join(blocks)


def build_suff_user_prompt(question: str, context: str) -> str:
    return f"""QUESTION:
{question}

CONTEXT:
{context}

Please ONLY use the above CONTEXT as evidence.
Decide whether it is sufficient, and if not, list the missing facts
in the required JSON format."""


def build_query_from_missing(
    question: str, missing_facts: Any, max_facts: int | None = None
) -> str:
    """Build a retrieval query from missing facts. Upstream signature also
    takes `dataset_name` purely to pick `max_facts = 1 if triviaqa else 3`;
    TriviaQA is not one of this suite's benchmarks, so the non-TriviaQA
    branch's value is the module-level default here (see
    MAX_GAP_FACTS_FOR_QUERY). The concatenation logic itself is verbatim."""
    if max_facts is None:
        max_facts = MAX_GAP_FACTS_FOR_QUERY
    if not missing_facts:
        return question

    phrases = []
    for item in missing_facts[:max_facts]:
        if not isinstance(item, dict):
            continue
        target = (item.get("target") or "").strip()
        slot = (item.get("slot") or "").strip()
        description = (item.get("description") or "").strip()
        if target and slot:
            phrases.append(f"{target} {slot}")
        elif description:
            phrases.append(description)

    return question if not phrases else question + " " + " ".join(phrases)


def should_force_first_retrieval(turn: int, evidence_context: str, verdict: bool) -> bool:
    """Force one retrieval when the first-turn gate is overconfident on empty evidence."""
    return turn == 0 and bool(verdict) and not str(evidence_context or "").strip()


def format_missing_facts_for_selector(missing_facts: Any, max_facts: int = 3) -> str:
    """Format missing-fact hints for the selector prompt."""
    if not missing_facts:
        return "None."
    lines = []
    for item in missing_facts[:max_facts]:
        if not isinstance(item, dict):
            continue
        category = (item.get("category") or "").strip()
        target = (item.get("target") or "").strip()
        slot = (item.get("slot") or "").strip()
        description = (item.get("description") or "").strip()
        core = f"{target} {slot}".strip() if target and slot else description
        if core:
            lines.append(f"- [{category or 'other'}] {core}")
    return "\n".join(lines) if lines else "None."


def merge_evidence_only(
    titles: list[str], texts: list[str], evidence_ids_per_doc: list[list[int]]
) -> str:
    """Keep only selected evidence sentences in a merged block."""
    blocks = []
    for title, raw_text, sentence_ids in zip(
        titles, texts, evidence_ids_per_doc, strict=False
    ):
        if not raw_text or not sentence_ids:
            continue
        sentences = split_wiki_sentences(raw_text)
        chosen = []
        for index in sentence_ids:
            if 1 <= index <= len(sentences):
                chosen.append(sentences[index - 1])
        if chosen:
            blocks.append(f"[{title or 'N/A'}] EVIDENCE:\n- " + "\n- ".join(chosen))
    return EVIDENCE_SEPARATOR.join(blocks)


def append_evidence_context(
    previous: str, new_text: str, sep: str = EVIDENCE_SEPARATOR
) -> str:
    """Append new evidence text while avoiding exact duplication. APPEND-ONLY:
    the accumulated context is never overwritten or truncated."""
    previous = (previous or "").strip()
    new_text = (new_text or "").strip()
    if not new_text:
        return previous
    if not previous:
        return new_text
    if new_text in previous:
        return previous
    return previous + sep + new_text


def update_task_with_evidence(task_content: str, query: str, evidence_block: str) -> str:
    """Append retrieved evidence to the reasoner's task content."""
    evidence_block = (evidence_block or "").strip()
    if not evidence_block:
        return task_content
    return f"{task_content}Query: {query}\nRetrieved Document: {evidence_block}\n"


def build_selector_user_prompt(
    question: str,
    titles: list[str],
    texts: list[str],
    missing_facts: Any,
    return_top_k: int = EVIDENCE_TOP_K,
    max_sents_per_doc: int = MAX_SENTS_PER_DOC,
) -> tuple[str | None, dict[int, tuple[int, int]]]:
    """Verbatim port of the per-question half of upstream's
    `concat_and_pick_sentences_batch`: number every sentence of every
    retrieved document globally, attach the gap-item hints, and cap the
    selection at `return_top_k`. Returns (user_prompt, id_map); a None
    prompt means there were no usable sentences and no LLM call is made
    (upstream skips those entries too)."""
    sentences: list[tuple[int, int, str, str]] = []
    for doc_idx, (title, raw_text) in enumerate(zip(titles, texts, strict=False)):
        if not raw_text or raw_text.strip() == NO_RESULTS:
            continue
        parts = split_wiki_sentences(raw_text)
        if max_sents_per_doc is not None:
            parts = parts[:max_sents_per_doc]
        for local_sid, sentence_text in enumerate(parts, start=1):
            sentences.append((doc_idx, local_sid, title or "N/A", sentence_text))

    if not sentences:
        return None, {}

    numbered_lines = []
    id_map: dict[int, tuple[int, int]] = {}
    for global_id, (doc_idx, local_sid, title, sentence_text) in enumerate(sentences, start=1):
        numbered_lines.append(f"[{global_id}] ({title} | s#{local_sid}) {sentence_text}")
        id_map[global_id] = (doc_idx, local_sid)

    hints = format_missing_facts_for_selector(
        missing_facts, max_facts=MAX_GAP_FACTS_FOR_SELECTOR
    )
    user_prompt = (
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"MISSING FACTS TO FILL:\n{hints}\n\n"
        f"NUMBERED SENTENCES FROM RETRIEVED DOCUMENTS:\n"
        + "\n".join(numbered_lines)
        + "\n\n"
        + f"You may select up to {return_top_k} sentences.\n"
        + 'Return ONLY JSON with "evidence_global_ids".'
    )
    return user_prompt, id_map


def parse_selector_output(
    raw_output: str, id_map: dict[int, tuple[int, int]], n_docs: int
) -> list[list[int]]:
    """Verbatim port of upstream's own selector-output parsing: ints only,
    ids >= 1 only, unknown ids dropped, per-document sorted unique."""
    parsed = safe_json_load((raw_output or "").strip())
    if not isinstance(parsed, dict):
        parsed = {}

    global_ids: list[int] = []
    values = extract_evidence_global_ids(parsed)
    if isinstance(values, list):
        for value in values:
            try:
                value = int(value)
                if value >= 1:
                    global_ids.append(value)
            except Exception:  # noqa: BLE001
                pass

    per_doc: list[list[int]] = [[] for _ in range(n_docs)]
    for global_id in global_ids:
        if global_id not in id_map:
            continue
        doc_idx, local_sid = id_map[global_id]
        if 0 <= doc_idx < len(per_doc):
            per_doc[doc_idx].append(int(local_sid))
    return [sorted(set(x)) for x in per_doc]


# ===========================================================================
# Retrieval over THIS suite's shared per-question candidate collection.
# ===========================================================================


class SharedCorpusRetriever:
    """Document-level BM25 retrieval over exactly the documents this
    question makes visible -- the same materialized `DocumentRecord` set,
    reached through the same `LocalSearchTool.search` (BM25 + symbol-path
    channel, fused by Reciprocal Rank Fusion) that Sparse Retrieval
    already uses.

    `LocalSearchTool.search` ranks REGIONS; upstream S2G-RAG ranks whole
    DOCUMENTS. Region hits are therefore collapsed to their source
    document, first-hit-wins, preserving the fused ranking order -- which
    is exactly `bm25_search_batch`'s own "walk hits in rank order, keep
    the first `k` acceptable doc_ids" loop, including its
    `remove_repeat_docs` filter against already-seen doc_ids and its
    "No results found." sentinel when nothing survives.

    Never reads supporting_doc_ids, reference, or any other gold field --
    it is constructed from DocumentRecords alone, which structurally carry
    only doc_id/title/text.
    """

    def __init__(
        self,
        environment_root: Path,
        documents: list[DocumentRecord],
        *,
        remove_repeat_docs: bool = REMOVE_REPEAT_DOCS,
    ) -> None:
        self.environment = EvalDocumentEnvironment(environment_root, documents)
        self.search_tool = LocalSearchTool(environment_root)
        self.files = [self.environment.relative_path_for(doc.doc_id) for doc in documents]
        self.corpus = {doc.doc_id: doc for doc in documents}
        self.remove_repeat_docs = remove_repeat_docs
        self.search_calls = 0

    def search(
        self, query: str, past_doc_ids: list[str], k: int = TOP_DOCS
    ) -> tuple[list[str], list[str], list[str]]:
        """Returns (titles, texts, doc_ids) -- same triple, same sentinel
        behavior, as upstream's `bm25_search_batch` for one question."""
        past = set(past_doc_ids) if self.remove_repeat_docs else set()
        self.search_calls += 1
        # Upstream asks its searcher for `max(k, 50)` hits and then filters
        # down; the same head-room is requested here.
        hits = self.search_tool.search(
            str(query or "").strip(), self.files, limit=max(k, 50)
        )

        filtered_doc_ids: list[str] = []
        seen: set[str] = set()
        for hit in hits:
            doc_id = self.environment.doc_id_for_relative_path(hit.path)
            if doc_id is None or doc_id not in self.corpus:
                continue
            if doc_id in seen:
                continue
            if self.remove_repeat_docs and doc_id in past:
                continue
            seen.add(doc_id)
            filtered_doc_ids.append(doc_id)
            if len(filtered_doc_ids) >= k:
                break

        top_titles = [self.corpus[d].title or NO_RESULTS for d in filtered_doc_ids]
        top_texts = [self.corpus[d].text or NO_RESULTS for d in filtered_doc_ids]
        top_doc_ids = list(filtered_doc_ids)

        if not top_titles:
            top_doc_ids = [""]
            top_titles = [NO_RESULTS]
            top_texts = [NO_RESULTS]

        return top_titles, top_texts, top_doc_ids


class _ZeroTemperatureProvider(CountingOpenAIProvider):
    """Upstream runs the gate with `do_sample=False` and both reasoner
    calls with `temperature=0.0, do_sample=False`. Same override pattern
    `ant.external_wrappers.chainrag` and
    `ant.evaluation_suite.answer_extraction` already use."""

    def _responses_kwargs(self, prompt: str, max_output_tokens: int) -> dict:
        kwargs = super()._responses_kwargs(prompt, max_output_tokens)
        kwargs["temperature"] = 0
        return kwargs


# Upstream's own per-call generation budgets (call_suff_gate_batch's
# max_new_tokens=256; concat_and_pick_sentences_batch's max_new_tokens=64;
# main_batch's answer call max_new_tokens=128). All are >= the OpenAI
# Responses API's own hard minimum of 16.
JUDGE_MAX_OUTPUT_TOKENS = 256
SELECTOR_MAX_OUTPUT_TOKENS = 64
ANSWER_MAX_OUTPUT_TOKENS = 128


class S2GRAGAdapter:
    """S2G-RAG for the document substrate. `environment_root` is the
    materialized document directory this suite's benchmark adapters
    already produce -- the identical directory Sparse Retrieval searches.
    """

    name = "s2g_rag"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_turns: int = MAX_TURNS,
        top_docs: int = TOP_DOCS,
        evidence_top_k: int = EVIDENCE_TOP_K,
        remove_repeat_docs: bool = REMOVE_REPEAT_DOCS,
    ) -> None:
        self.model = model
        self.max_turns = max_turns
        self.top_docs = top_docs
        self.evidence_top_k = evidence_top_k
        self.remove_repeat_docs = remove_repeat_docs

    # -- the three LLM roles ------------------------------------------------

    def _judge(self, provider: Any, question: str, evidence_context: str) -> tuple[bool, list]:
        """The S2G-Judge. GPT-4.1 substituted for the paper's LoRA Llama
        judge; the prompt, the schema and the parsing are upstream's own
        `call_suff_gate_batch`, unchanged."""
        prompt = SUFF_SYSTEM_PROMPT + "\n\n" + build_suff_user_prompt(question, evidence_context)
        raw = provider.responses_text(prompt, max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS).text
        parsed = safe_json_load((raw or "").strip())
        if not isinstance(parsed, dict):
            return False, []
        sufficient = bool(parsed.get("sufficient", False))
        missing_facts = extract_gap_items(parsed)
        if sufficient or not isinstance(missing_facts, list):
            missing_facts = []
        return sufficient, missing_facts

    def _select_evidence(
        self,
        provider: Any,
        question: str,
        titles: list[str],
        texts: list[str],
        missing_facts: Any,
    ) -> tuple[str, bool]:
        """The sentence-level Evidence Extractor. Returns (evidence_block,
        made_llm_call) -- upstream skips the call entirely when a turn's
        retrieved documents yielded no usable sentences."""
        user_prompt, id_map = build_selector_user_prompt(
            question,
            titles,
            texts,
            missing_facts,
            return_top_k=self.evidence_top_k,
            max_sents_per_doc=MAX_SENTS_PER_DOC,
        )
        if user_prompt is None:
            return "", False
        raw = provider.responses_text(
            SELECTOR_SYSTEM_PROMPT + "\n\n" + user_prompt,
            max_output_tokens=SELECTOR_MAX_OUTPUT_TOKENS,
        ).text
        per_doc = parse_selector_output(raw, id_map, len(titles))
        return merge_evidence_only(titles, texts, per_doc), True

    # -- retrieval-backend / answer-stage hooks -----------------------------
    #
    # Everything above and below these three hooks is the algorithm itself
    # (judge -> gap -> query -> retrieve -> extract -> loop) and is shared,
    # unforked, by every corpus substrate. A substrate subclass overrides
    # ONLY these: which retriever wraps its candidate pool, which answer
    # prompt upstream would dispatch for it (upstream itself dispatches its
    # answer/selector prompts per dataset -- see `get_answer_system_prompt`),
    # and how the parsed Answer/Rationale pair becomes `final_answer`.

    ANSWER_SYSTEM_PROMPT: str = force_answer_prompt
    answer_max_output_tokens: int = ANSWER_MAX_OUTPUT_TOKENS
    corpus_substrate: str = "multi_hop_qa_documents"

    def deviations(self) -> list[dict[str, str]]:
        """Disclosed official -> adapted -> reason record, emitted into every
        AgentResult so it always travels with the results."""
        return DEVIATIONS

    def _build_retriever(self, example: TaskExample, environment_root: Path) -> Any:
        documents = [DocumentRecord(**d) for d in example.metadata["documents"]]
        return SharedCorpusRetriever(
            environment_root, documents, remove_repeat_docs=self.remove_repeat_docs
        )

    def _assemble_final_answer(self, answer: str, rationale: str, raw: str) -> str:
        """Extractive multi-hop QA is EM/F1-scored against a short gold
        span, so the `Answer:` field alone is the prediction -- exactly
        what upstream's own `main_batch` writes to `Reasoner Answer`."""
        del rationale, raw
        return answer

    def _answer(self, provider: Any, task_content: str) -> tuple[str, str, str]:
        raw = provider.responses_text(
            self.ANSWER_SYSTEM_PROMPT + "\n\n" + task_content,
            max_output_tokens=self.answer_max_output_tokens,
        ).text
        answer, rationale = extract_final_answer_and_rationale(raw, QUESTION_TYPE)
        return answer, rationale, raw

    # -- the turn loop ------------------------------------------------------

    def run(self, example: TaskExample, environment_root: Path) -> AgentResult:
        question = example.question

        started = time.time()
        provider = _ZeroTemperatureProvider(model=self.model)
        retriever = self._build_retriever(example, environment_root)

        # Upstream's `run_system` seeds the reasoner task with
        # f"Question: {question}\n" when --gpt is set (which it is here,
        # since GPT-4.1 is the reasoner).
        task_content = f"Question: {question}\n"
        evidence_context = ""
        raw_retrieved_concat = ""
        past_doc_ids: list[str] = []
        past_titles: list[str] = []
        trajectory: list[dict[str, Any]] = []

        turn = 0
        n_judge_calls = 0
        n_selector_calls = 0
        final_answer = "Answer not found"
        rationale = "Rationale not found"
        raw_answer = ""
        termination_reason = "max_turns_reached"

        while turn <= self.max_turns:
            verdict, missing_facts = self._judge(provider, question, evidence_context)
            n_judge_calls += 1

            forced_first_retrieval = should_force_first_retrieval(
                turn, evidence_context, verdict
            )
            if forced_first_retrieval:
                verdict = False
                missing_facts = []

            done = bool(verdict) or turn == self.max_turns
            need_retrieve = (not verdict) and turn < self.max_turns

            turn_record: dict[str, Any] = {
                "turn": turn,
                "sufficient": bool(verdict),
                "forced_first_retrieval": forced_first_retrieval,
                "gap_items": missing_facts,
                "evidence_context_chars": len(evidence_context),
            }

            if need_retrieve:
                query = build_query_from_missing(question, missing_facts)
                titles, texts, doc_ids = retriever.search(
                    query, past_doc_ids, k=self.top_docs
                )
                for doc_id, title in zip(doc_ids, titles, strict=False):
                    if doc_id and doc_id != NO_RESULTS and doc_id not in past_doc_ids:
                        past_doc_ids.append(doc_id)
                    if title and title != NO_RESULTS and title not in past_titles:
                        past_titles.append(title)

                new_raw = concat_raw_retrieved_docs(titles, texts)
                if new_raw:
                    raw_retrieved_concat = (
                        raw_retrieved_concat + EVIDENCE_SEPARATOR + new_raw
                        if raw_retrieved_concat
                        else new_raw
                    )

                merged, selector_called = self._select_evidence(
                    provider, question, titles, texts, missing_facts
                )
                if selector_called:
                    n_selector_calls += 1

                # APPEND-ONLY: never overwrites the accumulated context.
                evidence_context = append_evidence_context(evidence_context, merged)
                task_content = update_task_with_evidence(task_content, query, merged)

                turn_record["retrieval_query"] = query
                turn_record["retrieved_doc_ids"] = doc_ids
                turn_record["retrieved_titles"] = titles
                turn_record["evidence_block_chars"] = len(merged)

            if done:
                parsed_answer, rationale, raw_answer = self._answer(provider, task_content)
                final_answer = self._assemble_final_answer(parsed_answer, rationale, raw_answer)
                turn_record["answered"] = True
                termination_reason = (
                    "judge_declared_sufficient" if verdict else "max_turns_reached"
                )

            trajectory.append(turn_record)

            if not need_retrieve:
                break
            turn += 1

        token_usage = provider.drain_usage()
        llm_calls = provider.drain_call_count()
        elapsed = time.time() - started

        return AgentResult(
            benchmark=example.benchmark,
            task_id=example.task_id,
            method=self.name,
            final_answer=final_answer,
            trajectory=trajectory,
            evidence=[
                {"doc_id": doc_id, "title": retriever.corpus[doc_id].title}
                for doc_id in past_doc_ids
                if doc_id in retriever.corpus
            ],
            usage=UsageStats(
                llm_calls=llm_calls,
                tool_calls=retriever.search_calls,
                input_tokens=token_usage.input_tokens,
                output_tokens=token_usage.output_tokens,
                total_tokens=token_usage.total_tokens,
                estimated_cost_usd=token_usage.estimated_cost_usd,
                wall_clock_seconds=elapsed,
                unique_files_inspected=len(set(past_doc_ids)),
            ),
            termination_reason=termination_reason,
            metadata={
                "generation_model": self.model,
                "judge_model": self.model,
                "implementation_label": IMPLEMENTATION_LABEL,
                "paper_citation": PAPER_CITATION,
                "upstream_commit": UPSTREAM_COMMIT,
                "upstream_repo_note": UPSTREAM_REPO_NOTE,
                "deviations": self.deviations(),
                "corpus_substrate": self.corpus_substrate,
                "max_turns": self.max_turns,
                "top_docs": self.top_docs,
                "evidence_top_k": self.evidence_top_k,
                "max_gap_facts_for_query": MAX_GAP_FACTS_FOR_QUERY,
                "remove_repeat_docs": self.remove_repeat_docs,
                "turns_used": len(trajectory),
                "n_judge_calls": n_judge_calls,
                "n_selector_calls": n_selector_calls,
                "n_answer_calls": 1,
                "n_retrieval_calls": retriever.search_calls,
                "retrieval_is_local_and_free": True,
                "retrieved_doc_ids": list(past_doc_ids),
                "final_evidence_context_chars": len(evidence_context),
                "raw_retrieved_docs_chars": len(raw_retrieved_concat),
                "raw_answer_before_extraction": raw_answer,
                "rationale": rationale,
            },
        )


from ant.evaluation_suite.registry import register_agent  # noqa: E402

register_agent(S2GRAGAdapter())
