# ChainRAG fidelity audit

Performed by reading both the paper and the real, pinned official source
before writing any implementation code, per the governing spec's Part 1.

## Paper

Zhu, Sun, Cheng, Xu, Hu, Wang, Wang, Sun, Qu (nju-websoft). "Mitigating
Lost-in-Retrieval Problems in Retrieval Augmented Multi-Hop Question
Answering." ACL 2025 (Main, Long), pp. 22362-22375. arXiv:2502.14245.
Confirmed via web search: evaluated on MuSiQue/2Wiki/HotpotQA with
GPT-4o-mini/Qwen2.5-72B/GLM-4-Plus backbones. Method summary (paper's own
abstract): "lost-in-retrieval" -- key entities get lost during
sub-question decomposition, degrading retrieval and disrupting the
reasoning chain. ChainRAG "sequentially handles each sub-question by
completing missing key entities and retrieving relevant sentences from a
sentence graph for answer generation." No paper-vs-code discrepancy
found at the methodological level -- the released code faithfully
implements what the paper describes (see per-component detail below,
drawn from the code where the paper is underspecified, per the
governing spec's own priority order).

## Official repository

`https://github.com/nju-websoft/ChainRAG`, **pinned upstream commit
`0f115b1fa03698b5d9cd2662a26dfdfde62d074a`** (2025-10-28, "Reply context
retrieval and expansion logic" -- the latest commit at audit time, GPG-
verified). **License: GPL-3.0.** Files read in full: `retrieval.py`,
`LLM.py`, `sentence_graph.py`, `utils.py`, `requirements.txt`.

## Exact algorithm (from the real, pinned source)

- **Sentence splitting:** spaCy `en_core_web_sm`, `nlp(text).sents`.
- **Entity extraction:** spaCy NER (`doc.ents`) per sentence.
- **Embedding model:** OpenAI `text-embedding-3-small` (1536-dim,
  confirmed by the code's own zero-vector fallback `[0.0] * 1536`),
  batched at 20 texts/call.
- **Entity importance:** `BM25Okapi` over per-sentence entity-text lists;
  "high importance" = above the 40th percentile of each entity's mean
  BM25 score across sentences. **Audit note:** this `high_importance_
  entities` set is computed and stored on `self.high_tfidf_entities` but
  is **never actually read anywhere else in `retrieval.py`** -- confirmed
  by reading the full file; likely vestigial from an earlier version.
  Not used by the entity-graph edges either (those use the raw
  `entity_scores` weights directly, not the thresholded set). This
  adaptation preserves the computation faithfully (since it's real
  upstream behavior) but does not invent a use for it that upstream
  itself doesn't have.
- **Sentence graph** (`networkx.Graph`, one node per sentence): three
  edge types added over the same node set --
  1. **Similarity edges:** top-**k=10** most cosine-similar sentences per
     node (embedding-based).
  2. **Positional edges:** every sentence within **±3** positions in the
     original (per-document) sentence order.
  3. **Entity edges:** any sentence pair sharing at least one entity,
     weighted by the sum of that entity's BM25 importance score.
- **Question decomposition** (`decompose_question`, 2-stage, both LLM
  calls use the EXACT prompts below verbatim):
  1. A judge call asks whether the question `is_multi_hop` (JSON
     `{"is_multi_hop": bool}`, with few-shot examples in the prompt). If
     `false`, decomposition stops -- `[question]` is returned unchanged
     (single "sub-question" = the original question).
  2. If multi-hop, a second call decomposes into a JSON array of
     sub-questions, guided by an extensive prompt distinguishing
     parallel/sequential/comparative decomposition patterns and capping
     "usually 2 at most" sub-questions (a soft guideline in the prompt
     text, not a hard-enforced limit in code).
- **Sequential sub-question processing** (`plan()`), per sub-question:
  1. **Reference check + rewrite:** if the sub-question contains any of
     a fixed pronoun list (`this/that/the/these/those/it/they/he/she/
     his/her/its/their`) AND a prior sub-question/answer pair exists, an
     LLM call asks whether the current sub-question refers to that prior
     answer; if yes, a second LLM call rewrites the sub-question to be
     self-contained (replacing the reference with the actual entity).
  2. **Seed retrieval:** `find_top_k_sentences` -- cosine similarity
     against all sentence embeddings, keep the top **100**, then rerank
     with `BAAI/bge-reranker-large` (`FlagReranker`, local) down to the
     top **k=7**.
  3. Append a one-line "previous context summary" string (the last
     sub-question's own answer, if any) to the context.
  4. **Sufficiency check:** one LLM call (`can_answer_question`,
     yes/no). If yes, generate the answer (`force_answer`, one LLM
     call); if the answer isn't "unable"/"don't know", accept it and
     move to the next sub-question.
  5. If not sufficient: expand to the seed sentences' **1-hop**
     neighbors in the sentence graph (deduplicated, capped at
     **max_words=3000** total), re-rank via the reranker, re-run the
     sufficiency check + force-answer once more.
  6. If still not sufficient: expand hop-by-hop from **2 up to
     max_hops=3** (still capped at 3000 words total), with **no further
     sufficiency check** at this stage -- the loop unconditionally
     force-answers once the hop expansion finishes (or the word cap is
     hit).
- **Final answer, two variants, both always computed:**
  - `answer_with_subquestions`: LLM given ONLY the original question +
    the list of (sub-question, answer) text pairs -- no raw evidence
    sentences at all.
  - `answer_without_subquestions`: LLM given ONLY the deduplicated,
    reranker-sorted union of every sub-question's own retrieved context
    sentences -- no sub-question answers at all.
  - Per the governing spec's Section 9 (and confirmed this is a real,
    exact field name in the official output), **`answer_with_
    subquestions` is the precommitted canonical field**, chosen before
    any inference, not selected after seeing scores.
- **LLM call accounting (audit note on upstream's OWN counter):**
  upstream's own `llm_counter` **undercounts** physical calls whenever
  `decompose_question`'s judge call returns "multi-hop" (2 real API
  calls -- judge + decompose -- credited as only `+1` in `plan()`, since
  `plan()` increments once right before calling the whole
  `decompose_question` function). This adaptation counts every REAL
  physical call via `CountingOpenAIProvider`, not upstream's own
  (undercounting) convention.
- **Retry:** upstream wraps `custom_llm`/`custom_embedding` in `tenacity`
  (5 attempts, exponential backoff 4-10s, only on `ConnectionError`/
  `Timeout`). Per the governing spec's Section 14, this adaptation uses
  **our own** frozen retry policy instead (`ant.evaluation_suite.
  retry_policy.call_with_transient_retry`: 429/500/502/503/timeout/
  connection errors, max 2 retries, 1s/2s backoff) for both LLM and
  embedding calls -- a disclosed, spec-mandated substitution of the
  retry *mechanism* only, not of what counts as a retriable failure
  (both are transient-network-only policies).

## Model substitution (disclosed per Section 5)

Upstream: `gpt-4o-mini`, `temperature=0.2` (both hardcoded in
`custom_llm`). **This evaluation uses GPT-4.1, temperature=0** for
architecture-comparison consistency with every other method in this
suite. ChainRAG's own paper evaluates multiple interchangeable backbones
(GPT-4o-mini, Qwen2.5-72B, GLM-4-Plus), supporting the characterization
of this as a backbone substitution, not an algorithmic change. All
prompts and decision logic are otherwise preserved verbatim.

## Retrieval stack (preserved, per Section 6)

`text-embedding-3-small` (real OpenAI API calls -- **not free**, unlike
this suite's own local BGE-small dense baseline) -> cosine similarity
top-100 -> `BAAI/bge-reranker-large` via `FlagReranker` (local, verified
installed and working: `fastembed`... no -- `FlagEmbedding==1.4.2`,
confirmed the reranker loads and scores correctly on a local smoke pair
before any paid call). No BM25/RRF/shared `LocalSearchTool`/ANTMAN
evidence ranking is substituted anywhere in the retrieval path.

## Hyperparameters (frozen from the audit, before any inference)

| Parameter | Value | Source |
|---|---|---|
| Embedding candidate pool | 100 | `retrieval.py::find_top_k_sentences` |
| Reranked seed count (k) | 7 | `retrieval.py::plan` (`k=7`) |
| Similarity-edge k | 10 | `sentence_graph.py::build_sentence_graph` (`k=10`) |
| Positional-edge window | ±3 | `sentence_graph.py::build_sentence_graph` |
| Entity-importance percentile | 40th | `sentence_graph.py::calculate_bm25_importance` |
| Max context words | 3000 | `retrieval.py::plan` (`max_words = 3000`) |
| Max graph hops | 3 | `retrieval.py::plan` (`max_hops = 3`) |
| Sub-question decomposition cap | "usually 2" (soft, prompt-only) | `sentence_graph.py::decompose_question` |

No discrepancy found between the paper and the released code on any of
these values -- all are used exactly as released, none tuned against
this evaluation's own 45 examples.

## Implementation bug found and fixed during the smoke test (not a fidelity change)

The first smoke-test run (before this fix) produced `final_answer_with_
subquestions = "Unknown"` on all 3 examples: every sub-question hit the
per-sub-question safety-net `except Exception` and was recorded as
"Unable to process this sub-question due to error." Root-caused via a
targeted repro script that removed the broad catch: `can_answer_question`
and the reference-resolution check both called the OpenAI Responses API
with `max_output_tokens=10`, which the API hard-rejects (`HTTP 400
invalid_request_error: "Invalid 'max_output_tokens': integer below
minimum value. Expected a value >= 16, but got 10 instead."` -- verified
by reproducing the raw HTTP call and reading the response body). This is
a pure API-parameter bug in this adaptation's own token-budget constants
(`10` was never one of ChainRAG's own hyperparameters -- it does not
appear in the frozen hyperparameter table above), not a change to any
ChainRAG algorithm, prompt, or hyperparameter. Fixed by raising both call
sites from `max_output_tokens=10` to `max_output_tokens=16` (the same
minimum already used elsewhere in this codebase, e.g.
`openai_provider.py`'s own `yes/no` helper). Re-ran the smoke test after
the fix before proceeding to any fidelity/cost-gate judgment on real
output.

## Corpus adaptation (per Section 4 -- the only unavoidable adaptation)

Upstream's own `chain_rag()` loads one pre-flattened `context` string per
question from its own dataset JSONL files (`data/{hotpotqa,musique,
2wikimqa}.jsonl`) -- a single already-concatenated text blob, not a list
of separate documents. Our frozen 45 examples are each a list of 10-20
separate `DocumentRecord`s. This adaptation splits each document's text
into sentences **independently** (preserving document boundaries for the
post-hoc supporting-document-recall diagnostic) and concatenates the
resulting sentence lists in document order before graph construction --
producing the same flat, ordered sentence list upstream's own
single-text `build_graph(text)` would produce, without changing the
graph-construction algorithm itself. This is the only corpus-level
adaptation made; no other component was altered to accommodate it.
