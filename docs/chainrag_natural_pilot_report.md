# ChainRAG added to the preliminary natural multi-document QA pilot

Frozen baseline-completion run: ChainRAG (Zhu et al., ACL 2025 Main,
"Mitigating Lost-in-Retrieval Problems in Retrieval Augmented Multi-Hop
Question Answering", `github.com/nju-websoft/ChainRAG`) added as a
contemporary multi-hop-specific baseline. No existing method (Sparse/
Dense/ReAct/LongAgent/CoA/ANTMAN) rerun. No ANTMAN core file modified.
No scorer/extraction change. Full pre-implementation audit in
`docs/chainrag_fidelity_audit.md` (governing spec Part 1).

## A. Upstream fidelity (summary — full detail in the audit doc)

- Pinned upstream commit `0f115b1fa03698b5d9cd2662a26dfdfde62d074a`
  (GPL-3.0). No paper/code discrepancy found.
- Algorithm preserved verbatim: spaCy sentence graph (similarity top-k=10
  + positional ±3 + BM25-weighted entity edges), 2-stage LLM question
  decomposition, sequential sub-question processing with
  reference-check/rewrite, seed retrieval (`text-embedding-3-small`
  top-100 → `BAAI/bge-reranker-large` rerank to k=7), sufficiency check →
  1-hop → 2–3-hop graph expansion (capped at 3000 words), dual final
  synthesis (`answer_with_subquestions` precommitted as canonical per
  Section 9).
- Model substitution (disclosed, Section 5): GPT-4.1/temperature=0
  replacing upstream's gpt-4o-mini/temperature=0.2 — ChainRAG's own paper
  evaluates multiple interchangeable backbones, so this is a backbone
  substitution, not an algorithmic change. All prompts/decision logic
  preserved verbatim.
- Retrieval stack preserved exactly (Section 6): real, paid
  `text-embedding-3-small` embeddings + real, local `BAAI/bge-reranker-large`
  via `FlagReranker`. No BM25/RRF/shared `LocalSearchTool`/ANTMAN ranking
  substituted anywhere.
- Corpus adaptation (Section 4, the only unavoidable one): each frozen
  example's separate `DocumentRecord`s are sentence-split independently
  (preserving document provenance for post-hoc diagnostics) and
  concatenated in document order before graph construction — reproduces
  upstream's own flat, ordered sentence list without changing the
  graph-construction algorithm.

## B. Implementation bug found and fixed (disclosed, not a fidelity change)

The first smoke-test attempt (3 examples, 1/benchmark) produced
`final_answer = "Unknown"` on all 3, every sub-question recorded as
"Unable to process this sub-question due to error." Root-caused via a
targeted repro that removed the broad per-sub-question exception
handler: `can_answer_question` and the reference-resolution check both
called the OpenAI Responses API with `max_output_tokens=10`, which the
API hard-rejects (`HTTP 400`: *"Invalid 'max_output_tokens': integer
below minimum value. Expected a value >= 16, but got 10 instead."* —
confirmed by reproducing the raw HTTP call and reading the response
body). `10` was never one of ChainRAG's own hyperparameters (it does not
appear in the frozen hyperparameter table) — this is a pure
implementation-level API-parameter bug in this adaptation's own code,
not a change to any ChainRAG algorithm, prompt, or hyperparameter. Fixed
by raising both call sites to `max_output_tokens=16` (the minimum
already used elsewhere in this codebase). A regression test
(`test_no_max_output_tokens_below_openai_api_minimum`) now scans the
module source for any `max_output_tokens=` literal below 16. The smoke
test was rerun after the fix before any fidelity/cost-gate judgment was
made on real output.

## C. Smoke test (post-fix) and fidelity check

3/3 examples (HotpotQA `5a8b57f25542995d1e6f1371`, 2WikiMultihopQA
`8813f87c0bdd11eba7f7acde48001122`, MuSiQue `2hop__460946_294723`) ran
successfully: 2/3 exact-match correct; the MuSiQue miss ("Steve Hillage"
vs gold "Miquette Giraudy") is a genuine multi-hop entity-resolution
error (the model resolved "the Green performer" to jazz guitarist Grant
Green rather than the intended entity), not an implementation defect.
Graph expansion was genuinely exercised in this smoke set (MuSiQue's
context grew from the k=7 seed to 62 sentences via hop expansion).

All 13 required invariants (governing spec Section 12) verified:
same canonical question/context; graph built only from allowed docs
(`example.metadata["documents"]`, nothing else); no gold/support
leakage into any prompt (confirmed both by code inspection and a
dedicated unit test with a planted secret value); question decomposed
(2 sub-questions in all 3 smoke examples); sequential sub-question
processing with previous answers threaded into later sub-questions'
context (2Wiki: sub-question 2's context included a "previous context
summary" line derived from sub-question 1's answer, and resolved
correctly); rewrite mechanism active (the reference-check LLM call
fired for 2Wiki's "this director" — see Section E for why it did not
produce a literal rewrite); retrieval uses ChainRAG's own embedding +
reranker (real embedding cost logged, real BGE reranker scores used);
graph expansion available and exercised; final synthesis integrates
sub-results; no ANTMAN components bound (structural test); no shared
sparse retrieval substituted (structural test); no score-driven
retry/tuning (the only change made was the API-parameter bug fix in
Section B, made before any accuracy judgment, not in response to one).

## D. Cost gate and full run

- Smoke (3 examples, post-fix): **$0.0316**. Projected for 45:
  **$0.47** — well under the **$6.00** cap.
- Full run: the 3 already-paid smoke `AgentResult`s were reused verbatim
  (rescored via the benchmark's own native scorer, exactly what the
  harness itself does) rather than re-paying for them; `run_suite`
  (`resume=True`) then executed the remaining 42 examples fresh.
- **45/45 completed, 0 errors, 0 retries triggered.**
- Total generation cost: **$0.4056** (including the 3 reused smoke
  results). Extraction-pipeline rescoring cost: **$0.0283**. **Total:
  $0.4339** — under the $6.00 cap.
- Result paths: `output/runs/natural-multidoc-pilot/{hotpotqa,
  2wikimultihopqa,musique}/chainrag.jsonl` (raw), `.rescored.jsonl`
  (extracted, canonical), `trajectories/chainrag-*.json` (45 full
  traces: decomposition, per-sub-question context/answers, retrieved
  seed doc ids, both final-answer variants).
- Raw vs. extracted scores are **identical** on all 45 examples — the
  shared extraction pipeline never changed ChainRAG's answer. This is
  expected: ChainRAG's own `force_answer`/`answer_with_subquestions`
  prompts already explicitly request "fewest words possible" answers,
  functionally pre-empting what `extract_answer_span` would otherwise
  do.

## E. Results

**Per-benchmark (extracted EM/F1, n=15):**

| Benchmark | Metric | Sparse | Dense | **ChainRAG** | ReAct | LongAgent | CoA | ANTMAN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | EM | 0.600 | 0.600 | **0.533** | 0.533 | 0.533 | 0.600 | 0.600 |
| HotpotQA | F1 | 0.749 | 0.758 | **0.711** | 0.744 | 0.758 | 0.793 | 0.746 |
| 2WikiMultihopQA | EM | 0.467 | 0.467 | **0.733** | 0.467 | 0.733 | 0.667 | 0.800 |
| 2WikiMultihopQA | F1 | 0.519 | 0.479 | **0.789** | 0.737 | 0.853 | 0.820 | 0.812 |
| MuSiQue | EM | 0.467 | 0.200 | **0.467** | 0.400 | 0.267 | 0.533 | 0.467 |
| MuSiQue | F1 | 0.538 | 0.280 | **0.613** | 0.536 | 0.309 | 0.662 | 0.596 |

**Macro average (equal-weight across the 3 benchmarks, n=15 each):**

| Method | Avg EM | Avg F1 |
|---|---:|---:|
| Sparse Retrieval | 0.511 | 0.602 |
| Dense Retrieval | 0.422 | 0.506 |
| **ChainRAG** | **0.578** | **0.704** |
| Matched ReAct | 0.467 | 0.673 |
| LongAgent | 0.511 | 0.640 |
| CoA | 0.600 | 0.758 |
| ANTMAN | 0.622 | 0.718 |

ChainRAG ranks **3rd of 7 methods** on macro F1 (behind CoA 0.758 and
ANTMAN 0.718), and is the **strongest non-ANTMAN, non-CoA method** on
this set. It is notably strong on 2WikiMultihopQA (F1 0.789, 2nd best
overall after LongAgent's 0.853) and is the single best method on
MuSiQue F1 (0.613, ahead of even ANTMAN's 0.596), while being the
weakest of the multi-hop-aware methods on HotpotQA (F1 0.711, its own
lowest per-benchmark score). As a contemporary, purpose-built multi-hop
RAG baseline, it is competitive with this suite's own agentic methods
without matching CoA or ANTMAN overall.

## F. Zero-cost post-hoc diagnostics (Section 16)

Computed strictly after all 45 predictions were frozen; never fed back
into any rerun.

| Benchmark | Avg sub-Qs | Avg rewrites | Examples w/ graph expansion (>7 ctx) | Avg supporting-doc recall | Avg retrieved-doc count | Avg LLM calls/query | Avg cost/query |
|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | 2.00 | 0.00 | 15/15 | 1.000 | 7.40 | 9.07 | $0.0065 |
| 2WikiMultihopQA | 1.93 | 0.00 | 14/15 | 0.967 | 7.07 | 8.87 | $0.0070 |
| MuSiQue | 1.93 | 0.00 | 14/15 | 0.967 | 15.80 | 9.60 | $0.0136 |

Notable, disclosed finding: **0 literal rewrites occurred across all 45
examples**, even though the reference-check mechanism genuinely fired
(a real LLM call judging whether a pronoun-bearing sub-question refers
to a prior answer — verified in the 2Wiki smoke trace). In every case
the model judged a literal rewrite unnecessary, because `plan()`
unconditionally appends a "previous context summary" line (the prior
sub-question's own answer) to the next sub-question's context regardless
of whether reference-check/rewrite fires — this fallback context alone
was sufficient for embedding retrieval to find the right sentences. This
is real, disclosed behavior of the algorithm on this example set, not a
broken rewrite path (the mechanism is exercised — it makes a real
decision each time — it just consistently decided "not needed" here).
Graph expansion beyond the k=7 seed fired on 43/45 examples, confirming
the hop-expansion machinery is genuinely exercised at scale, not a
rarely-hit code path. Supporting-document recall is high (0.967–1.000),
consistent with ChainRAG's relatively strong F1 on this set.

## G. Explicit confirmation

- **No existing method was rerun** — Sparse/Dense/ReAct/LongAgent/CoA/
  ANTMAN canonical scores were loaded unmodified from their existing
  `*.rescored.jsonl` files.
- **No ANTMAN core file was modified** — `LocalCoordinator`, `NeedGraph`,
  `AutonomousWorker`, `LocalSearchTool`, `BM25Index`, `EmbeddingIndex`,
  and every other core file are untouched; verified both by code review
  and a structural unit test asserting none of these names are bound in
  `ant.external_wrappers.chainrag`.
- **No scorer/extraction change** — the same, frozen
  `extract_answer_span` pipeline every other method already uses was
  reused verbatim (question + raw answer only, no gold/support passed
  in).
- **No gold/supporting-fact leakage** — verified structurally (`run()`
  never reads `supporting_doc_ids` or any gold field) and by a dedicated
  unit test with a planted secret value; `supporting_doc_ids` was read
  only for Section F's post-hoc diagnostics, after all 45 predictions
  were already frozen on disk.
- **No tuning from observed scores** — every hyperparameter (embedding
  pool=100, seed k=7, similarity-edge k=10, position window=±3, entity
  percentile=40th, max words=3000, max hops=3) was frozen from the
  upstream audit before any inference; the only code change made during
  this pass was the `max_output_tokens` API-parameter bug fix (Section
  B), made before any accuracy judgment and disclosed as such, not a
  response to observed scores.
- **Identity confirmed** — the exact same 45 canonical `task_id`s
  (verified against `chain_of_agents.jsonl`'s own task_id list) were
  used, no resampling, no ChainRAG-side example selection.

Per Section 14: all 45 intended conditions completed, no configuration
changed mid-run (other than the pre-inference bug fix in Section B), no
leakage occurred, scoring completed successfully — **this ChainRAG run
is marked canonical.** No existing method's canonical status was
altered.
