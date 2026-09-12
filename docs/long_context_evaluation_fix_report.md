# Long-context evaluation fix report: answer-extraction scoring + filler source-overlap/leakage

Evaluation/infrastructure fixes only, per the governing spec. **Not
touched:** ANT core coordination, worker routing, Need Graph logic,
search behavior, retrieval/indexing behavior, LongAgent, Matched ReAct,
method prompts, tool budgets. This report is the required 9-section
summary of that pass.

---

## 1. Existing answer-contract audit

Performed by reading real code and a real live trajectory before writing
anything new, not inferred from function names.

- `ant.evaluation_suite.answer_contract.condense_to_answer_span` IS
  already applied, identically, as the last step of all five
  document-track methods' own `run()`. Confirmed both by reading each
  agent's source and by inspecting a live trajectory
  (`output/runs/natural-multidoc-pilot/hotpotqa/trajectories/
  ant_document-5a8b57f25542995d1e6f1371.json`): `raw_answer_before_
  condensation` held an ~1800-character multi-paragraph analysis;
  `final_answer` (post-condensation) held `"Yes, both were American."`
  — condensation clearly ran and clearly shortened the answer.
- It is LLM-based (one `responses_text` call per answer), not
  deterministic.
- Every benchmark adapter's own `score()` (`score_hotpot_style`,
  `NiahPlusAdapter.score`, `MuSiQueAdapter.score`) calls
  `score_qa(result.final_answer, ground_truths)` — scoring already used
  the POST-condensation answer, not raw pre-condensation text. Confirmed
  by direct code inspection.
- **Why verbose examples survived anyway:** `condense_to_answer_span`'s
  own prompt instructs the model to "extract OR RESTATE" the minimal
  answer span. The word "restate" explicitly permits a rewritten
  grammatical sentence rather than a literal minimal substring. The
  model's own notion of "minimal" stopped at a complete sentence
  (`"Yes, both were American."`) rather than the bare token (`"yes"`)
  HotpotQA's own gold convention expects. This is the root cause the new
  layer below fixes.

`condense_to_answer_span` itself was **not modified** (frozen, per
instruction — it is method-generation-time code, out of scope for an
evaluation-side fix). The fix is a separate, later, scoring-time-only
layer.

## 2. New answer extraction contract

`ant.evaluation_suite.answer_extraction.extract_answer_span(question,
raw_model_answer)` — applied only by a rescoring script over
already-saved predictions, never during generation, never rerunning any
agent.

- **Inputs:** `question` and `raw_model_answer` only. No gold answer,
  aliases, supporting facts, doc labels, needle locations, method
  identity, benchmark-specific metadata, or score information is ever
  passed in or read.
- **Deterministic path (preferred, zero cost):** canonical yes/no
  normalization only — `^\s*(yes|no)\b` (case-insensitive) →
  `"yes"`/`"no"`. Nothing else is rewritten deterministically.
- **LLM path (fallback):** one frozen prompt, `gpt-4.1`, `temperature=0`
  (via a `_ZeroTemperatureProvider(CountingOpenAIProvider)` subclass that
  overrides `_responses_kwargs` — the same subclassing pattern
  `CountingOpenAIProvider` itself already uses — without touching frozen
  `ant/providers/openai_provider.py` at all), structured JSON output
  (`{"extracted_span": "..."}`), `max_output_tokens=128`. The call is
  logged and its tokens/cost counted separately from every other
  generation call (via `CountingOpenAIProvider`'s own counters).
- **Hard safety gate:** except for yes/no normalization, the extracted
  text MUST be a verbatim, whitespace/case-insensitive substring of the
  raw answer (`_is_verbatim_substring`). Any violation — a parse failure,
  an invented entity, a "repaired" wrong answer, anything not literally
  present in the raw text — is rejected and the ORIGINAL raw answer is
  returned unchanged (`rejected_hallucination=True`, but the physical
  call is still logged/costed).
- **Never tuned on ANT's own scores.** The prompt/logic was frozen after
  writing 11 unit tests (`tests/test_answer_extraction.py`, covering all
  7 required cases A–G) and one small live validation (3 real calls,
  ~$0.0005), and was not touched afterward based on how the full
  rescoring pass turned out.

## 3. Re-scored natural QA table (raw vs. extracted, all 5 methods × 3 benchmarks)

Computed by rescoring the existing, already-saved 225 natural-pilot
predictions (no rerun) — 15 tasks/benchmark × 5 methods × 3 benchmarks.

| Benchmark | Method | raw EM | raw F1 | extracted EM | extracted F1 |
|---|---|---|---|---|---|
| HotpotQA | ant_document | 0.400 | 0.601 | **0.533** | **0.741** |
| HotpotQA | direct_document | 0.667 | 0.817 | 0.600 | 0.800 |
| HotpotQA | longagent | 0.533 | 0.758 | 0.467 | 0.730 |
| HotpotQA | matched_react_document | 0.467 | 0.704 | 0.533 | 0.744 |
| HotpotQA | retrieval_document | 0.400 | 0.597 | 0.533 | 0.721 |
| 2WikiMultihopQA | ant_document | 0.733 | 0.760 | 0.800 | 0.833 |
| 2WikiMultihopQA | direct_document | 0.600 | 0.678 | 0.600 | 0.689 |
| 2WikiMultihopQA | longagent | 0.733 | 0.842 | 0.733 | 0.853 |
| 2WikiMultihopQA | matched_react_document | 0.400 | 0.685 | 0.467 | 0.737 |
| 2WikiMultihopQA | retrieval_document | 0.467 | 0.501 | 0.467 | 0.518 |
| MuSiQue | ant_document | 0.467 | 0.596 | 0.467 | 0.596 |
| MuSiQue | direct_document | 0.467 | 0.633 | 0.467 | 0.633 |
| MuSiQue | longagent | 0.267 | 0.309 | 0.267 | 0.329 |
| MuSiQue | matched_react_document | 0.400 | 0.536 | 0.400 | 0.536 |
| MuSiQue | retrieval_document | 0.467 | 0.538 | 0.467 | 0.564 |

Extracted EM/F1 is now the primary metric; raw remains an audit column.
Every one of these 15 rows is one call of the SAME, method-agnostic
extractor — no method-specific tuning. Full per-row data (including the
supplementary rescored `multineedle-scaling`/`single-needle-scaling`/
`contamination-study` tables, 345 additional rows) is in the sibling
`*.rescored.jsonl` files next to each original `output/runs/**/*.jsonl`.

## 4. Score-change audit

Across all 570 rescored rows (225 natural pilot + 225 multineedle-scaling
+ 90 single-needle-scaling + 30 contamination-study): **0 rows had
`rejected_hallucination=True`** — the extractor never proposed a
non-substring span on real data (the hard safety gate was never needed
to intervene on this data, though it remains active). 457/570 rows used
the LLM path; 113/570 resolved deterministically (yes/no).

**134 rows changed by ≥0.3 F1** (saved to
`output/runs/answer_extraction_sanity_audit_delta_ge_0.3.json`, with gold
answers included there ONLY for this post-hoc audit):

- **84 improved.** Overwhelmingly the exact verbose-but-correct pattern
  the fix targets: `"Yes, both were American."` → `"yes"` (raw F1 0.40 →
  1.0), `"No, the Laleli Mosque is in Laleli and the Esma Sultan Mansion
  is in Ortaköy."` → `"no"` (raw F1 0.14 → 1.0), `"United States
  ambassador to Ghana, United States ambassador to Czechoslovakia, Chief
  of Protocol of the United States"` → `"Chief of Protocol of the United
  States"` (raw F1 0.32 → 0.67). None of these changed factual content —
  every one is a formatting fix (a full sentence/list shortened to the
  span the question actually asked for).
- **50 got worse**, and this is reported honestly rather than buried:
  - **48/50** are one systematic pattern: gold expects the full span
    `"Greenwich Village, New York City"`; several methods happened to
    already produce exactly that string verbatim (raw EM=1.0, raw
    F1=1.0); the extractor judged `"Greenwich Village"` alone to be the
    "shortest span that answers the question" and shortened a
    already-correct answer past the point the benchmark's own gold
    convention wanted (extracted F1 drops to 0.57, EM to 0). This is a
    real, quantified limitation of the extractor's "shortest span"
    heuristic on multi-part gold answers, not a hallucination and not a
    factual change — `"Greenwich Village"` remains true, just
    incomplete relative to gold's exact string.
  - **1/50**: from a raw answer listing three true facts about the same
    person (`"United States Ambassador to Ghana; United States
    Ambassador to Czechoslovakia; Chief of Protocol of the United
    States"`, gold = `"Chief of Protocol"`), the extractor picked
    `"United States Ambassador to Ghana"` — a different, also-true item
    from the same list, not the gold-matching one. A genuine
    extraction-precision miss among multiple valid candidates (case D of
    the required test matrix), not an invention and not a repair of a
    false claim.
  - **1/50**: in a counterfactualized single-needle example, the
    extractor dropped a parenthetical restating the substituted entity
    (`"the Main Building (also referred to as Ossenfer Malquorin)"` →
    `"the Main Building"`), losing the appositive gold expected intact.
- **Confirmed: no case changed factual content**, invented an entity, or
  "repaired" a wrong raw answer into a different, more-correct one (test
  case E's guarantee held on all 570 real rows, matching the unit-test
  guarantee). No STOP condition from Step 7 was triggered. Net effect
  across all 570 rows is a clear, substantial improvement (see Section 3
  and the per-method aggregate in the sibling `.rescored.jsonl` files),
  with the 50-row over-shortening pattern documented as a known,
  quantified trade-off for future work (e.g. tightening the extraction
  prompt to prefer the longest span still satisfying "shortest that
  fully answers" when gold conventions expect a compound span) — not
  something this frozen pass tuned away after seeing scores.

## 5. Filler leakage audit (old, pre-fix construction, reconstructed live)

The original controlled runs did not persist their materialized filler
documents (only condition specs — `third_party/manifests/long_context/
{multineedle_scaling,single_needle_scaling,contamination_study}
_manifest.json` — and generation outputs were saved). Construction is
fully deterministic (fixed `NIAH_PLUS_SEED`, no run-order/score
dependence), so the old, pre-fix filler pool was reconstructed exactly
by replaying every one of the 67 frozen manifest conditions through the
now-fixed builders and reading off `excluded_source_overlap_count`/
`excluded_answer_leakage_count` — i.e., counting how many of the
old-code's actual filler candidates the new rule would have (and now
does) exclude:

| Experiment | Construction | Conditions audited | Source-overlap exclusions | Answer-leakage exclusions |
|---|---|---|---|---|
| multineedle-scaling | multi-needle, HotpotQA | 45 | 0 | **63** (avg 1.4/condition) |
| single-needle-scaling | Condition C, SQuAD | 18 | **4,824** (avg 268/condition) | 0 |
| contamination-study, Condition A | SQuAD | 2 | **536** (avg 268/condition) | 0 |
| contamination-study, Condition C | SQuAD | 2 | **536** (avg 268/condition) | 0 |
| **Total** | | **67** | **5,896** | **63** |

(Condition B of the contamination study is a no-context memorization
probe with no filler documents at all — `ant.evaluation_suite.
answer_contract` — and is unaffected by this fix.)

**Single-needle (SQuAD) was dominated by source-overlap**: SQuAD packs
many distinct questions per article, so ~268 filler candidates per
instance (out of a ~400-row pool) shared a source article with the
needle under the old `row["id"] != needle_row["id"]`-only check — this
matches the concrete, previously-confirmed case where a real materialized
haystack's `doc_0014.txt` contained the needle's own un-substituted
sentence. **Multi-needle (HotpotQA) was dominated by direct
answer-string leakage instead** (63 instances, ~1.4/condition) — source
overlap was actually zero in this data (HotpotQA's per-question document
sets rarely repeated the same source title across the sampled conditions
here), but individual filler passages did sometimes contain the literal
gold-answer string.

## 6. Clean construction validation (live, zero-cost, real data)

Built fresh real instances (no LLM calls, network reads only) with the
fixed construction and validated directly against the real materialized
documents:

- **Source overlap = 0** across 9 fresh instances (3 single-needle
  Condition A, 3 fully-counterfactualized Condition C, 3 multi-needle),
  each checked against its own `needle_source_titles`.
- **Prohibited leakage = 0**: no non-needle document contains the gold
  answer, the original pre-substitution answer, or (for Condition C) any
  substituted entity string, checked with the same word-boundary rule
  the construction itself uses.
- **Context length preserved**: actual/target ratio 1.001–1.022 across
  all 9 instances (target 8000 tokens) — well within the existing
  whole-document tolerance, unaffected by exclusion removing ~65-70% of
  the raw SQuAD filler pool.
- **Early/Middle/Late still correct**: relative needle depth
  early=0.0 < middle≈0.49 < late=1.0 (single-needle, question_index 0).
- **Paired-length prefix-compatibility preserved**: a 4000-token and an
  8000-token instance (same question_index, "early" position) share an
  exact-text-prefix filler sequence (`long[: len(short)] == short`),
  confirming the paired 32K/64K/128K construction property survives the
  fix (filler order is seeded independently of context length; a longer
  target just keeps drawing from the same deterministic order).
- **No gold/support metadata reaches `EvalDocumentEnvironment`**:
  structurally confirmed — `DocumentRecord` has only `doc_id`/`title`/
  `text` fields (no gold/support field exists on the type at all), and
  `EvalDocumentEnvironment.__init__` only accepts `documents:
  list[DocumentRecord]`, with no parameter through which
  `NiahPlusExample.metadata` could ever reach it.

All 7 of these (A–G) are now also permanent regression tests in
`tests/test_niah_plus.py` (10 new tests), exercised under a
leakage-heavy synthetic fixture specifically designed to stress the
exclusion logic, not just the easy case.

## 7. Canonicality decision

- **Still canonical (documents unaffected by Part B):** the natural
  multi-document pilot's document construction (HotpotQA/2WikiMultihopQA/
  MuSiQue — real benchmark documents, no synthetic filler). Its
  *reported EM/F1 numbers* in prior docs are superseded by Section 3's
  extracted-EM/F1 table (Part A), not because the documents changed, but
  because the scoring metric improved.
- **PRE-FIX / NON-CANONICAL** for any needle-retrieval/position/scaling
  claim, effective immediately: `multineedle-scaling` (all 225
  generations), `single-needle-scaling` (all 90 generations),
  `contamination-study` Conditions A and C (24 of 30 generations;
  Condition B unaffected). Any Early/Middle/Late positional analysis or
  32K/64K/128K scaling interpretation built on these — including
  `docs/long_context_hotpot_audit_and_expansion_report.md` Section C's
  "128K position trace audit" — is marked non-canonical for that purpose.
  Canonicality banners were added to the top of `docs/
  long_context_final_report.md`, `docs/long_context_decision_memo.md`,
  and `docs/long_context_hotpot_audit_and_expansion_report.md` pointing
  here; nothing in those files was deleted or rewritten.
- **Must be rerun** before any needle-retrieval/position/scaling claim
  can be made canonically again: `multineedle-scaling`,
  `single-needle-scaling`, and `contamination-study` Conditions A/C,
  against the now-fixed construction (Section 8).

## 8. Rerun proposal (not executed this pass)

| Experiment | Generations | Historical actual cost (pre-fix run) | Historical wall-clock (sequential sum) |
|---|---|---|---|
| multineedle-scaling | 225 (5 methods × 45 conditions) | $21.82 | 0.89 h |
| single-needle-scaling | 90 (5 methods × 18 conditions) | $12.45 | 0.46 h |
| contamination-study (A+C only; B unaffected) | 20 (5 methods × 4 conditions) | ~$1.63 (proportional share of the $2.44 total 30-gen cost) | ~0.06 h |
| **Total** | **335** | **~$35.90** | **~1.4 h sequential** |

- **Cost basis:** real, measured per-row cost from the existing
  `output/runs/{multineedle-scaling,single-needle-scaling,
  contamination-study}/*.jsonl` (the `cost` field each generation already
  logs), summed per experiment. The fixed construction changes which
  filler documents are chosen, not their approximate token volume
  (Section 6: context length preserved within ~2%), so per-generation
  cost should land close to these historical figures — a **20% margin**
  for retries/variance gives a planning estimate of **~$43**, comfortably
  inside typical remaining budget for this study (see `docs/
  long_context_budget_ledger.md`).
- **Wall-clock:** ~1.4 hours if run strictly sequentially in one process
  (sum of historical per-row wall-clock times). `ant.evaluation_suite.
  runner.run_suite` has no built-in concurrency, but prior passes in this
  session ran different methods as separate parallel background
  processes; under that pattern, wall-clock is closer to the
  slowest single method's own total (longagent: ~14 min for
  multineedle-scaling, ~6 min for single-needle-scaling) — realistically
  **20–35 minutes** if launched as parallel per-method background runs.
- **What would change operationally:** none of the manifests need
  hand-editing — `build_single_needle_instance`/
  `build_fully_counterfactualized_single_needle_instance`/
  `build_multi_needle_instance` now produce cleaned documents
  automatically for the exact same `(context_length_tokens, position,
  question_index)` triples already frozen in the three manifest files;
  the rerun is "regenerate against the same conditions, fixed
  construction," not "redesign the experiment."
- **Not executed in this pass**, per explicit instruction — this is a
  proposal only, pending a separate go-ahead.

## 9. Integrity confirmation

- ANT core coordination, worker routing, Need Graph logic, search
  behavior, retrieval/indexing behavior, LongAgent, Matched ReAct: **not
  modified.**
- No method prompt was changed. `condense_to_answer_span` (generation-time
  answer condensation) is untouched and unmodified.
- No score-driven tuning: the extractor's prompt/logic was frozen after
  writing the 11 unit tests and one small live validation call, before
  the full 570-row rescoring pass ran; it was not adjusted afterward
  based on the rescoring results (including the 50-row over-shortening
  pattern documented honestly in Section 4, left as-is rather than
  patched post hoc).
- No method-specific evaluation rule: `extract_answer_span` is called
  identically for all five methods, with no method identity ever passed
  in or branched on.
- All five methods use the same shared answer extractor
  (`ant.evaluation_suite.answer_extraction.extract_answer_span`), called
  once per prediction row regardless of which method produced it.
- No gold/support metadata is exposed at inference time: `documents`
  passed into every method remain plain `DocumentRecord` (doc_id/title/
  text only, structurally — Section 6); the answer extractor itself
  receives only `question`/`raw_model_answer`, never gold answers,
  aliases, or needle locations (Section 2); construction-time leakage
  checks (Section 5/6) are checked and excluded before any document ever
  reaches `EvalDocumentEnvironment`, never after.

**Tests/lint/type checks:** full suite 622/622 passed
(`.venv/Scripts/python -m pytest tests/`), `ruff check`/`ruff format
--check` clean on all new/modified files, `pyright` clean on
`answer_extraction.py`/`niah_plus.py` (0 errors) — the 49 pre-existing
`pyright` errors project-wide are in unrelated test files
(`test_no_evolution_runtime.py`, `test_worker_autonomy.py`, stub
`WorkerReasoner` protocol mismatches) not touched by this pass.
