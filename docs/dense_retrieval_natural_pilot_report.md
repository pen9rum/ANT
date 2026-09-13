# Dense Retrieval baseline added to the preliminary natural multi-document QA pilot

Frozen baseline-completion run. No existing method rerun. No ANTMAN core
file modified. No scoring/extraction change. See `docs/
dense_retrieval_baseline_audit.md` for the full pre-implementation audit
(Part 1 of the governing spec).

## A. Implementation audit (summary — full detail in the audit doc)

- **Sparse Retrieval** (`src/ant/agents/retrieval_document.py`): BM25 +
  symbol-path exact-match, fused via Reciprocal Rank Fusion
  (`ant.tools.local.LocalSearchTool.search`), `limit=8`, iterative up to
  3 LLM-decided query-refinement rounds (`_TIER2_MAX_ROUNDS`), then
  `synthesize()` + `condense_to_answer_span()`. Variable call count
  (up to 5).
- **Dense Retrieval** (new: `src/ant/agents/dense_retrieval_document.py`):
  dense-ONLY, one-shot. **Critical finding from the audit:** the
  existing `dense_search()`/`build_embedding_index` path chunks at
  Python-symbol granularity (`build_symbol_index` skips every non-`.py`
  file) and would return an **empty index for every document-QA
  example** — not usable as-is. This new adapter instead reuses
  `DenseEmbedder`/`EmbeddingIndex` (frozen model/math, unmodified) with
  `ant.tools.local._retrieval_regions` for chunking — the exact same
  paragraph-aware block boundaries Sparse's own BM25 index already uses,
  giving byte-identical chunk boundaries, not merely a "closest
  equivalent." One embedding search (`limit=8`, same as Sparse), then
  the identical `synthesize()` + `condense_to_answer_span()` stage.
  Exactly 2 physical LLM calls, always (deterministic) — fewer than
  Sparse's variable up-to-5, because Dense is deliberately one-shot per
  the governing spec.
- **Embedding model:** `BAAI/bge-small-en-v1.5` (the repository's own
  frozen default, `ant.retrieval.dense.DEFAULT_EMBEDDING_MODEL`; no env
  override present; **not chosen for this pass** — it is the only
  embedding model configured anywhere in the repo). Dimension 384. Runs
  locally via `fastembed`/ONNX (confirmed installed, `fastembed==0.8.0`)
  — **zero API cost** for every embedding call.
- **top-k:** 8, identical to Sparse.
- **Chunking:** `_retrieval_regions` (paragraph-aware blocks, ≤~7
  non-blank lines, breaking at blank lines) — identical to Sparse.
- **Synthesis model:** GPT-4.1, same `synthesize()` method Sparse calls.

## B. Run status

- **45/45 conditions completed** (15 HotpotQA + 15 2WikiMultihopQA + 15
  MuSiQue), 0 errors, 0 retries.
- Preflight (before any paid call): all 45 examples verified against
  question hash + document-content hash + document count, matched
  against the already-canonical `direct_document.jsonl` rows for the
  same `task_id`s — 45/45 OK, 0 problems. (Full 45-row table generated;
  omitted here for length — see `.pytest_tmp/dense_preflight_result.json`
  from this session, or rerun `dense_preflight.py`'s logic to reproduce.)
- 10 pre-run unit tests (`tests/test_dense_retrieval_document.py`), using
  real local embeddings (free) and a mocked LLM boundary, all passed
  before any paid call — including a live behavioral proof that ranking
  favors a semantically-relevant, lexically-unrelated passage over a
  keyword-stuffed irrelevant one (rules out BM25 influence directly,
  not just by import inspection).
- Total cost: **$0.3133** (generation) + **$0.0289** (rescoring) =
  **$0.3422 total** — under the $2.00 hard cap.
- Wall-clock was notably slower than CoA's equivalent run (~25+ minutes
  for 45 examples) because each `run()` call instantiates a fresh
  `DenseEmbedder()` (reloading the ONNX model) rather than reusing one
  across calls — a real, disclosed inefficiency of this evaluation-only
  adapter's per-call design, not a correctness issue (confirmed via
  process CPU monitoring during the run: actively computing, not
  hung).
- Result paths: `output/runs/natural-multidoc-pilot/{hotpotqa,
  2wikimultihopqa,musique}/dense_retrieval_document.jsonl` (raw),
  `.rescored.jsonl` (extracted, canonical), `trajectories/
  dense_retrieval_document-*.json` (45 full traces, including retrieved
  chunk IDs and scores).
- Commit: see end of document.

## C. Main result table (extracted EM/F1)

| Benchmark | Metric | Direct | Sparse | Dense | ReAct | LongAgent | CoA | ANTMAN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| HotpotQA | EM | 0.667 | 0.600 | 0.600 | 0.533 | 0.533 | 0.600 | 0.600 |
| HotpotQA | F1 | 0.817 | 0.749 | 0.758 | 0.744 | 0.758 | 0.793 | 0.746 |
| 2WikiMultihopQA | EM | 0.600 | 0.467 | 0.467 | 0.467 | 0.733 | 0.667 | 0.800 |
| 2WikiMultihopQA | F1 | 0.689 | 0.519 | 0.479 | 0.737 | 0.853 | 0.820 | 0.812 |
| MuSiQue | EM | 0.467 | 0.467 | 0.200 | 0.400 | 0.267 | 0.533 | 0.467 |
| MuSiQue | F1 | 0.633 | 0.538 | 0.280 | 0.536 | 0.309 | 0.662 | 0.596 |

**Macro average (equal-weight across the 3 benchmarks, n=15 each):**

| Method | Avg EM | Avg F1 |
|---|---:|---:|
| Direct | 0.578 | 0.713 |
| Sparse Retrieval | 0.511 | 0.602 |
| **Dense Retrieval** | **0.422** | **0.506** |
| Matched ReAct | 0.467 | 0.673 |
| LongAgent | 0.511 | 0.640 |
| CoA | 0.600 | 0.758 |
| ANTMAN | 0.622 | 0.718 |

Dense Retrieval has the **lowest macro EM and macro F1 of all seven
methods** on this n=15/benchmark preliminary set, driven mainly by a
sharp MuSiQue drop (F1 0.280, roughly half of Sparse's 0.538 on the same
15 questions).

## D. Dense retrieval diagnostics by benchmark

Supporting-fact metadata used ONLY post-hoc, after predictions were
already frozen (never during retrieval/generation):

| Benchmark | Method | Supporting-doc recall@8 | All-required-docs-retrieved rate | Mean # supporting docs retrieved |
|---|---|---:|---:|---:|
| HotpotQA | Sparse | 0.900 | 0.800 | 1.80 |
| HotpotQA | **Dense** | **0.967** | **0.933** | 1.93 |
| 2WikiMultihopQA | Sparse | 0.883 | 0.667 | 2.20 |
| 2WikiMultihopQA | **Dense** | 0.833 | 0.667 | 2.00 |
| MuSiQue | Sparse | 0.800 | 0.600 | 1.60 |
| MuSiQue | **Dense** | **0.733** | **0.467** | 1.47 |

**Sparse/Dense retrieved-set overlap@k (audit only):**

| Benchmark | Mean overlap@k |
|---|---:|
| HotpotQA | 0.679 |
| 2WikiMultihopQA | 0.620 |
| MuSiQue | 0.517 |

Dense's recall pattern tracks its answer-quality pattern across
benchmarks: it has the *highest* recall of the two methods on HotpotQA
(0.967, where Dense also slightly out-F1's Sparse) and the *lowest* on
MuSiQue (0.733, where Dense's F1 gap is largest). Overlap@k is
moderate-to-low (0.52–0.68), confirming Sparse and Dense are retrieving
genuinely different (not identical) evidence sets, not just re-deriving
the same top-8 under different scores.

## E. Short interpretation (measured facts only)

- Dense Retrieval is competitive with Sparse Retrieval on HotpotQA
  (F1 0.758 vs 0.749, higher recall too) but substantially weaker on
  MuSiQue (F1 0.280 vs 0.538, lower recall too) — a benchmark-dependent
  result, not a uniform ranking.
- Dense has the lowest macro F1/EM of all seven methods in this table;
  this reflects the MuSiQue result dominating an equal-weight 3-benchmark
  average, not a uniform "worst on every benchmark" pattern.
- Recall@8 and answer F1 move together across benchmarks for Dense
  (highest recall + best relative F1 on HotpotQA, lowest recall + worst
  relative F1 on MuSiQue) — consistent with, though not proof of, recall
  being a real driver of the benchmark-to-benchmark F1 gap.
- Sparse and Dense retrieve meaningfully different evidence (overlap@k
  0.52–0.68, never near 1.0) — the two baselines are not redundant with
  each other.
- No superiority claim is made beyond what the numbers show: Dense is
  not shown to be better than Sparse in general, only better on
  HotpotQA specifically and substantially worse on MuSiQue specifically,
  in this n=15/benchmark preliminary set.

## F. Explicit confirmation

- **No existing method was rerun** — Direct/Sparse/ReAct/LongAgent/
  CoA/ANTMAN canonical scores were loaded unmodified from their existing
  `*.rescored.jsonl` files.
- **No ANTMAN core file was modified** — `dense_search()`,
  `build_embedding_index`, `build_symbol_index`, `LocalSearchTool`,
  `LocalCoordinator`, and every other core file are untouched; the new
  adapter is a standalone file reusing only `DenseEmbedder`/
  `EmbeddingIndex`/`_retrieval_regions` by import, unmodified.
- **No scorer/extraction change** — the same `condense_to_answer_span`
  and `extract_answer_span` pipeline every other method already uses was
  reused verbatim.
- **No gold/supporting-fact leakage** — verified both structurally
  (`DenseRetrievalDocumentAgent.run()` never reads `supporting_doc_ids`
  or any gold field; confirmed by a dedicated unit test with a planted
  secret value) and by construction (supporting-fact metadata was read
  for Section D's diagnostics only, after all 45 predictions were
  already frozen on disk).
- **No tuning from observed scores** — the embedding model, chunking
  function, and top-k were all fixed from auditing the existing
  repository configuration (Part 1/Section 4), before any prediction was
  generated; no value here was chosen or adjusted based on this run's
  own EM/F1 results.

Per Section 14: all 45 intended conditions completed, no configuration
changed mid-run, no leakage occurred, scoring completed successfully —
**this Dense Retrieval run is marked canonical.** No existing method's
canonical status was altered.
