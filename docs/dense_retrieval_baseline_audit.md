# Dense Retrieval baseline: pre-implementation audit

Performed by reading the real code before writing anything, per the
governing spec's Part 1.

## A. Current Sparse Retrieval baseline

- **File:** `src/ant/agents/retrieval_document.py` (`RetrievalDocumentAgent`,
  registered as `retrieval_document`).
- **Search primitive:** `ant.tools.local.LocalSearchTool.search()` --
  BM25 (`_bm25_channel_rank`) fused with a symbol/filename-stem exact-match
  channel (`_symbol_path_channel_rank`) via Reciprocal Rank Fusion
  (`_reciprocal_rank_fusion`). No embedding involved anywhere in this path.
- **Indexing method:** `_territory_index` builds one `BM25Index` over
  every file's `_retrieval_regions`-chunked blocks (paragraph-aware:
  non-blank lines grouped up to ~7 lines, breaking at blank lines or
  definition boundaries) across the whole file scope -- cached per
  sorted file-set, rebuilt from scratch per distinct scope (no
  cross-example persistence in the document-QA track, since each
  question gets its own `environment_root`).
- **top-k / limit:** `limit=8` (`search_tool.search(query, all_files,
  limit=8)`).
- **Synthesis:** `OpenAIProvider.synthesize(question=..., evidence=...)`
  -- ANT core's own shared answer-generation method (same one other
  methods use), given the full (untruncated) evidence list.
- **Number of LLM calls:** variable, up to 5 -- up to `_TIER2_MAX_ROUNDS
  = 3` round-decision calls (`responses_json` against `_TIER2_QUERY_PROMPT`,
  each deciding `"enough"` or a `"next_query"`), + 1 `synthesize()` call,
  + 1 `condense_to_answer_span()` call. Fewer if the model declares
  "enough" before round 3.
- **Rounds:** **multiple, iterative** -- an LLM decides after each
  search call whether to search again with a refined query or stop.
  This is the key structural difference from the mandated Dense design
  (Section 6: one-shot, no iteration) -- disclosed, not silently matched
  (see Dense Retrieval definition below).

## B. Existing dense infrastructure

- **File:** `src/ant/retrieval/dense.py` (`DenseEmbedder`, `EmbeddingEntry`,
  `EmbeddingIndex`, `build_embedding_index`, `get_shared_embedder`).
  Consumed by `ant.tools.local.LocalSearchTool.dense_search()`.
- **Embedding model:** `DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"`
  (overridable only via the `ANT_EMBEDDING_MODEL` env var -- confirmed
  **not set** in this environment, so the frozen default is what's
  actually in effect). Verified live: `DenseEmbedder().model_name ==
  "BAAI/bge-small-en-v1.5"`, embedding dimension **384**.
- **Runs locally:** yes, via `fastembed` (ONNX runtime) -- confirmed
  installed (`fastembed==0.8.0`) in this environment. **Zero API cost**
  for every embedding call; only answer generation/extraction costs
  anything.
- **Chunk granularity (existing `build_embedding_index` path):** one
  embedding per **Python symbol** (class/function), via
  `ant.tools.symbol_index.build_symbol_index` -- an AST-based extractor.
- **CRITICAL FINDING:** `build_symbol_index` contains `if not
  relative.endswith(".py"): continue` -- it unconditionally skips every
  non-Python file. Document-QA environments are `doc_0000.txt`,
  `doc_0001.txt`, ... Confirmed directly: this produces **zero symbols,
  hence an empty embedding index**, for any document-QA environment.
  **`dense_search()` is therefore NOT usable as-is on this substrate --
  it would return `[]` for every query.** It is not "already exposed as
  a standalone baseline" for documents; as implemented today it isn't
  functional here at all, only for code repositories.
- **Normalization:** L2-normalized rows (`vectors / norms`, zero-norm
  guarded to 1.0), confirmed in `_embed_entries`.
- **Similarity function:** cosine similarity via normalized dot product
  (`pool @ query`, `EmbeddingIndex.search`).
- **Index persistence:** `.entries.json` (metadata) + `.vectors.npy`
  (float32 array), written to temp names and atomically replaced.
- **Query encoding:** `embedder.embed([query])` on the **raw** query
  text -- no BGE-style instruction/prefix is applied to queries anywhere
  in the existing code (confirmed in `dense_search()`); this baseline
  matches that exactly, rather than introducing an optimization ANTMAN's
  own dense path doesn't already use.
- **Result limit:** caller-specified (`limit` parameter on `search()`).
- **Standalone baseline status:** dense search currently exists **only**
  inside `LocalSearchTool`, called by `AutonomousWorker`'s own tool
  choices during repo-QA tasks. It has never been exposed as a
  standalone, non-agentic baseline anywhere in `src/ant/agents/` or
  `src/ant/external_wrappers/`, and (per the finding above) could not
  have been used as one for documents without new chunking code.

## Resulting design decision (frozen before any paid call)

A new, evaluation-only adapter,
`src/ant/agents/dense_retrieval_document.py`
(`DenseRetrievalDocumentAgent`, registered as `dense_retrieval_document`),
was written. It:

- Reuses `DenseEmbedder` and `EmbeddingIndex` **unmodified** from
  `ant.retrieval.dense` (frozen model, frozen similarity math).
- Reuses `ant.tools.local._retrieval_regions` **unmodified** -- the
  exact same paragraph-aware chunker Sparse Retrieval's own BM25 index
  already uses for every file (it has no `.py`-only restriction) --
  giving Dense Retrieval **byte-identical chunk boundaries** to Sparse,
  not merely a "closest equivalent."
- Builds a fresh, **in-memory-only** index per question, scoped to that
  question's own document set. Never touches ANTMAN's persistent,
  symbol-only `.ant/dense/` cache, never shares state across examples.
- Performs exactly **one** embedding search (`limit=8`, same as
  Sparse), then the same `synthesize()` call, the same Condition-B
  regrounding hook, and the same `condense_to_answer_span()` contract
  every other document-track method uses.
- Makes **no BM25 call, no RRF fusion, no sparse fallback, no iterative
  refinement, no query rewriting, no reranking** -- exactly the
  dense-only, one-shot design Section 3/6 require.
- Physical LLM call count: **exactly 2, always** (1 `synthesize()` + 1
  `condense_to_answer_span()`) -- deterministic, unlike Sparse's
  variable up-to-5. This asymmetry (iterative Sparse vs. one-shot Dense)
  is intentional per Section 6's own explicit instruction and is
  reported, not hidden.
- No ANTMAN core file was modified.
