# RULER: design note only (not implemented this pass)

Per Section 16 of the long-context evaluation spec, this is a design note
describing how the substrate built in this pass (`ant.evaluation_suite.
document_scope`, `ant.tools.document_tools`) could later support a
RULER-style (Hsieh et al. 2024) multi-needle-in-a-haystack / multi-hop
tracing evaluation. **No RULER task is implemented or run in this pass.**

## What RULER needs that the current substrate does not yet provide

RULER's core tasks (single/multi-needle retrieval, multi-hop tracing,
aggregation, QA-in-context) are synthetic: a needle (or several) is
inserted into a long, mostly-irrelevant "haystack" document at controlled
positions, and the model must retrieve or trace it. This is a different
construction primitive from anything this pass builds -- MuSiQue/HotpotQA/
2WikiMultihopQA are naturally-occurring multi-document QA, not synthetic
needle insertion.

## How the existing substrate maps onto RULER's needs

- **Haystack construction**: `DocumentRecord`/`materialize_documents`
  already handle "many documents, one ordered corpus, verbatim text" --
  the same shape a synthetic haystack needs. A RULER adapter would
  generate synthetic `DocumentRecord`s (or synthetic filler chunks
  interspersed within one long document) rather than loading them from a
  benchmark's own dataset.
- **Positional control**: `build_positional_variants` already solves
  exactly RULER's "needle depth" parameter for the single-needle case --
  splicing a designated "needle" document to an early/middle/late position
  is structurally the same operation as splicing a supporting document.
  Generalizing it to N arbitrary target positions (not just three fixed
  buckets) instead of three named ones is a small, mechanical extension.
- **Multi-needle / multi-hop tracing**: `chunk_documents`'
  deterministic, boundary-respecting chunking already gives stable
  `global_chunk_index` addressing, which a multi-needle task would use to
  place several needles at independently controlled chunk offsets and
  verify the model's answer references the correct combination.
- **Retrieval/search parity**: the same `search`/`view`/`navigate`
  primitives (`ant.tools.document_tools`, `ant.tools.local.
  LocalSearchTool`) that Retrieval/Matched-ReAct/ANT already share on real
  benchmarks would apply unchanged to a synthetic RULER haystack, since
  none of them depend on document content being naturally-occurring text.
- **ANT document adapter**: `_document_territories` (one territory per
  document) already generalizes to "one territory per synthetic haystack
  segment" with no change -- the territory-construction logic never reads
  benchmark-specific structure, only document boundaries.

## What a future RULER pass would still need to build

1. A synthetic haystack/needle generator (not present anywhere in this
   codebase) -- this is real, non-trivial implementation work, not a
   config flag.
2. A programmatic (not GPT-judge, not EM/F1) scorer for exact-needle-value
   recall, matching RULER's own official scoring convention.
3. A decision on context lengths to test (RULER's own headline results
   sweep up to 128K+ tokens) and which of this suite's 5 methods can
   meaningfully run at each length without hitting `MAX_CONTEXT_TOKENS`
   (`ant.agents.direct_document`) or LongAgent's own per-member chunk
   budget in a degenerate way.

None of this is implemented in this pass -- this note exists only so a
later RULER implementation starts from an explicit map of what's already
reusable, rather than re-deriving it.
