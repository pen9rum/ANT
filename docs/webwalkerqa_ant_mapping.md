# WebWalkerQA: benchmark audit and ANT-mapping design

Status: **design document only**. No ANT code (core or evaluation-suite)
is modified by this document. No Track B implementation has started.
Written before any implementation, per Stage 6's own instruction: "Do not
assume [a territory-analogue mapping] is correct without auditing the
benchmark. ... Do not modify ANT core yet. First write the mapping/design."

All benchmark facts below are sourced from a dedicated research pass that
directly fetched and quoted the primary paper (arXiv:2501.07572, ACL 2025,
`aclanthology.org/2025.acl-long.508/`) and the official repo (confirmed
live via HTTP redirect: `github.com/Alibaba-NLP/WebWalker` now redirects
to `github.com/Alibaba-NLP/DeepResearch`, where WebWalker lives at
`WebAgent/WebWalker/`).

## 1. Benchmark structure

- 680 human-verified question-answer pairs across four domains:
  Conference (24.0%), Organization (7.9%), Education (46.3%), Game
  (24.0%). Bilingual (Chinese 60.5%, English 39.5%).
- Two question types: **single-source** (answerable from one page, depth
  2-4) and **multi-source** (requires combining info across pages,
  combined depth 2-8), each stratified Easy/Medium/Hard.
- A task is: a root URL `U_root` plus a query `Q`. The agent must answer
  by exploring **only within that website** -- not open web search, not
  BrowseComp-Plus-style corpus retrieval.

## 2. Available navigation primitives

**Purely click-based, no search function.** At each step the agent
observes `(pt, lt)`: current page content `pt` (markdown-extracted via
BeautifulSoup) plus a set of clickable sublinks `lt`. The agent picks one
link to click next. No formal `click(url)`/`search(query)`/`answer(text)`
API is specified in the paper -- actions are described narratively. **Hard
ceiling: at most 15 exploration steps** (the paper's own explorer-agent
budget).

## 3. Scoring

**LLM-judged (GPT-4 in the original paper), not exact-match** -- the paper
explicitly states exact-match was rejected as infeasible given variable
answer lengths, using CoT-prompted comparison against the ground truth
instead. Per this suite's own judge policy (`docs/evaluation_plan.md`),
the JUDGE MODEL will be standardized to this suite's own centralized
GPT-5 (not WebWalkerQA's own original GPT-4 judge) when this track is
actually implemented -- the SCORING SEMANTICS (LLM-judged correctness
comparison against a reference answer) are preserved unchanged; only the
judge model is standardized, per this suite's own stated fairness policy.

## 4. Does a hierarchical/structural search space exist?

**The paper does not formally model sites as sitemaps/directory trees.**
It notes qualitatively that its four domains were chosen because their
pages have "rich clickable content, offering substantial depth for
exploration" and that institutional sites have "more structured" paths
via clickable buttons -- suggestive of implicit hierarchy, but there is
**no explicit sitemap/tree API exposed to the agent, no directory-listing
primitive**, and no analogue to a repository's own file-tree listing.
Navigation is link-following/associative at each step; only the
benchmark's own difficulty labels (depth 2-8) encode structure, and that
encoding is never revealed to the agent itself.

This is the central disanalogy with Track A: a code repository's
directory/file/symbol structure is a real, queryable, pre-existing
artifact (`EvalRepoEnvironment.iter_files()`, `discover_territories()`);
a website's link graph is discovered incrementally, one click at a time,
with no equivalent pre-existing structural listing available up front.

## 5. How WebWalker itself decomposes exploration

Two roles, loop-coordinated (not a fixed pipeline, not a fixed number of
roles beyond two):

- **Explorer**: a Thought-Action-Observation ReAct loop that clicks links.
- **Critic**: invoked after each explorer step; maintains/updates an
  incremental memory of accumulated relevant information and judges
  whether it's sufficient to answer. If sufficient, terminates the loop
  and produces the answer; otherwise the explorer continues (hard cap: 15
  steps regardless).
- The memory-update mechanics themselves are **not specified at
  implementation detail** in the paper text -- described only narratively
  as "accumulating relevant observations." Flagged explicitly as a gap in
  the primary source, not filled in with an invented mechanism here.
- Explicit design motivation: managing the "potentially large size" of
  accumulated history -- i.e. WebWalker's own Critic role is ITSELF a
  rudimentary context-management mechanism, directly comparable in
  *purpose* (not mechanism) to SLIM's periodic summarization and PACE's
  multi-granularity compression (see `docs/web_track_baselines.md`).

## 6. Candidate ANT mapping -- audited, not assumed correct

The candidate abstraction proposed before this audit was:

> website/domain -> sections/navigation subtrees/topic regions -> specialized workers

Having now audited the benchmark, this mapping requires real qualification
rather than direct adoption:

- **"Territory" (Track A: a pre-existing, statically-discoverable file
  set a worker owns) has no direct Track B equivalent.** There is no
  pre-existing structural listing of a website's own sections the way
  `discover_territories()` gets a real directory tree up front. A Track B
  "territory" would have to be **runtime-discovered and provisional**:
  e.g., the set of pages reachable from one first-level link off the root,
  expanded incrementally as the agent clicks -- structurally closer to
  ANT's own runtime Need Graph revision than to its static territory
  discovery. This is a genuine architectural difference to design for
  explicitly, not paper over by reusing the word "territory" for something
  that behaves differently.
- **What a "worker" would own**: plausibly, one first-level section/link
  off the root URL (mirroring the paper's own domain examples, e.g. a
  conference site's "Schedule" vs. "Speakers" vs. "Venue" sections) --
  but this is a DESIGN HYPOTHESIS, not verified against the benchmark's
  own data (the paper doesn't label pages by "section" in a way this
  audit found), and would need validation against real WebWalkerQA
  examples before being trusted for anything beyond a first implementation
  attempt.
- **What a Need Graph node would mean**: a sub-question or information
  need the current page/section doesn't yet resolve (e.g. "find the
  keynote speaker's affiliation," discovered only after loading the
  relevant page) -- this maps more cleanly, since Track A's own Need Graph
  nodes are already runtime-discovered from evidence gaps, not
  pre-existing structure; Track B's navigation is ALREADY runtime-only, so
  this part of ANT's architecture is arguably a more natural fit for
  Track B than territory discovery is.
- **What "progress" / "local exhaustion" would mean**: progress = new,
  previously-unseen relevant content found on a newly-clicked page;
  exhaustion = a worker (section) has clicked through all of its own
  reachable links (or hit some local step budget) without finding
  anything new relevant to its assigned need -- a workable analogue to
  Track A's own file-exhaustion-based recovery, though the "how many
  clicks constitutes exhaustion" threshold is a new parameter with no
  Track A precedent to copy from.

**Conclusion**: the proposed mapping is directionally plausible but NOT
validated against real WebWalkerQA data in this pass, and the
"territory" concept specifically needs to become runtime-provisional
rather than statically discovered -- a real architectural adaptation, not
a drop-in relabeling. This is exactly the kind of design decision Stage 6
asked to be surfaced before any implementation, not decided unilaterally
here. No ANT code changes follow from this document.

## 7. Comparison to BrowseComp-Plus (why it's a genuinely different shape)

BrowseComp-Plus evaluates against a **fixed, curated ~100K-document
corpus**, not the live web -- explicitly designed to isolate retriever
quality from agent reasoning quality (avoiding live-search-API
non-reproducibility). Scored by a fixed judge (Qwen3-32B in the original
paper; standardized to this suite's own GPT-5 when implemented) plus a
secondary IR-metrics track (recall, nDCG) against labeled relevance
judgments. This is **flat search-and-retrieve, no site hierarchy or
traversal concept at all** -- structurally closer to Retrieval/Matched
ReAct's own existing Track A shape (a bounded evidence pool to search)
than to WebWalkerQA's link-traversal shape. Any ANT mapping for
BrowseComp-Plus specifically would look more like Track A's own
territory-over-a-fixed-corpus pattern than the runtime-discovery pattern
WebWalkerQA needs -- these two Track B benchmarks likely need **different**
ANT-mapping designs, not one shared design; this document covers
WebWalkerQA only, per Stage 6's own scope.
