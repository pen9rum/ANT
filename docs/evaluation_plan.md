# Evaluation plan: two tracks

Status: frozen planning document, written before any formal benchmark run
on either track. Describes the current accepted structure; supersedes any
earlier single-track framing. If this document and the actual code/
manifests ever disagree, that is a bug in one of the two, not an
ambiguity to interpret away.

## Why two tracks

We want to test ANT on two structurally distinct information-seeking
settings before drawing any generalization claim:

1. **Repository information seeking** (Track A) -- a codebase with a real
   directory/file/symbol structure, where ANT's own territory/worker
   decomposition has an obvious structural analogue to exploit.
2. **Structured web/deep information seeking** (Track B) -- websites with
   link-graph navigation, a genuinely different structural shape (see
   `docs/webwalkerqa_ant_mapping.md` for whether/how ANT's own
   territory/Need-Graph concepts map onto this at all -- not assumed).

Additional heterogeneous benchmarks are added only if they materially
strengthen the generalization claim beyond these two settings -- not by
default.

## No universal score

Each benchmark keeps its own native scoring semantics (SWE-QA-Pro's 5-axis
rubric, RepoProbe's weighted checklist, WebWalkerQA's/BrowseComp-Plus's
own judged-correctness or IR metrics). This suite does not implement or
report a universal cross-benchmark score -- only the centralized GPT-5
judge model is standardized (see `evaluation_suite/judge.py`), never a
universal rubric.

---

## Track A -- Repository information seeking

### Primary benchmarks

1. **SWE-QA-Pro** (`src/ant/benchmarks/sweqa_pro.py`)
2. **RepoProbe** (`src/ant/benchmarks/repoprobe.py`)

### Primary methods

| method | status | notes |
|---|---|---|
| Direct | active | `ant.agents.direct.DirectAgent` |
| Retrieval | active | `ant.agents.retrieval.RetrievalAgent` |
| Matched ReAct | active | `ant.agents.matched_react.MatchedReActAgent` |
| ANT | active | `ant.agents.ant_adapter.AntAgent` |
| SWE-QA-Pro official ToolCallingAgent | active | `ant.external_wrappers.sweqa_pro_native_agent.SweQaProNativeAgent` -- verified live this session (2/2 real dev-question smoke runs succeeded) |
| RepoGraph | active (as an augmentation, not a standalone agent) | evaluated system is precisely **"Matched ReAct + RepoGraph"** -- `MatchedReActAgent(extra_tools={"search_repograph": ...})`; see `third_party/manifests/repograph/manifest.json` |

### Deferred / supplementary

| method | status | reason |
|---|---|---|
| HiFiRepoQA | `deferred` | method implementation exists, partial execution verified live, but its own knowledge-base-building step is operationally expensive (whole-repo function/class summarization, no rate-limit backoff, on the order of hours for one mid-sized repo's first-time index) -- not part of the current primary set. All integration work preserved. |
| DeepRepoQA | `blocked_external_release` | public implementation depends on an unavailable PyArmor native runtime (confirmed via direct `ModuleNotFoundError`, not assumed) plus an unavailable `VOYAGE_API_KEY`. No source available to patch. Not reimplemented from the paper. All integration work preserved. |
| SWE-agent / Agentless | not started | only if later needed |

---

## Track B -- General / web information seeking

### Primary benchmarks

3. **WebWalkerQA** (arXiv:2501.07572, ACL 2025) -- see
   `docs/webwalkerqa_ant_mapping.md` for the full benchmark audit and
   ANT-mapping design (Stage 6 -- design only, no implementation yet).
4. **BrowseComp-Plus** (arXiv:2508.06600, ACL 2026) -- a fixed ~100K-
   document corpus retrieval benchmark, structurally different from
   WebWalkerQA (flat search-and-retrieve vs. single-site hierarchical
   traversal) -- see `docs/webwalkerqa_ant_mapping.md`'s own comparison
   section.

### Primary methods

| method | status | notes |
|---|---|---|
| simple retrieval/RAG baseline | not started | where benchmark-compatible; design pending Track B implementation |
| ReAct-style single agent | not started | Track B analogue of Matched ReAct |
| ANT | not started | pending the territory-mapping design in `docs/webwalkerqa_ant_mapping.md` |
| WebWalker | fidelity/setup plan only | see `docs/web_track_baselines.md` |
| SLIM | fidelity/setup plan only | see `docs/web_track_baselines.md` |
| PACE | fidelity/setup plan only -- **currently blocked**, no working code found (see below) |

### Secondary

| method | status | notes |
|---|---|---|
| AggAgent | secondary scaling experiment, not a main baseline | see `docs/web_track_baselines.md` |

### Reserve benchmarks (not implemented now)

- **AssistantBench** -- first reserve.
- **FRAMES** -- second reserve.

Reason for reserve status: we first want to test ANT on the two
structurally clear settings above. AssistantBench/FRAMES are added only
if the four-primary-benchmark story turns out insufficient for the
generalization claim. No adapter/planning work exists yet for either; none
is being started in this pass.

---

## Track A: RepoProbe compatibility matrix

Static analysis only -- no RepoProbe experiments launched. For each
Track A method, whether it can in principle operate on RepoProbe, and
what would limit it.

| method | RepoProbe-compatible? | limitation / note |
|---|---|---|
| Direct | Yes | No repository access at all -- benchmark-agnostic by construction. |
| Retrieval | Yes | `EvalRepoEnvironment` file-universe policy (content-based, not extension-list) already covers all of RepoProbe's 50 repos regardless of language -- verified in the earlier repo-file-universe audit (RepoProbe sample included Swift/C#/Lua/Ruby/PHP repos, all correctly covered). |
| Matched ReAct | Yes | Same `EvalRepoEnvironment` coverage; `LocalSearchTool`'s own lexical/symbol tools are language-agnostic (regex/line-based, not Python-specific). |
| ANT | Yes | Territory discovery (`discover_territories`) and worker cards operate on `EvalRepoEnvironment.iter_files()` output directly -- no Python-specific assumption found in that code path. |
| RepoGraph (as "Matched ReAct + RepoGraph") | **Partial -- 8/50 repos only** | RepoGraph's own `find_files()` filters to `.py` files unconditionally (confirmed by direct inspection, not assumed) -- it is a **no-op tool** (always returns "not found") for any RepoProbe repo whose content isn't Python. Per this suite's own RepoProbe repo-language audit (from the earlier repo-file-universe pass): only 8/50 sampled RepoProbe repos are Python-primary. For the other 42, "Matched ReAct + RepoGraph" degrades to plain Matched ReAct behavior (the extra tool is available but never finds anything) -- this must be disclosed in any RepoProbe result table, not silently averaged in as if the tool were contributing everywhere. |
| SWE-QA-Pro official ToolCallingAgent | **Not scientifically meaningful to force onto RepoProbe** | Tightly coupled to SWE-QA-Pro-specific interfaces: `agent.query(question, repo_path)` and its own prompts/tools are written and tuned specifically for SWE-QA-Pro's own dataset shape and repo set (`eval/repos.txt`'s own pinned 26 repos). Porting it to RepoProbe's differently-structured dataset (its own CSV schema, its own Docker-isolated harness, its own repo set) would require rewriting the adapter layer in ways that go beyond "point it at a different repo" -- there is no evidence the official authors intended or tested this. **Decision: main table shows "SWE-QA-Pro official agent: SWE-QA-Pro only."** Do not force it onto RepoProbe. |

This matrix will be re-checked (not re-derived from assumption) before any
formal RepoProbe run actually launches, since RepoProbe's own repo set
could differ from the 50-repo sample audited here.

---

## Fairness policy (both tracks, restated)

- GPT-4.1 generation for every reproduced inference-time method where the
  official implementation is genuinely model-pluggable and swapping the
  model does not change the method itself. Trained-weight or method-
  intrinsic-model baselines (e.g. PACE's fixed BGE-M3/Gemini-2.5-Flash-Lite
  auxiliary models, WebWalker's own untested-with-GPT-4.1 status) are kept
  as-is and marked semi-controlled/external-reference, never silently
  swapped.
- Centralized GPT-5 judge for every LLM-judged benchmark, uniformly (see
  `evaluation_suite/judge.py`).
- No ANT WorkerCards, Need Graph, recovery, or other ANT-specific
  coordination mechanisms are ever injected into a baseline method --
  confirmed for every method above by direct code inspection, not
  assumed.
- Generation and judging are kept as separate calls/passes for every
  method (no benchmark's own judge is reused as a generation-time signal).
