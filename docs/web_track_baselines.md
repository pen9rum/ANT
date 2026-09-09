# Web track (Track B) candidate baselines: fidelity/setup plans

Status: **plans only** -- no code stood up, no live execution attempted,
no broad implementation. Per Stage 7's own instruction: "Prepare
fidelity/setup plans, but do not broadly execute yet." All facts below
are from a dedicated research pass with direct primary-source citations
(paper text quotes, repo URLs, HTTP-verified links) -- confidence levels
and unresolved gaps are stated explicitly, not smoothed over.

Standardization policy: GPT-4.1 only where a method is genuinely
model-pluggable AND swapping the model would not itself change the
method. Where trained weights or a method-intrinsic model are load-
bearing, the original model is preserved and the comparison is marked
semi-controlled, per `docs/evaluation_plan.md`'s fairness policy.

---

## WebWalker

- **Paper/repo**: arXiv:2501.07572 (ACL 2025); repo now at
  `github.com/Alibaba-NLP/DeepResearch/WebAgent/WebWalker/` (the original
  `Alibaba-NLP/WebWalker` URL 301-redirects here -- verified via raw HTTP
  header, not just search).
- **Architecture to preserve faithfully**: the Explorer + Critic loop
  described in `docs/webwalkerqa_ant_mapping.md` section 5 -- a
  ReAct-style Explorer plus a per-step Critic that maintains memory and
  decides sufficiency. Do not collapse this to a single-agent loop; the
  two-role structure is the method's own defining feature.
- **Stack**: ReAct + Qwen-Agent + LangChain, with Crawl4AI for page
  rendering (`crawl4ai-setup`/`crawl4ai-doctor` required). Repo contains
  `README.md`, `requirements.txt`, `src/app.py` (Streamlit demo),
  `rag_system.py`, `evaluate.py` (GPT-4-based scoring). No obfuscation
  found.
- **Model assumptions**: model-pluggable via Qwen-Agent's supported
  backends (`OPENAI_API_KEY` and/or `DASHSCOPE_API_KEY`). The paper's own
  experiments predate GPT-4.1's release, so GPT-4.1 substitution is
  architecturally plausible but **not paper-verified** -- must be
  confirmed live before being treated as validated, not assumed from
  "the code looks pluggable."
- **Setup plan (not executed)**: clone the `DeepResearch` monorepo (large
  -- confirm a shallow/sparse clone is possible to avoid pulling
  unrelated subprojects), isolate `WebAgent/WebWalker/`'s own
  dependencies in a fresh venv, run `crawl4ai-setup`, verify a bare
  Explorer+Critic loop completes one real question against a real small
  site before any benchmark-scale attempt.

## SLIM

- **Paper/repo**: "Lost in the Maze: Overcoming Context Limitations in
  Long-Horizon Agentic Search," arXiv:2510.18939, COLM 2026;
  `github.com/howard-yen/SLIM` (MIT). SLIM = "Simple Lightweight
  Information Management" (confirmed in both README and paper text).
- **Mechanism to preserve faithfully**: three components -- (1) a
  lightweight **search tool** returning top-k title/URL/snippet only
  (not full pages), (2) a **browse tool** that fetches a URL and extracts
  only the most-relevant SECTION of its content (similarity-matched, not
  a full-page dump), (3) a **periodic summarization module** that
  compresses the full trajectory every *n* turns. This is a
  single-agent, enhanced-ReAct architecture -- NOT multi-agent, do not
  add roles that aren't in the original.
- **IMPORTANT CORRECTION** (flagged by the research pass, confirmed via
  direct full-text search of the paper for the literal string
  "GPT-4.1"): the paper's own reported experiments used **o3, o4-mini,
  and Claude-4-Sonnet only** -- GPT-4.1 does NOT appear anywhere in the
  paper. An earlier, uncorroborated claim that SLIM was GPT-4.1-tested is
  **false and must not be repeated**. GPT-4.1 substitution is
  architecturally plausible (SLIM uses LiteLLM, which lists OpenAI as a
  supported provider) but empirically unverified in the source paper --
  any future run using GPT-4.1 with SLIM must be labeled as this suite's
  own untested substitution, not as reproducing a paper-verified
  configuration.
- **Benchmarks evaluated by the authors**: BrowseComp (300-sample) and
  Humanity's Last Exam (300-sample, text-only). **WebWalkerQA is NOT used
  in this paper** -- if SLIM is run against WebWalkerQA in this suite,
  that is a NEW pairing this suite is introducing, not something to cite
  as paper-precedented.
- **Setup plan (not executed)**: `git clone --recursive` + `uv sync`;
  requires a `SERPER_API_KEY` (paid search API, not currently configured
  in this environment -- a real, disclosed prerequisite before any live
  run) plus an LLM provider key.

## PACE

- **Paper**: "PACE: Predictive Adaptive Context Extraction for
  Long-Horizon LLM Agents," ACL 2026 Main,
  `aclanthology.org/2026.acl-long.1252/`. **No arXiv preprint exists**
  (confirmed via thorough, targeted search -- other same-named papers
  exist and were explicitly ruled out as unrelated).
- **Mechanism** (extracted from the camera-ready PDF, not an arXiv HTML
  render -- flagged medium confidence, not independently double-checked
  against a second extraction pass): frames context management as
  "Next Step Prediction" -- encodes query + recent history into dense
  vectors, scores cached historical chunks' relevance to the PREDICTED
  next action via cosine similarity (temperature-0.3 softmax), and uses a
  dynamic "compression pressure" signal to assign each historical chunk
  one of four granularities (full text / detailed summary / brief
  summary / placeholder-only). This is meaningfully more elaborate than
  SLIM's periodic-summarization approach -- attention-inspired
  multi-granularity compression, not fixed-interval compression.
- **BLOCKING FINDING, verified directly**: the ACL Anthology page's own
  "Code/Data" link (`anonymous.4open.science/r/PACE-B000/`) returns
  **HTTP 401 `{"error":"not_connected"}`** on direct request -- the
  anonymized code-sharing link is not connected to any actual repository
  content. A full-text search of the paper itself found **no code
  repository URL anywhere in the paper body**. **Net conclusion: no
  functioning public PACE implementation could be located.** This
  supersedes an earlier, incorrect automated summary that claimed a repo
  was present (traced to a thin/incomplete page fetch, not a real
  finding) -- that earlier claim is retracted here.
- **Model assumptions**: evaluated across WebSailor-32B,
  Tongyi-DeepResearch-30B, DeepSeek-V3.1-671B, Claude-4-Sonnet as
  backbones, with **BGE-M3** (fixed embedding model) and **Gemini 2.5
  Flash-Lite** (fixed summary-generation model) as auxiliary components.
  No GPT model of any kind appears among the evaluated backbones.
  GPT-4.1 pluggability is **unconfirmed** -- do not assume it without
  code to verify against.
- **Benchmarks evaluated**: BrowseComp, BrowseComp-ZH, WideSearch, GAIA,
  xbench-DeepResearch, and **WebWalkerQA** (direct overlap with this
  track's own primary benchmark).
- **Setup plan**: **blocked**. No functioning code exists to stand up.
  If a working implementation surfaces later (e.g. the authors fix the
  anonymous link post-review), re-audit before proceeding -- do not
  attempt to reimplement PACE's own mechanism from the paper text alone,
  for the same reason DeepRepoQA's obfuscated internals were not
  reimplemented: a from-scratch reimplementation would not be a faithful
  reproduction of the actual released method (there is no released
  method to reproduce here at all).

## AggAgent (secondary scaling baseline)

- **Paper/repo**: "Agentic Aggregation for Parallel Scaling of
  Long-Horizon Agentic Tasks," arXiv:2604.11753, COLM 2026;
  `github.com/princeton-pli/AggAgent` (Apache 2.0), also on PyPI
  (`pip install aggagent`).
- **What it does**: NOT a task-solving agent -- an aggregation layer over
  K independently-sampled parallel trajectories for the SAME task,
  synthesizing one final answer via four internal tools
  (`get_solution`/`search_trajectory`/`get_segment`/`finish`). This is
  why it's a secondary SCALING experiment, not a main baseline: it
  requires K existing rollouts to aggregate over, not a single-pass
  question-answering setup comparable to the other Track B methods.
- **Model assumptions**: explicitly GPT-4.1-pluggable -- the README's own
  usage example is `AggAgent(model="gpt-4.1", task="browsecomp")`. Also
  supports local vLLM-served models and Gemini.
- **Setup plan (not executed, lower priority given secondary status)**:
  `pip install aggagent`; requires OpenAI/Gemini keys plus a
  `SERPER_API_KEY` for web-search-dependent benchmarks, and separately
  downloadable rollout/trajectory datasets to reproduce published
  results. Not scheduled before the four Track B primary methods have
  their own setup attempted.

## Summary

| method | code available? | GPT-4.1 status | blocking issue |
|---|---|---|---|
| WebWalker | Yes, open | plausible, unverified | none identified yet |
| SLIM | Yes, open | plausible via LiteLLM, **paper did NOT test GPT-4.1** | needs `SERPER_API_KEY` |
| PACE | **No -- broken/absent code link** | unconfirmed, no GPT backbone in paper | **no runnable implementation exists** |
| AggAgent | Yes, open, PyPI | confirmed GPT-4.1-pluggable | needs `SERPER_API_KEY` for some benchmarks; secondary priority |

No further action taken this pass. Do not broadly execute any of these
until each has passed its own static/fidelity check the way SWE-QA-Pro's
official agent and RepoGraph did in Track A.
