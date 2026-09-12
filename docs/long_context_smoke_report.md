# Long-context / multi-document evaluation: implementation + behavioral smoke report

Scope reminder (per the governing spec): this is an IMPLEMENTATION +
BEHAVIORAL SMOKE report only. No full benchmark evaluation, no RULER, no
formal Lost-in-the-Middle experiment, no ANT SWE-QA-Pro/RepoProbe-Python
runs, and no LongAgent full-repository runs were launched. The 6-task
smoke result table below is a behavioral/wiring check, not a performance
claim.

## 1. Exact benchmark variants and dataset revisions used

Full audit in `docs/long_context_dataset_audit.md`. Summary:

| Benchmark | HF path | Config | Split | Why |
|---|---|---|---|---|
| HotpotQA | `hotpotqa/hotpot_qa` | `distractor` | `validation` | Original release, not LongBench (see below); `test` split has hidden/blank answers |
| 2WikiMultihopQA | `framolfese/2WikiMultihopQA` | `default` | `validation` | `xanhho/2WikiMultihopQA` and LongBench both use a deprecated HF loading script; this mirror is parquet-backed and schema-confirmed identical to HotpotQA's |
| MuSiQue | `dgslibisey/MuSiQue` | `default` | `validation` | Original release; distinct `paragraphs`/`is_supporting`/`answer_aliases` schema |

**LongBench was rejected**, not merely deprioritized, on three independently verified grounds: (1) `datasets>=5.0` refuses to run its deprecated loading script and no Parquet-convert branch exists for either `THUDM/LongBench` or `zai-org/LongBench`; (2) downloading and parsing `data.zip` directly showed `context` is a single flattened, pre-concatenated string, destroying original document boundaries; (3) LongBench carries no supporting-facts field at all, which the no-leakage design (Section 2 below) depends on for diagnostics.

Scoring: official EM/F1 (`normalize_answer` -> lowercase, remove punctuation, remove articles, collapse whitespace; MuSiQue additionally takes max over `[answer] + answer_aliases`), byte-verified against each benchmark's own GitHub eval script.

## 2. Files / commits changed

Committed at `257bce5` ("Add long-document/multi-document evaluation substrate for ANT"):

- `docs/long_context_dataset_audit.md`, `docs/ruler_design_note.md` (this file)
- `src/ant/evaluation_suite/document_scope.py` -- `DocumentRecord`, `EvalDocumentEnvironment`, `materialize_documents`, `chunk_documents`, `build_positional_variants`
- `src/ant/evaluation_suite/qa_metrics.py` -- official EM/F1
- `src/ant/tools/document_tools.py` -- `view_document`, `navigate_chunk`, `chunk_by_global_index`
- `src/ant/benchmarks/_hotpot_style.py`, `hotpotqa.py`, `twowikimultihopqa.py`, `musique.py`
- `src/ant/agents/direct_document.py`, `retrieval_document.py`, `matched_react_document.py`, `ant_document_adapter.py`
- `src/ant/external_wrappers/longagent.py` -- bounded concurrent member dispatch (see Section 6 below); **no other file in `ant/coordinator/`, `ant/indexing/`, or `ant/domain/` was touched**
- Tests: `test_document_scope.py`, `test_qa_metrics.py`, `test_document_tools.py`, `test_hotpot_style_benchmarks.py`, `test_musique_benchmark.py`, `test_ant_document_adapter.py`, plus 3 new tests added to `test_longagent.py`
- `third_party/manifests/long_context/smoke_manifest_6task.json` -- the frozen 6-task manifest

Full existing suite: **514/514 tests pass** (up from 501 before this pass), zero regressions to any frozen repository-evaluation code.

## 3. ANT change classification (the central question)

**Zero algorithmic core changes.** Not one line of `ant/coordinator/local.py`, `ant/indexing/cards.py`, `ant/indexing/territories.py`, or `ant/domain/models.py` was modified. `ant_document_adapter.py` bypasses only `discover_territories` (a repository-directory-clustering heuristic that does not apply to a flat directory of materialized documents) and replaces it with `_document_territories`: one `Territory` per materialized document, built only from `doc_id`/`title`/relative-filename -- deterministic, gold-independent, task-independent. `build_worker_cards`, `IndexStore`, `LocalCoordinator`, and `.ask()` are called completely unmodified, identically to the repository-QA `AntAgent`.

This was verified, not merely asserted: `tests/test_ant_document_adapter.py::test_ant_core_runs_unmodified_on_a_document_substrate_and_finds_the_right_document` feeds document-derived territories/workers into the real, unmodified `LocalCoordinator.ask()` (no reasoner -- the same mechanical-fallback pipeline the repository-QA equivalent test already exercises) and confirms it correctly finds the document containing the answer.

Category classification per the spec's own A/B/C scheme: **A only** (interface/generalization-only: a new territory-construction function, using existing schema-only `Territory`/`WorkerCard` models). No B (repository-specific cleanup) or C (algorithmic behavior change) was needed or performed.

**Disclosed asymmetry, not hidden**: ANT's own `AutonomousWorker` tool loop (frozen inside `local.py`) only meaningfully exercises `search` on a document substrate -- giving it the new `view`/`navigate` primitives would require modifying that frozen loop, a Category-C-adjacent change this pass declined to make. Matched ReAct's document variant genuinely has all three primitives (confirmed used in practice: the smoke run's own trajectories show `view` and `navigate` both actually invoked, not just implemented-and-unused -- see Section 8).

The one intentional, execution-only exception to "zero core changes" is **LongAgent's bounded concurrent member dispatch** (`external_wrappers/longagent.py`, not `ant/coordinator/`) -- see Section 6.

## 4. The frozen 6-task smoke manifest

`third_party/manifests/long_context/smoke_manifest_6task.json`. Selection policy: first N rows in each benchmark's own HF validation-split row order (deterministic, gold-independent), confirmed non-empty/loader-clean via a live diagnostic pass before freezing.

| Benchmark | Task ID | Question |
|---|---|---|
| hotpotqa | `5a8b57f25542995d1e6f1371` | Were Scott Derrickson and Ed Wood of the same nationality? |
| hotpotqa | `5a8c7595554299585d9e36b6` | What government position was held by the woman who portrayed Corliss Archer in *Kiss and Tell*? |
| 2wikimultihopqa | `8813f87c0bdd11eba7f7acde48001122` | Who is the mother of the director of *Polish-Russian War (Film)*? |
| 2wikimultihopqa | `61a46987092f11ebbdaeac1f6bf848b6` | Which film came out first, *Blind Shaft* or *The Mask Of Fu Manchu*? |
| musique | `2hop__460946_294723` | Who is the spouse of the Green performer? |
| musique | `2hop__252311_366220` | Who founded the company that distributed the film UHF? |

## 5. Results table (n=6 per benchmark cell; diagnostic only, not a performance claim)

All 30/30 generations (5 methods x 6 tasks) completed with zero errors.

| Method | MuSiQue F1 | HotpotQA F1 | 2Wiki F1 | Avg calls/task | Avg tokens/task | Total cost | Avg wall-clock/task |
|---|---|---|---|---|---|---|---|
| direct_document | 0.400 | 0.667 | 1.000 | 1.0 | ~1,489 | $0.018 | ~1.0s |
| retrieval_document | 0.000 | 0.005 | 0.038 | 3.0 | ~3,342 | $0.056 | ~5.0s |
| matched_react_document | 0.100 | 0.067 | 0.308 | 7.0 | ~9,039 | $0.126 | ~8.1s |
| longagent | 0.083 | 0.077 | 0.336 | 4.3 | ~4,006 | $0.060 | ~4.3s |
| ant_document | 0.005 | 0.015 | 0.045 | 27.5 | ~22,552 | $0.345 | ~29.1s |

**Total smoke test cost: $0.6046, 257 physical LLM calls, 242,563 tokens.**

### Methodological concern this table surfaces (do not tune around it -- report it)

The large F1 gap is dominated by **answer verbosity, not correctness**. Both `retrieval_document` and `ant_document` use the shared, unmodified `provider.synthesize()` prompt, which produces long, structured, multi-paragraph answers (e.g. "**Concrete technical aspects asked by the question:** ..."). HotpotQA/2Wiki/MuSiQue's official gold answers are short phrases (`"yes"`, `"Chief of Protocol"`). Word-overlap F1's precision term is heavily diluted by a ~150-300 word response competing against a 1-3 word gold string, even when the answer is substantively correct -- confirmed directly by inspection: `ant_document` on hotpotqa task `5a8b57f25542...` correctly identifies both Scott Derrickson and Ed Wood as American (i.e. the correct "yes") inside a long analysis, but scores 0.0 F1 because "yes" never appears as an isolated token match against the verbose prediction. `direct_document`'s own prompt explicitly instructs "final answer only, no explanation," which is why it scores far higher despite doing no document access at all. This is a real, disclosed evaluation-design gap (a metric/prompt mismatch), not evidence that retrieval/coordination hurts, and not something this pass corrected -- correcting it now, after seeing the scores, would be exactly the score-driven tuning the spec prohibits. **A larger pilot should fix this BEFORE running, by standardizing a "concise final answer" instruction across every method's own finish/synthesis prompt, applied uniformly and pre-registered, not derived from these six scores.**

## 6. LongAgent behavioral summary

Bounded concurrent member dispatch was implemented (`DEFAULT_MEMBER_CONCURRENCY=4`, disclosed and logged in every result's `metadata["member_concurrency"]`) and verified to preserve member prompt/context/outputs, leader inputs/policy, member count, and round semantics exactly -- 3 new tests confirm correct per-member response mapping under genuine thread overlap, that concurrency is real (calls actually overlap in time) and bounded (never exceeds the configured limit), and that `member_concurrency=1` remains effectively sequential. All 16 LongAgent tests pass.

On the smoke tasks: **n_members=1 for 4/6 tasks** (HotpotQA/2Wiki documents are short enough to fit in one 2000-token chunk, the paper's own disclosed default) and **n_members=2 for the 2 MuSiQue tasks** (20 paragraphs). No CONFLICT action ever fired (`conflict_used=False` on all 6). Leader rounds ranged 1-3. This means the concurrency feature, while implemented and tested, was **not meaningfully exercised** on these particular short documents -- an honest finding, not an engineered one: these benchmark documents are simply too short for LongAgent's static-partition design to require more than one or two members.

## 7. ANT behavioral summary

Real dynamic coordination behavior occurred on the smoke tasks, not merely static single-worker routing:

| Task | Territories | Workers activated | Need nodes created | Reroutes | Recovery events |
|---|---|---|---|---|---|
| hotpotqa task 1 | 10 | 2 | 1 | 2 | 0 |
| hotpotqa task 2 | 10 | 2 | 1 | 1 | 0 |
| 2wiki task 1 | 10 | 2 | 1 | 1 | 0 |
| 2wiki task 2 | 10 | 2 | 1 | 2 | 0 |
| musique task 1 | 20 | 9 | 1 | 2 | 0 |
| musique task 2 | 20 | 2 | 0 | 1 | 0 (1 need revision) |

Territory construction is one-territory-per-document (deterministic, gold-independent by construction -- see Section 3). Workers activated matched or closely tracked each task's true supporting-document count (2) in 5/6 tasks, and one MuSiQue task activated 9/20 workers (broader multi-worker exploration). Reroutes (a need_id whose recovery state shows more than one distinct worker tried) occurred on every single task, meaning the coordinator did move off its first-assigned worker at runtime at least once per task -- genuine runtime-adaptive behavior, not static one-shot dispatch. No stuck-episode recovery was triggered on this small a sample. `workers_available == total_territories` on every task, confirming every document became exactly one addressable worker with no silent merging or loss.

## 8. Confirmation: ReAct and ANT use matched document primitives

`matched_react_document`'s tool set (`search`, `view`, `navigate`) is genuinely exercised, not just implemented: across the 6 smoke tasks it used `search` in all 6, `view` in 4, and `navigate` in 1 (2 navigate calls on hotpotqa task 2). `search` for Retrieval/Matched-ReAct/ANT all route through the identical `LocalSearchTool.search()` over the identical `EvalDocumentEnvironment`-derived file list -- the same corpus, same limits (Section 10's "do not give ANT a stronger retrieval engine" is satisfied by construction, not by a separate parity audit). The disclosed limitation from Section 3 stands: ANT's own worker loop only meaningfully used `search` in this run (confirmed: `ant_document`'s tool_calls counts reflect only search/dense_search-style calls, since `AutonomousWorker`'s frozen loop has no `view`/`navigate` primitive to invoke) -- full three-way tool parity was not, and structurally could not be, achieved without a Category-C-adjacent core change.

## 9. Position-perturbation (Lost-in-the-Middle) utility validation

Exactly one dry-run example was executed (no formal evaluation launched), against a REAL frozen smoke task: hotpotqa `5a8b57f25542995d1e6f1371` (10 documents, `supporting_doc_ids=["doc1","doc4"]`). Verified directly: all three variants (early/middle/late) preserve the full 10-document set with identical text/titles, the distractor documents' relative order is preserved across all three variants, and the supporting pair is correctly spliced to the very front, an interior split point, and the very end respectively. Question/answer and every document's own text were never touched -- only ordering changed. 12 corresponding permanent unit tests were also added (`test_document_scope.py`) covering the general case.

## 10. Blockers / methodological concerns

1. **F1/verbosity mismatch (Section 5)** -- the dominant finding of this smoke pass; must be fixed by prompt standardization before any scored evaluation, not fixed by re-scoring these six tasks after the fact.
2. **Tool-primitive asymmetry (Sections 3, 8)** -- ANT cannot be given `view`/`navigate` without touching frozen core; disclosed, not resolved.
3. **LongAgent's static partition barely engages on short documents** (Section 6) -- a real substrate-fit question for a larger pilot: LongAgent's design target is 128K-token documents, and MuSiQue/HotpotQA/2Wiki's individual tasks are far shorter, so a larger pilot should consider whether concatenating multiple tasks' contexts (or picking longer-context tasks specifically) is needed to exercise LongAgent's actual mechanism at all.
4. **n=6 supports no statistical claim of any kind** -- every score above is a behavioral/wiring smoke signal only.

## 11. Recommendation

**GO for a larger long-context pilot**, conditional on fixing item 1 (concise-answer prompt standardization, pre-registered before running, applied identically across all 5 methods) before that pilot's scores are treated as meaningful. The substrate-agnosticism claim -- the same ANT coordination mechanism (Need Graph, runtime worker routing, reroute-on-stuck) transferring from repository information-seeking to multi-document information-seeking with zero algorithmic core changes -- is verified, not merely asserted, at the wiring level: ANT correctly built 10-20 document-territory workers per task, activated a plausible subset of them, revised/rerouted at runtime, and (in every task examined) actually surfaced the two documents that ground the real gold answer, using the exact same frozen `LocalCoordinator.ask()` that runs repository QA. **The most important result of this pass is that transfer, not which method's six numeric scores happen to look best -- and by that standard, this pass succeeded.**
