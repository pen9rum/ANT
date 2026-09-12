# HotpotQA failure audit, 15-task natural pilot expansion, and 128K position trace audit

Three goals, diagnosis + replication only: (1) diagnose ANT's apparent
HotpotQA weakness from existing traces, (2) expand the natural pilot from
5 to 15 tasks/benchmark without changing any method, (3) audit the
apparent 128K position sensitivity from existing traces. ANT's core
coordination was not modified; no prompts were tuned based on scores; no
frozen repository benchmark results were touched.

New API spend this pass: **$3.77** (150 new natural-pilot generations;
the pre-existing 75 generations, $2.30, were reused verbatim per
instruction -- combined total for all 225 natural-pilot generations is
$6.07, confirmed by summing the now-complete output files).

## A. HotpotQA 5-task failure audit (existing traces only, no reruns)

| task_id | gold | support docs found+selected? | workers activated | ANT F1 | Direct F1 | Failure class |
|---|---|---|---|---|---|---|
| `5a8b57f2...` | "yes" | Yes, both | doc1, doc4 (2) | 0.40 | 1.00 | **H** (metric artifact -- answer "Yes, both were American." is correct) |
| `5a8c7595...` | "Chief of Protocol" | Yes, both | doc1, doc5→doc6 (3, 1 extra) | 0.32 | 0.33 | **H** (metric artifact -- answer lists all 3 real positions incl. the gold one; **Direct makes the identical over-inclusion error**, F1=0.33) |
| `5a85ea09...` | "Animorphs" | **No, neither** | doc5, doc7 (wrong territories) | 0.00 | 1.00 | **A+C** (search/candidate-ranking failure compounded by need-decomposition failure -- see detail below) |
| `5adbf0a2...` | "no" | Yes, both | doc5, doc6 (2, after 2 reroutes + 1 recovery) | 0.14 | 1.00 | **H** (metric artifact -- answer "No, the Laleli Mosque is in Laleli and the Esma Sultan Mansion is in Ortaköy." is correct; recovery mechanism worked) |
| `5a8e3ea9...` | "Greenwich Village, New York City" | Yes, both | doc3, doc9 (2) | 1.00 | 1.00 | None -- exact match |

**Dominant failure mode: metric/verbosity artifact (Category H), not architecture.** In 4 of 5 tasks, ANT's own document search and worker routing found and selected BOTH correct supporting documents; the low reported F1 in 3 of those 4 cases (0.40, 0.32, 0.14) is entirely a consequence of ANT's synthesis producing a complete, semantically-correct, but non-terse sentence ("Yes, both were American." vs. gold "yes"), which word-overlap F1 penalizes. Critically, **Direct makes the exact same over-inclusion error on task 2** (F1=0.33, listing all 3 government positions instead of just "Chief of Protocol") -- this specific failure is not ANT-specific at all, it is a property of the question's own gold-answer ambiguity shared by any method that answers completely.

Only task 3 (`5a85ea09...`, gold "Animorphs") is a genuine content failure, and it is architecturally real, not a metric artifact: ANT answered "not stated in the provided documents" (an honest abstention, not a hallucination) after activating the wrong two territories (doc5, doc7) throughout. Direct, seeing the full 10-document context in one shot, answered "Animorphs" perfectly (F1=1.00) without needing to route anywhere.

**Trace-level root cause for task 3** (inspecting `node_executions`/`graph_delta` directly): at round 0, the coordinator's own initial worker-candidate ranking for the root need placed `worker-doc-doc5` and `worker-doc-doc7` as the ONLY two candidates (`candidate_worker_ranks: {doc7: 1, doc5: 2}`) -- the correct territories (`worker-doc-doc2` "The Hork-Bajir Chronicles", `worker-doc-doc8` "Animorphs") were **never candidates at any of the 6 rounds this task ran**. After 2 rounds of no progress, the Orchestrator decomposed the root need into two children -- but BOTH child hypotheses ("does 'Science Fantasy (magazine)' match the criteria", "does a Square Enix companion book series match the criteria") stayed anchored to the same wrong 2-worker candidate pool, never widening to the other 8 available territories. This is a combined **Category A (initial search/candidate-ranking failure)** + **Category C (need decomposition reinforced the wrong hypothesis space rather than escaping it)** -- explicitly NOT Category B (routing failure with evidence available but wrong worker picked -- the correct workers were never even offered as candidates) and NOT Category D (the "recovery" mechanism proper never triggered here at all; `recovery_events=0` for this task, distinct from task 4's genuine, successful recovery).

### Regime check (Section 2 of the governing spec)

| task_id | docs | total context tokens | supporting docs | Direct F1 | ANT activated workers | ANT calls |
|---|---|---|---|---|---|---|
| `5a8b57f2...` | 10 | 1,066 | 2 | 1.00 | 2 | -- |
| `5a8c7595...` | 10 | 1,166 | 2 | 0.33 | 3 | -- |
| `5a85ea09...` | 10 | 1,416 | 2 | 1.00 | 2 | -- |
| `5adbf0a2...` | 10 | 893 | 2 | 1.00 | 2 | -- |
| `5a8e3ea9...` | 10 | 1,191 | 2 | 1.00 | 2 | -- |

HotpotQA distractor contexts are 850-1,450 tokens total across 10
documents -- trivially small for a modern LLM's context window. **The
hypothesis is supported, not forced**: Direct answers all 5 tasks in
exactly 2 physical calls each at ~$0.003 apiece, with zero search/routing
surface to fail on; ANT's coordination machinery (search ranking, Need
decomposition, multi-round dispatch) adds real complexity and a real
failure mode (task 3) for a corpus small enough that "read everything"
is already a complete, correct strategy. This does not mean coordination
is never useful (see Section D below, where it shows a real, significant
advantage on MuSiQue against LongAgent specifically) -- it means HotpotQA
distractor specifically is a compact-enough regime that adaptive
coordination has more to lose than gain.

## B. Natural 15-task results (all 5 methods x 3 benchmarks, 225 generations, zero errors)

| Method | Benchmark | EM | F1 | calls | tokens | cost |
|---|---|---|---|---|---|---|
| direct_document | hotpotqa | 0.667 | 0.817 | 30 | 26,107 | $0.053 |
| direct_document | 2wikimultihopqa | 0.600 | 0.678 | 30 | 20,539 | $0.042 |
| direct_document | musique | 0.467 | 0.633 | 30 | 36,743 | $0.075 |
| retrieval_document | hotpotqa | 0.400 | 0.597 | 49 | 53,317 | $0.147 |
| retrieval_document | 2wikimultihopqa | 0.467 | 0.501 | 59 | 62,229 | $0.168 |
| retrieval_document | musique | 0.467 | 0.538 | 61 | 63,716 | $0.174 |
| matched_react_document | hotpotqa | 0.467 | 0.704 | 78 | 61,061 | $0.154 |
| matched_react_document | 2wikimultihopqa | 0.400 | 0.685 | 121 | 120,139 | $0.293 |
| matched_react_document | musique | 0.400 | 0.536 | 139 | 175,055 | $0.408 |
| longagent | hotpotqa | 0.533 | 0.758 | 62 | 44,063 | $0.111 |
| longagent | 2wikimultihopqa | 0.733 | 0.842 | 81 | 60,870 | $0.153 |
| longagent | musique | 0.267 | 0.309 | 119 | 120,634 | $0.291 |
| ant_document | hotpotqa | 0.400 | 0.601 | 472 | 391,844 | $1.066 |
| ant_document | 2wikimultihopqa | 0.733 | 0.760 | 516 | 435,984 | $1.146 |
| ant_document | musique | 0.467 | 0.596 | 714 | 667,982 | $1.788 |

Task selection for the 10 new tasks/benchmark: rows 6-15 (0-indexed 5-14)
of each benchmark's own validation-split row order -- the next 10
deterministic rows immediately following the already-used first 5, frozen
to `third_party/manifests/long_context/natural_pilot_manifest_10new_
per_benchmark.json` before any inference, never inspected for
score/difficulty/gold content beforehand.

**At n=15, ANT's HotpotQA F1 (0.601) is far less of an outlier than the
n=5 pilot's own 0.372 suggested** -- still below Direct (0.817) and
LongAgent (0.758), but the gap narrowed substantially, consistent with
Section A's finding that much of the n=5 gap was verbosity/small-sample
noise. **2WikiMultihopQA no longer looks uniformly "favorable" for ANT**:
its perfect 1.000 at n=5 regressed to 0.760 at n=15 (still competitive,
but not a clean win). **MuSiQue's earlier "Direct high-F1-but-zero-EM"
pattern also regressed toward normal** at n=15 (Direct EM=0.467, F1=0.633
-- see Section E for the full discussion).

## C. Paired statistics (ANT vs. each other method, per benchmark)

Seed=42, 10,000 bootstrap replicates, using this repo's own frozen
`ant.evaluation_suite.bootstrap_ci.paired_bootstrap_ci` utility (not a new
implementation).

| Benchmark | Comparison | mean F1 ANT | mean F1 other | mean EM ANT | mean EM other | W/L/T | F1 delta | 95% CI | Significant? |
|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | ANT vs Direct | 0.601 | 0.817 | 0.400 | 0.667 | 2/6/7 | -0.216 | [-0.456, +0.020] | No |
| hotpotqa | ANT vs Retrieval | 0.601 | 0.597 | 0.400 | 0.400 | 4/2/9 | +0.004 | [-0.171, +0.151] | No |
| hotpotqa | ANT vs Matched ReAct | 0.601 | 0.704 | 0.400 | 0.467 | 3/4/8 | -0.103 | [-0.325, +0.102] | No |
| hotpotqa | **ANT vs LongAgent** | 0.601 | 0.758 | 0.400 | 0.533 | 2/4/9 | **-0.157** | **[-0.321, -0.012]** | **Yes (ANT worse)** |
| 2wikimultihopqa | ANT vs Direct | 0.760 | 0.678 | 0.733 | 0.600 | 3/2/10 | +0.082 | [-0.119, +0.305] | No |
| 2wikimultihopqa | ANT vs Retrieval | 0.760 | 0.501 | 0.733 | 0.467 | 5/2/8 | +0.259 | [-0.007, +0.523] | No (borderline) |
| 2wikimultihopqa | ANT vs Matched ReAct | 0.760 | 0.685 | 0.733 | 0.400 | 6/2/7 | +0.075 | [-0.075, +0.248] | No |
| 2wikimultihopqa | ANT vs LongAgent | 0.760 | 0.842 | 0.733 | 0.733 | 1/3/11 | -0.082 | [-0.208, +0.013] | No (borderline) |
| musique | ANT vs Direct | 0.596 | 0.633 | 0.467 | 0.467 | 2/4/9 | -0.038 | [-0.244, +0.176] | No |
| musique | ANT vs Retrieval | 0.596 | 0.538 | 0.467 | 0.467 | 2/4/9 | +0.058 | [-0.124, +0.280] | No |
| musique | ANT vs Matched ReAct | 0.596 | 0.536 | 0.467 | 0.400 | 2/2/11 | +0.059 | [-0.148, +0.267] | No |
| musique | **ANT vs LongAgent** | 0.596 | 0.309 | 0.467 | 0.267 | 5/1/9 | **+0.287** | **[+0.087, +0.511]** | **Yes (ANT better)** |

**Only two comparisons of the twelve reach significance at n=15, and they
point in opposite directions**: ANT is significantly worse than LongAgent
on HotpotQA, and significantly better than LongAgent on MuSiQue. Every
ANT-vs-Direct/Retrieval/Matched-ReAct comparison, on all three
benchmarks, has a CI that includes zero -- **no significant difference
established at this sample size**, consistent with the governing
instruction not to overclaim.

## D. Regime analysis (descriptive only, n=15, no fitted model)

All 15 HotpotQA and all 15 2WikiMultihopQA tasks share the identical
10-document distractor structure (no hop-count variation to test within
either benchmark at this sample); MuSiQue's 15 tasks are all `2hop__`
questions (also no internal hop-count variation available). The only
regime variables with real within-benchmark spread are context length and
ANT's own coordination-churn signals:

| Split (median-based) | avg (ANT F1 - Direct F1) | n |
|---|---|---|
| context tokens >= median (1,374) | -0.088 | 23 |
| context tokens < median | -0.025 | 22 |
| activated workers >= median (2) | -0.062 | 41 |
| activated workers < median | -0.008 | 4 |
| reroutes >= 2 | **-0.128** | 16 |
| reroutes < 2 | -0.018 | 29 |
| >=1 recovery event | **-0.115** | 12 |
| 0 recovery events | -0.036 | 33 |

**This is a hypothesis test, and the result does not support the
hypothesis in its stated form.** ANT does NOT show a stronger relative
advantage on longer-context tasks in this pool (if anything, the sign is
mildly negative for above-median-length tasks, though the gap is small
and n=22/23 is not enough to trust the direction). What DOES show a clear
pattern: tasks that trigger more internal coordination churn (2+
reroutes, or any recovery event) have a **notably worse** relative
delta than calmer tasks. The most defensible reading is not "distributed
evidence helps ANT" but rather "reroutes/recovery are a symptom of a
genuinely hard-to-route task, and even a successful-looking recovery
(task 4 in Section A found both correct documents) does not fully close
the F1 gap against Direct's simpler, format-advantaged answer." No hop-
count-based regime claim can be made from this data at all -- both
HotpotQA and MuSiQue's samples are hop-count-uniform.

## E. MuSiQue interpretation: F1 vs. EM

At n=15, the extreme n=5 pattern ("Direct F1=0.427, EM=0.00") **did not
replicate** -- Direct's own EM/F1 gap narrowed to 0.467/0.633 (a 0.166
gap, not 0.427). ANT's EM/F1 (0.467/0.596, a 0.129 gap) is now essentially
comparable to Direct's, not dramatically different. Matched ReAct
(0.400/0.536) and Retrieval (0.467/0.538) show similar-sized gaps.
**LongAgent stands out as the one method whose EM (0.267) and F1 (0.309)
are BOTH markedly lower than every other method** -- not a metric-only
effect, a genuine correctness gap, and the direct cause of its
significant loss to ANT in Section C. **Conclusion: MuSiQue's earlier
"Direct wins on F1 but not EM" story was largely an n=5 artifact; at
n=15 the more accurate read is that ANT and Direct are statistically
tied, while LongAgent is a real, significant step behind both.**

## F. Position trace audit (128K, 5 paired Early/Middle/Late ANT trajectories, existing traces only)

Position-SPECIFIC needle doc_ids (not a shared reference across
positions -- doc_ids are reassigned per instance since insertion position
shifts the whole list) were read directly from the frozen manifest and
compared against each trajectory's own `worker_ids`/`evidence` paths.

| Q | Early F1 | Middle F1 | Late F1 | Needle worker(s) activated (E/M/L) | First divergence |
|---|---|---|---|---|---|
| 0 | 0.33 | 0.40 | 0.40 | 2/2, 2/2, 2/2 | None -- needle always found; F1 spread is answer-length variance only |
| 1 | 0.32 | 0.30 | 0.30 | 2/2, 2/2, 2/2 | None -- needle always found; all 3 answers list the same 3 positions (same over-inclusion pattern as Section A) |
| 2 | 1.00 | 0.22 | 1.00 | **0/2, 0/2, 0/2** | Needle NEVER found at any position -- Early/Late got the correct answer via other (filler) evidence; Middle got a worse-worded but still substantively-related answer via different filler evidence |
| 3 | 1.00 | 0.22 | 1.00 | 2/2, 2/2, 2/2 | None -- needle always found; Middle's own answer is MORE verbose ("No, they are not located in the same neighborhood." vs "No.") despite identical evidence coverage, and Middle also shows more Need-Graph churn (3 needs, 2 reroutes vs 0/1 at Early/Late) |
| 4 | 1.00 | 1.00 | 1.00 | 2/2, 2/2, **1/2** | Middle activated only 1 of 2 needle workers, yet still answered correctly (F1=1.00) -- evidence redundancy, not a failure |

**Classification**: Questions 0, 1, 3, 4 show the needle worker(s) found
regardless of position (or, for Q4's middle case, a partial miss that
still didn't hurt the score) -- the observed F1 variation across
positions in these four questions is driven entirely by answer-length/
verbosity (same Category-H metric artifact identified in Sections A and
E), not by evidence-access failure. This is **Category E (LLM
stochasticity / no structural difference)** for 4 of 5 questions.
Question 2 is the one structurally interesting case: the needle was never
activated at ANY position, so its own F1 swing cannot be a position-
driven needle-access effect (there was no needle access at any position)
-- it instead reflects which OTHER (non-needle) evidence happened to be
surfaced at each position, itself a downstream consequence of how depth-
based insertion shifts which documents land "near" which territories.
This is closer to **Category C (sparse-selection: ANT activates only a
few of many territories, and which few varies by position)** than to any
position-specific degradation of the needle itself.

## G. Position conclusion

**Choice: 2 -- evidence points mainly to search/index/routing-order
sensitivity and answer-verbosity/metric artifacts, not genuine positional
degradation of context utilization.** In 4 of 5 paired questions, ANT
found the SAME needle evidence regardless of Early/Middle/Late placement;
the F1 dip at Middle in 3 of those 4 cases traces to longer, still-correct
answers, not worse evidence access. The one case with a real structural
difference (Q2) never found the needle at any position at all, so it
cannot demonstrate a Middle-specific weakness either. **The term "Lost-in-
the-Middle" is not used to describe this data set -- it is not supported.**

## H. Method integrity confirmation

- ANT's coordination core (`ant/coordinator/`, `ant/indexing/`,
  `ant/domain/`) was not touched this pass -- no code changes were made
  at all in this pass; every table above is read directly from already-
  materialized JSONL/trajectory files or newly-generated rows from the
  UNCHANGED agent classes already frozen in prior passes.
- No prompt tuning, tool-budget changes, retrieval changes, worker-count
  changes, territory changes, or concise-answer-contract changes were
  made.
- The existing 5-task-per-benchmark outputs were reused verbatim
  (`run_suite`'s own resume-by-task-id logic skipped all 15 already-
  present rows across the 3 benchmarks x 5 methods; only the 30 new
  task-slots x 5 methods = 150 new generations were run).
- No benchmark-specific heuristic was added to any method.
- No implementation defect was discovered that required stopping and
  reporting separately (the HotpotQA task-3 and NIAH+ Q2 findings above
  are architectural/behavioral observations about ANT's search-ranking
  and Need-decomposition dynamics on small corpora, not code defects).

## I. Recommendation

1. **Is HotpotQA a real weakness or just a 5-task artifact?** Mostly a
   5-task artifact, with one real, small architectural weak point. At
   n=15, ANT's HotpotQA F1 (0.601) is statistically indistinguishable
   from Direct/Retrieval/Matched ReAct (all CIs include zero); it is
   significantly behind LongAgent specifically. The single genuine
   content failure identified in Section A (a search-ranking +
   decomposition miss on a compact 10-document corpus) is real and
   worth tracking, but is one failure mode among 15 tasks, not a
   systemic pattern.
2. **Does MuSiQue remain inconclusive?** No longer as inconclusive as at
   n=5 -- ANT and Direct are now statistically tied (CI includes zero,
   EM/F1 gaps comparable to other methods), while ANT is significantly
   AHEAD of LongAgent. The earlier "Direct wins on F1 alone" framing does
   not replicate at n=15.
3. **Is 2Wiki still favorable after expansion?** No -- 2Wiki's perfect
   1.000 at n=5 regressed to a competitive-but-not-significant 0.760 at
   n=15 (every comparison's CI includes zero, though ANT vs Retrieval and
   ANT vs LongAgent are the two closest to significance in either
   direction). The n=5 "clearly favorable" read does not survive
   expansion.
4. **Should the final paper keep all three natural QA benchmarks?**
   Yes -- each now shows a genuinely different, non-redundant picture:
   HotpotQA (ANT roughly tied with 3/4 methods, behind LongAgent),
   2WikiMultihopQA (ANT roughly tied with everyone), MuSiQue (ANT tied
   with 3/4 methods, ahead of LongAgent). Dropping any one would lose a
   distinct data point in an otherwise small evidence base.
5. **Should we run a larger position study, or is the current effect
   better treated as an ordering/path-sensitivity diagnostic?** The
   latter -- treat it as a diagnostic, not a phenomenon requiring a
   larger dedicated study yet. 4 of 5 paired questions show no
   position-dependent evidence-access difference at all; the one
   exception never found its needle regardless of position, so a larger
   study at the SAME construction would likely just add more instances
   of the same two already-understood patterns (verbosity variance,
   sparse-selection filler-sensitivity) rather than surface a new
   phenomenon. If a position study is pursued later, it should first
   address the filler-pool source-overlap confound already flagged in
   `docs/long_context_decision_memo.md` Section 8, since Q2's own
   "correct answer without touching the needle" behavior is consistent
   with that same confound appearing in the multi-needle construction
   too, not only single-needle.
