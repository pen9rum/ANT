# Long-context evaluation, Parts A & B: final report

Governing spec: Part A (short-answer output contract + 6-task re-run) and
Part B (Needle-in-a-Haystack PLUS reconstruction + 12-condition x 5-method
smoke). Commits: `257bce5`, `673074a` (prior track) through this pass's
own commits (`9f8826c` concise-answer contract, `d8b69d6` NIAH+
reconstruction, `28aed4b` frozen manifest, plus the test-only commit
`79f9125`). Both parts are IMPLEMENTATION + BEHAVIORAL SMOKE only -- no
full benchmark run was launched, ANT's coordination algorithm was not
modified, and no method was tuned based on any score observed during this
pass.

## 1. Concise-answer normalization

**Exact shared contract** (`src/ant/evaluation_suite/answer_contract.py`):

> "Return only the minimal answer span required by the question. Do not
> provide explanation, justification, citations, or reasoning unless the
> question explicitly requires them."

Applied as ONE shared, purely POST-HOC function
(`condense_to_answer_span`) -- the LAST thing every one of the five
methods does to its own already-fully-computed raw answer, using the SAME
provider instance each method already has in scope. Never inserted into
any method's reasoning/search/synthesis PROMPTS.

**Files changed**: `src/ant/agents/direct_document.py`,
`retrieval_document.py`, `matched_react_document.py`,
`src/ant/agents/ant_document_adapter.py`, `src/ant/external_wrappers/
longagent.py` (behind a new `apply_concise_answer_contract: bool = False`
constructor flag, since this class is shared with the repository-QA
track -- default False leaves every existing/other caller's behavior
byte-for-byte unchanged; only this pass's document-track driver
constructs it with `True`).

**Did any reasoning/search behavior change?** No, by construction, for
four of five methods: the condensation call happens strictly after
Direct's/Retrieval's/Matched-ReAct's/LongAgent's own answer is already
fully decided, so their search rounds, tool budgets, and leader/member
protocol are provably untouched (LongAgent's own leader/member round
counts, NEW_STATE/CONFLICT occurrences are identical with the flag on or
off in the unit tests). For ANT specifically, an alternative design
(appending the instruction into the shared `question` string passed to
`coordinator.ask()`) was considered and REJECTED, because `local.py`
reuses that exact string for lexical term extraction
(`TOKEN_RE.findall(question)`) and worker-candidate ranking -- polluting
it would have been a real, if inadvertent, search-behavior change. Instead
`condense_to_answer_span` runs strictly after `coordinator.ask()` returns,
on the already-final `state.answer` -- ANT's routing, Need Graph
structure, and recovery state are unaffected by definition, verified by a
dedicated unit test (`test_document_agents_concise_contract.py`) that
stubs `LocalCoordinator` and confirms the shared function is called
exactly once, after `.ask()` returns, with that exact answer.

## 2. Re-run of the original 6-task smoke (before/after)

Same frozen 6-task manifest (2 MuSiQue + 2 HotpotQA + 2 2WikiMultihopQA),
same 5 methods, 30/30 generations, zero errors, both passes.

| Method | Before (avg F1) | After (avg F1) |
|---|---|---|
| direct_document | 0.689 | 0.689 (unchanged -- already concise) |
| retrieval_document | 0.014 | 0.233 |
| matched_react_document | 0.158 | 0.548 |
| longagent | 0.165 | 0.617 |
| ant_document | 0.022 | 0.583 |

Spot-checked directly (not just aggregate numbers): ANT's hotpotqa
`5a8b57f25542...` answer changed from a multi-paragraph analysis
containing "...both were described as American..." (F1 ≈ 0.015 against
gold `"yes"`) to `"Yes, both were American."` (F1 = 0.4) -- same
underlying finding, reformatted. ANT's own coordination diagnostics
(workers activated, reroutes, needs created) show ordinary run-to-run LLM
variance between the two passes (e.g. one MuSiQue task went from 9 to 2
activated workers) -- expected non-determinism from re-running a live,
stochastic API twice, not evidence the contract changed ANT's search
behavior (which the unit-test-level guarantee above already establishes
independently of any live run's variance). **Diagnostic only, n=6 per
method** -- this is a behavioral validation, not a performance claim.

## 3. NIAH+ fidelity audit summary

Full audit: `docs/niah_plus_fidelity_audit.md`. Headline finding: the
released benchmark's own repository (`zuucan/NeedleInAHaystack-PLUS`)
contains exactly 3 files (README + 2 figure images), zero generation
code, confirmed live via the GitHub API tree listing. Its one data file is
a single opaque Google Drive download with no accompanying script --
checked reachable, deliberately NOT downloaded and used, since trusting
an unverifiable blob with no code to audit against would itself be
"silently inventing" undisclosed trust.

This pass builds a disclosed **reconstruction** from the paper's own
quoted protocol (arXiv:2402.11550 Section 3.1, fetched and cross-verified
live): SQuAD-based single-needle with fictional-entity substitution,
HotpotQA-based multi-needle, the paper's own literal depth grids (10-point
0-100% single-needle; {0,33,66,100}% multi-needle), context lengths
{32K, 128K} (this pass's own two smoke points, with 128K exactly matching
the paper's stated upper bound). Metric: this suite's own official EM/F1
(the paper's own "ACC" is referenced but never formally defined in the
fetched text). Every gap between "what the paper says" and "what this
pass assumes" is itemized in the audit's own Section 7 table -- no claim
below should be read as "NIAH+ results" without that qualification.

## 4. Frozen 12-condition manifest

`third_party/manifests/long_context/niah_plus_smoke_manifest_12condition.json`.
2 task types (single_needle, multi_needle) x 2 context lengths (32,000 /
128,000 target tokens) x 3 positions (early/middle/late), one
deterministic instance each (`question_index=0`, fixed before any
generation). Live-verified actual token counts land within ~1% of target
(e.g. 128,160 actual vs 128,000 target for single-needle; 128,004 vs
128,000 for multi-needle).

## 5. Main diagnostic table (n=1 per condition -- behavioral smoke only)

| TaskType | Length | Position | Method | EM | F1 | Calls | Tokens | Cost | Wall(s) |
|---|---|---|---|---|---|---|---|---|---|
| single_needle | 32000 | early | direct_document | 0.00 | 0.00 | 2 | 33187 | $0.066 | 3.2 |
| single_needle | 32000 | early | retrieval_document | 0.00 | 0.00 | 5 | 8159 | $0.020 | 7.1 |
| single_needle | 32000 | early | matched_react_document | 0.00 | 0.00 | 9 | 10858 | $0.025 | 10.7 |
| single_needle | 32000 | early | longagent | 0.00 | 0.00 | 21 | 42168 | $0.094 | 8.2 |
| single_needle | 32000 | early | ant_document | 0.00 | 0.00 | 53 | 101960 | $0.220 | 54.4 |
| single_needle | 32000 | middle | direct_document | 0.00 | 0.00 | 2 | 33187 | $0.066 | 2.9 |
| single_needle | 32000 | middle | retrieval_document | 0.00 | 0.00 | 5 | 8288 | $0.020 | 8.5 |
| single_needle | 32000 | middle | matched_react_document | 0.00 | 0.00 | 9 | 10406 | $0.025 | 9.9 |
| single_needle | 32000 | middle | longagent | 0.00 | 0.00 | 26 | 84833 | $0.183 | 11.9 |
| single_needle | 32000 | middle | ant_document | 0.00 | 0.00 | 57 | 114053 | $0.261 | 72.5 |
| single_needle | 32000 | late | direct_document | 0.00 | 0.00 | 2 | 33187 | $0.066 | 3.0 |
| single_needle | 32000 | late | retrieval_document | 0.00 | 0.00 | 5 | 8360 | $0.021 | 6.5 |
| single_needle | 32000 | late | matched_react_document | 0.00 | 0.00 | 8 | 9021 | $0.022 | 8.4 |
| single_needle | 32000 | late | longagent | 0.00 | 0.00 | 26 | 79114 | $0.171 | 11.4 |
| single_needle | 32000 | late | ant_document | 0.00 | 0.00 | 52 | 101145 | $0.237 | 71.8 |
| single_needle | 128000 | early | direct_document | 0.00 | 0.00 | 2 | 131908 | $0.264 | 8.4 |
| single_needle | 128000 | early | retrieval_document | 0.00 | 0.00 | 5 | 9348 | $0.023 | 10.8 |
| single_needle | 128000 | early | matched_react_document | 0.00 | 0.00 | 19 | 47621 | $0.103 | 21.4 |
| single_needle | 128000 | early | longagent | 0.00 | 0.00 | 73 | 162672 | $0.355 | 24.5 |
| single_needle | 128000 | early | ant_document | 0.00 | 0.50 | 30 | 84471 | $0.183 | 32.4 |
| single_needle | 128000 | middle | direct_document | 0.00 | 0.00 | 2 | 131908 | $0.264 | 8.6 |
| single_needle | 128000 | middle | retrieval_document | 0.00 | 0.00 | 5 | 9437 | $0.024 | 10.2 |
| single_needle | 128000 | middle | matched_react_document | 0.00 | 0.00 | 14 | 24305 | $0.054 | 14.2 |
| single_needle | 128000 | middle | longagent | 0.00 | 0.00 | 73 | 161604 | $0.352 | 22.6 |
| single_needle | 128000 | middle | ant_document | 0.00 | 0.00 | 28 | 137957 | $0.288 | 32.1 |
| single_needle | 128000 | late | direct_document | 0.00 | 0.00 | 2 | 131908 | $0.264 | 7.5 |
| single_needle | 128000 | late | retrieval_document | 0.00 | 0.00 | 5 | 8805 | $0.021 | 7.3 |
| single_needle | 128000 | late | matched_react_document | 0.00 | 0.00 | 9 | 11068 | $0.026 | 9.4 |
| single_needle | 128000 | late | longagent | 0.00 | 0.00 | 81 | 276059 | $0.588 | 27.1 |
| single_needle | 128000 | late | ant_document | 0.00 | 0.00 | 27 | 82979 | $0.179 | 30.7 |
| multi_needle | 32000 | early | direct_document | 1.00 | 1.00 | 2 | 33837 | $0.068 | 3.2 |
| multi_needle | 32000 | early | retrieval_document | 0.00 | 0.40 | 3 | 2590 | $0.007 | 7.3 |
| multi_needle | 32000 | early | matched_react_document | 0.00 | 0.29 | 4 | 1816 | $0.005 | 4.3 |
| multi_needle | 32000 | early | longagent | 0.00 | 0.33 | 22 | 41675 | $0.086 | 8.6 |
| multi_needle | 32000 | early | ant_document | 0.00 | 0.40 | 23 | 12257 | $0.032 | 24.2 |
| multi_needle | 32000 | middle | direct_document | 1.00 | 1.00 | 2 | 33837 | $0.068 | 3.4 |
| multi_needle | 32000 | middle | retrieval_document | 0.00 | 0.20 | 3 | 2510 | $0.007 | 3.8 |
| multi_needle | 32000 | middle | matched_react_document | 0.00 | 0.40 | 4 | 1812 | $0.005 | 3.5 |
| multi_needle | 32000 | middle | longagent | 0.00 | 0.40 | 22 | 41222 | $0.085 | 7.7 |
| multi_needle | 32000 | middle | ant_document | 0.00 | 0.33 | 23 | 11901 | $0.030 | 19.6 |
| multi_needle | 32000 | late | direct_document | 1.00 | 1.00 | 2 | 33837 | $0.068 | 3.0 |
| multi_needle | 32000 | late | retrieval_document | 0.00 | 0.40 | 3 | 2514 | $0.007 | 4.0 |
| multi_needle | 32000 | late | matched_react_document | 0.00 | 0.40 | 4 | 1812 | $0.005 | 4.0 |
| multi_needle | 32000 | late | longagent | 0.00 | 0.40 | 22 | 41955 | $0.089 | 7.9 |
| multi_needle | 32000 | late | ant_document | 0.00 | 0.33 | 23 | 12632 | $0.032 | 21.9 |
| multi_needle | 128000 | early | direct_document | 1.00 | 1.00 | 2 | 134532 | $0.269 | 9.1 |
| multi_needle | 128000 | early | retrieval_document | 0.00 | 0.40 | 3 | 2644 | $0.007 | 6.8 |
| multi_needle | 128000 | early | matched_react_document | 0.00 | 0.40 | 4 | 2036 | $0.005 | 4.1 |
| multi_needle | 128000 | early | longagent | 1.00 | 1.00 | 76 | 162400 | $0.341 | 20.5 |
| multi_needle | 128000 | early | ant_document | 0.00 | 0.25 | 23 | 12950 | $0.034 | 21.4 |
| multi_needle | 128000 | middle | direct_document | 1.00 | 1.00 | 2 | 134532 | $0.269 | 9.0 |
| multi_needle | 128000 | middle | retrieval_document | 0.00 | 0.40 | 3 | 2788 | $0.008 | 6.5 |
| multi_needle | 128000 | middle | matched_react_document | 0.00 | 0.29 | 4 | 2000 | $0.005 | 3.3 |
| multi_needle | 128000 | middle | longagent | 0.00 | 0.40 | 76 | 162328 | $0.339 | 19.5 |
| multi_needle | 128000 | middle | ant_document | 0.00 | 0.40 | 23 | 12135 | $0.031 | 21.4 |
| multi_needle | 128000 | late | direct_document | 1.00 | 1.00 | 2 | 134532 | $0.269 | 8.9 |
| multi_needle | 128000 | late | retrieval_document | 0.00 | 0.40 | 3 | 2746 | $0.007 | 6.5 |
| multi_needle | 128000 | late | matched_react_document | 0.00 | 0.00 | 4 | 2009 | $0.005 | 3.9 |
| multi_needle | 128000 | late | longagent | 1.00 | 1.00 | 76 | 162999 | $0.344 | 21.7 |
| multi_needle | 128000 | late | ant_document | 0.00 | 0.40 | 19 | 10932 | $0.027 | 17.5 |

60/60 generations completed, zero errors.

### The single-needle 0.0 finding (all 5 methods, all positions) -- a real, disclosed limitation, not a bug

Every method except ANT (one condition) scored exactly 0.0 F1 on every
single-needle condition. Inspected directly, not assumed: every method
answered **"Saint Bernadette Soubirous"** -- the REAL, original SQuAD
answer to "To whom did the Virgin Mary allegedly appear in 1858 in
Lourdes France?" -- ignoring the fictional-entity substitution
("Zorvath Quennelin") actually present in the provided context. This is
exactly the failure mode the paper's own fictional-entity-substitution
technique exists to detect: the model answered from memorized world
knowledge rather than the given long context, regardless of how much
search/retrieval/coordination machinery sat in front of it. `question_
index=0` (SQuAD's own first training-set row with a verbatim-matching
answer) happens to be extremely well-known trivia -- a limitation of
this n=1-per-condition smoke's fixed, deterministic question choice, not
something this pass altered after seeing the result (the question was
fixed before any generation, per the audit's own disclosed policy). A
larger pilot should sample multiple, deliberately less-memorized SQuAD
questions.

**One partial exception, worth reporting honestly rather than either
hiding or overclaiming**: `ant_document` on `niah_single_128000_early_0`
answered `"Saint Bernadette Soubirous; Zorvath Quennelin (conflict)"` --
the ONLY one of all 30 single-needle generations (5 methods x 6
conditions) that surfaced the context-grounded fictional entity at all,
explicitly flagging the discrepancy with its own memorized-knowledge
prior. This scored F1=0.5 (partial credit). This is a single instance
(n=1), not evidence ANT "solves" the memorization-shortcut problem --
but it is a genuine, inspected difference in behavior, not an artifact.

## 6. LongAgent behavior

| Condition | Members | Rounds | NEW_STATE | CONFLICT |
|---|---|---|---|---|
| single-needle, 32K (all 3 positions) | 18 | 1 | Yes | 2/3 positions |
| single-needle, 128K (all 3 positions) | 70-81 | 1 | Yes | 1/3 positions |
| multi-needle, 32K (all 3 positions) | 19 | 1 | Yes | No |
| multi-needle, 128K (all 3 positions) | 73-76 | 1 | Yes | No |

At `DEFAULT_CHUNK_SIZE_TOKENS=2000`, 32K -> 18-19 members and 128K ->
70-81 members -- remarkably close to the governing spec's own back-of-
envelope estimate ("32K -> ~16 members, 128K -> ~64 members"), reached
without forcing it (member count is a pure function of
`chunk_document`'s real tokenization, never adjusted to hit a target).
CONFLICT fired genuinely on 3 of 12 conditions (all single-needle) --
real dynamic leader behavior, not a static rubber-stamp "answer
immediately" pattern, even though it never changed the final (memorized-
knowledge) answer on those specific conditions.

## 7. ANT behavior

| Condition | Territories = Workers available | Workers activated | Reroutes | Recovery | Need revisions |
|---|---|---|---|---|---|
| single-needle, 32K | 158 | 4 (all 3 positions) | 2 | 0-1 | 0 |
| single-needle, 128K | 633 | 6-7 | 1 | 0 | 0 |
| multi-needle, 32K | 233 | 2 (all 3 positions) | 1-2 | 0 | 0 |
| multi-needle, 128K | 924 | **2 (all 3 positions, unchanged from 32K)** | 1 | 0 | 0 |

`workers_available == total_territories` on every condition (every
document became exactly one addressable worker, confirmed at up to 924
territories with zero loss/merging). Reroutes occurred on every single
condition -- the coordinator moved off its first-assigned worker at least
once per task even at this scale.

## 8. Position sanity check (EARLY vs MIDDLE vs LATE) -- n=1, not evidence

Single-needle: all 5 methods score 0.0 at all 3 positions, both lengths
(the memorization shortcut dominates regardless of where the needle
sits, so no positional signal is visible at all in this smoke).
Multi-needle: `ant_document` scores 0.40/0.33/0.33 (32K) and
0.25/0.40/0.40 (128K) across early/middle/late -- no consistent
directional pattern; `matched_react_document` similarly varies
0.29-0.40 without a clear trend; `longagent`'s only non-1.0-or-0.4
variation is the single-needle CONFLICT firing at middle/late but not
early for 32K. **None of this supports any Lost-in-the-Middle-style claim
in either direction -- n=1 per condition is explicitly a sanity check
only, per the governing spec's own instruction, not a positional-bias
finding.**

## 9. Scaling sanity check (32K vs 128K) -- two points, no extrapolation

- **LongAgent**: member count scales ~3.9x (18->70, 19->73) for a 4x
  context increase -- consistent with a static, proportional chunk
  partition (the paper's own design), and cost scales similarly
  (~$0.09-0.18 at 32K -> ~$0.34-0.59 at 128K).
- **ANT**: territories scale ~4x (matching the underlying document count
  growth, by construction -- this is mechanical, not a coordination
  choice), but **activated workers do NOT scale proportionally**:
  single-needle grows only 4->6-7 (~1.5-1.75x for a 4x context increase),
  and multi-needle stays EXACTLY FLAT at 2 activated workers across both
  32K and 128K. Cost tracks activated-worker count, not territory count:
  multi-needle cost is nearly flat (~$0.03 at both lengths); single-needle
  cost is actually LOWER at 128K ($0.18-0.29) than at 32K ($0.22-0.26) in
  two of three position pairs, driven by fewer recovery/reroute cycles at
  128K in this particular run, not a general claim.

This is exactly the qualitative difference research question (3) in the
governing spec asks about -- **observed directly in this smoke's own
numbers, without changing either method to produce it**: LongAgent's
static partition inherently couples compute to context length; ANT's
runtime worker activation does not, at least not on this NIAH+
reconstruction's own document-boundary structure. Two points per method
is not a scaling law -- this is reported as an observed contrast, not
extrapolated beyond 128K or claimed as "sublinear" in any fitted sense.

## 10. Total API cost of this task

- Part A re-run (30 generations): included in the original $0.60 + this
  pass's re-run cost (not separately budgeted beforehand, but of the same
  order as the original 6-task smoke, since document sizes are unchanged).
- Part B NIAH+ smoke (60 generations): **$7.04**, 1,139 physical LLM
  calls, 3,307,746 total tokens.
- **Combined total for this pass's own new API spend: approximately $7-8.**

## 11. Methodological blockers / concerns

1. **Single-needle memorization shortcut (Section 5 above)** dominates
   this pass's single-needle results end to end -- a real limitation of
   `question_index=0`'s own fixed identity, not a coordination-quality
   signal. A larger pilot needs multiple, deliberately obscure SQuAD
   questions (and ideally a check that the chosen question's answer is
   NOT independently guessable) before single-needle scores mean
   anything.
2. **NIAH+ is a reconstruction, not the released benchmark** (Section 3) --
   every number in Sections 5-9 is scoped to this pass's own construction
   choices, disclosed in full in `docs/niah_plus_fidelity_audit.md`.
3. **n=1 per condition** (Sections 8/9) supports behavioral observation
   only, explicitly not statistical or scaling claims.
4. **ANT's wall-clock cost is real**: 30-72s per condition, notably
   slower than every other method (Direct: 3-9s; Retrieval: 4-11s;
   Matched ReAct: 3-21s; LongAgent: 8-27s) -- worth tracking as a
   practical concern for a larger pilot's own compute budget, even though
   its token/dollar cost is competitive or better than LongAgent's at
   128K.

## 12. Recommendation

**GO for a larger long-context pilot**, on both tracks, with two
preconditions carried forward from this pass's own findings: (a) the
concise-answer contract is now validated and should be treated as
permanently frozen infrastructure (Part A, Section 2) -- do not revert to
the old verbose-synthesis-only scoring; (b) any larger NIAH+-reconstruction
pilot must sample multiple, less-memorized single-needle questions before
that track's scores are treated as meaningful at all, per Section 11.1.

The three questions this pass was actually built to answer:

1. **Does LongAgent behave normally in its native long-context regime?**
   Yes -- member count scales almost exactly as the paper's own design
   predicts (near-linear with context length at a fixed chunk size), and
   CONFLICT resolution genuinely fired on real conditions, not just
   NEW_STATE rounds.
2. **Can ANT operate on the same 32K/128K information universe with the
   exact same coordination core?** Yes, verified directly: zero
   algorithmic core changes, the same frozen `LocalCoordinator.ask()`
   scaled cleanly to 924 document-territories with no failures across all
   12 conditions.
3. **Do the two architectures already exhibit measurably different
   scaling behavior without changing either method?** Yes, clearly, in
   this smoke's own numbers (Section 9): LongAgent's compute couples to
   context length; ANT's activated-worker count and cost do not, most
   starkly on the multi-needle track (flat at 2 workers across a 4x
   context increase). This is the single most important result of this
   pass -- not which method's twelve numeric scores happen to look best.
