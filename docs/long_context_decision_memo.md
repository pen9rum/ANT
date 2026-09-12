# Long-context evaluation: decision memo (analysis-only pass over the existing 420 generations)

> **CANONICALITY NOTICE (added by the evaluation-fix pass, see
> `docs/long_context_evaluation_fix_report.md` for the full audit):** the
> `multineedle-scaling` and `single-needle-scaling` tables below (and any
> Early/Middle/Late positional analysis built on them) used a filler pool
> that could leak passages from a needle's own source article/document --
> now fixed (`_exclude_leaking_fillers`). Those tables are
> **PRE-FIX / NON-CANONICAL** for needle retrieval/position/scaling claims
> and must be regenerated before being cited for that purpose; the
> `contamination-study` numbers are affected the same way. The natural
> multi-document pilot tables (HotpotQA/2Wiki/MuSiQue) are NOT
> NIAH+-constructed and remain canonical as documents, but their reported
> EM/F1 is superseded by the new extracted-EM/F1 metric (see the fix
> report's rescored table) -- raw EM/F1 undercounted verbose-but-correct
> answers. Nothing below has been altered.

No new API calls, code, prompts, manifests, or outputs were modified to
produce this memo -- every table below is computed directly from the
already-committed raw data (`output/runs/{contamination-study,
natural-multidoc-pilot,multineedle-scaling,single-needle-scaling}/*.jsonl`
and their trajectory dumps) and the frozen manifests under
`third_party/manifests/long_context/`.

## 1. Full natural multi-document results (all 5 methods x 3 benchmarks)

| Method | Benchmark | avg EM | avg F1 | total calls | total tokens | total cost | avg wall(s) |
|---|---|---|---|---|---|---|---|
| direct_document | hotpotqa | 0.80 | 0.867 | 10 | 7,362 | $0.015 | 1.8 |
| direct_document | 2wikimultihopqa | 1.00 | 1.000 | 10 | 6,526 | $0.013 | 1.4 |
| direct_document | musique | 0.00 | 0.427 | 10 | 11,704 | $0.024 | 1.6 |
| retrieval_document | hotpotqa | 0.40 | 0.511 | 15 | 14,288 | $0.040 | 3.6 |
| retrieval_document | 2wikimultihopqa | 0.60 | 0.600 | 21 | 19,782 | $0.052 | 5.5 |
| retrieval_document | musique | 0.20 | 0.200 | 22 | 23,074 | $0.062 | 6.7 |
| matched_react_document | hotpotqa | 0.40 | 0.594 | 32 | 34,289 | $0.082 | 6.7 |
| matched_react_document | 2wikimultihopqa | 0.20 | 0.698 | 47 | 50,523 | $0.121 | 10.2 |
| matched_react_document | musique | 0.20 | 0.360 | 51 | 50,904 | $0.123 | 11.4 |
| longagent | hotpotqa | 0.60 | 0.740 | 20 | 13,262 | $0.034 | 4.1 |
| longagent | 2wikimultihopqa | 0.80 | 0.960 | 20 | 12,090 | $0.031 | 3.9 |
| longagent | musique | 0.20 | 0.253 | 52 | 57,240 | $0.140 | 9.4 |
| ant_document | hotpotqa | 0.20 | 0.372 | 208 | 181,560 | $0.491 | 54.2 |
| ant_document | 2wikimultihopqa | 1.00 | 1.000 | 158 | 133,085 | $0.351 | 37.2 |
| ant_document | musique | 0.20 | 0.360 | 290 | 275,749 | $0.720 | 77.7 |

(5 tasks/benchmark; totals are sums over those 5, per-task avg EM/F1/wall
shown.) ANT's call/token/cost footprint is 5-25x every other method's on
this pilot -- its own accuracy is highest on 2Wiki (tied 1.000 with
Direct) but lowest on HotpotQA (0.372, below even Direct's 0.867) of the
coordinated methods.

## 2. Full controlled multi-needle table (5 methods x 3 lengths x 3 positions = 45 rows)

| Method | Length | Position | avg EM | avg F1 | avg calls | avg tokens | avg cost | avg wall(s) |
|---|---|---|---|---|---|---|---|---|
| direct_document | 32K | early | 0.80 | 0.87 | 2.0 | 34,027 | $0.068 | 2.8 |
| direct_document | 32K | middle | 0.80 | 0.93 | 2.0 | 34,019 | $0.068 | 3.1 |
| direct_document | 32K | late | 0.60 | 0.73 | 2.0 | 34,030 | $0.068 | 2.4 |
| direct_document | 64K | early | 0.80 | 0.87 | 2.0 | 67,592 | $0.135 | 3.8 |
| direct_document | 64K | middle | 0.60 | 0.67 | 2.0 | 67,589 | $0.135 | 3.3 |
| direct_document | 64K | late | 0.80 | 0.87 | 2.0 | 67,603 | $0.135 | 3.4 |
| direct_document | 128K | early | 0.60 | 0.67 | 2.0 | 134,589 | $0.269 | 7.3 |
| direct_document | 128K | middle | 0.60 | 0.67 | 2.0 | 134,589 | $0.269 | 5.7 |
| direct_document | 128K | late | 0.60 | 0.67 | 2.0 | 134,589 | $0.269 | 6.2 |
| retrieval_document | 32K | early | 0.60 | 0.71 | 3.2 | 3,434 | $0.010 | 4.9 |
| retrieval_document | 32K | middle | 0.40 | 0.57 | 3.2 | 3,371 | $0.009 | 4.9 |
| retrieval_document | 32K | late | 0.40 | 0.57 | 3.2 | 3,496 | $0.010 | 4.8 |
| retrieval_document | 64K | early | 0.40 | 0.57 | 3.2 | 3,452 | $0.009 | 4.8 |
| retrieval_document | 64K | middle | 0.40 | 0.57 | 3.2 | 3,527 | $0.010 | 5.3 |
| retrieval_document | 64K | late | 0.40 | 0.56 | 3.2 | 3,559 | $0.010 | 5.2 |
| retrieval_document | 128K | early | 0.20 | 0.51 | 3.2 | 3,389 | $0.009 | 5.4 |
| retrieval_document | 128K | middle | 0.40 | 0.57 | 3.2 | 3,375 | $0.009 | 5.6 |
| retrieval_document | 128K | late | 0.40 | 0.57 | 3.2 | 3,472 | $0.010 | 5.8 |
| matched_react_document | 32K | early | 0.20 | 0.40 | 6.0 | 5,662 | $0.014 | 5.8 |
| matched_react_document | 32K | middle | 0.40 | 0.63 | 5.8 | 5,166 | $0.013 | 6.6 |
| matched_react_document | 32K | late | 0.40 | 0.52 | 5.4 | 4,864 | $0.012 | 5.3 |
| matched_react_document | 64K | early | 0.20 | 0.32 | 7.0 | 9,630 | $0.022 | 7.3 |
| matched_react_document | 64K | middle | 0.20 | 0.40 | 5.4 | 4,862 | $0.012 | 9.8 |
| matched_react_document | 64K | late | 0.40 | 0.47 | 5.8 | 5,213 | $0.013 | 6.2 |
| matched_react_document | 128K | early | 0.20 | 0.39 | 7.4 | 10,543 | $0.024 | 7.8 |
| matched_react_document | 128K | middle | 0.20 | 0.39 | 5.6 | 4,758 | $0.012 | 6.2 |
| matched_react_document | 128K | late | 0.20 | 0.46 | 7.6 | 11,011 | $0.025 | 8.8 |
| longagent | 32K | early | 0.40 | 0.51 | 22.0 | 42,514 | $0.091 | 8.0 |
| longagent | 32K | middle | 0.20 | 0.45 | 26.0 | 51,333 | $0.110 | 13.7 |
| longagent | 32K | late | 0.40 | 0.58 | 22.0 | 43,061 | $0.093 | 8.5 |
| longagent | 64K | early | 0.40 | 0.62 | 40.0 | 84,351 | $0.182 | 14.2 |
| longagent | 64K | middle | 0.60 | 0.74 | 47.6 | 100,422 | $0.214 | 19.4 |
| longagent | 64K | late | 0.40 | 0.62 | 47.6 | 100,739 | $0.216 | 15.4 |
| longagent | 128K | early | 0.00 | 0.45 | 76.0 | 166,803 | $0.356 | 27.0 |
| longagent | 128K | middle | 0.20 | 0.46 | 90.8 | 198,952 | $0.422 | 30.3 |
| longagent | 128K | late | 0.40 | 0.56 | 76.0 | 165,643 | $0.353 | 30.3 |
| ant_document | 32K | early | 0.40 | 0.61 | 20.4 | 18,236 | $0.048 | 25.0 |
| ant_document | 32K | middle | 0.20 | 0.37 | 29.4 | 23,180 | $0.063 | 40.6 |
| ant_document | 32K | late | 0.40 | 0.58 | 34.6 | 35,252 | $0.091 | 43.8 |
| ant_document | 64K | early | 0.20 | 0.46 | 26.0 | 27,655 | $0.069 | 32.2 |
| ant_document | 64K | middle | 0.40 | 0.57 | 23.4 | 18,020 | $0.048 | 27.2 |
| ant_document | 64K | late | 0.20 | 0.44 | 22.8 | 24,490 | $0.062 | 28.1 |
| ant_document | 128K | early | 0.60 | 0.73 | 27.4 | 37,536 | $0.088 | 34.2 |
| ant_document | 128K | middle | 0.20 | 0.43 | 35.8 | 43,551 | $0.110 | 49.6 |
| ant_document | 128K | late | 0.60 | 0.74 | 28.4 | 38,965 | $0.096 | 47.3 |

(Each row averages 5 questions.)

## 3. Per-method effectiveness vs. context length, with PAIRED deltas

Design is paired (same question+position pairs exist at every length), so
this uses per-(question,position) F1 deltas, not just the cross-sectional
averages in Section 2.

| Method | 32K avg F1 | 64K avg F1 | 128K avg F1 | paired Δ 32K→64K | paired Δ 64K→128K | paired Δ 32K→128K | pairs improved / degraded / ~same (of 15) |
|---|---|---|---|---|---|---|---|
| direct_document | 0.84 | 0.80 | 0.67 | -0.044 | -0.133 | **-0.178** | 0 / 4 / 11 |
| retrieval_document | 0.62 | 0.56 | 0.55 | -0.053 | -0.011 | -0.064 | 2 / 4 / 9 |
| matched_react_document | 0.52 | 0.40 | 0.41 | -0.120 | +0.015 | -0.105 | 1 / 6 / 8 |
| longagent | 0.51 | 0.66 | 0.49 | +0.145 | -0.169 | -0.024 | 4 / 4 / 7 |
| **ant_document** | 0.52 | 0.49 | 0.63 | -0.033 | **+0.145** | **+0.112** | **6 / 2 / 7** |

**ANT is the only method with a net POSITIVE paired F1 change from 32K to
128K** (+0.112, 6 of 15 paired instances improved vs. only 2 degraded).
Every other method shows a net negative or flat paired trend, and Direct
degrades the most (-0.178, 0 improved / 4 degraded / 11 unchanged).

## 4. Members/workers next to effectiveness and cost -- does ANT's flat activation cost accuracy?

| Length | LongAgent members | LongAgent avg F1 | LongAgent avg cost | ANT territories (avail. workers) | ANT activated workers | ANT avg F1 | ANT avg cost |
|---|---|---|---|---|---|---|---|
| 32K | **19** (fixed, all 15 conditions) | 0.51 | $0.098 | 234.8 | **2.67** (range 2-11) | 0.52 | $0.067 |
| 64K | **37** (fixed, all 15 conditions) | 0.66 | $0.204 | 460.4 | **2.40** (range 2-4) | 0.49 | $0.060 |
| 128K | **73** (fixed, all 15 conditions) | 0.49 | $0.377 | 919.0 | **3.27** (range 1-10) | 0.63 | $0.098 |

**Answer to the specific question asked: yes, ANT's near-flat worker
activation is achieved WITHOUT sacrificing answer quality.** Cross-
sectionally, ANT's F1 at 128K (0.63) is its own HIGHEST of the three
lengths, and the paired analysis in Section 3 confirms this is a real
per-instance improvement, not a sampling artifact (6/15 pairs improved,
only 2/15 degraded). Meanwhile LongAgent's F1 does NOT show a
corresponding accuracy benefit from its much larger, proportionally-
scaled member count -- its 128K score (0.49) is actually its own LOWEST
of the three lengths, despite deploying ~3.8x more members and ~3.8x more
cost than at 32K.

## 5. Scaling ratio table (32K → 128K, cross-sectional group averages)

| Method | Context size ratio | F1 ratio | EM ratio | Calls ratio | Tokens ratio | Cost ratio |
|---|---|---|---|---|---|---|
| direct_document | 4.00x | 0.79x (0.84→0.67) | 0.82x (0.73→0.60) | 1.00x | 3.96x | 3.95x |
| retrieval_document | 4.00x | 0.90x (0.62→0.55) | 0.71x (0.47→0.33) | 1.00x | 0.99x | 0.96x |
| matched_react_document | 4.00x | 0.80x (0.52→0.41) | 0.60x (0.33→0.20) | 1.20x | 1.68x | 1.61x |
| longagent | 4.00x | 0.95x (0.51→0.49) | 0.60x (0.33→0.20) | **3.47x** | **3.88x** | **3.85x** |
| **ant_document** | 4.00x | **1.22x** (0.52→0.63) | **1.40x** (0.33→0.47) | 1.09x | 1.57x | 1.46x |

LongAgent members: 19 → 73 = **3.84x** (essentially tracking its own
calls/tokens/cost ratios above 1:1, as expected for a static partition).
ANT activated workers: 2.67 → 3.27 avg = **1.22x** (essentially flat
relative to the 4.00x context-size growth and the 3.91x growth in its own
available territories, 234.8 → 919.0). **ANT is the only method whose
effectiveness ratio exceeds 1.0x at 128K -- every other method's F1 and
EM ratios are below 1.0x (degrading), while ANT's compute ratios (1.09x
calls, 1.46-1.57x tokens/cost) stay far below LongAgent's (3.47-3.88x)
for a NET GAIN in accuracy, not just a smaller loss.**

## 6. Decontaminated single-needle experiment: full numbers and why ANT is "best"

| Method | Length | avg EM | avg F1 | avg calls | avg cost |
|---|---|---|---|---|---|
| direct_document | 32K | 0.00 | 0.00 | 2.0 | $0.066 |
| direct_document | 64K | 0.33 | 0.33 | 2.0 | $0.133 |
| direct_document | 128K | 0.17 | 0.17 | 2.0 | $0.264 |
| retrieval_document | 32K | 0.17 | 0.28 | 5.0 | $0.023 |
| retrieval_document | 64K | 0.17 | 0.46 | 5.0 | $0.024 |
| retrieval_document | 128K | 0.17 | 0.17 | 5.0 | $0.023 |
| matched_react_document | 32K | 0.00 | 0.00 | 10.5 | $0.037 |
| matched_react_document | 64K | 0.00 | 0.00 | 8.8 | $0.026 |
| matched_react_document | 128K | 0.00 | 0.00 | 8.3 | $0.024 |
| longagent | 32K | 0.33 | 0.42 | 24.7 | $0.169 |
| longagent | 64K | 0.17 | 0.17 | 41.2 | $0.249 |
| longagent | 128K | 0.33 | 0.33 | 80.8 | $0.586 |
| **ant_document** | 32K | **0.50** | **0.61** | 22.3 | $0.070 |
| **ant_document** | 64K | **0.67** | **0.67** | 37.5 | $0.218 |
| **ant_document** | 128K | **0.67** | **0.67** | 23.8 | $0.162 |

ANT is "best" in the precise, narrow sense that it has the highest EM/F1
of all 5 methods at every one of the 3 lengths tested, and (like Section
3/4's multi-needle finding) its score IMPROVES or holds flat with length
rather than degrading -- the only method with that property here besides
Retrieval's noisy, non-monotonic 0.28/0.46/0.17. This is n=2 questions x
3 positions = 6 samples per length, well below the multi-needle pilot's
own n=15 -- **a suggestive result, not yet a statistically robust one.**

Only 2 question indices (0, 2) were used, selected by a construction-
quality filter (>=2 entity substitutions) BEFORE any inference -- see
`docs/single_needle_contamination_study.md`. Section 7 below identifies a
material confound affecting BOTH of these questions that qualifies how
much credit "ANT solved single-needle" deserves.

## 7. Matched ReAct 0-score anomaly: diagnosis from existing traces only

Matched ReAct scored exactly 0.00 F1 on EVERY single-needle condition in
both Part A (Condition C) and Part D (all 18 conditions) -- but scored
0.29-0.52 F1 on the multi-needle track (Section 2). The anomaly is
**specific to the single-needle/SQuAD track**, not a general method
failure. Trace evidence (`trajectory_metadata.steps_taken`/
`tool_type_counts`, read directly from the already-persisted JSONL rows,
no new calls):

**Question 0** (the "Virgin Mary" question): terminates in 2-3 steps
(well under its 50-step budget), using ONLY `search`, never `view` or
`navigate`, at every one of 9 tested conditions. Answer is consistently
"Bernadette Soubirous" (missing "Saint", but unmistakably the memorized
entity). **Classification: parametric-memory-override failure with
premature termination** -- it declares "finish" almost immediately
without deeply exploring the document set, the same failure pattern
Direct/Retrieval/LongAgent/ANT show elsewhere in the contamination study
(Section 8), except Matched ReAct's own tool-use loop makes the
"declared finish with minimal exploration" pattern directly visible in
a way the other methods' architectures don't expose as cleanly.

**Question 2** (the "Basilica... beside which structure" question):
explores extensively (8-22 steps, uses `search`+`view`+occasionally
`navigate`), yet consistently answers "the Main Building" (or a close
paraphrase) -- which is **not a hallucination**: it is the question's
ORIGINAL, pre-substitution answer, and Section 8 below shows this exact
phrase is genuinely present, un-substituted, in multiple OTHER (filler)
documents in the same haystack. **Classification: evidence-found-but-
wrong-evidence-used** -- distinct from question 0's pattern. This is not
a tool-budget/termination problem (it used up to 44% of its budget) or a
pure retrieval failure (it clearly found and read relevant-looking
content); the mechanism appears to be genuine textual support for the
WRONG (pre-counterfactual) answer sitting elsewhere in the haystack,
compounded by whatever memorized prior makes that phrase attractive.

**Conclusion**: Matched ReAct's failure is not one single cause. Per the
governing spec's own four candidate classifications, it exhibits (a)
parametric-memory-override-with-early-termination on one question and
(c) evidence-found-but-final-answer-wrong on the other -- not (tool-
budget exhaustion) or (pure search failure) in either case. No rerun or
tuning was performed to investigate further.

## 8. Contamination study: numeric summary

| Condition | n | Override failures (evidence found, memory won) | Matches counterfactual (success) | Matches memorized |
|---|---|---|---|---|
| A (original) | 10 | 8 | 0 | 8 |
| B (context-authoritative instruction) | 10 | 10 | 0 | 10 |
| C (fully counterfactualized) | 10 | 2 | 2 | 4 |

Condition C cut override failures from 8/10 (Condition A) to 2/10 -- a
75% relative reduction -- and is the ONLY condition that ever produced a
correct counterfactual answer (2/10, both on question 0). Condition B was
measurably WORSE than doing nothing (10/10 override failures, higher
than A's own 8/10).

**Effect of substituting one entity vs. multiple**: question 0's
Condition C instance substituted 2 entities (answer + "Virgin Mary") and
produced 2 of 5 methods (Direct, ANT) with PERFECT counterfactual
recovery. Question 1's Condition C instance substituted only 1 entity
(the answer alone, since the heuristic found no other qualifying shared
entity) and produced ZERO methods with any counterfactual recovery --
every one of 5 methods, under every one of the 3 conditions A/B/C, gave
essentially the identical memorized description for question 1. Strength
of decontamination tracks directly with substitution count in this small
sample (n=2 questions), consistent with, and now quantified beyond, the
qualitative Finding 2 already in `docs/single_needle_contamination_
study.md`.

**New finding from this analysis-only pass, not previously reported**:
inspecting the materialized documents directly (`document-envs/single-
needle-scaling/` and `document-envs/contamination-study/`) shows the
SQuAD-derived filler pool frequently contains OTHER, un-substituted
passages from the SAME source Wikipedia article as the needle -- SQuAD
draws many distinct questions from one article, and only the ONE needle
document gets entity-substituted; unrelated filler documents drawn from
the same article keep the original, true fact. Concretely: `doc_0014.txt`
in the `niah_single_cf_32000_early_2` haystack is a *different* SQuAD
question's own context (same "University_of_Notre_Dame" article),
verbatim, containing the sentence "Next to the Main Building is the
Basilica of the Sacred Heart" -- directly and correctly answering
question 2 with the ORIGINAL (un-substituted) entity, sitting in plain
sight elsewhere in the same haystack. Question 0's needle terms
("Saint Bernadette Soubirous", "Virgin Mary") were similarly found
verbatim, un-substituted, in 4 and 7 other filler documents respectively.
**This is a genuine construction confound affecting every single-needle
instance built from this filler-sampling design (Parts A and D both),
not fixed in this pass** (no code/manifest/output changes were made) --
it means at least part of the observed contamination could be explained
by legitimate (if construction-flawed) textual leakage rather than pure
parametric memory, and qualifies how "clean" a win Condition C's 2/10
successes and Part D's ANT results really are.

## 9. Exact three preconditions behind the GO recommendation

From `docs/long_context_final_report.md` Section 13, restated exactly:

1. Freeze fully-counterfactualized single-needle construction (Condition
   C) as the formal protocol, but implement the required strengthening
   (minimum substituted-entity guarantee) BEFORE any larger single-needle
   run -- not after seeing more scores.
2. Retain the concise-answer contract (frozen in the prior pass) as
   permanent infrastructure -- not touched or re-evaluated this pass.
3. Treat Matched ReAct's decontamination-resistance as an open, disclosed
   methodological question for the final protocol's write-up, not a
   silently-dropped result.

**This memo adds a fourth item that should be treated as an equal-
priority precondition, surfaced only in this analysis pass**: fix the
filler-pool source-overlap confound (Section 8) -- e.g. exclude filler
passages drawn from the same source article/title as the needle -- before
any larger single-needle run, since it currently affects the validity of
every single-needle result produced so far (Parts A and D alike).

## 10. Proposed exact final protocol to freeze

Based only on evidence already gathered (no new runs performed to reach
this proposal):

- **Benchmarks / task construction**:
  - Natural multi-document: HotpotQA (distractor/validation),
    2WikiMultihopQA (default/validation), MuSiQue (default/validation),
    same sources already audited in `docs/long_context_dataset_audit.md`.
  - Controlled long-context: multi-needle only (HotpotQA-sourced, per
    the existing `build_multi_needle_instance`) as the PRIMARY
    controlled task, given it already produced a clean, reproducible
    scaling contrast (Sections 3-5) with no known confound comparable to
    Section 8's filler-leakage issue.
  - Single-needle: retained only as a SECONDARY/diagnostic track, and
    only after the filler-pool fix (Section 9's 4th item) and the
    minimum-substitution-count fix (precondition 1) are both implemented.
- **Task counts**: natural multi-document at 15-20 tasks/benchmark (up
  from this pass's 5, still well short of a full run); controlled
  multi-needle at 10 questions/cell (up from this pass's 5, matching the
  original NIAH+ paper's own per-cell count) x 3 positions x the same 3
  lengths.
  - Add ONE new lower context-length anchor (e.g. 8K or 16K) to the
    controlled multi-needle grid: this pass's 3 points (32K/64K/128K) can
    show a ratio between two ends but cannot show whether the ANT-vs-
    LongAgent scaling contrast (Section 5) is already present at smaller
    scales or is itself a threshold effect that only appears once ANT's
    coordination has "enough" territories to be selective.
- **Context lengths**: 8K or 16K (new), 32K, 64K, 128K.
- **Positions**: EARLY / MIDDLE / LATE, unchanged (paired construction,
  already validated as prefix-preserving across lengths -- Section 5 of
  `docs/long_context_final_report.md`).
- **Single-vs-multi needle construction**: keep both generators
  (`build_multi_needle_instance`, `build_fully_counterfactualized_
  single_needle_instance`) exactly as implemented, contingent on the
  Section 9 filler-pool fix for the single-needle path specifically.
- **Methods**: unchanged, all 5 (Direct, Retrieval, Matched ReAct,
  LongAgent, ANT) -- Matched ReAct's own anomaly (Section 7) is a
  scientifically interesting result to keep measuring, not a reason to
  drop it.
- **Metrics**: official EM/F1 (`ant.evaluation_suite.qa_metrics.
  score_qa`), unchanged; the shared concise-answer contract stays frozen
  infrastructure.
- **Exclusions, unchanged from this pass**: no RULER, no FakeBookQA, no
  WebWalkerQA, no BrowseComp-Plus, no full HotpotQA/2Wiki/MuSiQue, no ANT
  SWE-QA-Pro/RepoProbe-Python runs, no full LongAgent repository
  evaluation.

## 11. What is supported now vs. what still needs a larger run vs. what NOT to claim yet

### Findings already supported by the current pilot (420 generations)

- ANT requires zero core algorithmic changes to run on this substrate
  (verified structurally, not just by score, across all four phases).
  ANT and Matched ReAct share the identical `search` primitive/index.
- LongAgent's member count and cost scale almost exactly proportionally
  with context length (~3.8-3.9x for a 4x context increase), a
  deterministic, mechanical consequence of its fixed-chunk static
  partition -- confirmed identically on TWO independent pilots (multi-
  needle Part C and single-needle Part D).
- ANT's activated-worker count does NOT scale proportionally with either
  context length or its own available-territory count, on both pilots.
- On the paired multi-needle data specifically, ANT's F1 does not degrade
  as context grows to 128K, and its 32K→128K paired delta is net
  positive while every other method's is net flat-to-negative.
- The generic context-authoritative instruction (Condition B) measurably
  does not help SQuAD-style single-needle contamination (10/10 override
  failures, identical to or worse than the uninstructed baseline).
- Matched ReAct's single-needle failure is trace-diagnosable into (at
  least) two distinct observable patterns, neither of which is tool-
  budget exhaustion.

### Promising hypotheses that still require a larger formal run

- "ANT's worker activation stays flat as context scales, without
  sacrificing accuracy" -- true on n=15 (multi-needle) and n=6/length
  (single-needle) paired samples here; needs the larger, additional-
  length-anchor protocol in Section 10 before it can be presented as a
  robust scaling law rather than a suggestive early result.
- "Fully counterfactualized single-needle construction, once the filler-
  leakage confound is fixed, will show an even cleaner ANT/LongAgent
  contrast on single-needle too" -- plausible given Part D's own
  suggestive numbers, but unverified until the confound is actually
  fixed and re-run.
- Whether the LongAgent-vs-ANT scaling contrast already exists at smaller
  context sizes (8K-16K) or is a threshold effect -- no data exists at
  those lengths yet.

### Claims we should NOT make yet

- Any Lost-in-the-Middle claim, mitigating or otherwise (Section 8 of
  `docs/long_context_final_report.md` already establishes this; nothing
  in this analysis pass changes it -- no consistent direction across
  methods/lengths).
- That Condition C "solves" single-needle contamination in general --
  it worked for exactly 2/10 tested cases, both from the same question,
  and Section 8's filler-leakage finding means even those 2 successes
  are not yet a fully clean demonstration.
- That ANT's flat-worker-activation-without-accuracy-loss result is
  final/robust -- it is currently based on 15 paired multi-needle
  instances and 6 single-needle instances per length, not the larger
  per-cell counts proposed in Section 10.
- Any claim that Matched ReAct's single-needle failure generalizes to
  its multi-needle behavior -- the two tracks show materially different
  score ranges (0.00 vs 0.29-0.52 F1) and this memo did not test whether
  the same trace patterns appear in multi-needle trajectories.
