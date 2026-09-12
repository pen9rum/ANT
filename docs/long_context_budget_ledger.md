# Long-context evaluation: budget ledger (this pass)

Hard maximum for this pass: **$70.00**. Target: $50-60. Buffer: keep >=$10
before the hard ceiling; do not start a phase whose conservative estimate
would push projected cumulative spend past $70; if reaching $65 actual,
start no further non-essential phase.

All per-generation cost estimates below are read directly from the
previous pass's own measured `output/runs/niah-plus-smoke/*.jsonl` and
`output/runs/long-context-smoke-v2/*/*.jsonl` files (docs/long_context_
part_ab_report.md Sections 5-7), not invented.

## Phase estimates (written BEFORE launching each phase)

| Phase | Generations | Basis | Conservative estimate |
|---|---|---|---|
| A: contamination study | 30 (2 Q x 3 conditions x 5 methods, 32K MIDDLE only) | Single-needle 32K per-method-set cost ~$0.43 (measured); x2 questions x3 conditions, +30% margin for the extra regrounding call in Condition B | ~$4 |
| B: natural multi-doc pilot | 75 (15 tasks x 5 methods) | Original 6-task/30-gen smoke cost ~$0.60-1.20 total (tiny documents); scale x2.5 for 75 gens, +50% margin | ~$3 |
| C: controlled multi-needle scaling | 225 (45 instances x 5 methods) | Multi-needle per-method-set-of-5 measured: 32K ~$0.20, 128K ~$0.65; 64K interpolated ~$0.40; sum x 15 instance-groups (5 questions x 3 positions) per length... computed in full before launch below | ~$25-30 |
| D (optional): decontaminated single-needle scaling | <=90 (<=3x3x2x5) | Single-needle-scale costs are higher than multi-needle (more ANT worker activation); computed before launch, only if room remains | <=$15, only if cumulative+estimate <= $65 |

## Ledger (filled in as each phase actually runs)

| Phase | Planned gens | Estimated $ | Cumulative prior actual | Projected cumulative | Actual $ | Actual cumulative |
|---|---|---|---|---|---|---|
| A | 30 | ~$5 (pre-launch: 6 instances x per-method-set ~$0.56 (32K single-needle scale, measured) + condition-B regrounding overhead, +margin) | $0.00 | ~$5 | **$2.44** | **$2.44** |
| B | 75 | ~$3 | $2.44 | ~$5.44 | **$2.30** | **$4.74** |
| C | 225 | ~$25-30 (see detail below) | $4.74 | ~$30-35 | TBD | TBD |
| D | <=90 | <=$15 | TBD | TBD | TBD | TBD |

### Phase C pre-launch estimate detail

Frozen manifest: `third_party/manifests/long_context/multineedle_scaling_manifest.json`
(3 lengths x 3 positions x 5 questions = 45 instances x 5 methods = 225 generations).
Paired-across-lengths property EMPIRICALLY VERIFIED before launch (not just analytically
assumed): for question_index=0/position=early, the 32K filler document sequence is an exact
prefix of the 64K sequence, which is an exact prefix of the 128K sequence, with identical
question/answer/needle texts throughout -- confirmed by direct Python inspection.
Per-method-set-of-5 cost basis (measured, multi-needle from the original NIAH+ smoke):
32K ~$0.198, 128K ~$0.654, 64K interpolated ~$0.389 (Direct/LongAgent scale with tokens/
members; Retrieval/Matched ReAct/ANT stay roughly flat). Per-question (all 3 lengths, 1
position) ~$1.24; x3 positions ~$3.72/question; x5 questions ~$18.6. +50% margin for
question-dependent variance (recovery/reroute/conflict counts are not fixed) =>
**~$28 conservative estimate**. Cumulative prior actual: $4.74. Projected cumulative after
C: ~$33, comfortably under $65/$70.

### Phase A pre-launch estimate detail

Frozen manifest: `third_party/manifests/long_context/contamination_study_manifest.json`
(2 question indices x 3 conditions = 6 instances, 32K/MIDDLE only, x5 methods = 30 generations).
Per-instance-per-method-set cost basis (measured, single-needle 32K from the prior NIAH+ smoke):
Direct ~$0.066, Retrieval ~$0.02, Matched ReAct ~$0.025, LongAgent ~$0.183 (worst-case middle),
ANT ~$0.261 (worst-case middle) = ~$0.555/instance x 6 = ~$3.33, +Condition-B regrounding
overhead (2 instances x 5 methods x ~$0.015) ~$0.15, +30% margin => **~$5 conservative estimate**.
Cumulative prior actual: $0.00 (first phase this pass). Projected cumulative after A: ~$5,
well under $70.

(Updated live below as each phase completes -- see the end of this file
for the final tally.)
