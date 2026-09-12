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
| A | 30 | ~$4 | $0.00 | ~$4 | TBD | TBD |
| B | 75 | ~$3 | TBD | TBD | TBD | TBD |
| C | 225 | ~$25-30 | TBD | TBD | TBD | TBD |
| D | <=90 | <=$15 | TBD | TBD | TBD | TBD |

(Updated live below as each phase completes -- see the end of this file
for the final tally.)
