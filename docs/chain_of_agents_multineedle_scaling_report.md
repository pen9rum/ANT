# Chain-of-Agents added to the canonical multineedle-scaling matrix

Frozen evaluation run, not method development. CoA's implementation
(`src/ant/external_wrappers/chain_of_agents.py`, commit `a3b6310`),
construction, answer extraction, and scoring were all used unmodified.
No other method was touched.

## A. Run status

- **45/45 conditions completed**, 0 errors, 0 retries (615/615 physical
  calls succeeded on the first attempt).
- Total physical calls: 615. Total cost: **$8.5174** (generation) +
  **$0.0214** (post-hoc answer-extraction rescoring, same frozen pipeline
  every other method's outputs already went through) = **$8.5388 total**
  — under the $9.00 pre-run estimate and the $12 hard cap.
- Preflight (Section 2 of the governing spec): all 45 conditions verified
  byte-identical in real document content to the current, clean, post-fix
  construction code (`build_multi_needle_instance`), reading the same
  already-materialized canonical environments the other 5 methods already
  used — not regenerated, not resampled, not the archived pre-fix copies.
  One artifact was caught and disclosed during preflight, not silently
  patched: `serialize_repository` (the same function LongAgent already
  uses) picks up each environment's own `.materialized` sentinel file
  (written by the generation driver, containing only the literal text
  "ok") as if it were a document — a pre-existing, content-free artifact
  shared identically by CoA and LongAgent (both read `environment_root`
  via this function), never gold/needle-related, and negligible in size
  (~15–20 tokens against 32K–144K contexts). Not something this frozen
  pass is permitted to fix (would mean modifying `serialize_repository`,
  shared with LongAgent); disclosed here rather than ignored.
- Result paths: `output/runs/multineedle-scaling/chain_of_agents.jsonl`
  (raw), `chain_of_agents.rescored.jsonl` (extracted, canonical),
  `trajectories/chain_of_agents-*.json` (45 full traces).
- Commit for this run's code/report: see Section-17 commit hash at the
  end of this document.

## B. CoA scaling

| Length | Avg source tokens | Workers | Calls | Cost/query | EM | F1 |
|---|---:|---:|---:|---:|---:|---:|
| 32K | 36,466 | 5.0 | 7.0 | $0.081 | 0.600 | 0.738 |
| 64K | 72,725 | 10.0 | 12.0 | $0.165 | 0.600 | 0.726 |
| 128K | 145,009 | 20.0 | 22.0 | $0.323 | 0.267 | 0.424 |

Worker count is **exactly** proportional to length (5 → 10 → 20, exactly
2× per doubling) — CoA's fixed-budget chunking guarantees this by
construction, not an emergent property. Call count grows more slowly
(7 → 12 → 22) because manager+condense calls (a constant +2) dilute the
ratio at small chunk counts. EM/F1 hold roughly flat from 32K to 64K,
then drop sharply at 128K (F1 0.726 → 0.424) — reported as a measured
fact; Section G/I discuss what the trace data does and does not explain
about it.

## C. Six-method quality table (extracted EM/F1, canonical)

| Method | 32K EM/F1 | 64K EM/F1 | 128K EM/F1 | Overall EM/F1 |
|---|---|---|---|---|
| direct_document | 0.667/0.794 | 0.667/0.778 | 0.533/0.638 | 0.622/0.737 |
| retrieval_document | 0.800/0.862 | 0.800/0.862 | 0.800/0.861 | 0.800/0.862 |
| matched_react_document | 0.600/0.691 | 0.533/0.620 | 0.467/0.600 | 0.533/0.637 |
| longagent | 0.600/0.740 | 0.600/0.740 | 0.600/0.720 | 0.600/0.733 |
| ant_document | 0.667/0.752 | 0.600/0.695 | 0.667/0.729 | 0.644/0.725 |
| **chain_of_agents** | 0.600/0.738 | 0.600/0.726 | 0.267/0.424 | 0.489/0.629 |

At 32K/64K, CoA is competitive with LongAgent (both ~0.60 EM). At 128K,
CoA is the only method whose F1 drops substantially (0.424, vs. 0.60–0.86
for every other method) — the lowest 128K F1 of the six.

## D. Coordination scaling

| Length | ANTMAN active workers | LongAgent members | CoA workers |
|---|---:|---:|---:|
| 32K | 2.67 | 19.0 | 5.0 |
| 64K | 3.13 | 37.0 | 10.0 |
| 128K | 3.33 | 73.2 | 20.0 |

Canonical ANTMAN/LongAgent reference values were **loaded from the
existing canonical artifacts and reproduced exactly** (2.67/3.13/3.33 and
19.0/37.0/73.2) — not hardcoded into this analysis, matching Section 12's
requirement.

## E. Call/cost scaling

| Length | ANTMAN calls/cost | LongAgent calls/cost | CoA calls/cost |
|---|---|---|---|
| 32K | 33.9 / $0.104 | 24.7 / $0.104 | 7.0 / $0.081 |
| 64K | 35.9 / $0.135 | 40.0 / $0.181 | 12.0 / $0.165 |
| 128K | 33.6 / $0.148 | 76.2 / $0.355 | 22.0 / $0.323 |

Canonical ANTMAN/LongAgent reference values again reproduced exactly
(calls 33.9/35.9/33.6 and 24.7/40.0/76.2; cost $0.104/$0.135/$0.148 and
$0.104/$0.181/$0.355).

**128K ratios:**
- CoA/ANTMAN: worker ratio 6.00, call ratio 0.65, cost ratio 2.18
- LongAgent/ANTMAN: worker ratio 21.96, call ratio 2.27, cost ratio 2.40

CoA makes *fewer* physical calls than ANTMAN at 128K (22.0 vs 33.6) even
though it has 6× the "worker" count — each CoA worker call is one
occurrence, while ANTMAN's 33.6 calls come from a smaller number of
workers each making multiple calls (search/reasoning rounds). Worker
*count* and call *count* measure different things for these two methods;
neither ratio alone characterizes relative cost, which is why cost is
reported separately.

## F. Growth ratios 32K→128K

| Method | Active-agent growth | Call growth | Cost growth |
|---|---:|---:|---:|
| ANTMAN | 1.25 | 0.99 | 1.42 |
| LongAgent | 3.85 | 3.09 | 3.41 |
| CoA | 4.00 | 3.14 | 4.00 |

Both exhaustive baselines (LongAgent: static partition; CoA: sequential
chain) show active-agent/call/cost growth in the same rough band
(3.1–4.0×) as context grows 4×, tracking information-space size fairly
closely. ANTMAN's growth is far smaller across all three (0.99–1.42×).
This is the direct, measured answer to Section 14's question — reported
as the observation itself, not as a claim about which architecture is
"better."

## G. Position analysis (CoA)

| Length | Early F1 | Middle F1 | Late F1 | gap_F1 (M−mean(E,L)) | gap_EM |
|---|---:|---:|---:|---:|---:|
| 32K | 0.793 | 0.680 | 0.740 | −0.087 | +0.000 |
| 64K | 0.747 | 0.630 | 0.800 | −0.143 | −0.300 |
| 128K | 0.690 | 0.480 | **0.101** | +0.084 | +0.200 |

**Paired analysis across all 15 (length, question) pairs, Middle −
mean(Early, Late):**

| Metric | n | Mean | Median | Improved | Degraded | Tied | 95% CI (seed=42, 10k reps) |
|---|---:|---:|---:|---:|---:|---:|---|
| F1 | 15 | −0.0485 | 0.0000 | 2 | 7 | 6 | [−0.1600, +0.0809] |
| EM | 15 | −0.0333 | 0.0000 | 2 | 2 | 11 | [−0.2000, +0.1333] |

Both CIs cross zero — **no statistically credible Middle-specific penalty
for CoA** in this sample (consistent with Section 14: not overinterpreted
as "no Lost-in-the-Middle" either; the sample is small and noisy, tied
counts dominate).

The one thing the paired Middle-vs-outer framing does *not* capture:
**CoA's Late position at 128K specifically collapses** (F1 0.101,
substantially below its own Early 0.690 and Middle 0.480 at the same
length, and below every other length's Late score). This is a Late-,
not Middle-, positional effect, length-interacted, and outside the scope
of what Section 10's Middle-vs-outer analysis was designed to detect.
Reported as a distinct, real, measured fact — not folded into a
Lost-in-the-Middle claim, per Section 14's explicit instruction not to.

## H. Fidelity audit

Inspected one real trace per length (32K/64K/128K `early_0`), cross-checked
against each trace's own logged metadata (not re-asserted from code):

| Check | 32K | 64K | 128K |
|---|---|---|---|
| n_chunks == n_workers == worker_calls == len(worker trace steps) | ✓ (5) | ✓ (10) | ✓ (20) |
| Exactly one manager step | ✓ | ✓ | ✓ |
| Chunk index strictly increasing (source order preserved) | ✓ | ✓ | ✓ |
| Every CU ≤ 512-token cap | ✓ | ✓ | ✓ |
| total_physical_calls == worker_calls + manager_calls + 1 (condense) | ✓ (7=7) | ✓ (12=12) | ✓ (22=22) |
| No internal metadata key names (`needle_doc_ids`, `niah_metadata`) in trace text | ✓ | ✓ | ✓ |

`sum(chunk_token_counts)` is slightly below `source_token_count` in all
three (35,752 vs 36,232; 71,291 vs 72,221; 142,767 vs 144,656) — expected
and not a violation: chunk text is reconstructed via `" ".join(sentences)`
(single-space joins), which collapses the original's redundant
whitespace/newlines before re-encoding; `verify_full_coverage`'s own
assertion inside `run()` (whitespace-insensitive, content-exact) passed
for all 45 conditions or the run would have crashed. **No fidelity
violation found. No STOP triggered. All 45 results remain canonical.**

(The worker-isolation invariants — Wi sees only `ci`/`CU_{i-1}`/`q`, the
manager sees only `CU_l` — were verified structurally by the 15 unit
tests in `tests/test_chain_of_agents.py`, run against this exact,
unmodified, frozen code before any paid call in the original commit
`a3b6310`; this section's post-run check confirms the real traces are
consistent with what those tests already proved about the code path
itself.)

## I. Short factual summary

- CoA's worker count scales exactly linearly with context length (5 →
  10 → 20, exactly 2×/doubling) by construction (fixed per-chunk budget).
- CoA and LongAgent (the two exhaustive baselines) both show call/cost
  growth in the 3.1–4.0× band across a 4× context increase; ANTMAN's
  growth is 0.99–1.42× over the same range. Measured, not a claim about
  which is "better."
- CoA's 128K F1 (0.424) is the lowest among all six methods at any
  length in this matrix; its 32K/64K F1 (0.738/0.726) is mid-pack.
- CoA's Late position at 128K collapses specifically (F1 0.101) — a
  length-interacted Late effect, not a Middle effect.
- The paired Middle-vs-(Early,Late) bootstrap shows no statistically
  credible Middle-specific penalty for CoA (both F1 and EM CIs cross
  zero, n=15).
- At 128K, CoA makes fewer physical calls than ANTMAN (22.0 vs 33.6) but
  costs more per query ($0.323 vs $0.148) — call count and cost do not
  move together across these two methods.
- Preflight confirmed all 45 conditions read byte-identical (content-wise)
  canonical, post-fix, leakage-clean environments; one shared, disclosed,
  content-free `.materialized`-file artifact affects CoA and LongAgent
  identically and was not fixed (out of scope for this frozen pass).
- All 45 conditions completed with 0 errors and 0 retries; fidelity audit
  of 3 traces found no invariant violations.
