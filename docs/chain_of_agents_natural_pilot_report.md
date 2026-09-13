# Chain-of-Agents added to the preliminary natural multi-document QA pilot

Frozen baseline-completion run, not method development. CoA's
implementation (`src/ant/external_wrappers/chain_of_agents.py`, unchanged
since commit `a3b6310`), the shared `run_suite` harness, dataset sampling,
document assembly, answer extraction, and scoring were all reused
unmodified. No other method was touched. No code file changed in this
pass (verified: `git status` shows no modified/new source files).

## A. Run status

- **45/45 conditions completed** (15 HotpotQA + 15 2WikiMultihopQA + 15
  MuSiQue), 0 errors, 0 retries.
- Total physical calls: 135 (45 × 3 — 1 worker + 1 manager + 1 condense
  each, since every example fit in a single chunk; see Section B).
- Total cost: **$0.2512** (generation) + **$0.0289** (post-hoc
  answer-extraction rescoring) = **$0.2801 total** — well under the
  $0.3814 pre-run estimate and the $2.00 hard cap.
- Preflight (before any paid call): all 45 conditions verified against
  the exact frozen 15-task-per-benchmark ID sets already used by the
  other 5 methods (`third_party/manifests/long_context/
  natural_pilot_manifest_{15task,10new_per_benchmark}.json`) — same
  `task_id`s, same questions, same already-materialized environments
  (idempotent reuse via each benchmark adapter's own
  `prepare_environment`). One false-positive was caught and corrected
  during preflight, not silently ignored: an initial overly-strict
  allowed-metadata-key check flagged `supporting_doc_ids`/`question_type`/
  `level`/`answerable` as "unexpected" — these are legitimate,
  pre-existing benchmark-scoring fields already present on every one of
  the other 5 methods' identical `TaskExample.metadata` (read only by
  `benchmark.score()`, never by any agent's `run()`); the check was
  corrected to the real schema and re-verified clean (0 problems / 45).
- Result paths: `output/runs/natural-multidoc-pilot/{hotpotqa,
  2wikimultihopqa,musique}/chain_of_agents.jsonl` (raw, via `run_suite`),
  `chain_of_agents.rescored.jsonl` (extracted, canonical),
  `trajectories/chain_of_agents-*.json` (45 full traces).
- Commit for this report: see end of document.

## B. CoA per-benchmark results

| Benchmark | Avg source tokens | Avg workers | Calls/query | Cost/query | EM | F1 |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 1,613 | 1.00 | 3.00 | $0.0051 | 0.600 | 0.793 |
| 2WikiMultihopQA | 1,247 | 1.00 | 3.00 | $0.0048 | 0.667 | 0.820 |
| MuSiQue | 2,446 | 1.00 | 3.00 | $0.0068 | 0.533 | 0.662 |

Every one of the 45 examples fit inside a single 7,406-token chunk
budget — natural multi-document contexts here (10 short Wikipedia-style
documents) are far smaller than the 32K–128K controlled scaling
conditions. Per Section 4's instruction, this was **not** artificially
forced into a multi-worker shape: `n_chunks=1` was recorded faithfully
for all 45.

## C. Full six-method preliminary table (extracted EM/F1)

| Benchmark | Metric | Direct | Retrieval | ReAct | LongAgent | CoA | ANTMAN |
|---|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | EM | 0.667 | 0.600 | 0.533 | 0.533 | 0.600 | 0.600 |
| HotpotQA | F1 | 0.817 | 0.749 | 0.744 | 0.758 | **0.793** | 0.746 |
| 2WikiMultihopQA | EM | 0.600 | 0.467 | 0.467 | 0.733 | 0.667 | **0.800** |
| 2WikiMultihopQA | F1 | 0.689 | 0.519 | 0.737 | **0.853** | 0.820 | 0.812 |
| MuSiQue | EM | 0.467 | 0.467 | 0.400 | 0.267 | **0.533** | 0.467 |
| MuSiQue | F1 | 0.633 | 0.538 | 0.536 | 0.309 | **0.662** | 0.596 |

(Direct/Retrieval/ReAct/LongAgent/ANTMAN numbers loaded unmodified from
the existing canonical `*.rescored.jsonl` files — not rerun.)

## D. Macro average (equal-weight across the 3 benchmarks, n=15 each)

| Method | Avg EM | Avg F1 |
|---|---:|---:|
| Direct | 0.578 | 0.713 |
| Retrieval | 0.511 | 0.602 |
| Matched ReAct | 0.467 | 0.673 |
| LongAgent | 0.511 | 0.640 |
| **CoA** | **0.600** | **0.758** |
| ANTMAN | 0.622 | 0.718 |

CoA has the **highest macro F1** of all six methods and the second
highest macro EM (behind ANTMAN by 0.022). Reported as the measured
result; per Section 12, this is a preliminary general-effectiveness
number on n=15/benchmark, not a scaling or superiority claim.

## E. Fidelity audit

One real trace inspected per benchmark (HotpotQA `5a8b57f2...`, 2Wiki
`8813f87c...`, MuSiQue `2hop__460946_294723`):

| Check | HotpotQA | 2Wiki | MuSiQue |
|---|---|---|---|
| n_chunks == n_workers == worker_calls == worker trace steps | ✓ (1) | ✓ (1) | ✓ (1) |
| Exactly one manager step | ✓ | ✓ | ✓ |
| total_physical_calls == worker_calls + manager_calls + 1 (condense) | ✓ (3=3) | ✓ (3=3) | ✓ (3=3) |
| No internal metadata keys (`supporting_doc_ids`, etc.) in trace text | ✓ | ✓ | ✓ |

**No fidelity violation found. No STOP triggered. All 45 results
canonical.**

One pre-existing, already-documented extractor limitation surfaced once
(1/45, MuSiQue `2hop__460946_294723`): raw answer `"No spouse is
mentioned for Grant Green. Steve Hillage, who released the album
'Green', has a partner named Miquette Giraudy."` was deterministically
normalized to `"no"` by the extractor's yes/no regex (`^\s*(yes|no)\b`),
because the sentence happens to start with "No" as ordinary prose, not a
yes/no answer — dropping the real answer content (`raw_f1=0.20` →
`extracted_f1=0.0`). This is the frozen extractor's own known
behavior (unchanged this pass, out of scope to fix here), not a CoA
defect and not a hallucination (`rejected_hallucination=False` — the
deterministic path never calls the LLM or the hallucination gate at
all). 44/45 rows were unchanged-or-improved by extraction.

## F. Short factual summary

- CoA is competitive on short natural multi-document QA: highest macro
  F1 (0.758) and second-highest macro EM (0.600) of the six methods on
  this n=15/benchmark preliminary set.
- All 45 examples collapsed to a single CoA worker (1 chunk) — natural
  QA contexts here are far smaller than the controlled 32K–128K scaling
  conditions; this was recorded faithfully, not forced otherwise.
- No benchmark showed a surprising CoA failure; MuSiQue's single
  worsened-by-extraction case is a known, pre-existing, frozen-extractor
  edge case (a "No..." sentence-opener misread as a yes/no answer), not
  a CoA-specific issue.
- No scaling claim is made from this experiment — that conclusion
  belongs only to the separate 32K/64K/128K controlled multineedle
  matrix (`docs/chain_of_agents_multineedle_scaling_report.md`).
- Total cost $0.28, well under the $2 cap; 0 errors, 0 retries across
  all 45 conditions.
- Fidelity audit of 3 traces (one per benchmark) found no invariant
  violations; all 45 results are canonical.
