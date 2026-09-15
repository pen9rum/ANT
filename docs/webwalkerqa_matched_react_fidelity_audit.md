# Matched ReAct Web fidelity audit and correction

**Purpose of this pass**: the original N=60 Matched ReAct Web run scored
11.7% (7/60), far below WebWalkerQA's own published ReAct baseline
(~33.8% overall, GPT-4o). This document diagnoses why, applies only
paper-fidelity corrections (no performance tuning, no gold-answer
optimization), and reports the corrected result. ANTMAN has not been run
yet — that is deliberately deferred until this baseline is confirmed
healthy (see the Recommendation, section G).

## Provenance (never overwritten)

1. **Original mismatched ReAct** — `output/runs/webwalkerqa-matched-react-web-n60/`
   (11.7%, 7/60). Frozen, untouched.
2. **Fidelity audit** — this document + the research/analysis that
   produced it.
3. **Corrected ReAct validation (N=20)** —
   `output/runs/webwalkerqa-matched-react-web-corrected-v20/`.
4. **Corrected ReAct full (N=60)** —
   `output/runs/webwalkerqa-matched-react-web-corrected-full60/`
   (28.3%, 17/60). New namespace, new page-cache namespace
   (`.ant/eval-suite-web/webwalkerqa-corrected/`), same frozen manifest
   (`matched_react_web_n60.json`), same GPT-4.1, same 15-step budget.

## A. Fidelity comparison

| Component | Official ReAct | Original ours | Corrected ours |
|---|---|---|---|
| Observation | Markdown/text + `(button_text, url)` pairs, no truncation | Bare URLs only, text truncated to 3000 chars (avg. 22.3% of raw page retained) | `(anchor_text, url)` pairs; no arbitrary truncation (real API-safety guard only, never triggered in this run) |
| Links/buttons | BeautifulSoup-extracted `(text, url)`, incl. some `onclick` JS buttons | Bare URL list, zero anchor text | `(text, url)` pairs from `<a href>` (JS-button capture out of scope for this pass, disclosed) |
| History | Full raw (Thought, Action, Observation) accumulated every step, never truncated | One-line `"[N] navigated to URL (status)"` summary only — no page content retained | Full (Thought, Action, Observation) blocks retained every step, matching official's own accumulation semantics |
| Step budget | 15 (paper-stated, applies to all baselines) | 15 | 15 (unchanged — not touched per your explicit instruction) |
| Navigation | Click only, `urljoin(ROOT_URL).startswith(ROOT_URL)` string-prefix domain restriction; **no code exists confirming meta-refresh handling, but Crawl4AI's real browser rendering makes it inherent** | Click only, host-based same-site restriction (no path-prefix); did NOT follow client-side meta-refresh redirects | Same host-based restriction (a disclosed, minor divergence — official is path-prefix-scoped for 8/60 tasks, ours is domain-scoped; this makes ours *more* permissive, not harder); now follows meta-refresh (verified universal browser-standard behavior, bounded to 5 hops) |
| Prompt | Not released for the plain ReAct baseline (only WebWalker's own Explorer+Critic prompts exist) | This suite's own JSON-action-schema prompt (unavoidable — no official prompt to reproduce) | Unchanged prompt structure; only the *content it's fed* was corrected |
| Judge | Single GPT-4 CoT-QA call, no majority vote, paper-published exact prompt | 3× GPT-5 majority vote, this suite's own prompt | Unchanged (out of scope for this pass — see root-cause ranking; judge audit found it substantively defensible, not the primary driver) |

## B. Root causes of the original 11.7%, ranked (from the pre-fix audit, evidence recapped)

1. **Missing link anchor text** (implementation bug) — bare URLs shown
   instead of `(text, url)` pairs. Directly evidenced by all 9 original
   step-budget-exhaustion cases showing heavy repeated navigation (up to
   13/15 steps revisiting only 2-3 unique pages).
2. **Aggressive page-text truncation** (implementation bug) — only 22.3%
   of raw page text reached the model; official's own design deliberately
   avoids truncation via large-context backbones.
3. **Compressed history retention** (implementation/environment mismatch)
   — one-line URL summaries vs. official's full accumulated
   (Thought, Action, Observation) context.
4. **Live-web/environment drift** — narrow (a confirmed meta-refresh bug
   broke 2/60 tasks completely; most "inaccessible page" cases are PDFs,
   plausibly inaccessible to official's own pipeline too).
5. **Evaluation/judge mismatch** — real protocol difference, but a manual
   audit of all 7 correct + 15 incorrect original predictions found the
   judge's decisions substantively defensible in nearly every case.
6. **Genuine model difference** (GPT-4.1 vs. paper's GPT-4o/Qwen
   backbones) — real and unquantified. Model difference remains
   possible, but it is unlikely to explain the full 21.5-point gap given
   the confirmed observation and history mismatches (correcting your
   author's own note: not ruled out by "GPT-4.1 is generally stronger,"
   since agentic/browsing-task behavior doesn't reliably track general
   capability).
7. **Subset difficulty composition** — ruled out as a primary driver: the
   composition-adjusted expected accuracy (33.2%) barely differs from
   the paper's unweighted rate (33.8%).

**Confirmed post-fix**: causes #1-3 were the dominant drivers. Fixing
them alone (without touching the judge, the model, or the step budget)
moved accuracy from 11.7% to 28.3%, and step-budget exhaustion cases
dropped from 9/60 to 1/60 — direct, quantified corroboration that the
repeated-navigation pathology (caused by #1-3) was the primary bottleneck.

## C. Score comparison

| Run | N | Accuracy | Calls/q | Steps/q | Pages/q | Cost/q |
|---|---|---|---|---|---|---|
| Original Matched ReAct | 60 | 11.7% | 8.65 | 7.18 | 5.22 | $0.0501 |
| Corrected ReAct validation (same-20 old) | 20 | 35.0%* | 7.25 | 6.05 | 5.10 | $0.0463 |
| Corrected ReAct validation | 20 | 40.0% | 6.75 | 4.30 | 4.50 | $0.2302 |
| Corrected ReAct full | 60 | **28.3%** | 7.45 | 4.72 | 4.98 | $0.2467 |

\* The validation slice was deliberately stratified to include every
originally-correct case per difficulty bucket, so its OWN old-accuracy
(35.0%) is not comparable to the full-60 original rate (11.7%) — this is
a known artifact of the selection rule (disclosed at selection time), not
a discrepancy in the results. The fair full-corpus comparison is row 1
vs. row 4: **11.7% → 28.3%**.

Cost/query rose ~5x (no truncation + full accumulated history means
substantially larger prompts) — an expected, disclosed consequence of
the fidelity fixes, not a bug. Full-60 total cost: $14.80.

## D. Difficulty breakdown (corrected full-60)

| Difficulty | N | Score |
|---|---|---|
| Easy | 20 | 45.0% (9 correct) |
| Medium | 20 | 25.0% (5 correct) |
| Hard | 20 | 15.0% (3 correct) |

## E. Hop breakdown (corrected full-60)

| Hop type | N | Score |
|---|---|---|
| Single-source | 26 | 26.9% (7 correct) |
| Multi-source | 34 | 29.4% (10 correct) |

## F. Live-web validity

All 60 root URLs were re-verified accessible immediately before both the
validation-20 and full-60 corrected runs (live checks, not assumed from
the earlier environment-validation pass). Within the corrected full-60
trajectories, 18/60 (30%) encountered at least one inaccessible page
during navigation (predominantly PDF downloads — conference programs,
sponsorship brochures — which are plausibly inaccessible to the official
Crawl4AI pipeline too, since no PDF-extraction logic exists in the
released code either; this is a benchmark-level limitation, not
necessarily specific to our environment). Errors: 0/60. No task was
excluded or silently skipped. We cannot determine "fully solvable"
precisely without consulting gold paths (which this audit deliberately
never does) — 18/60 hitting a dead end partway through their trajectory
is the closest available proxy, and it did not prevent 17/60 from being
answered correctly.

## G. Recommendation

**Yes — the corrected Matched ReAct baseline (28.3%, 60/60 completed, 0
errors) is sufficiently faithful and healthy to use as the ANTMAN
comparison baseline.**

Basis:
- Three concrete, evidenced fidelity bugs were found and fixed (link
  labels, truncation, history retention), each independently justified
  by direct comparison against the official paper text and released
  Explorer code — not tuned for performance.
- The fix produced the exact behavioral signature the diagnosis
  predicted (step-budget exhaustion 9→1), not just a score change —
  strong evidence the correction addressed the real mechanism, not a
  coincidental improvement.
- 28.3% falls within your own pre-declared "~25-35%, reasonable
  baseline" acceptance band, and is close to (nine points below) the
  paper's own 33.8% overall GPT-4o rate — a plausible gap given genuine,
  disclosed differences (English-only subset with a harder, deliberately
  balanced difficulty mix; GPT-4.1 vs. GPT-4o/Qwen; live-web drift since
  the paper's 2025 data collection; a different, though audited-as-
  defensible, judge).
- Remaining known limitations (judge protocol difference, no JS-button
  capture, host- vs. path-scoped domain restriction) are disclosed, of
  low measured impact, and out of scope for this pass per your explicit
  "fidelity fixes only" instruction — not swept under the rug, just not
  blocking.

ANTMAN may now be run against this corrected baseline
(`output/runs/webwalkerqa-matched-react-web-corrected-full60/`), pending
your go-ahead.
