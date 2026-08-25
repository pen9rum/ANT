# Scoring heuristics

Every hand-tuned numeric constant in ANT's evidence reranking, worker
routing, colony evolution, and route-quality logic lives in one file:
[`src/ant/scoring_config.py`](../src/ant/scoring_config.py). This document
explains where each group of constants is used, why it exists, and which
ones are the weakest-justified (i.e. the first candidates for a sensitivity
pass before treating any single value as load-bearing in a paper claim).

None of these numbers are learned or derived from a formal model. They were
chosen empirically while building the corresponding mechanism and have not
individually been validated for sensitivity. Centralizing them did not
change any value or any system behavior (verified: the full test suite
passes unchanged before and after) — it exists so there is one place to
point to, one place to sweep, and one place to extend if a value becomes
runtime-adjustable later.

## Why this exists

Before this file, these constants were scattered as inline literals across
five modules (`ant.retrieval.relevance`, `ant.coordinator.local`,
`ant.evolution`, `ant.evaluation.runner`, `ant.retrieval.dense`). That made
two things hard:

1. **Defending a specific number.** "Why 12?" required finding the line it
   lives on and reading the surrounding comment, if one existed, in
   whichever of five files happened to hold it.
2. **Sweeping a value.** Testing "what if the dense-score bonus were half as
   strong" meant editing source and re-running, not passing a different
   config into an experiment.

## Groups

### `EvidenceScoringConfig` — `score_evidence()` (`ant/retrieval/relevance.py`)

The one canonical reranker for candidate evidence inside a worker's
territory (search hits, dense hits, symbol lookups all flow through it).
Additive point bonuses/penalties per signal: class/def presence, a
"definition"/"implementation" hint in the retrieval reason, source-path and
`.py`-extension bonuses, low-value-path penalties, a stemmed symbol-name
match, per-term lexical overlap, and `dense_score_weight` — the multiplier
on a 0..1 dense cosine similarity, i.e. the maximum a purely-semantic,
zero-lexical-overlap match can contribute.

`score_evidence()` takes `config: EvidenceScoringConfig` as an explicit
keyword argument (default `DEFAULT_SCORING_CONFIG.evidence`), so a
sensitivity pass can construct an alternate config and pass it directly
without editing source:

```python
from dataclasses import replace
from ant.scoring_config import DEFAULT_SCORING_CONFIG

weaker_dense = replace(DEFAULT_SCORING_CONFIG.evidence, dense_score_weight=6)
score_evidence(quote=..., path=..., terms=..., config=weaker_dense)
```

### `WorkerRoutingConfig` — `LocalCoordinator._rank_worker_scores()` and helpers (`ant/coordinator/local.py`)

The largest group, because worker routing is the most heuristic-dense part
of the pipeline. Covers: relevant-symbol weighting (inversely scaled by how
many workers a symbol matches), the source-worker follow-up bonus, the
seen-worker anti-repetition penalty, source-path/test-path ratio bonuses and
penalties, the memory-route replay bonus (with its minimum-overlap-ratio
gate), territory-hint bonuses, `dense_routing_weight` (the worker-card-level
counterpart to evidence's `dense_score_weight`), and the "genuinely close"
ratio used to decide whether a runner-up worker is recruited alongside the
top pick.

This group is read directly from `DEFAULT_SCORING_CONFIG.routing` inside
each function rather than threaded through as an explicit parameter (unlike
`score_evidence`) — routing runs through half a dozen small module-level
helper functions, and threading a config object through all of them for a
first pass wasn't worth the surface area. Swapping values for an experiment
currently means monkeypatching `ant.scoring_config.DEFAULT_SCORING_CONFIG`
(module-level, so patch before the coordinator is constructed) rather than
passing a config in per-call.

### `EvolutionConfig` — `evolve_workers()` (`ant/evolution.py`)

Thresholds for specialization (`min_specialization_routes`,
`min_specialization_group_routes`), coalition birth
(`min_coalition_count`), and merge (`merge_overlap_threshold`). These were
*already* exposed as `evolve_workers()`'s own keyword arguments before this
refactor — this file only centralizes their **defaults** so the "why 4, why
0.9" answer lives next to every other constant instead of only in the
function signature. Passing explicit values to `evolve_workers()` (as the
two-pass qibo/seaborn experiments do) still works exactly as before.

**`min_routes_for_health_check` / `healthy_route_ratio`** (added after a
real observed regression, not part of the original design): specialize,
birth, and merge all now skip touching a worker (for merge, either side of
a candidate pair) that already has at least `min_routes_for_health_check`
recorded routes with at least `healthy_route_ratio` of them flagged
`is_high_quality`. This was added after a three-generation repeated-evolve
experiment on qibo showed a concrete failure: cycle 2's merge folded an
already-adequate bridge worker into an overlapping-but-unrelated,
test-file-heavy bridge worker, and a question a pre-merge generation had
answered well (citing `distcircuit.py`'s actual implementation, correctness
8/10) dropped to citing only test-file assertions (correctness 4/10) once
the merge routed it to the diluted worker instead. Structural signals
(directory overlap, accumulated route *count*, recurring coalition count)
were all satisfied in that case; nothing checked whether the worker being
touched was already working. `is_high_quality` is derived from
correctness+completeness only (see `RouteQualityConfig` below), not
relevance/clarity — a worker that returns relevant-looking, well-written
but factually incomplete answers still counts as struggling, which is
deliberate: the point of specialize/birth is to dig into a topic further
when the *substance* isn't there yet, not when the surface signals already
look fine.

**`negative_presence_gate_ratio` (0.5)** — added alongside a broader fix:
`record_task_memory()` used to collapse every unresolved need from a task
into one aggregate route carrying only the original question's terms and a
single pass/fail flag, discarding the `UnresolvedNeed`'s own `need_type` and
`scope`. `evolve_workers` could see *that* a worker struggled but never
*what kind* of gap it struggled with. `MemoryRoute` now also gets a
per-need route (need_type/scope populated, `is_high_quality=False` by
construction, never visible to `matching_routes`) for each need still open
when a task ends. This gate is the first consumer: once at least half of a
worker's typed routes are `need_type="negative_presence"` (confirmed
directly on real questions — qibo asked about a tool not in that codebase;
seaborn asked about doc-build performance with no matching implementation),
specialize skips it, since reorganizing territory boundaries cannot fix "the
information was never in this repo." 0.5 (a bare majority) is unvalidated
like every other ratio in this file — the qualitative direction (skip a
majority-absence worker) is well-evidenced, the exact cutoff is not.

### `RouteQualityConfig` — eval runner (`ant/evaluation/runner.py`)

Decides which finished answers are trustworthy enough to feed back into
colony memory as a "high-quality" route (`correctness >= 8 and completeness
>= 8`, or an exact/contains match), and how much weight a recorded route
carries (`(correctness + completeness) / route_weight_divisor`).

### `DenseRetrievalConfig` — `ant/retrieval/dense.py`

`symbol_snippet_max_lines` (20): the hard cap on how many lines of a symbol
get embedded, regardless of the symbol's actual length. `embed_batch_size`
(256): purely a throughput/progress-reporting granularity, not a quality
lever.

## Weakest-justified values (start here for a sensitivity pass)

- **`symbol_snippet_max_lines = 20`.** The sharpest edge in this file. A
  function whose semantically distinguishing logic sits past line 20 of its
  body (e.g. behind a long parameter-validation preamble or docstring) is
  represented by an embedding that doesn't cover that logic at all. No
  principled justification for 20 specifically — it was chosen as "roughly a
  signature + docstring + a few lines," not validated against retrieval
  quality on long functions.
- **`dense_score_weight = 12` vs. `dense_routing_weight = 10`.** Two
  independently-chosen multipliers for conceptually the same thing (how much
  a 0..1 cosine similarity should weigh against lexical signals), tuned
  separately at the evidence-reranking layer and the worker-routing layer.
  Nothing currently checks or enforces that they should agree.
- **`merge_overlap_threshold = 0.9`.** A single global threshold for "these
  two workers' file sets overlap enough to merge," applied uniformly
  regardless of territory size — a 2-file worker and a 200-file worker are
  held to the same overlap ratio.
- **`min_specialization_routes = 4` / `min_specialization_group_routes =
  2`.** Directly determined this session's qibo evolve run producing 9
  `birth` events and zero `specialize` events (see the two-pass experiment
  write-up) — a different pair of numbers could plausibly have produced a
  different mechanism mix entirely, which is exactly why it's worth
  reporting as a swept parameter rather than a fixed given, if evolution
  behavior becomes a paper claim.
- **`min_routes_for_health_check = 3` / `healthy_route_ratio = 0.7`.** The
  motivating regression (see `EvolutionConfig` above) is real and the
  direction of the fix is well-evidenced, but these exact two numbers
  aren't: 3 routes is a small sample to call a worker's quality "settled,"
  and 70% is an unvalidated cutoff between "struggling" and "fine." A
  worker with 2 good routes and 1 bad one currently reads as unhealthy
  (2/3 = 0.67 < 0.7); one route flipping either way changes the verdict.

## Running a sensitivity/ablation pass

For `score_evidence`, construct an alternate `EvidenceScoringConfig` (or use
`dataclasses.replace` on `DEFAULT_SCORING_CONFIG.evidence`) and pass it via
the `config=` keyword — no monkeypatching needed.

For everything else (`WorkerRoutingConfig`, `DenseRetrievalConfig`, and any
direct use of `DEFAULT_SCORING_CONFIG.route_quality`), the values are read
from the shared `DEFAULT_SCORING_CONFIG` singleton at call time, so an
experiment script can monkeypatch the module-level object before running:

```python
import ant.scoring_config as scoring_config
from dataclasses import replace

scoring_config.DEFAULT_SCORING_CONFIG = replace(
    scoring_config.DEFAULT_SCORING_CONFIG,
    dense=replace(scoring_config.DEFAULT_SCORING_CONFIG.dense, symbol_snippet_max_lines=40),
)
```

Do this once per experiment process, before constructing any
`LocalCoordinator`/calling `evolve_workers`/etc. — several call sites read
`DEFAULT_SCORING_CONFIG` at call time (not import time), but a couple of
`evolve_workers`'s parameters resolve their default at *function-definition*
time (import time), so patching after `ant.evolution` has already been
imported won't retroactively change its keyword defaults — pass the desired
values to `evolve_workers()` explicitly in that case instead of relying on
the monkeypatch.
