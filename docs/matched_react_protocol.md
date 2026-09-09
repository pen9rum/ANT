# Matched ReAct baseline protocol (frozen)

Status: **frozen protocol specification**, written before any formal
benchmark evaluation run. This document exists so the protocol is
inspectable and reviewable independent of the code, and so a future change
to the code's actual behavior is a visible diff against this file, not a
silent drift. It describes `src/ant/agents/matched_react.py`'s
`MatchedReActAgent` as of commit-time; if the code and this document ever
disagree, that is a bug in one of the two, not an ambiguity to interpret
away.

## Generation model

`gpt-4.1` (constructor default `model="gpt-4.1"`), for every question, in
every run. Not a reasoning model -- no `reasoning.effort`/`text.verbosity`
controls apply (those are judge-only, see `evaluation_suite/judge.py`).

## Compute budget: PRIMARY protocol

- **Global tool-call cap: 50** (`DEFAULT_TOOL_CALL_BUDGET`), fixed once at
  agent construction, applied identically to every question in a
  benchmark run. Derived from the measured mean of 16 real ANT traces
  across 8 repos (qibo/seaborn/sphinx/yt-dlp/pennylane/sqlfluff/sanic/
  Pillow, this session's own no-evolution-runtime validation): mean 51.6
  tool calls/question, rounded to 50 -- a single number fixed BEFORE
  looking at any individual question this cap is later applied to, not
  derived from the questions it will be scored against.
- **No per-question ANT-derived budget.** `run()` reads no
  question-specific field to set its own budget. An earlier version of
  this class read `example.metadata["matched_react_tool_call_budget"]` --
  a retrospective per-question override equal to that exact question's
  own already-measured real ANT tool-call count -- and the actual smoke-
  test driver that produced this session's first 4-question SWE-QA-Pro
  scores was using it. That mechanism has been removed from `run()`
  entirely; it no longer exists as a silent code path.
- **No separate LLM-call cap.** The loop bound (`for step in
  range(budget)`) caps the number of DECISION *iterations* to at most
  `budget`; an iteration is not the same thing as a physical model call
  (see below). One additional forced-synthesis call may occur if the
  budget is exhausted without an explicit `finish` (see Early stopping,
  below) -- not counted against the 50-iteration budget, a genuine extra
  call.
- **`llm_calls` counts physical model/API invocations, not decision
  iterations.** `run()` uses `CountingOpenAIProvider`
  (`evaluation_suite/counting_provider.py`), which overrides
  `responses_text()` -- the sole choke point every physical call goes
  through, including `responses_json()`'s own internal JSON-repair pass
  when a step's raw response wasn't valid JSON on the first try. An
  earlier version of this method counted `llm_calls` with a manual "+1
  per loop iteration," which undercounted by one for every repair that
  actually fired (tokens/cost were always correct via `drain_usage()`;
  only this diagnostic counter was wrong). Fixed: `llm_calls =
  provider.drain_call_count()`, read once at the end of `run()`. Since a
  repair pass can occur, `llm_calls` can now legitimately exceed the
  50-iteration budget -- that is correct, not a bug.
- **Compute-matched (retrospective per-question) comparison is SECONDARY
  ONLY**, and not implemented in this pass. If it returns later, it must
  be an explicit, separately-labeled analysis (e.g. a wrapper constructing
  a distinct `MatchedReActAgent` per question) -- never a per-example
  metadata lookup inside the shared agent's own `run()`.

## Per-tool result limits

Default: `ANT_PARITY_TOOL_LIMITS` -- copied exactly from
`AutonomousWorker`'s own real, operative per-call limits (the
reasoner-driven `_execute_tool` path, the only one the real `AntAgent`
baseline ever exercises in this harness):

| tool | limit |
|---|---|
| search | 4 |
| dense_search | 4 |
| navigate | 2 |
| references | 2 |
| callers | 2 |
| callees | 2 |
| assignments | 2 |
| imports | 2 |
| subclasses | 4 |

An earlier version of this class called every tool with a uniform
`limit=6`, justified after the fact as "generous compensation" for being
a single agent -- that justification was never specified anywhere before
being written down, so per the fairness-closure audit's own rule it does
not count as a prior experimental reason to keep an unequal primitive.
`ANT_PARITY_TOOL_LIMITS` is now the default.

A deliberately more generous **alternate, sensitivity-only** configuration
exists (`GENEROUS_SENSITIVITY_TOOL_LIMITS`, uniform 6 for every tool) --
selectable only via an explicit `tool_result_limits=` constructor
argument, never the default, and not run as part of this pass.

## Repository scope

Both Matched ReAct and Retrieval (and, as of this pass, ANT's own
`AntAgent`) build their file listing from `EvalRepoEnvironment(
environment_root).iter_files()` (`evaluation_suite/repo_scope.py`) -- the
identical scope definition all three now use, in place of ANT core's own
closed `TEXT_EXTENSIONS` allowlist. See `docs/repo_file_universe_policy.md`
for the full extension-distribution audit and the general, content-based
text-detection policy this implements (not an ever-growing extension
list). This IS implementation parity with ANT, achieved entirely
evaluation-side -- `ant/environment/repo.py` (frozen ANT core) was never
modified; ANT's default runtime outside this evaluation suite still uses
its own `RepoEnvironment`/`TEXT_EXTENSIONS` unchanged.

## Context limit / observation truncation policy

- Each decision call is capped at `max_output_tokens=512` for the model's
  own response.
- The observation prompt shown at each step (`_format_history`) includes
  **every** prior step in the run (no step-count window), but for each
  step shows at most 5 tool results, each result's quote truncated to 200
  characters. This differs from `RetrievalAgent`'s own windowing (last 8
  evidence items, 300-char quotes) -- a legitimate difference: Retrieval
  has only 3 rounds total, so no long-history growth is possible; Matched
  ReAct's budget (up to 50 steps) makes an unbounded step-count window a
  real, if not yet observed, growth risk.
- No explicit input-token cap is enforced by this harness itself; it
  relies on GPT-4.1's own context window (1,047,576 tokens). **Statically
  quantified** (no new LLM calls; real per-step history entries from the
  largest real trajectory on disk -- 26 real steps -- were reused/cycled
  to extrapolate to the full 50-step budget): worst-case prompt size at
  the final step of a full 50-step run is ~37,300 characters, roughly
  9,000-12,000 tokens depending on the chars-per-token estimate used --
  about 1% of GPT-4.1's context window. Normal execution **cannot
  plausibly exceed the model's context** under the current protocol; this
  is a documented, quantified conclusion, not an untested assumption. If
  a future full run somehow hits a real context-limit failure anyway
  (e.g. a pathological question producing unusually long tool-result
  quotes), that would itself be a genuine, reportable finding -- not
  something to pre-emptively guard against by inventing a new truncation
  policy in this pass.

## Early stopping semantics

The agent may emit `{"finish": "<answer>"}` at any step; the loop breaks
immediately and unused budget is simply never spent (observed empirically
in development smoke runs: as few as 5/50 steps used before a voluntary
finish). If the budget is exhausted with no explicit `finish`, one forced
`synthesize()` call runs over whatever evidence was gathered (mirroring
ANT's own "an explicit abstention beats silence" posture) -- this call is
extra, not deducted from the 50-call budget.

## Failed-call accounting

- A decision that is not valid JSON, or valid JSON missing both `tool`
  and `finish` (after `_normalize_decision`'s shorthand-key tolerance --
  see the fairness-closure Matched-ReAct-competence audit for why this
  exists), still consumes one unit of the tool-call budget (one loop
  iteration) with zero tool executed and an empty `results: []` recorded
  in the trajectory. This mirrors a real ReAct agent's own confusion
  being genuine, comparable overhead -- it is not silently retried for
  free.
- No `try`/`except` wraps tool execution. An exception raised by a tool
  call propagates out of `run()` entirely, uncaught -- relying on the
  evaluation harness's own per-example failure isolation
  (`evaluation_suite/runner.py`'s `run_suite`) to keep the rest of a batch
  alive. This is symmetric with `AutonomousWorker`, which also has no
  per-call exception handling -- not a Matched-ReAct-specific weakness.

## Change log

- Initial version written after: the decision-schema parsing fix, the
  budget-protocol fix (removing the per-question metadata override), the
  tool-parity fix (`callers` now tries `indexed_callers` first), the
  repository-scope fix (both baselines on `RepoEnvironment.iter_files()`),
  and the per-tool result-limit normalization (`ANT_PARITY_TOOL_LIMITS`
  as the default).
- Updated after: the physical-LLM-call-accounting fix
  (`CountingOpenAIProvider`, `llm_calls` now counts physical model calls
  including `responses_json()`'s repair pass, not decision iterations),
  the content-based repository-file-universe policy replacing
  `RepoEnvironment.iter_files()` with `EvalRepoEnvironment.iter_files()`
  (see `docs/repo_file_universe_policy.md`), and the static context/budget
  sanity check (worst-case ~1% of GPT-4.1's context window at budget=50 --
  no overflow risk under normal execution). Any further change to
  `matched_react.py`'s protocol behavior should update this file in the
  same commit.
