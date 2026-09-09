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
  range(budget)`) caps the number of DECISION calls to at most `budget`;
  each iteration makes exactly one decision call, so total decision LLM
  calls <= 50. One additional forced-synthesis call may occur if the
  budget is exhausted without an explicit `finish` (see Early stopping,
  below) -- not counted against the 50-call budget, a genuine extra call.
  A known, minor accounting nuance: `OpenAIProvider.responses_json`
  internally issues up to one additional "repair" API call per decision
  when the raw response isn't valid JSON -- that repair call's tokens/cost
  ARE included in the run's total usage (`drain_usage()`), but it does
  NOT increment `llm_calls` a second time (the field counts external call
  sites in `run()`'s own loop, i.e. iterations, not raw HTTP calls). This
  can cause `llm_calls` to under-report the true number of OpenAI API
  calls by up to 1 per malformed decision -- cost/token totals are not
  affected, only this one diagnostic counter.
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

Both Matched ReAct and Retrieval build their file listing from ANT's own
`RepoEnvironment(environment_root).iter_files()` -- the identical scope
definition ANT's own territory discovery uses (`IGNORED_DIRS` +
`TEXT_EXTENSIONS` allowlist). This is implementation PARITY with ANT, not
"unrestricted whole-repository access" -- see the fairness-closure repo-
scope audit for exact excluded-extension counts (`.rst`/`.sql` in
particular) and why this is a shared evaluation-infrastructure limitation,
not fixed in this pass (`RepoEnvironment` is frozen ANT core).

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
  relies on GPT-4.1's own context window. This has not been stress-tested
  at high step counts (development smoke runs so far: 5-26 steps used).
  If a future full run hits a real context-limit failure at high step
  counts, that is a genuine finding to report, not a hidden risk to
  silently guard against pre-emptively without evidence it occurs.

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

- This document was written after: the decision-schema parsing fix, the
  budget-protocol fix (removing the per-question metadata override), the
  tool-parity fix (`callers` now tries `indexed_callers` first), the
  repository-scope fix (both baselines on `RepoEnvironment.iter_files()`),
  and the per-tool result-limit normalization (`ANT_PARITY_TOOL_LIMITS`
  as the default). Any further change to `matched_react.py`'s protocol
  behavior should update this file in the same commit.
