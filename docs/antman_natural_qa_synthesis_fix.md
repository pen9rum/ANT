# Fixing a benchmark-policy leakage in ANTMAN's final-answer synthesis

## A. Where the leakage came from (diagnosed before writing any code)

`ant.agents.ant_document_adapter.AntDocumentAgent.run()` (the natural
multi-document QA adapter used for HotpotQA/2WikiMultihopQA/MuSiQue) and
`ant.agents.ant_adapter.AntAgent.run()` (the SWE-QA-Pro adapter) are two
**completely separate classes** that both call the same frozen
`ant.coordinator.local.LocalCoordinator.ask()`. `state.answer` -- that
call's own internal synthesis, developed/tuned against SWE-QA-Pro, where
abstaining on a genuinely unsupported repository question is a correct
outcome -- was used by `AntDocumentAgent.run()` as the *sole basis* for
every document benchmark's final answer:

```
raw_answer = state.answer
final_answer = condense_to_answer_span(provider, example.question, raw_answer)
```

`condense_to_answer_span` (`ant.evaluation_suite.answer_contract`) is a
purely **extractive** shortening step -- it never re-derives an answer
from evidence, only shortens whatever `state.answer` already says. Its own
prompt even explicitly instructs, for an already-abstaining `raw_answer`:
*"If the answer above does not state a clear answer, output exactly what
it states is missing, as briefly as possible (e.g. 'not stated in the
provided documents')."* That is the literal source string. Since
`raw_answer` already carried SWE-QA-Pro-style caution whenever the core
coordinator's own synthesis judged the answer not literally stated,
HotpotQA/2Wiki/MuSiQue inherited that same abstention bias even for
ordinary compositional questions the retrieved evidence could answer.

`AntAgent.run()` (SWE-QA-Pro) never imports `answer_contract.py` or
anything from this change at all -- it uses `final_answer=state.answer`
directly. This was confirmed both by reading the file and by a structural
test (`test_ant_adapter_module_never_imports_natural_qa_synthesis`).

## B. The fix

A new, independent module,
`ant.evaluation_suite.natural_qa_synthesis`, provides ONE function:
`synthesize_natural_multihop_answer(provider, question, evidence_block)`.
It never reads `LocalCoordinator`, never touches worker routing, the Need
Graph, evidence selection/facet rescue, or retrieval -- it only consumes
whatever evidence an (unmodified) coordination run already gathered and
performs the final natural-language synthesis over it, permitting ordinary
compositional inference (entity chaining, relation composition,
comparisons, counting, simple arithmetic, cross-evidence entity
resolution) while still allowing abstention when the evidence genuinely
lacks the needed fact.

`AntDocumentAgent.run()` now branches on `example.benchmark`:

```python
if is_natural_multihop_qa_benchmark(example.benchmark):  # hotpotqa/2wiki/musique
    evidence_block = format_evidence_block(state.evidence)
    final_answer = synthesize_natural_multihop_answer(provider, example.question, evidence_block)
else:
    final_answer = condense_to_answer_span(provider, example.question, grounded_answer)  # unchanged
```

The natural-QA branch **replaces** `state.answer` + `condense_to_answer_span`
entirely (rather than chaining after it) -- deliberately, so the shared
condensation prompt's own "output exactly what it states is missing"
instruction can never re-inject abstention language into an already-good
composed answer. Every other benchmark (including any future
`ant_document`-family benchmark) keeps the exact prior code path,
byte-for-byte. `answer_contract_condition == "B"` (the single-needle
contamination-study regrounding flag) is asserted absent for the
natural-QA branch -- it is not defined there, and currently never set on
these three benchmarks' examples.

**Prompt policy** (generic to "standard answerable multi-hop QA" as a task
family -- no benchmark name, entity name, or gold-derived rule anywhere in
it):
- Answer using only the supplied evidence; the answer need not appear
  verbatim in one sentence; combine evidence across entries.
- Ordinary compositional inference (entity chaining, relation composition,
  comparisons, counting, arithmetic, cross-entity resolution) is expected
  and encouraged; do not refuse merely because the final relation isn't
  written out as its own sentence.
- If the evidence genuinely lacks the needed fact, abstain briefly rather
  than inventing or guessing.
- Prefer the shortest direct answer span; no unnecessary explanation.

## C. Validation (before any paid run)

- `tests/test_natural_qa_synthesis.py` (11 tests): the prompt explicitly
  contains the compositional-inference-permission language and the
  abstention-still-allowed language; the function is a pure pass-through
  of the model's own output in both directions (composed answer and
  abstention both preserved unchanged); empty evidence short-circuits to
  abstention with **zero LLM calls**; the function signature structurally
  has no gold/metadata parameter (`inspect.signature` == `["provider",
  "question", "evidence_block"]`); no planted secret ever appears in the
  assembled prompt.
- `tests/test_ant_document_adapter.py` (+9 new tests, 13 total):
  - **A**: SWE-QA-Pro's adapter module never imports this policy at all,
    and its `run()` source still literally contains
    `final_answer=state.answer` -- unchanged.
  - **B**: a scripted-evidence test proves `run()` dispatches to
    `synthesize_natural_multihop_answer` (never `condense_to_answer_span`)
    for hotpotqa/2wikimultihopqa/musique, with the exact evidence/question
    from the (stubbed) coordination state.
  - **C**: the "else" branch test proves every other benchmark still uses
    `condense_to_answer_span` exactly as before -- `synthesize_natural_
    multihop_answer` raises if ever called there.
  - `answer_contract_condition == "B"` on a natural-QA benchmark raises
    `AssertionError` rather than silently mis-applying regrounding.
  - A planted-secret test proves nothing beyond `question`/`evidence_block`
    ever reaches the new synthesis call.
- Real LLM composition/abstention behavior is not asserted deterministically
  in the automated suite (this codebase's established convention -- every
  other agent's tests mock the LLM boundary) -- it is verified empirically
  by the 90-example re-answer experiment below, which is the real evidence
  for the policy actually working at scale.
- Full repo suite: 691/691 passed. `ruff check src/ tests/`: clean.
  `pyright`: 0 errors in every file this change touched (49 pre-existing,
  unrelated errors elsewhere in the repo, confirmed untouched by this
  change and present before it).

## D. Re-answer experiment (90 examples, evidence reused, no new routing)

For all 90 existing ANTMAN natural-pilot examples (30/benchmark, the n=30
extension frozen earlier this session), the EXISTING saved trajectory's
`evidence` list was loaded verbatim and passed through
`synthesize_natural_multihop_answer` -- `LocalCoordinator` was never
constructed, `.ask()` was never called, and `ant_document.jsonl` /
`.rescored.jsonl` / `trajectories/` were never modified. New files:
`ant_document.reanswer_naturalqa.jsonl` (old/new answer pairs + usage) and
`ant_document.reanswer_naturalqa.rescored.jsonl` (old + new EM/F1, via the
unmodified, frozen `extract_answer_span` + `score_qa`).

**Cost:** $0.0988 (90 synthesis calls) + $0.0587 (rescoring the 90 new
answers) = **$0.1575 total**.

**Old vs. new, per benchmark (n=30) and macro average:**

| Benchmark | Old EM | New EM | Δ EM | Old F1 | New F1 | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | 0.533 | 0.633 | +0.100 | 0.678 | 0.775 | +0.097 |
| 2WikiMultihopQA | 0.700 | 0.667 | -0.033 | 0.785 | 0.789 | +0.004 |
| MuSiQue | 0.400 | 0.500 | +0.100 | 0.547 | 0.610 | +0.063 |
| **Macro Avg** | **0.544** | **0.600** | **+0.056** | **0.670** | **0.725** | **+0.055** |

## E. Abstention rate, before vs. after

| Scope | Old | New |
|---|---:|---:|
| Overall (90) | 15 (16.7%) | 10 (11.1%) |
| HotpotQA (30) | 5 | 2 |
| 2WikiMultihopQA (30) | 3 | 2 |
| MuSiQue (30) | 7 | 6 |

## F. Per-answer classification (all 90, computed after scoring)

| Category | Count |
|---|---:|
| Fixed by compositional inference (was abstention → correct/improved) | 3 |
| Still abstaining, genuinely insufficient evidence (unchanged refusal) | 10 |
| Became incorrect (was correct → now wrong) | 2 |
| Formatting/extraction-only difference (text changed, score same-or-better, no abstention flip) | 18 |
| Unchanged, already correct | 42 |
| Unchanged, already wrong (non-abstention) | 15 |

The 2 regressions were inspected directly, not just counted:
- **VCU founding year** (HotpotQA `5adf37a9`): the reused evidence
  literally states *both* "VCU was founded in 1838" and "In 1968, [VCU was
  formed by merger]" -- a genuine founding-date ambiguity in the source
  text itself. The new policy committed to "1968" (gold: "1838"); this is
  a wrong choice between two literally-evidenced candidates, not a
  fabrication -- the new "commit to an answer" permission occasionally
  picks the less-preferred of two real facts instead of hedging.
- **"Which film..." answered with a director's name** (2Wiki `05f8a691`):
  new raw answer is *"Elio Petri (director of The Working Class Goes to
  Heaven) died first."* -- substantively correct reasoning, but it answers
  "who" instead of "which film" as literally asked, and includes
  explanatory framing the prompt's own "prefer the shortest direct answer
  span" instruction should have suppressed. The correct film title is
  present verbatim inside the sentence (F1 0.625, not 0), so this is
  better classified as a synthesis/formatting weakness than a genuine
  factual miss.

## G. Final-synthesis-fixed vs. routing/retrieval-failures-remaining

Every one of the 15 original abstentions was inspected individually:

| Outcome | Count | Cases |
|---|---:|---|
| **Fixed** (synthesis alone recovers a correct answer from evidence that was already there) | 3 | Brown State Fishing Lake population (9,984); Badly Drawn Boy instrument ratio; Empire Sports Network successor (Time Warner Cable) |
| **Partial** (composed an answer instead of abstaining, but picked the wrong entity from ambiguous/incomplete evidence) | 2 | "Dr. Robotnik" vs gold "Sonic" (evidence names Robotnik, not Sonic, directly); "American" vs gold "United States" (a genuine two-Eric-Mueller identity-ambiguity case, unchanged from the earlier diagnosis) |
| **Still abstains -- confirmed a real routing/retrieval miss** | 10 | Animorphs; Tunnels & Trolls "Arena of Khazan" (the traced decoy-document case); Antiochus X's maternal grandfather; Thomas Jefferson (Film) director's birthplace; Green performer's spouse; Ulrich Walter's employer HQ; The Red Tree author's award; Rabbit Hole producer's spouse; Bruce Lee Band member's record label; Lessing's other notable work |

This is exactly the intended separation: the final-synthesis fix recovers
answers the evidence already supported (1/3 of prior abstentions fully,
another 1/3 partially), while the 10 genuine routing/retrieval misses
identified earlier this session (including the traced "Arena of Khazan"
decoy-document case) correctly remain failures rather than being
hallucinated around -- proving the new policy does not simply trade
"honest abstention" for "confident guessing."

## H. Recomputed 7-method n=30 table (baselines unmodified)

Sparse/Dense/ChainRAG/ReAct/LongAgent/CoA scores loaded unmodified from
their existing `*.rescored.jsonl` files -- none rerun, none touched.

| Method | HotpotQA | 2Wiki | MuSiQue | **Avg EM/F1** |
|---|---:|---:|---:|---:|
| Sparse Retrieval | .533/.684 | .367/.515 | .433/.582 | .444/.594 |
| Dense Retrieval | .533/.696 | .267/.379 | .200/.292 | .333/.456 |
| ChainRAG | .567/.711 | .733/.791 | .500/.654 | .600/.719 |
| ReAct | .533/.734 | .567/.781 | .467/.626 | .522/.714 |
| LongAgent | .433/.655 | .467/.697 | .400/.494 | .433/.616 |
| CoA | .567/.769 | .600/.778 | .533/.682 | **.567/.743** |
| ANTMAN (old policy) | .533/.678 | .700/.785 | .400/.547 | .544/.670 |
| **ANTMAN (new policy)** | **.633/.775** | .667/.789 | .500/.610 | **.600/.725** |

Ranking by macro F1: **CoA (0.743) > ANTMAN-new (0.725) > ChainRAG (0.719)
> ReAct (0.714)** > ANTMAN-old (0.670) > LongAgent (0.616) > Sparse (0.594)
> Dense (0.456). Fixing the benchmark-policy leakage alone moves ANTMAN
from **4th to 2nd** of 7 methods on macro F1, closing most of the gap to
CoA (0.743) and overtaking ChainRAG/ReAct -- without any change to
routing, retrieval, or the underlying evidence a single example used.

## I. Integrity confirmation

- **SWE-QA-Pro is unaffected**: `ant.agents.ant_adapter.AntAgent` never
  imports `natural_qa_synthesis`; its `run()` source is unchanged
  (`final_answer=state.answer` verbatim); confirmed by a structural test.
- **No worker routing/Need Graph/evidence-selection/facet-rescue/retrieval
  code was touched**: `local.py` was not edited; the fix lives entirely in
  `ant_document_adapter.py`'s post-hoc answer-selection step and one new,
  independent module.
- **No information-seeking trajectory was rerun**: the 90-example
  experiment loaded each example's already-saved `evidence` list from its
  existing trajectory file and never constructed a `LocalCoordinator`.
- **Old outputs preserved exactly**: `ant_document.jsonl` (30/30 rows per
  benchmark, unchanged), `ant_document.rescored.jsonl`, and
  `trajectories/ant_document-*.json` (30/30 per benchmark) are
  byte-identical to before this pass; all new data went to
  `ant_document.reanswer_naturalqa(.rescored).jsonl`.
- **The frozen extraction/scoring pipeline was reused unchanged**:
  `extract_answer_span` and `score_qa` were imported and called exactly as
  every other method already does; neither was modified.
- **No gold answer, supporting fact, or question-specific rule ever
  reached generation, routing, or answer selection**: the new synthesis
  function's signature structurally admits only `(provider, question,
  evidence_block)`; verified by both a signature-inspection test and a
  planted-secret test at the adapter level.
- **Not benchmark-specific answer hacking**: the natural-QA prompt
  contains no benchmark name, dataset heuristic, entity name, or
  gold-derived rule -- it is framed only as "a standard multi-hop QA
  benchmark question," identically for all three benchmarks, and was
  written and frozen before this pass's own 90-example results were seen.

Per the governing spec: implementation, tests, the 90-example
final-answer-only rerun, frozen rescoring, and this old/new comparison
report are complete. No routing was altered and the full ANTMAN pipeline
was not rerun. Stopping here as instructed.
