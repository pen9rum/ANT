# Long-context / multi-document QA dataset audit (frozen, written before implementation)

Status: pre-implementation audit. Records exactly which dataset variant is
used for each of the three benchmarks in this track (MuSiQue, HotpotQA,
2WikiMultihopQA), and why LongBench's versions were NOT used, despite being
the initially preferred option. All findings below were obtained by loading
real rows from each candidate source live (2026-09-12), not from memory.

## Decision: original per-benchmark datasets, NOT LongBench

LongBench (`THUDM/LongBench`, also mirrored at `zai-org/LongBench`) was
audited first, as instructed, since LongAgent itself was evaluated in a
LongBench-style long-context setting. It fails on three independent grounds:

1. **Not straightforward to load.** Its Hugging Face repo ships only a
   `LongBench.py` *loading script* plus a `data.zip` -- the modern
   `datasets` library (5.0.1, installed here) refuses to execute dataset
   loading scripts at all ("Dataset scripts are no longer supported").
   There is no Hub-side auto-converted Parquet branch either (`refs/convert/
   parquet` returns 404 for both `THUDM/LongBench` and `zai-org/LongBench`).
   The only way to load it is to download `data.zip` (114MB) directly and
   parse the JSONL files inside it by hand, bypassing `load_dataset`
   entirely.
2. **`context` is a single flat pre-concatenated string, not structured
   documents.** Inspected real rows from `data/hotpotqa.jsonl`,
   `data/2wikimqa.jsonl`, and `data/musique.jsonl` directly: every row's
   `context` field is one string of the form `"Passage 1:\n<title>\n<body>\n\nPassage 2:\n..."`,
   with no separate title/id/boundary metadata. This does not satisfy
   Section 2's requirement to preserve original document boundaries,
   order, and titles/IDs as first-class structured data (it CAN be
   regex-split back apart on `"Passage \d+:\n"`, but that is a
   LongBench-specific parsing hack, not a faithful reconstruction of the
   original benchmark's own document structure).
3. **No supporting-fact annotations at all.** None of the three LongBench
   configs carry anything equivalent to `supporting_facts`/`is_supporting`.
   This makes LongBench's version unusable for the Lost-in-the-Middle
   positional-perturbation utility (Section 15), which requires knowing
   which passage(s) are relevant in order to move them early/middle/late.

None of these are LongAgent-specific concerns -- they are properties of how
LongBench itself repackages these three benchmarks for single-context-window
stress testing, a different purpose (raw length scaling) than this track's
own goal (faithful multi-document structure + controllable supporting-fact
position). Per the explicit "audit first, do not silently mix variants"
instruction, the conclusion is to use each benchmark's own original,
structured release instead.

## Exact sources used

All three were loaded live and verified to have the fields claimed below.

### MuSiQue

- **Source**: `dgslibisey/MuSiQue` on Hugging Face (parquet-backed, loads
  directly with `datasets>=5.0`, no script).
- **Config**: default. **Split used: `validation`** (no `test` split exists
  in this mirror -- consistent with MuSiQue's own official practice of
  keeping the true test set held out behind a leaderboard submission).
- **Row schema** (verified from a real row): `id` (e.g.
  `"2hop__460946_294723"`, hop-count-prefixed per MuSiQue's own ID
  convention), `question`, `answer`, `answer_aliases` (list, may be empty),
  `answerable` (bool), `question_decomposition` (MuSiQue's signature
  single-hop decomposition, not used by this track), `paragraphs`: a list
  of `{idx: int, title: str, paragraph_text: str, is_supporting: bool}` --
  this is the "answerable" MuSiQue variant's native schema, confirmed
  field-for-field against the official `musique_ans` release format.
- **Document count per question**: variable (real row observed: 20
  paragraphs), matching MuSiQue's own distractor-scaling-by-hop-count
  design.

### HotpotQA

- **Source**: `hotpotqa/hotpot_qa` on Hugging Face (the dataset's own
  canonical org/name, parquet-backed).
- **Config**: **`distractor`** (not `fullwiki`). **Split used:
  `validation`.**
  - Audited both configs' split lists live: `distractor` only has
    `train`/`validation`; `fullwiki` additionally lists a `test` split, but
    a real row from it has `answer=None` and empty `supporting_facts` --
    it is the hidden/blank leaderboard test set, not usable for scoring.
    `distractor`'s own test set is likewise not distributed with answers.
    `validation` is therefore the only split with complete, real gold
    answers and supporting-fact annotations.
- **Row schema** (verified from a real row): `id`, `question`, `answer`
  (single string), `type` (`"bridge"` or `"comparison"`), `level`
  (`"easy"`/`"medium"`/`"hard"`), `supporting_facts`: `{title: list[str],
  sent_id: list[int]}`, `context`: `{title: list[str], sentences:
  list[list[str]]}` (one sentence-list per document, parallel to `title`).
- **Document count per question**: exactly 10 (2 gold + 8 distractors, by
  the distractor setting's own design).

### 2WikiMultihopQA

- **Source**: `framolfese/2WikiMultihopQA` on Hugging Face. (The more
  "obvious" mirror, `xanhho/2WikiMultihopQA`, has the same
  script-based-loading problem as LongBench -- `"Dataset scripts are no
  longer supported, but found 2WikiMultihopQA.py"` -- and was rejected on
  that basis.)
- **Config**: default. **Split used: `validation`.**
  - Audited live: `test` split rows have `answer=""` (empty string) --
    again the hidden-leaderboard pattern. `validation` has real, complete
    answers.
- **Row schema** (verified from a real row): `id`, `question`, `answer`,
  `type` (e.g. `"compositional"`), `evidences`, `supporting_facts`:
  `{title: list[str], sent_id: list[int]}`, `context`: `{title: list[str],
  sentences: list[list[str]]}` -- **structurally identical to HotpotQA's
  own schema** (2WikiMultihopQA's own paper explicitly adopted HotpotQA's
  format for cross-benchmark tooling compatibility; confirmed directly by
  comparing the two real rows field-for-field, not assumed).
- **Document count per question**: 10 in the row inspected (consistent
  with 2WikiMultihopQA's own distractor-style design, mirroring HotpotQA).

## Practical consequence for the loader implementation

Because HotpotQA and 2WikiMultihopQA share an identical `context`/
`supporting_facts` schema, one shared parsing path handles both; MuSiQue's
distinct `paragraphs`-list schema gets its own path. Both paths converge on
the SAME generic internal representation (`DocumentTaskExample`: ordered
list of `{doc_id, title, text}` plus a question and gold answer(s)) that
`EvalDocumentEnvironment` and every method consume identically -- see
`src/ant/benchmarks/musique.py`, `hotpotqa.py`, `twowikimultihopqa.py`.

**Supporting-fact / `is_supporting` fields are read ONLY by the benchmark
adapter's own scoring/diagnostic path and by the (not-yet-run)
Lost-in-the-Middle perturbation-construction utility -- never placed on the
`TaskExample` object any inference method receives.** See
`docs/long_context_no_leakage_policy.md`-equivalent note inline in
`ant/evaluation_suite/document_scope.py`'s own module docstring for the
mechanical guarantee.

## Official scoring metric (verified, not assumed)

All three benchmarks' own official evaluation scripts were fetched directly
and compared:

- HotpotQA: `hotpotqa/hotpot/hotpot_evaluate_v1.py`
- MuSiQue: `StonyBrookNLP/musique/metrics/answer.py` (imported by
  `evaluate_v1.0.py`)
- 2WikiMultihopQA: `Alab-NII/2wikimultihop/2wikimultihop_evaluate_v1.1.py`

All three use a **byte-identical** `normalize_answer` (lowercase -> remove
punctuation -> remove articles `a`/`an`/`the` -> collapse whitespace) before
computing exact-match and token-overlap F1. MuSiQue's own metric additionally
takes the max EM/F1 over `[answer] + answer_aliases` (confirmed in
`metric_max_over_ground_truths`); HotpotQA/2WikiMultihopQA have no aliases
field, so their ground-truth set is just `[answer]`. This track implements
one shared, byte-faithful port of this exact normalization + max-over-
ground-truths logic, used identically for all three benchmarks -- see
`ant/evaluation_suite/qa_metrics.py`. No GPT-5 judge is used for these tasks.

## Explicitly not added this pass

NarrativeQA, Qasper, InfiniteBench, NoLiMa, WebWalkerQA, BrowseComp-Plus,
and RULER are out of scope for this pass (RULER gets a design note only,
Section 16 of the parent task).
