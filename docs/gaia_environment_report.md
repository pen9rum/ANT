# GAIA environment report

**Scope of this pass**: engineering and environment construction only.
**Zero paid inference was run.** No LLM API call of any kind was made
against any GAIA question, real or synthetic. No benchmark evaluation was
executed. This report freezes an environment and a proposed design; it
contains no scores.

**Which GAIA**: the original ICLR 2024 benchmark
(`gaia-benchmark/GAIA`, arXiv:2311.12983), 2023 **validation** split.
Not GAIA2, not any community re-release.

---

## A. Dataset status — BLOCKED on gated access

### What was attempted, and exactly what failed

| Step | Result |
|---|---|
| `HfApi().dataset_info("gaia-benchmark/GAIA")` | **OK** — public metadata. Reports `gated="auto"`, `private=False` |
| `HfApi().list_repo_files(..., repo_type="dataset")` | **OK** — 119 file paths listed (public metadata) |
| `hf_hub_download(..., "2023/validation/metadata.jsonl")` | **FAILED — `GatedRepoError`, HTTP 401** |
| `datasets-server.huggingface.co/{size,splits,info}` | **FAILED** — "does not exist, or is not accessible without authentication (private or gated)" |

Verbatim failure:

```
GatedRepoError: 401 Client Error.
Cannot access gated repo for url
https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation/metadata.jsonl
Access to dataset gaia-benchmark/GAIA is restricted.
You must have access to it and be authenticated to access it. Please log in.
```

Environment state: `HF_TOKEN` **unset**, `HUGGING_FACE_HUB_TOKEN`
**unset**, no cached token in `~/.cache/huggingface`.

No workaround was attempted. No mirror, scraped copy, or unofficial
derivative was searched for or used.

### What the user must do to unblock

1. Sign in to a Hugging Face account.
2. Visit the `gaia-benchmark/GAIA` dataset page on huggingface.co and
   accept the dataset's terms. The gate is configured **`gated: auto`**,
   read live from the dataset card — meaning **access is granted
   automatically on accepting the terms; there is no manual review
   queue and no waiting on a maintainer.** The gate's own text is:
   > "To avoid contamination and data leakage, you agree to not reshare
   > this dataset outside of a gated or private repository on the HF hub."
3. Expose that account's token to the process, either by running
   `huggingface-cli login` or by setting `HF_TOKEN`.
4. Install the optional dependency: `pip install 'ant-codebase[bench]'`
   (provides `datasets`).

Then `GaiaAdapter(source="live").load_examples()` works unchanged.
Everything else in this report is already built and tested.

### Redistribution discipline

Per those terms, **no GAIA question, answer, or attachment byte is
committed anywhere in this repository.** Raw downloads land in
`.gaia-data/`, newly added to `.gitignore`. The only GAIA-origin file in
version control is the official **scorer** (code, not data — see §D).

---

## B. Dataset audit — what could be established WITHOUT the gate

Part 2's audit is mostly blocked. However, the HF **file listing** is
public metadata even for a gated repo, and it yields a genuine,
zero-inference, zero-gold attachment audit. Reported separately from the
numbers that remain unverified.

### B.1 Verified live from public metadata

The 2023 validation split ships **42 files**, of which **4 are metadata
parquet** (`metadata.parquet`, `metadata.level{1,2,3}.parquet`, named in
the dataset card's `configs`) and **38 are task attachments**:

| Extension | Count | Modality |
|---|---|---|
| `.xlsx` | 13 | tabular |
| `.png` | 8 | image |
| `.mp3` | 3 | audio |
| `.pdf` | 3 | document |
| `.jpg` | 2 | image |
| `.zip` | 2 | archive |
| `.csv` | 1 | tabular |
| `.docx` | 1 | office doc |
| `.jsonld` | 1 | structured text |
| `.pdb` | 1 | text (Protein Data Bank) |
| `.pptx` | 1 | office doc |
| `.py` | 1 | text/code |
| `.txt` | 1 | text |
| **Total** | **38** | |

Headline: **spreadsheets are the single largest attachment category
(14 of 38 counting `.csv`), and images are second (10 of 38).** That
finding directly drove two engineering decisions in §C — building a
dependency-free `.xlsx` reader, and declaring image/audio explicitly
unsupported rather than quietly skipping them.

For context, the test split's 75 attachments skew similarly
(`.xlsx` 16, `.pdf` 12, `.txt` 12, `.png` 10, `.csv` 5, `.jpg` 5,
`.mp3` 4, plus `.parquet` 4 metadata and a long tail).

### B.2 NOT verified — requires the gate

These remain open and **must be recomputed live from the real data**
before any subset is frozen (per this project's standing "recompute
fresh, never assume" rule):

- **Total N.** The paper states "a developer set of 166 annotated
  questions"; the widely-reported figure for the HF validation split is
  **165**. This one-row discrepancy is unresolved and I will not pick a
  side from secondary sources.
- **Per-level distribution for validation.** The paper's Table 4 gives
  whole-dataset level counts (Level 1: 146, Level 2: 245, Level 3: 75
  = 466 total), **not** the validation-only split.
- **Attachment-to-task mapping.** 38 attachment *files* implies **at
  most** 38 tasks with attachments (GAIA's `file_name` is a single
  scalar field per row), so roughly 23% of ~165 — but the exact count
  needs the metadata table.
- **Capability-category breakdown from question text.** Requires reading
  the questions. The adapter already computes the task-observable half
  of this (attachment modality → territories) with no gold involved, so
  this becomes a one-command audit the moment access lands.

---

## C. Environment — what was built

All new, all additive. **No existing file's behaviour was modified**
(the only edits to pre-existing files are a `.gitignore` entry and a
`pyproject.toml` ruff exclusion; see §F).

| Module | Role |
|---|---|
| `src/ant/benchmarks/gaia.py` | `GaiaAdapter` — `load_examples` / `prepare_environment` / `score`, per `benchmarks/base.BenchmarkAdapter` |
| `src/ant/evaluation_suite/gaia_scope.py` | Substrate: attachment resolution, modality policy, parsers, territories |
| `src/ant/evaluation_suite/gaia_scorer.py` | Loader + `FINAL ANSWER:` extraction over the vendored official scorer |
| `src/ant/evaluation_suite/gaia_fixtures.py` | Deterministic builder for synthetic fixture attachments |
| `src/ant/agents/gaia_tools.py` | The **shared** tool registry + coordinator seam adapter |
| `third_party/manifests/gaia/scorer.py` | Official GAIA scorer, vendored byte-for-byte |
| `third_party/manifests/gaia/PROVENANCE.md` | Pin, license, upstream quirks |
| `third_party/manifests/gaia/synthetic_fixtures.json` | Hand-authored fixtures (zero GAIA content) |

### C.1 Implemented tools (the shared surface)

Five primitives on `GaiaToolRegistry`, the **only** object in the
substrate able to reach a backend or an attachment:

| Tool | Status |
|---|---|
| `search(query, limit)` | Implemented; backend injected. **Raises if no backend is wired** — never returns `[]` for a misconfiguration |
| `open_url(url)` | Implemented; backend injected; same fail-loud rule |
| `inspect_file(offset, limit)` | Implemented — text/archive/PDF(conditional) |
| `inspect_table(max_rows)` | Implemented — `.csv`, `.tsv`, `.xlsx` |
| `compute(expression)` | Implemented as a **restricted AST arithmetic evaluator** |

**`compute()` is deliberately not a Python sandbox.** GAIA does include
tasks that want arbitrary code execution, but wiring `exec` to
model-chosen strings is an arbitrary-code-execution primitive, and
building a genuinely safe sandbox is separate reviewed work. The safe
subset ships; the gap is declared rather than papered over.

### C.2 Modality support — and what fails explicitly

| Modality | Validation files | Status | Behaviour |
|---|---|---|---|
| Tabular (`.xlsx`/`.csv`) | 14 | **Supported** | Stdlib-only OOXML + `csv` reader |
| Text (`.txt`/`.py`/`.jsonld`/`.pdb`) | 4 | **Supported** | Direct read |
| Archive (`.zip`) | 2 | **Supported** | Member listing |
| PDF (`.pdf`) | 3 | **Conditional** | Supported iff PyMuPDF importable; else explicit `UnsupportedModalityError` |
| Office doc (`.docx`/`.pptx`) | 2 | **Not implemented** | Explicit error |
| Image (`.png`/`.jpg`) | 10 | **UNSUPPORTED** | Explicit error |
| Audio (`.mp3`) | 3 | **UNSUPPORTED** | Explicit error |
| Video (`.mov`/`.mp4`) | 0 (val) | **UNSUPPORTED** | Explicit error |

**Image and audio are the fairness line, not an oversight.** Adding OCR
or ASR for ANTMAN alone would be exactly the privileged-capability
failure the evaluation design forbids. They are declared `UNSUPPORTED`
at the **substrate** level, so the refusal binds every method
identically. Three tests assert these raise rather than returning empty
or partial content.

`NOT_IMPLEMENTED` vs `UNSUPPORTED` is a deliberate distinction:
the former is a cheap scope decision, the latter is a fairness
judgement. Both fail loudly; only the remediation differs.

**Honest consequence**: ~13 of 38 attachment-bearing validation tasks
(image + audio) cannot be fairly solved from their attachment in this
environment. That is a known, declared coverage ceiling — it must be
reported alongside any future accuracy number, never silently absorbed.

### C.3 Leakage prevention

Following the established `_audit_only` pattern
(`benchmarks/webwalkerqa.py`):

- **Agent-visible**: `question`, `task_id`, and metadata keys
  `level`, `file_name`, `has_attachment`, `attachment_modality`,
  `attachment_supported`, `territories`, `source`.
- **`reference`**: the gold `Final answer`. Read only by `score()`.
- **`metadata["_audit_only"]`**: GAIA's `Annotator Metadata`
  (`Steps` = a literal worked solution, `Tools` = the exact capability
  list) plus a copy of the final answer. **No generation path reads
  this key.**

Structural, not just conventional: `derive_territories()` and
`GaiaEnvironment.__init__` have **no parameter** through which an answer
or annotator metadata could be threaded — asserted by signature tests,
mirroring `web_scope.build_territory_from_discovered`'s own precedent.

`level` is intentionally agent-visible: it is a difficulty label, not
solution content, it is needed to stratify results, and
`benchmarks/repoprobe.py` already keeps its `difficulty` visible.

---

## D. Scorer — official, vendored verbatim, not reimplemented

**The real official scorer was obtained and is used directly.** This was
possible because the GAIA *leaderboard Space* is public even though the
*dataset* is gated:

| Field | Value |
|---|---|
| Source | HF Space `gaia-benchmark/leaderboard`, `scorer.py` |
| Revision | `9f133d71362e77b3539f1514f31b9c101a545fec` |
| SHA-256 | `0d44c07f3046eec521697c22e3eaca8719cc81e422a8eaf32695c5f22bdac6e2` |
| License | Apache-2.0 |

`gaia_scorer.py` loads that file via `importlib` and **verifies its
SHA-256 at import time** — a stray reformat fails loudly instead of
silently shifting every score. Nothing is reimplemented from memory, so
the "diff against the true official implementation" caveat the task
anticipated **does not apply**: this *is* the official implementation.

The official **system prompt** was likewise vendored verbatim from the
same Space's `content.py`, including the `FINAL ANSWER: [YOUR FINAL
ANSWER]` template.

**What is ours, and disclosed as ours**: `extract_final_answer()`. The
official scorer takes an *already-extracted* `model_answer` (submissions
are JSONL of `{task_id, model_answer}`), so template-stripping happens
on the submitter's side and has no official implementation to vendor.
Ours is pure and deterministic — regex and string ops, zero LLM calls
— and deliberately conservative: it strips template scaffolding only and
never touches casing, internal punctuation, or units, because the
official scorer owns all of that and double-normalizing could change
verdicts.

### Upstream quirks preserved (not "fixed")

Documented in `PROVENANCE.md` and pinned by tests: list comparison is
**order-sensitive**; a length mismatch is an immediate `False`; only
`$`, `%`, `,` are stripped as units; `normalize_str` removes *all*
whitespace (`sea gull` == `seagull`); list elements compare with
punctuation *kept* while whole strings compare with it *removed*.

One consequence is worth flagging to whoever reads a future results
table: `"1,000, apples"` scores **False** against `"1000, apples"`,
because the comma-grouped number splits into two list elements. **Format
compliance is load-bearing, not cosmetic** — which is exactly why the
official system prompt says "don't use comma to write your number". A
method that ignores the format instruction loses points on formatting
alone. `final_answer_template_found` is recorded in `MetricResult.metadata`
so format failures stay distinguishable from content failures.

Scoring is `JudgeType.DETERMINISTIC`: one call, **no judge, no LLM, zero
cost**. GAIA is the only substrate in this suite whose scoring is free.

---

## E. Design — GAIA as a heterogeneous substrate, not a WebWalker clone

`web_scope.py` models a territory as *a region of one website*, because
WebWalkerQA is single-root-site traversal: every territory is the same
kind of thing, reached the same way, and the only question is which
subtree.

**GAIA is the opposite shape.** A GAIA task is not "navigate one site";
it is "this needs a spreadsheet read AND a web lookup AND an arithmetic
step". The heterogeneity is across **information kinds**, not regions of
one homogeneous space. So a `GaiaTerritory` is *a kind of information
source with its own access primitive*:

| Territory | When present | Access primitive |
|---|---|---|
| `WEB` | always | `search` / `open_url` |
| `COMPUTATION` | always | `compute` |
| `ATTACHMENT` | iff `file_name` | `inspect_file` |
| `TABULAR` | iff attachment is tabular | `inspect_table` |
| `DOCUMENT` | iff attachment is text/PDF/office | `inspect_file` |

Two properties are enforced structurally:

1. Territories derive from **task-observable signals only** — attachment
   presence and file extension. Never gold, never annotator metadata.
2. Territory derivation is **not an answer heuristic**. It says "a
   spreadsheet is present, so tabular reading is in scope", never "this
   question is type X, so do Y". `WEB` and `COMPUTATION` are granted
   **unconditionally** precisely so no question-text classifier creeps in.

An unsupported territory is marked `supported=False` on the territory
object itself, so a routing layer sees a capability gap as a first-class
fact rather than discovering it as an exception several rounds later.

**The coordinator is untouched.** `src/ant/coordinator/local.py` was not
modified. ANTMAN consumes GAIA through the existing
`search_tool_factory` seam via
`build_gaia_search_tool_factory(registry)`, which returns a duck-typed
`GaiaCoordinatorSearchTool`. The coordinator's role stays exactly the
generic unresolved-Need → territory → evidence → Need-revision loop. No
GAIA-specific answer heuristics, no benchmark answers anywhere.

Where GAIA has no analogue for a code-navigation primitive
(`callers`/`subclasses`/…), the seam returns empty **honestly** rather
than faking keyword matches that would enter the coordinator looking
like structural evidence.

### Shared tools — the fairness invariant

> ANTMAN must never hold a capability a Matched-ReAct-shaped baseline
> over the same substrate would not also hold.

Enforced by **identity, not review discipline**. `GaiaToolRegistry` is
the sole holder of every backend and the attachment. A ReAct-shaped
agent drives it by tool name through `invoke()`; ANTMAN drives *the same
registry instance* through the seam adapter, which owns no backend and
adds no method reaching around it. Tests assert `react.registry is
antman.registry`, that both report identical `available_tools()`, and
that an unsupported modality is unreadable through **both** paths.

A shared call log falls out for free: whichever shape makes a call, one
ledger records it.

---

## F. Test results

All tests are static/mocked. **Zero network, zero LLM calls.**

### New GAIA tests — 138, all passing

| File | Tests | Covers |
|---|---|---|
| `tests/test_gaia_scorer.py` | 40 | Official scorer normalization, SHA pin, `FINAL ANSWER:` extraction |
| `tests/test_gaia_scope.py` | 38 | Modality policy, explicit failures, attachment paths, parsers, territory signatures |
| `tests/test_gaia_tools.py` | 33 | Tool surface, deterministic logging, shared-registry fairness, coordinator seam |
| `tests/test_gaia_adapter.py` | 27 | Loading, **gold-leakage prevention**, environment prep, deterministic scoring |

Required coverage, mapped:

1. **Loader never exposes gold** — `test_annotator_metadata_never_appears_outside_audit_only` serializes every agent-visible surface and asserts planted sentinels are absent; plus an allowlist test pinning the exact visible key set. ✅
2. **Attachment paths resolve** — including a path-traversal refusal test. ✅
3. **Official scorer normalization** — numeric/units/commas, lists, case, whitespace, punctuation, **and explicit must-fail cases**. ✅
4. **Deterministic tool-call logging** — byte-identical logs across identical runs; asserted the log carries no timestamp/duration field. ✅
5. **Both agent shapes over one registry** — stub ReAct-shaped and ANTMAN-shaped agents, identity-asserted. ✅
6. **Unsupported modality fails explicitly** — image and audio raise through both access paths. ✅

### Existing suite — unchanged

`ruff check .` — **clean**. (This pass also fixed a *pre-existing*
repo-wide lint failure: 12 findings in the vendored RepoProbe prompt
template. `third_party` is now excluded from ruff, which is correct on
principle — vendored verbatim artifacts must never be reformatted, and
an actual `ruff --fix` over GAIA's scorer would break its SHA pin.)

Full-suite status is recorded in the commit message and the final
hand-off notes, including one **environment-provisioning** exclusion
(`tests/test_chainrag.py`) that is unrelated to this work — it fails at
import on missing optional dependencies (`networkx`, `spacy`) in the
isolated sandbox venv built for this pass, and did so *before* any GAIA
file existed. Nothing in this pass imports or touches ChainRAG.

---

## G. Proposed experimental design — NOT LAUNCHED

No inference was run. This is a recommendation for human review.

### Recommendation: **(B), a predeclared subset** — with a caveat

Recommend **B**, defined **only** by task-observable capability
requirements, never by gold answers and never by preliminary model
performance.

**Exact subset-selection rule** (deterministic, auditable, computable
without reading a single answer):

> Include a validation task **iff** every modality it requires is
> `SUPPORTED` in `gaia_scope.MODALITY_SUPPORT` at a pinned commit.
> Concretely: include the task **iff** it has no attachment, **or** its
> attachment's extension maps to `TEXT`, `TABULAR`, or `ARCHIVE`.
> Exclude `IMAGE`, `AUDIO`, `VIDEO`, `OFFICE_DOC`, `UNKNOWN`, and
> `PDF` unless PyMuPDF is pinned into the environment (in which case
> `PDF` moves to included, and that choice is frozen in the manifest).

Rationale: the excluded tasks are ones **no method in the roster can
fairly attempt**, because the substrate refuses the modality for
everyone equally. Including them would add a constant ~13-task floor of
guaranteed failures that compresses every method toward zero and
measures the substrate's OCR/ASR gap rather than the coordination
question under study.

**The caveat, stated plainly**: this is a *capability-coverage* subset,
not an *information-seeking* subset, and it is **not** comparable to
published full-validation GAIA numbers. Any result must be reported as
"GAIA-validation, supported-modality subset, N=<recomputed>", with the
excluded count and reasons printed alongside. If comparability to the
literature is the priority, run **(A)** the full split and report the
unsupported-modality failures explicitly as a capability ceiling — that
is defensible too, just answering a different question. I would run B as
the primary and keep A available as a disclosed secondary.

**Freeze order** (nothing here is done yet): get access → recompute N
and level distribution live → apply the rule above → write
`third_party/manifests/gaia/manifest.json` with the selected task IDs
and the exclusion ledger → **only then** consider inference.

### Proposed baseline roster — none launched

| Role | Method | Notes |
|---|---|---|
| Tool-limited reference | **Direct** | No tools; establishes the parametric-knowledge floor. Meaningful on GAIA, where the paper's own point is that tools are what matter |
| Matched baseline | **Matched ReAct (GAIA)** | Identical `GaiaToolRegistry`, identical budget. The load-bearing comparison |
| Retrieval reference | **Search-only** | Search + open_url, no attachment/compute — isolates web-seeking from tool orchestration |
| System under test | **ANTMAN (GAIA)** | Via `search_tool_factory`; coordinator unmodified |
| Published context | see below | Cited for orientation only, **never** as a matched comparison |

**Published reference numbers**, for context only — different models,
tooling, budgets, and in some cases a different split, so they bound
expectations rather than serve as controls:
- GAIA's own paper: **GPT-4 with plugins ≈ 15%**, **humans ≈ 92%**
  (arXiv:2311.12983) — the canonical framing number.
- Modern agent scaffolds report substantially higher validation-split
  figures (e.g. published claims in the ~35–65% range depending on model
  and tooling). These should be re-verified from primary sources before
  being printed in any results table, not taken from this sentence.

**Budget note**: GAIA scoring is deterministic and free, so a sweep's
entire cost is generation. That materially changes the cost model
relative to RepoProbe/SWE-QA-Pro, where 3 judge calls per task dominate.

---

## H. Open items for the human reviewer

1. **Grant HF access** (§A) — the only true blocker.
2. **Decide PDF**: add PyMuPDF as a pinned dependency (moves 3 tasks in)
   or leave conditional. Currently conditional and honest either way.
3. **Decide `.docx`/`.pptx`**: 2 tasks; stdlib-extractable if wanted.
4. **Decide A vs B** (§G) before anything is frozen.
5. **Review `compute()`'s scope**: arithmetic-only today. Broadening it
   to real code execution is a separate, security-reviewed decision.
6. **Confirm no inference** is launched until 1–5 are settled.
