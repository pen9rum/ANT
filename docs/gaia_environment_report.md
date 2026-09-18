# GAIA environment report

> **PASS 2 ADDENDUM (2026-09-17) — read §I–§L at the end of this file
> before acting on §A, §C.1, §C.2 or §G.** A second pass extended the
> substrate (PDF/DOCX/PPTX now genuinely supported; `compute()` replaced
> by a real sandboxed Python interpreter), implemented §G's subset rule,
> and produced a grounded cost estimate. **The dataset audit is still
> blocked**: the HF gate was reported as open but is not — see §I for
> the exact evidence. Sections A, C.1, C.2 and G below are the pass-1
> text, kept for the record and superseded where §I–§L say so.

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

---

# PASS 2 (2026-09-17)

**Still zero paid inference.** No LLM call of any kind was made against
any GAIA question, real or synthetic. Nothing in this pass ran a
benchmark. §L is an *estimate* built from previously-measured runs of
other benchmarks, not from a GAIA run.

---

## I. The HF gate is NOT open — audit still blocked

This pass was commissioned on the premise that access had been
unblocked and "independently verified working" via
`HfApi().dataset_info("gaia-benchmark/GAIA", token=...)` succeeding.
**That verification is not a test of access**, and access is in fact
still refused.

### What is true

| Check | Result |
|---|---|
| `HF_TOKEN` present in `.env` (both checkouts) | **Yes**, 37 chars |
| `whoami-v2` with that token | **OK** — user `Xaviuer`, token type `access_token` |
| Token scopes | fine-grained token named `Gaia`, **`canReadGatedRepos: true`** |
| `HfApi().dataset_info("gaia-benchmark/GAIA")` | **OK** — `gated="auto"`, `private=False` |

### What is also true, and is decisive

| Check | Result |
|---|---|
| `GET /api/datasets/gaia-benchmark/GAIA/auth-check` | **HTTP 403** |
| `hf_hub_download(".../2023/validation/metadata.jsonl")` | **HTTP 403 `GatedRepoError`** |
| `load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")` | **`DatasetNotFoundError`: gated** |

Verbatim:

```
403 Client Error. Cannot access gated repo for url
https://huggingface.co/datasets/gaia-benchmark/GAIA/resolve/main/2023/validation/metadata.jsonl.
Access to dataset gaia-benchmark/GAIA is restricted and
you are not in the authorized list.
```

### Why `dataset_info` succeeding misled the check

`gaia-benchmark/GAIA` is **`gated: auto`, `private: false`**. For such a
repo the *repo metadata endpoint is public* — `dataset_info` returns
name, tags, file list and gate configuration to anyone, authenticated or
not. It answers "does this repo exist and is it gated?", **not** "may I
read it?". Only `auth-check` (or an actual file fetch) answers the
second question. Note also that the status code moved from pass 1's
**401** to **403**: 401 was "you are not authenticated", 403 is "you are
authenticated and still not on the list" — i.e. the token was added
successfully and the *gate itself* was never accepted.

The token is correctly scoped (`canReadGatedRepos: true`), so **no new
token is needed.** What is needed is one browser action:

> Sign in as `Xaviuer`, open
> <https://huggingface.co/datasets/gaia-benchmark/GAIA>, and click the
> button accepting the dataset's terms. The gate is auto-approval —
> access is granted immediately, with no review queue.

Then `python scripts/freeze_gaia_manifest.py` completes the entire audit
and freezes the subset in one command (§K).

**No workaround was attempted.** No mirror, scraped copy, or unofficial
derivative was searched for or used, per the dataset's own gate terms
and pass 1's stated discipline.

### Consequence for this report

Everything in part 1 of the brief — exact N, validation-split level
distribution, per-task attachment mapping, per-task capability
classification — **remains unmeasured.** It is not estimated below and
it is not carried over from secondary sources. The one-row 165-vs-166
discrepancy stays open. What *is* real is the **public file listing**
(pass 1 §B.1), and where this report projects a count from it, it says
so and states the assumption.

---

## J. Substrate extensions — what changed

### J.1 Modality support table (supersedes §C.2)

| Modality | Val. files (public metadata) | Pass 1 | **Pass 2** | Implementation |
|---|---|---|---|---|
| Tabular (`.xlsx`/`.csv`) | 14 | Supported | **Supported** | stdlib OOXML + `csv` |
| Text (`.txt`/`.py`/`.jsonld`/`.pdb`) | 4 | Supported | **Supported** | direct read |
| Archive (`.zip`) | 2 | Supported | **Supported** | member listing |
| PDF (`.pdf`) | 3 | *Conditional* | **Supported** | **PyMuPDF, pinned** |
| Office doc (`.docx`/`.pptx`) | 2 | *Not implemented* | **Supported** | **stdlib OOXML** |
| Image (`.png`/`.jpg`) | 10 | UNSUPPORTED | **UNSUPPORTED** | explicit error |
| Audio (`.mp3`) | 3 | UNSUPPORTED | **UNSUPPORTED** | explicit error |
| Video | 0 (val) | UNSUPPORTED | **UNSUPPORTED** | explicit error |

**Image/audio/video are untouched and remain unconditionally
UNSUPPORTED at the substrate level**, exactly as pass 1 designed.
Nothing in this pass added, or created a route to, any perception
capability — including the new sandbox, which cannot reach the
attachment at all (§J.3). A test explicitly pins this so a future edit
cannot loosen the fairness line by accident.

Net effect: **5 of the 13 previously-unreadable attachment files (3 PDF
+ 1 DOCX + 1 PPTX) become readable**, and the honest coverage ceiling
drops from ~13 tasks to the 13 image/audio tasks only.

**PDF** — PyMuPDF `>=1.24,<2`, pinned in a new `gaia`
optional-dependency extra in `pyproject.toml`. Deliberately not added to
core dependencies: that would make every other benchmark's CI install a
~20 MiB native wheel for a capability only GAIA uses. Imported as
`pymupdf`, not the deprecated `fitz` alias (pinned by a test, so a
future PyMuPDF bump that removes the alias cannot silently fail a
sweep). Pages are joined with a **form feed** so "on page N …" questions
keep their page boundary. **No OCR** is attempted — a scanned,
image-only PDF yields empty text rather than silently wrong text,
because OCR is a vision capability and vision is UNSUPPORTED for
everybody.

**DOCX/PPTX** — read with the **standard library only** (`zipfile` +
`xml.etree`), matching `_read_xlsx`'s existing precedent rather than
adding `python-docx`/`python-pptx`. Deterministic, no LLM. Handles: text
fragmented across `<w:r>` runs (real Word files split a sentence at
every formatting change — a naive one-`<w:t>`-per-paragraph reader
truncates silently), table-cell text, `<w:tab>`/`<w:br>`, DrawingML
shape text, and **speaker notes**. Slide parts are ordered by the
**numeric** suffix, not lexicographically — `slide10.xml` sorts before
`slide9.xml` as a string, which would silently renumber a deck, and
"what is on the Nth slide" is a real GAIA question shape. A fixture with
eleven slides exists specifically to catch that. Documented scope limits
(headers/footers/footnotes not read; table geometry flattened;
`presentation.xml` slide order not consulted) live on the functions.

**A change worth flagging to a reviewer**: `support_status_for()` no
longer *probes* the environment. Pass 1 resolved PDF support by
attempting an import, which made the substrate's declared capability set
vary by machine — and therefore would have made a frozen subset
non-reproducible, since a manifest built on a machine without PyMuPDF
would silently exclude different tasks. Declared support is now a fixed
property of the code. A missing pinned dependency raises the new
`MissingSubstrateDependencyError`, which is deliberately **not** an
`UnsupportedModalityError`: one means "this machine is provisioned
wrongly, run `pip install`", the other means "this substrate withholds
this capability from everyone on purpose". Conflating them would let a
forgotten install show up in a results table as a principled coverage
ceiling. `check_substrate_dependencies()` is a preflight that the
manifest freezer runs first.

### J.2 Tool surface (supersedes §C.1)

`compute(expression)` — the restricted-AST arithmetic evaluator — has
been **removed and replaced** by `run_python(code)`. The rename is
deliberate: a tool whose name understates what it does is a
documentation bug in a security-relevant place.

| Tool | Status |
|---|---|
| `search(query, limit)` | unchanged |
| `open_url(url)` | unchanged |
| `inspect_file(offset, limit)` | text / archive / **PDF / DOCX / PPTX** |
| `inspect_table(max_rows)` | unchanged |
| **`run_python(code)`** | **NEW — real sandboxed interpreter (§J.3)** |

The fairness invariant is preserved exactly as pass 1 built it:
execution limits are fields on the **registry**, not parameters of the
call, so `run_python` takes `(self, code)` and nothing else. A
caller-chosen timeout would have meant "ANTMAN got 60s of compute and
the baseline got 10s" could happen silently. Tests assert
`react.registry is antman.registry` and that both shapes report the same
budget.

### J.3 The sandbox — mechanism writeup

**This is the security-sensitive component.** The authoritative writeup
is the module docstring of
`src/ant/evaluation_suite/gaia_sandbox.py`, which is written to be read
on its own; what follows reproduces it in full because the brief asked
for this section to be reviewable standalone.

#### Layer 1 — OS level (parent process)

* **Process separation.** `subprocess.Popen([interpreter, ...])`. A
  genuinely separate OS process with its own interpreter and address
  space. **There is no `exec()`/`eval()` of model-chosen source in the
  harness process anywhere.**
* **Interpreter flags `-S -s -P`.** `-S` skips `site`, so
  **site-packages is not importable** (stdlib only). `-s` drops the
  per-user site directory. `-P` stops the script's own directory being
  prepended to `sys.path`, so **this repository is not importable from
  inside the sandbox** even though the child script physically lives in
  it. `-I` was deliberately *not* used: it implies `-E`, which would
  discard `PYTHONHASHSEED=0` and reintroduce hash randomisation, and the
  environment is already fully controlled.
* **Base interpreter, not the venv's.** On Windows a venv's
  `Scripts/python.exe` is a trampoline that re-execs the base
  interpreter as a *second* process, colliding with
  `ActiveProcessLimit = 1`. Found the hard way (every call failed with
  "Unable to create process"); now pinned by a test.
* **Scrubbed environment.** The child's `env` is an **allowlist**, never
  a copy of `os.environ`: `SystemRoot`/`SystemDrive`/`windir`/`COMSPEC`
  and a `PATH` pinned to the system directory on Windows,
  `/usr/bin:/bin` on POSIX, plus `TMP`/`TEMP`/`TMPDIR` pointed at
  scratch. **No inherited variable reaches it** — not `HF_TOKEN`, not
  `OPENAI_API_KEY`, not `PYTHONPATH`. Verified by a test that plants
  sentinel secrets in the harness environment and asserts the child
  cannot see them.
* **Ephemeral scratch directory as cwd.** A fresh `mkdtemp()` per call,
  removed in a `finally`. Never the repository checkout, never a
  caller-supplied path. Verified by test to start empty on every call
  and to be gone afterwards.
* **Wall-clock timeout** via `communicate(timeout=…)`, then a kill of
  the whole process **tree**: `os.killpg` against a session created with
  `start_new_session=True` on POSIX, `TerminateJobObject` on Windows.
  Killing only the direct child would leave grandchildren running.
* **Memory limit.** POSIX: `resource.setrlimit(RLIMIT_AS)` plus
  `RLIMIT_CPU`, `RLIMIT_FSIZE`, `RLIMIT_CORE`, installed by the child
  *before it reads a single byte of user code*. Windows: a **Job
  Object** (`CreateJobObjectW` + `SetInformationJobObject` with
  `JOBOBJECT_EXTENDED_LIMIT_INFORMATION`) carrying
  `JOB_OBJECT_LIMIT_PROCESS_MEMORY`, `JOB_OBJECT_LIMIT_JOB_MEMORY`,
  `JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 1` and
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, applied through `ctypes`. That
  last flag means that if the harness itself crashes, Windows tears the
  sandboxed process down with it. **Fail closed**: if a limit cannot be
  installed on either platform, the call raises rather than running
  unbounded.
* **The spawn/limit race is closed.** A Job Object can only be attached
  after `CreateProcess` returns, which normally leaves a window in which
  the child is alive and unconstrained. Here the child **blocks on
  `sys.stdin.read()`** and the parent does not write the code until
  after the job is attached, so the child provably never saw the code
  while unconstrained.
* **The result channel is a file, not stdout.** User code controls
  stdout and could forge any sentinel placed there, so the child writes
  a JSON result inside scratch *after* user code has stopped. Its
  **absence** is the unambiguous signal that the child was killed.

#### Layer 2 — interpreter level (child process)

A **PEP 578 audit hook** (`sys.addaudithook`). Two properties earn it
its place: a hook **cannot be removed** once installed (CPython exposes
no `removeaudithook`), and the events are raised from inside CPython's
own C implementations, so pure-Python re-routing
(`getattr(os, "sys" + "tem")`, `builtins.__dict__["open"]`) does not
evade it. The policy:

* **Filesystem** — `open` and the `os.*`/`shutil.*` path events are
  bounds-checked. Read+write is allowed **only** inside the ephemeral
  scratch directory; read-only is additionally allowed under
  `sys.base_prefix`/`sys.prefix`, which is what makes `import json`
  work. Everything else — this repository, `.env`, the user's home
  directory — is refused. Write intent is inferred from both `mode` and
  `flags`, and anything ambiguous is treated as a write (fail closed).
* **Network** — `socket.*` audit events denied; `socket`, `ssl`,
  `http`, `urllib`, `requests`, `httpx`, `ftplib`, `smtplib` and
  friends denied at import.
* **Process creation** — `subprocess.*`, `os.system`, `os.exec*`,
  `os.spawn*`, `os.fork*`, `os.startfile` denied as events;
  `subprocess`, `multiprocessing`, `pty` denied at import.
* **Native code** — `ctypes` denied at import and `ctypes.*` denied as
  events, because `ctypes` is the one stdlib module that converts Python
  into arbitrary machine code — exactly the thing an audit hook cannot
  see past.
* **Environment mutation** — `os.putenv`, `os.unsetenv`, `os.chdir`
  denied.

#### A real bypass, found and closed

The first version of the hook was **defeated in testing**:

```python
import importlib
importlib.import_module("subprocess")   # succeeded
import subprocess                       # refused
```

CPython raises the `import` audit event from
`PyImport_ImportModuleLevelObject` — i.e. from `builtins.__import__` and
the `IMPORT_NAME` opcode — but **not** from `importlib.import_module`,
which drops straight into the pure-Python `_bootstrap._gcd_import`. This
was found by a test, not by reading documentation.

Two things close it:

1. A **`sys.meta_path` finder** refusing the same module list, which
   every import route converges on. It is *removable* (`sys.meta_path`
   is an ordinary list), so it is not relied on alone.
2. The **private extension modules are blocked alongside their public
   wrappers** — `_ctypes`, `_socket`, `_winapi`, `_multiprocessing`.
   This is the load-bearing half: `ctypes/__init__.py` reaches its
   native half with a plain `from _ctypes import …` **statement**, which
   fires the unremovable audit hook. The native-code route therefore
   stays shut even against an attacker who has defeated every removable
   layer.

Probed directly. All refused:

| Attack | Result |
|---|---|
| `importlib.import_module("subprocess")` | `SandboxDenied` |
| `sys.meta_path.pop(0)` then `import ctypes` | `SandboxDenied` (audit hook) |
| `sys.meta_path.pop(0)` + `import_module("ctypes")` → `memmove` | `SandboxDenied` on `_ctypes` |
| `sys.meta_path.clear()` + `import_module("_ctypes")` | child dies; reported as killed |
| `os.system`, `subprocess.run`, `socket.connect`, `urlopen` | `SandboxDenied` |
| read `.env` / repo source / list `$HOME` | `SandboxDenied` |
| write or delete anything outside scratch | `SandboxDenied` |
| `import numpy` / `import ant` | `ModuleNotFoundError` |
| allocate 3.2 GiB under a 256 MiB cap | `MemoryError` |
| `while True: pass` | killed at the wall clock |
| read the task's own attachment | `SandboxDenied` |

#### What this does NOT protect against — read this part twice

**I am not fully confident this isolation is robust against a determined
adversary**, and this list exists so that nobody has to take my word for
it either way.

1. **This is not a container, a VM, a seccomp filter, or a jail.** There
   is no kernel-enforced syscall restriction, no namespace, no chroot.
   The child runs as the **same OS user with the same privileges** as
   the harness. If the interpreter-level guard is bypassed, nothing
   below it stops a syscall.
2. **The network guard is interpreter-level only.** There is no firewall
   rule and no network namespace. It rests entirely on the import denial
   plus `socket.*` audit events. Any path to raw syscalls — a CPython
   bug, an unaudited C extension, a native-code trick I did not
   anticipate — bypasses it completely. If genuine network isolation
   matters, run this inside a container with networking disabled; that
   is the correct fix and this module is not it.
3. **Audit-hook coverage is only as complete as CPython's event list,
   and I already found one gap the hard way** (the `importlib` bypass
   above). That one is closed; I cannot prove there is not another. The
   event list also drifts between CPython releases.
4. **A CPython escape is a real, unpatched-by-me risk.** Audit hooks
   have had bypasses historically. The native-code route is closed at
   three independent points and I attacked it deliberately — **that is
   evidence, not proof.** Those are the attacks I thought of.
5. **Every escape probe above was run on Windows only.** The mechanisms
   are portable and the `_ctypes` import block is platform-independent,
   but the same probes have **not** been executed on Linux, and this
   project's sweeps run on Linux (DeltaAI). **The POSIX
   `resource.setrlimit` path has had no live exercise at all.** Re-run
   `tests/test_gaia_sandbox.py` on the target machine before trusting
   Linux behaviour. This is the single biggest gap in the verification.
6. **`SandboxDenied` is catchable.** User code can wrap a denied call in
   `try/except` and retry or probe. The operation still never happens —
   the hook raises *before* the effect — but the sandbox is a wall, not
   a tripwire, and it does not report probing back to the caller.
7. **CPU is bounded only by the wall clock** (plus `RLIMIT_CPU` on
   POSIX). A busy loop consumes one core for the whole timeout. There is
   no CPU-share cgroup.
8. **Windows has no in-process `RLIMIT_AS` equivalent** — the memory cap
   there is entirely the Job Object's.
9. **TOCTOU on path checks.** Paths are resolved with
   `os.path.realpath` at check time; a symlink swapped between the check
   and the open is theoretically exploitable. The only writable
   directory is one created microseconds earlier, which makes this
   narrow, not absent.
10. **Disk usage inside scratch is capped only on POSIX**
    (`RLIMIT_FSIZE`); the Windows Job Object caps memory, not disk.

**My honest assessment**: this is solid defence in depth against *an LLM
doing something careless or mildly adversarial*, which is the actual
threat model of an evaluation harness. It is **not** a safe place to run
code from a genuinely hostile party. I would not object to this being
run as-is on a benchmark sweep; I would object to it being described as
"sandboxed" without the qualifications above.

#### Deliberate functional limits (not security, but they shape results)

* **Stdlib only.** `-S` means no `numpy`, no `pandas`. This is a
  fairness and reproducibility choice as much as a security one: with
  site-packages visible, the substrate's capability would depend on
  whichever machine ran the sweep. `math`, `statistics`, `fractions`,
  `decimal`, `datetime`, `itertools`, `re`, `json` and `csv` cover
  GAIA's computational needs, and the limit binds **every method
  identically**.
* **No attachment access from inside the sandbox.** The scratch
  directory starts empty; the attachment is reached only through
  `inspect_file`/`inspect_table`, which apply the modality policy. This
  keeps the modality policy un-bypassable — otherwise `run_python` would
  be an image reader and the vision fairness line would be decorative.

---

## K. The capability-covered subset — rule implemented, freeze pending

§G recommended Option B but could not implement it. The **rule is now
real, tested code** (`src/ant/evaluation_suite/gaia_subset.py`); the
**freeze is one command away** and is blocked only by §I.

### K.1 The rule

> Include a validation task **iff** every modality it requires is
> `SUPPORTED` in `gaia_scope.MODALITY_SUPPORT`. Concretely: include it
> iff it declares **no attachment**, **or** its attachment's file
> extension maps to a SUPPORTED modality — now `TEXT`, `TABULAR`,
> `ARCHIVE`, **`PDF`** and **`OFFICE_DOC`**. Exclude it iff that
> modality is `UNSUPPORTED`/`NOT_IMPLEMENTED` — `IMAGE`, `AUDIO`,
> `VIDEO`, `UNKNOWN`.

### K.2 The leakage boundary is structural, not conventional

The rule may not look at the gold answer, `Annotator Metadata`, model
performance, or the question text. That is enforced the same way pass 1
enforced territory derivation — **by there being no parameter through
which it could arrive**:

* `select_subset()` takes `SubsetCandidate`s, and that dataclass's
  fields are exactly `{task_id, level, file_name, question_chars}`.
  There is **no** `reference`/`answer`/`annotator_metadata`/`score`
  field. A test asserts the exact field set and that it is disjoint from
  a forbidden-name list.
* `candidates_from_examples()` is the projection where the boundary is
  crossed in the *safe* direction: a `TaskExample` carries gold and
  `_audit_only`, a `SubsetCandidate` cannot. A test plants sentinels in
  both and asserts neither survives the projection.
* `level` is carried for **reporting only**. Rather than asserting that
  in a comment, a test **blanks every level and asserts the partition is
  identical**. Another does the same for question length.
* No run has happened, so nothing in the module *can* read a score.

### K.3 The manifest file

`third_party/manifests/gaia/manifest.json` **exists and is committed**,
in its honest **not-frozen** form: `status: "blocked_on_gated_access"`.

Every field describing the **rule**, the **declared modality policy**
and the **resolved dependency versions** is real and final. Every field
that would describe the **selected tasks** is `null` — deliberately
`null` rather than `0`/`[]`, because *"not computable yet"* and
*"computed, found none"* must not share a representation.
`require_frozen_manifest()` refuses the file as-is, so it cannot
silently back a run and produce a results file with N=0 that reads like
a completed sweep. Its structure follows the `third_party/manifests/`
convention (provenance block + selection rule + ids + ledger + a
`status` gate), matching the RepoProbe and SWE-QA-Pro manifests.

`scripts/freeze_gaia_manifest.py` does the whole job in one command once
the gate is accepted: preflight the dependencies, load the live split,
apply the rule, print the audit, rewrite the manifest with
`status: "frozen"`. It **deliberately refuses to fall back to synthetic
fixtures** — a manifest silently frozen over fixtures would be worse
than no manifest — and it prints only counts, distributions and task
IDs, never dataset content.

### K.4 Projected sizes — labelled as projection, not measurement

Derived from the **verified public file listing** (pass 1 §B.1), which
is real metadata, under one stated assumption.

| Quantity | Value | Basis |
|---|---|---|
| Validation total N | **UNMEASURED** (165 or 166 — unresolved) | §I |
| Attachment files | 38 | verified public listing |
| — now readable | **25** (xlsx 13, pdf 3, zip 2, csv 1, docx 1, jsonld 1, pdb 1, pptx 1, py 1, txt 1) | §J.1 |
| — still refused | **13** (png 8, jpg 2, mp3 3) | §J.1 |
| Projected excluded | **13** — 10 IMAGE, 3 AUDIO, 0 VIDEO | see assumption |
| Projected retained | **152 or 153** | N − 13 |

**Assumption, stated because it is the weak link**: that each of the 38
attachment files belongs to exactly one task. GAIA's `file_name` is a
scalar per row, so **at most** 38 tasks have attachments; if two tasks
share a file, both the exclusion count and the retained count shift.
This is exactly the per-task attachment mapping §I could not obtain.
**These numbers must be recomputed live before anything is frozen** and
are given here only so the cost estimate in §L has a scale.

Retained **level** distribution and retained **file-type** distribution
cannot be projected at all — the validation-split level breakdown has
never been public (the paper's Table 4 is whole-dataset), and the file
listing carries no level information. Both are computed and printed by
the freeze script.

---

## L. Cost estimate — six methods over the retained subset

### L.1 GAIA's cost model is structurally different, and that is the headline

**GAIA scoring is deterministic and free.** The official
`question_scorer` is vendored verbatim and run once per task:
`JudgeType.DETERMINISTIC`, `n_judge_calls = 0`, `judge_cost_usd = 0.0`.
**There is no LLM judge.** A GAIA sweep's entire cost is generation.

That is a different cost model from every other substrate in this suite.
For contrast, from the **measured** SWE-QA-Pro N=80 run: judging cost
**$0.01866 per task per method** (3 GPT-5 calls). Over 6 methods × 152
tasks that is a **$17.02 line item GAIA simply does not have** — and for
the cheap methods it *dominates*: `direct` cost $0.0046/task to generate
and $0.0187/task to judge, i.e. **judging was 4× generation**. On GAIA,
`direct` costs $0.0046/task, full stop.

### L.2 Grounding — real measurements, not guesses

Per-task **generation** cost, measured from completed runs already in
this project, all `gpt-4.1` at the repo's own pinned pricing
($2.00/$8.00 per 1M in/out, `providers/pricing.py`):

| Method | Source run | n | mean $ | sd | p50 | p90 | max | LLM calls | in tok | out tok |
|---|---|---|---|---|---|---|---|---|---|---|
| Direct | SWE-QA-Pro 80 | 80 | **0.0046** | 0.0011 | 0.0046 | 0.0061 | 0.0074 | 1.0 | 85 | 553 |
| Sparse Retrieval | SWE-QA-Pro 80 | 80 | **0.0194** | 0.0051 | 0.0192 | 0.0246 | 0.0411 | 3.6 | 4 154 | 1 383 |
| Dense Retrieval | RepoProbe dense | 83 | **0.0143** | 0.0029 | 0.0141 | 0.0182 | 0.0252 | 1.0 | 1 764 | 1 349 |
| ReAct (matched) | SWE-QA-Pro 80 | 80 | **0.1094** | 0.1261 | 0.0595 | 0.3007 | 0.5062 | 18.8 | 47 104 | 1 896 |
| ANTMAN | SWE-QA-Pro 80 | 80 | **0.4986** | 0.4274 | 0.4028 | 0.9861 | 2.8193 | 53.7 | 200 860 | 12 112 |
| **S2G-RAG** | — | — | **UNMEASURED** | — | — | — | — | — | — | — |

**S2G-RAG has no implementation and no measured run in this repository**
(the main checkout's branch is named `s2g-rag-baseline`, but no such
module exists). Its figure below is an extrapolation from the pipeline
shape — decompose → per-subquestion retrieval → graph expansion →
synthesise, i.e. roughly 5–8 LLM calls, about 2–3× sparse retrieval —
and it is the **least trustworthy row in this section**.

GAIA-specific fixed prompt overhead, **measured directly** from the
vendored artefacts (`o200k_base`):

| Component | Tokens |
|---|---|
| Official GAIA system prompt (vendored verbatim from the leaderboard Space) | **155** |
| `GaiaToolRegistry.tool_specs()` rendered as JSON | **266** |
| Fixed per-call floor for a tool-using method | **421** |

The synthetic fixtures' attachment render costs (8–180 tokens) were
measured but are **not** used as a basis — they are hand-authored toys,
not representative of real GAIA attachments, whose sizes could not be
measured (§I).

### L.3 Estimate

Per-task costs carried over unchanged from §L.2, scaled to the
**projected** retained N = 152 (§K.4). Same model (`gpt-4.1`), same
pricing, no judge.

| Method | $/task | × 152 tasks | Basis |
|---|---|---|---|
| Direct | 0.0046 | **$0.70** | measured |
| Dense Retrieval | 0.0143 | **$2.17** | measured |
| Sparse Retrieval | 0.0194 | **$2.95** | measured |
| S2G-RAG | ~0.04–0.06 | **$6–9** | **extrapolated** |
| ReAct (matched) | 0.1094 | **$16.63** | measured |
| ANTMAN | 0.4986 | **$75.79** | measured |
| **Total, one pass of all six** | | **≈ $104–107** | |
| *Judge cost* | *0.0000* | ***$0.00*** | *deterministic official scorer* |

### L.4 Honest uncertainty on transferring these numbers to GAIA

These are **real measurements of the same six methods on a different
substrate**, not measurements on GAIA. Directional adjustments, none of
which are applied above because none can be measured yet:

* **Likely cheaper on GAIA**: the retrieval methods carry no repository
  corpus. Sparse and dense retrieval spent most of their 1.7k–4.2k input
  tokens on repo chunks; a GAIA task's context is a question, at most
  one attachment, and some page text. **Direct** is essentially exact
  already (question + the 155-token system prompt, versus SWE-QA-Pro's
  85-token question-only input).
* **Could be more expensive on GAIA**: `open_url` returns whole web
  pages, which are frequently larger than a code chunk, and GAIA's
  multi-hop questions need more search rounds than a single-repo
  question does. This pushes **ReAct** and **ANTMAN** — the two
  loop-shaped methods, and the two that dominate the total — in the
  opposite direction.
* **The variance is larger than the mean is comfortable with.** ReAct's
  sd (0.126) exceeds its mean (0.109); ANTMAN's max (2.82) is 5.7× its
  mean. Budget against the **p90** column, not the mean: at p90, ReAct
  is $45.71 and ANTMAN $149.89 over 152 tasks.
* **N is unmeasured.** 152 is a projection resting on the
  one-file-per-task assumption (§K.4).

**A defensible planning figure: $100–150 for one full six-method pass at
gpt-4.1, with ANTMAN about 70% of it, and a realistic worst case around
$250.** I would not quote tighter than that without a 10-task pilot on
the real data — which, at ANTMAN's measured mean, would cost roughly
**$7** and would convert every estimate in this section into a
measurement. That pilot is the obvious next step once the gate opens,
and it is the user's decision, not this pass's.

---

## M. Test and regression status

| Suite | Result |
|---|---|
| GAIA tests (`test_gaia_{adapter,scope,tools,scorer,sandbox,subset}.py`) | **258 passed** (138 → 258; **120 new**) |
| Full suite `pytest tests/ -q` | **1065 passed, 1 skipped, 20 failed** |
| `ruff check .` | **clean** |

**The 20 failures are pre-existing and environment-only, and this was
verified rather than assumed.** They are all in
`tests/test_chainrag.py`, `tests/test_dense_retrieval_document.py` and
`tests/test_dense_retrieval_repo.py`, and every one resolves to
`ModuleNotFoundError: No module named 'fastembed'` — the optional
`dense` extra is not installed in this sandbox's venv. **The identical
20 failures were reproduced on the base commit with this pass's changes
stashed.** Nothing in this pass imports or touches ChainRAG or dense
retrieval.

(For the record: pass 1 attributed `test_chainrag.py`'s failure to
`networkx`/`spacy`. In this venv those are present; the actually-missing
dependencies were `rank_bm25` — installed during this pass, which is why
the module now collects at all — and `fastembed`, which is not.)

Isolation held. The only file touched outside GAIA's own is
`pyproject.toml`, which gains one new optional-dependency group.
ANTMAN's coordinator, other benchmarks' adapters, and the shared
`search_tool_factory` call sites are untouched.

---

## N. Open items for the human reviewer (supersedes §H)

1. **Accept the GAIA gate in a browser as `Xaviuer`** (§I). This is the
   only blocker, it is a single click, and the token is already correct.
   Then run `python scripts/freeze_gaia_manifest.py`.
2. **Review the sandbox** (§J.3), in particular limit 2 (network is
   interpreter-level only) and limit 5 (**never exercised on Linux**,
   which is where sweeps run). If the answer is "run it in a container",
   that is a reasonable call and this component is ready to be dropped
   into one.
3. ~~Decide PDF~~ — done, pinned (§J.1).
4. ~~Decide `.docx`/`.pptx`~~ — done, stdlib (§J.1).
5. ~~Decide A vs B~~ — B is implemented; **A remains available** and the
   pass-1 caveat stands unchanged: a supported-modality subset is **not
   comparable to published full-validation GAIA numbers** and must be
   reported as "GAIA-validation, supported-modality subset,
   N=<recomputed>" with the exclusion ledger printed alongside.
6. **Consider a 10-task pilot (~$7)** before committing to a full sweep,
   to replace §L's transferred estimates with measurements.
7. **Confirm no inference** is launched until 1–2 are settled. **None
   was launched in this pass.**
