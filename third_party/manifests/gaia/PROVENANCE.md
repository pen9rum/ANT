# Vendored GAIA official artifacts

## `scorer.py`

**Vendored byte-for-byte verbatim. Do not edit, reformat, lint, or
"improve" this file.** It is the benchmark's own scorer, not ours; any
local change silently makes our numbers non-comparable to the official
leaderboard. `ant.evaluation_suite.gaia_scorer` asserts its SHA-256 at
import time so a stray edit fails loudly instead of quietly changing
every GAIA score.

| Field | Value |
|---|---|
| Source | Hugging Face Space `gaia-benchmark/leaderboard`, file `scorer.py` |
| Revision (Space commit) | `9f133d71362e77b3539f1514f31b9c101a545fec` |
| SHA-256 of vendored bytes | `0d44c07f3046eec521697c22e3eaca8719cc81e422a8eaf32695c5f22bdac6e2` |
| Size | 3212 bytes |
| License | Apache-2.0 (declared in the Space's own `README.md` front matter) |
| Retrieved | 2026-09-16, via `huggingface_hub.hf_hub_download(repo_type="space")` |

This is **code only** — it contains no GAIA question, answer, or
attachment content, so vendoring it into version control does not
conflict with the dataset's own redistribution terms (see below).

The Space is public and ungated; only the *dataset* repo
(`gaia-benchmark/GAIA`) is gated. That asymmetry is why this pass could
vendor the real official scorer without dataset access.

### Public API used

- `question_scorer(model_answer: str, ground_truth: str) -> bool`

Everything else (`normalize_number_str`, `split_string`, `normalize_str`)
is the scorer's own internals, re-exported by `gaia_scorer` only so unit
tests can exercise the normalization layers individually.

### Known quirks of the official implementation (preserved, not fixed)

These are faithfully kept because matching the official leaderboard
matters more than our opinion of the code:

1. `normalize_number_str` returns `float("inf")` on an unparseable
   answer rather than raising, so a non-numeric answer to a numeric
   question scores `False` instead of erroring.
2. It strips only `$`, `%`, and `,` — no other unit is handled.
3. List answers are split on `,` **or** `;`, compared pairwise and
   **order-sensitively**, and a length mismatch is an immediate `False`.
4. Per-element list comparison uses `remove_punct=False`, while the
   whole-string branch uses `remove_punct=True`. This asymmetry is
   deliberate upstream (see the "question with the fish" comment).
5. `normalize_str` removes *all* whitespace (so `sea gull` == `seagull`).
6. The scorer `print()`s a line per call and can emit a `UserWarning`.
   `gaia_scorer` suppresses that chatter at the call boundary without
   touching this file.
7. `import numpy as np` is unused upstream but retained verbatim.

## Dataset content is deliberately NOT vendored

`gaia-benchmark/GAIA`'s own gate text reads:

> To avoid contamination and data leakage, you agree to not reshare this
> dataset outside of a gated or private repository on the HF hub.

and the leaderboard adds "Please do not repost the public dev set, nor
use it in training data for your models." No GAIA question, answer, or
attachment byte is committed anywhere in this repository. Raw downloads
land in `.gaia-data/` (gitignored). `synthetic_fixtures.json` in this
directory is hand-authored by us and contains **no** GAIA material.
