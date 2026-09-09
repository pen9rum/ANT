# Repository file-universe policy: audit and rationale

Status: frozen policy specification for `src/ant/evaluation_suite/repo_scope.py`'s
`EvalRepoEnvironment`, written from a general, benchmark-independent audit
-- not from noticing sphinx uses `.rst` or sqlfluff uses `.sql` and patching
those two in. If the code and this document ever disagree, that is a bug
to fix, not an ambiguity to interpret away.

## What was audited, and how

File-extension distributions across 34 already-cloned/shallow-cloned
repositories: all 26 SWE-QA-Pro repos already checked out under `repos/`,
plus 8 RepoProbe repos shallow-cloned specifically for this audit
(selected for language diversity against RepoProbe's own `primary_language`
metadata: Swift, C#, Lua, Ruby, PHP, C++, Rust x2 -- languages ANT core's
`TEXT_EXTENSIONS` allowlist had **zero** coverage for). No gold answers
were read at any point; only raw repository file trees and (for the
"can a question legitimately depend on this" question) SWE-QA-Pro's own
`qa_type`/`cluster` taxonomy metadata, which is benchmark documentation,
not an answer.

## The old policy's failure mode, precisely

`ant.environment.repo.TEXT_EXTENSIONS` is a closed, ~20-entry allowlist
(`.c .cc .cfg .conf .css .go .h .html .java .js .json .jsx .md .py .rs .sh
.toml .ts .tsx .txt .yaml .yml`, plus exact-name `Dockerfile`/`Makefile`/
`README`). Measured directly against RepoProbe's own `primary_language`
distribution across its 50 repos: **Swift (4 repos), C# (3), Lua (3),
Ruby (2), PHP (1)** have no allowlisted extension at all -- for a repo
whose primary language is one of these, the old policy would show ANT's
own worker population (and, before this pass's fix, every baseline reusing
the same allowlist) almost nothing of the actual codebase.

## Extension-distribution audit (real data, 34 repos)

Top extensions found post-`IGNORED_DIRS` pruning, across all 34 repos
(files / repos-containing-it / classification / old-policy verdict):

| extension | files | repos | classification | old (`TEXT_EXTENSIONS`) |
|---|---|---|---|---|
| `.py` | 7439 | 27 | source | included |
| `.cs` | 2796 | 1 | source (C#) | **excluded** |
| `.yml` | 2456 | 34 | config | included |
| `.sql` | 1473 | 1 | source/fixture (SQL) | **excluded** |
| `.rst` | 1136 | 22 | documentation | **excluded** |
| `.txt` | 1077 | 26 | text | included |
| `.md` | 880 | 32 | documentation | included |
| `.f90` | 862 | 1 | source (Fortran) | **excluded** |
| `.php` | 710 | 1 | source | **excluded** |
| `.rb` | 673 | 3 | source (Ruby) | **excluded** |
| (no extension) | 663 | 34 | mixed (LICENSE, dotfiles, scripts) | name-based only |
| `.rs` | 452 | 3 | source (Rust) | included |
| `.yaml` | 297 | 21 | config | included |
| `.json` | 285 | 24 | config/data | included |
| `.svg` | 226 | 21 | markup (icons/diagrams) | **excluded** |
| `.js` | 217 | 13 | source | included |
| `.swift` | 186 | 1 | source | **excluded** |
| `.html` | 169 | 22 | markup | included |
| `.lua` | 141 | 3 | source | **excluded** |
| `.css` | 131 | 26 | stylesheet | included |
| `.h` | 124 | 5 | source (C header) | included |
| `.csproj` | 121 | 1 | build config (XML) | **excluded** |
| `.dat` | 113 | 3 | mixed (some binary, some text fixture) | **excluded** |
| `.csv` | 111 | 4 | structured text data | **excluded** |
| `.c` | 111 | 3 | source | included |
| `.sass` | 100 | 1 | stylesheet | **excluded** |
| `.po` | 99 | 1 | translation source (gettext) | **excluded** |
| `.cpp` | 91 | 2 | source (C++) | **excluded** |
| `.toml` | 74 | 17 | config | included |
| `.hy` | 74 | 1 | source (Hy -- hylang/hy's own language) | **excluded** |
| `.ini` | 28 | 15 | config | **excluded** |

Binary/generated categories (`.png .pkl .dat[some] .bmp .jpg .tif .fli
.mo .ttf .woff .zip .pdf .npy .icns ...`) were confirmed correctly
excluded already and remain excluded under the new policy -- via content
sniffing, not an extension list (see below), so this also covers formats
that never showed up in this specific 34-repo sample.

## Can a benchmark question legitimately depend on an excluded file?

Checked via SWE-QA-Pro's own `qa_type`/`cluster` taxonomy metadata (not
gold answers): the sampled sqlfluff questions include 3/10 in cluster
`"0.1: SQL / structured grammar / templating Engine"`, and the sampled
sphinx questions include one in cluster `"6.2: build pipeline / doc
building / Sphinx / cloud provisioning"` -- both directly naming
territory the old allowlist excluded outright (`.sql` fixtures, `.rst`
docs). Plausible, disclosed dependency, not certain -- gold answers were
never read to confirm it further, per the audit's own constraint.

## The principled policy: content-based, not an ever-growing extension list

An extension allowlist is inherently reactive -- it is correct only for
languages/formats someone thought to add before a repo needing them
showed up. `EvalRepoEnvironment` (in `evaluation_suite/repo_scope.py`,
evaluation-side, not `ant.environment.repo`) replaces it with:

1. **Same `IGNORED_DIRS`** as ANT core (reused directly, not re-derived --
   VCS internals, `.venv`/`node_modules`/dependency directories, `build`/
   `dist` outputs, caches).
2. **Content-based text detection**, not extension matching: read a
   bounded prefix (8 KiB) of each candidate file; a NUL byte anywhere in
   it is treated as conclusive evidence of binary content (the same
   heuristic git and most text/binary detectors use) -- with a UTF-8/ASCII
   decode check as a secondary signal. This is why `.svg`/`.csv`/`.dat`/
   `.po`/`.sass`/`.cs`/`.rb`/`.php`/`.swift`/`.lua`/`.f90`/`.cpp`/`.hy`/
   `.ini` -- and any extension this policy has never seen before -- are
   now included when they're genuinely text, with zero new allowlist
   entries required.
3. **A 1 MiB size cap** (`MAX_TEXT_FILE_BYTES`) -- what "continue
   excluding... large generated artifacts" means under a content-based
   policy: it doesn't matter what produced an oversized file, only
   whether it's plausibly something a person reads.
4. **A short, fixed, industry-standard lockfile-basename exclusion**
   (`package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`, `Cargo.lock`,
   `Gemfile.lock`, `poetry.lock`, `composer.lock`, `go.sum`,
   `Pipfile.lock`, `mix.lock`) -- these pass the text sniff but carry
   almost no narrative signal and are frequently huge; matched by exact,
   well-known basename (not extension, since `package-lock.json` shares
   its extension with genuinely meaningful `.json` files).
5. **Notebooks (`.ipynb`) excluded**, explicitly decided, not
   overlooked: raw notebook JSON is high-noise for a lexical/embedding
   index (execution counts, cell-output blobs, frequently base64-embedded
   images, no clean separation from the actual prose/code without
   dedicated cell-by-cell extraction). Building that extraction well is a
   distinct feature with real implementation risk this pass does not take
   on. Measured: only 8/34 audited repos contain any `.ipynb` files at
   all, and even there it's a small fraction of total content -- a
   disclosed, revisitable scope decision.

Extensionless-but-meaningful files (`Dockerfile`, `Makefile`, `README`,
`LICENSE`, ...) no longer need their own exact-name allowlist at all: they
pass the content sniff on their own merits (small, valid text), which is
strictly more general than enumerating names.

## Measured effect (real data, all 34 audited repos)

| | old (`TEXT_EXTENSIONS`) | new (`EvalRepoEnvironment`) |
|---|---|---|
| total files covered | 14,040 | 24,642 (+75.5%) |
| lockfiles leaked through | n/a (not a concept) | **0** (confirmed) |
| `.ipynb` included | 0 (extension not listed) | **0** (excluded by policy) |
| `.png`/binaries included | 0 | **0** (content-sniffed out) |

## Can this be implemented evaluation-side, without touching frozen core?

**Yes**, and it is. Traced every consumer of `RepoEnvironment`
(`grep -rn RepoEnvironment src/ant`): the only core function that actually
uses one is `ant.indexing.territories.discover_territories(repo:
RepoEnvironment)`, which calls only `repo.root` and `repo.iter_files()` --
no `isinstance` check anywhere in the codebase requires the literal
`RepoEnvironment` class. `EvalRepoEnvironment` is a plain subclass
(`@dataclass(frozen=True) class EvalRepoEnvironment(RepoEnvironment)`)
overriding only `iter_files()` -- Liskov-substitutable everywhere a
`RepoEnvironment` is expected, not a monkeypatch, not a runtime mutation
of the existing class, and `ant/environment/repo.py` itself is imported
only for its `IGNORED_DIRS` constant, never modified.

Three evaluation-suite call sites (`ant.agents.ant_adapter.AntAgent`,
`ant.agents.matched_react.MatchedReActAgent`,
`ant.agents.retrieval.RetrievalAgent`) now construct `EvalRepoEnvironment`
instead of core's own `RepoEnvironment` directly. **ANT's default runtime
outside this evaluation suite is completely unaffected**: `ant.cli`,
`ant.evaluation.runner`, and `ant.git_refresh` all still construct core's
own `RepoEnvironment` directly and were not touched -- `TEXT_EXTENSIONS`
remains ANT's own default behavior everywhere this evaluation suite isn't
involved.

No frozen core file was modified to implement this policy.
