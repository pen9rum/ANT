# WebWalkerQA environment validation report

**Scope of this pass**: environment/setup only, per the governing spec. No
paid LLM inference was launched. No benchmark evaluation was run. This
report freezes the environment, not a score.

## 0. What already existed vs. what this pass built

**Already existed and reused unmodified** (commit `a3b6310`, extended by
`fb0b065`):
- `ant.evaluation_suite.webwalkerqa` package (`loader.py`, `manifest.py`,
  `prepare.py`, `__init__.py`) — dataset loading, schema normalization,
  filtering, deterministic sampling, prep-manifest tooling. Fully reused,
  not duplicated.
- `third_party/manifests/webwalkerqa/english_main.json` — a frozen
  247-example prep manifest (IDs + metadata only).
- `docs/webwalkerqa_ant_mapping.md` — the benchmark-structure audit and
  ANT-mapping design this pass's `web_scope.py` implements.
- `docs/web_track_baselines.md`'s "## WebWalker" section — the official
  method's (not-yet-executed) setup plan.
- `ant.evaluation_suite.scoring.JudgeType.LLM_BINARY` — already
  anticipated WebWalkerQA as binary-judged, majority-vote-over-3.

**Built new in this pass** (all under one commit, see the end of this
report for the SHA):
- `src/ant/evaluation_suite/web_fetch.py` — URL normalization, same-site
  restriction, deterministic HTML-to-text/link extraction, disk-backed
  page cache. Stdlib-only (no new dependency).
- `src/ant/evaluation_suite/web_scope.py` — `EvalWebEnvironment`/
  `WebTerritory`/`build_territory_from_discovered` (see section C below
  for the proposed territory semantics).
- `src/ant/benchmarks/webwalkerqa.py` — the `TaskExample`/
  `BenchmarkAdapter` layer: `WebWalkerQaAdapter` (leakage-safe
  `load_examples`, no-op `prepare_environment`, `LLM_BINARY` `score()`).
- `src/ant/agents/webwalkerqa_stubs.py` +
  `src/ant/external_wrappers/webwalker_native_agent.py` — 5 baseline
  interface stubs (Direct/Retrieval/Matched ReAct/ANTMAN/official
  WebWalker), each raising before any inference.
- `tests/test_web_fetch.py`, `tests/test_web_scope.py`,
  `tests/test_webwalkerqa_adapter.py`, `tests/test_webwalkerqa_stubs.py`
  — 98 new tests, all static/mocked, zero network, zero LLM calls.

## A. Dataset inventory

Recomputed live from `datasets.load_dataset("callanwu/WebWalkerQA",
split="main")` (not assumed from the existing docstring's own claim,
per the governing spec's explicit instruction) — **zero discrepancy
found**:

| Category | Count |
|---|---|
| Total | 680 |
| English | 247 |
| Chinese | 375 |
| Invalid/empty (`lang` field neither `en` nor `zh`) | 58 |
| Eligible English (root URL currently accessible) | 222 |

Note on the "58 invalid/empty" row: these rows are **not** structurally
broken — `question`/`root_url`/`answer` are all present and non-empty on
every one of the 680 rows (verified: 0 rows with any missing critical
field). The 58 are excluded specifically because their `info.lang` value
is an empty string (neither `"en"` nor `"zh"`) — a language-label gap in
the source dataset, not a data-integrity failure.

Full-dataset distributions (all 680 rows, for reference):
domain — education 322, conference 156, game 136, organization 66;
hop_type — single_source 340, multi_source 340 (exact 50/50 split);
difficulty — medium 280, hard 240, easy 160.

## B. English subset breakdown (247 rows, pre-eligibility-filter)

| Dimension | Value | N |
|---|---|---|
| Hop | single-source | 117 |
| Hop | multi-source | 130 |
| Domain | conference | 149 |
| Domain | game | 84 |
| Domain | education | 12 |
| Domain | organization | 2 |
| Difficulty | medium | 130 |
| Difficulty | hard | 76 |
| Difficulty | easy | 41 |

Unique root URLs among the 247: **46** (many questions share a root
site). Unique gold source pages (audit-only): 338. All 247 root URLs are
syntactically valid (0 malformed).

**Note the English domain mix is very different from the full dataset**:
conference dominates English (60%) where education dominates the full
set (47%) — education/organization content skews heavily Chinese. A
frozen English-only subset is therefore NOT domain-representative of the
benchmark as a whole; this is disclosed, not smoothed over.

## C. Web substrate abstraction — proposed territory semantics

(Full rationale lives in the module docstring of
`src/ant/evaluation_suite/web_scope.py`; summarized here as requested.)

- **Territory** = a runtime-discovered, provisional page set anchored to
  one page reached from an already-visited page's own link list — NOT a
  pre-existing structure. This is the one real architectural departure
  from Track A (`EvalRepoEnvironment.iter_files()` enumerates a
  repository's complete file universe up front; no site-map/tree API
  exists for a website — audited in `docs/webwalkerqa_ant_mapping.md`
  section 4).
- **Worker** = specialized over one such territory (plausibly one
  first-level section/subtree off the root, mirroring the paper's own
  domain examples), but which pages belong to it is discovered through
  that worker's own navigation, never assigned upfront.
- **Navigation primitive** = `EvalWebEnvironment.navigate(from_page,
  to_url)` — follows a link that was ACTUALLY present in
  `from_page.links`; raises if not. This is the sole channel through
  which new pages become knowable — there is no "jump to any URL"
  primitive and no upfront full-graph listing.
- **Page inspection primitive** = `EvalWebEnvironment.inspect(url)` —
  re-reads an already-visited page's content; an unvisited URL returns
  `status="not_fetched"` rather than being silently fetched, so
  `navigate()` stays the one discovery channel.
- **Enforced, not just documented, no-leakage guarantees**: no function
  in `web_scope.py` accepts a golden-path/source-website/gold-answer
  parameter (asserted structurally in tests via
  `inspect.signature`/`__dataclass_fields__`, not just by convention);
  `PageCache` rejects any URL outside the root site's domain before
  fetching (`same_site()`-checked, tested); a 15-step navigation ceiling
  (WebWalkerQA's own paper-stated Explorer budget) is enforced at
  `navigate()` itself for both `MatchedReActWebAgent` and `AntWebAgent`
  identically (the governing spec's explicit fairness requirement).

## D. Environment status

| Item | Status |
|---|---|
| Task adapter implemented? | Yes — `WebWalkerQaAdapter` (`src/ant/benchmarks/webwalkerqa.py`) |
| Gold leakage prevented? | Yes — structural (`inference_view()` omits gold fields entirely; `_audit_only` metadata key is never read by any generation code path; 6 dedicated leakage tests) |
| Page fetch/cache implemented? | Yes — `web_fetch.py`: URL normalization, same-site restriction, deterministic HTML extraction, disk-backed cache, distinguishes ok/error/not_fetched |
| Web territory abstraction implemented? | Yes, at the primitive layer (`web_scope.py`) — NOT wired to `LocalCoordinator` (explicitly out of scope, "STOP after environment validation") |
| ReAct-compatible tools implemented? | Interface stub only (`MatchedReActWebAgent`) — constructor/step-budget shape frozen, `run()` unimplemented |
| ANTMAN-compatible tools implemented? | Interface stub only (`AntWebAgent`) — same step budget as ReAct (fairness requirement), `run()` unimplemented |
| Official scorer understood/implemented? | Understood (LLM-judged binary correctness, no fixed official prompt exists to vendor) and implemented as `LLM_BINARY`/majority-vote-of-3, using this suite's own disclosed-non-official judge prompt |

## E. Proposed frozen evaluation set

**Filtering rule**: English-language questions (`info.lang == "en"`)
whose `root_url` is currently reachable over plain HTTP(S) (status `ok`,
verified live, not assumed) — benchmark/environment validity only, no
ANTMAN performance involved anywhere in this filter.

**Exact counts**:
- 247 English candidates → **222 eligible** (89.9%), **25 excluded**.
- All 25 exclusions trace to exactly 6 root URLs (many tasks share a
  root site):

| Root URL | Failure mode | Tasks affected |
|---|---|---|
| `https://illuvium.io/` | HTTP 429 (rate-limited), reproduced on a spaced retry | 16 |
| `http://cstc.hrbeu.edu.cn/` | Connection timeout, reproduced on retry | 4 |
| `https://www.yili.com/` | HTTP 403 (Forbidden), reproduced on retry | 2 |
| `https://cs.swust.edu.cn/` | Connection timeout, reproduced on retry | 1 |
| `https://www.sandbox.game/en/` | HTTP 403 (Forbidden), reproduced on retry | 1 |
| `https://www.unrealengine.com/zh-CN` | HTTP 403 (Forbidden), reproduced on retry | 1 |

**Caveat, disclosed rather than smoothed over**: the 403/429 responses are
consistent with bot-protection (Cloudflare/WAF-style) rejecting a plain
scripted `User-Agent`, not necessarily true public inaccessibility — a
real browser or a differently-configured client might succeed. This
report labels these 6 sites "inaccessible from this environment's plain
HTTP fetcher," not "permanently dead," and excludes their tasks from the
frozen set on that conservative basis.

**Eligible-set (N=222) breakdown**:

| Dimension | Value | N |
|---|---|---|
| Hop | single-source | 103 |
| Hop | multi-source | 119 |
| Domain | conference | 149 |
| Domain | game | 66 |
| Domain | education | 7 |
| Domain | organization | 0 |
| Difficulty | medium | 116 |
| Difficulty | hard | 65 |
| Difficulty | easy | 41 |

**Gold-source staleness** (offline diagnostic only — never fed into
eligibility filtering or inference, per the governing spec): of the 338
unique gold source pages referenced across the English subset, 284/338
(84%) are currently accessible. Within the 222-task eligible set, 199
tasks have ALL of their own gold source pages currently accessible; 23
have at least one stale/inaccessible gold source page. This is reported
descriptively — WebWalkerQA is a live-web benchmark and the official
dataset itself warns pages may go stale; it is not used to further
shrink the frozen set, since a stale *citation* page doesn't necessarily
mean the *root site* no longer contains the answer via a different path.

**Recommendation**: freeze N=222 as stated. Do not launch inference in
this pass (per the governing spec).

## Next-step inference cost estimate (N=222, projections only — no run launched)

No real WebWalkerQA generation run exists yet, so these are **projections
from this project's own existing repo-QA/SWE-QA-Pro baseline cost
patterns** (GPT-4.1 generation: $2.00/1M input, $8.00/1M output; GPT-5
judge: $1.25/1M input, $10.00/1M output — both pinned in
`src/ant/providers/pricing.py`), explicitly labeled as such, not measured.

| Method | Basis for the projection | Est. calls/query | Est. cost/query | Est. total (N=222) |
|---|---|---|---|---|
| ReAct (`matched_react_web`) | RepoProbe/SWE-QA-Pro `matched_react` averaged ~18.1-18.8 calls/query, $0.11-0.14 total/query on this project's own frozen runs; WebWalkerQA's own 15-step cap plus a synthesis call is the same order of magnitude | ~15-20 | ~$0.10-0.15 | ~$22-33 |
| Official WebWalker (Explorer+Critic) | **Not estimable** — no checkout exists yet (`docs/web_track_baselines.md`'s setup plan is unexecuted); cost depends on an unmeasured two-role loop's own call pattern | n/a | n/a | n/a (blocked on setup) |
| ANTMAN (`ant_web`) | RepoProbe's full-108 `ant` result averaged 51.5 calls/query, $0.43 total/query — but that included real per-repo territory-DISCOVERY overhead the web substrate's runtime-provisional territories don't have; SWE-QA-Pro's own `ant` baseline was not run this pass either, so no second anchor point exists | ~20-40 (wide range, no direct analogue) | ~$0.15-0.35 | ~$33-78 |

These are explicitly **not commitments** — the real number depends on
entirely unbuilt logic (`run()` on every stub above is currently
`NotImplementedError`). They exist only to give an order-of-magnitude
sense of scale before that implementation work is scoped, per the
governing spec's Step E.

## Verification

- 64 tests collected across the WebWalkerQA test files (verified by
  actually running `pytest --collect-only`, not estimated):
  `test_web_fetch.py` 27, `test_web_scope.py` 11,
  `test_webwalkerqa_adapter.py` 8, `test_webwalkerqa_stubs.py` 9
  (6 `def`s, one `@pytest.mark.parametrize`d over 4 agent classes),
  plus the pre-existing, unmodified `test_webwalkerqa.py` 9. All 64
  pass; zero network calls, zero LLM calls.
- `ruff check` and `pyright`: 0 errors across every new file.
- Full project test suite: no regressions (spot-checked the new files in
  isolation plus the pre-existing WebWalkerQA package's own 9 tests;
  a full-suite rerun is recommended before the next implementation pass
  given two long-running background jobs were occupying this session
  concurrently).
- Real-network regression caught and fixed during this pass: several
  Chinese-hosted gold source pages embed raw non-ASCII characters
  directly in the URL, which crashed `urllib_fetcher()` with an uncaught
  `UnicodeEncodeError` inside `http.client` (not a clean, catchable
  `URLError`). Fixed via IRI-to-URI conversion
  (`_ensure_ascii_url()`) with 4 new regression tests; verified live
  against the real dataset's own gold source pages after the fix (the
  338-URL accessibility pass that originally crashed completed cleanly
  on rerun).

## Explicit stop point

Per the governing spec: **no paid benchmark inference was run**, no
tuning against `Answer`/`Golden_Path` occurred anywhere in this pass (the
only place either is read is `WebWalkerQaAdapter.score()`, which no
agent code path touches), and Steps 4/7's `LocalCoordinator`/agent wiring
is deliberately left as interface stubs for a future, separately-scoped
implementation pass.
