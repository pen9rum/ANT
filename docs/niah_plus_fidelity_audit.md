# Needle-in-a-Haystack PLUS: fidelity audit

Written BEFORE any implementation, per Part B Section 5 of the follow-up
long-context spec. All facts below were verified live against the actual
paper and repository on 2026-09-12 (not recalled from prior familiarity
with the earlier LongAgent adaptation work, though that work's own
`docs/longagent_fidelity_audit.md` Section 2 already established the
headline finding this audit confirms again independently: no runnable
benchmark-generation code exists anywhere).

## 1. Exact paper version

Zhao, Zu, Xu, Lu, He, Ding, Gui, Zhang, Huang. "LongAgent: Scaling
Language Models to 128k Context through Multi-Agent Collaboration."
arXiv:2402.11550 (also published as the EMNLP 2024 paper this evaluation
suite already cites for the LongAgent adaptation, same work). Fetched live
from `https://arxiv.org/html/2402.11550v2` (the HTML-rendered v2, the
current version as of this audit) -- **NIAH+'s benchmark description
appears in that paper's own Section 3.1 ("Evaluation Protocol")**, not in
a separate NIAH+-specific paper.

## 2. Exact benchmark/resource URL and available files

`https://github.com/zuucan/NeedleInAHaystack-PLUS` (the same author-linked
repository already identified in the LongAgent fidelity audit). Verified
live via the GitHub API (`git/trees?recursive=1`), not just browsing:

| File | Size | Contents |
|---|---|---|
| `README.md` | 2,761 bytes | Description, data-format schema, results-visualization figures, acknowledgement, citation |
| `singleQA.jpg` | 2,089,246 bytes | A results figure (accuracy heatmap image), not data |
| `multiQA.jpg` | 2,200,219 bytes | A results figure (accuracy heatmap image), not data |

Repository metadata: default branch `main`, 3 commits total, all made
within one hour on 2024-03-04 (`ac26932`, `b8b7664`, `128aa29`, all
"Update README.md"), never updated since. **No generation code, no
dataset files, no notebooks, no scripts of any kind exist in this
repository.**

## 3. Whether benchmark-generation code exists

**No.** Confirmed by direct enumeration of the full repository tree
(3 files total, none executable/data). No fork, branch, or release
contains anything else. This is the same "zero lines of code" finding as
the LongAgent inference-code search in `docs/longagent_fidelity_audit.md`
Section 2, independently reconfirmed here for the benchmark-construction
side specifically.

## 4. Whether the data itself exists publicly

**Only as an opaque, non-code-accompanied download**, not as inspectable
data: the README points to one Google Drive file
(`https://drive.google.com/file/d/1aov5kwy4DRYWgxu4Ulaf3omx3uNd3M2r/view`).
This link was checked for reachability (`curl -I`, live, 2026-09-12) and
returns HTTP 200 -- the file exists and the page loads -- but:

- It is a single opaque archive with **no accompanying generation script,
  no README inside it (unknown without downloading), and no way to verify
  its contents were produced by the exact procedure the paper describes**
  rather than some undocumented variant.
- Google Drive links of this kind commonly require an interactive
  virus-scan/size-confirmation step for programmatic (non-browser)
  retrieval, which was not attempted further.

**Decision, disclosed rather than silently made**: this pass does NOT
download and use that opaque file. Blindly trusting an unverifiable
external blob with no accompanying code would itself be "silently
inventing" trust in an unaudited artifact -- the same failure mode the
governing spec's "do not silently invent missing protocol details"
warns against, just inverted (trusting undocumented data instead of
inventing an undocumented procedure). Instead, Section 6 below builds a
**faithful reconstruction** from the paper's own disclosed construction
protocol and disclosed source datasets (SQuAD, HotpotQA), clearly labeled
as a reconstruction, never claimed to be the literal released benchmark.

## 5. Construction protocol, as disclosed in the paper text (exact quotes)

Fetched from `https://arxiv.org/html/2402.11550v2`, Section 3.1, cross-
verified across two independent fetches of the same section (identical
quotes both times):

### Context lengths (both task types)
> "The length of haystack ranges from 1,000 to 128,000 tokens with equal
> intervals, totaling 15 different lengths."

The paper does **not** enumerate the 15 exact numeric values anywhere in
the quoted text. The phrase "with equal intervals" is consistent with
`numpy.linspace(1000, 128000, 15)` rounded to integers -- the well-known
default schedule from the original NeedleInAHaystack generator
(gkamradt/LLMTest_NeedleInAHaystack) this benchmark explicitly credits in
its Acknowledgement section -- but this is this audit's own inference
from the wording and the explicit lineage credit, not a verbatim paper
statement of the 15 numbers. **Flagged as an ambiguity, not resolved by
assumption**: Section 6 below does not rely on reproducing all 15 values,
only two round smoke points (32K, 128K) the calling spec itself
specifies, with 128K coinciding exactly with the paper's own stated upper
bound and 32K treated explicitly as a round intermediate smoke value, not
claimed as one of the paper's own literal 15 grid points.

### Single-document QA depths
> "The depth of the needle ranges from 0% to 100% with equal intervals,
> totaling 10 different depths."

10 equal intervals over [0%, 100%] is `linspace(0, 100, 10)` =
{0, 11.1, 22.2, 33.3, 44.4, 55.6, 66.7, 77.8, 88.9, 100}. No exact 50%
grid point exists in this scheme.

### Multi-document QA depths
> "The two needles are randomly scattered at the depth of 0%, 33%, 66%,
> and 100% of the haystack, resulting in 6 combinations."

Exactly 4 depth labels, 2-needle unordered pairs from 4 values = C(4,2) +
4 same-value pairs is not 6 either; the paper's own arithmetic (6
combinations from 4 depths) is consistent with **unordered pairs of
distinct depths**: C(4,2) = 6 exactly. This audit takes the paper's own 4
literal values (0%, 33%, 66%, 100%) as authoritative and uses 3 of the 6
possible pairs for EARLY/MIDDLE/LATE (see Section 6).

### Source datasets and construction method
> Single-document: "SQuAD is a large-scale machine reading comprehension
> dataset... We randomly select 100 samples from the training set of
> SQuAD... documents from other samples in the training set are randomly
> selected as haystacks." Entities "directly related to the answer within
> the documents" are replaced with "fictional entities."
>
> Multi-document: "We construct test samples based on HotpotQA... we
> select 60 questions from the validation set." Answers require
> "information from two or more documents to reason the final answer."

This confirms: (a) single-needle haystack filler is **other SQuAD
training-set passages**, not Paul Graham essays (the original NIAH's own
filler) or any other corpus; (b) the needle passage itself has its answer
entity replaced with a fictional one, specifically to defeat world-
knowledge shortcuts rather than genuine long-context retrieval; (c)
multi-needle needles are HotpotQA's own two supporting documents for a
selected validation question. **The paper does not specify what fills
the REST of the multi-document haystack** (beyond the 2 needle documents)
to reach 32K-128K tokens -- HotpotQA's own distractor documents per
question are far too few and short (10 documents, a few hundred tokens
total) to reach even the smallest 1K length, let alone 128K. This is a
second explicit ambiguity, resolved in Section 6 by using other HotpotQA
validation documents (from different, unrelated questions) as filler,
disclosed as this audit's own reconstruction decision, not a paper-stated
mechanism.

### Official metric
> Referenced only as "ACC" (accuracy) in the results figures/table
> captions. The paper's Appendix does not formally specify the exact
> string-matching procedure (exact match on the fictional/gold entity?
> substring containment? something else?) anywhere in the fetched text.

**This is the single most consequential ambiguity for scoring.** Per the
governing spec's Section 16 ("Use the benchmark's official deterministic
scoring semantics if available... do not use GPT-5 judging unless the
benchmark itself actually requires one"), and given ACC's exact
computation is unspecified, this pass uses this evaluation suite's own
already-audited, byte-verified official EM/F1 scorer
(`ant.evaluation_suite.qa_metrics.score_qa`, the same one used for
HotpotQA/2Wiki/MuSiQue) as the closest defensible deterministic
substitute for an unspecified "ACC" -- reported as EM/F1, not relabeled
as "ACC", so the substitution is visible rather than silently presented
as the paper's own metric.

### Sample counts and seeds
> "For each length and depth, we randomly select 10 needles to construct
> 10 test samples" (single-document); "randomly select 10 needles to
> construct 10 test samples for evaluation" (multi-document, worded
> identically).

No random seed value is given anywhere in the fetched text. This pass
uses a fixed, disclosed seed (see Section 6) for full reproducibility of
its own reconstruction -- not a recovered paper seed, since none exists
to recover.

## 6. This pass's faithful reconstruction (explicitly separated from the released benchmark)

Implemented in `src/ant/evaluation_suite/niah_plus.py`. Every decision
below is a reconstruction choice, disclosed, not a verified detail of the
original released benchmark:

- **Single-needle source**: `rajpurkar/squad`, `train` split (matching
  the paper's own "training set of SQuAD" for both needle AND haystack
  filler, since the paper explicitly uses train-set documents for both
  roles). Needle = one example's `context` field with its
  `answers.text[0]` span replaced everywhere it occurs (case-sensitive
  exact string match) by a deterministic fictional entity name; the gold
  answer becomes that fictional entity. Haystack filler = other, unrelated
  SQuAD train examples' own `context` fields, concatenated until the
  target token count is reached.
- **Multi-needle source**: this evaluation suite's own already-audited
  `hotpotqa` benchmark adapter (`hotpotqa/hotpot_qa`, `distractor` config,
  `validation` split -- the exact same source `docs/long_context_dataset_
  audit.md` already establishes). Needles = the question's own 2
  supporting documents (by `supporting_doc_ids`, construction-only
  metadata never exposed to any inference method, same discipline as
  `document_scope.py`'s existing Lost-in-the-Middle utility). Haystack
  filler = OTHER, unrelated HotpotQA validation questions' own documents
  (never the same question's own distractors alone, since those are far
  too few/short to reach 32K-128K) -- this is this audit's own filler-
  source decision, not a paper-specified mechanism (see Section 5).
- **Depth control**: reuses the existing `DocumentRecord`/document-list
  abstraction (`ant.evaluation_suite.document_scope`), generalizing
  `build_positional_variants`'s early/middle/late splicing to an
  arbitrary depth PERCENTAGE measured in cumulative TOKEN position (via
  `tiktoken`, `cl100k_base`, the same encoding `chunk_documents` already
  uses) across the filler+needle document list, not merely list-index
  position -- so "depth 33%" means "roughly 33% of the way through the
  haystack by token count," matching the paper's own token-percentage
  framing rather than a coarser document-count approximation.
- **EARLY/MIDDLE/LATE mapping** (Section 6 of the governing spec: "map to
  fixed documented paper-consistent depths," chosen before any generation
  or inference, independent of scores):
  - Single-needle: EARLY=0%, MIDDLE=50%, LATE=100%. 0%/100% are literal
    paper grid points; 50% is a natural midpoint label even though the
    paper's own 10-point equal-interval grid has no exact 50% value
    (nearest grid points are 44.4%/55.6%) -- disclosed, not hidden.
  - Multi-needle (2 needles, paper's own literal 4-value grid {0,33,66,100}):
    EARLY = needles at (0%, 33%), MIDDLE = needles at (33%, 66%), LATE =
    needles at (66%, 100%) -- three of the paper's own six valid
    unordered pairs, chosen to span early/middle/late while using ONLY
    the paper's own literal depth values, never an interpolated one.
- **Context lengths**: exactly 32,000 and 128,000 tokens (the calling
  spec's own two smoke points), measured as the total token count of the
  fully assembled haystack+needle(s) text via `tiktoken` `cl100k_base` --
  "within documented tolerance" (Section 19's test requirement) is
  defined as within 1% of the target, since exact equality is not always
  achievable with whole filler documents.
- **Scoring**: `ant.evaluation_suite.qa_metrics.score_qa` (EM/F1), same as
  every other benchmark in this suite -- reported honestly as EM/F1, not
  relabeled "ACC" (see Section 5).
- **Seed**: a fixed constant (`NIAH_PLUS_SEED = 20260912`, this audit's
  own write date, chosen arbitrarily but fixed and disclosed) drives all
  `random`-module sampling (which SQuAD haystack passages, which HotpotQA
  filler questions) -- every instance this generator produces is
  reproducible from (task_type, context_length, position, question_index)
  alone.
- **Sample count**: this pass generates exactly ONE deterministic instance
  per (task type x context length x position) cell -- 2x2x3 = 12
  conditions total, per Section 14 of the governing spec, not the paper's
  own 10-samples-per-cell design (explicitly out of scope: "do not
  increase sample count yet").

## 7. Summary: exact released behavior vs. this pass's reconstruction

| Aspect | Exact released NIAH+ behavior | This pass |
|---|---|---|
| Generation code | None published | Newly written, disclosed as reconstruction |
| Raw data | One opaque Google Drive archive, unverified | Not used -- built fresh from SQuAD/HotpotQA |
| Context lengths | 15 values, exact numbers not quoted in paper | 2 values (32K, 128K) per this task's own scope |
| Single-needle depths | 10 exact values (paper-specified grid) | 3 of those 10 grid points (0%, ~50% label, 100%) |
| Multi-needle depths | 4 exact values, 6 pairs (paper-specified) | 3 of the 6 valid pairs |
| Metric | "ACC", exact computation unspecified | EM/F1 (this suite's own official scorer), disclosed substitution |
| Samples/cell | 10 | 1 (explicit smoke-only scope) |
| Multi-needle filler source | Unspecified in paper | Other HotpotQA validation documents (this pass's own decision) |

No claim in this pass's final report may describe results against this
reconstruction as "NIAH+ results" without this qualification.
