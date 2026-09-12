# LongAgent fidelity audit (frozen, written before implementation)

Status: **pre-implementation fidelity audit**, per explicit instruction that
Phase 1 (this document) must exist and be read before any adapter code is
written. If the eventual `src/ant/external_wrappers/longagent.py` ever
disagrees with what this document records as the intended protocol, that is
a bug in the code, not an ambiguity to interpret away -- update this
document deliberately, don't let it silently drift.

## 1. Exact paper identity

- **Title (EMNLP camera-ready)**: "LONGAGENT: Achieving Question Answering
  for 128k-Token-Long Documents through Multi-Agent Collaboration"
- **Venue**: Proceedings of EMNLP 2024 (Main), pages 16310-16324, Miami,
  Florida, USA.
- **ACL Anthology**: https://aclanthology.org/2024.emnlp-main.912/
- **Same work, earlier arXiv preprint**: arXiv:2402.11550, "LongAgent:
  Scaling Language Models to 128k Context through Multi-Agent
  Collaboration", submitted 2024-02-18, revised 2024-03-13 (v2). Same
  author list, same method, slightly different title -- confirmed by
  cross-reading both abstracts (fetched live, not from memory): identical
  divide-and-conquer / leader-member / inter-member-communication
  description in both.
- **Authors**: Jun Zhao, Can Zu, Xu Hao (Hao Xu), Yi Lu, Wei He, Yiwen
  Ding, Tao Gui, Qi Zhang, Xuanjing Huang (Fudan University NLP group).
- This audit cites the arXiv v2 HTML (https://arxiv.org/html/2402.11550v2)
  as the primary source text, since it is directly fetchable and openly
  licensed, cross-checked against the ACL Anthology abstract page for the
  EMNLP camera-ready framing. Both were fetched live during this audit
  (2026-09-11), not reconstructed from prior training-data familiarity --
  per the standing instruction that paper/code wins over any prior
  description.

## 2. Code availability -- **no usable implementation exists, official or unofficial**

This is the single most important finding of this audit and it changes
what "faithful" can mean here.

- **No GitHub link is given anywhere in the paper, the ACL Anthology page,
  the arXiv abstract, the Hugging Face paper page, or the author's own
  homepage** (https://zhaojun-nlpr.github.io/, checked live -- no mention
  of LongAgent at all).
- The only author-linked repository found anywhere is
  **https://github.com/zuucan/NeedleInAHaystack-PLUS**, which is the
  paper's own evaluation *benchmark* (the single/multi-document QA test
  set used to report results), not the LongAgent inference code. Its full
  contents were enumerated: `README.md`, two `.jpg` figures, and a pointer
  to Google-Drive-hosted test data. **Zero lines of leader/member/chunk/
  conflict-resolution code exist in this repository.**
- A third-party repository, **https://github.com/vyomakesh09/longagent**,
  surfaces in search results under the paper's title. It was inspected via
  the GitHub API tree listing, not just its README: it is a
  `kyegomez/Python-Package-Template` scaffold (CI config, docs scaffolding,
  `pyproject.toml`, etc.) with **every single Python file at 0 bytes**
  (`example.py`, `package/__init__.py`, `package/main.py`,
  `package/subfolder/__init__.py`, `package/subfolder/main.py` -- all
  confirmed 0-byte blobs via `git/trees?recursive=1`). It is not an
  independent reimplementation either; it is an empty template with the
  paper's name attached. It contributes nothing and was not used for
  anything beyond this negative finding.
- No OpenReview supplementary code link was found (the OpenReview page for
  this submission, https://openreview.net/forum?id=SqDrBaZVqn, did not
  expose reviewable content through the fetch used here).

**Conclusion**: there is no runnable reference implementation to port,
official or otherwise. Per the standing instruction for exactly this
situation, the adapter below is reconstructed directly from the paper's
own methodology section and appendix prompt tables, and must always be
labeled:

> **"paper-faithful LongAgent adaptation"** -- never "exact reproduction",
> since no reference implementation exists to reproduce exactly against.

## 3. Protocol, as extracted from the paper text

Everything below is quoted or closely paraphrased from Section 2
("Method") and Appendix B of arXiv:2402.11550v2, fetched directly rather
than assumed.

### 3.1 Leader action space

The leader samples one action per round from a fixed three-way space
(Eq. 1, Section 2.3): **`a ~ Leader(a | S, q)`, a ∈ {NEW_STATE, CONFLICT,
ANSWER}**.

- **NEW_STATE**: "the information contained in the preceding rounds is
  insufficient" -- the leader generates a new instruction and broadcasts
  another round.
- **CONFLICT**: the leader "perceives conflicting answers from members" --
  triggers the inter-member communication / peer-review sub-protocol
  (3.4 below).
- **ANSWER**: "the leader deems the currently collected information
  sufficient" -- terminates the loop and emits the final answer.

The user's brief in this task uses the word **"QUERY"** for what the paper
itself calls **NEW_STATE** (there is no action literally named "QUERY" in
the paper). This audit uses the paper's own terminology (`NEW_STATE`)
throughout the implementation and its accounting fields, and records the
correspondence here explicitly rather than silently renaming it or
silently inventing a fourth action.

### 3.2 Chunking

- "we partition the long text x into n chunks {c1,c2,...,cn} of
  predefined size and distribute them accordingly to n members
  {m1,m2,...,mn}" (Section 2.1) -- **one chunk per member, strictly
  one-to-one**, no variation, no member handling more than one chunk.
- The paper's own chunk-size ablation (Section 4.2 / Figure 5) sweeps
  **{500, 1000, 1500, 2000} tokens** and reports: "when the chunk size
  increases from 500 to 2,000, the hallucination problem is alleviated."
  **No single default chunk size is stated for the paper's main reported
  results** -- this is a real gap in the released text, not an oversight
  in this audit. Decision (Section 4 below): use **2000 tokens**, the
  largest value the paper's own ablation tested and the one it reports as
  best for hallucination mitigation, disclosed as a paper-grounded choice
  rather than a documented main-experiment default.
- No chunk overlap is described anywhere in the text.

### 3.3 Round structure / member communication

- Round 1 (and every subsequent NEW_STATE round, per the trajectory
  examples in Appendix B): the leader **broadcasts the same instruction to
  all n members simultaneously** ("the leader systematically breaks q down
  into multiple sub-questions and organizes members to collaborate in
  searching for clues from their respective chunks", Section 2.1). No
  mechanism for addressing a strict subset of members in an ordinary
  NEW_STATE round is described.
- Each member's prompt (Appendix B, Table 4 template) gives it **only its
  own assigned chunk plus the leader's current instruction** -- never the
  full document, never other members' chunks, never other members'
  responses (except during CONFLICT resolution, 3.4 below). Members
  respond in a fixed JSON-ish schema wrapping a `<scratchpad>`-style
  free-text answer.
- **The leader itself never sees raw document text.** It only ever sees
  the aggregated member responses for the current round ("the leader
  needs to complete the task based on the query results they return",
  Section 2.1, step 4). This is a structural property, not an
  implementation detail this adapter can legitimately deviate from without
  ceasing to be LongAgent: giving the leader direct chunk access would be
  a different, better-informed coordinator than the paper describes.

### 3.4 Conflict resolution (inter-member communication)

Section 2.4, Eqs. 2-4: when the leader emits `CONFLICT`, it "first
identifies the member IDs where answers conflict", then **those specific
members share their chunks pairwise and re-answer with the merged
document**: if member i hallucinated from chunk `ci` and member j
answered correctly from chunk `cj`, member j (or the corrected member) is
re-queried with `cj ⊕ cj'` (concatenation of the two chunks). "The
majority of members experiencing hallucination tend to correct their
original responses upon receiving the chunk containing the correct
answers." This is the paper's entire "inter-member communication
mechanism" -- it is pairwise chunk-sharing triggered by the leader
detecting a conflict, not a general any-to-any member chat.

### 3.5 Termination

- **ANSWER** is the only clean termination path; output schema (Appendix
  B, Table 5 style): `{"type": "answer", "content": "..."}`.
- **No explicit maximum round count is given anywhere in the paper.**
  Appendix B's worked examples show 2-4 rounds completing in practice for
  both single-hop and multi-hop settings, but this is descriptive, not a
  documented cap.

### 3.6 Model dependencies

- **Primary reported system**: LLaMA-2-7B, base 4k context, **fine-tuned**
  via supervised fine-tuning on synthetic leader/member interaction
  trajectories -- "we utilize GPT-4 to generate 1,000 interaction
  trajectories for each task to train the Leader" (Section 3.3), and
  training documents were "extended... to 2500-3000 tokens through
  concatenation."
- **Training-free, prompt-only variant, reported as a MAIN result (not an
  ablation)**: "For LongAgent supported by more powerful models like
  GPT-3.5, fine-tuning is not necessary. Through prompting, GPT-3.5 can
  simultaneously act as a leader and members with specific skills."
  (Section 4.1, "Overall Performance"). This variant is reported to beat
  GPT-4 by 6.78%/1.5% (single/multi-doc) despite GPT-3.5's smaller (16k)
  native context window. **This is the load-bearing precedent for Phase
  2's model-substitution decision** -- see Section 5 below.
- No member cap of any kind is stated ("no explicit numeric cap is
  specified anywhere in the paper" -- confirmed by direct query against
  the paper text, not inferred).

## 4. What had to be reconstructed / adapted (repository QA is not the paper's own task)

The paper's own tasks are single/multi-document QA over prose documents
(Needle-in-a-Haystack-PLUS-style). Adapting this to *repository* QA
(RepoProbe-Python, SWE-QA-Pro) requires decisions the paper does not make,
each recorded here rather than silently absorbed into the code:

1. **What "the long document" is.** The paper assumes one continuous
   document is handed in already assembled. For a repository, this adapter
   deterministically serializes the SAME eligible-file universe the rest
   of this evaluation suite already uses (`EvalRepoEnvironment.iter_files()`
   -- the identical file set Direct/Retrieval/Matched ReAct/MR+RepoGraph
   see, not a bespoke walk), in **stable sorted-relative-path order**
   (simple, deterministic, domain-neutral -- the paper gives no
   repository-specific ordering to prefer over this), with an explicit
   `===== FILE: <relative/path> =====` boundary marker before each file's
   verbatim text. No file is summarized, filtered by relevance to the
   question, or reordered by any heuristic -- see Section 6 (explicitly
   excluded techniques) below.
2. **Chunk size**: 2000 tokens (the paper's own best-tested ablation
   value; see 3.2). Chunking is a straightforward token-count sliding
   window over the serialized document text (via `tiktoken`, added as a
   new, disclosed, narrowly-scoped dependency purely for token counting --
   it changes nothing about any existing frozen baseline), splitting
   exactly at the target token count with no overlap (the paper describes
   no overlap) and no chunk-boundary awareness beyond "at the token
   count" (the paper does not describe sentence/file-boundary-aware
   chunking either; adding it would be inventing repository-specific
   intelligence LongAgent does not have, which is explicitly disallowed).
3. **Round cap**: the paper states no maximum round count. Running
   genuinely unbounded is not viable for an automated evaluation harness
   (a leader that never emits ANSWER would hang a task forever and burn
   unbounded API cost) -- so this adapter imposes an explicit, disclosed
   **engineering safety bound**, not a paper-derived value: 8 NEW_STATE
   rounds (chosen generously above the paper's own observed 2-4-round
   examples), after which the run force-terminates with
   `termination_reason="round_budget_exhausted"` and the leader is asked
   once more, directly, to answer with whatever it has. This mirrors how
   this same evaluation suite already handles other open-ended baselines
   reaching their own budget ceiling (Matched ReAct's `budget_exhausted`,
   the official SWE-QA-Pro agent's "Forced after max iterations"), not a
   LongAgent-specific invention.
4. **Member cap**: the paper states no cap. Per explicit instruction, none
   is invented here either -- the adapter instantiates exactly
   `ceil(len(serialized_document_tokens) / chunk_size)` members, however
   large that is for a given repository, and the actual resulting count is
   measured and reported in Section 7 (Phase 5 smoke results) rather than
   silently capped.
5. **Model**: GPT-4.1 for both leader and members (Phase 2 decision, see
   Section 5).

## 5. Model substitution: GPT-4.1 in place of GPT-3.5/fine-tuned LLaMA-2-7B

The paper itself already reports a **training-free, prompt-only GPT-3.5
leader+member configuration as a main result**, not merely a hypothetical
this audit is inventing (Section 3.6 above). Using GPT-4.1 -- a strictly
more capable, more recent, still-training-free/prompt-only model, in the
exact same "prompt GPT-N to act as leader and members" role the paper
itself already validated -- is a same-paradigm capability upgrade, not a
new methodological regime the paper never tested. This is recorded
explicitly as a **model substitution**: results under GPT-4.1 characterize
"LongAgent's coordination architecture, run on this evaluation suite's
standardized inference model", not "a reproduction of the paper's own
GPT-3.5 numbers" and never "a reproduction of the paper's fine-tuned
LLaMA-2-7B numbers" -- those are a different model entirely (fine-tuned,
not prompted) and no claim of matching them is made anywhere in this
adapter or its outputs.

## 6. Explicitly excluded (would silently turn this into ANT, or into a smarter system than LongAgent)

Per direct instruction, the adapter must not use, anywhere:

- ANT WorkerCards, territories, Need Graph, runtime graph revision,
  routing, progress controller, stuck recovery.
- RepoGraph, AST-based smart partitioning, or any other semantic
  repository decomposition.
- Oracle/gold-answer information of any kind.
- Question-specific hand-written partitioning (chunking must not look at
  the question before splitting the document).
- Any relevance-based file filtering before member assignment (this would
  be a retrieval step the paper's own protocol does not have -- see
  Critical Question 4 in the final report).

The implementation is checked against this list explicitly before the
smoke test (Critical Question 9 in the final report).

## 7. Open fidelity caveats (carried into every result this baseline produces)

1. No default main-experiment chunk size is stated in the paper; 2000
   tokens is a defensible, paper-grounded choice, not a documented one.
2. The paper gives no repository-domain guidance at all -- every decision
   in Section 4 is this adapter's own necessary extension of a
   prose-document method to a code-repository setting, disclosed rather
   than presented as prescribed by the paper.
3. GPT-4.1 is a model substitution (Section 5) -- defensible under the
   paper's own training-free-variant precedent, but never to be reported
   as reproducing either of the paper's own two headline configurations
   (fine-tuned LLaMA-2-7B, or GPT-3.5).
4. The round cap (8) and the absence of a member cap are both disclosed
   engineering decisions under real-world constraints the paper's own
   text does not have to contend with (an automated multi-benchmark
   evaluation harness with a real budget), not paper findings.
5. No reference implementation exists anywhere to diff this adapter's
   behavior against line-by-line. Fidelity here means "matches the
   paper's stated protocol as closely as a repository-QA setting permits",
   not "byte-identical to a known-good implementation" -- because no such
   implementation is available to compare against.
