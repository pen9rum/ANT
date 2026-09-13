# Chain-of-Agents (CoA) fidelity audit

Written before implementation, same discipline as `docs/longagent_fidelity_audit.md`.

## 1. Paper

Zhang, Sun, Chen, Pfister, Zhang, Arık. "Chain of Agents: Large Language
Models Collaborating on Long-Context Tasks." NeurIPS 2024 (poster).
arXiv:2406.02818. Authors: Penn State University + Google Cloud AI
Research.

## 2. Official/runnable implementation search (no API cost — web search only)

- The paper's own project page (`https://yszh8.github.io/chain-of-agents/`,
  first author's site) links only to the Penn State NLP Lab's general
  GitHub org (`github.com/psunlpgroup`), with no paper-specific repo
  mentioned or linked.
- Searching `psunlpgroup`'s own repositories for "chain" returns zero
  matches.
- No Google-affiliated org (`google-research`, `google-deepmind`, etc.)
  repo for this paper was found.
- Three third-party, non-author-linked community repos exist
  (`rudrankriyam/Chain-of-Agents`, `richardwhiteii/chain-of-agents`,
  `AdamCodd/Chain-of-agents`) — none claim author affiliation, none were
  used as a reference here.

**Conclusion: no official or unofficial-but-author-linked runnable
implementation exists.** This is a **paper-faithful adaptation**,
reconstructed directly from the paper's own Section 3 (method) and its
worker/manager prompt descriptions, exactly the same disclosure posture
as this repo's existing LongAgent adapter.

## 3. Algorithm fidelity (Section 3 of the paper)

Implemented exactly as specified by the task brief, itself drawn from the
paper's own query-based (QA) variant:

- Chunks processed **sequentially**, in original source order, one
  worker per chunk.
- Each worker `Wi` receives only `(ci, CU_{i-1}, q)` — never future
  chunks, never other workers' chunks, never the full document.
- `CU_0 = ""` (empty) for the first worker.
- The manager receives **only** `CU_l` (the last communication unit) and
  `q` — never the individual per-chunk outputs, never the source text
  directly.
- No retrieval, no relevance-based chunk skipping, no early stopping, no
  question-aware pruning before the worker call — every chunk is always
  processed. This is the defining contrast with ANT/Retrieval in this
  suite, and is deliberately preserved even though it makes CoA the most
  call-expensive baseline at long context lengths.

## 4. Disclosed engineering decisions (not paper-specified, not tuned on scores)

- **Sentence-aware chunking** (paper: "greedily pack sentences under the
  agent's context window budget", Section 3.1) implemented via a
  lightweight, regex-based sentence splitter (`split on [.!?] + whitespace
  + capital, or newline`) — not a true NLP sentence tokenizer (no such
  dependency exists in this repo; adding one was judged out of scope for
  a smoke-test-only baseline). Disclosed limitation, same "heuristic, not
  true NLP" posture as `niah_plus.py`'s own proper-noun regex.
- **`agent_window_tokens = 8192`**, the paper's own reported 8K-agent
  setting, used unchanged as the smoke-test configuration per instruction
  — never tuned from results.
- **Worker output cap (`MAX_CU_OUTPUT_TOKENS`)**: the paper does not state
  an exact CU length limit; a fixed 512-token cap is applied so the
  chunk-budget arithmetic (`budget = k - tokens(q) - tokens(instruction) -
  MAX_CU_OUTPUT_TOKENS - safety_margin`) has a computable, worst-case-safe
  right-hand side. Disclosed engineering choice, not from the paper, not
  tuned from results.
- **Safety margin (200 tokens)**: reserved for prompt-formatting/wrapper
  overhead beyond raw text token counts — the same kind of disclosed
  engineering margin `direct_document.py`'s own `MAX_CONTEXT_TOKENS`
  already uses below the true context window, not a paper value.
- Model: GPT-4.1 for both worker and manager (same backbone as every
  other method in this suite) — a disclosed substitution for the paper's
  own multi-backbone experiments (which include GPT-3.5/4, Gemini,
  Claude), never claimed as a specific paper configuration reproduction.

## 5. What this adapter does NOT do

Per the task brief and to keep this a recognizable CoA baseline rather
than gradually acquiring ANT-specific mechanisms: no Need Graph, no
routing, no worker specialization, no retrieval, no explicit
critique/recovery step, no answer-confidence heuristics. The final-answer
extraction step (`condense_to_answer_span`) is the SAME frozen,
method-agnostic post-hoc layer every other method in this suite already
uses — not a CoA-specific scorer.
