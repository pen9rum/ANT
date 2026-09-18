"""RepoDistill component 2/3: CABA -- Compression-Aware Budget Allocation
(paper section 2.2).

Paper: Yin, Ding, Zhang, Wang, Wang, Ni, Cui. "RepoDistill: Distilling
Repository Knowledge through Compression-Aware Budget Allocation and
Policy Optimization." Findings of ACL 2026.
https://aclanthology.org/2026.findings-acl.217/

Reimplemented from the published description (see
`repodistill_graph.py`'s header for why no code port was possible).

Section 2.2, as implemented. "Given a function f, query q, and token
budget t (where t is computed as the function's token length multiplied
by the budget rate assigned by CAPO), RepoDistill executes a two-stage
compression pipeline":

  Stage 1 -- Perplexity-Guided Block Segmentation. "we compute each
  line's perplexity and designate a line as the start of a block if its
  perplexity exceeds neighboring lines by at alpha (set to 0.2) times the
  standard deviation."

  Stage 2 -- MMR-Guided Block Selection (Equations 3 and 4):

      MMR(b) = lambda * AMI(b, q) - (1 - lambda) * max_{b_j in S} Sim(b, b_j)
      AMI(b, q) = PPL(q) - PPL(q | b)

  with lambda = 0.1. "We adopt a greedy strategy: in each iteration, we
  select the block b* that yields the highest MMR score and fits within
  the remaining token budget. This process continues until the budget is
  exhausted or no further blocks can be added." And: "Following
  LongCodeZip, we use Qwen2.5-Coder-0.5B to compute PPL and Sim(b, b_j)."

ALL OF THIS IS LOCAL AND FREE. Qwen2.5-Coder-0.5B runs on CPU via
transformers/torch (both already project dependencies). Not one token of
CABA touches a paid API -- see `repodistill.py`'s cost accounting.

================================================================
DEVIATIONS FROM THE PAPER'S EXACT SPEC (official -> adapted -> reason)
================================================================

1. PERPLEXITY / SIMILARITY MODEL: NONE. Qwen2.5-Coder-0.5B is used
   exactly as the paper specifies. It was confirmed downloadable and
   runnable on CPU in this environment before this module was written
   (~0.6s for a short forward pass, float32, torch 2.x CPU build). This
   is the one auxiliary model where paper fidelity was achievable without
   new engineering, so no substitution was made.

2. "EXCEEDS NEIGHBORING LINES BY AT alpha TIMES THE STANDARD DEVIATION".
   Official setting: that sentence, verbatim; neither "neighboring" nor
   which standard deviation is defined further.
   Adapted setting: line i starts a block when
   `ppl[i] > mean(ppl[i-1], ppl[i+1]) + alpha * stdev(ppl)`, where the
   neighbours are the immediately adjacent available lines and stdev is
   taken over the whole function's per-line perplexities.
   Reason: this is the most direct literal reading of the sentence, and
   it matches the stated intuition ("abrupt increases in perplexity
   indicate transitions to new semantic units") -- the comparison has to
   be local (against neighbours) while the scale has to be global (the
   function's own variability), or alpha would not be dimensionless.

3. Sim(b, b_j) REPRESENTATION.
   Official setting: "we use Qwen2.5-Coder-0.5B to compute PPL and
   Sim(b, b_j)" -- the paper does not say how a causal LM is turned into
   a block representation.
   Adapted setting: cosine similarity between mean-pooled final hidden
   states of each block, from the same Qwen2.5-Coder-0.5B forward pass.
   Reason: mean-pooled last-hidden-state is the standard way to get a
   sentence/segment vector out of a decoder-only LM without a separate
   embedding head, and it keeps the paper's constraint that this model
   (not a second one) supplies Sim.

4. PER-LINE PERPLEXITY VIA ONE FORWARD PASS.
   Official setting: "we compute each line's perplexity" -- no procedure
   given.
   Adapted setting: one forward pass over the whole function, then
   per-token negative log-likelihoods are bucketed to their source line
   via the tokenizer's offset mapping, and each line's perplexity is
   exp(mean NLL over its own tokens).
   Reason: this is both the cheapest (1 pass, not 1-per-line) and the
   more faithful reading -- a line scored in isolation would have no
   preceding context, and the whole premise of the segmentation signal is
   that a line is surprising GIVEN what came before it.

5. DEGENERATE "NOTHING FITS THE BUDGET" CASE -- stated in full at its own
   call site, in `compress_unit` below.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np

# Paper section 2.2, stated explicitly: "a line as the start of a block if
# its perplexity exceeds neighboring lines by at alpha (set to 0.2) times
# the standard deviation."
SEGMENTATION_ALPHA = 0.2

# Paper Equation 3: "lambda (set to 0.1) is a trade-off parameter that
# controls the balance between relevance and diversity."
MMR_LAMBDA = 0.1

# Paper section 2.2: "Following LongCodeZip, we use Qwen2.5-Coder-0.5B to
# compute PPL and Sim(b, b_j)."
PERPLEXITY_MODEL = os.getenv("ANT_REPODISTILL_PPL_MODEL", "Qwen/Qwen2.5-Coder-0.5B")

# Guard against a pathological single "function" (e.g. a 5000-line
# module-level class body) pushing one forward pass past the model's own
# context window. Not a paper parameter -- a runtime safety bound. Blocks
# beyond the cap are still segmented, just on truncated perplexity signal.
MAX_PPL_TOKENS = 4096


@dataclass(frozen=True)
class Block:
    """One "semantically coherent block" produced by stage 1."""

    line_start: int  # 1-based, relative to the unit's own snippet
    line_end: int
    text: str
    n_tokens: int


class PerplexityScorer(Protocol):
    """The Qwen2.5-Coder-0.5B-shaped interface stages 1 and 2 need.

    Declared as a Protocol so tests can inject a deterministic stub
    instead of loading a real 0.5B model for every unit test -- the same
    mock-the-expensive-boundary convention this suite already uses for
    the LLM provider. Note the stub is a TEST convenience only: a real
    benchmark run uses `QwenPerplexityScorer` below, i.e. the paper's own
    model.
    """

    def count_tokens(self, text: str) -> int: ...

    def line_perplexities(self, text: str) -> list[float]: ...

    def perplexity(self, text: str, prefix: str = "") -> float: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...


class QwenPerplexityScorer:
    """Paper-faithful PPL/Sim backend: Qwen2.5-Coder-0.5B via transformers.

    Loaded lazily on first use and cached per (model_name) at module
    level, because a repo-QA benchmark asks many questions against the
    same process and re-loading a 0.5B model per question would dominate
    wall-clock. float32 on CPU is deliberate -- CPU float16 inference in
    torch is slower than float32, not faster, and numerical stability of
    a perplexity readout matters more here than memory.

    Local and free. No network at inference time (weights come from the
    HuggingFace cache), no API cost, ever.
    """

    def __init__(self, model_name: str = PERPLEXITY_MODEL) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self._model.eval()

    # --- tokens ---

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])

    # --- stage 1: per-line perplexity ---

    def line_perplexities(self, text: str) -> list[float]:
        """exp(mean token NLL) per line of `text`, from ONE forward pass
        (deviation 4). Lines contributing no tokens (e.g. a blank line
        swallowed into a neighbour's token) get `nan`, which the
        segmentation step treats as "no boundary signal here" rather than
        as a zero.
        """
        lines = text.splitlines()
        if not lines:
            return []
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_PPL_TOKENS,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        if len(input_ids) < 2:
            return [float("nan")] * len(lines)

        ids = self._torch.tensor([input_ids])
        with self._torch.no_grad():
            logits = self._model(ids).logits
        log_probs = self._torch.log_softmax(logits[0, :-1].float(), dim=-1)
        targets = ids[0, 1:]
        # Negate the TENSOR, then convert -- `-x.tolist()` would apply the
        # unary minus to the list, not to the values.
        token_nll = (-log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)).tolist()

        # Character offset -> line index, computed once for the whole text.
        line_of_char: list[int] = []
        for line_index, line in enumerate(lines):
            line_of_char.extend([line_index] * (len(line) + 1))
        if not line_of_char:
            return [float("nan")] * len(lines)

        sums = [0.0] * len(lines)
        counts = [0] * len(lines)
        # token_nll[i] is the loss for predicting token i+1, so it is
        # attributed to the line that token i+1 starts on.
        for i, nll in enumerate(token_nll):
            start_char = offsets[i + 1][0]
            if start_char >= len(line_of_char):
                continue
            line_index = line_of_char[start_char]
            sums[line_index] += nll
            counts[line_index] += 1

        out: list[float] = []
        for total, count in zip(sums, counts, strict=True):
            out.append(math.exp(min(total / count, 70.0)) if count else float("nan"))
        return out

    # --- stage 2: AMI via conditional perplexity ---

    def perplexity(self, text: str, prefix: str = "") -> float:
        """PPL(text) when `prefix` is empty, else PPL(text | prefix).

        Only the TEXT tokens' losses are averaged -- the prefix is
        conditioning context, never part of the quantity being scored.
        That distinction is what makes Equation 4's
        AMI(b, q) = PPL(q) - PPL(q | b) a measure of how much block b
        helps predict the query, rather than a measure of how surprising
        b itself is.
        """
        text_ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(text_ids) < 1:
            return float("nan")
        prefix_ids = (
            self._tokenizer(
                prefix, add_special_tokens=False, truncation=True, max_length=MAX_PPL_TOKENS
            )["input_ids"]
            if prefix
            else []
        )
        all_ids = [*prefix_ids, *text_ids]
        if len(all_ids) < 2:
            return float("nan")
        ids = self._torch.tensor([all_ids])
        with self._torch.no_grad():
            logits = self._model(ids).logits
        log_probs = self._torch.log_softmax(logits[0, :-1].float(), dim=-1)
        targets = ids[0, 1:]
        token_nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # Losses for the text's own tokens start where the prefix ends.
        scored = token_nll[max(len(prefix_ids) - 1, 0) :]
        if scored.numel() == 0:
            return float("nan")
        return math.exp(min(float(scored.mean()), 70.0))

    # --- stage 2: Sim(b, b_j) ---

    def embed(self, texts: list[str]) -> np.ndarray:
        """Mean-pooled final hidden states, L2-normalized (deviation 3)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: list[np.ndarray] = []
        for text in texts:
            ids = self._tokenizer(
                text or " ",
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_PPL_TOKENS,
            )["input_ids"]
            if ids.shape[1] == 0:
                ids = self._torch.tensor([[self._tokenizer.eos_token_id or 0]])
            with self._torch.no_grad():
                hidden = self._model(ids, output_hidden_states=True).hidden_states[-1]
            vectors.append(hidden[0].float().mean(dim=0).numpy())
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


_scorer_cache: dict[str, QwenPerplexityScorer] = {}


def get_shared_perplexity_scorer(model_name: str = PERPLEXITY_MODEL) -> QwenPerplexityScorer:
    """One 0.5B model per process, shared across every question -- mirrors
    `ant.retrieval.dense.get_shared_embedder`'s own rationale."""
    if model_name not in _scorer_cache:
        _scorer_cache[model_name] = QwenPerplexityScorer(model_name)
    return _scorer_cache[model_name]


# ---------------------------------------------------------------------------
# Stage 1: Perplexity-Guided Block Segmentation
# ---------------------------------------------------------------------------


def segment_into_blocks(
    text: str, scorer: PerplexityScorer, *, alpha: float = SEGMENTATION_ALPHA
) -> list[Block]:
    """Split one code unit into semantically coherent blocks using
    perplexity spikes as boundaries (paper section 2.2, stage 1;
    deviation 2 for the exact inequality)."""
    lines = text.splitlines()
    if not lines:
        return []
    if len(lines) == 1:
        return [Block(1, 1, lines[0], scorer.count_tokens(lines[0]))]

    perplexities = scorer.line_perplexities(text)
    if len(perplexities) != len(lines):
        perplexities = (perplexities + [float("nan")] * len(lines))[: len(lines)]

    finite = [p for p in perplexities if not math.isnan(p)]
    # Fewer than two measurable lines means there is no variability to
    # threshold against -- the whole unit is one block, which is the
    # correct degenerate answer, not an error.
    if len(finite) < 2:
        return [Block(1, len(lines), text, scorer.count_tokens(text))]
    stdev = float(np.std(finite))

    boundaries = {0}
    for i in range(1, len(lines)):
        current = perplexities[i]
        if math.isnan(current):
            continue
        neighbours = [
            perplexities[j]
            for j in (i - 1, i + 1)
            if 0 <= j < len(perplexities) and not math.isnan(perplexities[j])
        ]
        if not neighbours:
            continue
        if current > (sum(neighbours) / len(neighbours)) + alpha * stdev:
            boundaries.add(i)

    starts = sorted(boundaries)
    blocks: list[Block] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        block_text = "\n".join(lines[start:end])
        if not block_text.strip():
            continue
        blocks.append(
            Block(
                line_start=start + 1,
                line_end=end,
                text=block_text,
                n_tokens=scorer.count_tokens(block_text),
            )
        )
    return blocks


# ---------------------------------------------------------------------------
# Stage 2: MMR-Guided Block Selection
# ---------------------------------------------------------------------------


def select_blocks_mmr(
    blocks: list[Block],
    query: str,
    token_budget: int,
    scorer: PerplexityScorer,
    *,
    mmr_lambda: float = MMR_LAMBDA,
) -> list[Block]:
    """Greedy MMR selection under a token budget (Equations 3 and 4).

    Returns the selected blocks in ORIGINAL SOURCE ORDER, not selection
    order: the output is code that a downstream LLM has to read, and
    emitting a function's tail before its signature because the tail won
    the first MMR round would be actively misleading. The paper is silent
    on emission order; selection order is what MMR governs.
    """
    if not blocks or token_budget <= 0:
        return []

    # AMI(b, q) = PPL(q) - PPL(q | b), Equation 4. PPL(q) is constant
    # across blocks, so it is computed once.
    base_ppl = scorer.perplexity(query)
    ami: list[float] = []
    for block in blocks:
        conditional = scorer.perplexity(query, prefix=block.text)
        if math.isnan(conditional) or math.isnan(base_ppl):
            ami.append(0.0)
        else:
            ami.append(base_ppl - conditional)

    vectors = scorer.embed([block.text for block in blocks])
    similarity = vectors @ vectors.T if vectors.size else np.zeros((len(blocks), len(blocks)))

    selected: list[int] = []
    remaining = set(range(len(blocks)))
    budget_left = token_budget

    while remaining:
        best_index = -1
        best_score = -math.inf
        for index in sorted(remaining):
            if blocks[index].n_tokens > budget_left:
                continue
            redundancy = max((float(similarity[index][j]) for j in selected), default=0.0)
            score = mmr_lambda * ami[index] - (1.0 - mmr_lambda) * redundancy
            # Ties broken by the lower index (sorted iteration + strict >)
            # so selection is deterministic across runs.
            if score > best_score:
                best_score = score
                best_index = index
        if best_index < 0:
            # "This process continues until the budget is exhausted or no
            # further blocks can be added."
            break
        selected.append(best_index)
        remaining.discard(best_index)
        budget_left -= blocks[best_index].n_tokens

    return [blocks[i] for i in sorted(selected)]


def compress_unit(
    snippet: str,
    query: str,
    budget_rate: float,
    scorer: PerplexityScorer,
    *,
    alpha: float = SEGMENTATION_ALPHA,
    mmr_lambda: float = MMR_LAMBDA,
) -> str:
    """Full CABA pipeline for one retrieved code unit.

    "Given a function f, query q, and token budget t (where t is computed
    as the function's token length multiplied by the budget rate assigned
    by CAPO)".

    The two saturating rates are handled without spending any model
    compute, because the paper defines them as exact: 0% is "Fully
    filtered (empty)" and 100% is "Original text (no compression)" (see
    the verbatim budget semantics in `repodistill.py`). Short-circuiting
    them is a fidelity point, not an optimization -- running segmentation
    and MMR at rate 1.0 could drop a block to rounding and would NOT be
    "the original text".
    """
    if budget_rate <= 0.0:
        return ""
    if budget_rate >= 1.0:
        return snippet
    token_budget = int(scorer.count_tokens(snippet) * budget_rate)
    if token_budget <= 0:
        return ""
    blocks = segment_into_blocks(snippet, scorer, alpha=alpha)
    if not blocks:
        return ""
    kept = select_blocks_mmr(blocks, query, token_budget, scorer, mmr_lambda=mmr_lambda)
    if kept:
        return "\n".join(block.text for block in kept)

    # DEVIATION 5 (official -> adapted -> reason).
    # Official setting: greedy MMR "continues until the budget is
    # exhausted or no further blocks can be added" -- with no special
    # case for "no block was ever small enough to add".
    # Adapted setting: when greedy selection retains NOTHING because every
    # block individually exceeds the budget, fall back to the single
    # highest-AMI block truncated at a line boundary to fit the budget.
    # Reason: the empty result is a real, reachable degenerate case (a
    # short function that segments into one or two large blocks under an
    # aggressive 25% budget), and taking it literally would silently
    # collapse a 25% decision into a 0% one. The paper's own prompt draws
    # exactly that distinction -- "0% : Fully filtered (empty)" is a
    # SEPARATE option from "25% : Aggressive compression (essential
    # information only)" -- so returning empty for a non-zero budget would
    # contradict the policy's own stated decision. Truncating at a line
    # boundary respects the token budget while keeping the 0%/non-0%
    # distinction meaningful.
    base_ppl = scorer.perplexity(query)
    def _ami(block: Block) -> float:
        conditional = scorer.perplexity(query, prefix=block.text)
        if math.isnan(conditional) or math.isnan(base_ppl):
            return 0.0
        return base_ppl - conditional

    best = max(blocks, key=lambda b: (_ami(b), -b.line_start))
    kept_lines: list[str] = []
    used = 0
    for line in best.text.splitlines():
        n_tokens = scorer.count_tokens(line)
        if kept_lines and used + n_tokens > token_budget:
            break
        kept_lines.append(line)
        used += n_tokens
    return "\n".join(kept_lines)
