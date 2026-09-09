from __future__ import annotations

from dataclasses import dataclass

from ant.domain import TokenUsage
from ant.providers import OpenAIProvider
from ant.providers.pricing import estimate_cost_usd

# Centralized judge configuration (evaluation-suite philosophy correction):
# the fairness condition this suite targets is "same benchmark + same
# benchmark-specific scoring semantics + same judge model for every
# compared system" -- NOT "reproduce each paper's own original judge
# model". Every LLM-judged benchmark adapter in this suite calls THIS one
# function, so there is exactly one place that decides which model judges
# everything, never a per-benchmark choice that could silently drift.
#
# DEFAULT_JUDGE_MODEL is pinned to the exact GPT-5 snapshot SWE-QA-Pro's
# own official judge already uses (gpt-5-2025-08-07, verified against
# their pinned-commit source, already priced in ant.providers.pricing) --
# chosen as OUR standard because it's already integrated and already
# proven working in this codebase, not because a paper mandates it. A
# benchmark whose own paper happens to use a different judge (RepoProbe's
# own code defaults to gpt-5.2, its paper used Claude Sonnet 4.5;
# WebWalkerQA used GPT-4; BrowseComp-Plus used GPT-4.1) simply gets judged
# by this same GPT-5 snapshot instead -- that is not a fidelity gap to
# apologize for, it is the point: internal comparability across every
# system we run through this suite matters more than matching each paper's
# own, mutually-inconsistent judge choices.
DEFAULT_JUDGE_MODEL = "gpt-5-2025-08-07"
DEFAULT_JUDGE_REASONING_EFFORT = "low"
DEFAULT_JUDGE_VERBOSITY = "low"


@dataclass
class JudgeCallResult:
    text: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    model: str


def call_judge(
    *,
    system: str,
    user: str,
    model: str = DEFAULT_JUDGE_MODEL,
    max_output_tokens: int = 1024,
) -> JudgeCallResult:
    """The one call site every benchmark adapter's scorer routes through.
    Uses the Responses API's `reasoning`/`text.verbosity` controls (the
    shape GPT-5-class reasoning models require -- no `temperature`, which
    these models reject) -- same call shape originally verified against
    SWE-QA-Pro's own official judge.py, now the shared, benchmark-agnostic
    path for every adapter, not just that one.
    """
    provider = OpenAIProvider(model=model)
    client = provider.client()
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        reasoning={"effort": DEFAULT_JUDGE_REASONING_EFFORT},
        text={"verbosity": DEFAULT_JUDGE_VERBOSITY},
        max_output_tokens=max_output_tokens,
    )
    text = (getattr(response, "output_text", None) or "").strip()
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    usage_for_cost = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    cost = estimate_cost_usd(model, usage_for_cost)
    return JudgeCallResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        estimated_cost_usd=cost,
        model=model,
    )
