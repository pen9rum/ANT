from __future__ import annotations

from ant.domain import TokenUsage

# Prices are configurable because model pricing changes. Defaults are conservative placeholders.
# Values are USD per 1M tokens.
#
# gpt-5.4-mini and gpt-5 confirmed against https://developers.openai.com/api/docs/pricing
# on 2026-08-27 (standard-tier rates; batch/flex/fast-mode tiers differ and aren't tracked
# here since TokenUsage doesn't record which tier a call used). gpt-5.4-mini is what
# ANT_MODEL is set to for every run in this project so far (orchestrator + workers); the
# judge is hash-locked to the dated snapshot "gpt-5-2025-08-07"
# (OFFICIAL_SWE_QA_PRO_JUDGE_MODEL in evaluation/judge.py), priced the same as its base
# "gpt-5" per the same pricing page.
DEFAULT_PRICING_PER_MILLION = {
    "gpt-5.4-nano": {"input": 0.05, "output": 0.40},
    "gpt-5.4-nano-2026-03-17": {"input": 0.05, "output": 0.40},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5-2025-08-07": {"input": 1.25, "output": 10.00},
}


def estimate_cost_usd(model: str, usage: TokenUsage) -> float:
    pricing = DEFAULT_PRICING_PER_MILLION.get(model)
    if pricing is None:
        return 0.0
    input_cost = usage.input_tokens / 1_000_000 * pricing["input"]
    output_cost = usage.output_tokens / 1_000_000 * pricing["output"]
    return round(input_cost + output_cost, 8)
