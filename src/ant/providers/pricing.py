from __future__ import annotations

from ant.domain import TokenUsage

# Prices are configurable because model pricing changes. Defaults are conservative placeholders.
# Values are USD per 1M tokens.
DEFAULT_PRICING_PER_MILLION = {
    "gpt-5.4-nano": {"input": 0.05, "output": 0.40},
    "gpt-5.4-nano-2026-03-17": {"input": 0.05, "output": 0.40},
}


def estimate_cost_usd(model: str, usage: TokenUsage) -> float:
    pricing = DEFAULT_PRICING_PER_MILLION.get(model)
    if pricing is None:
        return 0.0
    input_cost = usage.input_tokens / 1_000_000 * pricing["input"]
    output_cost = usage.output_tokens / 1_000_000 * pricing["output"]
    return round(input_cost + output_cost, 8)
