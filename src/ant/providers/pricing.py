from __future__ import annotations

from ant.domain import TokenUsage

# Prices are configurable because model pricing changes. Defaults are conservative placeholders.
# Values are USD per 1M tokens.
#
# gpt-4.1, gpt-5.4-mini, and gpt-5 confirmed against
# https://developers.openai.com/api/docs/pricing on 2026-08-27 (standard-tier rates;
# batch/flex/fast-mode tiers differ and aren't tracked here since TokenUsage doesn't record
# which tier a call used). gpt-4.1 is the OpenAIProvider default (see config.load_dotenv's
# override=True unconditionally re-applying .env's ANT_MODEL=gpt-4.1 on every construction,
# even over an explicitly-set os.environ["ANT_MODEL"]) and is what every run in this project
# actually used through 2026-08-27, despite several of them setting ANT_MODEL=gpt-5.4-mini
# beforehand -- that setting never took effect. gpt-5.4-mini is kept here for whenever a
# caller passes model="gpt-5.4-mini" explicitly (the only way to actually select it while
# the override bug stands). The judge is hash-locked to the dated snapshot
# "gpt-5-2025-08-07" (OFFICIAL_SWE_QA_PRO_JUDGE_MODEL in evaluation/judge.py, passed
# explicitly so the override bug never touched it), priced the same as its base "gpt-5" per
# the same pricing page.
DEFAULT_PRICING_PER_MILLION = {
    "gpt-5.4-nano": {"input": 0.05, "output": 0.40},
    "gpt-5.4-nano-2026-03-17": {"input": 0.05, "output": 0.40},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-5-2025-08-07": {"input": 1.25, "output": 10.00},
    # OpenRouter slug for the same model, same $/1M rate confirmed on
    # openrouter.ai/openai/gpt-4.1 (2026-09-23) -- OpenRouter passes this
    # model through at OpenAI's own list price, no markup observed.
    "openai/gpt-4.1": {"input": 2.00, "output": 8.00},
}


def estimate_cost_usd(model: str, usage: TokenUsage) -> float:
    pricing = DEFAULT_PRICING_PER_MILLION.get(model)
    if pricing is None and "/" in model:
        # A provider-routed name (e.g. OpenRouter's "openai/gpt-4.1", set via
        # ANT_MODEL when OPENAI_BASE_URL points at openrouter.ai instead of
        # api.openai.com -- see .env's own OPENAI_BASE_URL comment). This
        # table is priced from OpenAI's own listed rates; it is used here as
        # an APPROXIMATION for the same underlying model routed through
        # OpenRouter, which is usually but not guaranteed to bill at the
        # same per-token rate. Treat estimated_cost_usd as indicative, not
        # a substitute for OpenRouter's own dashboard/invoice, whenever
        # ANT_MODEL carries a "provider/" prefix.
        pricing = DEFAULT_PRICING_PER_MILLION.get(model.split("/", 1)[1])
    if pricing is None:
        return 0.0
    input_cost = usage.input_tokens / 1_000_000 * pricing["input"]
    output_cost = usage.output_tokens / 1_000_000 * pricing["output"]
    return round(input_cost + output_cost, 8)
