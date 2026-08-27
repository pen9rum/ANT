from ant.domain import TokenUsage
from ant.providers.pricing import estimate_cost_usd


def test_estimate_cost_usd_is_nonzero_for_the_models_this_project_actually_uses() -> None:
    # Regression test: DEFAULT_PRICING_PER_MILLION only had entries for
    # gpt-5.4-nano, so every real run this whole project has used --
    # ANT_MODEL=gpt-5.4-mini for the orchestrator/workers, and the
    # hash-locked judge model gpt-5-2025-08-07 -- silently priced at $0
    # (estimate_cost_usd degrades to 0.0 on an unlisted model instead of
    # raising), with no error or warning anywhere. Confirmed on a real saved
    # trace: 184603 input / 6780 output tokens, reported as $0.0000.
    usage = TokenUsage(input_tokens=184603, output_tokens=6780, total_tokens=191383)

    orchestrator_cost = estimate_cost_usd("gpt-5.4-mini", usage)
    judge_cost = estimate_cost_usd("gpt-5-2025-08-07", usage)

    assert orchestrator_cost > 0
    assert judge_cost > 0
    # gpt-5's per-token rate is higher than gpt-5.4-mini's -- same usage
    # should cost more under it.
    assert judge_cost > orchestrator_cost


def test_estimate_cost_usd_degrades_to_zero_for_an_unlisted_model() -> None:
    usage = TokenUsage(input_tokens=1000, output_tokens=1000, total_tokens=2000)
    assert estimate_cost_usd("some-future-model-not-in-the-table-yet", usage) == 0.0
