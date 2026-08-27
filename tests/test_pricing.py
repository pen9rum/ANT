from ant.domain import TokenUsage
from ant.providers.pricing import estimate_cost_usd


def test_estimate_cost_usd_is_nonzero_for_the_models_this_project_actually_uses() -> None:
    # Regression test: DEFAULT_PRICING_PER_MILLION only had entries for
    # gpt-5.4-nano, so every real run's actual model -- gpt-4.1, the
    # OpenAIProvider default (see config.load_dotenv's override=True
    # unconditionally re-applying .env's ANT_MODEL=gpt-4.1 on every
    # construction, even over an explicitly-set os.environ["ANT_MODEL"]; a
    # script setting ANT_MODEL=gpt-5.4-mini beforehand never actually took
    # effect) -- and the hash-locked judge model gpt-5-2025-08-07 (passed
    # explicitly, so the override bug never touched it) were silently
    # priced at $0 (estimate_cost_usd degrades to 0.0 on an unlisted model
    # instead of raising), with no error or warning anywhere. Confirmed on
    # a real saved trace: 184603 input / 6780 output tokens, reported as
    # $0.0000.
    usage = TokenUsage(input_tokens=184603, output_tokens=6780, total_tokens=191383)

    run_cost = estimate_cost_usd("gpt-4.1", usage)
    judge_cost = estimate_cost_usd("gpt-5-2025-08-07", usage)

    # $2.00/$8.00 vs $1.25/$10.00 per 1M input/output -- neither rate
    # dominates the other, so only the exact figures are asserted, not a
    # blanket "more expensive" comparison between the two.
    assert run_cost == round(184603 / 1_000_000 * 2.00 + 6780 / 1_000_000 * 8.00, 8)
    assert judge_cost == round(184603 / 1_000_000 * 1.25 + 6780 / 1_000_000 * 10.00, 8)


def test_estimate_cost_usd_degrades_to_zero_for_an_unlisted_model() -> None:
    usage = TokenUsage(input_tokens=1000, output_tokens=1000, total_tokens=2000)
    assert estimate_cost_usd("some-future-model-not-in-the-table-yet", usage) == 0.0
