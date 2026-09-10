from __future__ import annotations

import pytest

from ant.evaluation_suite.bootstrap_ci import paired_bootstrap_ci


def test_same_seed_produces_byte_identical_result() -> None:
    deltas = [1.0, -0.5, 2.0, 0.0, -1.0, 0.5, 1.5, -0.25]
    a = paired_bootstrap_ci(deltas, seed=20260910, n_replicates=2000)
    b = paired_bootstrap_ci(deltas, seed=20260910, n_replicates=2000)
    assert a == b


def test_different_seed_can_produce_a_different_result() -> None:
    deltas = [1.0, -0.5, 2.0, 0.0, -1.0, 0.5, 1.5, -0.25]
    a = paired_bootstrap_ci(deltas, seed=1, n_replicates=2000)
    b = paired_bootstrap_ci(deltas, seed=2, n_replicates=2000)
    # Not asserting inequality is guaranteed in general, but for this
    # deltas set with a real spread it practically always differs --
    # regression-guards against an implementation that ignores the seed.
    assert (a.ci_low, a.ci_high) != (b.ci_low, b.ci_high)


def test_constant_deltas_collapse_to_a_point_interval() -> None:
    deltas = [2.0] * 10
    result = paired_bootstrap_ci(deltas, seed=42, n_replicates=1000)
    assert result.observed_mean == 2.0
    assert result.ci_low == 2.0
    assert result.ci_high == 2.0


def test_observed_mean_matches_plain_arithmetic_mean() -> None:
    deltas = [1.0, 2.0, 3.0, 4.0]
    result = paired_bootstrap_ci(deltas, seed=7, n_replicates=500)
    assert result.observed_mean == pytest.approx(2.5)


def test_ci_bounds_the_observed_mean_for_a_symmetric_distribution() -> None:
    deltas = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    result = paired_bootstrap_ci(deltas, seed=123, n_replicates=5000)
    assert result.ci_low <= result.observed_mean <= result.ci_high


def test_n_pairs_and_n_replicates_and_seed_are_recorded() -> None:
    deltas = [0.1, 0.2, 0.3]
    result = paired_bootstrap_ci(deltas, seed=99, n_replicates=1234)
    assert result.n_pairs == 3
    assert result.n_replicates == 1234
    assert result.seed == 99
    assert result.confidence == 0.95


def test_default_replicate_count_is_10000() -> None:
    from ant.evaluation_suite.bootstrap_ci import DEFAULT_N_REPLICATES

    assert DEFAULT_N_REPLICATES == 10_000


def test_empty_deltas_raises() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_ci([], seed=1)


def test_to_dict_round_trips_all_fields() -> None:
    result = paired_bootstrap_ci([1.0, -1.0], seed=5, n_replicates=100)
    d = result.to_dict()
    assert set(d) == {
        "observed_mean",
        "ci_low",
        "ci_high",
        "confidence",
        "n_replicates",
        "seed",
        "n_pairs",
    }
