from __future__ import annotations

import random
from dataclasses import dataclass

DEFAULT_N_REPLICATES = 10_000
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class BootstrapCI:
    observed_mean: float
    ci_low: float
    ci_high: float
    confidence: float
    n_replicates: int
    seed: int
    n_pairs: int

    def to_dict(self) -> dict:
        return {
            "observed_mean": self.observed_mean,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "n_replicates": self.n_replicates,
            "seed": self.seed,
            "n_pairs": self.n_pairs,
        }


def paired_bootstrap_ci(
    deltas: list[float],
    *,
    seed: int,
    n_replicates: int = DEFAULT_N_REPLICATES,
    confidence: float = DEFAULT_CONFIDENCE,
) -> BootstrapCI:
    """Analysis-only paired bootstrap over a list of task-level paired
    deltas (e.g. `ANT_score - MatchedReAct_score` per task, one value per
    task already present on BOTH sides of the comparison). Resamples the
    deltas WITH REPLACEMENT `n_replicates` times using a fixed,
    deterministic `random.Random(seed)` stream (same seed -> byte-for-byte
    identical CI, every time -- see the determinism tests), takes each
    replicate's own mean, and reports the empirical (percentile) interval
    covering the middle `confidence` fraction of those replicate means.

    This performs NO generation, scoring, or judging of any kind -- it
    consumes only a list of already-computed deltas and never touches a
    task example, an agent, or a provider. It must never be used to
    decide whether to rerun, expand, drop, or modify any method; it is a
    read-only summary statistic over a result set that is already final.
    """
    if not deltas:
        msg = "deltas must be non-empty"
        raise ValueError(msg)
    n = len(deltas)
    rng = random.Random(seed)
    observed_mean = sum(deltas) / n

    replicate_means = []
    for _ in range(n_replicates):
        resample_sum = 0.0
        for _ in range(n):
            resample_sum += deltas[rng.randrange(n)]
        replicate_means.append(resample_sum / n)
    replicate_means.sort()

    alpha = (1 - confidence) / 2
    lo_index = max(0, min(n_replicates - 1, int(alpha * n_replicates)))
    hi_index = max(0, min(n_replicates - 1, int((1 - alpha) * n_replicates) - 1))

    return BootstrapCI(
        observed_mean=observed_mean,
        ci_low=replicate_means[lo_index],
        ci_high=replicate_means[hi_index],
        confidence=confidence,
        n_replicates=n_replicates,
        seed=seed,
        n_pairs=n,
    )
