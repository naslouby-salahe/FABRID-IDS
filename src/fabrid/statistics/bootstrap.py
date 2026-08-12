"""Paired seed-bootstrap confidence intervals for effect-size reporting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    mean_difference: float
    median_difference: float
    confidence_interval_low: float
    confidence_interval_high: float
    confidence: float
    resamples: int


def paired_bootstrap_ci(
    paired_differences: np.ndarray,
    resamples: int,
    seed: int,
    confidence: float = _DEFAULT_CONFIDENCE,
) -> BootstrapResult:
    n = paired_differences.shape[0]
    if n == 0:
        raise ValueError("paired_bootstrap_ci requires at least one paired difference")
    if resamples < 1:
        raise ValueError(f"resamples must be at least 1, got {resamples}")
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(0, n, size=(resamples, n))
    resample_means = paired_differences[resample_indices].mean(axis=1)

    lower_quantile = (1.0 - confidence) / 2
    upper_quantile = 1.0 - lower_quantile

    return BootstrapResult(
        mean_difference=float(np.mean(paired_differences)),
        median_difference=float(np.median(paired_differences)),
        confidence_interval_low=float(np.quantile(resample_means, lower_quantile)),
        confidence_interval_high=float(np.quantile(resample_means, upper_quantile)),
        confidence=confidence,
        resamples=resamples,
    )
