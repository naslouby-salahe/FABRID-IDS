from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fabrid.domain.values import AnalysisSeed, MetricDifference, Probability, RowCount


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    mean_difference: MetricDifference
    median_difference: MetricDifference
    confidence_interval_low: MetricDifference
    confidence_interval_high: MetricDifference
    confidence: Probability
    resamples: RowCount


def paired_bootstrap_ci(
    paired_differences: tuple[MetricDifference, ...],
    resamples: RowCount,
    seed: AnalysisSeed,
    confidence: Probability,
) -> BootstrapResult:
    if not paired_differences:
        raise ValueError("paired bootstrap requires at least one paired difference")
    if resamples.value < 1:
        raise ValueError("paired bootstrap requires at least one resample")
    if confidence.value <= 0.0 or confidence.value >= 1.0:
        raise ValueError("bootstrap confidence must be strictly between zero and one")

    raw = np.asarray(
        tuple(difference.value for difference in paired_differences),
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed.value)
    resample_indices = rng.integers(
        0,
        raw.shape[0],
        size=(resamples.value, raw.shape[0]),
    )
    resample_means = raw[resample_indices].mean(axis=1)
    lower_quantile = (1.0 - confidence.value) / 2.0
    upper_quantile = 1.0 - lower_quantile
    return BootstrapResult(
        mean_difference=MetricDifference(float(np.mean(raw))),
        median_difference=MetricDifference(float(np.median(raw))),
        confidence_interval_low=MetricDifference(
            float(np.quantile(resample_means, lower_quantile))
        ),
        confidence_interval_high=MetricDifference(
            float(np.quantile(resample_means, upper_quantile))
        ),
        confidence=confidence,
        resamples=resamples,
    )
