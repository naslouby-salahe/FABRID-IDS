"""Record-level detection and budget-compliance metrics.

All functions take already-computed per-client (and, where relevant,
per-attack-subtype) confusion counts or rates rather than raw score arrays —
that keeps this module free of any dependency on the detector, scoring, or
calibration layers, matching the one-directional architecture where
evaluation only consumes typed results.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NewType

ClientId = NewType("ClientId", str)
AttackSubtype = NewType("AttackSubtype", str)


@dataclass(frozen=True, slots=True)
class ClientWeight:
    value: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"client weight must be in [0, 1], got {self.value}")


@dataclass(frozen=True, slots=True)
class TruePositiveRate:
    value: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"TPR must be in [0, 1], got {self.value}")


@dataclass(frozen=True, slots=True)
class FalsePositiveRate:
    value: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"FPR must be in [0, 1], got {self.value}")


def client_macro_recall(subtype_tpr: Mapping[AttackSubtype, TruePositiveRate]) -> float:
    """Mean subtype TPR for one client: R_k."""
    if not subtype_tpr:
        raise ValueError("client_macro_recall requires at least one attack subtype")
    return sum(tpr.value for tpr in subtype_tpr.values()) / len(subtype_tpr)


def federation_macro_recall(client_recall: Mapping[ClientId, float]) -> float:
    """MacroRecall = mean over clients of their per-client macro recall."""
    if not client_recall:
        raise ValueError("federation_macro_recall requires at least one client")
    return sum(client_recall.values()) / len(client_recall)


def worst_client_recall(client_recall: Mapping[ClientId, float]) -> float:
    """WorstClientRecall = min over clients of their per-client macro recall."""
    if not client_recall:
        raise ValueError("worst_client_recall requires at least one client")
    return min(client_recall.values())


def federation_fpr(
    client_fpr: Mapping[ClientId, FalsePositiveRate],
    client_weight: Mapping[ClientId, ClientWeight],
) -> float:
    """FPR_fed = sum_k w_k FPR_k."""
    if client_fpr.keys() != client_weight.keys():
        raise ValueError("client_fpr and client_weight must share the same client set")
    return sum(client_weight[k].value * client_fpr[k].value for k in client_fpr)


def budget_usage_ratio(federation_fpr_value: float, budget: float) -> float:
    """BUR = FPR_fed / B_FP. Never clamped."""
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    return federation_fpr_value / budget


def budget_violation_ratio(bur: float) -> float:
    """BVR = max(0, BUR - 1)."""
    return max(0.0, bur - 1.0)


@dataclass(frozen=True, slots=True)
class FprDispersion:
    median: float
    iqr: float
    minimum: float
    maximum: float
    coefficient_of_variation: float | None  # None (reported as NA) when mean FPR is 0


def fpr_dispersion(client_fpr: Mapping[ClientId, FalsePositiveRate]) -> FprDispersion:
    if not client_fpr:
        raise ValueError("fpr_dispersion requires at least one client")
    values = sorted(fpr.value for fpr in client_fpr.values())
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std_dev = math.sqrt(variance)
    coefficient_of_variation = None if mean == 0.0 else std_dev / mean
    return FprDispersion(
        median=_percentile(values, 0.50),
        iqr=_percentile(values, 0.75) - _percentile(values, 0.25),
        minimum=values[0],
        maximum=values[-1],
        coefficient_of_variation=coefficient_of_variation,
    )


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return sorted_values[lower_index] * (1 - fraction) + sorted_values[upper_index] * fraction


def false_alert_gini(client_false_alert_count: Mapping[ClientId, int]) -> float:
    """Gini concentration of false-alert counts across clients; 0 when total is 0."""
    if not client_false_alert_count:
        raise ValueError("false_alert_gini requires at least one client")
    counts = list(client_false_alert_count.values())
    total = sum(counts)
    if total == 0:
        return 0.0
    k = len(counts)
    absolute_differences = sum(abs(a - b) for a in counts for b in counts)
    return absolute_differences / (2 * k * total)
