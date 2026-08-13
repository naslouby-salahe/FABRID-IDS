from __future__ import annotations

import math
from dataclasses import dataclass

from fabrid.allocation.contracts import FederationWeights
from fabrid.domain.identifiers import AttackSubtypeId, ClientId
from fabrid.domain.values import (
    BudgetUsageRatio,
    BudgetViolationRatio,
    CoefficientOfVariation,
    FalseAlertCount,
    FalsePositiveBudget,
    FalsePositiveRate,
    MacroRecall,
    Probability,
    TruePositiveRate,
    WorstClientRecall,
)


@dataclass(frozen=True, slots=True)
class SubtypeRecall:
    subtype: AttackSubtypeId
    rate: TruePositiveRate


@dataclass(frozen=True, slots=True)
class ClientMacroRecall:
    client_id: ClientId
    recall: MacroRecall


@dataclass(frozen=True, slots=True)
class ClientFalsePositiveRate:
    client_id: ClientId
    rate: FalsePositiveRate


@dataclass(frozen=True, slots=True)
class ClientFalseAlerts:
    client_id: ClientId
    count: FalseAlertCount


@dataclass(frozen=True, slots=True)
class FprDispersion:
    median: FalsePositiveRate
    interquartile_range: FalsePositiveRate
    minimum: FalsePositiveRate
    maximum: FalsePositiveRate
    coefficient_of_variation: CoefficientOfVariation | None


def client_macro_recall(subtypes: tuple[SubtypeRecall, ...]) -> MacroRecall:
    if not subtypes:
        raise ValueError("client macro recall requires at least one attack subtype")
    subtype_ids = tuple(subtype.subtype for subtype in subtypes)
    if len(set(subtype_ids)) != len(subtype_ids):
        raise ValueError("client macro recall contains duplicate attack subtypes")
    return MacroRecall(sum(subtype.rate.value for subtype in subtypes) / len(subtypes))


def federation_macro_recall(clients: tuple[ClientMacroRecall, ...]) -> MacroRecall:
    if not clients:
        raise ValueError("federation macro recall requires at least one client")
    client_ids = tuple(client.client_id for client in clients)
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("federation macro recall contains duplicate clients")
    return MacroRecall(sum(client.recall.value for client in clients) / len(clients))


def worst_client_recall(clients: tuple[ClientMacroRecall, ...]) -> WorstClientRecall:
    if not clients:
        raise ValueError("worst-client recall requires at least one client")
    return WorstClientRecall(min(client.recall.value for client in clients))


def federation_fpr(
    client_rates: tuple[ClientFalsePositiveRate, ...],
    weights: FederationWeights,
) -> FalsePositiveRate:
    if not client_rates:
        raise ValueError("federation FPR requires at least one client")
    rate_clients = {client.client_id for client in client_rates}
    weight_clients = {client.client_id for client in weights.clients}
    if rate_clients != weight_clients:
        raise ValueError("client FPRs and federation weights must cover the same clients")
    return FalsePositiveRate(
        sum(
            weights.for_client(client.client_id).value * client.rate.value
            for client in client_rates
        )
    )


def budget_usage_ratio(
    federation_rate: FalsePositiveRate,
    budget: FalsePositiveBudget,
) -> BudgetUsageRatio | None:
    if budget.value == 0.0:
        return None
    return BudgetUsageRatio(federation_rate.value / budget.value)


def budget_violation_ratio(usage: BudgetUsageRatio | None) -> BudgetViolationRatio | None:
    if usage is None:
        return None
    return BudgetViolationRatio(max(0.0, usage.value - 1.0))


def fpr_dispersion(
    client_rates: tuple[ClientFalsePositiveRate, ...],
) -> FprDispersion:
    if not client_rates:
        raise ValueError("FPR dispersion requires at least one client")
    values = tuple(sorted(client.rate.value for client in client_rates))
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    standard_deviation = math.sqrt(variance)
    coefficient = (
        None
        if mean == 0.0
        else CoefficientOfVariation(standard_deviation / mean)
    )
    return FprDispersion(
        median=FalsePositiveRate(_percentile(values, 0.50)),
        interquartile_range=FalsePositiveRate(
            _percentile(values, 0.75) - _percentile(values, 0.25)
        ),
        minimum=FalsePositiveRate(values[0]),
        maximum=FalsePositiveRate(values[-1]),
        coefficient_of_variation=coefficient,
    )


def _percentile(sorted_values: tuple[float, ...], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def false_alert_gini(clients: tuple[ClientFalseAlerts, ...]) -> Probability:
    if not clients:
        raise ValueError("false-alert Gini requires at least one client")
    counts = tuple(client.count.value for client in clients)
    total = sum(counts)
    if total == 0:
        return Probability(0.0)
    client_count = len(counts)
    absolute_differences = sum(abs(left - right) for left in counts for right in counts)
    return Probability(absolute_differences / (2.0 * client_count * total))
