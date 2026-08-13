from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fabrid.allocation.contracts import FederationWeights
from fabrid.artifacts.score import ScorePartitionArtifact
from fabrid.domain.enums import AttackSplit, BenignSplit
from fabrid.domain.identifiers import AttackSubtypeId, ClientId
from fabrid.domain.values import FalsePositiveBudget, FalsePositiveRate, MacroRecall, Threshold

_BUDGET_TOLERANCE = 1.0e-12


@dataclass(frozen=True, slots=True)
class ClientPooledValidation:
    client_id: ClientId
    benign_frontier: ScorePartitionArtifact
    attack_validation: ScorePartitionArtifact

    def __post_init__(self) -> None:
        if self.benign_frontier.coordinate.client_id != self.client_id:
            raise ValueError("benign frontier artifact belongs to another client")
        if self.attack_validation.coordinate.client_id != self.client_id:
            raise ValueError("attack validation artifact belongs to another client")
        if self.benign_frontier.split is not BenignSplit.FRONTIER:
            raise ValueError("pooled baseline requires BENIGN_FRONTIER scores")
        if self.attack_validation.split is not AttackSplit.VALIDATION:
            raise ValueError("pooled baseline requires ATTACK_VALIDATION scores")


@dataclass(frozen=True, slots=True)
class FederationPooledValidation:
    clients: tuple[ClientPooledValidation, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("pooled baseline requires at least one client")
        client_ids = tuple(client.client_id for client in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("pooled baseline contains duplicate clients")


@dataclass(frozen=True, slots=True)
class PooledSharedResult:
    threshold: Threshold
    macro_recall: MacroRecall
    federation_fpr: FalsePositiveRate


def pooled_candidate_thresholds(
    validation: FederationPooledValidation,
) -> tuple[Threshold, ...]:
    values = {
        record.score.value
        for client in validation.clients
        for record in client.benign_frontier.records
    }
    return tuple(Threshold(value) for value in sorted(values)) + (Threshold(math.inf),)


def _client_fpr(client: ClientPooledValidation, threshold: Threshold) -> FalsePositiveRate:
    scores = client.benign_frontier.scores.values
    if scores.size == 0:
        return FalsePositiveRate(0.0)
    return FalsePositiveRate(float(np.mean(scores > threshold.value)))


def _client_macro_recall(
    client: ClientPooledValidation,
    threshold: Threshold,
) -> MacroRecall:
    grouped: dict[AttackSubtypeId, list[float]] = {}
    for record in client.attack_validation.records:
        if record.attack_subtype is None:
            raise ValueError("attack validation record is missing attack subtype")
        grouped.setdefault(record.attack_subtype, []).append(record.score.value)
    if not grouped:
        raise ValueError("client has no attack-validation subtypes")
    recalls = tuple(
        float(np.mean(np.asarray(scores, dtype=np.float64) > threshold.value))
        for scores in grouped.values()
    )
    return MacroRecall(sum(recalls) / len(recalls))


def select_pooled_shared_threshold(
    validation: FederationPooledValidation,
    weights: FederationWeights,
    budget: FalsePositiveBudget,
) -> PooledSharedResult:
    validation_clients = {client.client_id for client in validation.clients}
    weight_clients = {client.client_id for client in weights.clients}
    if validation_clients != weight_clients:
        raise ValueError("pooled baseline and federation weights must share clients")

    best: PooledSharedResult | None = None
    for threshold in pooled_candidate_thresholds(validation):
        federation_fpr = FalsePositiveRate(
            sum(
                weights.for_client(client.client_id).value
                * _client_fpr(client, threshold).value
                for client in validation.clients
            )
        )
        if federation_fpr.value > budget.value + _BUDGET_TOLERANCE:
            continue
        macro_recall = MacroRecall(
            sum(_client_macro_recall(client, threshold).value for client in validation.clients)
            / len(validation.clients)
        )
        candidate = PooledSharedResult(
            threshold=threshold,
            macro_recall=macro_recall,
            federation_fpr=federation_fpr,
        )
        if best is None or candidate.macro_recall.value > best.macro_recall.value:
            best = candidate

    if best is None:
        raise ValueError("pooled baseline has no budget-feasible threshold")
    return best
