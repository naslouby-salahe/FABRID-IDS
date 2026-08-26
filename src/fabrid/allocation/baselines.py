from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fabrid.allocation.problem import AllocationProblem
from fabrid.config import (
    TIGHT_TOLERANCE,
    AttackSplit,
    AttackSubtypeId,
    BenignSplit,
    ClientId,
    FalsePositiveRate,
    MacroRecall,
    Threshold,
)
from fabrid.detector.scoring import ScorePartitionArtifact


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


def _pooled_candidate_thresholds(
    validation: FederationPooledValidation,
) -> tuple[Threshold, ...]:
    values = {
        record.score for client in validation.clients for record in client.benign_frontier.records
    }
    return tuple(sorted(values)) + (math.inf,)


def _attack_subtypes(client: ClientPooledValidation) -> tuple[AttackSubtypeId, ...]:
    subtypes: set[AttackSubtypeId] = set()
    for record in client.attack_validation.records:
        if record.attack_subtype is None:
            raise ValueError("attack validation record is missing attack subtype")
        subtypes.add(record.attack_subtype)
    return tuple(sorted(subtypes))


def select_pooled_shared_threshold(
    validation: FederationPooledValidation,
    problem: AllocationProblem,
) -> PooledSharedResult:
    weights = problem.weights
    validation_clients = {client.client_id for client in validation.clients}
    weight_clients = {client.client_id for client in weights.clients}
    if validation_clients != weight_clients:
        raise ValueError("pooled baseline and federation weights must share clients")
    thresholds = _pooled_candidate_thresholds(validation)
    threshold_values = np.asarray(thresholds, dtype=np.float64)
    per_client_fpr: list[np.ndarray] = []
    per_client_recall: list[np.ndarray] = []
    for client in validation.clients:
        benign = np.asarray(
            [record.score for record in client.benign_frontier.records],
            dtype=np.float64,
        )
        benign_sorted = np.sort(benign)
        if benign_sorted.size == 0:
            fpr_curve = np.zeros(threshold_values.shape, dtype=np.float64)
        else:
            fpr_curve = (
                benign_sorted.size - np.searchsorted(benign_sorted, threshold_values, side="right")
            ) / benign_sorted.size
        per_client_fpr.append(fpr_curve)
        subtypes = _attack_subtypes(client)
        if not subtypes:
            raise ValueError("client has no attack-validation subtypes")
        subtype_curves: list[np.ndarray] = []
        for subtype in subtypes:
            values = np.asarray(
                [
                    record.score
                    for record in client.attack_validation.records
                    if record.attack_subtype == subtype
                ],
                dtype=np.float64,
            )
            values_sorted = np.sort(values)
            if values_sorted.size == 0:
                tpr_curve = np.zeros(threshold_values.shape, dtype=np.float64)
            else:
                tpr_curve = (
                    values_sorted.size
                    - np.searchsorted(values_sorted, threshold_values, side="right")
                ) / values_sorted.size
            subtype_curves.append(tpr_curve)
        per_client_recall.append(np.mean(np.stack(subtype_curves), axis=0))
    fpr_matrix = np.stack(per_client_fpr)
    weight_vector = np.asarray(
        [weights.for_client(client.client_id) for client in validation.clients],
        dtype=np.float64,
    )
    federation_fpr: np.ndarray = weight_vector @ fpr_matrix
    macro_recall = np.mean(np.stack(per_client_recall), axis=0)
    feasible = federation_fpr <= problem.budget + TIGHT_TOLERANCE
    feasible_indices = np.nonzero(feasible)[0]
    if feasible_indices.size == 0:
        raise ValueError("pooled baseline has no budget-feasible threshold")
    best_index = int(feasible_indices[np.argmax(macro_recall[feasible_indices])])
    return PooledSharedResult(
        threshold=thresholds[best_index],
        macro_recall=float(macro_recall[best_index]),
        federation_fpr=float(federation_fpr[best_index]),
    )
