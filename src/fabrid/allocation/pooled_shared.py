"""POOLED_SHARED: centralized diagnostic, explicitly non-federated/non-deployable.

Pools client benign-frontier and attack-validation scores under one absolute
threshold selected by grid search to maximize mean per-client Macro Recall
subject to the weighted federation FPR budget. Purpose: determine whether
client-specific rate allocation adds value beyond one centrally selected
global score cutoff. Never call this a deployable federated policy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from fabrid.evaluation.record_level import AttackSubtype, ClientId


@dataclass(frozen=True, slots=True)
class ClientValidationData:
    benign_frontier_scores: np.ndarray
    attack_scores_by_subtype: Mapping[AttackSubtype, np.ndarray]


@dataclass(frozen=True, slots=True)
class PooledSharedResult:
    threshold: float
    macro_recall: float
    weighted_fpr: float


def pooled_candidate_thresholds(
    client_data: Mapping[ClientId, ClientValidationData],
) -> tuple[float, ...]:
    """Distinct pooled benign-frontier scores change which rows alert; +inf yields zero alerts."""
    pooled = np.concatenate(
        [
            data.benign_frontier_scores
            for data in client_data.values()
            if data.benign_frontier_scores.size
        ]
    )
    unique_scores = sorted({float(s) for s in pooled})
    return (*unique_scores, math.inf)


def _client_fpr(scores: np.ndarray, threshold: float) -> float:
    if scores.size == 0:
        return 0.0
    return float(np.mean(scores > threshold))


def _client_macro_recall(
    attack_scores_by_subtype: Mapping[AttackSubtype, np.ndarray], threshold: float
) -> float:
    subtype_tprs = [
        float(np.mean(scores > threshold))
        for scores in attack_scores_by_subtype.values()
        if scores.size > 0
    ]
    if not subtype_tprs:
        raise ValueError("client has no attack rows in any eligible subtype")
    return sum(subtype_tprs) / len(subtype_tprs)


def select_pooled_shared_threshold(
    client_data: Mapping[ClientId, ClientValidationData],
    weight: Mapping[ClientId, float],
    budget: float,
    candidate_thresholds: Sequence[float],
) -> PooledSharedResult:
    if not client_data:
        raise ValueError("select_pooled_shared_threshold requires at least one client")
    if client_data.keys() != weight.keys():
        raise ValueError("client_data and weight must share the same client set")
    if not candidate_thresholds:
        raise ValueError("candidate_thresholds must be non-empty")

    best: PooledSharedResult | None = None
    for threshold in candidate_thresholds:
        weighted_fpr = sum(
            weight[client_id] * _client_fpr(data.benign_frontier_scores, threshold)
            for client_id, data in client_data.items()
        )
        if weighted_fpr > budget + 1e-12:
            continue
        macro_recall = sum(
            _client_macro_recall(data.attack_scores_by_subtype, threshold)
            for data in client_data.values()
        ) / len(client_data)
        if best is None or macro_recall > best.macro_recall:
            best = PooledSharedResult(
                threshold=threshold, macro_recall=macro_recall, weighted_fpr=weighted_fpr
            )

    if best is None:
        return PooledSharedResult(threshold=math.inf, macro_recall=0.0, weighted_fpr=0.0)
    return best
