from __future__ import annotations

import numpy as np

from fabrid.allocation.baselines.pooled_shared import (
    ClientPooledValidation,
    FederationPooledValidation,
    select_pooled_shared_threshold,
)
from fabrid.domain.enums import AllocationPolicy
from fabrid.domain.identifiers import AttackSubtypeId
from fabrid.domain.values import DetectorSeed, FalsePositiveRate, Threshold, TruePositiveRate
from fabrid.evaluation.metrics import (
    ClientFalsePositiveRate,
    ClientMacroRecall,
    SubtypeRecall,
    budget_usage_ratio,
    client_macro_recall,
    federation_fpr,
    federation_macro_recall,
    worst_client_recall,
)
from fabrid.pipeline.allocation import LoadedSeedScores, equal_client_weights
from fabrid.pipeline.diagnostics import DiagnosticPolicyEvidence
from fabrid.protocol.models import BudgetLevel


def _subtypes(scores: LoadedSeedScores, client_index: int) -> tuple[AttackSubtypeId, ...]:
    records = scores.clients[client_index].evaluation.attack_test.records
    return tuple(
        sorted(
            {
                record.attack_subtype
                for record in records
                if record.attack_subtype is not None
            },
            key=lambda subtype: subtype.value,
        )
    )


def _client_fpr(scores: LoadedSeedScores, client_index: int, threshold: Threshold) -> FalsePositiveRate:
    records = scores.clients[client_index].evaluation.benign_test.records
    if not records:
        return FalsePositiveRate(0.0)
    values = np.fromiter((record.score.value for record in records), dtype=np.float64)
    return FalsePositiveRate(float(np.mean(values > threshold.value)))


def _client_recall(scores: LoadedSeedScores, client_index: int, threshold: Threshold) -> ClientMacroRecall:
    client = scores.clients[client_index]
    recalls: list[SubtypeRecall] = []
    for subtype in _subtypes(scores, client_index):
        values = np.fromiter(
            (
                record.score.value
                for record in client.evaluation.attack_test.records
                if record.attack_subtype == subtype
            ),
            dtype=np.float64,
        )
        recalls.append(
            SubtypeRecall(
                subtype=subtype,
                rate=TruePositiveRate(float(np.mean(values > threshold.value))),
            )
        )
    return ClientMacroRecall(client.client_id, client_macro_recall(tuple(recalls)))


def run_pooled_shared_diagnostic(
    detector_seed: DetectorSeed,
    budget: BudgetLevel,
    scores: LoadedSeedScores,
) -> DiagnosticPolicyEvidence:
    weights = equal_client_weights(scores.population)
    validation = FederationPooledValidation(
        tuple(
            ClientPooledValidation(
                client_id=client.client_id,
                benign_frontier=client.frontier.benign_frontier,
                attack_validation=client.frontier.attack_validation,
            )
            for client in scores.clients
        )
    )
    selected = select_pooled_shared_threshold(validation, weights, budget.value)
    recalls = tuple(
        _client_recall(scores, index, selected.threshold)
        for index in range(len(scores.clients))
    )
    rates = tuple(
        ClientFalsePositiveRate(
            scores.clients[index].client_id,
            _client_fpr(scores, index, selected.threshold),
        )
        for index in range(len(scores.clients))
    )
    federation_rate = federation_fpr(rates, weights)
    return DiagnosticPolicyEvidence(
        detector_seed=detector_seed,
        budget=budget,
        policy=AllocationPolicy.POOLED_SHARED,
        macro_recall=federation_macro_recall(recalls),
        worst_client_recall=worst_client_recall(recalls),
        federation_fpr=federation_rate,
        budget_usage=budget_usage_ratio(federation_rate, budget.value),
    )
