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
