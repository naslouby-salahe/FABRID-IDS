from __future__ import annotations

import numpy as np

from fabrid.allocation.contracts import Allocation, FederationWeights
from fabrid.calibration.final_calibration import FinalCalibrationInputs, calibrate_final_thresholds
from fabrid.calibration.order_statistic import alerts_above_threshold
from fabrid.domain.identifiers import AttackSubtypeId
from fabrid.domain.scores import ScoreVector
from fabrid.domain.values import DetectorSeed, FalsePositiveRate, TruePositiveRate
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
from fabrid.pipeline.allocation import LoadedSeedScores
from fabrid.pipeline.diagnostics import DiagnosticPolicyEvidence
from fabrid.protocol.models import BudgetLevel


def _vector(values: tuple[float, ...]) -> ScoreVector:
    return ScoreVector(np.asarray(values, dtype=np.float64))


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
