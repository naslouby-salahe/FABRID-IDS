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
