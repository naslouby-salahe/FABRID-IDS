from __future__ import annotations

import numpy as np

from fabrid.allocation.baselines.pooled_shared import (
    ClientPooledValidation,
    FederationPooledValidation,
    select_pooled_shared_threshold,
)
from fabrid.domain.enums import AllocationPolicy
from fabrid.domain.identifiers import AttackSubtypeId
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
from fabrid.pipeline.allocation import LoadedSeedScores, equal_client_weights
from fabrid.pipeline.diagnostics import DiagnosticPolicyEvidence
from fabrid.protocol.models import BudgetLevel
