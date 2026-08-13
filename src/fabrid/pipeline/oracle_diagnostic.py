from __future__ import annotations

import numpy as np

from fabrid.allocation.diagnostics.test_oracle import (
    OracleAccessToken,
    OracleAuthorization,
    allocate_test_oracle,
)
from fabrid.allocation.frontier import FederationFrontierInputs
from fabrid.allocation.frontier_inputs import (
    AttackSubtypeScores,
    FrontierScorePopulation,
    build_client_frontier_inputs,
)
from fabrid.allocation.problem import build_allocation_problem, merge_full_allocation
from fabrid.domain.coordinates import ExperimentCoordinate
from fabrid.domain.enums import DatasetId, ExperimentId, ExperimentVariantId, WeightMode
from fabrid.domain.scores import ScoreVector
from fabrid.pipeline.allocation import LoadedSeedScores, evaluate_policy, equal_client_weights
from fabrid.pipeline.diagnostics import DiagnosticPolicyEvidence
from fabrid.protocol.models import BudgetLevel, FabridProtocol
