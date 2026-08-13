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
from fabrid.domain.identifiers import CampaignId
from fabrid.domain.scores import ScoreVector
from fabrid.evaluation.evaluator import EvaluationProvenance
from fabrid.pipeline.allocation import LoadedSeedScores, evaluate_policy, equal_client_weights
from fabrid.pipeline.diagnostics import DiagnosticPolicyEvidence
from fabrid.protocol.models import BudgetLevel, FabridProtocol


def _test_population(scores: LoadedSeedScores, client_index: int) -> FrontierScorePopulation:
    client = scores.clients[client_index]
    attack_records = client.evaluation.attack_test.records
    subtypes = tuple(
        sorted(
            {
                record.attack_subtype
                for record in attack_records
                if record.attack_subtype is not None
            },
            key=lambda subtype: subtype.value,
        )
    )
    attack_test = tuple(
        AttackSubtypeScores(
            subtype=subtype,
            scores=ScoreVector(
                np.fromiter(
                    (
                        record.score.value
                        for record in attack_records
                        if record.attack_subtype == subtype
                    ),
                    dtype=np.float64,
                )
            ),
        )
        for subtype in subtypes
    )
    benign = client.frontier.benign_frontier.scores
    return FrontierScorePopulation(
        client_id=client.client_id,
        benign_frontier=benign,
        attack_validation=attack_test,
    )
