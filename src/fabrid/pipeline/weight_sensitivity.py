from __future__ import annotations

from dataclasses import dataclass

from fabrid.allocation.contracts import FederationWeights
from fabrid.allocation.weight_sensitivity import (
    ClientDatasetCount,
    DatasetCounts,
    dataset_count_reference_weights,
    preregistered_weight_scenarios,
)
from fabrid.artifacts.layout import ArtifactLayout
from fabrid.domain.enums import ExperimentId, WeightMode
from fabrid.domain.identifiers import CampaignId
from fabrid.domain.values import DetectorSeed
from fabrid.evaluation.evaluator import EvaluationProvenance
from fabrid.evaluation.results import SeedBudgetEvaluation
from fabrid.pipeline.allocation import LoadedSeedScores, run_seed_budget
from fabrid.pipeline.materialization import MaterializedSeedBudget, materialize_seed_budget
from fabrid.pipeline.training import PreparedFederation
from fabrid.protocol.models import FabridProtocol


@dataclass(frozen=True, slots=True)
class WeightSensitivityExecution:
    evaluations: tuple[SeedBudgetEvaluation, ...]
    artifacts: tuple[MaterializedSeedBudget, ...]


def nbaiot_dataset_count_reference_weights(
    prepared: PreparedFederation,
) -> FederationWeights:
    return dataset_count_reference_weights(
        DatasetCounts(
            tuple(
                ClientDatasetCount(
                    client_id=client.dataset.client_id,
                    rows=client.dataset.benign.row_count,
                )
                for client in prepared.clients
            )
        )
    )


def run_weight_sensitivity_seed(
    campaign_id: CampaignId,
    detector_seed: DetectorSeed,
    scores: LoadedSeedScores,
    provenance: EvaluationProvenance,
    reference_weights: FederationWeights,
    protocol: FabridProtocol,
    layout: ArtifactLayout,
) -> WeightSensitivityExecution:
    evaluations: list[SeedBudgetEvaluation] = []
    artifacts: list[MaterializedSeedBudget] = []

    for scenario in preregistered_weight_scenarios(reference_weights):
        for budget_level in protocol.budgets:
            run = run_seed_budget(
                campaign_id=campaign_id,
                experiment_id=ExperimentId.WEIGHT_SENSITIVITY,
                variant_id=scenario.variant_id,
                detector_seed=detector_seed,
                budget_level=budget_level,
                scores=scores,
                protocol=protocol,
                provenance=provenance,
                weights=scenario.weights,
                weight_mode=WeightMode.DATASET_COUNT_PROXY,
            )
            evaluations.append(run.evaluation)
            artifacts.append(materialize_seed_budget(run, layout))

    return WeightSensitivityExecution(
        evaluations=tuple(evaluations),
        artifacts=tuple(artifacts),
    )
