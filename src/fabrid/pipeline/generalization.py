from __future__ import annotations

from dataclasses import dataclass

from fabrid.allocation.frontier_inputs import FrontierScoreArtifacts
from fabrid.artifacts.layout import ArtifactLayout
from fabrid.artifacts.score import ScorePartitionArtifact
from fabrid.datasets.nbaiot.specification import (
    NBAIOT_DUAL_BOTNET_FAMILY_POPULATION,
    NBAIOT_PRIMARY_POPULATION,
)
from fabrid.domain.enums import AttackSplit, ExperimentId, ExperimentVariantId
from fabrid.domain.identifiers import AttackSubtypeId, CampaignId
from fabrid.domain.population import ClientPopulation
from fabrid.domain.values import DetectorSeed
from fabrid.evaluation.evaluator import (
    ClientEvaluationArtifacts,
    EvaluationProvenance,
)
from fabrid.evaluation.results import SeedBudgetEvaluation
from fabrid.pipeline.allocation import (
    LoadedClientScores,
    LoadedSeedScores,
    run_seed_budget,
)
from fabrid.pipeline.materialization import (
    MaterializedSeedBudget,
    materialize_seed_budget,
)
from fabrid.protocol.models import AttackFoldRotation, FabridProtocol


@dataclass(frozen=True, slots=True)
class GeneralizationExecution:
    evaluations: tuple[SeedBudgetEvaluation, ...]
    artifacts: tuple[MaterializedSeedBudget, ...]


def _filter_attack_artifact(
    artifact: ScorePartitionArtifact,
    expected_split: AttackSplit,
    allowed_subtypes: tuple[AttackSubtypeId, ...],
) -> ScorePartitionArtifact:
    if artifact.split is not expected_split:
        raise ValueError(
            f"generalization expected {expected_split.value}, got {artifact.split.value}"
        )
    filtered = ScorePartitionArtifact(
        coordinate=artifact.coordinate,
        split=artifact.split,
        records=tuple(
            record
            for record in artifact.records
            if record.attack_subtype in allowed_subtypes
        ),
    )
    if not filtered.records:
        raise ValueError(
            f"client {artifact.coordinate.client_id.value} has no rows for requested "
            f"{expected_split.value} subtypes"
        )
    return filtered


def restrict_generalization_scores(
    scores: LoadedSeedScores,
    population: ClientPopulation,
    validation_subtypes: tuple[AttackSubtypeId, ...],
    test_subtypes: tuple[AttackSubtypeId, ...],
) -> LoadedSeedScores:
    clients: list[LoadedClientScores] = []
    for client_id in population.clients:
        source = scores.for_client(client_id)
        clients.append(
            LoadedClientScores(
                client_id=client_id,
                frontier=FrontierScoreArtifacts(
                    benign_frontier=source.frontier.benign_frontier,
                    attack_validation=_filter_attack_artifact(
                        source.frontier.attack_validation,
                        AttackSplit.VALIDATION,
                        validation_subtypes,
                    ),
                ),
                evaluation=ClientEvaluationArtifacts(
                    final_calibration=source.evaluation.final_calibration,
                    benign_test=source.evaluation.benign_test,
                    attack_test=_filter_attack_artifact(
                        source.evaluation.attack_test,
                        AttackSplit.TEST,
                        test_subtypes,
                    ),
                ),
            )
        )
    return LoadedSeedScores(tuple(clients))


def _rotation_test_subtypes(
    protocol: FabridProtocol,
    rotation: AttackFoldRotation,
) -> tuple[AttackSubtypeId, ...]:
    return tuple(
        subtype
        for test_fold in rotation.test_folds
        for subtype in protocol.generalization.fold(test_fold).subtypes
    )


def _run_variant(
    campaign_id: CampaignId,
    experiment_id: ExperimentId,
    variant_id: ExperimentVariantId,
    detector_seed: DetectorSeed,
    scores: LoadedSeedScores,
    provenance: EvaluationProvenance,
    protocol: FabridProtocol,
    layout: ArtifactLayout,
) -> GeneralizationExecution:
    evaluations: list[SeedBudgetEvaluation] = []
    artifacts: list[MaterializedSeedBudget] = []
    for budget_level in protocol.budgets:
        run = run_seed_budget(
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            variant_id=variant_id,
            detector_seed=detector_seed,
            budget_level=budget_level,
            scores=scores,
            protocol=protocol,
            provenance=provenance,
        )
        evaluations.append(run.evaluation)
        artifacts.append(materialize_seed_budget(run, layout))
    return GeneralizationExecution(
        evaluations=tuple(evaluations),
        artifacts=tuple(artifacts),
    )


def run_attack_subtype_generalization_seed(
    campaign_id: CampaignId,
    detector_seed: DetectorSeed,
    scores: LoadedSeedScores,
    provenance: EvaluationProvenance,
    protocol: FabridProtocol,
    layout: ArtifactLayout,
) -> GeneralizationExecution:
    evaluations: list[SeedBudgetEvaluation] = []
    artifacts: list[MaterializedSeedBudget] = []
    for rotation in protocol.generalization.rotations:
        restricted = restrict_generalization_scores(
            scores=scores,
            population=NBAIOT_PRIMARY_POPULATION,
            validation_subtypes=protocol.generalization.fold(
                rotation.validation_fold
            ).subtypes,
            test_subtypes=_rotation_test_subtypes(protocol, rotation),
        )
        execution = _run_variant(
            campaign_id=campaign_id,
            experiment_id=ExperimentId.ATTACK_SUBTYPE_DISJOINT,
            variant_id=rotation.variant_id,
            detector_seed=detector_seed,
            scores=restricted,
            provenance=provenance,
            protocol=protocol,
            layout=layout,
        )
        evaluations.extend(execution.evaluations)
        artifacts.extend(execution.artifacts)
    return GeneralizationExecution(
        evaluations=tuple(evaluations),
        artifacts=tuple(artifacts),
    )


def run_botnet_family_generalization_seed(
    campaign_id: CampaignId,
    detector_seed: DetectorSeed,
    scores: LoadedSeedScores,
    provenance: EvaluationProvenance,
    protocol: FabridProtocol,
    layout: ArtifactLayout,
) -> GeneralizationExecution:
    population = NBAIOT_DUAL_BOTNET_FAMILY_POPULATION
    if population.size != protocol.generalization.botnet_eligible_client_count:
        raise ValueError(
            "N-BaIoT dual-family population does not match the frozen protocol count"
        )

    subset_provenance = provenance.subset(population)
    evaluations: list[SeedBudgetEvaluation] = []
    artifacts: list[MaterializedSeedBudget] = []
    for direction in protocol.generalization.family_directions:
        restricted = restrict_generalization_scores(
            scores=scores,
            population=population,
            validation_subtypes=protocol.generalization.family(
                direction.validation_family
            ).subtypes,
            test_subtypes=protocol.generalization.family(direction.test_family).subtypes,
        )
        execution = _run_variant(
            campaign_id=campaign_id,
            experiment_id=ExperimentId.BOTNET_FAMILY_DISJOINT,
            variant_id=direction.variant_id,
            detector_seed=detector_seed,
            scores=restricted,
            provenance=subset_provenance,
            protocol=protocol,
            layout=layout,
        )
        evaluations.extend(execution.evaluations)
        artifacts.extend(execution.artifacts)
    return GeneralizationExecution(
        evaluations=tuple(evaluations),
        artifacts=tuple(artifacts),
    )
