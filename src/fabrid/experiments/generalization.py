from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from fabrid.allocation.problem import FrontierScoreArtifacts, equal_client_weights
from fabrid.artifacts.paths import ExperimentCoordinate
from fabrid.config import (
    AttackSubtypeId,
    DatasetId,
    DetectorSeed,
    ExperimentId,
    ExperimentVariantId,
    FabridConfig,
    WeightMode,
)
from fabrid.datasets.registry import ClientPopulation
from fabrid.detector.scoring import ScorePartitionArtifact
from fabrid.experiments.matched_budget import (
    ClientEvaluationArtifacts,
    EvaluationProvenance,
    LoadedClientScores,
    LoadedSeedScores,
    SeedBudgetRun,
    oracle_loaded,
    run_seed_budget,
)


def filter_attack_partition(
    artifact: ScorePartitionArtifact,
    subtypes: tuple[AttackSubtypeId, ...],
) -> ScorePartitionArtifact:
    allowed = set(subtypes)
    records = tuple(record for record in artifact.records if record.attack_subtype in allowed)
    return ScorePartitionArtifact(
        coordinate=artifact.coordinate, split=artifact.split, records=records
    )


def restricted_seed_scores(
    loaded: LoadedSeedScores,
    validation_subtypes: tuple[AttackSubtypeId, ...],
    test_subtypes: tuple[AttackSubtypeId, ...],
) -> LoadedSeedScores:
    clients = tuple(
        LoadedClientScores(
            client_id=client.client_id,
            frontier=FrontierScoreArtifacts(
                benign_frontier=client.frontier.benign_frontier,
                attack_validation=filter_attack_partition(
                    client.frontier.attack_validation, validation_subtypes
                ),
            ),
            evaluation=ClientEvaluationArtifacts(
                client_id=client.client_id,
                final_calibration=client.evaluation.final_calibration,
                benign_test=client.evaluation.benign_test,
                attack_test=filter_attack_partition(client.evaluation.attack_test, test_subtypes),
            ),
        )
        for client in loaded.clients
    )
    return LoadedSeedScores(clients=clients)


@dataclass(frozen=True, slots=True)
class _GeneralizationDirection:
    variant_id: ExperimentVariantId
    validation_subtypes: tuple[AttackSubtypeId, ...]
    test_subtypes: tuple[AttackSubtypeId, ...]


def _run_generalization_seed(
    seed: DetectorSeed,
    loaded: LoadedSeedScores,
    config: FabridConfig,
    provenance: EvaluationProvenance,
    dataset_id: DatasetId,
    experiment_id: ExperimentId,
    directions: Iterable[_GeneralizationDirection],
) -> tuple[SeedBudgetRun, ...]:
    runs: list[SeedBudgetRun] = []
    for direction in directions:
        restricted = restricted_seed_scores(
            loaded, direction.validation_subtypes, direction.test_subtypes
        )
        oracle_scores = oracle_loaded(restricted)
        for budget_level in config.budgets:
            runs.append(
                run_seed_budget(
                    ExperimentCoordinate(
                        experiment_id=experiment_id,
                        variant_id=direction.variant_id,
                        dataset_id=dataset_id,
                        detector_seed=seed,
                        budget_id=budget_level.budget_id,
                        budget=budget_level.value,
                        weight_mode=WeightMode.EQUAL_CLIENT,
                    ),
                    restricted,
                    oracle_scores,
                    config,
                    provenance,
                    equal_client_weights(restricted.population),
                )
            )
    return tuple(runs)


def run_attack_subtype_generalization_seed(
    seed: DetectorSeed,
    loaded: LoadedSeedScores,
    config: FabridConfig,
    provenance: EvaluationProvenance,
    dataset_id: DatasetId,
) -> tuple[SeedBudgetRun, ...]:
    return _run_generalization_seed(
        seed,
        loaded,
        config,
        provenance,
        dataset_id,
        ExperimentId.ATTACK_SUBTYPE_DISJOINT,
        tuple(
            _GeneralizationDirection(
                rotation.variant_id,
                config.generalization.fold(rotation.validation_fold).subtypes,
                tuple(
                    subtype
                    for fold_id in rotation.test_folds
                    for subtype in config.generalization.fold(fold_id).subtypes
                ),
            )
            for rotation in config.generalization.rotations
        ),
    )


def run_botnet_family_generalization_seed(
    seed: DetectorSeed,
    loaded: LoadedSeedScores,
    config: FabridConfig,
    provenance: EvaluationProvenance,
    dataset_id: DatasetId,
    dual_botnet: ClientPopulation,
) -> tuple[SeedBudgetRun, ...]:
    if dual_botnet.size != config.generalization.botnet_eligible_client_count:
        raise ValueError("dual-botnet population size does not match the configured client count")
    family_loaded = LoadedSeedScores(
        clients=tuple(
            client for client in loaded.clients if client.client_id in dual_botnet.clients
        )
    )
    if not family_loaded.clients:
        raise ValueError("dual-botnet population has no loaded clients")
    return _run_generalization_seed(
        seed,
        family_loaded,
        config,
        provenance,
        dataset_id,
        ExperimentId.BOTNET_FAMILY_DISJOINT,
        tuple(
            _GeneralizationDirection(
                direction.variant_id,
                config.generalization.family(direction.validation_family).subtypes,
                config.generalization.family(direction.test_family).subtypes,
            )
            for direction in config.generalization.family_directions
        ),
    )
