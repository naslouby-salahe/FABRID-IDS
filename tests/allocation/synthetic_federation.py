from __future__ import annotations

import numpy as np

from fabrid.allocation.problem import (
    AllocationProblem,
    ClientFrontierInputs,
    FederationFrontierInputs,
    FrontierScoreArtifacts,
    build_allocation_problem,
    build_client_frontier_inputs,
    equal_client_weights,
)
from fabrid.artifacts.paths import ScoreCoordinate
from fabrid.config import (
    LOCAL_TARGET_RATE_CAP,
    AnomalyScore,
    AttackSplit,
    AttackSubtypeId,
    BenignSplit,
    ClientId,
    DatasetId,
    DetectorSeed,
    FalsePositiveBudget,
    Label,
    RowCount,
    TargetFalsePositiveRate,
    UtilityEligibilityConfig,
)
from fabrid.datasets.registry import ClientPopulation
from fabrid.detector.scoring import ScorePartitionArtifact, ScoreRecord

ALPHA_GRID = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05)


def eligibility_config() -> UtilityEligibilityConfig:
    return UtilityEligibilityConfig(
        minimum_attack_validation_rows=100,
        minimum_eligible_subtypes=2,
        minimum_rows_per_subtype=50,
    )


def inputs_for(
    client_ids: tuple[ClientId, ...],
    *,
    attack_rows_per_subtype: RowCount = 300,
    subtypes: tuple[AttackSubtypeId, ...] = ("bashlite_scan", "mirai_scan"),
) -> FederationFrontierInputs:
    client_inputs: list[ClientFrontierInputs] = []
    for index, client_id in enumerate(client_ids):
        frontier, validation = synthetic_client_scores(
            client_id,
            seed=index + 1,
            attack_rows_per_subtype=attack_rows_per_subtype,
            subtypes=subtypes,
        )
        artifacts = FrontierScoreArtifacts(benign_frontier=frontier, attack_validation=validation)
        client_inputs.append(
            build_client_frontier_inputs(
                artifacts, ALPHA_GRID, calibration_row_count=frontier.row_count
            )
        )
    return FederationFrontierInputs(clients=tuple(client_inputs))


def synthetic_records(
    client_id: ClientId,
    scores: np.ndarray,
    split: BenignSplit | AttackSplit,
    *,
    label: Label,
    subtype: AttackSubtypeId | None = None,
) -> tuple[ScoreRecord, ...]:
    subtype_part = "" if subtype is None else f"|{subtype}"
    return tuple(
        ScoreRecord(
            sample_id=f"{client_id}|{split.value}{subtype_part}|{index}",
            source_file="synthetic.csv",
            source_row=index,
            score=float(score),
            label=label,
            attack_subtype=subtype,
            timestamp=None,
        )
        for index, score in enumerate(scores)
    )


def synthetic_partition(
    client_id: ClientId,
    scores: np.ndarray,
    split: BenignSplit | AttackSplit,
    *,
    label: Label,
    subtype: AttackSubtypeId | None = None,
    seed: DetectorSeed = 0,
) -> ScorePartitionArtifact:
    return ScorePartitionArtifact(
        coordinate=ScoreCoordinate(
            dataset_id=DatasetId.NBAIOT, detector_seed=seed, client_id=client_id
        ),
        split=split,
        records=synthetic_records(client_id, scores, split, label=label, subtype=subtype),
    )


def synthetic_client_scores(
    client_id: ClientId,
    *,
    benign_frontier_count: RowCount = 2_000,
    attack_rows_per_subtype: RowCount = 300,
    seed: DetectorSeed = 0,
    separation: AnomalyScore = 2.0,
    subtypes: tuple[AttackSubtypeId, ...] = ("bashlite_scan", "mirai_scan"),
) -> tuple[ScorePartitionArtifact, ScorePartitionArtifact]:
    rng = np.random.default_rng(seed)
    benign = rng.normal(size=benign_frontier_count)
    attack_scores = rng.normal(loc=separation, size=attack_rows_per_subtype * len(subtypes))
    frontier = synthetic_partition(
        client_id,
        benign,
        BenignSplit.FRONTIER,
        label=Label.BENIGN,
        seed=seed,
    )
    attack_records = tuple(
        record
        for subtype in subtypes
        for record in synthetic_records(
            client_id,
            attack_scores,
            AttackSplit.VALIDATION,
            label=Label.ATTACK,
            subtype=subtype,
        )
    )
    validation = ScorePartitionArtifact(
        coordinate=frontier.coordinate,
        split=AttackSplit.VALIDATION,
        records=attack_records,
    )
    return frontier, validation


def synthetic_population(client_ids: tuple[ClientId, ...]) -> ClientPopulation:
    return ClientPopulation(clients=client_ids)


def synthetic_problem(
    client_ids: tuple[ClientId, ...],
    budget: FalsePositiveBudget,
    *,
    maximum_target_rate: TargetFalsePositiveRate = LOCAL_TARGET_RATE_CAP,
    attack_rows_per_subtype: RowCount = 300,
    subtypes: tuple[AttackSubtypeId, ...] = ("bashlite_scan", "mirai_scan"),
) -> AllocationProblem:
    return build_allocation_problem(
        inputs_for(
            client_ids,
            attack_rows_per_subtype=attack_rows_per_subtype,
            subtypes=subtypes,
        ),
        equal_client_weights(synthetic_population(client_ids)),
        budget,
        eligibility_config(),
        maximum_target_rate,
    )
