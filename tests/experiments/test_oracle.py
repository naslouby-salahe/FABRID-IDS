from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
from pydantic import TypeAdapter

from fabrid.allocation.optimization import SolverStage
from fabrid.allocation.problem import build_allocation_problem, equal_client_weights
from fabrid.artifacts.paths import ExperimentCoordinate
from fabrid.config import (
    AllocationPolicy,
    AttackSplit,
    BenignSplit,
    DatasetId,
    ExperimentId,
    ExperimentVariantId,
    FabridConfig,
    Label,
    WeightMode,
)
from fabrid.detector.scoring import ScorePartitionArtifact, ScoreRecord
from fabrid.experiments.matched_budget import (
    ClientEvaluationArtifacts,
    CompletedPolicyRun,
    EvaluationProvenance,
    LoadedClientScores,
    LoadedSeedScores,
    build_frontier_inputs,
    oracle_loaded,
    run_seed_budget,
)
from tests.support import smoke_protocol

from ..allocation.synthetic_federation import synthetic_records

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _partition(
    records: tuple[ScoreRecord, ...],
    client_id: str,
    split: BenignSplit | AttackSplit,
) -> ScorePartitionArtifact:
    from fabrid.detector.scoring import ScoreCoordinate

    return ScorePartitionArtifact(
        coordinate=ScoreCoordinate(
            dataset_id=DatasetId.NBAIOT, detector_seed=0, client_id=client_id
        ),
        split=split,
        records=records,
    )


def _client_scores(client_id: str) -> LoadedClientScores:
    from fabrid.allocation.problem import FrontierScoreArtifacts

    benign_frontier = synthetic_records(
        client_id,
        np.linspace(0.0, 1.0, 400),
        BenignSplit.FRONTIER,
        label=Label.BENIGN,
    )
    attack_validation = synthetic_records(
        client_id,
        np.linspace(0.5, 1.0, 100),
        AttackSplit.VALIDATION,
        label=Label.ATTACK,
        subtype="mirai_scan",
    ) + synthetic_records(
        client_id,
        np.linspace(0.5, 1.0, 100),
        AttackSplit.VALIDATION,
        label=Label.ATTACK,
        subtype="bashlite_scan",
    )
    final_cal = synthetic_records(
        client_id,
        np.linspace(0.0, 1.0, 150),
        BenignSplit.FINAL_CAL,
        label=Label.BENIGN,
    )
    benign_test = synthetic_records(
        client_id,
        np.linspace(0.0, 1.0, 200),
        BenignSplit.TEST,
        label=Label.BENIGN,
    )
    attack_test = synthetic_records(
        client_id,
        np.linspace(0.5, 1.0, 100),
        AttackSplit.TEST,
        label=Label.ATTACK,
        subtype="mirai_scan",
    ) + synthetic_records(
        client_id,
        np.linspace(0.5, 1.0, 100),
        AttackSplit.TEST,
        label=Label.ATTACK,
        subtype="bashlite_scan",
    )
    return LoadedClientScores(
        client_id=client_id,
        frontier=FrontierScoreArtifacts(
            benign_frontier=_partition(benign_frontier, client_id, BenignSplit.FRONTIER),
            attack_validation=_partition(attack_validation, client_id, AttackSplit.VALIDATION),
        ),
        evaluation=ClientEvaluationArtifacts(
            client_id=client_id,
            final_calibration=_partition(final_cal, client_id, BenignSplit.FINAL_CAL),
            benign_test=_partition(benign_test, client_id, BenignSplit.TEST),
            attack_test=_partition(attack_test, client_id, AttackSplit.TEST),
        ),
    )


def _provenance() -> EvaluationProvenance:
    return TypeAdapter(EvaluationProvenance).validate_python(
        {
            "model_sha256": "a" * 64,
            "score_sha256": "b" * 64,
            "split_sha256": "c" * 64,
            "feature_sha256": "d" * 64,
            "protocol_sha256": "e" * 64,
            "git_commit": "f" * 40,
        }
    )


def _matched_budget_coordinate(config: FabridConfig) -> ExperimentCoordinate:
    budget_level = config.budgets[0]
    return ExperimentCoordinate(
        experiment_id=ExperimentId.MATCHED_BUDGET,
        variant_id=ExperimentVariantId.PRIMARY,
        dataset_id=DatasetId.NBAIOT,
        detector_seed=0,
        budget_id=budget_level.budget_id,
        budget=budget_level.value,
        weight_mode=WeightMode.EQUAL_CLIENT,
    )


def _oracle_run(config: FabridConfig, loaded: LoadedSeedScores):
    run = run_seed_budget(
        _matched_budget_coordinate(config),
        loaded,
        oracle_loaded(loaded),
        config,
        _provenance(),
        equal_client_weights(loaded.population),
    )
    return next(
        policy_run
        for policy_run in run.policy_runs
        if isinstance(policy_run, CompletedPolicyRun)
        and policy_run.evaluation.policy is AllocationPolicy.TEST_ORACLE
    )


def test_test_oracle_solves_the_test_attack_problem_optimally() -> None:
    config = smoke_protocol()
    loaded = LoadedSeedScores(clients=(_client_scores("a"), _client_scores("b")))
    oracle_run = _oracle_run(config, loaded)
    oracle_inputs = build_frontier_inputs(oracle_loaded(loaded), config.alpha_grid)
    oracle_problem = build_allocation_problem(
        oracle_inputs,
        equal_client_weights(loaded.population),
        config.budgets[0].value,
        config.utility_eligibility,
        config.maximum_target_rate,
    )
    curves = oracle_problem.frontier.eligible_curves()
    assert curves is not None
    weights = oracle_problem.weights.allocation_weights
    weight_by_client = {entry.client_id: entry.weight for entry in weights.clients}
    candidates = {
        curve.client_id: tuple((point.target_rate, float(point.utility)) for point in curve.points)
        for curve in curves.clients
    }
    client_ids = tuple(candidates)
    best: float | None = None
    for combo in product(*(candidates[client_id] for client_id in client_ids)):
        total_cost = sum(
            weight_by_client[client_id] * rate
            for (rate, _), client_id in zip(combo, client_ids, strict=True)
        )
        if total_cost > oracle_problem.remaining_budget + 1e-9:
            continue
        mean_utility = sum(
            utility for (_, utility), _ in zip(combo, client_ids, strict=True)
        ) / len(client_ids)
        if best is None or mean_utility > best + 1e-12:
            best = mean_utility
    assert best is not None
    stage = next(
        stage
        for stage in oracle_run.solver.stages
        if stage.stage is SolverStage.MACRO_PRIMARY_UTILITY
    )
    assert -float(stage.objective) == best, (
        "the oracle must achieve the exhaustive optimum of the test-attack problem"
    )


def test_test_oracle_is_recorded_but_excluded_from_practical_policies() -> None:
    config = smoke_protocol()
    loaded = LoadedSeedScores(clients=(_client_scores("a"), _client_scores("b")))
    run = run_seed_budget(
        _matched_budget_coordinate(config),
        loaded,
        oracle_loaded(loaded),
        config,
        _provenance(),
        equal_client_weights(loaded.population),
    )
    oracle_present = any(
        isinstance(policy_run, CompletedPolicyRun)
        and policy_run.evaluation.policy is AllocationPolicy.TEST_ORACLE
        for policy_run in run.policy_runs
    )
    assert oracle_present
