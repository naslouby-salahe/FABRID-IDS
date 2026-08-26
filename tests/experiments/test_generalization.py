from __future__ import annotations

import numpy as np
from pydantic import TypeAdapter

from fabrid.allocation.problem import FrontierScoreArtifacts, equal_client_weights
from fabrid.artifacts.paths import ExperimentCoordinate, ScoreCoordinate
from fabrid.config import (
    AllocationPolicy,
    AttackSplit,
    AttackSubtypeId,
    BenignSplit,
    BotnetFamily,
    ClientId,
    DatasetId,
    ExperimentId,
    ExperimentVariantId,
    FabridConfig,
    Label,
    RowCount,
    WeightMode,
)
from fabrid.datasets.registry import ClientPopulation
from fabrid.detector.scoring import ScorePartitionArtifact, ScoreRecord
from fabrid.evaluation.metrics import CompletedPolicyEvaluation
from fabrid.experiments.generalization import (
    restricted_seed_scores,
    run_attack_subtype_generalization_seed,
    run_botnet_family_generalization_seed,
)
from fabrid.experiments.matched_budget import (
    ClientEvaluationArtifacts,
    CompletedPolicyRun,
    EvaluationProvenance,
    LoadedClientScores,
    LoadedSeedScores,
    run_seed_budget,
)
from tests.support import smoke_application, smoke_protocol

from ..allocation.synthetic_federation import synthetic_records


def _partition(
    records: tuple[ScoreRecord, ...],
    client_id: ClientId,
    split: BenignSplit | AttackSplit,
) -> ScorePartitionArtifact:
    return ScorePartitionArtifact(
        coordinate=ScoreCoordinate(
            dataset_id=DatasetId.NBAIOT, detector_seed=0, client_id=client_id
        ),
        split=split,
        records=records,
    )


def _client_scores(
    client_id: ClientId,
    validation_subtypes: tuple[AttackSubtypeId, ...],
    test_subtypes: tuple[AttackSubtypeId, ...],
    *,
    rows_per_subtype: RowCount = 60,
) -> LoadedClientScores:
    benign_frontier = synthetic_records(
        client_id,
        np.linspace(0.0, 1.0, 400),
        BenignSplit.FRONTIER,
        label=Label.BENIGN,
    )
    attack_validation = tuple(
        record
        for subtype in validation_subtypes
        for record in synthetic_records(
            client_id,
            np.linspace(0.5, 1.0, rows_per_subtype),
            AttackSplit.VALIDATION,
            label=Label.ATTACK,
            subtype=subtype,
        )
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
    attack_test = tuple(
        record
        for subtype in test_subtypes
        for record in synthetic_records(
            client_id,
            np.linspace(0.5, 1.0, rows_per_subtype),
            AttackSplit.TEST,
            label=Label.ATTACK,
            subtype=subtype,
        )
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


def _fold_subtypes(config: FabridConfig) -> tuple[AttackSubtypeId, ...]:
    return tuple(subtype for fold in config.generalization.folds for subtype in fold.subtypes)


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


def test_restricted_seed_scores_restricts_attack_partitions() -> None:
    loaded = LoadedSeedScores(
        clients=(
            _client_scores("a", ("mirai", "bashlite"), ("mirai", "bashlite")),
            _client_scores("b", ("mirai", "bashlite"), ("mirai", "bashlite")),
        )
    )
    restricted = restricted_seed_scores(loaded, ("mirai",), ("bashlite",))
    assert len(restricted.clients) == 2
    for client in restricted.clients:
        assert {record.attack_subtype for record in client.frontier.attack_validation.records} == {
            "mirai"
        }
        assert {record.attack_subtype for record in client.evaluation.attack_test.records} == {
            "bashlite"
        }


def test_run_seed_budget_falls_back_for_ineligible_client() -> None:
    config = smoke_protocol()
    loaded = LoadedSeedScores(
        clients=(
            _client_scores("a", ("mirai", "bashlite"), ("mirai",), rows_per_subtype=200),
            _client_scores("b", ("mirai", "bashlite"), ("mirai",), rows_per_subtype=15),
        )
    )
    budget_level = config.budgets[0]
    run = run_seed_budget(
        ExperimentCoordinate(
            experiment_id=ExperimentId.MATCHED_BUDGET,
            variant_id=ExperimentVariantId.PRIMARY,
            dataset_id=DatasetId.NBAIOT,
            detector_seed=0,
            budget_id=budget_level.budget_id,
            budget=budget_level.value,
            weight_mode=WeightMode.EQUAL_CLIENT,
        ),
        loaded,
        config,
        _provenance(),
        equal_client_weights(loaded.population),
    )
    for policy in run.evaluation.policies:
        assert isinstance(policy, CompletedPolicyEvaluation), policy.reason
    fabrid_run = next(
        policy_run
        for policy_run in run.policy_runs
        if isinstance(policy_run, CompletedPolicyRun)
        and policy_run.evaluation.policy is AllocationPolicy.FABRID_MACRO
    )
    decisions = fabrid_run.allocation.decisions
    assert {decision.client_id for decision in decisions} == {"a", "b"}
    fallback_target = fabrid_run.allocation.decision("b").target_rate
    assert fallback_target == min(config.budgets[0].value, config.maximum_target_rate)


def test_attack_subtype_generalization_uses_disjoint_fold_rotations() -> None:
    config = smoke_protocol()
    subtypes = _fold_subtypes(config)
    loaded = LoadedSeedScores(
        clients=(
            _client_scores("a", subtypes, subtypes),
            _client_scores("b", subtypes, subtypes),
        )
    )
    for rotation in config.generalization.rotations:
        validation_subtypes = config.generalization.fold(rotation.validation_fold).subtypes
        test_subtypes = tuple(
            subtype
            for fold_id in rotation.test_folds
            for subtype in config.generalization.fold(fold_id).subtypes
        )
        restricted = restricted_seed_scores(loaded, validation_subtypes, test_subtypes)
        for client in restricted.clients:
            validation_subtype_ids = {
                record.attack_subtype for record in client.frontier.attack_validation.records
            }
            test_subtype_ids = {
                record.attack_subtype for record in client.evaluation.attack_test.records
            }
            assert validation_subtype_ids == set(validation_subtypes)
            assert test_subtype_ids == set(test_subtypes)
            assert validation_subtype_ids.isdisjoint(test_subtype_ids)
    runs = run_attack_subtype_generalization_seed(
        0,
        loaded,
        config,
        _provenance(),
        DatasetId.NBAIOT,
    )
    assert len(runs) == len(config.generalization.rotations) * len(config.budgets)
    assert {run.evaluation.experiment.experiment_id for run in runs} == {
        ExperimentId.ATTACK_SUBTYPE_DISJOINT
    }
    assert {run.evaluation.experiment.variant_id for run in runs} == {
        rotation.variant_id for rotation in config.generalization.rotations
    }


def test_botnet_family_generalization_uses_disjoint_family_folds() -> None:
    config = smoke_protocol()
    application = smoke_application()
    bashlite_subtypes = config.generalization.family(BotnetFamily.BASHLITE).subtypes
    mirai_subtypes = config.generalization.family(BotnetFamily.MIRAI).subtypes
    dual_botnet_ids = application.datasets.nbaiot.dual_botnet_devices()
    excluded = application.datasets.nbaiot.dual_botnet_excluded_devices
    dual_botnet = ClientPopulation(dual_botnet_ids)
    family_clients = dual_botnet_ids[:2]
    outsider = excluded[0]
    family_subtypes = bashlite_subtypes + mirai_subtypes
    loaded = LoadedSeedScores(
        clients=(
            _client_scores(family_clients[0], family_subtypes, family_subtypes),
            _client_scores(family_clients[1], family_subtypes, family_subtypes),
            _client_scores(outsider, family_subtypes, family_subtypes),
        )
    )
    for direction in config.generalization.family_directions:
        validation_subtypes = config.generalization.family(direction.validation_family).subtypes
        test_subtypes = config.generalization.family(direction.test_family).subtypes
        restricted = restricted_seed_scores(loaded, validation_subtypes, test_subtypes)
        for client in restricted.clients:
            validation_subtype_ids = {
                record.attack_subtype for record in client.frontier.attack_validation.records
            }
            test_subtype_ids = {
                record.attack_subtype for record in client.evaluation.attack_test.records
            }
            assert validation_subtype_ids <= set(validation_subtypes)
            assert test_subtype_ids <= set(test_subtypes)
            assert validation_subtype_ids.isdisjoint(test_subtype_ids)
    runs = run_botnet_family_generalization_seed(
        0,
        loaded,
        config,
        _provenance(),
        DatasetId.NBAIOT,
        dual_botnet,
    )
    assert len(runs) == len(config.generalization.family_directions) * len(config.budgets)
    assert {run.evaluation.experiment.experiment_id for run in runs} == {
        ExperimentId.BOTNET_FAMILY_DISJOINT
    }
    assert {run.evaluation.experiment.variant_id for run in runs} == {
        direction.variant_id for direction in config.generalization.family_directions
    }
    for run in runs:
        assert {record.client_id for record in run.records} == set(family_clients)
        assert outsider not in {record.client_id for record in run.records}
