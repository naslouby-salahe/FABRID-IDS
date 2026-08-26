from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from fabrid.allocation.problem import FrontierScoreArtifacts, equal_client_weights
from fabrid.artifacts.paths import ExperimentCoordinate, ScoreCoordinate
from fabrid.config import (
    AllocationPolicy,
    AnomalyScore,
    AttackSplit,
    BenignSplit,
    ClientId,
    DatasetId,
    ExperimentId,
    ExperimentVariantId,
    Label,
    TargetFalsePositiveRate,
    Threshold,
    WeightMode,
)
from fabrid.detector.scoring import ScorePartitionArtifact, ScoreRecord
from fabrid.evaluation.metrics import ClientResultRecord
from fabrid.experiments.matched_budget import (
    ClientEvaluationArtifacts,
    EvaluationProvenance,
    LoadedClientScores,
    LoadedSeedScores,
    run_seed_budget,
)
from tests.support import smoke_protocol

from ..allocation.synthetic_federation import synthetic_records

ScoreRecordTransform = Callable[[ScoreRecord], ScoreRecord]


def _artifact(
    client_id: ClientId,
    split: BenignSplit | AttackSplit,
    records: tuple[ScoreRecord, ...],
) -> ScorePartitionArtifact:
    return ScorePartitionArtifact(
        coordinate=ScoreCoordinate(
            dataset_id=DatasetId.NBAIOT, detector_seed=0, client_id=client_id
        ),
        split=split,
        records=records,
    )


def _client_scores(client_id: ClientId) -> LoadedClientScores:
    benign_frontier = synthetic_records(
        client_id,
        np.linspace(0.0, 1.0, 400),
        BenignSplit.FRONTIER,
        label=Label.BENIGN,
    )
    attack_validation = synthetic_records(
        client_id,
        np.linspace(0.5, 1.0, 200),
        AttackSplit.VALIDATION,
        label=Label.ATTACK,
        subtype="mirai",
    ) + synthetic_records(
        client_id,
        np.linspace(0.5, 1.0, 200),
        AttackSplit.VALIDATION,
        label=Label.ATTACK,
        subtype="bashlite",
    )
    final_calibration = synthetic_records(
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
        np.linspace(0.5, 1.0, 200),
        AttackSplit.TEST,
        label=Label.ATTACK,
        subtype="mirai",
    )
    return LoadedClientScores(
        client_id=client_id,
        frontier=FrontierScoreArtifacts(
            benign_frontier=_artifact(client_id, BenignSplit.FRONTIER, benign_frontier),
            attack_validation=_artifact(client_id, AttackSplit.VALIDATION, attack_validation),
        ),
        evaluation=ClientEvaluationArtifacts(
            client_id=client_id,
            final_calibration=_artifact(client_id, BenignSplit.FINAL_CAL, final_calibration),
            benign_test=_artifact(client_id, BenignSplit.TEST, benign_test),
            attack_test=_artifact(client_id, AttackSplit.TEST, attack_test),
        ),
    )


def _loaded() -> LoadedSeedScores:
    return LoadedSeedScores(clients=(_client_scores("a"), _client_scores("b")))


def _provenance() -> EvaluationProvenance:
    return EvaluationProvenance(
        model_sha256="0" * 64,
        score_sha256="3" * 64,
        split_sha256="1" * 64,
        feature_sha256="4" * 64,
        protocol_sha256="5" * 64,
        git_commit="f" * 40,
    )


def _run(loaded: LoadedSeedScores) -> tuple[ClientResultRecord, ...]:
    config = smoke_protocol()
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
    return run.records


def _apply_records(
    records: tuple[ScoreRecord, ...],
    transform: ScoreRecordTransform,
) -> tuple[ScoreRecord, ...]:
    return tuple(transform(record) for record in records)


def _perturbed(
    loaded: LoadedSeedScores,
    split: BenignSplit | AttackSplit,
    transform: ScoreRecordTransform,
) -> LoadedSeedScores:
    clients: list[LoadedClientScores] = []
    for client in loaded.clients:
        if split is BenignSplit.FRONTIER:
            frontier = FrontierScoreArtifacts(
                benign_frontier=_artifact(
                    client.client_id,
                    BenignSplit.FRONTIER,
                    _apply_records(client.frontier.benign_frontier.records, transform),
                ),
                attack_validation=client.frontier.attack_validation,
            )
            clients.append(replace(client, frontier=frontier))
        elif split is AttackSplit.VALIDATION:
            frontier = FrontierScoreArtifacts(
                benign_frontier=client.frontier.benign_frontier,
                attack_validation=_artifact(
                    client.client_id,
                    AttackSplit.VALIDATION,
                    _apply_records(client.frontier.attack_validation.records, transform),
                ),
            )
            clients.append(replace(client, frontier=frontier))
        elif split is BenignSplit.FINAL_CAL:
            clients.append(
                replace(
                    client,
                    evaluation=ClientEvaluationArtifacts(
                        client_id=client.client_id,
                        final_calibration=_artifact(
                            client.client_id,
                            BenignSplit.FINAL_CAL,
                            _apply_records(client.evaluation.final_calibration.records, transform),
                        ),
                        benign_test=client.evaluation.benign_test,
                        attack_test=client.evaluation.attack_test,
                    ),
                )
            )
        elif split is BenignSplit.TEST:
            clients.append(
                replace(
                    client,
                    evaluation=ClientEvaluationArtifacts(
                        client_id=client.client_id,
                        final_calibration=client.evaluation.final_calibration,
                        benign_test=_artifact(
                            client.client_id,
                            BenignSplit.TEST,
                            _apply_records(client.evaluation.benign_test.records, transform),
                        ),
                        attack_test=client.evaluation.attack_test,
                    ),
                )
            )
        elif split is AttackSplit.TEST:
            clients.append(
                replace(
                    client,
                    evaluation=ClientEvaluationArtifacts(
                        client_id=client.client_id,
                        final_calibration=client.evaluation.final_calibration,
                        benign_test=client.evaluation.benign_test,
                        attack_test=_artifact(
                            client.client_id,
                            AttackSplit.TEST,
                            _apply_records(client.evaluation.attack_test.records, transform),
                        ),
                    ),
                )
            )
        else:
            raise ValueError(f"unsupported partition {split}")
    return LoadedSeedScores(clients=tuple(clients))


_PRACTICAL_POLICIES = frozenset(AllocationPolicy)
_ALLOCATION_POLICIES = (
    AllocationPolicy.EQ_FPR,
    AllocationPolicy.GREEDY,
    AllocationPolicy.FABRID_MACRO,
    AllocationPolicy.FABRID_CVAR,
)


def _with_score(record: ScoreRecord, score: AnomalyScore) -> ScoreRecord:
    return replace(record, score=score)


def _with_label(record: ScoreRecord, label: Label) -> ScoreRecord:
    return replace(record, label=label)


def _practical_alphas(
    records: tuple[ClientResultRecord, ...],
) -> tuple[tuple[AllocationPolicy, ClientId, TargetFalsePositiveRate], ...]:
    return tuple(
        (record.policy, record.client_id, record.alpha_selected)
        for record in records
        if record.policy in _PRACTICAL_POLICIES
    )


def _allocation_alphas(
    records: tuple[ClientResultRecord, ...],
) -> tuple[tuple[AllocationPolicy, ClientId, TargetFalsePositiveRate], ...]:
    return tuple(
        (record.policy, record.client_id, record.alpha_selected)
        for record in records
        if record.policy in _ALLOCATION_POLICIES
    )


def _practical_thresholds(
    records: tuple[ClientResultRecord, ...],
) -> tuple[tuple[AllocationPolicy, ClientId, Threshold], ...]:
    return tuple(
        (record.policy, record.client_id, record.threshold)
        for record in records
        if record.policy in _PRACTICAL_POLICIES
    )


def test_t02_test_label_integrity_is_enforced_at_the_artifact_boundary() -> None:
    loaded = _loaded()
    with pytest.raises(ValueError):
        _perturbed(loaded, BenignSplit.TEST, lambda record: _with_label(record, Label.ATTACK))
    with pytest.raises(ValueError):
        _perturbed(loaded, AttackSplit.TEST, lambda record: _with_label(record, Label.BENIGN))


def test_t07_score_hash_identity_across_policies() -> None:
    records = _run(_loaded())
    hashes = {record.score_sha256 for record in records}
    assert hashes == {_provenance().score_sha256}


def test_t03_test_score_perturbation_leaves_allocation_unchanged() -> None:
    baseline = _practical_alphas(_run(_loaded()))
    jittered = _perturbed(
        _loaded(),
        AttackSplit.TEST,
        lambda record: _with_score(record, min(1.0, record.score + 0.01)),
    )
    assert _practical_alphas(_run(jittered)) == baseline


def test_t04_benign_test_perturbation_leaves_allocated_rates_unchanged() -> None:
    baseline = _allocation_alphas(_run(_loaded()))
    jittered = _allocation_alphas(
        _run(
            _perturbed(
                _loaded(),
                BenignSplit.TEST,
                lambda record: _with_score(record, max(0.0, record.score - 0.01)),
            )
        )
    )
    assert jittered == baseline


def test_t05_final_calibration_perturbation_moves_threshold_not_alpha() -> None:
    baseline = _run(_loaded())
    jittered = _run(
        _perturbed(
            _loaded(),
            BenignSplit.FINAL_CAL,
            lambda record: _with_score(record, record.score * 0.5),
        )
    )
    assert _practical_alphas(jittered) == _practical_alphas(baseline)
    assert _practical_thresholds(baseline) != _practical_thresholds(jittered)


def test_t06_validation_attack_perturbation_moves_allocated_rates_not_equal_fpr() -> None:
    baseline = _practical_alphas(_run(_loaded()))
    perturbed = _practical_alphas(
        _run(
            _perturbed(
                _loaded(),
                AttackSplit.VALIDATION,
                lambda record: _with_score(record, min(1.0, record.score + 0.02)),
            )
        )
    )
    for policy, client_id, alpha in baseline:
        if policy is AllocationPolicy.EQ_FPR:
            assert (policy, client_id, alpha) in perturbed
    moved = [
        item
        for item in baseline
        if item[0] in (AllocationPolicy.GREEDY, AllocationPolicy.FABRID_MACRO)
        and item not in perturbed
    ]
    assert moved, "validation-attack perturbation must move at least one allocated rate"
