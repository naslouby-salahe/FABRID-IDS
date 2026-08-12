from __future__ import annotations

import numpy as np
import torch

from fabrid.config.attack_folds import FoldId, FoldRotation, load_attack_folds
from fabrid.config.protocol import (
    AttackSplitFraction,
    BenignSplitFractions,
    SolverSettings,
    UtilityEligibilityGuardrails,
)
from fabrid.data.nbaiot_reader import RawDeviceData
from fabrid.data.partitioner import (
    RowCount,
    compute_attack_split_boundary,
    compute_benign_split_boundaries,
)
from fabrid.data.preprocessing import fit_feature_scaler
from fabrid.detector.model import Autoencoder, AutoencoderArchitecture
from fabrid.evaluation.record_level import AttackSubtype, ClientId
from fabrid.experiments.generalization import run_attack_subtype_disjoint_rotation
from fabrid.schemas.allocation import AllocationPolicy
from fabrid.schemas.score_artifact import DetectorSeed
from fabrid.scoring.score_generation import generate_score_artifact

_BENIGN_FRACTIONS = BenignSplitFractions(0.5, 0.7, 0.8)
_ATTACK_FRACTION = AttackSplitFraction(0.2)
_GRID = tuple(round(0.001 * i, 4) for i in range(11))
_GUARDRAILS = UtilityEligibilityGuardrails(
    min_attack_validation_rows=20, min_eligible_subtypes=1, min_rows_per_eligible_subtype=5
)
_SETTINGS = SolverSettings(mip_rel_gap=0.0, time_limit_seconds=10.0, accept_mip_gap_leq=1e-5)

# fold 0: bashlite_scan, bashlite_tcp, mirai_ack, mirai_udp
# fold 1: bashlite_junk, bashlite_udp, mirai_scan
# fold 2: bashlite_combo, mirai_syn, mirai_udpplain
_SUBTYPES = ("bashlite_scan", "mirai_udp", "mirai_scan", "mirai_syn")


def _client_artifact(seed: int, client_id: ClientId):
    rng = np.random.default_rng(seed)
    benign = rng.normal(0, 0.1, size=(300, 4)).astype(np.float64)
    attacks = {
        AttackSubtype(name): rng.normal(1.0, 0.1, size=(150, 4)).astype(np.float64)
        for name in _SUBTYPES
    }
    raw = RawDeviceData(
        client_id=client_id, benign_features=benign, attack_features_by_subtype=attacks
    )
    benign_boundaries = compute_benign_split_boundaries(RowCount(300), _BENIGN_FRACTIONS)
    attack_boundaries = {
        subtype: compute_attack_split_boundary(RowCount(features.shape[0]), _ATTACK_FRACTION)
        for subtype, features in attacks.items()
    }
    scaler = fit_feature_scaler(benign)
    torch.manual_seed(seed)
    model = Autoencoder(AutoencoderArchitecture(n_features=4, hidden_dims=(3,)))
    return generate_score_artifact(
        client_id,
        "n-baiot",
        DetectorSeed(seed),
        raw,
        benign_boundaries,
        attack_boundaries,
        scaler,
        model,
    )


def test_run_attack_subtype_disjoint_rotation_returns_valid_metrics() -> None:
    fold_config = load_attack_folds()
    rotation = FoldRotation(validation_fold=FoldId(0), test_folds=(FoldId(1), FoldId(2)))
    artifacts = {
        ClientId("1"): _client_artifact(0, ClientId("1")),
        ClientId("2"): _client_artifact(1, ClientId("2")),
        ClientId("3"): _client_artifact(2, ClientId("3")),
    }

    result = run_attack_subtype_disjoint_rotation(
        artifacts,
        _GRID,
        _GUARDRAILS,
        fold_config,
        rotation,
        budget=0.05,
        alpha_max=0.05,
        solver_settings=_SETTINGS,
    )

    assert AllocationPolicy.EQ_FPR in result.macro_recall_by_policy
    for macro_recall in result.macro_recall_by_policy.values():
        assert 0.0 <= macro_recall <= 1.0
    for worst_recall in result.worst_client_recall_by_policy.values():
        assert 0.0 <= worst_recall <= 1.0
    for bur in result.bur_by_policy.values():
        assert bur >= 0.0


def test_rotation_uses_only_frozen_fold_config() -> None:
    fold_config = load_attack_folds()
    rotation = fold_config.rotations[0]
    assert rotation.validation_fold not in rotation.test_folds
    validation_subtypes = set(fold_config.validation_subtypes(rotation))
    test_subtypes = set(fold_config.test_subtypes(rotation))
    assert validation_subtypes.isdisjoint(test_subtypes)
