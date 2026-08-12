from __future__ import annotations

import numpy as np
import torch

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
from fabrid.experiments.main_experiment import run_conservative_minimax_at_budget
from fabrid.schemas.score_artifact import DetectorSeed
from fabrid.scoring.score_generation import generate_score_artifact

_BENIGN_FRACTIONS = BenignSplitFractions(0.5, 0.7, 0.8)
_ATTACK_FRACTION = AttackSplitFraction(0.2)
_GRID = tuple(round(0.001 * i, 4) for i in range(11))
_GUARDRAILS = UtilityEligibilityGuardrails(
    min_attack_validation_rows=20, min_eligible_subtypes=2, min_rows_per_eligible_subtype=5
)
_SETTINGS = SolverSettings(mip_rel_gap=0.0, time_limit_seconds=10.0, accept_mip_gap_leq=1e-9)


def _client_artifact(seed: int, client_id: ClientId):
    rng = np.random.default_rng(seed)
    benign = rng.normal(0, 0.1, size=(300, 4)).astype(np.float64)
    attacks = {
        AttackSubtype("mirai_scan"): rng.normal(1.0, 0.1, size=(150, 4)).astype(np.float64),
        AttackSubtype("bashlite_udp"): rng.normal(1.0, 0.1, size=(150, 4)).astype(np.float64),
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


def test_conservative_minimax_returns_valid_recalls() -> None:
    artifacts = {
        ClientId("1"): _client_artifact(0, ClientId("1")),
        ClientId("2"): _client_artifact(1, ClientId("2")),
        ClientId("3"): _client_artifact(2, ClientId("3")),
    }
    result = run_conservative_minimax_at_budget(
        artifacts, _GRID, _GUARDRAILS, budget=0.01, solver_settings=_SETTINGS
    )
    assert result is not None
    macro_recall, worst_recall = result
    assert 0.0 <= macro_recall <= 1.0
    assert 0.0 <= worst_recall <= 1.0


def test_conservative_minimax_none_when_no_eligible_clients() -> None:
    tiny_guardrails = UtilityEligibilityGuardrails(
        min_attack_validation_rows=10_000, min_eligible_subtypes=2, min_rows_per_eligible_subtype=5
    )
    artifacts = {ClientId("1"): _client_artifact(0, ClientId("1"))}
    result = run_conservative_minimax_at_budget(
        artifacts, _GRID, tiny_guardrails, budget=0.01, solver_settings=_SETTINGS
    )
    assert result is None
