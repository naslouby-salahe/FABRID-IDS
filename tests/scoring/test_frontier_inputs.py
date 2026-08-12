from __future__ import annotations

import numpy as np
import torch

from fabrid.config.protocol import AttackSplitFraction, BenignSplitFractions
from fabrid.data.nbaiot_reader import RawDeviceData
from fabrid.data.partitioner import (
    RowCount,
    compute_attack_split_boundary,
    compute_benign_split_boundaries,
)
from fabrid.data.preprocessing import fit_feature_scaler
from fabrid.detector.model import Autoencoder, AutoencoderArchitecture
from fabrid.evaluation.record_level import AttackSubtype, ClientId
from fabrid.schemas.score_artifact import DetectorSeed
from fabrid.scoring.frontier_inputs import (
    all_test_auroc,
    attack_test_scores_by_subtype,
    benign_final_cal_scores,
    benign_test_scores,
    build_client_frontier_inputs,
)
from fabrid.scoring.score_generation import generate_score_artifact

_BENIGN_FRACTIONS = BenignSplitFractions(0.5, 0.7, 0.8)
_ATTACK_FRACTION = AttackSplitFraction(0.2)
_GRID = (0.0, 0.01, 0.02, 0.05)


def _artifact():
    rng = np.random.default_rng(0)
    benign = rng.normal(0, 0.1, size=(200, 4)).astype(np.float64)
    attacks = {
        AttackSubtype("mirai_scan"): rng.normal(1.0, 0.1, size=(120, 4)).astype(np.float64),
        AttackSubtype("bashlite_udp"): rng.normal(1.0, 0.1, size=(80, 4)).astype(np.float64),
    }
    raw = RawDeviceData(
        client_id=ClientId("1"), benign_features=benign, attack_features_by_subtype=attacks
    )
    benign_boundaries = compute_benign_split_boundaries(RowCount(200), _BENIGN_FRACTIONS)
    attack_boundaries = {
        subtype: compute_attack_split_boundary(RowCount(features.shape[0]), _ATTACK_FRACTION)
        for subtype, features in attacks.items()
    }
    scaler = fit_feature_scaler(benign)
    torch.manual_seed(0)
    model = Autoencoder(AutoencoderArchitecture(n_features=4, hidden_dims=(3,)))
    return generate_score_artifact(
        ClientId("1"),
        "n-baiot",
        DetectorSeed(0),
        raw,
        benign_boundaries,
        attack_boundaries,
        scaler,
        model,
    )


def test_build_client_frontier_inputs_shapes() -> None:
    artifact = _artifact()
    inputs = build_client_frontier_inputs(artifact, _GRID)
    assert len(inputs.subtype_confusion_by_candidate) == len(_GRID)
    assert set(inputs.subtype_validation_row_counts.keys()) == {
        AttackSubtype("mirai_scan"),
        AttackSubtype("bashlite_udp"),
    }
    assert inputs.benign_frontier_scores.shape[0] > 0


def test_confusion_counts_are_monotone_in_alpha() -> None:
    artifact = _artifact()
    inputs = build_client_frontier_inputs(artifact, _GRID)
    tp_by_alpha = [
        counts[AttackSubtype("mirai_scan")].true_positive
        for counts in inputs.subtype_confusion_by_candidate
    ]
    for lower, higher in zip(tp_by_alpha, tp_by_alpha[1:], strict=False):
        assert higher >= lower


def test_final_cal_and_test_scores_nonempty() -> None:
    artifact = _artifact()
    assert benign_final_cal_scores(artifact).shape[0] > 0
    assert benign_test_scores(artifact).shape[0] > 0
    attack_test = attack_test_scores_by_subtype(artifact)
    assert set(attack_test.keys()) == {AttackSubtype("mirai_scan"), AttackSubtype("bashlite_udp")}


def test_all_test_auroc_in_valid_range() -> None:
    artifact = _artifact()
    auroc = all_test_auroc(artifact)
    assert 0.0 <= auroc <= 1.0
