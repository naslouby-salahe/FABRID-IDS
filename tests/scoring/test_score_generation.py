from __future__ import annotations

import numpy as np
import pytest
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
from fabrid.schemas.score_artifact import DetectorSeed, Label
from fabrid.scoring.score_generation import generate_score_artifact

_BENIGN_FRACTIONS = BenignSplitFractions(0.5, 0.7, 0.8)
_ATTACK_FRACTION = AttackSplitFraction(0.2)


def _raw_data(seed: int, n_benign: int, attack_counts: dict[str, int]) -> RawDeviceData:
    rng = np.random.default_rng(seed)
    benign = rng.normal(0, 0.1, size=(n_benign, 4)).astype(np.float64)
    attacks = {
        AttackSubtype(name): rng.normal(1.0, 0.1, size=(n, 4)).astype(np.float64)
        for name, n in attack_counts.items()
    }
    return RawDeviceData(
        client_id=ClientId("1"), benign_features=benign, attack_features_by_subtype=attacks
    )


def test_generates_records_for_benign_and_all_attack_subtypes() -> None:
    raw = _raw_data(0, n_benign=20, attack_counts={"mirai_scan": 10, "bashlite_udp": 8})
    benign_boundaries = compute_benign_split_boundaries(RowCount(20), _BENIGN_FRACTIONS)
    attack_boundaries = {
        AttackSubtype("mirai_scan"): compute_attack_split_boundary(RowCount(10), _ATTACK_FRACTION),
        AttackSubtype("bashlite_udp"): compute_attack_split_boundary(RowCount(8), _ATTACK_FRACTION),
    }
    scaler = fit_feature_scaler(raw.benign_features)
    torch.manual_seed(0)
    model = Autoencoder(AutoencoderArchitecture(n_features=4, hidden_dims=(3,)))

    artifact = generate_score_artifact(
        ClientId("1"),
        "n-baiot",
        DetectorSeed(0),
        raw,
        benign_boundaries,
        attack_boundaries,
        scaler,
        model,
    )

    assert len(artifact.records) == 20 + 10 + 8
    benign_records = [r for r in artifact.records if r.label is Label.BENIGN]
    attack_records = [r for r in artifact.records if r.label is Label.ATTACK]
    assert len(benign_records) == 20
    assert len(attack_records) == 18
    assert all(r.attack_type is None for r in benign_records)
    assert all(r.attack_type is not None for r in attack_records)


def test_skips_subtype_absent_from_raw_data() -> None:
    raw = _raw_data(1, n_benign=10, attack_counts={"mirai_scan": 5})
    benign_boundaries = compute_benign_split_boundaries(RowCount(10), _BENIGN_FRACTIONS)
    attack_boundaries = {
        AttackSubtype("mirai_scan"): compute_attack_split_boundary(RowCount(5), _ATTACK_FRACTION),
        AttackSubtype("mirai_udp"): compute_attack_split_boundary(RowCount(5), _ATTACK_FRACTION),
    }
    scaler = fit_feature_scaler(raw.benign_features)
    torch.manual_seed(0)
    model = Autoencoder(AutoencoderArchitecture(n_features=4, hidden_dims=(3,)))

    artifact = generate_score_artifact(
        ClientId("1"),
        "n-baiot",
        DetectorSeed(0),
        raw,
        benign_boundaries,
        attack_boundaries,
        scaler,
        model,
    )
    subtypes_present = {r.attack_type for r in artifact.records if r.attack_type is not None}
    assert subtypes_present == {AttackSubtype("mirai_scan")}


def test_mismatched_benign_row_count_rejected() -> None:
    raw = _raw_data(2, n_benign=10, attack_counts={})
    wrong_boundaries = compute_benign_split_boundaries(RowCount(999), _BENIGN_FRACTIONS)
    scaler = fit_feature_scaler(raw.benign_features)
    torch.manual_seed(0)
    model = Autoencoder(AutoencoderArchitecture(n_features=4, hidden_dims=(3,)))
    with pytest.raises(ValueError):
        generate_score_artifact(
            ClientId("1"), "n-baiot", DetectorSeed(0), raw, wrong_boundaries, {}, scaler, model
        )


def test_sample_ids_are_unique() -> None:
    raw = _raw_data(3, n_benign=15, attack_counts={"mirai_scan": 6})
    benign_boundaries = compute_benign_split_boundaries(RowCount(15), _BENIGN_FRACTIONS)
    attack_boundaries = {
        AttackSubtype("mirai_scan"): compute_attack_split_boundary(RowCount(6), _ATTACK_FRACTION),
    }
    scaler = fit_feature_scaler(raw.benign_features)
    torch.manual_seed(0)
    model = Autoencoder(AutoencoderArchitecture(n_features=4, hidden_dims=(3,)))
    artifact = generate_score_artifact(
        ClientId("1"),
        "n-baiot",
        DetectorSeed(0),
        raw,
        benign_boundaries,
        attack_boundaries,
        scaler,
        model,
    )
    sample_ids = [r.sample_id for r in artifact.records]
    assert len(sample_ids) == len(set(sample_ids))
