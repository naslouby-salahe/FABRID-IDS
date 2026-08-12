"""T01 against the real score-generation pipeline: no sample_id crosses partitions."""

from __future__ import annotations

import numpy as np
import torch

from fabrid.audit.split_leakage import check_partition_exclusivity
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
from fabrid.scoring.score_generation import generate_score_artifact

_BENIGN_FRACTIONS = BenignSplitFractions(0.5, 0.7, 0.8)
_ATTACK_FRACTION = AttackSplitFraction(0.2)


def test_generated_artifact_has_no_cross_partition_sample_id_collisions() -> None:
    rng = np.random.default_rng(0)
    benign = rng.normal(0, 0.1, size=(97, 4)).astype(np.float64)
    attacks = {
        AttackSubtype("mirai_scan"): rng.normal(1.0, 0.1, size=(53, 4)).astype(np.float64),
        AttackSubtype("bashlite_udp"): rng.normal(1.0, 0.1, size=(31, 4)).astype(np.float64),
    }
    raw = RawDeviceData(
        client_id=ClientId("1"), benign_features=benign, attack_features_by_subtype=attacks
    )

    benign_boundaries = compute_benign_split_boundaries(RowCount(97), _BENIGN_FRACTIONS)
    attack_boundaries = {
        subtype: compute_attack_split_boundary(RowCount(features.shape[0]), _ATTACK_FRACTION)
        for subtype, features in attacks.items()
    }
    scaler = fit_feature_scaler(benign)
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

    sample_ids_by_partition: dict[str, list[str]] = {}
    for record in artifact.records:
        sample_ids_by_partition.setdefault(record.split_id.value, []).append(record.sample_id)

    # 6 distinct partitions expected: 4 benign + 2 attack.
    assert len(sample_ids_by_partition) == 6
    check_partition_exclusivity(sample_ids_by_partition)
