from __future__ import annotations

import numpy as np
import pytest

from fabrid.config import (
    AttackSplit,
    AttackSplitConfig,
    BenignSplit,
    BenignSplitConfig,
    Label,
)
from fabrid.datasets.registry import (
    AttackFeatureBlock,
    AttackSplitBoundary,
    AttackSplitCounts,
    BenignSplitBoundaries,
    BenignSplitCounts,
    DeviceDataset,
    FeatureMatrix,
    compute_attack_split_boundary,
    compute_benign_split_boundaries,
    plan_device_splits,
)


def test_benign_boundaries_cover_population_without_duplication() -> None:
    boundaries = compute_benign_split_boundaries(
        total_rows=1000, train_end=0.1, frontier_end=0.5, final_cal_end=0.6
    )
    counts = boundaries.counts()
    assert counts.total == 1000
    assert counts.train == 100
    assert counts.frontier == 400
    assert counts.final_cal == 100
    assert counts.test == 400
    for index in range(1000):
        assert boundaries.split_of(index) is not None


def test_benign_floor_rule_boundaries_inside_population() -> None:
    boundaries = compute_benign_split_boundaries(
        total_rows=9, train_end=0.1, frontier_end=0.5, final_cal_end=0.9
    )
    assert boundaries.train_end < boundaries.frontier_end < boundaries.final_cal_end
    assert boundaries.final_cal_end <= 9
    counts = boundaries.counts()
    assert counts.total == 9
    assert counts.train >= 0
    assert counts.frontier >= 0
    assert counts.final_cal >= 0
    assert counts.test >= 0


def test_benign_split_of_classifies_row_indices() -> None:
    boundaries = BenignSplitBoundaries(train_end=2, frontier_end=5, final_cal_end=7, total_rows=10)
    assert boundaries.split_of(0) is BenignSplit.TRAIN
    assert boundaries.split_of(1) is BenignSplit.TRAIN
    assert boundaries.split_of(2) is BenignSplit.FRONTIER
    assert boundaries.split_of(4) is BenignSplit.FRONTIER
    assert boundaries.split_of(5) is BenignSplit.FINAL_CAL
    assert boundaries.split_of(6) is BenignSplit.FINAL_CAL
    assert boundaries.split_of(7) is BenignSplit.TEST
    assert boundaries.split_of(9) is BenignSplit.TEST
    with pytest.raises(ValueError):
        boundaries.split_of(10)


def test_benign_boundaries_reject_non_monotonic() -> None:
    with pytest.raises(ValueError):
        BenignSplitBoundaries(train_end=5, frontier_end=2, final_cal_end=7, total_rows=10)


def test_attack_split_counts_and_classification() -> None:
    boundary = compute_attack_split_boundary(total_rows=100, validation_end=0.2)
    counts = boundary.counts()
    assert counts == AttackSplitCounts(validation=20, test=80)
    assert counts.total == 100
    assert boundary.split_of(0) is AttackSplit.VALIDATION
    assert boundary.split_of(19) is AttackSplit.VALIDATION
    assert boundary.split_of(20) is AttackSplit.TEST
    assert boundary.split_of(99) is AttackSplit.TEST
    with pytest.raises(ValueError):
        boundary.split_of(100)


def test_attack_boundary_validation() -> None:
    with pytest.raises(ValueError):
        AttackSplitBoundary(validation_end=101, total_rows=100)
    with pytest.raises(ValueError):
        AttackSplitBoundary(validation_end=5, total_rows=-1)


def test_label_enum_values() -> None:
    assert Label.BENIGN.value == "benign"
    assert Label.ATTACK.value == "attack"


def test_benign_split_counts_properties() -> None:
    counts = BenignSplitCounts(train=10, frontier=20, final_cal=30, test=40)
    assert counts.total == 100


def test_plan_device_splits_matches_floor_boundaries() -> None:
    device = DeviceDataset(
        client_id="device",
        benign_source_file="benign.csv",
        benign=FeatureMatrix(np.zeros((1000, 2))),
        attacks=(
            AttackFeatureBlock(
                subtype="scan",
                source_file="scan.csv",
                features=FeatureMatrix(np.zeros((100, 2))),
            ),
        ),
    )
    plan = plan_device_splits(
        device,
        BenignSplitConfig(train_end=0.1, frontier_end=0.5, final_cal_end=0.6),
        AttackSplitConfig(validation_end=0.2),
    )
    assert plan.benign.counts() == BenignSplitCounts(
        train=100, frontier=400, final_cal=100, test=400
    )
    assert plan.attack_boundary("scan").counts() == AttackSplitCounts(validation=20, test=80)
