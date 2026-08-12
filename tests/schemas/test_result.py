from __future__ import annotations

import pytest

from fabrid.evaluation.record_level import AttackSubtype, ClientId
from fabrid.schemas.allocation import AllocationPolicy
from fabrid.schemas.result import ResultRow, SolverStatus, WeightMode


def _row(**overrides: object) -> ResultRow:
    defaults: dict[str, object] = dict(
        experiment_id="exp-1",
        dataset_id="n-baiot",
        seed=0,
        budget_id="B_0.01",
        budget_value=0.01,
        weight_mode=WeightMode.EQUAL_CLIENT,
        policy=AllocationPolicy.EQ_FPR,
        client_id=ClientId("1"),
        alpha_selected=0.01,
        threshold=1.5,
        calibration_n=1000,
        nominal_weight=1 / 9,
        realized_weight=1 / 9,
        n_benign_test=500,
        n_attack_test=300,
        attack_subtype=AttackSubtype("mirai_scan"),
        true_positive=250,
        false_negative=50,
        false_positive=5,
        true_negative=495,
        fpr=0.01,
        tpr=0.833,
        macro_attack_recall=0.833,
        false_alert_count=5,
        solver_status=SolverStatus.NOT_APPLICABLE,
        solver_objective=None,
        solver_gap=None,
        solver_runtime_ms=None,
        model_sha256="a" * 64,
        score_sha256="b" * 64,
        split_sha256="c" * 64,
        feature_sha256="d" * 64,
        protocol_sha256="e" * 64,
        git_commit="deadbeef",
    )
    defaults.update(overrides)
    return ResultRow(**defaults)  # type: ignore[arg-type]


def test_valid_row_constructs() -> None:
    row = _row()
    assert row.policy is AllocationPolicy.EQ_FPR
    assert row.weight_mode is WeightMode.EQUAL_CLIENT


def test_negative_seed_rejected() -> None:
    with pytest.raises(ValueError):
        _row(seed=-1)


def test_negative_confusion_counts_rejected() -> None:
    with pytest.raises(ValueError):
        _row(true_positive=-1)
    with pytest.raises(ValueError):
        _row(false_positive=-1)


def test_out_of_range_rates_rejected() -> None:
    with pytest.raises(ValueError):
        _row(fpr=1.5)
    with pytest.raises(ValueError):
        _row(tpr=-0.1)
