from __future__ import annotations

from pathlib import Path

from fabrid.evaluation.record_level import AttackSubtype, ClientId
from fabrid.schemas.allocation import AllocationPolicy
from fabrid.schemas.result import ResultRow, SolverStatus, WeightMode
from fabrid.schemas.result_io import read_result_rows_csv, write_result_rows_csv


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


def test_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    rows = [_row(client_id=ClientId("1")), _row(client_id=ClientId("2"), attack_subtype=None)]
    path = tmp_path / "results.csv"
    write_result_rows_csv(rows, path)
    loaded = read_result_rows_csv(path)
    assert loaded == rows


def test_none_solver_fields_round_trip(tmp_path: Path) -> None:
    row = _row(solver_objective=1.5, solver_gap=0.0, solver_runtime_ms=42.0)
    path = tmp_path / "results.csv"
    write_result_rows_csv([row], path)
    loaded = read_result_rows_csv(path)
    assert loaded[0].solver_objective == 1.5
    assert loaded[0].solver_gap == 0.0
    assert loaded[0].solver_runtime_ms == 42.0


def test_empty_rows_writes_header_only(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    write_result_rows_csv([], path)
    assert read_result_rows_csv(path) == []
    assert "experiment_id" in path.read_text()


def test_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "results.csv"
    write_result_rows_csv([_row()], path)
    assert path.exists()
