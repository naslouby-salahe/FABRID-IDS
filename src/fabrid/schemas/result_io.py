"""Persist `ResultRow` records as CSV — the artifact format for generated tables/figures.

The column order matches `ResultRow`'s field order exactly, so this file is
the single place that defines the on-disk schema; nothing downstream should
reconstruct or reorder columns by hand.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import asdict, fields
from pathlib import Path

from fabrid.evaluation.record_level import AttackSubtype, ClientId
from fabrid.schemas.allocation import AllocationPolicy
from fabrid.schemas.result import ResultRow, SolverStatus, WeightMode

_FIELD_NAMES = tuple(f.name for f in fields(ResultRow))


def _serialize_value(value: object) -> str:
    # StrEnum's __str__ already yields its plain value; ClientId/AttackSubtype are
    # NewType(str, ...) so str() is their identity. None becomes the empty string.
    return "" if value is None else str(value)


def write_result_rows_csv(rows: Sequence[ResultRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_FIELD_NAMES)
        for row in rows:
            values = asdict(row)
            writer.writerow([_serialize_value(values[name]) for name in _FIELD_NAMES])


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _row_from_csv_dict(record: dict[str, str]) -> ResultRow:
    return ResultRow(
        experiment_id=record["experiment_id"],
        dataset_id=record["dataset_id"],
        seed=int(record["seed"]),
        budget_id=record["budget_id"],
        budget_value=float(record["budget_value"]),
        weight_mode=WeightMode(record["weight_mode"]),
        policy=AllocationPolicy(record["policy"]),
        client_id=ClientId(record["client_id"]),
        alpha_selected=float(record["alpha_selected"]),
        threshold=float(record["threshold"]),
        calibration_n=int(record["calibration_n"]),
        nominal_weight=float(record["nominal_weight"]),
        realized_weight=float(record["realized_weight"]),
        n_benign_test=int(record["n_benign_test"]),
        n_attack_test=int(record["n_attack_test"]),
        attack_subtype=AttackSubtype(record["attack_subtype"])
        if record["attack_subtype"]
        else None,
        true_positive=int(record["true_positive"]),
        false_negative=int(record["false_negative"]),
        false_positive=int(record["false_positive"]),
        true_negative=int(record["true_negative"]),
        fpr=float(record["fpr"]),
        tpr=float(record["tpr"]),
        macro_attack_recall=float(record["macro_attack_recall"]),
        false_alert_count=int(record["false_alert_count"]),
        solver_status=SolverStatus(record["solver_status"]),
        solver_objective=_optional_float(record["solver_objective"]),
        solver_gap=_optional_float(record["solver_gap"]),
        solver_runtime_ms=_optional_float(record["solver_runtime_ms"]),
        model_sha256=record["model_sha256"],
        score_sha256=record["score_sha256"],
        split_sha256=record["split_sha256"],
        feature_sha256=record["feature_sha256"],
        protocol_sha256=record["protocol_sha256"],
        git_commit=record["git_commit"],
    )


def read_result_rows_csv(path: Path) -> list[ResultRow]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [_row_from_csv_dict(record) for record in reader]
