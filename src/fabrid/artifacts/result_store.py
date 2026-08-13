from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from fabrid.artifacts.digests import digest_file
from fabrid.domain.identifiers import ArtifactDigest
from fabrid.evaluation.results import ClientResultRecord

_RESULT_SCHEMA = (
    ("experiment_id", pl.String),
    ("dataset_id", pl.String),
    ("seed", pl.Int64),
    ("budget_id", pl.String),
    ("budget_value", pl.Float64),
    ("weight_mode", pl.String),
    ("policy", pl.String),
    ("client_id", pl.String),
    ("alpha_selected", pl.Float64),
    ("threshold", pl.Float64),
    ("calibration_n", pl.Int64),
    ("nominal_weight", pl.Float64),
    ("realized_weight", pl.Float64),
    ("n_benign_test", pl.Int64),
    ("n_attack_test", pl.Int64),
    ("attack_subtype", pl.String),
    ("true_positive", pl.Int64),
    ("false_negative", pl.Int64),
    ("false_positive", pl.Int64),
    ("true_negative", pl.Int64),
    ("fpr", pl.Float64),
    ("tpr", pl.Float64),
    ("macro_attack_recall", pl.Float64),
    ("false_alert_count", pl.Int64),
    ("solver_status", pl.String),
    ("solver_objective", pl.Float64),
    ("solver_gap", pl.Float64),
    ("solver_runtime_ms", pl.Float64),
    ("model_sha256", pl.String),
    ("score_sha256", pl.String),
    ("split_sha256", pl.String),
    ("feature_sha256", pl.String),
    ("protocol_sha256", pl.String),
    ("git_commit", pl.String),
)


@dataclass(frozen=True, slots=True)
class StoredResultArtifact:
    digest: ArtifactDigest
    path: Path


def _serialize(record: ClientResultRecord) -> tuple[object, ...]:
    experiment = record.allocation.experiment
    solver = record.solver
    return (
        experiment.experiment_id.value,
        experiment.dataset_id.value,
        experiment.detector_seed.value,
        experiment.budget_id.value,
        experiment.budget.value,
        experiment.weight_mode.value,
        record.allocation.policy.value,
        record.client_id.value,
        record.calibration.target_rate.value,
        record.calibration.threshold.value,
        record.calibration.calibration_count.value,
        record.weights.nominal.value,
        record.weights.realized.value,
        record.benign_test_count.value,
        record.attack_test_count.value,
        None if record.attack_subtype is None else record.attack_subtype.value,
        record.confusion.true_positive.value,
        record.confusion.false_negative.value,
        record.confusion.false_positive.value,
        record.confusion.true_negative.value,
        record.metrics.false_positive_rate.value,
        record.metrics.true_positive_rate.value,
        record.metrics.macro_attack_recall.value,
        record.metrics.false_alert_count.value,
        solver.status.value,
        None if solver.final_objective is None else solver.final_objective.value,
        None if solver.final_gap is None else solver.final_gap.value,
        None if solver.total_runtime is None else solver.total_runtime.value,
        record.provenance.detector.model.value,
        record.provenance.scores.score_bundle_digest().value,
        record.provenance.scores.split_manifest.value,
        record.provenance.detector.feature_manifest.value,
        record.provenance.scores.protocol.value,
        record.provenance.git_commit.value,
    )


def write_result_records(
    records: tuple[ClientResultRecord, ...],
    path: Path,
) -> StoredResultArtifact:
    if not records:
        raise ValueError("result artifact requires at least one completed policy record")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        [_serialize(record) for record in records],
        schema=_RESULT_SCHEMA,
        orient="row",
    )
    frame.write_parquet(path, compression="zstd", statistics=True)
    return StoredResultArtifact(digest=digest_file(path), path=path)
