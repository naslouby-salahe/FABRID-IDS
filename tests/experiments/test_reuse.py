from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import TypeAdapter

from fabrid.artifacts.json import write_typed_json
from fabrid.artifacts.paths import ArtifactPaths
from fabrid.config import DatasetId, PreprocessingStatus
from fabrid.datasets.registry import ClientPopulation
from fabrid.execution.prerequisites import (
    PreparedFederation,
    PreprocessingProvenance,
    checkpoint_matches_config,
    resolve_preprocessing,
)
from tests.support import isolated_paths


def _paths(tmp_path: Path) -> ArtifactPaths:
    return isolated_paths(tmp_path, raw_data_root=tmp_path / "data" / "raw")


def _provenance(
    dataset_id: DatasetId,
    raw_digest: str,
    config_digest: str,
    *,
    feature_manifest_digest: str = "0" * 64,
    split_manifest_digest: str = "1" * 64,
    features_digest: str = "e" * 64,
) -> PreprocessingProvenance:
    return PreprocessingProvenance(
        dataset_id=dataset_id,
        schema_version="2.0",
        raw_digest=raw_digest,
        preprocessing_config_digest=config_digest,
        feature_manifest_digest=feature_manifest_digest,
        split_manifest_digest=split_manifest_digest,
        features_digest=features_digest,
        built_at="2026-08-14T00:00:00Z",
    )


def _write_valid_preprocessing(paths: ArtifactPaths, dataset_id: DatasetId) -> None:
    paths.preprocessing_dir(dataset_id).mkdir(parents=True, exist_ok=True)
    write_typed_json(
        _provenance(dataset_id, "a" * 64, "b" * 64),
        TypeAdapter(PreprocessingProvenance),
        paths.preprocessing_provenance_path(dataset_id),
    )
    paths.preprocessing_features_path(dataset_id).write_bytes(b"parquet")
    paths.preprocessing_split_manifest_path(dataset_id).write_bytes(b"manifest")


def test_split_manifest_enforces_disjoint_boundary_contract() -> None:
    from fabrid.validation.datasets import (
        SplitAttackBoundaryManifest,
        SplitClientManifest,
        SplitManifest,
    )

    valid_attack = SplitAttackBoundaryManifest(
        subtype="mirai_ack", validation_end=50, total_rows=100
    )
    valid_client = SplitClientManifest(
        client_id="a",
        benign_train_end=100,
        benign_frontier_end=200,
        benign_final_cal_end=300,
        benign_total_rows=400,
        attacks=(valid_attack,),
    )
    SplitManifest(dataset_id=DatasetId.NBAIOT, clients=(valid_client,))
    with pytest.raises(ValueError):
        SplitClientManifest(
            client_id="a",
            benign_train_end=200,
            benign_frontier_end=100,
            benign_final_cal_end=300,
            benign_total_rows=400,
            attacks=(valid_attack,),
        )
    with pytest.raises(ValueError):
        SplitAttackBoundaryManifest(subtype="mirai_ack", validation_end=150, total_rows=100)
    duplicate_subtype = SplitAttackBoundaryManifest(
        subtype="mirai_ack", validation_end=60, total_rows=100
    )
    with pytest.raises(ValueError):
        SplitClientManifest(
            client_id="a",
            benign_train_end=100,
            benign_frontier_end=200,
            benign_final_cal_end=300,
            benign_total_rows=400,
            attacks=(valid_attack, duplicate_subtype),
        )
    empty_client = SplitClientManifest(
        client_id="a",
        benign_train_end=100,
        benign_frontier_end=200,
        benign_final_cal_end=300,
        benign_total_rows=400,
        attacks=(),
    )
    with pytest.raises(ValueError):
        SplitManifest(dataset_id=DatasetId.NBAIOT, clients=(empty_client, empty_client))


def test_scores_reuse_gate_requires_matching_provenance(tmp_path: Path) -> None:
    import json

    from fabrid.execution.prerequisites import ScoreProvenance, scores_reuse_valid

    paths = _paths(tmp_path)
    dataset_id = DatasetId.NBAIOT
    expected = ScoreProvenance(
        checkpoint_digest="0" * 64,
        preprocessing_digest="1" * 64,
        scoring_config_digest="2" * 64,
    )
    assert not scores_reuse_valid(paths, dataset_id, 0, expected)
    provenance_path = paths.score_provenance_path(dataset_id, 0)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_bytes(
        json.dumps(
            {
                "checkpoint_digest": expected.checkpoint_digest,
                "preprocessing_digest": expected.preprocessing_digest,
                "scoring_config_digest": expected.scoring_config_digest,
            },
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )
    assert scores_reuse_valid(paths, dataset_id, 0, expected)
    changed_checkpoint = ScoreProvenance(
        checkpoint_digest="f" * 64,
        preprocessing_digest="1" * 64,
        scoring_config_digest="2" * 64,
    )
    assert not scores_reuse_valid(paths, dataset_id, 0, changed_checkpoint)


def test_scores_reuse_gate_rejects_preprocessing_digest_drift(tmp_path: Path) -> None:
    import json

    from fabrid.execution.prerequisites import ScoreProvenance, scores_reuse_valid

    paths = _paths(tmp_path)
    dataset_id = DatasetId.NBAIOT
    stored = ScoreProvenance(
        checkpoint_digest="0" * 64,
        preprocessing_digest="1" * 64,
        scoring_config_digest="2" * 64,
    )
    provenance_path = paths.score_provenance_path(dataset_id, 0)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_bytes(
        json.dumps(
            {
                "checkpoint_digest": stored.checkpoint_digest,
                "preprocessing_digest": stored.preprocessing_digest,
                "scoring_config_digest": stored.scoring_config_digest,
            }
        ).encode("utf-8")
    )
    changed_preprocessing = stored.model_copy(update={"preprocessing_digest": "9" * 64})
    assert not scores_reuse_valid(paths, dataset_id, 0, changed_preprocessing)


def test_scores_reuse_gate_rejects_scoring_config_drift(tmp_path: Path) -> None:
    import json

    from fabrid.execution.prerequisites import ScoreProvenance, scores_reuse_valid

    paths = _paths(tmp_path)
    dataset_id = DatasetId.NBAIOT
    stored = ScoreProvenance(
        checkpoint_digest="0" * 64,
        preprocessing_digest="1" * 64,
        scoring_config_digest="2" * 64,
    )
    provenance_path = paths.score_provenance_path(dataset_id, 0)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_bytes(
        json.dumps(
            {
                "checkpoint_digest": stored.checkpoint_digest,
                "preprocessing_digest": stored.preprocessing_digest,
                "scoring_config_digest": stored.scoring_config_digest,
            }
        ).encode("utf-8")
    )
    changed_scoring = stored.model_copy(update={"scoring_config_digest": "9" * 64})
    assert not scores_reuse_valid(paths, dataset_id, 0, changed_scoring)


def test_preprocessing_build_when_missing(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    status = resolve_preprocessing(
        paths,
        DatasetId.NBAIOT,
        "a" * 64,
        "b" * 64,
        "2.0",
        overwrite=False,
    )
    assert status is PreprocessingStatus.BUILD


def test_preprocessing_reuse_when_provenance_matches(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_valid_preprocessing(paths, DatasetId.NBAIOT)
    status = resolve_preprocessing(
        paths,
        DatasetId.NBAIOT,
        "a" * 64,
        "b" * 64,
        "2.0",
        overwrite=False,
    )
    assert status is PreprocessingStatus.REUSE


def test_preprocessing_invalidates_on_raw_change(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_valid_preprocessing(paths, DatasetId.NBAIOT)
    status = resolve_preprocessing(
        paths,
        DatasetId.NBAIOT,
        "c" * 64,
        "b" * 64,
        "2.0",
        overwrite=False,
    )
    assert status is PreprocessingStatus.REBUILD


def test_preprocessing_invalidates_on_config_change(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_valid_preprocessing(paths, DatasetId.NBAIOT)
    status = resolve_preprocessing(
        paths,
        DatasetId.NBAIOT,
        "a" * 64,
        "d" * 64,
        "2.0",
        overwrite=False,
    )
    assert status is PreprocessingStatus.REBUILD


def test_preprocessing_invalidates_on_protocol_version_change(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    _write_valid_preprocessing(paths, DatasetId.NBAIOT)
    status = resolve_preprocessing(
        paths,
        DatasetId.NBAIOT,
        "a" * 64,
        "b" * 64,
        "3.0",
        overwrite=False,
    )
    assert status is PreprocessingStatus.REBUILD


def test_preprocessing_invalidates_on_unreadable_provenance(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.preprocessing_dir(DatasetId.NBAIOT).mkdir(parents=True, exist_ok=True)
    paths.preprocessing_provenance_path(DatasetId.NBAIOT).write_text("{not json", encoding="utf-8")
    status = resolve_preprocessing(
        paths,
        DatasetId.NBAIOT,
        "a" * 64,
        "b" * 64,
        "2.0",
        overwrite=False,
    )
    assert status is PreprocessingStatus.REBUILD


def test_preprocessing_builds_when_split_manifest_missing(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.preprocessing_dir(DatasetId.NBAIOT).mkdir(parents=True, exist_ok=True)
    write_typed_json(
        _provenance(DatasetId.NBAIOT, "a" * 64, "b" * 64),
        TypeAdapter(PreprocessingProvenance),
        paths.preprocessing_provenance_path(DatasetId.NBAIOT),
    )
    paths.preprocessing_features_path(DatasetId.NBAIOT).write_bytes(b"parquet")
    status = resolve_preprocessing(
        paths,
        DatasetId.NBAIOT,
        "a" * 64,
        "b" * 64,
        "2.0",
        overwrite=False,
    )
    assert status is PreprocessingStatus.BUILD


def test_preprocessing_explicit_overwrite_rebuilds(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    status = resolve_preprocessing(
        paths,
        DatasetId.NBAIOT,
        "a" * 64,
        "b" * 64,
        "2.0",
        overwrite=True,
    )
    assert status is PreprocessingStatus.REBUILD


def test_preprocessing_relevant_config_ignores_unrelated_protocol_fields() -> None:
    from fabrid.validation.datasets import preprocessing_relevant_config
    from tests.support import smoke_application

    application = smoke_application()
    baseline = preprocessing_relevant_config(application).model_dump_json()
    new_time_limit = application.protocol.solver.time_limit_seconds + 5
    unrelated = application.model_copy(
        update={
            "protocol": application.protocol.model_copy(
                update={
                    "solver": application.protocol.solver.model_copy(
                        update={"time_limit_seconds": new_time_limit}
                    )
                }
            )
        }
    )
    assert preprocessing_relevant_config(unrelated).model_dump_json() == baseline
    relevant = application.model_copy(
        update={
            "protocol": application.protocol.model_copy(
                update={
                    "benign_splits": application.protocol.benign_splits.model_copy(
                        update={"train_end": application.protocol.benign_splits.train_end / 2}
                    )
                }
            )
        }
    )
    assert preprocessing_relevant_config(relevant).model_dump_json() != baseline


def test_checkpoint_reuse_requires_matching_protocol() -> None:
    from fabrid.detector.autoencoder import AutoencoderArchitecture
    from fabrid.detector.training import (
        CheckpointMetadata,
        DetectorInputSpace,
        FederatedTrainingSettings,
    )
    from fabrid.validation.datasets import FeatureManifest, SplitManifest

    manifest = FeatureManifest(dataset_id=DatasetId.NBAIOT, columns=("f0", "f1", "f2"))
    prepared = PreparedFederation(
        population=ClientPopulation(("a",)),
        feature_manifest=manifest,
        split_manifest=SplitManifest(dataset_id=DatasetId.NBAIOT, clients=()),
        provenance=_provenance(DatasetId.NBAIOT, "a" * 64, "b" * 64),
        raw_digest="c" * 64,
        config_digest="d" * 64,
    )
    training = FederatedTrainingSettings(
        learning_rate=0.001, local_epochs=1, rounds=2, batch_size=128
    )
    saved = CheckpointMetadata(
        dataset_id=DatasetId.NBAIOT,
        detector_seed=0,
        feature_count=3,
        hidden_layers=(8, 4),
        input_space=DetectorInputSpace.PER_CLIENT_ZSCORE,
        training=training,
        preprocessing_digest=prepared.provenance.combined_digest,
        model_digest="0" * 64,
        scaler_digests=(),
    )
    architecture = AutoencoderArchitecture(feature_count=3, hidden_layers=(8, 4))
    assert checkpoint_matches_config(saved, prepared, 0, architecture, training)
    assert not checkpoint_matches_config(
        saved.model_copy(update={"hidden_layers": (16, 4)}),
        prepared,
        0,
        architecture,
        training,
    )
    assert not checkpoint_matches_config(
        saved.model_copy(update={"feature_count": 2}), prepared, 0, architecture, training
    )
    assert not checkpoint_matches_config(
        saved.model_copy(update={"dataset_id": DatasetId.GOTHAM}),
        prepared,
        0,
        architecture,
        training,
    )
    assert not checkpoint_matches_config(saved, prepared, 1, architecture, training)
    assert not checkpoint_matches_config(
        saved,
        prepared,
        0,
        architecture,
        FederatedTrainingSettings(learning_rate=0.01, local_epochs=1, rounds=2, batch_size=128),
    )
