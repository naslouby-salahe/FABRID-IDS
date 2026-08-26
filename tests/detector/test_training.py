from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from fabrid.artifacts.checkpoints import load_scaler, save_scaler
from fabrid.artifacts.json import digest_file
from fabrid.artifacts.paths import DetectorCoordinate, ScoreCoordinate
from fabrid.config import (
    BatchSize,
    BenignSplit,
    DatasetId,
    FederatedRoundCount,
    Label,
    LearningRate,
    LocalEpochCount,
)
from fabrid.datasets.registry import FeatureMatrix
from fabrid.detector.autoencoder import Autoencoder, AutoencoderArchitecture, resolve_device
from fabrid.detector.scoring import (
    ScorePartitionArtifact,
    ScoreRecord,
    load_score_partition,
    persist_score_partition,
    require_exclusive_sample_ids,
    score_feature_matrix,
)
from fabrid.detector.training import (
    CheckpointMetadata,
    ClientScaler,
    ClientTrainingData,
    DetectorInputSpace,
    FeatureScaler,
    FederatedScalers,
    FederatedTrainingData,
    FederatedTrainingSettings,
    fit_feature_scaler,
    load_detector_checkpoint,
    save_detector_checkpoint,
    train_federated_autoencoder,
)
from tests.support import isolated_paths


def _settings(
    *,
    learning_rate: LearningRate = 0.01,
    local_epochs: LocalEpochCount = 2,
    rounds: FederatedRoundCount = 2,
    batch_size: BatchSize = 16,
) -> FederatedTrainingSettings:
    return FederatedTrainingSettings(
        learning_rate=learning_rate,
        local_epochs=local_epochs,
        rounds=rounds,
        batch_size=batch_size,
    )


def test_fit_feature_scaler_and_transform() -> None:
    features = FeatureMatrix(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]))
    scaler = fit_feature_scaler(features)
    transformed = scaler.transform(features)
    assert transformed.values.shape == features.values.shape
    column_means = transformed.values.mean(axis=0)
    assert np.allclose(column_means, 0.0, atol=1e-9)
    column_stds = transformed.values.std(axis=0)
    assert np.allclose(column_stds, 1.0, atol=1e-9)


def test_scaler_rejects_degenerate_statistics() -> None:
    with pytest.raises(ValueError):
        FeatureScaler(mean=np.zeros(2), standard_deviation=np.zeros(2))
    with pytest.raises(ValueError):
        FeatureScaler(mean=np.zeros(2), standard_deviation=np.zeros(3))


def test_client_training_weight_is_row_count() -> None:
    features = FeatureMatrix(np.ones((7, 3)))
    client = ClientTrainingData(client_id="a", features=features)
    assert client.weight == 7
    assert client.weight == features.row_count


def test_client_training_data_rejects_empty_matrix() -> None:
    empty = FeatureMatrix(np.empty((0, 3)))
    with pytest.raises(ValueError, match="at least one row"):
        ClientTrainingData(client_id="a", features=empty)


def test_federated_training_data_rejects_duplicate_clients() -> None:
    features = FeatureMatrix(np.ones((2, 3)))
    first = ClientTrainingData(client_id="a", features=features)
    duplicate = ClientTrainingData(client_id="a", features=features)
    with pytest.raises(ValueError, match="duplicate clients"):
        FederatedTrainingData(clients=(first, duplicate))


def test_train_federated_autoencoder_converges_on_toy_data() -> None:
    rng = np.random.default_rng(0)
    benign = FeatureMatrix(rng.normal(size=(200, 8)))
    attack = FeatureMatrix(rng.normal(loc=3.0, size=(200, 8)))
    training = FederatedTrainingData(
        clients=(
            ClientTrainingData(client_id="a", features=benign),
            ClientTrainingData(client_id="b", features=benign),
        )
    )
    model = train_federated_autoencoder(
        training,
        AutoencoderArchitecture(feature_count=8, hidden_layers=(4,)),
        _settings(local_epochs=3, rounds=3, batch_size=32),
        seed=1,
        device=resolve_device(),
    )
    device = resolve_device()
    benign_scores = score_feature_matrix(model, benign, device, batch_size=16)
    attack_scores = score_feature_matrix(model, attack, device, batch_size=16)
    assert float(np.median(attack_scores)) > float(np.median(benign_scores))


def test_train_federated_autoencoder_reports_only_completed_rounds() -> None:
    features = FeatureMatrix(np.ones((16, 4)))
    rounds: list[tuple[FederatedRoundCount, FederatedRoundCount]] = []
    train_federated_autoencoder(
        FederatedTrainingData(clients=(ClientTrainingData(client_id="a", features=features),)),
        AutoencoderArchitecture(feature_count=4, hidden_layers=(2,)),
        _settings(local_epochs=1, rounds=3, batch_size=8),
        seed=7,
        device=torch.device("cpu"),
        round_progress_reporter=lambda completed, total: rounds.append((completed, total)),
    )
    assert rounds == [(1, 3), (2, 3), (3, 3)]


def test_scale_then_train_has_lower_error_on_scaled_batch() -> None:
    rng = np.random.default_rng(11)
    raw = FeatureMatrix(rng.normal(loc=40.0, scale=8.0, size=(256, 6)))
    scaler = fit_feature_scaler(raw)
    scaled = scaler.transform(raw)
    model = train_federated_autoencoder(
        FederatedTrainingData(clients=(ClientTrainingData(client_id="a", features=scaled),)),
        AutoencoderArchitecture(feature_count=6, hidden_layers=(3,)),
        _settings(local_epochs=4, rounds=3, batch_size=32),
        seed=5,
        device=resolve_device(),
    )
    device = resolve_device()
    scaled_error = float(np.mean(score_feature_matrix(model, scaled, device, batch_size=32)))
    raw_error = float(np.mean(score_feature_matrix(model, raw, device, batch_size=32)))
    assert scaled_error < raw_error


def test_training_is_deterministic_given_seed() -> None:
    rng = np.random.default_rng(3)
    features = FeatureMatrix(rng.normal(size=(64, 6)))
    training = FederatedTrainingData(
        clients=(ClientTrainingData(client_id="a", features=features),)
    )
    architecture = AutoencoderArchitecture(feature_count=6, hidden_layers=(3,))
    settings = _settings()
    first = train_federated_autoencoder(
        training, architecture, settings, seed=9, device=torch.device("cpu")
    )
    second = train_federated_autoencoder(
        training, architecture, settings, seed=9, device=torch.device("cpu")
    )
    for first_tensor, second_tensor in zip(
        first.state_dict().values(), second.state_dict().values(), strict=True
    ):
        assert torch.equal(first_tensor, second_tensor)


def test_score_feature_matrix_empty_input() -> None:
    model = Autoencoder(AutoencoderArchitecture(feature_count=4, hidden_layers=(2,)))
    scores = score_feature_matrix(
        model, FeatureMatrix(np.empty((0, 4))), torch.device("cpu"), batch_size=16
    )
    assert scores.shape == (0,)


def _artifact(client_id: str, scores: np.ndarray) -> ScorePartitionArtifact:
    return ScorePartitionArtifact(
        coordinate=ScoreCoordinate(
            dataset_id=DatasetId.NBAIOT, detector_seed=0, client_id=client_id
        ),
        split=BenignSplit.FRONTIER,
        records=tuple(
            ScoreRecord(
                sample_id=f"{client_id}|frontier|{index}",
                source_file="f.csv",
                source_row=index,
                score=float(score),
                label=Label.BENIGN,
                attack_subtype=None,
                timestamp=None,
            )
            for index, score in enumerate(scores)
        ),
    )


def test_t01_duplicate_sample_id_across_partitions_fails() -> None:
    first = _artifact("a", np.array([0.1, 0.2]))
    second = ScorePartitionArtifact(
        coordinate=first.coordinate,
        split=BenignSplit.TEST,
        records=(
            ScoreRecord(
                sample_id=first.records[0].sample_id,
                source_file="f.csv",
                source_row=0,
                score=0.3,
                label=Label.BENIGN,
                attack_subtype=None,
                timestamp=None,
            ),
        ),
    )
    with pytest.raises(ValueError, match="multiple partitions"):
        require_exclusive_sample_ids((first, second))


def test_score_partition_round_trip(tmp_path: Path) -> None:
    paths = isolated_paths(tmp_path)
    artifact = _artifact("a", np.array([0.1, 0.2, 0.3]))
    coordinate = artifact.coordinate
    path = paths.score_path(coordinate, BenignSplit.FRONTIER)
    file_digest = persist_score_partition(path, artifact)
    assert file_digest == digest_file(path)
    loaded = load_score_partition(path, coordinate, BenignSplit.FRONTIER)
    assert loaded.records == artifact.records
    assert loaded.row_count == 3
    assert loaded.digest() == artifact.digest()


def test_save_load_detector_checkpoint_round_trip(tmp_path: Path) -> None:
    paths = isolated_paths(tmp_path)
    coordinate = DetectorCoordinate(dataset_id=DatasetId.NBAIOT, detector_seed=4)
    model = Autoencoder(AutoencoderArchitecture(feature_count=5, hidden_layers=(3, 2)))
    scaler = FeatureScaler(mean=np.zeros(5), standard_deviation=np.ones(5))
    scalers = FederatedScalers(
        clients=(
            ClientScaler(client_id="a", scaler=scaler),
            ClientScaler(client_id="b", scaler=scaler),
        )
    )
    metadata = save_detector_checkpoint(
        paths,
        coordinate,
        model,
        scalers,
        _settings(learning_rate=0.001, local_epochs=1, rounds=2, batch_size=16),
        "a" * 64,
    )
    assert metadata.detector_seed == 4
    assert metadata.input_space is DetectorInputSpace.PER_CLIENT_ZSCORE
    assert metadata.preprocessing_digest == "a" * 64
    loaded_model, loaded_scalers, loaded_metadata = load_detector_checkpoint(paths, coordinate)
    assert loaded_metadata == metadata
    for original, restored in zip(
        model.state_dict().values(), loaded_model.state_dict().values(), strict=True
    ):
        assert torch.equal(original, restored)
    assert loaded_scalers.for_client("a").mean.shape == (5,)


def test_save_load_scaler_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "scaler.safetensors"
    mean = np.array([1.0, 2.0])
    std = np.array([0.5, 0.5])
    save_scaler(mean, std, path)
    loaded_mean, loaded_std = load_scaler(path)
    assert np.array_equal(loaded_mean, mean)
    assert np.array_equal(loaded_std, std)


def test_checkpoint_metadata_requires_input_space() -> None:
    with pytest.raises(ValidationError, match="input_space"):
        CheckpointMetadata.model_validate(
            {
                "dataset_id": DatasetId.NBAIOT,
                "detector_seed": 0,
                "feature_count": 3,
                "hidden_layers": (2,),
                "training": {
                    "learning_rate": 0.001,
                    "local_epochs": 1,
                    "rounds": 1,
                    "batch_size": 16,
                },
                "preprocessing_digest": "1" * 64,
                "model_digest": "0" * 64,
                "scaler_digests": [],
            }
        )


def test_checkpoint_metadata_combined_digest_is_sensitive_to_preprocessing_digest() -> None:
    base = CheckpointMetadata(
        dataset_id=DatasetId.NBAIOT,
        detector_seed=0,
        feature_count=3,
        hidden_layers=(4,),
        input_space=DetectorInputSpace.PER_CLIENT_ZSCORE,
        training=_settings(),
        preprocessing_digest="a" * 64,
        model_digest="0" * 64,
        scaler_digests=(),
    )
    changed = base.model_copy(update={"preprocessing_digest": "b" * 64})
    assert base.combined_digest != changed.combined_digest
