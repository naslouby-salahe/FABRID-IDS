from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fabrid.artifacts.paths import DetectorCoordinate
from fabrid.config import (
    EXTERNAL_SOURCE_FILE,
    DatasetId,
    ExternalReplicationConfig,
    UtilityEligibilityConfig,
)
from fabrid.datasets.cic_iot_diad import ExternalFederation
from fabrid.datasets.registry import (
    AttackFeatureBlock,
    DeviceDataset,
    FeatureMatrix,
    plan_device_splits,
)
from fabrid.detector.autoencoder import Autoencoder, AutoencoderArchitecture
from fabrid.detector.training import (
    CheckpointMetadata,
    ClientScaler,
    DetectorInputSpace,
    FederatedScalers,
    FederatedTrainingSettings,
    fit_feature_scaler,
)
from fabrid.errors import DatasetError
from fabrid.experiments.external_replication import (
    ExternalTrainedSeed,
    devices_with_benign_train,
    external_loaded_scores,
    external_split_plans,
    load_external_federation_cache,
    load_mmap_external_federation_cache,
    load_or_train_external_seed,
    require_trained_feature_width,
    save_external_federation_cache,
    save_mmap_external_federation_cache,
)
from tests.support import isolated_paths, production_application


def _device(client_id: str, benign_rows: int, feature_count: int) -> DeviceDataset:
    benign = FeatureMatrix(
        np.ones((benign_rows, feature_count), dtype=np.float64)
        if benign_rows
        else np.empty((0, feature_count), dtype=np.float64)
    )
    attacks = (
        AttackFeatureBlock(
            subtype="syn",
            source_file="syn.csv",
            features=FeatureMatrix(np.ones((8, feature_count), dtype=np.float64)),
        ),
    )
    return DeviceDataset(
        client_id=client_id,
        benign_source_file=EXTERNAL_SOURCE_FILE,
        benign=benign,
        attacks=attacks,
    )


def _federation(*devices: DeviceDataset) -> ExternalFederation:
    feature_count = devices[0].benign.feature_count
    columns = tuple(f"f{index}" for index in range(feature_count))
    return ExternalFederation(devices=devices, kept_columns=columns, dropped_rows=0)


def _trained(feature_count: int, client_ids: tuple[str, ...]) -> ExternalTrainedSeed:
    model = Autoencoder(AutoencoderArchitecture(feature_count=feature_count, hidden_layers=(2,)))
    scalers = FederatedScalers(
        clients=tuple(
            ClientScaler(
                client_id=client_id,
                scaler=fit_feature_scaler(
                    FeatureMatrix(np.ones((4, feature_count), dtype=np.float64))
                ),
            )
            for client_id in client_ids
        )
    )
    return ExternalTrainedSeed(
        seed=0,
        model=model,
        scalers=scalers,
        metadata=CheckpointMetadata(
            dataset_id=DatasetId.NBAIOT,
            detector_seed=0,
            feature_count=feature_count,
            hidden_layers=(2,),
            input_space=DetectorInputSpace.PER_CLIENT_ZSCORE,
            training=FederatedTrainingSettings(
                learning_rate=0.01, local_epochs=1, rounds=1, batch_size=8
            ),
            preprocessing_digest="9" * 64,
            model_digest="0" * 64,
            scaler_digests=(),
        ),
    )


def _tiny_external(*, train_end: float = 0.5) -> ExternalReplicationConfig:
    application = production_application()
    external = application.external_replication
    detector = external.detector.model_copy(
        update={
            "hidden_layers": (2,),
            "learning_rate": 0.01,
            "local_epochs": 1,
            "rounds": 1,
            "batch_size": 8,
            "seeds": (0,),
        }
    )
    splits = external.benign_splits.model_copy(update={"train_end": train_end})
    return external.model_copy(update={"detector": detector, "benign_splits": splits})


def test_width_mismatch_raises_dataset_error(tmp_path: Path) -> None:
    application = production_application()
    external = application.external_replication
    trained = _trained(3, ("aa",))
    features = FeatureMatrix(np.ones((6, 4), dtype=np.float64))
    with pytest.raises(DatasetError, match="feature width"):
        require_trained_feature_width(features, trained)
    federation = _federation(_device("aa", 20, 4))
    plans = external_split_plans(federation, external)
    paths = isolated_paths(tmp_path)
    with pytest.raises(DatasetError, match="feature width"):
        external_loaded_scores(
            federation, plans, trained.seed, trained, external.dataset_id, 8, paths
        )


def test_empty_train_device_is_not_scored(tmp_path: Path) -> None:
    application = production_application()
    external = application.external_replication
    trained = _trained(3, ("aa", "bb"))
    federation = _federation(_device("aa", 20, 3), _device("bb", 0, 3))
    plans = external_split_plans(federation, external)
    assert plans[1].benign.train_end == 0
    paths = isolated_paths(tmp_path)
    loaded = external_loaded_scores(
        federation, plans, trained.seed, trained, external.dataset_id, 8, paths
    )
    assert tuple(client.client_id for client in loaded.clients) == ("aa",)
    scorable = devices_with_benign_train(federation.devices, external)
    assert tuple(device.client_id for device in scorable) == ("aa",)


def test_split_plans_match_plan_device_splits() -> None:
    external = production_application().external_replication
    device = _device("aa", 40, 3)
    federation = _federation(device)
    assert external_split_plans(federation, external) == (
        plan_device_splits(device, external.benign_splits, external.attack_split),
    )


def test_external_federation_cache_round_trip_and_key_invalidation(tmp_path: Path) -> None:
    federation = _federation(_device("aa", 40, 3))
    cache_path = tmp_path / "external_federation.pkl"
    cache_key = "a" * 64
    save_external_federation_cache(cache_path, cache_key, federation)
    reused = load_external_federation_cache(cache_path, cache_key)
    assert reused is not None
    assert reused.kept_columns == federation.kept_columns
    assert tuple(device.client_id for device in reused.devices) == ("aa",)
    np.testing.assert_array_equal(
        reused.devices[0].benign.values,
        federation.devices[0].benign.values,
    )
    assert load_external_federation_cache(cache_path, "b" * 64) is None


def test_memory_mapped_external_federation_cache_round_trip(tmp_path: Path) -> None:
    federation = _federation(_device("aa", 40, 3))
    cache_path = tmp_path / "external_federation.pkl"
    cache_key = "a" * 64
    save_mmap_external_federation_cache(cache_path, cache_key, federation)

    reused = load_mmap_external_federation_cache(cache_path, cache_key)

    assert reused is not None
    assert isinstance(reused.devices[0].benign.values, np.memmap)
    np.testing.assert_array_equal(
        reused.devices[0].benign.values,
        federation.devices[0].benign.values,
    )
    assert load_mmap_external_federation_cache(cache_path, "b" * 64) is None


def test_load_or_train_external_seed_builds_then_reuses(tmp_path: Path) -> None:
    external = _tiny_external()
    paths = isolated_paths(tmp_path)
    federation = _federation(_device("aa", 40, 3))
    plans = external_split_plans(federation, external)
    first = load_or_train_external_seed(federation, plans, paths, external, 0)
    assert first.metadata.dataset_id == external.dataset_id
    assert first.metadata.feature_count == 3
    assert first.metadata.hidden_layers == (2,)
    assert paths.checkpoint_dir(
        DetectorCoordinate(dataset_id=external.dataset_id, detector_seed=0)
    ).is_dir()
    second = load_or_train_external_seed(federation, plans, paths, external, 0)
    assert second.metadata.model_digest == first.metadata.model_digest
    assert tuple(client.client_id for client in second.scalers.clients) == ("aa",)


def test_load_or_train_external_seed_rebuilds_on_manifest_change(tmp_path: Path) -> None:
    external = _tiny_external()
    paths = isolated_paths(tmp_path)
    first_federation = _federation(_device("aa", 40, 3))
    first = load_or_train_external_seed(
        first_federation,
        external_split_plans(first_federation, external),
        paths,
        external,
        0,
    )
    wider_federation = _federation(_device("aa", 40, 4))
    rebuilt = load_or_train_external_seed(
        wider_federation,
        external_split_plans(wider_federation, external),
        paths,
        external,
        0,
    )
    assert rebuilt.metadata.feature_count == 4
    assert rebuilt.metadata.model_digest != first.metadata.model_digest


def test_load_or_train_external_seed_rebuilds_on_split_change(tmp_path: Path) -> None:
    federation = _federation(_device("aa", 40, 3))
    paths = isolated_paths(tmp_path)
    first_external = _tiny_external(train_end=0.4)
    first = load_or_train_external_seed(
        federation,
        external_split_plans(federation, first_external),
        paths,
        first_external,
        0,
    )
    second_external = _tiny_external(train_end=0.6)
    rebuilt = load_or_train_external_seed(
        federation,
        external_split_plans(federation, second_external),
        paths,
        second_external,
        0,
    )
    assert rebuilt.metadata.model_digest != first.metadata.model_digest


def test_external_protocol_uses_external_eligibility() -> None:
    application = production_application()
    primary = application.protocol
    eligibility = application.external_replication.eligibility.model_copy(
        update={
            "minimum_attack_validation_rows": 7,
            "minimum_eligible_subtypes": 3,
            "minimum_rows_per_subtype": 11,
        }
    )
    external = application.external_replication.model_copy(update={"eligibility": eligibility})
    protocol = external.protocol_for(primary)
    assert protocol.utility_eligibility == UtilityEligibilityConfig(
        minimum_attack_validation_rows=7,
        minimum_eligible_subtypes=3,
        minimum_rows_per_subtype=11,
    )
    assert protocol.alpha_grid == primary.alpha_grid
    assert protocol.payload_sizing == primary.payload_sizing
    assert protocol.practical_gates == primary.practical_gates
    assert protocol.generalization == primary.generalization
    assert protocol.event_gate == primary.event_gate
    assert protocol.sensitivity == primary.sensitivity
