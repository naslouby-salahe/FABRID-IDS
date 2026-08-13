from __future__ import annotations

from dataclasses import dataclass

from fabrid.datasets.common import DeviceDataset
from fabrid.datasets.nbaiot.reader import read_device_directory
from fabrid.datasets.nbaiot.specification import NBAIOT_PRIMARY_POPULATION
from fabrid.datasets.splitting import (
    AttackSubtypeBoundary,
    DeviceSplitPlan,
    compute_attack_split_boundary,
    compute_benign_split_boundaries,
)
from fabrid.detector.model import Autoencoder
from fabrid.detector.persistence import DetectorArtifactSet, save_detector_state
from fabrid.detector.preprocessing import (
    ClientScaler,
    FederatedScalers,
    FeatureScaler,
    fit_feature_scaler,
)
from fabrid.detector.training import (
    ClientTrainingData,
    FederatedTrainingConfig,
    FederatedTrainingData,
    train_federated_autoencoder,
)
from fabrid.domain.coordinates import DetectorCoordinate
from fabrid.domain.enums import DatasetId
from fabrid.domain.identifiers import CampaignId, ClientId
from fabrid.domain.values import DetectorSeed
from fabrid.pipeline.context import PipelinePaths
from fabrid.protocol.detector import DETECTOR_HYPERPARAMETERS
from fabrid.protocol.models import FabridProtocol


@dataclass(frozen=True, slots=True)
class PreparedClient:
    dataset: DeviceDataset
    split_plan: DeviceSplitPlan
    scaler: FeatureScaler


@dataclass(frozen=True, slots=True)
class PreparedFederation:
    clients: tuple[PreparedClient, ...]

    def __post_init__(self) -> None:
        if not self.clients:
            raise ValueError("prepared federation requires at least one client")
        client_ids = tuple(client.dataset.client_id for client in self.clients)
        if len(set(client_ids)) != len(client_ids):
            raise ValueError("prepared federation contains duplicate clients")

    def for_client(self, client_id: ClientId) -> PreparedClient:
        for client in self.clients:
            if client.dataset.client_id == client_id:
                return client
        raise KeyError(client_id.value)

    @property
    def scalers(self) -> FederatedScalers:
        return FederatedScalers(
            tuple(
                ClientScaler(
                    client_id=client.dataset.client_id,
                    scaler=client.scaler,
                )
                for client in self.clients
            )
        )


@dataclass(frozen=True, slots=True)
class TrainedDetectorSeed:
    coordinate: DetectorCoordinate
    model: Autoencoder
    prepared_federation: PreparedFederation
    artifacts: DetectorArtifactSet


def prepare_nbaiot_federation(
    paths: PipelinePaths,
    protocol: FabridProtocol,
) -> PreparedFederation:
    dataset_root = paths.raw_dataset_root(DatasetId.NBAIOT)
    clients: list[PreparedClient] = []
    for client_id in NBAIOT_PRIMARY_POPULATION.clients:
        dataset = read_device_directory(client_id, dataset_root / client_id.value)
        benign_boundaries = compute_benign_split_boundaries(
            dataset.benign.row_count,
            protocol.benign_splits,
        )
        attack_boundaries = tuple(
            AttackSubtypeBoundary(
                subtype=attack.subtype,
                boundary=compute_attack_split_boundary(
                    attack.features.row_count,
                    protocol.attack_split,
                ),
            )
            for attack in dataset.attacks
        )
        split_plan = DeviceSplitPlan(
            benign=benign_boundaries,
            attacks=attack_boundaries,
        )
        scaler = fit_feature_scaler(
            dataset.benign.prefix(benign_boundaries.train_end)
        )
        clients.append(
            PreparedClient(
                dataset=dataset,
                split_plan=split_plan,
                scaler=scaler,
            )
        )
    return PreparedFederation(tuple(clients))


def train_detector_seed(
    campaign_id: CampaignId,
    detector_seed: DetectorSeed,
    prepared: PreparedFederation,
    paths: PipelinePaths,
) -> TrainedDetectorSeed:
    training_data = FederatedTrainingData(
        tuple(
            ClientTrainingData(
                client_id=client.dataset.client_id,
                features=client.scaler.transform(
                    client.dataset.benign.prefix(client.split_plan.benign.train_end)
                ),
            )
            for client in prepared.clients
        )
    )
    hyperparameters = DETECTOR_HYPERPARAMETERS
    model = train_federated_autoencoder(
        training_data,
        FederatedTrainingConfig(
            hidden_layers=hyperparameters.hidden_layers,
            learning_rate=hyperparameters.learning_rate,
            local_epochs=hyperparameters.local_epochs,
            rounds=hyperparameters.rounds,
            batch_size=hyperparameters.batch_size,
            seed=detector_seed,
        ),
    )
    coordinate = DetectorCoordinate(
        dataset_id=DatasetId.NBAIOT,
        detector_seed=detector_seed,
    )
    artifact_set = save_detector_state(
        paths.artifacts.detector_root(campaign_id, coordinate),
        model,
        prepared.scalers,
    )
    return TrainedDetectorSeed(
        coordinate=coordinate,
        model=model,
        prepared_federation=prepared,
        artifacts=artifact_set,
    )
