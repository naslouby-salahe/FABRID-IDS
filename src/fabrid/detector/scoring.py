from __future__ import annotations

from dataclasses import dataclass

from fabrid.artifacts.score import ScorePartitionArtifact, ScoreRecord
from fabrid.datasets.common import DeviceDataset
from fabrid.datasets.splitting import DeviceSplitPlan
from fabrid.detector.model import Autoencoder, reconstruction_error_scores
from fabrid.detector.preprocessing import FeatureScaler
from fabrid.domain.coordinates import ScoreCoordinate
from fabrid.domain.enums import AttackSplit, BenignSplit, DatasetId, Label
from fabrid.domain.identifiers import SampleId
from fabrid.domain.values import AnomalyScore, DetectorSeed, SourceRowIndex


@dataclass(frozen=True, slots=True)
class ClientScoreArtifacts:
    benign_train: ScorePartitionArtifact
    benign_frontier: ScorePartitionArtifact
    benign_final_cal: ScorePartitionArtifact
    benign_test: ScorePartitionArtifact
    attack_validation: ScorePartitionArtifact
    attack_test: ScorePartitionArtifact

    @property
    def all(self) -> tuple[ScorePartitionArtifact, ...]:
        return (
            self.benign_train,
            self.benign_frontier,
            self.benign_final_cal,
            self.benign_test,
            self.attack_validation,
            self.attack_test,
        )


def _record(
    dataset: DeviceDataset,
    source_file,
    source_row: SourceRowIndex,
    score: AnomalyScore,
    label: Label,
    attack_subtype,
) -> ScoreRecord:
    subtype_component = "benign" if attack_subtype is None else attack_subtype.value
    return ScoreRecord(
        sample_id=SampleId(
            f"{dataset.client_id.value}:{subtype_component}:{source_row.value}"
        ),
        source_file=source_file,
        source_row=source_row,
        score=score,
        label=label,
        attack_subtype=attack_subtype,
        timestamp=None,
    )


def generate_client_score_artifacts(
    dataset_id: DatasetId,
    detector_seed: DetectorSeed,
    dataset: DeviceDataset,
    split_plan: DeviceSplitPlan,
    scaler: FeatureScaler,
    model: Autoencoder,
) -> ClientScoreArtifacts:
    if dataset.benign.row_count != split_plan.benign.total_rows:
        raise ValueError("benign split plan does not match dataset row count")

    coordinate = ScoreCoordinate(
        dataset_id=dataset_id,
        detector_seed=detector_seed,
        client_id=dataset.client_id,
    )
    benign_records: dict[BenignSplit, list[ScoreRecord]] = {
        split: [] for split in BenignSplit
    }
    benign_scores = reconstruction_error_scores(
        model,
        scaler.transform(dataset.benign),
    )
    for index in range(dataset.benign.row_count.value):
        source_row = SourceRowIndex(index)
        split = split_plan.benign.split_of(source_row)
        benign_records[split].append(
            _record(
                dataset=dataset,
                source_file=dataset.benign_source_file,
                source_row=source_row,
                score=AnomalyScore(float(benign_scores.values[index])),
                label=Label.BENIGN,
                attack_subtype=None,
            )
        )

    attack_validation_records: list[ScoreRecord] = []
    attack_test_records: list[ScoreRecord] = []
    for attack in dataset.attacks:
        boundary = split_plan.attack_boundary(attack.subtype)
        if attack.features.row_count != boundary.total_rows:
            raise ValueError(
                f"attack split plan does not match {attack.subtype.value} row count"
            )
        scores = reconstruction_error_scores(
            model,
            scaler.transform(attack.features),
        )
        for index in range(attack.features.row_count.value):
            source_row = SourceRowIndex(index)
            split = boundary.split_of(source_row)
            record = _record(
                dataset=dataset,
                source_file=attack.source_file,
                source_row=source_row,
                score=AnomalyScore(float(scores.values[index])),
                label=Label.ATTACK,
                attack_subtype=attack.subtype,
            )
            if split is AttackSplit.VALIDATION:
                attack_validation_records.append(record)
            else:
                attack_test_records.append(record)

    return ClientScoreArtifacts(
        benign_train=ScorePartitionArtifact(
            coordinate, BenignSplit.TRAIN, tuple(benign_records[BenignSplit.TRAIN])
        ),
        benign_frontier=ScorePartitionArtifact(
            coordinate,
            BenignSplit.FRONTIER,
            tuple(benign_records[BenignSplit.FRONTIER]),
        ),
        benign_final_cal=ScorePartitionArtifact(
            coordinate,
            BenignSplit.FINAL_CAL,
            tuple(benign_records[BenignSplit.FINAL_CAL]),
        ),
        benign_test=ScorePartitionArtifact(
            coordinate, BenignSplit.TEST, tuple(benign_records[BenignSplit.TEST])
        ),
        attack_validation=ScorePartitionArtifact(
            coordinate,
            AttackSplit.VALIDATION,
            tuple(attack_validation_records),
        ),
        attack_test=ScorePartitionArtifact(
            coordinate,
            AttackSplit.TEST,
            tuple(attack_test_records),
        ),
    )
