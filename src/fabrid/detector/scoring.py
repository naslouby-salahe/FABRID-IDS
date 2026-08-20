from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import torch

from fabrid.artifacts.json import digest_file, digest_text
from fabrid.artifacts.parquet import read_parquet_models, write_parquet_models
from fabrid.artifacts.paths import ScoreCoordinate
from fabrid.config import (
    AnomalyScore,
    ArtifactDigest,
    AttackSplit,
    AttackSubtypeId,
    BatchSize,
    BenignSplit,
    EnvironmentText,
    EventTimestamp,
    Label,
    RowCount,
    SampleId,
    SourceFileId,
    SourceRowIndex,
)
from fabrid.datasets.registry import FeatureMatrix
from fabrid.detector.autoencoder import Autoencoder


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    sample_id: SampleId
    source_file: SourceFileId
    source_row: SourceRowIndex
    score: AnomalyScore
    label: Label
    attack_subtype: AttackSubtypeId | None
    timestamp: EventTimestamp | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", Label(self.label))
        if self.label is Label.ATTACK and self.attack_subtype is None:
            raise ValueError("attack records must carry an attack subtype")
        if self.label is Label.BENIGN and self.attack_subtype is not None:
            raise ValueError("benign records must not carry an attack subtype")

    def relabeled(self, sample_id: SampleId) -> ScoreRecord:
        record = object.__new__(ScoreRecord)
        object.__setattr__(record, "sample_id", sample_id)
        object.__setattr__(record, "source_file", self.source_file)
        object.__setattr__(record, "source_row", self.source_row)
        object.__setattr__(record, "score", self.score)
        object.__setattr__(record, "label", self.label)
        object.__setattr__(record, "attack_subtype", self.attack_subtype)
        object.__setattr__(record, "timestamp", self.timestamp)
        return record


@dataclass(frozen=True, slots=True)
class AttackSubtypeScores:
    subtype: AttackSubtypeId
    scores: np.ndarray


@dataclass(frozen=True, slots=True)
class ScorePartitionArtifact:
    coordinate: ScoreCoordinate
    split: BenignSplit | AttackSplit
    records: tuple[ScoreRecord, ...]
    _digest: ArtifactDigest | None = field(default=None, repr=False, compare=False)
    _score_values: np.ndarray | None = field(default=None, repr=False, compare=False)
    _subtype_scores: tuple[AttackSubtypeScores, ...] | None = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.split in BenignSplit:
            if any(record.label is not Label.BENIGN for record in self.records):
                raise ValueError("benign score partitions may contain only benign records")
        elif any(record.label is not Label.ATTACK for record in self.records):
            raise ValueError("attack score partitions may contain only attack records")

    @property
    def row_count(self) -> RowCount:
        return len(self.records)

    def score_values(self) -> np.ndarray:
        if self._score_values is None:
            values = np.fromiter(
                (record.score for record in self.records),
                dtype=np.float64,
                count=self.row_count,
            )
            object.__setattr__(self, "_score_values", values)
        assert self._score_values is not None
        return self._score_values

    def subtype_scores(self) -> tuple[AttackSubtypeScores, ...]:
        if self._subtype_scores is None:
            scores = np.fromiter(
                (record.score for record in self.records if record.attack_subtype is not None),
                dtype=np.float64,
            )
            subtypes = np.fromiter(
                (
                    record.attack_subtype
                    for record in self.records
                    if record.attack_subtype is not None
                ),
                dtype=object,
            )
            if subtypes.size == 0:
                grouped: tuple[AttackSubtypeScores, ...] = ()
            else:
                order = np.argsort(subtypes, kind="stable")
                sorted_scores = scores[order]
                sorted_subtypes = subtypes[order]
                boundaries = np.flatnonzero(
                    np.concatenate(
                        (
                            np.asarray([True]),
                            sorted_subtypes[1:] != sorted_subtypes[:-1],
                            np.asarray([True]),
                        )
                    )
                )
                grouped = tuple(
                    AttackSubtypeScores(
                        subtype=cast(AttackSubtypeId, sorted_subtypes[start]),
                        scores=sorted_scores[start:end],
                    )
                    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
                )
            object.__setattr__(self, "_subtype_scores", grouped)
        assert self._subtype_scores is not None
        return self._subtype_scores

    def digest(self) -> ArtifactDigest:
        if self._digest is None:
            parts: list[EnvironmentText] = [
                self.coordinate.dataset_id.value,
                str(self.coordinate.detector_seed),
                self.coordinate.client_id,
                self.split.value,
            ]
            for record in sorted(self.records, key=lambda item: item.sample_id):
                parts.extend(
                    (
                        record.sample_id,
                        record.source_file,
                        str(record.source_row),
                        repr(record.score),
                        record.label.value,
                        ("none" if record.attack_subtype is None else record.attack_subtype),
                        ("none" if record.timestamp is None else repr(record.timestamp)),
                    )
                )
            digest = digest_text(tuple(parts))
            object.__setattr__(self, "_digest", digest)
        assert self._digest is not None
        return self._digest


def require_exclusive_sample_ids(artifacts: tuple[ScorePartitionArtifact, ...]) -> None:
    seen: set[SampleId] = set()
    for artifact in artifacts:
        for record in artifact.records:
            if record.sample_id in seen:
                raise ValueError(f"sample id {record.sample_id} appears in multiple partitions")
            seen.add(record.sample_id)


def score_feature_matrix(
    model: Autoencoder,
    features: FeatureMatrix,
    device: torch.device,
    batch_size: BatchSize,
) -> np.ndarray:
    model.to(device)
    model.eval()
    inputs = torch.as_tensor(features.values, dtype=torch.float32, device=device)
    if inputs.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)
    with torch.no_grad():
        batches: list[np.ndarray] = []
        for start in range(0, inputs.shape[0], batch_size):
            batch = inputs[start : start + batch_size]
            reconstructed = model(batch)
            per_row_mse = torch.mean((batch - reconstructed) ** 2, dim=1)
            batches.append(per_row_mse.cpu().numpy().astype(np.float64))
    return np.concatenate(batches)


def build_score_partition(
    coordinate: ScoreCoordinate,
    split: BenignSplit | AttackSplit,
    scores: np.ndarray,
    label: Label,
    source_file: SourceFileId,
    subtype: AttackSubtypeId | None = None,
) -> ScorePartitionArtifact:
    subtype_part = "" if subtype is None else f"|{subtype}"
    records = tuple(
        ScoreRecord(
            sample_id=f"{coordinate.client_id}|{split.value}{subtype_part}|{index}",
            source_file=source_file,
            source_row=index,
            score=float(score),
            label=label,
            attack_subtype=subtype,
            timestamp=None,
        )
        for index, score in enumerate(scores)
    )
    return ScorePartitionArtifact(coordinate=coordinate, split=split, records=records)


def persist_score_partition(path: Path, artifact: ScorePartitionArtifact) -> ArtifactDigest:
    write_parquet_models(path, artifact.records)
    return digest_file(path)


def load_score_partition(
    path: Path,
    coordinate: ScoreCoordinate,
    split: BenignSplit | AttackSplit,
) -> ScorePartitionArtifact:
    records = read_parquet_models(path, ScoreRecord)
    return ScorePartitionArtifact(coordinate=coordinate, split=split, records=records)
