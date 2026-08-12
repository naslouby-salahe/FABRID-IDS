"""Immutable persisted score artifact: one per dataset x detector_seed x client x split.

Every allocation/calibration/evaluation module downstream reads these records
only — none of them may recompute or mutate a score.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from fabrid.data.partitioner import AttackSplit, BenignSplit, RowCount
from fabrid.evaluation.record_level import AttackSubtype, ClientId

SplitId = BenignSplit | AttackSplit


class Label(StrEnum):
    BENIGN = "benign"
    ATTACK = "attack"


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    sample_id: str
    source_file: str
    source_row: RowCount
    split_id: SplitId
    score: float
    label: Label
    attack_type: AttackSubtype | None
    timestamp: float | None

    def __post_init__(self) -> None:
        if self.label is Label.ATTACK and self.attack_type is None:
            raise ValueError("attack-labeled records must carry an attack_type")
        if self.label is Label.BENIGN and self.attack_type is not None:
            raise ValueError("benign-labeled records must not carry an attack_type")


@dataclass(frozen=True, slots=True)
class DetectorSeed:
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(f"detector seed must be non-negative, got {self.value}")


@dataclass(frozen=True, slots=True)
class ScoreArtifact:
    """All scored rows for one dataset x detector_seed x client x split coordinate."""

    dataset_id: str
    detector_seed: DetectorSeed
    client_id: ClientId
    records: tuple[ScoreRecord, ...]

    def __post_init__(self) -> None:
        sample_ids = [record.sample_id for record in self.records]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("duplicate sample_id within a single score artifact")

    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.dataset_id.encode("utf-8"))
        digest.update(str(self.detector_seed.value).encode("utf-8"))
        digest.update(self.client_id.encode("utf-8"))
        for record in sorted(self.records, key=lambda r: r.sample_id):
            digest.update(record.sample_id.encode("utf-8"))
            digest.update(repr(record.score).encode("utf-8"))
            digest.update(record.label.value.encode("utf-8"))
            digest.update(str(record.split_id.value).encode("utf-8"))
        return digest.hexdigest()

    def scores_by_split(self, split_id: SplitId) -> tuple[ScoreRecord, ...]:
        return tuple(record for record in self.records if record.split_id == split_id)
