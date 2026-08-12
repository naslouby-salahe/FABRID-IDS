from __future__ import annotations

import hashlib
from dataclasses import dataclass

from fabrid.artifacts.layout import ScoreSplit
from fabrid.domain.coordinates import ScoreCoordinate
from fabrid.domain.enums import Label
from fabrid.domain.identifiers import ArtifactDigest, AttackSubtypeId, SampleId, SourceFileId
from fabrid.domain.values import AnomalyScore, EventTimestamp, SourceRowIndex


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
        if self.label is Label.ATTACK and self.attack_subtype is None:
            raise ValueError("attack records must carry an attack subtype")
        if self.label is Label.BENIGN and self.attack_subtype is not None:
            raise ValueError("benign records must not carry an attack subtype")


@dataclass(frozen=True, slots=True)
class ScorePartitionArtifact:
    coordinate: ScoreCoordinate
    split: ScoreSplit
    records: tuple[ScoreRecord, ...]

    def __post_init__(self) -> None:
        sample_ids = tuple(record.sample_id for record in self.records)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("duplicate sample id in score partition artifact")

        if self.split.value.startswith("benign_"):
            if any(record.label is not Label.BENIGN for record in self.records):
                raise ValueError("benign score partitions may contain only benign records")
        elif any(record.label is not Label.ATTACK for record in self.records):
            raise ValueError("attack score partitions may contain only attack records")

    def digest(self) -> ArtifactDigest:
        digest = hashlib.sha256()
        self._update_digest(digest, self.coordinate.dataset_id.value)
        self._update_digest(digest, str(self.coordinate.detector_seed.value))
        self._update_digest(digest, self.coordinate.client_id.value)
        self._update_digest(digest, self.split.value)

        for record in sorted(self.records, key=lambda item: item.sample_id.value):
            self._update_digest(digest, record.sample_id.value)
            self._update_digest(digest, record.source_file.value)
            self._update_digest(digest, str(record.source_row.value))
            self._update_digest(digest, repr(record.score.value))
            self._update_digest(digest, record.label.value)
            self._update_digest(
                digest,
                "" if record.attack_subtype is None else record.attack_subtype.value,
            )
            self._update_digest(
                digest,
                "" if record.timestamp is None else repr(record.timestamp.value),
            )

        return ArtifactDigest(digest.hexdigest())

    @staticmethod
    def _update_digest(digest: hashlib._Hash, value: str) -> None:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
