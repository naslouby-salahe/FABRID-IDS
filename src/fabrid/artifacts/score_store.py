from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl

from fabrid.artifacts.digests import digest_file
from fabrid.artifacts.layout import ScoreSplit
from fabrid.artifacts.score import ScorePartitionArtifact, ScoreRecord
from fabrid.domain.coordinates import ScoreCoordinate
from fabrid.domain.enums import DatasetId, Label
from fabrid.domain.identifiers import (
    ArtifactDigest,
    AttackSubtypeId,
    ClientId,
    SampleId,
    SourceFileId,
)
from fabrid.domain.values import (
    AnomalyScore,
    DetectorSeed,
    EventTimestamp,
    SourceRowIndex,
)

SerializedScoreRow = tuple[
    str,
    int,
    str,
    str,
    str,
    str,
    int,
    float,
    str,
    str | None,
    float | None,
]

_SCORE_SCHEMA = (
    ("dataset_id", pl.String),
    ("detector_seed", pl.Int64),
    ("client_id", pl.String),
    ("split", pl.String),
    ("sample_id", pl.String),
    ("source_file", pl.String),
    ("source_row", pl.Int64),
    ("score", pl.Float64),
    ("label", pl.String),
    ("attack_subtype", pl.String),
    ("timestamp", pl.Float64),
)


@dataclass(frozen=True, slots=True)
class StoredScoreArtifact:
    logical_digest: ArtifactDigest
    file_digest: ArtifactDigest
    path: Path


def _serialize_record(
    artifact: ScorePartitionArtifact,
    record: ScoreRecord,
) -> SerializedScoreRow:
    return (
        artifact.coordinate.dataset_id.value,
        artifact.coordinate.detector_seed.value,
        artifact.coordinate.client_id.value,
        artifact.split.value,
        record.sample_id.value,
        record.source_file.value,
        record.source_row.value,
        record.score.value,
        record.label.value,
        None if record.attack_subtype is None else record.attack_subtype.value,
        None if record.timestamp is None else record.timestamp.value,
    )


def write_score_partition(
    artifact: ScorePartitionArtifact,
    path: Path,
) -> StoredScoreArtifact:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        [_serialize_record(artifact, record) for record in artifact.records],
        schema=_SCORE_SCHEMA,
        orient="row",
    )
    frame.write_parquet(path, compression="zstd", statistics=True)
    return StoredScoreArtifact(
        logical_digest=artifact.digest(),
        file_digest=digest_file(path),
        path=path,
    )


def read_score_partition(
    path: Path,
    coordinate: ScoreCoordinate,
    split: ScoreSplit,
) -> ScorePartitionArtifact:
    frame = pl.read_parquet(path)
    records: list[ScoreRecord] = []
    for raw_row in frame.iter_rows(named=False):
        row = cast(SerializedScoreRow, raw_row)
        dataset_id, detector_seed, client_id, serialized_split = (
            DatasetId(row[0]),
            DetectorSeed(row[1]),
            ClientId(row[2]),
            row[3],
        )
        if dataset_id != coordinate.dataset_id:
            raise ValueError("score artifact dataset id does not match requested coordinate")
        if detector_seed != coordinate.detector_seed:
            raise ValueError("score artifact seed does not match requested coordinate")
        if client_id != coordinate.client_id:
            raise ValueError("score artifact client id does not match requested coordinate")
        if serialized_split != split.value:
            raise ValueError("score artifact split does not match requested split")

        records.append(
            ScoreRecord(
                sample_id=SampleId(row[4]),
                source_file=SourceFileId(row[5]),
                source_row=SourceRowIndex(row[6]),
                score=AnomalyScore(row[7]),
                label=Label(row[8]),
                attack_subtype=(
                    None if row[9] is None else AttackSubtypeId(row[9])
                ),
                timestamp=(
                    None if row[10] is None else EventTimestamp(row[10])
                ),
            )
        )

    return ScorePartitionArtifact(
        coordinate=coordinate,
        split=split,
        records=tuple(records),
    )
