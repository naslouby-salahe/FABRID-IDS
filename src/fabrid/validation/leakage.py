from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from fabrid.artifacts.layout import ScoreSplit
from fabrid.domain.identifiers import SampleId


class PartitionExclusivityValidationError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class PartitionSamples:
    split: ScoreSplit
    sample_ids: tuple[SampleId, ...]

    def __post_init__(self) -> None:
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("a partition contains duplicate sample identifiers")


def validate_partition_exclusivity(
    partitions: tuple[PartitionSamples, ...],
) -> None:
    if not partitions:
        raise ValueError("partition exclusivity requires at least one partition")
    splits = tuple(partition.split for partition in partitions)
    if len(set(splits)) != len(splits):
        raise ValueError("partition exclusivity contains duplicate split identities")
    counts = Counter(
        sample_id
        for partition in partitions
        for sample_id in partition.sample_ids
    )
    duplicates = tuple(
        sorted(
            (sample_id for sample_id, count in counts.items() if count > 1),
            key=lambda sample_id: sample_id.value,
        )
    )
    if duplicates:
        raise PartitionExclusivityValidationError(
            "sample identifiers appear in more than one split"
        )
