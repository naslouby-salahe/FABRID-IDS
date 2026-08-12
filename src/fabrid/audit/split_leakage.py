"""T01: partition exclusivity. No `sample_id` may appear in more than one partition."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence


class PartitionExclusivityError(Exception):
    pass


def check_partition_exclusivity(sample_ids_by_partition: Mapping[str, Sequence[str]]) -> None:
    counts = Counter(
        sample_id for sample_ids in sample_ids_by_partition.values() for sample_id in sample_ids
    )
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    if duplicates:
        raise PartitionExclusivityError(
            f"sample_id(s) present in more than one partition: {duplicates}"
        )
