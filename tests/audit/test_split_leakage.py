from __future__ import annotations

import pytest

from fabrid.audit.split_leakage import PartitionExclusivityError, check_partition_exclusivity


def test_no_duplicates_passes() -> None:
    check_partition_exclusivity({"train": ["a", "b"], "test": ["c", "d"]})


def test_duplicate_across_partitions_raises() -> None:
    with pytest.raises(PartitionExclusivityError):
        check_partition_exclusivity({"train": ["a", "b"], "test": ["b", "c"]})


def test_duplicate_within_partition_raises() -> None:
    with pytest.raises(PartitionExclusivityError):
        check_partition_exclusivity({"train": ["a", "a"]})


def test_empty_partitions_pass() -> None:
    check_partition_exclusivity({"train": [], "test": []})
