from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from fabrid.artifacts.parquet import write_parquet_frame
from fabrid.config import AttackSplit, BenignSplit, DatasetId, PreprocessedColumn
from fabrid.execution.prerequisites import (
    load_preprocessed_attack_partition,
    load_preprocessed_partition,
)
from tests.support import isolated_paths


def test_partition_loaders_filter_client_and_split(tmp_path: Path) -> None:
    paths = isolated_paths(tmp_path)
    dataset_id = DatasetId.NBAIOT
    feature_columns = ("f0", "f1")
    frame = pl.DataFrame(
        {
            PreprocessedColumn.CLIENT_ID: ["a", "a", "a", "b"],
            PreprocessedColumn.SOURCE_FILE: ["benign.csv", "benign.csv", "scan.csv", "benign.csv"],
            PreprocessedColumn.SOURCE_ROW: [0, 1, 0, 0],
            PreprocessedColumn.TIMESTAMP: [None, None, None, None],
            PreprocessedColumn.SPLIT: [
                BenignSplit.TRAIN.value,
                BenignSplit.TEST.value,
                AttackSplit.VALIDATION.value,
                BenignSplit.TRAIN.value,
            ],
            PreprocessedColumn.ATTACK_SUBTYPE: [None, None, "scan", None],
            "f0": [1.0, 2.0, 3.0, 4.0],
            "f1": [10.0, 20.0, 30.0, 40.0],
        }
    )
    write_parquet_frame(paths.preprocessing_features_path(dataset_id), frame)
    train_a = load_preprocessed_partition(
        paths, dataset_id, "a", BenignSplit.TRAIN, feature_columns
    )
    assert np.array_equal(train_a.values, np.array([[1.0, 10.0]]))
    test_a = load_preprocessed_partition(paths, dataset_id, "a", BenignSplit.TEST, feature_columns)
    assert np.array_equal(test_a.values, np.array([[2.0, 20.0]]))
    attack = load_preprocessed_attack_partition(
        paths, dataset_id, "a", "scan", AttackSplit.VALIDATION, feature_columns
    )
    assert np.array_equal(attack.values, np.array([[3.0, 30.0]]))
