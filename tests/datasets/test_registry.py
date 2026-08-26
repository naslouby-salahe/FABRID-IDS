from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fabrid.config import DatasetId
from fabrid.datasets.registry import (
    AttackFeatureBlock,
    ClientPopulation,
    DeviceDataset,
    FeatureMatrix,
    resolve_raw_dataset_root,
)
from fabrid.errors import DatasetError
from tests.support import event_evidence, production_application


def test_feature_matrix_validates_shape_and_dtype() -> None:
    matrix = FeatureMatrix(np.zeros((4, 3), dtype=np.float64))
    assert matrix.row_count == 4
    assert matrix.feature_count == 3
    with pytest.raises(ValueError):
        FeatureMatrix(np.zeros((4,), dtype=np.float64))
    with pytest.raises(ValueError):
        FeatureMatrix(np.zeros((2, 2), dtype=object))
    with pytest.raises(ValueError):
        FeatureMatrix(np.array([[1.0, np.nan], [0.0, 1.0]]))


def test_client_population_validates() -> None:
    population = ClientPopulation(("a", "b", "c"))
    assert population.size == 3
    with pytest.raises(ValueError):
        ClientPopulation(())
    with pytest.raises(ValueError):
        ClientPopulation(("a", "a"))


def test_attack_feature_block_and_device_dataset() -> None:
    layout = production_application().datasets.nbaiot
    bashlite = layout.bashlite_files[0]
    mirai = layout.mirai_files[0]
    benign = FeatureMatrix(np.zeros((10, 2)))
    attack = AttackFeatureBlock(
        subtype=bashlite.subtype,
        source_file=f"{layout.bashlite_directory}/{bashlite.filename}",
        features=FeatureMatrix(np.ones((5, 2))),
    )
    device = DeviceDataset(
        client_id=layout.devices[0],
        benign_source_file=layout.benign_filename,
        benign=benign,
        attacks=(attack,),
    )
    assert device.attack(bashlite.subtype) is attack
    missing_subtype = mirai.subtype
    with pytest.raises(KeyError):
        device.attack(missing_subtype)
    mismatched = AttackFeatureBlock(
        subtype=bashlite.subtype,
        source_file=bashlite.filename,
        features=FeatureMatrix(np.ones((5, 3))),
    )
    with pytest.raises(ValueError):
        DeviceDataset(
            client_id=layout.devices[1],
            benign_source_file=layout.benign_filename,
            benign=benign,
            attacks=(mismatched,),
        )


def test_event_provenance_evidence_requires_every_criterion() -> None:
    evidence = event_evidence()
    assert evidence.passed(evidence.criteria[0].criterion)
    incomplete = evidence.criteria[:-1]
    model = type(evidence)
    with pytest.raises(ValueError):
        model(criteria=incomplete)


def test_resolve_raw_dataset_root_keeps_dataset_trees_as_siblings() -> None:
    catalog = production_application().datasets
    root = Path("/data/raw")
    resolved = resolve_raw_dataset_root(root, DatasetId.NBAIOT, catalog)
    assert resolved == root / catalog.nbaiot.directory_name
    cic_root = root / catalog.cic_iot_diad.directory_name
    with pytest.raises(DatasetError, match="dataset directory"):
        resolve_raw_dataset_root(cic_root, DatasetId.NBAIOT, catalog)
    with pytest.raises(DatasetError, match="dataset directory"):
        resolve_raw_dataset_root(root / catalog.nbaiot.directory_name, DatasetId.NBAIOT, catalog)
