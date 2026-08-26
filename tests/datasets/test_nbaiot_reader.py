from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fabrid.config import ClientId, NbaiotDatasetConfig
from fabrid.datasets.nbaiot import read_device_directory
from fabrid.errors import DatasetError
from tests.support import production_application


def _layout() -> NbaiotDatasetConfig:
    return production_application().datasets.nbaiot


def _write_feature_csv(path: Path, values: np.ndarray) -> None:
    header = ",".join(f"f{index}" for index in range(1, values.shape[1] + 1))
    np.savetxt(path, values, delimiter=",", header=header, comments="")


def _write_device_tree(root: Path, device: ClientId) -> None:
    layout = _layout()
    device_dir = root / device
    bashlite_dir = device_dir / layout.bashlite_directory
    mirai_dir = device_dir / layout.mirai_directory
    device_dir.mkdir(parents=True)
    bashlite_dir.mkdir(parents=True)
    mirai_dir.mkdir(parents=True)
    _write_feature_csv(
        device_dir / layout.benign_filename,
        np.random.default_rng(0).normal(size=(20, 3)),
    )
    for index, mapping in enumerate(layout.bashlite_files, start=1):
        _write_feature_csv(bashlite_dir / mapping.filename, np.full((5, 3), float(index)))
    for index, mapping in enumerate(layout.mirai_files, start=1):
        _write_feature_csv(mirai_dir / mapping.filename, np.full((5, 3), float(index + 10)))


def test_read_device_directory(tmp_path: Path) -> None:
    layout = _layout()
    client_id = layout.devices[0]
    _write_device_tree(tmp_path, client_id)
    device = read_device_directory(client_id, tmp_path / client_id, layout)
    assert device.client_id == client_id
    assert device.benign.row_count == 20
    assert {block.subtype for block in device.attacks} == {
        mapping.subtype for mapping in (*layout.bashlite_files, *layout.mirai_files)
    }


def test_read_device_directory_missing_benign_raises(tmp_path: Path) -> None:
    (tmp_path / "device").mkdir()
    layout = _layout()
    missing = tmp_path / "device"
    with pytest.raises(DatasetError):
        read_device_directory("device", missing, layout)


def test_primary_population_matches_nine_devices() -> None:
    layout = _layout()
    assert len(layout.devices) == 9


def test_dual_botnet_family_population_is_seven_devices() -> None:
    layout = _layout()
    dual = layout.dual_botnet_devices()
    assert len(dual) == 7
    assert set(dual).issubset(layout.devices)
