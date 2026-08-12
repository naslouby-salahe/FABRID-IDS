from __future__ import annotations

from fabrid.audit.reproducibility import DatasetChecksum, capture_reproducibility_metadata


def test_capture_reproducibility_metadata_populates_all_fields() -> None:
    checksums = (DatasetChecksum(dataset_id="n-baiot", sha256="a" * 64),)
    metadata = capture_reproducibility_metadata(seeds=(0, 1, 2), dataset_checksums=checksums)

    assert metadata.os_platform
    assert metadata.python_version
    assert metadata.numpy_version
    assert metadata.scipy_version
    assert metadata.torch_version
    assert metadata.cpu_model
    assert metadata.ram_total_bytes > 0
    assert metadata.solver_backend
    assert len(metadata.git_commit) == 40
    assert metadata.seeds == (0, 1, 2)
    assert metadata.dataset_checksums == checksums


def test_cuda_fields_consistent_with_availability() -> None:
    metadata = capture_reproducibility_metadata(seeds=(0,), dataset_checksums=())

    if metadata.cuda_available:
        assert metadata.cuda_version is not None
        assert metadata.gpu_name is not None
    else:
        assert metadata.cuda_version is None
        assert metadata.gpu_name is None
