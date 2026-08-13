from __future__ import annotations

from fabrid.domain.enums import DatasetId, SolverBackend
from fabrid.domain.identifiers import ArtifactDigest
from fabrid.domain.values import DetectorSeed
from fabrid.validation.reproducibility import (
    CudaAvailability,
    DatasetChecksum,
    capture_reproducibility_metadata,
)


def test_reproducibility_metadata_uses_typed_provenance_fields() -> None:
    checksums = (DatasetChecksum(DatasetId.NBAIOT, ArtifactDigest("a" * 64)),)
    seeds = (DetectorSeed(0), DetectorSeed(1), DetectorSeed(2))

    metadata = capture_reproducibility_metadata(seeds, checksums)

    assert metadata.os_platform.value
    assert metadata.python_version.value
    assert metadata.numpy_version.value
    assert metadata.scipy_version.value
    assert metadata.torch_version.value
    assert metadata.cpu_model.value
    assert metadata.ram_total.value >= 0
    assert metadata.solver_backend is SolverBackend.SCIPY_MILP
    assert metadata.seeds == seeds
    assert metadata.dataset_checksums == checksums


def test_cuda_metadata_is_consistent_with_explicit_availability_enum() -> None:
    metadata = capture_reproducibility_metadata((DetectorSeed(0),), ())

    if metadata.cuda_availability is CudaAvailability.AVAILABLE:
        assert metadata.cuda_version is not None
        assert metadata.gpu_name is not None
    else:
        assert metadata.cuda_version is None
        assert metadata.gpu_name is None
