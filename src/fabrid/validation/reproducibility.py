from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import scipy
import torch

from fabrid.domain.enums import DatasetId, SolverBackend
from fabrid.domain.identifiers import ArtifactDigest, GitCommit
from fabrid.domain.values import DetectorSeed

_BYTES_PER_KIB = 1024


class CudaAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class EnvironmentText:
    value: str

    def __post_init__(self) -> None:
        if not self.value.strip():
            raise ValueError("environment text must not be empty")


@dataclass(frozen=True, slots=True)
class MemoryBytes:
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("memory bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class DatasetChecksum:
    dataset_id: DatasetId
    digest: ArtifactDigest


@dataclass(frozen=True, slots=True)
class ReproducibilityMetadata:
    os_platform: EnvironmentText
    python_version: EnvironmentText
    numpy_version: EnvironmentText
    scipy_version: EnvironmentText
    torch_version: EnvironmentText
    cuda_availability: CudaAvailability
    cuda_version: EnvironmentText | None
    gpu_name: EnvironmentText | None
    cpu_model: EnvironmentText
    ram_total: MemoryBytes
    solver_backend: SolverBackend
    git_commit: GitCommit
    command_line: EnvironmentText
    seeds: tuple[DetectorSeed, ...]
    dataset_checksums: tuple[DatasetChecksum, ...]

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("reproducibility metadata requires detector seeds")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("reproducibility metadata contains duplicate seeds")
        dataset_ids = tuple(checksum.dataset_id for checksum in self.dataset_checksums)
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("reproducibility metadata contains duplicate dataset checksums")


def _git_commit() -> GitCommit:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        capture_output=True,
        text=True,
        check=True,
    )
    return GitCommit(result.stdout.strip())


def _cpu_model() -> EnvironmentText:
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return EnvironmentText(line.split(":", 1)[1].strip())
        except OSError:
            pass
    return EnvironmentText(platform.processor() or "unknown")


def _ram_total() -> MemoryBytes:
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        return MemoryBytes(int(line.split()[1]) * _BYTES_PER_KIB)
        except OSError:
            pass
    return MemoryBytes(0)


def capture_reproducibility_metadata(
    seeds: tuple[DetectorSeed, ...],
    dataset_checksums: tuple[DatasetChecksum, ...],
) -> ReproducibilityMetadata:
    cuda_available = torch.cuda.is_available()
    return ReproducibilityMetadata(
        os_platform=EnvironmentText(platform.platform()),
        python_version=EnvironmentText(platform.python_version()),
        numpy_version=EnvironmentText(np.__version__),
        scipy_version=EnvironmentText(scipy.__version__),
        torch_version=EnvironmentText(torch.__version__),
        cuda_availability=(
            CudaAvailability.AVAILABLE if cuda_available else CudaAvailability.UNAVAILABLE
        ),
        cuda_version=(
            None
            if not cuda_available or torch.version.cuda is None
            else EnvironmentText(torch.version.cuda)
        ),
        gpu_name=(
            EnvironmentText(torch.cuda.get_device_name(0)) if cuda_available else None
        ),
        cpu_model=_cpu_model(),
        ram_total=_ram_total(),
        solver_backend=SolverBackend.SCIPY_MILP,
        git_commit=_git_commit(),
        command_line=EnvironmentText(" ".join(sys.argv) or "python"),
        seeds=seeds,
        dataset_checksums=dataset_checksums,
    )
