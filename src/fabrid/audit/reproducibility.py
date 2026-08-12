"""Reproducibility metadata capture: environment/hardware/software fingerprint persisted
alongside every experiment run, so results can be traced back to the exact conditions that
produced them.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import scipy
import torch

_BYTES_PER_KIB = 1024


@dataclass(frozen=True, slots=True)
class DatasetChecksum:
    dataset_id: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ReproducibilityMetadata:
    os_platform: str
    python_version: str
    numpy_version: str
    scipy_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: str | None
    gpu_name: str | None
    cpu_model: str
    ram_total_bytes: int
    solver_backend: str
    git_commit: str
    command_line: str
    seeds: tuple[int, ...]
    dataset_checksums: tuple[DatasetChecksum, ...]


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _cpu_model() -> str:
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _ram_total_bytes() -> int:
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * _BYTES_PER_KIB
        except OSError:
            pass
    return 0


def capture_reproducibility_metadata(
    seeds: tuple[int, ...],
    dataset_checksums: tuple[DatasetChecksum, ...],
) -> ReproducibilityMetadata:
    cuda_available = torch.cuda.is_available()
    return ReproducibilityMetadata(
        os_platform=platform.platform(),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_version=torch.version.cuda if cuda_available else None,
        gpu_name=torch.cuda.get_device_name(0) if cuda_available else None,
        cpu_model=_cpu_model(),
        ram_total_bytes=_ram_total_bytes(),
        solver_backend=f"scipy.optimize.milp (HiGHS, scipy {scipy.__version__})",
        git_commit=_git_commit(),
        command_line=" ".join(sys.argv),
        seeds=seeds,
        dataset_checksums=dataset_checksums,
    )
