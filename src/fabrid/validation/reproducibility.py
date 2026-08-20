from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

import numpy as np
import polars as pl
import scipy
import torch
from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator

from fabrid.artifacts.json import write_typed_json
from fabrid.artifacts.paths import ArtifactPaths
from fabrid.config import (
    DetectorSeed,
    EnvironmentText,
    GitCommit,
    MemoryBytes,
    NonNegativeInt,
)
from fabrid.validation.datasets import DatasetChecksum

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProgressState:
    phase: EnvironmentText
    completed: NonNegativeInt
    total: NonNegativeInt
    detail: EnvironmentText
    updated_at: EnvironmentText


def report_progress(
    logger: logging.Logger,
    paths: ArtifactPaths,
    phase: EnvironmentText,
    completed: NonNegativeInt,
    total: NonNegativeInt,
    detail: EnvironmentText = "",
) -> None:
    percent = 100.0 * completed / total if total else 0.0
    state_detail = detail if detail else "in progress"
    logger.info("[PROGRESS] %s %d/%d (%.1f%%) %s", phase, completed, total, percent, state_detail)
    write_typed_json(
        ProgressState(
            phase=phase,
            completed=completed,
            total=total,
            detail=state_detail,
            updated_at=now_iso(),
        ),
        TypeAdapter(ProgressState),
        paths.progress_state_path(),
    )


class CudaAvailability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ReproducibilityMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    os_platform: EnvironmentText
    python_version: EnvironmentText
    numpy_version: EnvironmentText
    scipy_version: EnvironmentText
    polars_version: EnvironmentText
    torch_version: EnvironmentText
    cuda_availability: CudaAvailability
    cuda_version: EnvironmentText | None
    gpu_name: EnvironmentText | None
    cpu_model: EnvironmentText
    ram_total_bytes: MemoryBytes
    solver_backend: EnvironmentText
    git_commit: GitCommit
    command_line: EnvironmentText
    started_at: EnvironmentText
    finished_at: EnvironmentText | None
    seeds: tuple[DetectorSeed, ...]
    dataset_checksums: tuple[DatasetChecksum, ...]

    @model_validator(mode="after")
    def _validate(self) -> ReproducibilityMetadata:
        if not self.seeds:
            raise ValueError("reproducibility metadata requires detector seeds")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("reproducibility metadata contains duplicate seeds")
        dataset_ids = tuple(checksum.dataset_id for checksum in self.dataset_checksums)
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError("reproducibility metadata contains duplicate dataset checksums")
        return self


def resolve_git_commit() -> GitCommit:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _cpu_model() -> EnvironmentText:
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or "unknown"


def _ram_total_bytes() -> MemoryBytes:
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
    return 0


def capture_reproducibility_metadata(
    seeds: tuple[DetectorSeed, ...],
    dataset_checksums: tuple[DatasetChecksum, ...],
    command_line: EnvironmentText,
    started_at: EnvironmentText,
    finished_at: EnvironmentText | None,
) -> ReproducibilityMetadata:
    cuda_available = torch.cuda.is_available()
    return ReproducibilityMetadata(
        os_platform=platform.platform(),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        polars_version=pl.__version__,
        torch_version=torch.__version__,
        cuda_availability=(
            CudaAvailability.AVAILABLE if cuda_available else CudaAvailability.UNAVAILABLE
        ),
        cuda_version=torch.version.cuda if cuda_available else None,
        gpu_name=torch.cuda.get_device_name(0) if cuda_available else None,
        cpu_model=_cpu_model(),
        ram_total_bytes=_ram_total_bytes(),
        solver_backend="scipy.optimize.milp",
        git_commit=resolve_git_commit(),
        command_line=command_line,
        started_at=started_at,
        finished_at=finished_at,
        seeds=seeds,
        dataset_checksums=dataset_checksums,
    )


def now_iso() -> EnvironmentText:
    return datetime.now(UTC).isoformat(timespec="seconds")
