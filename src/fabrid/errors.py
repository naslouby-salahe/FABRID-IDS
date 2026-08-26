from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fabrid.config import EnvironmentText, ExperimentId, FailureReason


class FabridError(Exception):
    def __init__(self, message: EnvironmentText) -> None:
        self.message = message
        super().__init__(message)


class ConfigurationError(FabridError):
    def __init__(self, message: EnvironmentText, path: Path | None = None) -> None:
        self.path = path
        super().__init__(message)


class DatasetError(FabridError):
    def __init__(self, message: EnvironmentText, path: Path | None = None) -> None:
        self.path = path
        super().__init__(message)


class ArtifactError(FabridError):
    def __init__(self, message: EnvironmentText, path: Path | None = None) -> None:
        self.path = path
        super().__init__(message)


class ExecutionError(FabridError):
    def __init__(
        self,
        message: EnvironmentText,
        experiment_id: ExperimentId | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        super().__init__(message)


class SolverError(FabridError):
    def __init__(self, message: EnvironmentText) -> None:
        super().__init__(message)


class SolverInvalidError(FabridError):
    def __init__(self, reason: FailureReason) -> None:
        self.reason = reason
        super().__init__(reason)
