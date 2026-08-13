from __future__ import annotations

from dataclasses import dataclass


def _require_unit_interval(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _require_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _require_positive_integer(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(frozen=True, slots=True)
class DetectorSeed:
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(f"detector seed must be non-negative, got {self.value}")


@dataclass(frozen=True, slots=True)
class AnalysisSeed:
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(f"analysis seed must be non-negative, got {self.value}")


@dataclass(frozen=True, slots=True)
class RowCount:
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(f"row count must be non-negative, got {self.value}")


@dataclass(frozen=True, slots=True)
class FeatureCount:
    value: int

    def __post_init__(self) -> None:
        _require_positive_integer("feature count", self.value)


@dataclass(frozen=True, slots=True)
class LayerWidth:
    value: int

    def __post_init__(self) -> None:
        _require_positive_integer("layer width", self.value)


@dataclass(frozen=True, slots=True)
class LocalEpochCount:
    value: int

    def __post_init__(self) -> None:
        _require_positive_integer("local epoch count", self.value)


@dataclass(frozen=True, slots=True)
class FederatedRoundCount:
    value: int

    def __post_init__(self) -> None:
        _require_positive_integer("federated round count", self.value)


@dataclass(frozen=True, slots=True)
class BatchSize:
    value: int

    def __post_init__(self) -> None:
        _require_positive_integer("batch size", self.value)


@dataclass(frozen=True, slots=True)
class LearningRate:
    value: float

    def __post_init__(self) -> None:
        if self.value <= 0.0:
            raise ValueError(f"learning rate must be positive, got {self.value}")


@dataclass(frozen=True, slots=True)
class SourceRowIndex:
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(f"source row index must be non-negative, got {self.value}")


@dataclass(frozen=True, slots=True)
class EventTimestamp:
    value: float


@dataclass(frozen=True, slots=True)
class DurationSeconds:
    value: float

    def __post_init__(self) -> None:
        _require_non_negative("duration", self.value)


@dataclass(frozen=True, slots=True)
class EventRatePerClientHour:
    value: float

    def __post_init__(self) -> None:
        _require_non_negative("event rate per client hour", self.value)


@dataclass(frozen=True, slots=True)
class PercentagePoints:
    value: float

    def __post_init__(self) -> None:
        _require_non_negative("percentage points", self.value)


@dataclass(frozen=True, slots=True)
class MetricDifference:
    value: float


@dataclass(frozen=True, slots=True)
class FalseAlertCount:
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(f"false-alert count must be non-negative, got {self.value}")


@dataclass(frozen=True, slots=True)
class Probability:
    value: float

    def __post_init__(self) -> None:
        _require_unit_interval("probability", self.value)


@dataclass(frozen=True, slots=True)
class DetectionUtility:
    value: float

    def __post_init__(self) -> None:
        _require_unit_interval("detection utility", self.value)


@dataclass(frozen=True, slots=True)
class FalsePositiveRate:
    value: float

    def __post_init__(self) -> None:
        _require_unit_interval("false-positive rate", self.value)


@dataclass(frozen=True, slots=True)
class TruePositiveRate:
    value: float

    def __post_init__(self) -> None:
        _require_unit_interval("true-positive rate", self.value)


@dataclass(frozen=True, slots=True)
class PooledRecall:
    value: float

    def __post_init__(self) -> None:
        _require_unit_interval("pooled recall", self.value)


@dataclass(frozen=True, slots=True)
class MacroF1:
    value: float

    def __post_init__(self) -> None:
        _require_unit_interval("macro F1", self.value)


@dataclass(frozen=True, slots=True)
class BalancedAccuracy:
    value: float

    def __post_init__(self) -> None:
        _require_unit_interval("balanced accuracy", self.value)


@dataclass(frozen=True, slots=True)
class TargetFalsePositiveRate:
    value: float

    def __post_init__(self) -> None:
        _require_unit_interval("target false-positive rate", self.value)


@dataclass(frozen=True, slots=True)
class FalsePositiveBudget:
    value: float

    def __post_init__(self) -> None:
        _require_unit_interval("false-positive budget", self.value)


@dataclass(frozen=True, slots=True)
class ClientWeight:
    value: float

    def __post_init__(self) -> None:
        _require_unit_interval("client weight", self.value)


@dataclass(frozen=True, slots=True)
class Threshold:
    value: float


@dataclass(frozen=True, slots=True)
class AnomalyScore:
    value: float


@dataclass(frozen=True, slots=True)
class MacroRecall:
    value: float

    def __post_init__(self) -> None:
        _require_unit_interval("macro recall", self.value)


@dataclass(frozen=True, slots=True)
class WorstClientRecall:
    value: float

    def __post_init__(self) -> None:
        _require_unit_interval("worst-client recall", self.value)


@dataclass(frozen=True, slots=True)
class BudgetUsageRatio:
    value: float

    def __post_init__(self) -> None:
        _require_non_negative("budget-usage ratio", self.value)


@dataclass(frozen=True, slots=True)
class BudgetViolationRatio:
    value: float

    def __post_init__(self) -> None:
        _require_non_negative("budget-violation ratio", self.value)


@dataclass(frozen=True, slots=True)
class CoefficientOfVariation:
    value: float

    def __post_init__(self) -> None:
        _require_non_negative("coefficient of variation", self.value)


@dataclass(frozen=True, slots=True)
class SolverObjective:
    value: float


@dataclass(frozen=True, slots=True)
class SolverGap:
    value: float

    def __post_init__(self) -> None:
        _require_non_negative("solver gap", self.value)


@dataclass(frozen=True, slots=True)
class SolverRuntimeMilliseconds:
    value: float

    def __post_init__(self) -> None:
        _require_non_negative("solver runtime", self.value)
