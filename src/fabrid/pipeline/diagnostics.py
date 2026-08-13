from __future__ import annotations

from dataclasses import dataclass

from fabrid.domain.enums import AllocationPolicy
from fabrid.domain.values import (
    BudgetUsageRatio,
    DetectorSeed,
    FalsePositiveRate,
    MacroRecall,
    WorstClientRecall,
)
from fabrid.protocol.models import BudgetLevel


@dataclass(frozen=True, slots=True)
class DiagnosticPolicyEvidence:
    detector_seed: DetectorSeed
    budget: BudgetLevel
    policy: AllocationPolicy
    macro_recall: MacroRecall
    worst_client_recall: WorstClientRecall
    federation_fpr: FalsePositiveRate
    budget_usage: BudgetUsageRatio | None


@dataclass(frozen=True, slots=True)
class SeedDiagnosticEvidence:
    detector_seed: DetectorSeed
    cells: tuple[DiagnosticPolicyEvidence, ...]
