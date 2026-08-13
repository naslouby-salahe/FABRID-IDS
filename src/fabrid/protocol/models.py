from __future__ import annotations

from dataclasses import dataclass

from fabrid.domain.enums import (
    AttackFoldId,
    BotnetFamily,
    BudgetId,
    DecisionOperator,
    ExperimentalUnit,
    ExperimentVariantId,
    FallbackPolicy,
    OptimizationVariableKind,
    RetrainingPolicy,
    SolverBackend,
    ThresholdTiePolicy,
)
from fabrid.domain.identifiers import AttackSubtypeId
from fabrid.domain.values import (
    BatchSize,
    BudgetUsageRatio,
    DetectorSeed,
    DurationSeconds,
    EventRatePerClientHour,
    FalsePositiveBudget,
    FederatedRoundCount,
    LayerWidth,
    LearningRate,
    LocalEpochCount,
    PercentagePoints,
    Probability,
    RowCount,
    SolverGap,
    TargetFalsePositiveRate,
)


@dataclass(frozen=True, slots=True)
class ScoreContract:
    decision_operator: DecisionOperator
    tie_policy: ThresholdTiePolicy


@dataclass(frozen=True, slots=True)
class DetectorHyperparameters:
    hidden_layers: tuple[LayerWidth, ...]
    learning_rate: LearningRate
    local_epochs: LocalEpochCount
    rounds: FederatedRoundCount
    batch_size: BatchSize

    def __post_init__(self) -> None:
        if not self.hidden_layers:
            raise ValueError("detector requires at least one hidden layer")


@dataclass(frozen=True, slots=True)
class DetectorProtocol:
    seeds: tuple[DetectorSeed, ...]
    retraining_policy: RetrainingPolicy
    hyperparameters: DetectorHyperparameters

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("detector protocol requires at least one seed")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("detector seeds must be unique")


@dataclass(frozen=True, slots=True)
class BudgetLevel:
    budget_id: BudgetId
    value: FalsePositiveBudget


@dataclass(frozen=True, slots=True)
class BenignSplitFractions:
    train_end: Probability
    frontier_end: Probability
    final_cal_end: Probability

    def __post_init__(self) -> None:
        if not (
            self.train_end.value
            <= self.frontier_end.value
            <= self.final_cal_end.value
        ):
            raise ValueError("benign split boundaries must be monotonically increasing")


@dataclass(frozen=True, slots=True)
class AttackSplitFraction:
    validation_end: Probability


@dataclass(frozen=True, slots=True)
class UtilityEligibility:
    minimum_attack_validation_rows: RowCount
    minimum_eligible_subtypes: RowCount
    minimum_rows_per_subtype: RowCount
    fallback_policy: FallbackPolicy


@dataclass(frozen=True, slots=True)
class SolverSettings:
    backend: SolverBackend
    variable_kind: OptimizationVariableKind
    requested_gap: SolverGap
    time_limit: DurationSeconds
    accepted_gap: SolverGap


@dataclass(frozen=True, slots=True)
class StatisticsProtocol:
    experimental_unit: ExperimentalUnit
    sign_flip_enumeration: RowCount
    significance: Probability
    holm_family_size: RowCount
    bootstrap_resamples: RowCount


@dataclass(frozen=True, slots=True)
class FabridMacroGate:
    minimum_macro_recall_gain: PercentagePoints
    minimum_passing_budgets: RowCount
    total_budgets: RowCount


@dataclass(frozen=True, slots=True)
class FabridMinimaxGate:
    minimum_worst_client_recall_gain: PercentagePoints
    maximum_macro_recall_loss: PercentagePoints
    minimum_passing_budgets: RowCount
    total_budgets: RowCount


@dataclass(frozen=True, slots=True)
class BudgetComplianceGate:
    maximum_median_usage: BudgetUsageRatio
    seed_usage_limit: BudgetUsageRatio
    minimum_seed_fraction_below_limit: Probability


@dataclass(frozen=True, slots=True)
class PracticalGates:
    fabrid_macro: FabridMacroGate
    fabrid_minimax: FabridMinimaxGate
    budget_compliance: BudgetComplianceGate


@dataclass(frozen=True, slots=True)
class EventGateSensitivity:
    dilation: tuple[DurationSeconds, ...]
    merge_gap: tuple[DurationSeconds, ...]
    minimum_event_length: tuple[DurationSeconds, ...]
    cooldown: tuple[DurationSeconds, ...]


@dataclass(frozen=True, slots=True)
class EventGate:
    dilation: DurationSeconds
    merge_gap: DurationSeconds
    minimum_event_length: DurationSeconds
    cooldown: DurationSeconds
    maximum_alarm_duty: Probability
    budgets_per_client_hour: tuple[EventRatePerClientHour, ...]
    sensitivity: EventGateSensitivity


@dataclass(frozen=True, slots=True)
class AttackFold:
    fold_id: AttackFoldId
    subtypes: tuple[AttackSubtypeId, ...]

    def __post_init__(self) -> None:
        if not self.subtypes:
            raise ValueError("attack fold must contain at least one subtype")
        if len(set(self.subtypes)) != len(self.subtypes):
            raise ValueError("attack fold contains duplicate subtypes")


@dataclass(frozen=True, slots=True)
class AttackFoldRotation:
    variant_id: ExperimentVariantId
    validation_fold: AttackFoldId
    test_folds: tuple[AttackFoldId, ...]

    def __post_init__(self) -> None:
        if not self.test_folds:
            raise ValueError("attack-fold rotation requires at least one test fold")
        if self.validation_fold in self.test_folds:
            raise ValueError("validation fold may not also be a test fold")
        if len(set(self.test_folds)) != len(self.test_folds):
            raise ValueError("attack-fold rotation contains duplicate test folds")


@dataclass(frozen=True, slots=True)
class BotnetFamilySubtypes:
    family: BotnetFamily
    subtypes: tuple[AttackSubtypeId, ...]

    def __post_init__(self) -> None:
        if not self.subtypes:
            raise ValueError("botnet family must contain at least one subtype")
        if len(set(self.subtypes)) != len(self.subtypes):
            raise ValueError("botnet family contains duplicate subtypes")


@dataclass(frozen=True, slots=True)
class BotnetFamilyDirection:
    variant_id: ExperimentVariantId
    validation_family: BotnetFamily
    test_family: BotnetFamily

    def __post_init__(self) -> None:
        if self.validation_family is self.test_family:
            raise ValueError("botnet-family validation and test families must differ")


@dataclass(frozen=True, slots=True)
class GeneralizationProtocol:
    folds: tuple[AttackFold, ...]
    rotations: tuple[AttackFoldRotation, ...]
    families: tuple[BotnetFamilySubtypes, ...]
    family_directions: tuple[BotnetFamilyDirection, ...]
    botnet_eligible_client_count: RowCount

    def __post_init__(self) -> None:
        fold_ids = tuple(fold.fold_id for fold in self.folds)
        rotation_ids = tuple(rotation.variant_id for rotation in self.rotations)
        family_ids = tuple(family.family for family in self.families)
        direction_ids = tuple(direction.variant_id for direction in self.family_directions)
        if len(set(fold_ids)) != len(fold_ids):
            raise ValueError("generalization protocol contains duplicate fold identities")
        if len(set(rotation_ids)) != len(rotation_ids):
            raise ValueError("generalization protocol contains duplicate rotation identities")
        if len(set(family_ids)) != len(family_ids):
            raise ValueError("generalization protocol contains duplicate botnet families")
        if len(set(direction_ids)) != len(direction_ids):
            raise ValueError("generalization protocol contains duplicate family directions")

    def fold(self, fold_id: AttackFoldId) -> AttackFold:
        for fold in self.folds:
            if fold.fold_id is fold_id:
                return fold
        raise KeyError(fold_id.value)

    def family(self, family: BotnetFamily) -> BotnetFamilySubtypes:
        for entry in self.families:
            if entry.family is family:
                return entry
        raise KeyError(family.value)


@dataclass(frozen=True, slots=True)
class AlphaGrid:
    values: tuple[TargetFalsePositiveRate, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("alpha grid must not be empty")
        raw = tuple(value.value for value in self.values)
        if raw != tuple(sorted(raw)):
            raise ValueError("alpha grid must be sorted")
        if len(set(raw)) != len(raw):
            raise ValueError("alpha grid values must be unique")
        if raw[0] != 0.0:
            raise ValueError("alpha grid must start at zero")

    @property
    def maximum(self) -> TargetFalsePositiveRate:
        return self.values[-1]


@dataclass(frozen=True, slots=True)
class FabridProtocol:
    score_contract: ScoreContract
    detector: DetectorProtocol
    alpha_grid: AlphaGrid
    budgets: tuple[BudgetLevel, ...]
    benign_splits: BenignSplitFractions
    attack_split: AttackSplitFraction
    utility_eligibility: UtilityEligibility
    solver: SolverSettings
    statistics: StatisticsProtocol
    practical_gates: PracticalGates
    generalization: GeneralizationProtocol
    allocation_sensitivity_replicates: RowCount
    conservative_utility_confidence: Probability
    event_gate: EventGate

    def __post_init__(self) -> None:
        if not self.budgets:
            raise ValueError("protocol requires at least one false-positive budget")
        budget_ids = tuple(level.budget_id for level in self.budgets)
        budget_values = tuple(level.value for level in self.budgets)
        if len(set(budget_ids)) != len(budget_ids):
            raise ValueError("false-positive budget identities must be unique")
        if len(set(budget_values)) != len(budget_values):
            raise ValueError("false-positive budget values must be unique")
        if any(level.value.value > self.alpha_grid.maximum.value for level in self.budgets):
            raise ValueError("false-positive budget may not exceed alpha-grid maximum")
