from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

TIGHT_TOLERANCE = 1.0e-12
PERCENTAGE_POINTS_PER_UNIT = 100.0
LOCAL_TARGET_RATE_CAP = 0.05
COARSE_TOLERANCE = 1.0e-9
LEXICOGRAPHIC_TOLERANCE = 1.0e-4
MIP_FEASIBILITY_TOLERANCE = 1.0e-8
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
DetectorSeed = NonNegativeInt
AnalysisSeed = NonNegativeInt
RowCount = NonNegativeInt
ZERO_ROW_COUNT: RowCount = 0
RowNumber = Annotated[int, Field(ge=1)]
FieldCount = NonNegativeInt
ClientCount = PositiveInt
CandidateCount = PositiveInt
CandidateIndex = NonNegativeInt
ReplicateIndex = NonNegativeInt
ArtifactCount = NonNegativeInt
DecisionVariableIndex = NonNegativeInt
SourceRowIndex = NonNegativeInt
FeatureCount = PositiveInt
LayerWidth = PositiveInt
LocalEpochCount = PositiveInt
FederatedRoundCount = PositiveInt
BatchSize = PositiveInt
SeedCount = PositiveInt
WorkerCount = PositiveInt
SplitFraction = Annotated[float, Field(ge=0.0, le=1.0)]
WeightGamma = NonNegativeFloat
FalseAlertCount = NonNegativeInt
ByteCount = NonNegativeInt
BitCount = PositiveInt
MemoryBytes = NonNegativeInt
LearningRate = PositiveFloat
DurationSeconds = NonNegativeFloat
EventRatePerClientHour = NonNegativeFloat
PercentagePoints = NonNegativeFloat
SolverGap = NonNegativeFloat
SolverRuntimeMilliseconds = NonNegativeFloat
BudgetUsageRatio = NonNegativeFloat
CoefficientOfVariation = NonNegativeFloat
UtilityDispersion = NonNegativeFloat
IncrementalBudgetCost = NonNegativeFloat
TargetFalsePositiveRate = Probability
FalsePositiveBudget = Probability
ClientWeight = Probability
DetectionUtility = Probability
TruePositiveRate = Probability
FalsePositiveRate = Probability
MacroRecall = Probability
WorstClientRecall = Probability
BalancedAccuracy = Probability
F1Score = Probability
Threshold = Annotated[float, Field()]
AnomalyScore = Annotated[float, Field()]
MetricDifference = Annotated[float, Field()]
PercentagePointDifference = Annotated[float, Field()]
SolverObjective = Annotated[float, Field()]
MarginalEfficiency = Annotated[float, Field()]
UtilityDifference = Annotated[float, Field()]
EventTimestamp = Annotated[float, Field()]
Identifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        strip_whitespace=True,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._: -]*$",
    ),
]
ClientId = Identifier
AttackSubtypeId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        strip_whitespace=True,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    ),
]
SampleId = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
SourceFileId = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
PREPROCESSED_SOURCE_FILE: SourceFileId = "preprocessed"
EXTERNAL_SOURCE_FILE: SourceFileId = "external"
ColumnName = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
EnvironmentText = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
CalibrationExceedanceLattice = EnvironmentText
FailureReason = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
ProtocolVersion = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
RelativePath = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
ArtifactName = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        strip_whitespace=True,
        pattern=r"^[a-z0-9][a-z0-9._-]*$",
    ),
]
ArtifactDigest = Annotated[
    str,
    StringConstraints(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
]
GitCommit = Annotated[
    str,
    StringConstraints(
        min_length=7,
        max_length=40,
        strip_whitespace=True,
        pattern=r"^[0-9a-f]{7,40}$",
    ),
]


class DatasetId(StrEnum):
    NBAIOT = "nbaiot"
    CIC_IOT_DIAD = "cic_iot_diad"
    GOTHAM = "gotham"
    CICIOMT = "ciciomt"


class BudgetId(StrEnum):
    FALSE_POSITIVE_0P001 = "false_positive_0p001"
    FALSE_POSITIVE_0P0025 = "false_positive_0p0025"
    FALSE_POSITIVE_0P005 = "false_positive_0p005"
    FALSE_POSITIVE_0P010 = "false_positive_0p010"
    FALSE_POSITIVE_0P020 = "false_positive_0p020"


class ExperimentId(StrEnum):
    MATCHED_BUDGET = "matched_budget"
    ATTACK_SUBTYPE_DISJOINT = "attack_subtype_disjoint"
    BOTNET_FAMILY_DISJOINT = "botnet_family_disjoint"
    WEIGHT_SENSITIVITY = "weight_sensitivity"
    ALLOCATION_STABILITY = "allocation_stability"
    CONSERVATIVE_UTILITY = "conservative_utility"
    UTILITY_HETEROGENEITY = "utility_heterogeneity"
    EXTERNAL_REPLICATION = "external_replication"
    EVENT_LEVEL = "event_level"


class ExperimentVariantId(StrEnum):
    PRIMARY = "primary"
    ATTACK_SUBTYPE_FOLD_0 = "attack_subtype_fold_0"
    ATTACK_SUBTYPE_FOLD_1 = "attack_subtype_fold_1"
    ATTACK_SUBTYPE_FOLD_2 = "attack_subtype_fold_2"
    BOTNET_BASHLITE_TO_MIRAI = "botnet_bashlite_to_mirai"
    BOTNET_MIRAI_TO_BASHLITE = "botnet_mirai_to_bashlite"
    WEIGHT_GAMMA_0 = "weight_gamma_0"
    WEIGHT_GAMMA_0P5 = "weight_gamma_0p5"
    WEIGHT_GAMMA_1 = "weight_gamma_1"
    WEIGHT_GAMMA_1P5 = "weight_gamma_1p5"
    CONSERVATIVE_MACRO = "conservative_macro"
    CONSERVATIVE_CVAR = "conservative_cvar"
    EXTERNAL_PRIMARY = "external_primary"
    EVENT_PRIMARY = "event_primary"


class AllocationPolicy(StrEnum):
    EQ_FPR = "eq_fpr"
    EQ_ALERT = "eq_alert"
    GREEDY = "greedy"
    FABRID_MACRO = "fabrid_macro"
    FABRID_CVAR = "fabrid_cvar"
    FABRID_MACRO_FINITE_SAFE = "fabrid_macro_finite_safe"
    FABRID_CVAR_FINITE_SAFE = "fabrid_cvar_finite_safe"
    FABRID_MACRO_NO_RATE_MINIMIZATION = "fabrid_macro_no_rate_minimization"
    COMMON_RATE_MACRO = "common_rate_macro"
    EQ_FPR_MATCHED_MACRO = "eq_fpr_matched_macro"
    EQ_FPR_MATCHED_CVAR = "eq_fpr_matched_cvar"
    POOLED_SHARED = "pooled_shared"


class ScorerDefinition(StrEnum):
    AE_RECONSTRUCTION = "ae_reconstruction"
    AUXILIARY_F75 = "auxiliary_f75"
    AE_PLUS_AUXILIARY_F75 = "ae_plus_auxiliary_f75"


class WeightMode(StrEnum):
    EQUAL_CLIENT = "equal_client"
    DATASET_COUNT_PROXY = "dataset_count_proxy"


class ExperimentalUnit(StrEnum):
    DETECTOR_SEED = "detector_seed"


class AttackFoldId(StrEnum):
    FOLD_0 = "fold_0"
    FOLD_1 = "fold_1"
    FOLD_2 = "fold_2"


class BotnetFamily(StrEnum):
    BASHLITE = "bashlite"
    MIRAI = "mirai"


class CampaignAnalysisId(StrEnum):
    PRIMARY_INFERENCE = "primary_inference"
    PRACTICAL_GATES = "practical_gates"


class FigureId(StrEnum):
    ARCHITECTURE = "architecture"
    BUDGET_MACRO_RECALL = "budget_macro_recall"
    BUDGET_WORST_CLIENT_RECALL = "budget_worst_client_recall"
    CLIENT_OPERATING_POINTS = "client_operating_points"
    ALLOCATION_STABILITY = "allocation_stability"
    CLIENT_UTILITY_CURVES = "client_utility_curves"
    GAIN_VS_HETEROGENEITY = "gain_vs_heterogeneity"
    REALIZED_FPR_LINES = "realized_fpr_lines"
    MACRO_ALLOCATION_HEATMAP = "macro_allocation_heatmap"


class AnalysisArtifactId(StrEnum):
    STABILITY_REPLICATES = "replicates"
    STABILITY_SUMMARY = "summary"
    UTILITY_HETEROGENEITY = "utility_heterogeneity"
    CONSERVATIVE_UTILITY_CURVES = "utility_curves"
    PAYLOAD_MEASUREMENT = "payload_measurement"
    EQUAL_RATE_CALIBRATION_FRONTIER = "equal_rate_calibration_frontier"


class ReportTableId(StrEnum):
    BUDGET_POLICY_SUMMARY = "budget_policy_summary"
    GENERALIZATION_ROTATIONS = "generalization_rotations"
    EXTERNAL_REPLICATION = "external_replication"
    SYSTEM_OVERHEAD = "system_overhead"
    DATASET_POPULATIONS = "dataset_populations"


class Label(StrEnum):
    BENIGN = "benign"
    ATTACK = "attack"


class BenignSplit(StrEnum):
    TRAIN = "benign_train"
    FRONTIER = "benign_frontier"
    FINAL_CAL = "benign_final_cal"
    TEST = "benign_test"


class AttackSplit(StrEnum):
    VALIDATION = "attack_validation"
    TEST = "attack_test"


class PreprocessedColumn(StrEnum):
    CLIENT_ID = "client_id"
    SOURCE_FILE = "source_file"
    SOURCE_ROW = "source_row"
    TIMESTAMP = "timestamp"
    SPLIT = "split"
    ATTACK_SUBTYPE = "attack_subtype"


class GateStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"


class EvidenceAvailability(StrEnum):
    AVAILABLE = "available"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class EventCriterionId(StrEnum):
    IMMUTABLE_CLIENT_ID = "immutable_client_id"
    PACKET_TIMESTAMP = "packet_timestamp"
    INTERVAL_PROVENANCE = "interval_provenance"
    DETERMINISTIC_SCORE_ASSOCIATION = "deterministic_score_association"
    WITHIN_CLIENT_ORDERING = "within_client_ordering"
    OBSERVATION_DURATION = "observation_duration"
    NON_OVERLAPPING_EVALUATION_PERIOD = "non_overlapping_evaluation_period"


class ReplicationEvidenceLevel(StrEnum):
    CONFIRMATORY = "confirmatory"
    SUPPORTIVE = "supportive"


class PreprocessingStatus(StrEnum):
    REUSE = "REUSE"
    BUILD = "BUILD"
    REBUILD = "REBUILD"


FileGlob = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class DetectorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    hidden_layers: tuple[LayerWidth, ...]
    learning_rate: PositiveFloat
    local_epochs: LocalEpochCount
    rounds: FederatedRoundCount
    batch_size: BatchSize
    score_batch_size: PositiveInt
    seeds: tuple[DetectorSeed, ...]

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.hidden_layers:
            raise ValueError("detector requires at least one hidden layer")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("detector seeds must be unique")
        return self


class BudgetLevel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    budget_id: BudgetId
    value: FalsePositiveBudget


class BenignSplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    train_end: Probability
    frontier_end: Probability
    final_cal_end: Probability

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not (self.train_end <= self.frontier_end <= self.final_cal_end):
            raise ValueError("benign split boundaries must be monotonically increasing")
        return self


class AttackSplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    validation_end: Probability


class UtilityEligibilityConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    minimum_attack_validation_rows: RowCount
    minimum_eligible_subtypes: RowCount
    minimum_rows_per_subtype: RowCount


class AuxiliaryScoreFeatureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    preprocessed_column: ColumnName
    source_column: ColumnName
    preprocessing: EnvironmentText
    interpretation: EnvironmentText
    development_selection_rule: EnvironmentText


class ScoringConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    primary_scorer: ScorerDefinition
    ablation_scorers: tuple[ScorerDefinition, ...]
    auxiliary_feature: AuxiliaryScoreFeatureConfig | None = None

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.ablation_scorers:
            raise ValueError("scoring requires at least one predeclared scorer")
        if len(set(self.ablation_scorers)) != len(self.ablation_scorers):
            raise ValueError("scorer ablation definitions must be unique")
        if self.primary_scorer not in self.ablation_scorers:
            raise ValueError("primary scorer must be included in scorer ablations")
        needs_auxiliary = self.primary_scorer is not ScorerDefinition.AE_RECONSTRUCTION
        if needs_auxiliary and self.auxiliary_feature is None:
            raise ValueError("auxiliary scorer requires an auxiliary-feature specification")
        if self.auxiliary_feature is None and any(
            scorer is not ScorerDefinition.AE_RECONSTRUCTION for scorer in self.ablation_scorers
        ):
            raise ValueError(
                "auxiliary scorer ablations require an auxiliary-feature specification"
            )
        return self


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    finite_sample_confidence: Probability

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not 0.0 < self.finite_sample_confidence < 1.0:
            raise ValueError("finite-sample confidence must be strictly between zero and one")
        return self


class SolverConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    requested_gap: SolverGap
    time_limit_seconds: DurationSeconds
    accepted_gap: SolverGap
    accepted_absolute_gap: SolverGap
    equal_alert_bisection_iterations: PositiveInt
    cvar_tail_fraction: Probability
    calibration_resolution_factor: NonNegativeFloat

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.time_limit_seconds <= 0:
            raise ValueError("solver time limit must be positive")
        return self


class PayloadSizingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    client_identifier_bytes: ByteCount
    nominal_count_bytes: ByteCount
    final_calibration_count_bytes: ByteCount
    validation_attack_count_bytes: ByteCount
    eligible_subtype_count_bytes: ByteCount
    flags_bytes: ByteCount
    protocol_digest_bytes: ByteCount
    utility_value_bytes: ByteCount
    server_response_bits: BitCount
    alpha_candidate_count: PositiveInt

    @property
    def metadata_envelope_bytes(self) -> ByteCount:
        return (
            self.client_identifier_bytes
            + self.nominal_count_bytes
            + self.final_calibration_count_bytes
            + self.validation_attack_count_bytes
            + self.eligible_subtype_count_bytes
            + self.flags_bytes
            + self.protocol_digest_bytes
        )


class StatisticsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    experimental_unit: ExperimentalUnit
    sign_flip_enumeration: RowCount
    significance: Probability
    holm_family_size: RowCount
    bootstrap_resamples: RowCount
    bootstrap_confidence: Probability


class FabridMacroGateConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    minimum_macro_recall: Probability
    minimum_passing_budgets: RowCount
    total_budgets: RowCount


class FabridCvarGateConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    minimum_worst_client_recall: Probability
    maximum_macro_recall_loss: PercentagePoints
    minimum_passing_budgets: RowCount
    total_budgets: RowCount


class BudgetComplianceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    maximum_median_usage: BudgetUsageRatio
    seed_usage_limit: BudgetUsageRatio
    minimum_seed_fraction_below_limit: Probability
    evaluated_budgets: tuple[BudgetId, ...]


class PracticalGateConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    fabrid_macro: FabridMacroGateConfig
    fabrid_cvar: FabridCvarGateConfig
    budget_compliance: BudgetComplianceConfig


class AttackFoldConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    fold_id: AttackFoldId
    subtypes: tuple[AttackSubtypeId, ...]

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.subtypes:
            raise ValueError("attack fold must contain at least one subtype")
        if len(set(self.subtypes)) != len(self.subtypes):
            raise ValueError("attack fold contains duplicate subtypes")
        return self


class AttackFoldRotationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    variant_id: ExperimentVariantId
    validation_fold: AttackFoldId
    test_folds: tuple[AttackFoldId, ...]

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.test_folds:
            raise ValueError("attack-fold rotation requires at least one test fold")
        if self.validation_fold in self.test_folds:
            raise ValueError("validation fold may not also be a test fold")
        if len(set(self.test_folds)) != len(self.test_folds):
            raise ValueError("attack-fold rotation contains duplicate test folds")
        return self


class BotnetFamilyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    family: BotnetFamily
    subtypes: tuple[AttackSubtypeId, ...]

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.subtypes:
            raise ValueError("botnet family must contain at least one subtype")
        if len(set(self.subtypes)) != len(self.subtypes):
            raise ValueError("botnet family contains duplicate subtypes")
        return self


class BotnetFamilyDirectionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    variant_id: ExperimentVariantId
    validation_family: BotnetFamily
    test_family: BotnetFamily

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.validation_family is self.test_family:
            raise ValueError("botnet-family validation and test families must differ")
        return self


class GeneralizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    folds: tuple[AttackFoldConfig, ...]
    rotations: tuple[AttackFoldRotationConfig, ...]
    families: tuple[BotnetFamilyConfig, ...]
    family_directions: tuple[BotnetFamilyDirectionConfig, ...]
    botnet_eligible_client_count: RowCount

    @model_validator(mode="after")
    def _validate(self) -> Self:
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
        return self

    def fold(self, fold_id: AttackFoldId) -> AttackFoldConfig:
        for fold in self.folds:
            if fold.fold_id is fold_id:
                return fold
        raise KeyError(fold_id.value)

    def family(self, family: BotnetFamily) -> BotnetFamilyConfig:
        for entry in self.families:
            if entry.family is family:
                return entry
        raise KeyError(family.value)


class WeightGammaVariantConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    gamma: WeightGamma
    variant_id: ExperimentVariantId


class EventSensitivityConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    dilation: tuple[DurationSeconds, ...]
    merge_gap: tuple[DurationSeconds, ...]
    minimum_event_length: tuple[DurationSeconds, ...]
    cooldown: tuple[DurationSeconds, ...]


class EventGateConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    dilation: DurationSeconds
    merge_gap: DurationSeconds
    minimum_event_length: DurationSeconds
    cooldown: DurationSeconds
    maximum_alarm_duty: Probability
    minimum_timestamp_parse_success: Probability
    maximum_out_of_order_fraction: Probability
    minimum_capture_seam_seconds: DurationSeconds
    budgets_per_client_hour: tuple[EventRatePerClientHour, ...]
    budget_bisection_iterations: PositiveInt
    sensitivity: EventSensitivityConfig


class SensitivityConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    allocation_sensitivity_replicates: RowCount
    stability_workers: WorkerCount
    conservative_utility_confidence: Probability
    stability_percentiles: tuple[Probability, Probability]
    weight_gamma_variants: tuple[WeightGammaVariantConfig, ...]


class FabridConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    protocol_version: ProtocolVersion
    dataset_id: DatasetId
    alpha_grid: tuple[TargetFalsePositiveRate, ...]
    budgets: tuple[BudgetLevel, ...]
    benign_splits: BenignSplitConfig
    attack_split: AttackSplitConfig
    utility_eligibility: UtilityEligibilityConfig
    solver: SolverConfig
    payload_sizing: PayloadSizingConfig
    statistics: StatisticsConfig
    practical_gates: PracticalGateConfig
    generalization: GeneralizationConfig
    event_gate: EventGateConfig
    sensitivity: SensitivityConfig
    detector: DetectorConfig
    scoring: ScoringConfig
    calibration: CalibrationConfig

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.budgets:
            raise ValueError("protocol requires at least one false-positive budget")
        budget_ids = tuple(level.budget_id for level in self.budgets)
        budget_values = tuple(level.value for level in self.budgets)
        if len(set(budget_ids)) != len(budget_ids):
            raise ValueError("false-positive budget identities must be unique")
        if len(set(budget_values)) != len(budget_values):
            raise ValueError("false-positive budget values must be unique")
        raw_grid = tuple(self.alpha_grid)
        if len(raw_grid) != self.payload_sizing.alpha_candidate_count:
            raise ValueError(
                f"alpha grid must contain {self.payload_sizing.alpha_candidate_count} values, "
                f"got {len(raw_grid)}"
            )
        if raw_grid != tuple(sorted(raw_grid)):
            raise ValueError("alpha grid must be sorted")
        if len(set(raw_grid)) != len(raw_grid):
            raise ValueError("alpha grid values must be unique")
        if abs(raw_grid[0]) > TIGHT_TOLERANCE:
            raise ValueError("alpha grid must start at zero")
        if abs(raw_grid[-1] - LOCAL_TARGET_RATE_CAP) > TIGHT_TOLERANCE:
            raise ValueError("alpha grid must end at the local cap")
        if any(level.value > raw_grid[-1] for level in self.budgets):
            raise ValueError("false-positive budget may not exceed alpha-grid maximum")
        return self

    @property
    def maximum_target_rate(self) -> TargetFalsePositiveRate:
        return self.alpha_grid[-1]

    @property
    def seeds(self) -> tuple[DetectorSeed, ...]:
        return self.detector.seeds


class ExternalEligibilityConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    minimum_benign_rows: RowCount
    minimum_attack_rows: RowCount
    minimum_attack_validation_rows: RowCount
    minimum_eligible_subtypes: RowCount
    minimum_rows_per_subtype: RowCount
    minimum_eligible_clients: RowCount
    maximum_fallback_fraction: Probability


class ExternalReplicationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    protocol_version: ProtocolVersion
    dataset_id: DatasetId
    detector: DetectorConfig
    budgets: tuple[BudgetLevel, ...]
    benign_splits: BenignSplitConfig
    attack_split: AttackSplitConfig
    eligibility: ExternalEligibilityConfig
    feature_parse_threshold: Probability
    solver: SolverConfig
    statistics: StatisticsConfig

    def protocol_for(self, primary: FabridConfig) -> FabridConfig:
        eligibility = self.eligibility
        return FabridConfig(
            protocol_version=self.protocol_version,
            dataset_id=self.dataset_id,
            alpha_grid=primary.alpha_grid,
            budgets=self.budgets,
            benign_splits=self.benign_splits,
            attack_split=self.attack_split,
            utility_eligibility=UtilityEligibilityConfig(
                minimum_attack_validation_rows=eligibility.minimum_attack_validation_rows,
                minimum_eligible_subtypes=eligibility.minimum_eligible_subtypes,
                minimum_rows_per_subtype=eligibility.minimum_rows_per_subtype,
            ),
            solver=self.solver,
            payload_sizing=primary.payload_sizing,
            statistics=self.statistics,
            practical_gates=primary.practical_gates,
            generalization=primary.generalization,
            event_gate=primary.event_gate,
            sensitivity=primary.sensitivity,
            detector=self.detector,
            scoring=ScoringConfig(
                primary_scorer=ScorerDefinition.AE_RECONSTRUCTION,
                ablation_scorers=(ScorerDefinition.AE_RECONSTRUCTION,),
                auxiliary_feature=None,
            ),
            calibration=primary.calibration,
        )


class EventLevelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    protocol_version: ProtocolVersion
    dataset_id: DatasetId
    event_gate: EventGateConfig
    time_to_detect_percentile: Probability
    detector: DetectorConfig
    budgets: tuple[BudgetLevel, ...]
    benign_splits: BenignSplitConfig
    attack_split: AttackSplitConfig


class PathSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    raw_data_root: Path
    preprocessed_root: Path
    outputs_root: Path
    results_root: Path

    def resolved_against(self, repository_root: Path) -> PathSettings:
        def resolve(path: Path) -> Path:
            return path if path.is_absolute() else (repository_root / path).resolve()

        return PathSettings(
            raw_data_root=resolve(self.raw_data_root),
            preprocessed_root=resolve(self.preprocessed_root),
            outputs_root=resolve(self.outputs_root),
            results_root=resolve(self.results_root),
        )


class AttackFileMapping(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    filename: Identifier
    subtype: AttackSubtypeId


class NbaiotDatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    directory_name: Identifier
    benign_filename: Identifier
    bashlite_directory: Identifier
    mirai_directory: Identifier
    devices: tuple[ClientId, ...]
    dual_botnet_excluded_devices: tuple[ClientId, ...]
    bashlite_files: tuple[AttackFileMapping, ...]
    mirai_files: tuple[AttackFileMapping, ...]
    weight_mode: WeightMode

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if not self.devices:
            raise ValueError("N-BaIoT requires at least one device")
        if len(set(self.devices)) != len(self.devices):
            raise ValueError("N-BaIoT device identities must be unique")
        excluded = set(self.dual_botnet_excluded_devices)
        unknown = excluded.difference(self.devices)
        if unknown:
            raise ValueError(f"excluded dual-botnet devices are not in the population: {unknown}")
        return self

    def dual_botnet_devices(self) -> tuple[ClientId, ...]:
        excluded = set(self.dual_botnet_excluded_devices)
        return tuple(device for device in self.devices if device not in excluded)


class CicIotDiadDatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    directory_name: Identifier
    packet_collection_directory: EnvironmentText
    device_column: ColumnName
    target_column: ColumnName
    benign_category: Identifier
    csv_glob: FileGlob
    excluded_features: tuple[ColumnName, ...]
    weight_mode: WeightMode


class GothamDatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    directory_name: Identifier
    device_column: ColumnName
    label_column: ColumnName
    benign_label: EnvironmentText
    timestamp_column: ColumnName
    csv_glob: FileGlob
    excluded_features: tuple[ColumnName, ...]
    weight_mode: WeightMode


class CiciomtDatasetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    directory_name: Identifier
    csv_glob: FileGlob
    timestamp_columns: tuple[ColumnName, ...]
    identity_columns: tuple[ColumnName, ...]
    weight_mode: WeightMode


class DatasetCatalog(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    nbaiot: NbaiotDatasetConfig
    cic_iot_diad: CicIotDiadDatasetConfig
    gotham: GothamDatasetConfig
    ciciomt: CiciomtDatasetConfig

    def directory_name(self, dataset_id: DatasetId) -> Identifier:
        if dataset_id is DatasetId.NBAIOT:
            return self.nbaiot.directory_name
        if dataset_id is DatasetId.CIC_IOT_DIAD:
            return self.cic_iot_diad.directory_name
        if dataset_id is DatasetId.GOTHAM:
            return self.gotham.directory_name
        return self.ciciomt.directory_name


class PreprocessingLayoutConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    feature_column_prefix: Identifier


class ExperimentEnablement(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    matched_budget: bool
    attack_subtype_disjoint: bool
    botnet_family_disjoint: bool
    weight_sensitivity: bool
    allocation_stability: bool
    conservative_utility: bool
    utility_heterogeneity: bool
    external_replication: bool
    event_level: bool

    def enabled(self, experiment_id: ExperimentId) -> bool:
        if experiment_id is ExperimentId.MATCHED_BUDGET:
            return self.matched_budget
        if experiment_id is ExperimentId.ATTACK_SUBTYPE_DISJOINT:
            return self.attack_subtype_disjoint
        if experiment_id is ExperimentId.BOTNET_FAMILY_DISJOINT:
            return self.botnet_family_disjoint
        if experiment_id is ExperimentId.WEIGHT_SENSITIVITY:
            return self.weight_sensitivity
        if experiment_id is ExperimentId.ALLOCATION_STABILITY:
            return self.allocation_stability
        if experiment_id is ExperimentId.CONSERVATIVE_UTILITY:
            return self.conservative_utility
        if experiment_id is ExperimentId.UTILITY_HETEROGENEITY:
            return self.utility_heterogeneity
        if experiment_id is ExperimentId.EXTERNAL_REPLICATION:
            return self.external_replication
        return self.event_level

    def only(self, experiment_id: ExperimentId) -> ExperimentEnablement:
        return ExperimentEnablement(
            matched_budget=experiment_id is ExperimentId.MATCHED_BUDGET,
            attack_subtype_disjoint=experiment_id is ExperimentId.ATTACK_SUBTYPE_DISJOINT,
            botnet_family_disjoint=experiment_id is ExperimentId.BOTNET_FAMILY_DISJOINT,
            weight_sensitivity=experiment_id is ExperimentId.WEIGHT_SENSITIVITY,
            allocation_stability=experiment_id is ExperimentId.ALLOCATION_STABILITY,
            conservative_utility=experiment_id is ExperimentId.CONSERVATIVE_UTILITY,
            utility_heterogeneity=experiment_id is ExperimentId.UTILITY_HETEROGENEITY,
            external_replication=experiment_id is ExperimentId.EXTERNAL_REPLICATION,
            event_level=experiment_id is ExperimentId.EVENT_LEVEL,
        )


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    seed_workers: WorkerCount
    evaluation_workers: WorkerCount
    event_workers: WorkerCount


class ApplicationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    protocol_version: ProtocolVersion
    paths: PathSettings
    datasets: DatasetCatalog
    preprocessing: PreprocessingLayoutConfig
    experiments: ExperimentEnablement
    execution: ExecutionConfig
    protocol: FabridConfig
    external_replication: ExternalReplicationConfig
    event_level: EventLevelConfig

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if self.protocol.protocol_version != self.protocol_version:
            raise ValueError("nested protocol_version must match the application protocol_version")
        names = tuple(self.datasets.directory_name(dataset_id) for dataset_id in DatasetId)
        if len(set(names)) != len(names):
            raise ValueError("dataset directory names must be unique within the raw-data root")
        return self

    def resolve_paths(self, repository_root: Path) -> ApplicationConfig:
        return self.model_copy(update={"paths": self.paths.resolved_against(repository_root)})

    @classmethod
    def from_yaml(cls, path: Path, *, repository_root: Path | None = None) -> ApplicationConfig:
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(f"cannot read configuration {path}") from error
        if not isinstance(loaded, dict):
            raise ValueError(f"configuration {path} must be a YAML mapping")
        try:
            config = cls.model_validate(loaded)
        except ValidationError as error:
            raise ValueError(f"invalid configuration {path}: {error}") from error
        root = Path.cwd() if repository_root is None else repository_root
        return config.resolve_paths(root)


PRODUCTION_CONFIG_PATH = Path("configs/fabrid.yaml")


def load_application_config() -> ApplicationConfig:
    return ApplicationConfig.from_yaml(PRODUCTION_CONFIG_PATH)
