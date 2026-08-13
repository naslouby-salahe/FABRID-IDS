from __future__ import annotations

from enum import StrEnum


class DatasetId(StrEnum):
    NBAIOT = "nbaiot"
    CIC_IOT_DIAD = "cic_iot_diad"
    GOTHAM = "gotham"
    CICIOMT = "ciciomt"


class SourceSplit(StrEnum):
    TRAIN = "train"
    TEST = "test"


class WeightMode(StrEnum):
    EQUAL_CLIENT = "equal_client"
    DATASET_COUNT_PROXY = "dataset_count_proxy"
    OPERATIONAL_RATE = "operational_rate"


class AllocationPolicy(StrEnum):
    EQ_FPR = "eq_fpr"
    EQ_ALERT = "eq_alert"
    GREEDY = "greedy"
    FABRID_MACRO = "fabrid_macro"
    FABRID_MINIMAX = "fabrid_minimax"
    POOLED_SHARED = "pooled_shared"
    TEST_ORACLE = "test_oracle"


class SolverStatus(StrEnum):
    OPTIMAL = "optimal"
    SOLVER_INVALID = "solver_invalid"
    NOT_APPLICABLE = "not_applicable"


class SolverBackend(StrEnum):
    SCIPY_MILP = "scipy.optimize.milp"


class OptimizationVariableKind(StrEnum):
    BINARY = "binary"


class DecisionOperator(StrEnum):
    STRICT_GREATER_THAN = "strict_greater_than"


class ThresholdTiePolicy(StrEnum):
    NON_ALERT = "non_alert"


class FallbackPolicy(StrEnum):
    EQUAL_FPR_AT_BUDGET = "equal_fpr_at_budget"


class EligibilityStatus(StrEnum):
    ELIGIBLE = "eligible"
    FALLBACK = "fallback"


class RetrainingPolicy(StrEnum):
    FROZEN_ACROSS_POLICIES = "frozen_across_policies"


class ExperimentalUnit(StrEnum):
    DETECTOR_SEED = "detector_seed"


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


class ArtifactKind(StrEnum):
    PROTOCOL = "protocol"
    DATASET_MANIFEST = "dataset_manifest"
    SPLIT_MANIFEST = "split_manifest"
    FEATURE_MANIFEST = "feature_manifest"
    DETECTOR = "detector"
    SCALER = "scaler"
    SCORE = "score"
    ALLOCATION = "allocation"
    RESULT = "result"
    ANALYSIS = "analysis"
    TABLE = "table"
    FIGURE = "figure"
    AUDIT = "audit"


class ExperimentId(StrEnum):
    MATCHED_BUDGET = "matched_budget"
    ATTACK_SUBTYPE_DISJOINT = "attack_subtype_disjoint"
    BOTNET_FAMILY_DISJOINT = "botnet_family_disjoint"
    WEIGHT_SENSITIVITY = "weight_sensitivity"
    CONSERVATIVE_MINIMAX = "conservative_minimax"
    EXTERNAL_REPLICATION = "external_replication"
    EVENT_LEVEL = "event_level"


class AuditStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"
