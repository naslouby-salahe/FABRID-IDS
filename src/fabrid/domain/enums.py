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


class FeatureColumnStatus(StrEnum):
    KEPT = "kept"
    EXCLUDED = "excluded"
    PARSE_REJECTED = "parse_rejected"


class WeightMode(StrEnum):
    EQUAL_CLIENT = "equal_client"
    DATASET_COUNT_PROXY = "dataset_count_proxy"
    OPERATIONAL_RATE = "operational_rate"


class BudgetId(StrEnum):
    FALSE_POSITIVE_0P001 = "false_positive_0p001"
    FALSE_POSITIVE_0P0025 = "false_positive_0p0025"
    FALSE_POSITIVE_0P005 = "false_positive_0p005"
    FALSE_POSITIVE_0P010 = "false_positive_0p010"
    FALSE_POSITIVE_0P020 = "false_positive_0p020"


class AllocationPolicy(StrEnum):
    EQ_FPR = "eq_fpr"
    EQ_ALERT = "eq_alert"
    GREEDY = "greedy"
    FABRID_MACRO = "fabrid_macro"
    FABRID_MINIMAX = "fabrid_minimax"
    POOLED_SHARED = "pooled_shared"
    TEST_ORACLE = "test_oracle"


class MetricId(StrEnum):
    MACRO_RECALL = "macro_recall"
    WORST_CLIENT_RECALL = "worst_client_recall"
    FEDERATION_FPR = "federation_fpr"
    BUDGET_USAGE_RATIO = "budget_usage_ratio"


class PrimaryContrastId(StrEnum):
    FABRID_MACRO_VS_EQ_FPR_MACRO_RECALL = "fabrid_macro_vs_eq_fpr_macro_recall"
    FABRID_MINIMAX_VS_EQ_FPR_WORST_CLIENT_RECALL = (
        "fabrid_minimax_vs_eq_fpr_worst_client_recall"
    )


class EvidenceAvailability(StrEnum):
    AVAILABLE = "available"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class HypothesisDecision(StrEnum):
    REJECT = "reject"
    RETAIN = "retain"


class GateStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"


class SolverStatus(StrEnum):
    OPTIMAL = "optimal"
    SOLVER_INVALID = "solver_invalid"
    NOT_APPLICABLE = "not_applicable"


class SolverStage(StrEnum):
    MACRO_PRIMARY_UTILITY = "macro_primary_utility"
    MACRO_MINIMUM_BUDGET = "macro_minimum_budget"
    MACRO_TIE_BREAK = "macro_tie_break"
    MINIMAX_WORST_CLIENT = "minimax_worst_client"
    MINIMAX_MEAN_UTILITY = "minimax_mean_utility"
    MINIMAX_MINIMUM_BUDGET = "minimax_minimum_budget"
    MINIMAX_TIE_BREAK = "minimax_tie_break"


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
    CONSERVATIVE_MINIMAX = "conservative_minimax"
    EXTERNAL_PRIMARY = "external_primary"
    EVENT_PRIMARY = "event_primary"


class AttackFoldId(StrEnum):
    FOLD_0 = "fold_0"
    FOLD_1 = "fold_1"
    FOLD_2 = "fold_2"


class BotnetFamily(StrEnum):
    BASHLITE = "bashlite"
    MIRAI = "mirai"


class AuditStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"
