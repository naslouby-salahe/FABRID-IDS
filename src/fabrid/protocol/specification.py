from __future__ import annotations

from fabrid.domain.enums import (
    DecisionOperator,
    ExperimentalUnit,
    FallbackPolicy,
    OptimizationVariableKind,
    RetrainingPolicy,
    SolverBackend,
    ThresholdTiePolicy,
)
from fabrid.domain.values import (
    BudgetUsageRatio,
    DetectorSeed,
    DurationSeconds,
    EventRatePerClientHour,
    FalsePositiveBudget,
    PercentagePoints,
    Probability,
    RowCount,
    SolverGap,
)
from fabrid.protocol.alpha_grid import build_alpha_grid
from fabrid.protocol.models import (
    AttackSplitFraction,
    BenignSplitFractions,
    BudgetComplianceGate,
    DetectorProtocol,
    EventGate,
    EventGateSensitivity,
    FabridMacroGate,
    FabridMinimaxGate,
    FabridProtocol,
    PracticalGates,
    ScoreContract,
    SolverSettings,
    StatisticsProtocol,
    UtilityEligibility,
)

PROTOCOL = FabridProtocol(
    score_contract=ScoreContract(
        decision_operator=DecisionOperator.STRICT_GREATER_THAN,
        tie_policy=ThresholdTiePolicy.NON_ALERT,
    ),
    detector=DetectorProtocol(
        seeds=tuple(DetectorSeed(seed) for seed in range(10)),
        retraining_policy=RetrainingPolicy.FROZEN_ACROSS_POLICIES,
    ),
    alpha_grid=build_alpha_grid(),
    budgets=tuple(
        FalsePositiveBudget(value)
        for value in (0.001, 0.0025, 0.005, 0.010, 0.020)
    ),
    benign_splits=BenignSplitFractions(
        train_end=Probability(0.50),
        frontier_end=Probability(0.70),
        final_cal_end=Probability(0.80),
    ),
    attack_split=AttackSplitFraction(validation_end=Probability(0.20)),
    utility_eligibility=UtilityEligibility(
        minimum_attack_validation_rows=RowCount(200),
        minimum_eligible_subtypes=RowCount(2),
        minimum_rows_per_subtype=RowCount(50),
        fallback_policy=FallbackPolicy.EQUAL_FPR_AT_BUDGET,
    ),
    solver=SolverSettings(
        backend=SolverBackend.SCIPY_MILP,
        variable_kind=OptimizationVariableKind.BINARY,
        requested_gap=SolverGap(0.0),
        time_limit=DurationSeconds(300.0),
        accepted_gap=SolverGap(1.0e-5),
    ),
    statistics=StatisticsProtocol(
        experimental_unit=ExperimentalUnit.DETECTOR_SEED,
        sign_flip_enumeration=RowCount(1024),
        significance=Probability(0.05),
        holm_family_size=RowCount(5),
        bootstrap_resamples=RowCount(50_000),
    ),
    practical_gates=PracticalGates(
        fabrid_macro=FabridMacroGate(
            minimum_macro_recall_gain=PercentagePoints(2.0),
            minimum_passing_budgets=RowCount(3),
            total_budgets=RowCount(5),
        ),
        fabrid_minimax=FabridMinimaxGate(
            minimum_worst_client_recall_gain=PercentagePoints(5.0),
            maximum_macro_recall_loss=PercentagePoints(2.0),
            minimum_passing_budgets=RowCount(3),
            total_budgets=RowCount(5),
        ),
        budget_compliance=BudgetComplianceGate(
            maximum_median_usage=BudgetUsageRatio(1.05),
            minimum_seed_fraction_below_limit=Probability(0.9),
        ),
    ),
    allocation_sensitivity_replicates=RowCount(500),
    conservative_utility_confidence=Probability(0.95),
    event_gate=EventGate(
        dilation=DurationSeconds(2.0),
        merge_gap=DurationSeconds(5.0),
        minimum_event_length=DurationSeconds(2.0),
        cooldown=DurationSeconds(10.0),
        maximum_alarm_duty=Probability(0.25),
        budgets_per_client_hour=tuple(
            EventRatePerClientHour(value) for value in (0.1, 0.2, 0.5)
        ),
        sensitivity=EventGateSensitivity(
            dilation=tuple(DurationSeconds(value) for value in (1.0, 2.0, 3.0)),
            merge_gap=tuple(DurationSeconds(value) for value in (3.0, 5.0, 10.0)),
            minimum_event_length=tuple(
                DurationSeconds(value) for value in (1.0, 2.0, 3.0)
            ),
            cooldown=tuple(DurationSeconds(value) for value in (5.0, 10.0, 20.0)),
        ),
    ),
)
