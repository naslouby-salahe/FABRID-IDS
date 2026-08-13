from __future__ import annotations

from fabrid.domain.values import (
    BatchSize,
    BudgetUsageRatio,
    DetectorSeed,
    DurationSeconds,
    FederatedRoundCount,
    LayerWidth,
    LearningRate,
    LocalEpochCount,
    PercentagePoints,
    Probability,
    RowCount,
    SolverGap,
)
from fabrid.protocol.specification import PROTOCOL


def test_detector_protocol_is_frozen_to_ten_seed_experiment() -> None:
    hyperparameters = PROTOCOL.detector.hyperparameters

    assert PROTOCOL.detector.seeds == tuple(DetectorSeed(seed) for seed in range(10))
    assert hyperparameters.hidden_layers == (LayerWidth(64), LayerWidth(16))
    assert hyperparameters.learning_rate == LearningRate(0.001)
    assert hyperparameters.local_epochs == LocalEpochCount(3)
    assert hyperparameters.rounds == FederatedRoundCount(10)
    assert hyperparameters.batch_size == BatchSize(64)


def test_split_and_solver_protocol_matches_roadmap() -> None:
    assert PROTOCOL.benign_splits.train_end == Probability(0.50)
    assert PROTOCOL.benign_splits.frontier_end == Probability(0.70)
    assert PROTOCOL.benign_splits.final_cal_end == Probability(0.80)
    assert PROTOCOL.attack_split.validation_end == Probability(0.20)
    assert PROTOCOL.alpha_grid.maximum.value == 0.05
    assert PROTOCOL.solver.requested_gap == SolverGap(0.0)
    assert PROTOCOL.solver.time_limit == DurationSeconds(300.0)
    assert PROTOCOL.solver.accepted_gap == SolverGap(1.0e-5)


def test_statistics_and_practical_gates_match_roadmap() -> None:
    assert PROTOCOL.statistics.sign_flip_enumeration == RowCount(1_024)
    assert PROTOCOL.statistics.significance == Probability(0.05)
    assert PROTOCOL.statistics.holm_family_size == RowCount(5)
    assert PROTOCOL.statistics.bootstrap_resamples == RowCount(50_000)
    assert PROTOCOL.statistics.bootstrap_confidence == Probability(0.95)

    macro = PROTOCOL.practical_gates.fabrid_macro
    assert macro.minimum_macro_recall_gain == PercentagePoints(2.0)
    assert macro.minimum_passing_budgets == RowCount(3)
    assert macro.total_budgets == RowCount(5)

    minimax = PROTOCOL.practical_gates.fabrid_minimax
    assert minimax.minimum_worst_client_recall_gain == PercentagePoints(5.0)
    assert minimax.maximum_macro_recall_loss == PercentagePoints(2.0)
    assert minimax.minimum_passing_budgets == RowCount(3)
    assert minimax.total_budgets == RowCount(5)

    compliance = PROTOCOL.practical_gates.budget_compliance
    assert compliance.maximum_median_usage == BudgetUsageRatio(1.05)
    assert compliance.seed_usage_limit == BudgetUsageRatio(1.10)
    assert compliance.minimum_seed_fraction_below_limit == Probability(0.9)
