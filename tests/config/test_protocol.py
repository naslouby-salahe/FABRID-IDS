from __future__ import annotations

from fabrid.config.protocol import load_protocol


def test_load_protocol_reads_frozen_yaml() -> None:
    protocol = load_protocol()

    assert protocol.benign_split_fractions.train_end_fraction == 0.50
    assert protocol.benign_split_fractions.frontier_end_fraction == 0.70
    assert protocol.benign_split_fractions.final_cal_end_fraction == 0.80
    assert protocol.attack_split_fraction.validation_end_fraction == 0.20
    assert protocol.alpha_max == 0.05
    assert protocol.solver_settings.mip_rel_gap == 0.0
    assert protocol.solver_settings.time_limit_seconds == 300.0
    assert protocol.solver_settings.accept_mip_gap_leq == 1.0e-5
    assert protocol.practical_gates.fabrid_macro.min_delta_macro_recall_pp == 2.0
    assert protocol.practical_gates.fabrid_macro.min_budgets_passing == 3
    assert protocol.practical_gates.fabrid_macro.of_budgets == 5
    assert protocol.practical_gates.fabrid_minimax.min_delta_worst_client_recall_pp == 5.0
    assert protocol.practical_gates.fabrid_minimax.max_macro_recall_loss_pp == 2.0
    assert protocol.practical_gates.budget_compliance.median_bur_leq == 1.05
    assert protocol.practical_gates.budget_compliance.seeds_with_bur_leq_1_10_min_fraction == 0.9
