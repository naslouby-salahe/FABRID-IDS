from __future__ import annotations

from fabrid.allocation.contracts import Allocation, AllocationDecision
from fabrid.analysis.stability import (
    AvailablePolicyStability,
    UnavailablePolicyStability,
    allocation_sensitivity_seeds,
    build_policy_stability,
)
from fabrid.domain.enums import AllocationPolicy, EvidenceAvailability
from fabrid.domain.identifiers import ClientId
from fabrid.domain.values import AnalysisSeed, RowCount, TargetFalsePositiveRate


def _allocation(policy: AllocationPolicy, rate: float) -> Allocation:
    return Allocation(
        policy=policy,
        decisions=(
            AllocationDecision(
                client_id=ClientId("device"),
                target_rate=TargetFalsePositiveRate(rate),
            ),
        ),
    )


def test_stability_is_available_only_when_all_preregistered_replicates_complete() -> None:
    expected = RowCount(3)
    evidence = build_policy_stability(
        policy=AllocationPolicy.FABRID_MACRO,
        allocations=(
            _allocation(AllocationPolicy.FABRID_MACRO, 0.01),
            _allocation(AllocationPolicy.FABRID_MACRO, 0.01),
            _allocation(AllocationPolicy.FABRID_MACRO, 0.02),
        ),
        solver_invalid_replicates=RowCount(0),
        expected_replicates=expected,
    )

    assert isinstance(evidence, AvailablePolicyStability)
    assert evidence.availability is EvidenceAvailability.AVAILABLE
    assert evidence.completed_replicates == expected
    assert evidence.solver_invalid_replicates == RowCount(0)
    assert evidence.analysis.clients[0].summary.modal_target_rate == TargetFalsePositiveRate(0.01)


def test_single_missing_replicate_blocks_stability_evidence() -> None:
    evidence = build_policy_stability(
        policy=AllocationPolicy.FABRID_MINIMAX,
        allocations=(
            _allocation(AllocationPolicy.FABRID_MINIMAX, 0.01),
            _allocation(AllocationPolicy.FABRID_MINIMAX, 0.02),
        ),
        solver_invalid_replicates=RowCount(1),
        expected_replicates=RowCount(3),
    )

    assert isinstance(evidence, UnavailablePolicyStability)
    assert evidence.availability is EvidenceAvailability.INSUFFICIENT_EVIDENCE
    assert evidence.completed_replicates == RowCount(2)
    assert evidence.solver_invalid_replicates == RowCount(1)


def test_preregistered_replicate_seed_schedule_is_deterministic_and_complete() -> None:
    seeds = allocation_sensitivity_seeds(RowCount(500), AnalysisSeed(7))

    assert len(seeds) == 500
    assert seeds == allocation_sensitivity_seeds(RowCount(500), AnalysisSeed(7))
