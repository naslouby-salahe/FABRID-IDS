from __future__ import annotations

import pytest

from fabrid.allocation.contracts import Allocation, AllocationDecision
from fabrid.analysis.stability import run_allocation_sensitivity, summarize_client_stability
from fabrid.domain.identifiers import ClientId
from fabrid.domain.values import AnalysisSeed, Probability, RowCount, TargetFalsePositiveRate


def test_stable_client_has_zero_instability() -> None:
    summary = summarize_client_stability((TargetFalsePositiveRate(0.01),) * 100)

    assert summary.modal_target_rate == TargetFalsePositiveRate(0.01)
    assert summary.modal_frequency == Probability(1.0)
    assert summary.instability == Probability(0.0)


def test_split_selection_reports_modal_frequency_and_instability() -> None:
    values = (
        (TargetFalsePositiveRate(0.01),) * 60
        + (TargetFalsePositiveRate(0.02),) * 40
    )

    summary = summarize_client_stability(values)

    assert summary.modal_frequency == Probability(0.6)
    assert summary.instability == Probability(0.4)
    assert summary.percentile_5.value <= summary.median_target_rate.value <= summary.percentile_95.value


def test_empty_stability_replicates_are_rejected() -> None:
    with pytest.raises(ValueError):
        summarize_client_stability(())


def test_allocation_sensitivity_aggregates_same_typed_population() -> None:
    first = ClientId("first")
    second = ClientId("second")

    def resample_and_allocate(seed: AnalysisSeed) -> Allocation:
        first_rate = TargetFalsePositiveRate(0.01 if seed.value % 2 == 0 else 0.02)
        return Allocation(
            decisions=(
                AllocationDecision(first, first_rate),
                AllocationDecision(second, TargetFalsePositiveRate(0.005)),
            )
        )

    result = run_allocation_sensitivity(
        resample_and_allocate=resample_and_allocate,
        replicates=RowCount(50),
        seed=AnalysisSeed(0),
    )

    second_summary = next(client.summary for client in result.clients if client.client_id == second)
    assert second_summary.instability == Probability(0.0)
