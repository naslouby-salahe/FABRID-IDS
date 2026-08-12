from __future__ import annotations

import pytest

from fabrid.evaluation.record_level import ClientId
from fabrid.frontier.stability import run_allocation_sensitivity, summarize_client_stability


def test_perfectly_stable_client_has_zero_instability() -> None:
    summary = summarize_client_stability((0.01,) * 100)
    assert summary.modal_alpha == pytest.approx(0.01)
    assert summary.modal_frequency == pytest.approx(1.0)
    assert summary.instability == pytest.approx(0.0)


def test_split_outcomes_report_modal_frequency_and_instability() -> None:
    # 60/40 split -> modal frequency 0.6, instability 0.4.
    values = (0.01,) * 60 + (0.02,) * 40
    summary = summarize_client_stability(values)
    assert summary.modal_alpha == pytest.approx(0.01)
    assert summary.modal_frequency == pytest.approx(0.6)
    assert summary.instability == pytest.approx(0.4)


def test_percentiles_bracket_the_distribution() -> None:
    values = tuple(float(i) for i in range(100))
    summary = summarize_client_stability(values)
    assert summary.percentile_5 < summary.median_alpha < summary.percentile_95


def test_empty_replicates_rejected() -> None:
    with pytest.raises(ValueError):
        summarize_client_stability(())


def test_run_allocation_sensitivity_aggregates_per_client() -> None:
    def resample_and_allocate(replicate_seed: int) -> dict[ClientId, float]:
        # deterministic pseudo-allocation keyed only by replicate_seed parity,
        # so the aggregate result is exactly predictable.
        alpha = 0.01 if replicate_seed % 2 == 0 else 0.02
        return {ClientId("1"): alpha, ClientId("2"): 0.005}

    summaries = run_allocation_sensitivity(resample_and_allocate, replicates=50, seed=0)
    assert set(summaries.keys()) == {ClientId("1"), ClientId("2")}
    assert summaries[ClientId("2")].instability == pytest.approx(0.0)


def test_invalid_replicates_rejected() -> None:
    with pytest.raises(ValueError):
        run_allocation_sensitivity(lambda _: {ClientId("1"): 0.01}, replicates=0)
