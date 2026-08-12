from __future__ import annotations

import pytest

from fabrid.evaluation.eventization import (
    AlertEvent,
    EventizationParameters,
    alarm_duty_fraction,
    eventize_alerts,
    events_per_hour,
    load_event_gate_config,
)

_PARAMS = EventizationParameters(
    dilation_seconds=2, merge_gap_seconds=5, min_event_length_seconds=2, cooldown_seconds=10
)


def test_empty_alerts_yield_no_events() -> None:
    assert eventize_alerts((), _PARAMS) == ()


def test_single_alert_dilates_to_minimum_length() -> None:
    events = eventize_alerts((100.0,), _PARAMS)
    assert len(events) == 1
    assert events[0].start_seconds == pytest.approx(100.0)
    assert events[0].end_seconds == pytest.approx(102.0)


def test_close_alerts_merge_into_one_event() -> None:
    # alerts at 0 and 3s: dilated to [0,2] and [3,5]; gap between them is
    # 3-2=1 <= merge_gap(5) -> merged into one event [0,5].
    events = eventize_alerts((0.0, 3.0), _PARAMS)
    assert len(events) == 1
    assert events[0].start_seconds == pytest.approx(0.0)
    assert events[0].end_seconds == pytest.approx(5.0)


def test_distant_alerts_stay_separate_events() -> None:
    events = eventize_alerts((0.0, 1000.0), _PARAMS)
    assert len(events) == 2


def test_cooldown_merges_events_that_would_otherwise_be_separate() -> None:
    # alerts at 0 and 8s dilate to [0,2] and [8,10]: gap is 6s, > merge_gap(5)
    # so they stay separate at the merge stage, but <= cooldown(10) so the
    # cooldown stage folds them into a single reported event.
    events = eventize_alerts((0.0, 8.0), _PARAMS)
    assert len(events) == 1


def test_events_per_hour() -> None:
    events = (AlertEvent(0.0, 2.0), AlertEvent(100.0, 102.0))
    rate = events_per_hour(events, observation_duration_seconds=3600.0)
    assert rate == pytest.approx(2.0)


def test_alarm_duty_fraction() -> None:
    events = (AlertEvent(0.0, 90.0),)
    duty = alarm_duty_fraction(events, observation_duration_seconds=3600.0)
    assert duty == pytest.approx(0.025)


def test_invalid_parameters_rejected() -> None:
    with pytest.raises(ValueError):
        EventizationParameters(
            dilation_seconds=-1,
            merge_gap_seconds=5,
            min_event_length_seconds=2,
            cooldown_seconds=10,
        )


def test_invalid_event_bounds_rejected() -> None:
    with pytest.raises(ValueError):
        AlertEvent(start_seconds=10.0, end_seconds=5.0)


def test_zero_observation_duration_rejected() -> None:
    with pytest.raises(ValueError):
        events_per_hour((), observation_duration_seconds=0.0)
    with pytest.raises(ValueError):
        alarm_duty_fraction((), observation_duration_seconds=0.0)


def test_load_event_gate_config_reads_frozen_yaml() -> None:
    config = load_event_gate_config()

    assert config.parameters.dilation_seconds == 2.0
    assert config.parameters.merge_gap_seconds == 5.0
    assert config.parameters.min_event_length_seconds == 2.0
    assert config.parameters.cooldown_seconds == 10.0
    assert config.max_alarm_duty == 0.25
    assert config.event_budgets_per_client_hour == (0.1, 0.2, 0.5)
    assert config.sensitivity_grid.dilation_seconds == (1.0, 2.0, 3.0)
    assert config.sensitivity_grid.merge_gap_seconds == (3.0, 5.0, 10.0)
    assert config.sensitivity_grid.min_event_length_seconds == (1.0, 2.0, 3.0)
    assert config.sensitivity_grid.cooldown_seconds == (5.0, 10.0, 20.0)
