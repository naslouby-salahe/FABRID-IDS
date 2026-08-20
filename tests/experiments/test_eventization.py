from __future__ import annotations

import numpy as np

from fabrid.experiments.event_level import (
    AlertEvent,
    EventizationParameters,
    alarm_duty,
    event_gini,
    eventize_alert_times,
)

_PARAMETERS = EventizationParameters(
    dilation=2.0, merge_gap=5.0, minimum_event_length=2.0, cooldown=10.0
)


def test_no_alerts_produces_no_events() -> None:
    assert eventize_alert_times(np.asarray([], dtype=np.float64), _PARAMETERS) == ()


def test_single_alert_below_minimum_length_is_dropped() -> None:
    events = eventize_alert_times(np.asarray([10.0], dtype=np.float64), _PARAMETERS)
    assert events == ()


def test_dense_alerts_merge_into_one_event() -> None:
    times = np.asarray([10.0, 10.5, 11.0, 11.2, 12.0], dtype=np.float64)
    events = eventize_alert_times(times, _PARAMETERS)
    assert len(events) == 1
    event = events[0]
    assert event.start_time == 10.0
    assert event.end_time == 12.0
    assert event.alarm_count == 5


def test_merge_gap_bridges_separate_blocks() -> None:
    times = np.asarray([0.0, 0.5, 4.5, 5.0], dtype=np.float64)
    events = eventize_alert_times(times, _PARAMETERS)
    assert len(events) == 1
    assert events[0].end_time - events[0].start_time >= 5.0


def test_wide_gap_produces_separate_events() -> None:
    times = np.asarray(
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 60.0, 60.5, 61.0, 61.5, 62.0, 62.5],
        dtype=np.float64,
    )
    events = eventize_alert_times(times, _PARAMETERS)
    assert len(events) == 2


def test_cooldown_absorbs_close_events() -> None:
    times = np.asarray([0.0, 0.5, 1.0, 5.0, 5.5, 6.0], dtype=np.float64)
    events = eventize_alert_times(times, _PARAMETERS)
    assert len(events) == 1


def test_ordering_invariance() -> None:
    times = np.asarray([12.0, 0.5, 11.0, 60.0, 10.5, 60.5], dtype=np.float64)
    sorted_events = eventize_alert_times(times, _PARAMETERS)
    shuffled = eventize_alert_times(np.flip(times), _PARAMETERS)
    assert [(e.start_time, e.end_time, e.alarm_count) for e in sorted_events] == [
        (e.start_time, e.end_time, e.alarm_count) for e in shuffled
    ]


def test_alarm_duty() -> None:
    event = AlertEvent(start_time=0.0, end_time=10.0, alarm_count=3)
    assert alarm_duty(event, 100.0) == 0.1
    assert alarm_duty(event, 40.0) == 0.25


def test_event_gini_uniform_is_zero() -> None:
    assert event_gini((5, 5, 5, 5)) == 0.0


def test_event_gini_concentrated_is_high() -> None:
    assert event_gini((10, 0, 0, 0)) > 0.7
