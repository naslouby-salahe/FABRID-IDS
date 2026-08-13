from __future__ import annotations

import pytest

from fabrid.domain.values import DurationSeconds, EventTimestamp
from fabrid.evaluation.eventization import (
    AlertEvent,
    EventizationRule,
    alarm_duty_fraction,
    eventize_alerts,
    events_per_hour,
    primary_eventization_rule,
    sensitivity_eventization_rules,
)

_RULE = EventizationRule(
    dilation=DurationSeconds(2.0),
    merge_gap=DurationSeconds(5.0),
    minimum_event_length=DurationSeconds(2.0),
    cooldown=DurationSeconds(10.0),
)


def _timestamps(*values: float) -> tuple[EventTimestamp, ...]:
    return tuple(EventTimestamp(value) for value in values)


def test_empty_alerts_yield_no_events() -> None:
    assert eventize_alerts((), _RULE) == ()


def test_single_alert_dilates_to_minimum_length() -> None:
    events = eventize_alerts(_timestamps(100.0), _RULE)
    assert len(events) == 1
    assert events[0].start.value == pytest.approx(100.0)
    assert events[0].end.value == pytest.approx(102.0)


def test_close_alerts_merge_into_one_event() -> None:
    events = eventize_alerts(_timestamps(0.0, 3.0), _RULE)
    assert len(events) == 1
    assert events[0].start.value == pytest.approx(0.0)
    assert events[0].end.value == pytest.approx(5.0)


def test_distant_alerts_stay_separate_events() -> None:
    events = eventize_alerts(_timestamps(0.0, 1000.0), _RULE)
    assert len(events) == 2


def test_cooldown_merges_events_that_would_otherwise_be_separate() -> None:
    events = eventize_alerts(_timestamps(0.0, 8.0), _RULE)
    assert len(events) == 1


def test_events_per_hour() -> None:
    events = (
        AlertEvent(EventTimestamp(0.0), EventTimestamp(2.0)),
        AlertEvent(EventTimestamp(100.0), EventTimestamp(102.0)),
    )
    rate = events_per_hour(events, DurationSeconds(3600.0))
    assert rate.value == pytest.approx(2.0)


def test_alarm_duty_fraction() -> None:
    events = (AlertEvent(EventTimestamp(0.0), EventTimestamp(90.0)),)
    duty = alarm_duty_fraction(events, DurationSeconds(3600.0))
    assert duty.value == pytest.approx(0.025)


def test_invalid_event_bounds_rejected() -> None:
    with pytest.raises(ValueError):
        AlertEvent(EventTimestamp(10.0), EventTimestamp(5.0))


def test_zero_observation_duration_rejected() -> None:
    with pytest.raises(ValueError):
        events_per_hour((), DurationSeconds(0.0))
    with pytest.raises(ValueError):
        alarm_duty_fraction((), DurationSeconds(0.0))


def test_primary_rule_comes_from_typed_protocol() -> None:
    rule = primary_eventization_rule()
    assert rule.dilation.value == 2.0
    assert rule.merge_gap.value == 5.0
    assert rule.minimum_event_length.value == 2.0
    assert rule.cooldown.value == 10.0


def test_sensitivity_grid_contains_all_81_preregistered_rules() -> None:
    rules = sensitivity_eventization_rules()
    assert len(rules) == 81
    assert len(set(rules)) == 81
