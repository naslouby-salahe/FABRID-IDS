from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from fabrid.domain.values import DurationSeconds, EventRatePerClientHour, EventTimestamp, Probability
from fabrid.protocol.models import EventGate
from fabrid.protocol.specification import PROTOCOL

_SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True, slots=True)
class EventizationRule:
    dilation: DurationSeconds
    merge_gap: DurationSeconds
    minimum_event_length: DurationSeconds
    cooldown: DurationSeconds


@dataclass(frozen=True, slots=True)
class AlertEvent:
    start: EventTimestamp
    end: EventTimestamp

    def __post_init__(self) -> None:
        if self.end.value < self.start.value:
            raise ValueError("alert event end must not precede its start")

    @property
    def duration(self) -> DurationSeconds:
        return DurationSeconds(self.end.value - self.start.value)


def primary_eventization_rule(event_gate: EventGate = PROTOCOL.event_gate) -> EventizationRule:
    return EventizationRule(
        dilation=event_gate.dilation,
        merge_gap=event_gate.merge_gap,
        minimum_event_length=event_gate.minimum_event_length,
        cooldown=event_gate.cooldown,
    )


def sensitivity_eventization_rules(
    event_gate: EventGate = PROTOCOL.event_gate,
) -> tuple[EventizationRule, ...]:
    return tuple(
        EventizationRule(
            dilation=dilation,
            merge_gap=merge_gap,
            minimum_event_length=minimum_event_length,
            cooldown=cooldown,
        )
        for dilation, merge_gap, minimum_event_length, cooldown in product(
            event_gate.sensitivity.dilation,
            event_gate.sensitivity.merge_gap,
            event_gate.sensitivity.minimum_event_length,
            event_gate.sensitivity.cooldown,
        )
    )


def _merge_with_gap(
    events: tuple[AlertEvent, ...],
    maximum_gap: DurationSeconds,
) -> tuple[AlertEvent, ...]:
    merged: list[AlertEvent] = []
    for event in events:
        if merged and event.start.value <= merged[-1].end.value + maximum_gap.value:
            previous = merged[-1]
            merged[-1] = AlertEvent(
                start=previous.start,
                end=EventTimestamp(max(previous.end.value, event.end.value)),
            )
        else:
            merged.append(event)
    return tuple(merged)


def eventize_alerts(
    alert_timestamps: tuple[EventTimestamp, ...],
    rule: EventizationRule,
) -> tuple[AlertEvent, ...]:
    if not alert_timestamps:
        return ()
    dilated = tuple(
        AlertEvent(
            start=timestamp,
            end=EventTimestamp(timestamp.value + rule.dilation.value),
        )
        for timestamp in sorted(alert_timestamps, key=lambda timestamp: timestamp.value)
    )
    merged = _merge_with_gap(dilated, rule.merge_gap)
    long_enough = tuple(
        event for event in merged if event.duration.value >= rule.minimum_event_length.value
    )
    return _merge_with_gap(long_enough, rule.cooldown)


def events_per_hour(
    events: tuple[AlertEvent, ...],
    observation_duration: DurationSeconds,
) -> EventRatePerClientHour:
    if observation_duration.value <= 0.0:
        raise ValueError("observation duration must be positive")
    return EventRatePerClientHour(
        len(events) / (observation_duration.value / _SECONDS_PER_HOUR)
    )


def alarm_duty_fraction(
    events: tuple[AlertEvent, ...],
    observation_duration: DurationSeconds,
) -> Probability:
    if observation_duration.value <= 0.0:
        raise ValueError("observation duration must be positive")
    total_event_seconds = sum(event.duration.value for event in events)
    if total_event_seconds > observation_duration.value:
        raise ValueError("event duration exceeds the observation interval")
    return Probability(total_event_seconds / observation_duration.value)
