"""Alarm eventization: merge raw alert timestamps into investigation-level events.

Only meaningful where `EVENT_DATA_GATE` has passed (client identity, packet
timestamp, and interval provenance all verified from source — see
`fabrid.audit` for the gate itself, which is data-dependent and out of scope
for this pure sequence-processing module).

Algorithm (a documented engineering interpretation of the roadmap's four
named parameters, since the roadmap fixes their values but not the merge
mechanics): each alert at time `t` dilates into `[t, t + dilation]`; dilated
intervals separated by at most `merge_gap` are merged; merged intervals
shorter than `min_event_length` are dropped; surviving events separated by
at most `cooldown` are further merged (an alert during the cooldown window
after an event is treated as part of that event, not a new one).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

_SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True, slots=True)
class EventizationParameters:
    dilation_seconds: float
    merge_gap_seconds: float
    min_event_length_seconds: float
    cooldown_seconds: float

    def __post_init__(self) -> None:
        for name, value in (
            ("dilation_seconds", self.dilation_seconds),
            ("merge_gap_seconds", self.merge_gap_seconds),
            ("min_event_length_seconds", self.min_event_length_seconds),
            ("cooldown_seconds", self.cooldown_seconds),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")


@dataclass(frozen=True, slots=True)
class AlertEvent:
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if self.end_seconds < self.start_seconds:
            raise ValueError(
                f"end_seconds ({self.end_seconds}) must be >= start_seconds ({self.start_seconds})"
            )

    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def eventize_alerts(
    alert_timestamps: Sequence[float], parameters: EventizationParameters
) -> tuple[AlertEvent, ...]:
    if not alert_timestamps:
        return ()

    sorted_timestamps = sorted(alert_timestamps)
    dilated = [(t, t + parameters.dilation_seconds) for t in sorted_timestamps]

    merged: list[list[float]] = []
    for start, end in dilated:
        if merged and start <= merged[-1][1] + parameters.merge_gap_seconds:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    long_enough = [
        interval
        for interval in merged
        if interval[1] - interval[0] >= parameters.min_event_length_seconds
    ]

    cooled: list[list[float]] = []
    for start, end in long_enough:
        if cooled and start <= cooled[-1][1] + parameters.cooldown_seconds:
            cooled[-1][1] = max(cooled[-1][1], end)
        else:
            cooled.append([start, end])

    return tuple(AlertEvent(start_seconds=s, end_seconds=e) for s, e in cooled)


def events_per_hour(events: tuple[AlertEvent, ...], observation_duration_seconds: float) -> float:
    if observation_duration_seconds <= 0:
        raise ValueError(
            f"observation_duration_seconds must be positive, got {observation_duration_seconds}"
        )
    return len(events) / (observation_duration_seconds / _SECONDS_PER_HOUR)


def alarm_duty_fraction(
    events: tuple[AlertEvent, ...], observation_duration_seconds: float
) -> float:
    if observation_duration_seconds <= 0:
        raise ValueError(
            f"observation_duration_seconds must be positive, got {observation_duration_seconds}"
        )
    total_event_seconds = sum(event.duration_seconds() for event in events)
    return total_event_seconds / observation_duration_seconds
