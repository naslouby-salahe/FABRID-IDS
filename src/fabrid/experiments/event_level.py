from __future__ import annotations

import ctypes
import gc
import logging
import os
import shutil
import sys
import tempfile
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from multiprocessing import get_context
from pathlib import Path
from types import TracebackType
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, TypeAdapter

from fabrid.artifacts.json import digest_text, write_typed_json
from fabrid.artifacts.parquet import read_parquet_models, write_parquet_models
from fabrid.artifacts.paths import ArtifactFileName, ArtifactPaths, ScoreCoordinate
from fabrid.config import (
    ZERO_ROW_COUNT,
    AnomalyScore,
    ApplicationConfig,
    ArtifactDigest,
    AttackSplit,
    AttackSubtypeId,
    BenignSplit,
    BudgetId,
    BudgetLevel,
    ClientId,
    DatasetCatalog,
    DatasetId,
    DetectorSeed,
    DurationSeconds,
    EventLevelConfig,
    EventRatePerClientHour,
    EventTimestamp,
    FederatedRoundCount,
    GateStatus,
    Label,
    Probability,
    RelativePath,
    RowCount,
    SourceFileId,
    SourceRowIndex,
    TargetFalsePositiveRate,
    Threshold,
)
from fabrid.datasets.ciciomt import audit_ciciomt_event_provenance
from fabrid.datasets.gotham import (
    GothamNumericClient,
    audit_gotham_event_provenance,
    gotham_capture_paths,
    load_gotham_numeric_features,
    load_gotham_timed_rows,
)
from fabrid.datasets.registry import (
    AttackSplitBoundary,
    AttackSubtypeBoundary,
    BenignSplitBoundaries,
    EventProvenanceEvidence,
    FeatureMatrix,
    compute_attack_split_boundary,
    compute_benign_split_boundaries,
)
from fabrid.detector.autoencoder import AutoencoderArchitecture, resolve_device
from fabrid.detector.calibration import alerts_above_threshold, calibrate_threshold
from fabrid.detector.scoring import (
    ScorePartitionArtifact,
    ScoreRecord,
    persist_score_partition,
    score_feature_matrix,
)
from fabrid.detector.training import (
    ClientScaler,
    ClientTrainingData,
    FederatedScalers,
    FederatedTrainingData,
    FederatedTrainingSettings,
    RoundProgressReporter,
    fit_feature_scaler,
    train_federated_autoencoder,
)
from fabrid.errors import DatasetError
from fabrid.validation.reproducibility import ProgressState, now_iso, report_progress

logger = logging.getLogger(__name__)
_SECONDS_PER_HOUR = 3600.0
_GOTHAM_EVENT_CACHE_VERSION = "gotham-event-client-v1"


@dataclass(frozen=True, slots=True)
class _GothamEventClientCache:
    directory: Path
    cache_key: ArtifactDigest
    clients: tuple[GothamNumericClient, ...]


class _GothamEventCacheClient(BaseModel):
    model_config = ConfigDict(frozen=True)
    client_id: ClientId
    source_files: tuple[SourceFileId, ...]
    source_rows: tuple[SourceRowIndex, ...]
    labels: tuple[Label, ...]
    attack_subtypes: tuple[AttackSubtypeId | None, ...]
    features: RelativePath


class _GothamEventCacheManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    cache_key: ArtifactDigest
    finite_validated: bool
    clients: tuple[_GothamEventCacheClient, ...]


_spawn_gotham_event_clients: tuple[GothamNumericClient, ...] | None = None


class EventLevelContext(Protocol):
    @property
    def config(self) -> ApplicationConfig: ...
    @property
    def paths(self) -> ArtifactPaths: ...
    def raw_dataset_root(self, dataset_id: DatasetId) -> Path: ...


class _ExclusiveInitializationLock(Protocol):
    def __enter__(self) -> bool: ...
    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None: ...


def _gate_status(evidence: EventProvenanceEvidence) -> GateStatus:
    passed = all(criterion.status is GateStatus.PASS for criterion in evidence.criteria)
    return GateStatus.PASS if passed else GateStatus.FAIL


@dataclass(frozen=True, slots=True)
class EventDataGateAssessment:
    dataset_id: DatasetId
    evidence: EventProvenanceEvidence
    status: GateStatus

    def __post_init__(self) -> None:
        if self.status is not _gate_status(self.evidence):
            raise ValueError("event gate status is inconsistent with its evidence")


def assess_event_data_gate(
    dataset_id: DatasetId,
    evidence: EventProvenanceEvidence,
) -> EventDataGateAssessment:
    return EventDataGateAssessment(
        dataset_id=dataset_id,
        evidence=evidence,
        status=_gate_status(evidence),
    )


@dataclass(frozen=True, slots=True)
class AlertEvent:
    start_time: EventTimestamp
    end_time: EventTimestamp
    alarm_count: RowCount


class EventMetricRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    dataset_id: DatasetId
    seed: DetectorSeed
    budget_id: BudgetId
    target_rate: TargetFalsePositiveRate
    client_id: ClientId
    false_alert_events_per_hour: EventRatePerClientHour
    attacked_stream_events_per_hour: EventRatePerClientHour
    attack_event_recall: Probability
    attack_event_miss_rate: Probability
    mean_time_to_detect: DurationSeconds
    median_time_to_detect: DurationSeconds
    percentile_90_time_to_detect: DurationSeconds
    alarm_duty_fraction: Probability


class EventBudgetRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    dataset_id: DatasetId
    seed: DetectorSeed
    budget_per_client_hour: EventRatePerClientHour
    client_id: ClientId
    threshold: Threshold
    false_alert_events_per_hour: EventRatePerClientHour
    attack_event_recall: Probability
    mean_time_to_detect: DurationSeconds


@dataclass(frozen=True, slots=True)
class EventMetricValues:
    false_alert_events_per_hour: EventRatePerClientHour
    attacked_stream_events_per_hour: EventRatePerClientHour
    attack_event_recall: Probability
    attack_event_miss_rate: Probability
    mean_time_to_detect: DurationSeconds
    median_time_to_detect: DurationSeconds
    percentile_90_time_to_detect: DurationSeconds
    alarm_duty_fraction: Probability


class EventRobustnessRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    dataset_id: DatasetId
    seed: DetectorSeed
    budget_id: BudgetId
    dilation: DurationSeconds
    merge_gap: DurationSeconds
    minimum_event_length: DurationSeconds
    cooldown: DurationSeconds
    false_alert_events_per_hour: EventRatePerClientHour
    attack_event_recall: Probability


class EventGiniRow(BaseModel):
    model_config = ConfigDict(frozen=True)
    dataset_id: DatasetId
    seed: DetectorSeed
    budget_id: BudgetId
    gini: Probability


@dataclass(frozen=True, slots=True)
class EventizationParameters:
    dilation: DurationSeconds
    merge_gap: DurationSeconds
    minimum_event_length: DurationSeconds
    cooldown: DurationSeconds


def _nominal_eventization(config: EventLevelConfig) -> EventizationParameters:
    gate = config.event_gate
    return EventizationParameters(
        dilation=gate.dilation,
        merge_gap=gate.merge_gap,
        minimum_event_length=gate.minimum_event_length,
        cooldown=gate.cooldown,
    )


@dataclass(frozen=True, slots=True)
class TimeInterval:
    start: EventTimestamp
    end: EventTimestamp

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("time interval end must not precede its start")


@dataclass(frozen=True, slots=True)
class EventObservation:
    client_id: ClientId
    timestamp: EventTimestamp
    score: AnomalyScore
    label: Label
    source_file: SourceFileId
    source_row: SourceRowIndex


@dataclass(frozen=True, slots=True)
class ClientAlertStreams:
    client_id: ClientId
    benign_times: np.ndarray
    attacked_times: np.ndarray
    attack_intervals: tuple[TimeInterval, ...]
    observation_duration: DurationSeconds

    def __post_init__(self) -> None:
        if self.observation_duration < 0.0:
            raise ValueError("observation duration must be non-negative")


def _observations_for_client(
    observations: tuple[EventObservation, ...],
    client_id: ClientId,
) -> tuple[EventObservation, ...]:
    return tuple(item for item in observations if item.client_id == client_id)


def _group_observations_by_client(
    observations: tuple[EventObservation, ...],
) -> Mapping[ClientId, tuple[EventObservation, ...]]:
    grouped: defaultdict[ClientId, list[EventObservation]] = defaultdict(list)
    for item in observations:
        grouped[item.client_id].append(item)
    return {client_id: tuple(grouped[client_id]) for client_id in sorted(grouped)}


def _attack_interval(observations: tuple[EventObservation, ...]) -> TimeInterval | None:
    attack_times = tuple(item.timestamp for item in observations if item.label is Label.ATTACK)
    if not attack_times:
        return None
    return TimeInterval(start=min(attack_times), end=max(attack_times))


def build_client_alert_streams(
    observations: tuple[EventObservation, ...],
    threshold: Threshold,
) -> tuple[ClientAlertStreams, ...]:
    if not observations:
        return ()
    clients = tuple(sorted({item.client_id for item in observations}))
    streams: list[ClientAlertStreams] = []
    for client_id in clients:
        items = _observations_for_client(observations, client_id)
        timestamps = np.asarray([item.timestamp for item in items], dtype=np.float64)
        scores = np.asarray([item.score for item in items], dtype=np.float64)
        labels = tuple(item.label for item in items)
        alerts = alerts_above_threshold(scores, threshold)
        benign_mask = np.array([label is Label.BENIGN for label in labels], dtype=np.bool_)
        interval = _attack_interval(items)
        streams.append(
            ClientAlertStreams(
                client_id=client_id,
                benign_times=np.sort(timestamps[alerts & benign_mask]),
                attacked_times=np.sort(timestamps[alerts]),
                attack_intervals=() if interval is None else (interval,),
                observation_duration=float(np.max(timestamps) - np.min(timestamps)),
            )
        )
    return tuple(streams)


def load_event_score_records(
    paths: ArtifactPaths,
    dataset_id: DatasetId,
    seed: DetectorSeed,
) -> tuple[ScoreRecord, ...]:
    seed_root = paths.score_provenance_path(dataset_id, seed).parent
    if not seed_root.is_dir():
        return ()
    records: list[ScoreRecord] = []
    for parquet in sorted(seed_root.rglob("*.parquet")):
        records.extend(read_parquet_models(parquet, ScoreRecord))
    return tuple(records)


@dataclass(frozen=True, slots=True)
class EventScoreLookup:
    records: tuple[ScoreRecord, ...]
    keys: tuple[tuple[SourceFileId, SourceRowIndex], ...]

    def __post_init__(self) -> None:
        if len(self.records) != len(self.keys):
            raise ValueError("score lookup records and keys must share length")
        if self.keys != tuple(sorted(self.keys)):
            raise ValueError("score lookup keys must be sorted")
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("score lookup contains duplicate (source_file, source_row) keys")

    @classmethod
    def from_records(cls, records: tuple[ScoreRecord, ...]) -> EventScoreLookup:
        ordered = tuple(sorted(records, key=lambda record: (record.source_file, record.source_row)))
        return cls(
            records=ordered,
            keys=tuple((record.source_file, record.source_row) for record in ordered),
        )

    def record_for(
        self, source_file: SourceFileId, source_row: SourceRowIndex
    ) -> ScoreRecord | None:
        key = (source_file, source_row)
        index = bisect_left(self.keys, key)
        if index < len(self.keys) and self.keys[index] == key:
            return self.records[index]
        return None


def collect_event_observations(
    dataset_id: DatasetId,
    raw_root: Path,
    catalog: DatasetCatalog,
    scores: tuple[ScoreRecord, ...],
) -> tuple[EventObservation, ...]:
    if not scores or dataset_id is not DatasetId.GOTHAM:
        return ()
    lookup = EventScoreLookup.from_records(scores)
    observations: list[EventObservation] = []
    for row in load_gotham_timed_rows(raw_root, catalog.gotham):
        record = lookup.record_for(row.source_file, row.source_row)
        if record is None:
            continue
        observations.append(
            EventObservation(
                client_id=row.client_id,
                timestamp=row.timestamp,
                score=record.score,
                label=row.label,
                source_file=row.source_file,
                source_row=row.source_row,
            )
        )
    return tuple(observations)


def eventize_alert_times(
    timestamps: np.ndarray,
    parameters: EventizationParameters,
) -> tuple[AlertEvent, ...]:
    sorted_times = np.sort(np.asarray(timestamps, dtype=np.float64))
    if sorted_times.size == 0:
        return ()
    dilated = _dilate_alert_times(sorted_times, parameters.dilation)
    merged = _merge_adjacent_intervals(dilated, parameters.merge_gap)
    candidates = tuple(
        block for block in merged if block.end - block.start >= parameters.minimum_event_length
    )
    return _apply_event_cooldown(candidates, sorted_times, parameters.cooldown)


def _dilate_alert_times(
    sorted_times: np.ndarray, dilation: DurationSeconds
) -> tuple[TimeInterval, ...]:
    blocks: list[TimeInterval] = []
    for timestamp in sorted_times:
        if blocks and timestamp - blocks[-1].end <= dilation:
            previous = blocks[-1]
            blocks[-1] = TimeInterval(start=previous.start, end=max(previous.end, timestamp))
            continue
        blocks.append(TimeInterval(start=timestamp, end=timestamp))
    return tuple(blocks)


def _merge_adjacent_intervals(
    blocks: tuple[TimeInterval, ...], merge_gap: DurationSeconds
) -> tuple[TimeInterval, ...]:
    merged: list[TimeInterval] = []
    for block in blocks:
        if merged and block.start - merged[-1].end <= merge_gap:
            previous = merged[-1]
            merged[-1] = TimeInterval(start=previous.start, end=block.end)
            continue
        merged.append(block)
    return tuple(merged)


def _apply_event_cooldown(
    candidates: tuple[TimeInterval, ...],
    sorted_times: np.ndarray,
    cooldown: DurationSeconds,
) -> tuple[AlertEvent, ...]:
    events: list[AlertEvent] = []
    for block in candidates:
        alarm_count = _alarm_count(sorted_times, block.start, block.end)
        if events and block.start - events[-1].end_time < cooldown:
            previous = events[-1]
            events[-1] = AlertEvent(
                start_time=previous.start_time,
                end_time=max(previous.end_time, block.end),
                alarm_count=previous.alarm_count + alarm_count,
            )
            continue
        events.append(
            AlertEvent(start_time=block.start, end_time=block.end, alarm_count=alarm_count)
        )
    return tuple(events)


def _detected_attack_count(
    attacked_times: np.ndarray, attack_intervals: tuple[TimeInterval, ...]
) -> RowCount:
    return sum(
        1
        for interval in attack_intervals
        if np.any((attacked_times >= interval.start) & (attacked_times <= interval.end))
    )


def _alarm_count(sorted_times: np.ndarray, start: EventTimestamp, end: EventTimestamp) -> RowCount:
    first = int(np.searchsorted(sorted_times, start, side="left"))
    last = int(np.searchsorted(sorted_times, end, side="right"))
    return last - first


def alarm_duty(event: AlertEvent, observation_duration: DurationSeconds) -> Probability:
    if observation_duration <= 0:
        raise ValueError("observation duration must be positive")
    return float((event.end_time - event.start_time) / observation_duration)


def event_gini(events_per_client: tuple[RowCount, ...]) -> Probability:
    counts = np.asarray(sorted(events_per_client), dtype=np.float64)
    if counts.size == 0 or float(counts.sum()) == 0:
        return 0.0
    cumulative = np.cumsum(counts)
    n = counts.size
    area = float(np.sum(cumulative) / cumulative[-1])
    return float((n + 1 - 2 * area) / n)


def _event_budget_threshold(
    observations: tuple[EventObservation, ...],
    observation_duration: DurationSeconds,
    budget_per_client_hour: EventRatePerClientHour,
    config: EventLevelConfig,
) -> Threshold | None:
    benign = tuple(item for item in observations if item.label is Label.BENIGN)
    if not benign or observation_duration <= 0.0:
        return None
    hours = observation_duration / _SECONDS_PER_HOUR
    parameters = _nominal_eventization(config)
    scores = np.asarray([item.score for item in benign], dtype=np.float64)
    times = np.asarray([item.timestamp for item in benign], dtype=np.float64)
    low = float(np.min(scores))
    high = float(np.max(scores))
    if low == high:
        return low

    def _rate(threshold: Threshold) -> EventRatePerClientHour:
        alerts = times[alerts_above_threshold(scores, threshold)]
        return len(eventize_alert_times(alerts, parameters)) / hours

    for _ in range(config.event_gate.budget_bisection_iterations):
        midpoint = (low + high) / 2.0
        if _rate(midpoint) <= budget_per_client_hour:
            high = midpoint
        else:
            low = midpoint
    if _rate(high) > budget_per_client_hour:
        return None
    return high


def run_event_level_metrics(
    config: EventLevelConfig,
    seed: DetectorSeed,
    budget: BudgetLevel,
    stream: ClientAlertStreams,
) -> EventMetricRow | None:
    values = _event_metric_values(stream, config)
    if values is None:
        return None
    return EventMetricRow(
        dataset_id=config.dataset_id,
        seed=seed,
        budget_id=budget.budget_id,
        target_rate=budget.value,
        client_id=stream.client_id,
        false_alert_events_per_hour=values.false_alert_events_per_hour,
        attacked_stream_events_per_hour=values.attacked_stream_events_per_hour,
        attack_event_recall=values.attack_event_recall,
        attack_event_miss_rate=values.attack_event_miss_rate,
        mean_time_to_detect=values.mean_time_to_detect,
        median_time_to_detect=values.median_time_to_detect,
        percentile_90_time_to_detect=values.percentile_90_time_to_detect,
        alarm_duty_fraction=values.alarm_duty_fraction,
    )


def _event_metric_values(
    stream: ClientAlertStreams,
    config: EventLevelConfig,
) -> EventMetricValues | None:
    parameters = _nominal_eventization(config)
    benign_events = eventize_alert_times(stream.benign_times, parameters)
    attacked_events = eventize_alert_times(stream.attacked_times, parameters)
    if not benign_events and not attacked_events:
        return None
    hours = stream.observation_duration / _SECONDS_PER_HOUR
    benign_rate = len(benign_events) / hours if hours > 0 else 0.0
    attacked_rate = len(attacked_events) / hours if hours > 0 else 0.0
    detected = _detected_attack_count(stream.attacked_times, stream.attack_intervals)
    detection_times: list[DurationSeconds] = []
    for interval in stream.attack_intervals:
        window = stream.attacked_times[
            (stream.attacked_times >= interval.start) & (stream.attacked_times <= interval.end)
        ]
        if window.size > 0:
            detection_times.append(float(float(np.min(window)) - interval.start))
    recall = detected / len(stream.attack_intervals) if stream.attack_intervals else 0.0
    miss_rate = 1.0 - recall
    duties = [alarm_duty(event, stream.observation_duration) for event in benign_events]
    duty_fraction = float(np.mean(duties)) if duties else 0.0
    if duty_fraction > config.event_gate.maximum_alarm_duty:
        logger.warning(
            "alarm duty %.3f exceeds the guardrail for %s", duty_fraction, stream.client_id
        )
    detection = np.asarray(detection_times, dtype=np.float64)
    return EventMetricValues(
        false_alert_events_per_hour=benign_rate,
        attacked_stream_events_per_hour=attacked_rate,
        attack_event_recall=recall,
        attack_event_miss_rate=miss_rate,
        mean_time_to_detect=float(np.mean(detection)) if detection.size else 0.0,
        median_time_to_detect=float(np.median(detection)) if detection.size else 0.0,
        percentile_90_time_to_detect=(
            float(np.percentile(detection, config.time_to_detect_percentile * 100.0))
            if detection.size
            else 0.0
        ),
        alarm_duty_fraction=duty_fraction,
    )


def _persist_event_rows(path: Path, rows: tuple[BaseModel, ...]) -> None:
    if not rows:
        return
    write_parquet_models(path, rows)


def _nominal_metric_and_gini_rows(
    config: EventLevelConfig,
    seed: DetectorSeed,
    budget_level: BudgetLevel,
    alert_streams: tuple[ClientAlertStreams, ...],
) -> tuple[tuple[EventMetricRow, ...], EventGiniRow | None]:
    nominal = _nominal_eventization(config)
    metric_rows: list[EventMetricRow] = []
    events_per_client: list[RowCount] = []
    for stream in alert_streams:
        row = run_event_level_metrics(config, seed, budget_level, stream)
        if row is None:
            continue
        metric_rows.append(row)
        events_per_client.append(len(eventize_alert_times(stream.benign_times, nominal)))
    gini_row = (
        EventGiniRow(
            dataset_id=config.dataset_id,
            seed=seed,
            budget_id=budget_level.budget_id,
            gini=event_gini(tuple(events_per_client)),
        )
        if events_per_client
        else None
    )
    return tuple(metric_rows), gini_row


def _eventization_grid(config: EventLevelConfig) -> tuple[EventizationParameters, ...]:
    sensitivity = config.event_gate.sensitivity
    return tuple(
        EventizationParameters(
            dilation=dilation,
            merge_gap=merge_gap,
            minimum_event_length=minimum_length,
            cooldown=cooldown,
        )
        for dilation, merge_gap, minimum_length, cooldown in product(
            sensitivity.dilation,
            sensitivity.merge_gap,
            sensitivity.minimum_event_length,
            sensitivity.cooldown,
        )
    )


def _attack_event_recall(stream: ClientAlertStreams) -> Probability:
    if not stream.attack_intervals:
        return 0.0
    return _detected_attack_count(stream.attacked_times, stream.attack_intervals) / len(
        stream.attack_intervals
    )


def _robustness_row(
    config: EventLevelConfig,
    seed: DetectorSeed,
    budget_id: BudgetId,
    parameters: EventizationParameters,
    stream: ClientAlertStreams,
) -> EventRobustnessRow:
    benign_events = eventize_alert_times(stream.benign_times, parameters)
    hours = stream.observation_duration / _SECONDS_PER_HOUR
    return EventRobustnessRow(
        dataset_id=config.dataset_id,
        seed=seed,
        budget_id=budget_id,
        dilation=parameters.dilation,
        merge_gap=parameters.merge_gap,
        minimum_event_length=parameters.minimum_event_length,
        cooldown=parameters.cooldown,
        false_alert_events_per_hour=len(benign_events) / hours if hours > 0 else 0.0,
        attack_event_recall=_attack_event_recall(stream),
    )


def _robustness_grid_rows(
    config: EventLevelConfig,
    seed: DetectorSeed,
    budget_id: BudgetId,
    alert_streams: tuple[ClientAlertStreams, ...],
) -> tuple[EventRobustnessRow, ...]:
    return tuple(
        _robustness_row(config, seed, budget_id, parameters, stream)
        for parameters in _eventization_grid(config)
        for stream in alert_streams
    )


def _client_event_budget_row(
    config: EventLevelConfig,
    seed: DetectorSeed,
    budget_per_client_hour: EventRatePerClientHour,
    items: tuple[EventObservation, ...],
    client_id: ClientId,
) -> EventBudgetRow | None:
    times = np.asarray([item.timestamp for item in items], dtype=np.float64)
    duration = float(np.max(times) - np.min(times)) if times.size else 0.0
    threshold = _event_budget_threshold(items, duration, budget_per_client_hour, config)
    if threshold is None:
        return None
    streams = build_client_alert_streams(items, threshold)
    if not streams:
        return None
    values = _event_metric_values(streams[0], config)
    if values is None:
        return None
    return EventBudgetRow(
        dataset_id=config.dataset_id,
        seed=seed,
        budget_per_client_hour=budget_per_client_hour,
        client_id=client_id,
        threshold=threshold,
        false_alert_events_per_hour=values.false_alert_events_per_hour,
        attack_event_recall=values.attack_event_recall,
        mean_time_to_detect=values.mean_time_to_detect,
    )


def _event_budget_rows(
    config: EventLevelConfig,
    seeds: tuple[DetectorSeed, ...],
    observations: tuple[EventObservation, ...],
    paths: ArtifactPaths,
) -> tuple[EventBudgetRow, ...]:
    observations_by_client = _group_observations_by_client(observations)
    rows: list[EventBudgetRow] = []
    total_seeds = len(seeds)
    for seed_index, seed in enumerate(seeds, start=1):
        report_progress(
            logger,
            paths,
            "event budget rows",
            seed_index,
            total_seeds,
            detail=f"seed {seed}",
        )
        for budget_per_client_hour in config.event_gate.budgets_per_client_hour:
            for client_id, client_observations in observations_by_client.items():
                row = _client_event_budget_row(
                    config,
                    seed,
                    budget_per_client_hour,
                    client_observations,
                    client_id,
                )
                if row is not None:
                    rows.append(row)
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class _EventLevelTables:
    metrics: tuple[EventMetricRow, ...]
    robustness: tuple[EventRobustnessRow, ...]
    gini: tuple[EventGiniRow, ...]
    budgets: tuple[EventBudgetRow, ...]

    @property
    def empty(self) -> bool:
        return not (self.metrics or self.robustness or self.gini or self.budgets)


def _event_level_tables(
    config: EventLevelConfig,
    seeds: tuple[DetectorSeed, ...],
    observations: tuple[EventObservation, ...],
    paths: ArtifactPaths,
) -> _EventLevelTables:
    metric_rows: list[EventMetricRow] = []
    robustness_rows: list[EventRobustnessRow] = []
    gini_rows: list[EventGiniRow] = []
    total_seeds = len(seeds)
    for seed_index, seed in enumerate(seeds, start=1):
        report_progress(
            logger,
            paths,
            "event tables",
            seed_index,
            total_seeds,
            detail=f"seed {seed}",
        )
        for budget_level in config.budgets:
            alert_streams = _streams_for_budget(observations, budget_level.value)
            seed_metrics, gini_row = _nominal_metric_and_gini_rows(
                config, seed, budget_level, alert_streams
            )
            metric_rows.extend(seed_metrics)
            if gini_row is not None:
                gini_rows.append(gini_row)
            robustness_rows.extend(
                _robustness_grid_rows(config, seed, budget_level.budget_id, alert_streams)
            )
    return _EventLevelTables(
        metrics=tuple(metric_rows),
        robustness=tuple(robustness_rows),
        gini=tuple(gini_rows),
        budgets=_event_budget_rows(config, seeds, observations, paths),
    )


def _combine_event_level_tables(tables: tuple[_EventLevelTables, ...]) -> _EventLevelTables:
    return _EventLevelTables(
        metrics=tuple(row for table in tables for row in table.metrics),
        robustness=tuple(row for table in tables for row in table.robustness),
        gini=tuple(row for table in tables for row in table.gini),
        budgets=tuple(row for table in tables for row in table.budgets),
    )


def _event_seed_tables_directory(paths: ArtifactPaths, seed: DetectorSeed) -> Path:
    return paths.event_analysis_dir() / "seed_tables" / f"seed-{seed:03d}"


def _release_event_seed_memory() -> None:
    gc.collect()
    if sys.platform == "linux":
        ctypes.CDLL(None).malloc_trim(0)


def _persist_event_seed_tables(
    paths: ArtifactPaths, seed: DetectorSeed, tables: _EventLevelTables
) -> None:
    directory = _event_seed_tables_directory(paths, seed)
    directory.mkdir(parents=True, exist_ok=True)
    _persist_event_rows(directory / ArtifactFileName.EVENT_METRICS, tables.metrics)
    _persist_event_rows(directory / ArtifactFileName.EVENT_ROBUSTNESS, tables.robustness)
    _persist_event_rows(directory / ArtifactFileName.EVENT_GINI, tables.gini)
    _persist_event_rows(directory / ArtifactFileName.EVENT_BUDGET, tables.budgets)
    (directory / ".complete").touch()


def _load_event_seed_tables(paths: ArtifactPaths, seed: DetectorSeed) -> _EventLevelTables | None:
    directory = _event_seed_tables_directory(paths, seed)
    if not (directory / ".complete").is_file():
        return None
    return _EventLevelTables(
        metrics=tuple(
            read_parquet_models(directory / ArtifactFileName.EVENT_METRICS, EventMetricRow)
        ),
        robustness=tuple(
            read_parquet_models(directory / ArtifactFileName.EVENT_ROBUSTNESS, EventRobustnessRow)
        ),
        gini=tuple(read_parquet_models(directory / ArtifactFileName.EVENT_GINI, EventGiniRow)),
        budgets=tuple(
            read_parquet_models(directory / ArtifactFileName.EVENT_BUDGET, EventBudgetRow)
        ),
    )


def _persist_event_level_tables(
    paths: ArtifactPaths,
    tables: _EventLevelTables,
) -> None:
    event_root = paths.event_analysis_dir()
    event_root.mkdir(parents=True, exist_ok=True)
    _persist_event_rows(event_root / ArtifactFileName.EVENT_METRICS, tables.metrics)
    _persist_event_rows(event_root / ArtifactFileName.EVENT_ROBUSTNESS, tables.robustness)
    _persist_event_rows(event_root / ArtifactFileName.EVENT_GINI, tables.gini)
    _persist_event_rows(event_root / ArtifactFileName.EVENT_BUDGET, tables.budgets)
    logger.info(
        "[SAVE] event metrics %d rows, robustness grid %d rows, event budgets %d rows",
        len(tables.metrics),
        len(tables.robustness),
        len(tables.budgets),
    )


def run_event_level(
    config: EventLevelConfig,
    paths: ArtifactPaths,
    seeds: tuple[DetectorSeed, ...],
    evidence: EventProvenanceEvidence,
    observations: tuple[EventObservation, ...],
) -> tuple[EventMetricRow, ...]:
    gate = assess_event_data_gate(config.dataset_id, evidence)
    if gate.status is not GateStatus.PASS:
        logger.info(
            "[GATE] event data gate %s; no event-level workload claim",
            gate.status.value,
        )
        return ()
    if not observations:
        logger.info(
            "[GATE] event data gate passed but no timed scored rows; no event-level claim",
        )
        return ()
    tables = _event_level_tables(config, seeds, observations, paths)
    if tables.empty:
        logger.info("event data gate passed but no alert-stream metrics; no event-level claim")
        return ()
    _persist_event_level_tables(paths, tables)
    return tables.metrics


def _streams_for_budget(
    observations: tuple[EventObservation, ...],
    target_rate: TargetFalsePositiveRate,
) -> tuple[ClientAlertStreams, ...]:
    clients = tuple(sorted({item.client_id for item in observations}))
    streams: list[ClientAlertStreams] = []
    for client_id in clients:
        items = _observations_for_client(observations, client_id)
        benign_scores = np.asarray(
            [item.score for item in items if item.label is Label.BENIGN],
            dtype=np.float64,
        )
        if benign_scores.size == 0:
            continue
        threshold = calibrate_threshold(benign_scores, target_rate)
        streams.extend(build_client_alert_streams(items, threshold))
    return tuple(streams)


def _audit_event_dataset(
    config: EventLevelConfig,
    raw_root: Path,
    catalog: DatasetCatalog,
) -> EventProvenanceEvidence:
    if config.dataset_id is DatasetId.GOTHAM:
        return audit_gotham_event_provenance(
            raw_root,
            catalog.gotham,
            config.event_gate.minimum_timestamp_parse_success,
            config.event_gate.maximum_out_of_order_fraction,
            config.event_gate.minimum_capture_seam_seconds,
        )
    if config.dataset_id is DatasetId.CICIOMT:
        return audit_ciciomt_event_provenance(raw_root, catalog.ciciomt)
    raise DatasetError(f"no event-provenance auditor for {config.dataset_id.value}")


def _event_training_settings(config: EventLevelConfig) -> FederatedTrainingSettings:
    detector = config.detector
    return FederatedTrainingSettings(
        learning_rate=detector.learning_rate,
        local_epochs=detector.local_epochs,
        rounds=detector.rounds,
        batch_size=detector.batch_size,
    )


def _client_benign_indices(client: GothamNumericClient) -> tuple[SourceRowIndex, ...]:
    return tuple(index for index, label in enumerate(client.labels) if label is Label.BENIGN)


def _rows_at(client: GothamNumericClient, indices: tuple[SourceRowIndex, ...]) -> FeatureMatrix:
    if not indices:
        return FeatureMatrix(np.empty((0, client.features.feature_count), dtype=np.float64))
    return FeatureMatrix(client.features.values[np.asarray(indices, dtype=np.int64)])


def _trainable_gotham_clients(
    clients: tuple[GothamNumericClient, ...],
    config: EventLevelConfig,
) -> tuple[GothamNumericClient, ...]:
    trainable: list[GothamNumericClient] = []
    for client in clients:
        benign_count = len(_client_benign_indices(client))
        train_end = compute_benign_split_boundaries(
            benign_count,
            config.benign_splits.train_end,
            config.benign_splits.frontier_end,
            config.benign_splits.final_cal_end,
        ).train_end
        if train_end > 0:
            trainable.append(client)
    return tuple(trainable)


@dataclass(frozen=True, slots=True)
class _ClientSplitPlans:
    benign: BenignSplitBoundaries
    attacks: tuple[AttackSubtypeBoundary, ...]

    def boundary_for(self, subtype: AttackSubtypeId) -> AttackSplitBoundary:
        for item in self.attacks:
            if item.subtype == subtype:
                return item.boundary
        raise ValueError(f"no split plan for attack subtype {subtype!r}")


def _client_split_plans(client: GothamNumericClient, config: EventLevelConfig) -> _ClientSplitPlans:
    benign = compute_benign_split_boundaries(
        len(_client_benign_indices(client)),
        config.benign_splits.train_end,
        config.benign_splits.frontier_end,
        config.benign_splits.final_cal_end,
    )
    counts: defaultdict[AttackSubtypeId, RowCount] = defaultdict(lambda: ZERO_ROW_COUNT)
    for subtype in client.attack_subtypes:
        if subtype is not None:
            counts[subtype] += 1
    attacks = tuple(
        AttackSubtypeBoundary(
            subtype=subtype,
            boundary=compute_attack_split_boundary(count, config.attack_split.validation_end),
        )
        for subtype, count in sorted(counts.items())
    )
    return _ClientSplitPlans(benign=benign, attacks=attacks)


def _score_record(
    client: GothamNumericClient,
    row_index: SourceRowIndex,
    score: AnomalyScore,
) -> ScoreRecord:
    return ScoreRecord(
        sample_id=f"{client.client_id}|{client.source_files[row_index]}|{client.source_rows[row_index]}",
        source_file=client.source_files[row_index],
        source_row=client.source_rows[row_index],
        score=score,
        label=client.labels[row_index],
        attack_subtype=client.attack_subtypes[row_index],
        timestamp=None,
    )


@dataclass(frozen=True, slots=True)
class _BenignScoreBucket:
    split: BenignSplit
    record: ScoreRecord


@dataclass(frozen=True, slots=True)
class _AttackScoreBucket:
    split: AttackSplit
    record: ScoreRecord


def _persist_score_buckets(
    paths: ArtifactPaths,
    coordinate: ScoreCoordinate,
    benign: tuple[_BenignScoreBucket, ...],
    attacks: tuple[_AttackScoreBucket, ...],
) -> None:
    for split in BenignSplit:
        records = tuple(item.record for item in benign if item.split is split)
        if not records:
            continue
        persist_score_partition(
            paths.score_path(coordinate, split),
            ScorePartitionArtifact(coordinate=coordinate, split=split, records=records),
        )
    for split in AttackSplit:
        records = tuple(item.record for item in attacks if item.split is split)
        if not records:
            continue
        persist_score_partition(
            paths.score_path(coordinate, split),
            ScorePartitionArtifact(coordinate=coordinate, split=split, records=records),
        )


def _persist_client_event_scores(
    paths: ArtifactPaths,
    coordinate: ScoreCoordinate,
    client: GothamNumericClient,
    scores: np.ndarray,
    config: EventLevelConfig,
) -> None:
    plans = _client_split_plans(client, config)
    benign: list[_BenignScoreBucket] = []
    attacks: list[_AttackScoreBucket] = []
    benign_rank = 0
    attack_ranks: defaultdict[AttackSubtypeId, RowCount] = defaultdict(lambda: ZERO_ROW_COUNT)
    for row_index, score in enumerate(scores):
        record = _score_record(client, row_index, float(score))
        if record.label is Label.BENIGN:
            benign.append(
                _BenignScoreBucket(
                    split=plans.benign.split_of(benign_rank),
                    record=record,
                )
            )
            benign_rank += 1
            continue
        subtype = record.attack_subtype
        if subtype is None:
            raise DatasetError("gotham attack score is missing an attack subtype")
        rank = attack_ranks[subtype]
        attacks.append(
            _AttackScoreBucket(
                split=plans.boundary_for(subtype).split_of(rank),
                record=record,
            )
        )
        attack_ranks[subtype] = rank + 1
    _persist_score_buckets(paths, coordinate, tuple(benign), tuple(attacks))


def gotham_seed_scores_materialized(
    paths: ArtifactPaths,
    dataset_id: DatasetId,
    seed: DetectorSeed,
    clients: tuple[GothamNumericClient, ...],
) -> bool:
    for client in clients:
        coordinate = ScoreCoordinate(
            dataset_id=dataset_id,
            detector_seed=seed,
            client_id=client.client_id,
        )
        splits = (
            BenignSplit.FRONTIER,
            BenignSplit.FINAL_CAL,
            BenignSplit.TEST,
            AttackSplit.VALIDATION,
            AttackSplit.TEST,
        )
        if not any(paths.score_path(coordinate, split).exists() for split in splits):
            return False
    return True


def _gotham_event_cache_directory(paths: ArtifactPaths, cache_key: ArtifactDigest) -> Path:
    return (
        paths.preprocessing_dir(DatasetId.GOTHAM)
        / "event_client_cache"
        / (f"clients_mmap_v1_{cache_key[:16]}")
    )


def _gotham_event_cache_key(event_root: Path, catalog: DatasetCatalog) -> ArtifactDigest:
    captures = gotham_capture_paths(event_root, catalog.gotham)
    fingerprints = tuple(
        f"{path.relative_to(event_root)}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
        for path in captures
    )
    return digest_text(
        (
            _GOTHAM_EVENT_CACHE_VERSION,
            str(event_root.resolve()),
            catalog.gotham.model_dump_json(),
            *fingerprints,
        )
    )


def _load_gotham_event_client_cache(
    directory: Path,
    expected_key: ArtifactDigest,
) -> tuple[GothamNumericClient, ...] | None:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = _GothamEventCacheManifest.model_validate_json(manifest_path.read_bytes())
        if manifest.cache_key != expected_key or manifest.finite_validated is not True:
            return None
        clients = tuple(
            GothamNumericClient(
                client_id=item.client_id,
                source_files=item.source_files,
                source_rows=item.source_rows,
                labels=item.labels,
                attack_subtypes=item.attack_subtypes,
                features=FeatureMatrix.from_cached_values(
                    np.load(directory / item.features, mmap_mode="r")
                ),
            )
            for item in manifest.clients
        )
    except (OSError, TypeError, ValueError):
        logger.warning("[CACHE] Gotham memory-mapped event cache is unreadable")
        return None
    logger.info("[CACHE] reused Gotham memory-mapped event clients (%d clients)", len(clients))
    return clients


def _save_gotham_event_client_cache(
    directory: Path,
    cache_key: ArtifactDigest,
    clients: tuple[GothamNumericClient, ...],
) -> None:
    if directory.exists():
        return
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gotham_event_clients_", dir=directory.parent))
    try:
        serialized_clients: list[_GothamEventCacheClient] = []
        for index, client in enumerate(clients):
            features_name = f"client_{index}_features.npy"
            np.save(staging / features_name, client.features.values, allow_pickle=False)
            serialized_clients.append(
                _GothamEventCacheClient(
                    client_id=client.client_id,
                    source_files=client.source_files,
                    source_rows=client.source_rows,
                    labels=client.labels,
                    attack_subtypes=client.attack_subtypes,
                    features=features_name,
                )
            )
        manifest = _GothamEventCacheManifest(
            cache_key=cache_key,
            finite_validated=True,
            clients=tuple(serialized_clients),
        )
        (staging / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        try:
            os.replace(staging, directory)
        except FileExistsError:
            shutil.rmtree(staging, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger.info("[CACHE] saved Gotham memory-mapped event clients (%d clients)", len(clients))


def _load_or_build_gotham_event_clients(
    paths: ArtifactPaths,
    event_root: Path,
    catalog: DatasetCatalog,
) -> _GothamEventClientCache:
    cache_key = _gotham_event_cache_key(event_root, catalog)
    directory = _gotham_event_cache_directory(paths, cache_key)
    clients = _load_gotham_event_client_cache(directory, cache_key)
    if clients is None:
        raw_clients = load_gotham_numeric_features(event_root, catalog.gotham)
        _save_gotham_event_client_cache(directory, cache_key, raw_clients)
        del raw_clients
        gc.collect()
        clients = _load_gotham_event_client_cache(directory, cache_key)
    if clients is None:
        raise DatasetError("Gotham memory-mapped event client cache could not be loaded")
    return _GothamEventClientCache(directory=directory, cache_key=cache_key, clients=clients)


def _initialize_gotham_event_worker(
    directory: Path,
    cache_key: ArtifactDigest,
    cache_initialization_lock: _ExclusiveInitializationLock,
) -> None:
    global _spawn_gotham_event_clients

    with cache_initialization_lock:
        clients = _load_gotham_event_client_cache(directory, cache_key)
    if clients is None:
        raise DatasetError("Gotham event worker could not load its immutable client cache")
    _spawn_gotham_event_clients = clients


def _materialize_gotham_event_seed_worker(
    paths: ArtifactPaths,
    config: EventLevelConfig,
    seed: DetectorSeed,
) -> DetectorSeed:
    if _spawn_gotham_event_clients is None:
        raise RuntimeError("Gotham event worker was not initialized")
    _materialize_gotham_event_scores(paths, config, seed, _spawn_gotham_event_clients)
    return seed


def _event_round_progress_reporter(
    paths: ArtifactPaths,
    seed: DetectorSeed,
) -> RoundProgressReporter:
    def report(
        round_index: FederatedRoundCount,
        total_rounds: FederatedRoundCount,
    ) -> None:
        write_typed_json(
            ProgressState(
                phase="event federated training",
                completed=round_index,
                total=total_rounds,
                detail=f"seed {seed}",
                updated_at=now_iso(),
            ),
            TypeAdapter(ProgressState),
            paths.event_training_progress_path(seed),
        )

    return report


def _materialize_gotham_event_scores(
    paths: ArtifactPaths,
    config: EventLevelConfig,
    seed: DetectorSeed,
    clients: tuple[GothamNumericClient, ...],
) -> None:
    trainable = _trainable_gotham_clients(clients, config)
    if not trainable:
        raise DatasetError(
            "Gotham event-level training requires at least one client with benign train rows"
        )
    if gotham_seed_scores_materialized(paths, config.dataset_id, seed, trainable):
        logger.info("[REUSE] gotham event scores seed=%d", seed)
        return
    fitted_scalers: list[ClientScaler] = []
    training_clients: list[ClientTrainingData] = []
    for client in trainable:
        benign_indices = _client_benign_indices(client)
        train_end = compute_benign_split_boundaries(
            len(benign_indices),
            config.benign_splits.train_end,
            config.benign_splits.frontier_end,
            config.benign_splits.final_cal_end,
        ).train_end
        train_features = _rows_at(client, benign_indices[:train_end])
        scaler = fit_feature_scaler(train_features)
        fitted_scalers.append(ClientScaler(client_id=client.client_id, scaler=scaler))
        training_clients.append(
            ClientTrainingData(
                client_id=client.client_id,
                features=scaler.transform(train_features),
            )
        )
    scalers = FederatedScalers(tuple(fitted_scalers))
    model = train_federated_autoencoder(
        FederatedTrainingData(clients=tuple(training_clients)),
        AutoencoderArchitecture(
            feature_count=trainable[0].features.feature_count,
            hidden_layers=config.detector.hidden_layers,
        ),
        _event_training_settings(config),
        seed,
        resolve_device(),
        _event_round_progress_reporter(paths, seed),
    )
    device = resolve_device()
    for client in trainable:
        coordinate = ScoreCoordinate(
            dataset_id=config.dataset_id,
            detector_seed=seed,
            client_id=client.client_id,
        )
        scores = score_feature_matrix(
            model,
            scalers.for_client(client.client_id).transform(client.features),
            device,
            config.detector.score_batch_size,
        )
        _persist_client_event_scores(paths, coordinate, client, scores, config)


def prepare_and_run_event_level(context: EventLevelContext) -> tuple[EventMetricRow, ...]:
    application = context.config
    event_config = application.event_level
    event_root = context.raw_dataset_root(event_config.dataset_id)
    evidence = _audit_event_dataset(event_config, event_root, application.datasets)
    gate = assess_event_data_gate(event_config.dataset_id, evidence)
    if gate.status is not GateStatus.PASS:
        logger.info(
            "[GATE] event data gate %s; no event-level workload claim",
            gate.status.value,
        )
        return ()
    if event_config.dataset_id is DatasetId.GOTHAM:
        client_cache = _load_or_build_gotham_event_clients(
            context.paths, event_root, application.datasets
        )
        trainable = _trainable_gotham_clients(client_cache.clients, event_config)
        pending_seeds = tuple(
            seed
            for seed in event_config.detector.seeds
            if not gotham_seed_scores_materialized(
                context.paths, event_config.dataset_id, seed, trainable
            )
        )
        if pending_seeds:
            workers = min(application.execution.event_workers, len(pending_seeds))
            logger.info(
                "[RUN] Gotham event scoring %d pending seeds with %d CUDA worker(s)",
                len(pending_seeds),
                workers,
            )
            if workers == 1:
                for seed in pending_seeds:
                    _materialize_gotham_event_scores(
                        context.paths, event_config, seed, client_cache.clients
                    )
                del trainable
                del client_cache
            else:
                cache_directory = client_cache.directory
                cache_key = client_cache.cache_key
                del trainable
                del client_cache
                gc.collect()
                spawn_context = get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=spawn_context,
                    initializer=_initialize_gotham_event_worker,
                    initargs=(cache_directory, cache_key, spawn_context.Lock()),
                ) as executor:
                    futures = tuple(
                        executor.submit(
                            _materialize_gotham_event_seed_worker,
                            context.paths,
                            event_config,
                            seed,
                        )
                        for seed in pending_seeds
                    )
                    for future in as_completed(futures):
                        completed_seed = future.result()
                        logger.info(
                            "[PROGRESS] Gotham event scoring seed=%d complete", completed_seed
                        )
        else:
            logger.info("[REUSE] all Gotham event seed scores are materialized")
            del trainable
            del client_cache
        gc.collect()
    seed_tables: list[_EventLevelTables] = []
    for seed in event_config.detector.seeds:
        cached_tables = _load_event_seed_tables(context.paths, seed)
        if cached_tables is not None:
            logger.info("[REUSE] event aggregation seed=%d", seed)
            seed_tables.append(cached_tables)
            continue
        observations = collect_event_observations(
            event_config.dataset_id,
            event_root,
            application.datasets,
            load_event_score_records(context.paths, event_config.dataset_id, seed),
        )
        if not observations:
            continue
        tables_for_seed = _event_level_tables(event_config, (seed,), observations, context.paths)
        _persist_event_seed_tables(context.paths, seed, tables_for_seed)
        seed_tables.append(tables_for_seed)
        del observations
        _release_event_seed_memory()
    tables = _combine_event_level_tables(tuple(seed_tables))
    if tables.empty:
        logger.info(
            "[GATE] event data gate passed but no timed scored rows; no event-level claim",
        )
        return ()
    _persist_event_level_tables(context.paths, tables)
    return tables.metrics
