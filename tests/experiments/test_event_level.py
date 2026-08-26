from __future__ import annotations

from pathlib import Path

import numpy as np

from fabrid.artifacts.paths import ScoreCoordinate
from fabrid.config import (
    BenignSplit,
    DatasetId,
    DetectorConfig,
    EventCriterionId,
    EventLevelConfig,
    GateStatus,
    Label,
)
from fabrid.detector.scoring import ScorePartitionArtifact, ScoreRecord, persist_score_partition
from fabrid.experiments.event_level import (
    ClientAlertStreams,
    EventDataGateAssessment,
    EventObservation,
    TimeInterval,
    assess_event_data_gate,
    build_client_alert_streams,
    collect_event_observations,
    load_event_score_records,
    prepare_and_run_event_level,
    run_event_level,
    run_event_level_metrics,
)
from tests.support import (
    event_evidence,
    isolated_context,
    isolated_paths,
    production_application,
)


def test_assess_event_data_gate_passes_with_full_evidence() -> None:
    assessment = assess_event_data_gate(DatasetId.GOTHAM, event_evidence())
    assert assessment.status is GateStatus.PASS


def test_assess_event_data_gate_fails_on_missing_criterion() -> None:
    evidence = event_evidence(failing=EventCriterionId.INTERVAL_PROVENANCE)
    assessment = assess_event_data_gate(DatasetId.GOTHAM, evidence)
    assert assessment.status is GateStatus.FAIL


def test_event_data_gate_assessment_rejects_inconsistent_status() -> None:
    try:
        EventDataGateAssessment(
            dataset_id=DatasetId.GOTHAM,
            evidence=event_evidence(),
            status=GateStatus.FAIL,
        )
    except ValueError:
        return
    raise AssertionError("inconsistent gate status must raise")


def _config() -> EventLevelConfig:
    return production_application().event_level


def _observations() -> tuple[EventObservation, ...]:
    rows: list[EventObservation] = []
    for index, timestamp in enumerate(
        (0.0, 1.0, 2.0, 3.0, 60.0, 61.0, 62.0, 63.0, 120.0, 121.0, 122.0, 123.0)
    ):
        rows.append(
            EventObservation(
                client_id="device_a",
                timestamp=timestamp,
                score=0.8 if timestamp >= 60.0 else 0.1,
                label=Label.BENIGN,
                source_file="capture.csv",
                source_row=index,
            )
        )
    for offset, timestamp in enumerate((10.0, 11.0, 12.0, 13.0, 40.0, 41.0, 42.0, 43.0)):
        rows.append(
            EventObservation(
                client_id="device_a",
                timestamp=timestamp,
                score=0.9,
                label=Label.ATTACK,
                source_file="capture.csv",
                source_row=100 + offset,
            )
        )
    for index in range(2000):
        rows.append(
            EventObservation(
                client_id="device_a",
                timestamp=200.0 + index * 0.01,
                score=0.0,
                label=Label.BENIGN,
                source_file="capture.csv",
                source_row=1000 + index,
            )
        )
    return tuple(rows)


def _alert_streams() -> tuple[ClientAlertStreams, ...]:
    return build_client_alert_streams(_observations(), threshold=0.5)


def test_run_event_level_gate_pass_without_observations_does_not_write(tmp_path: Path) -> None:
    paths = isolated_paths(tmp_path)
    rows = run_event_level(
        _config(),
        paths,
        (0,),
        event_evidence(),
        (),
    )
    assert rows == ()
    event_dir = paths.event_analysis_dir()
    assert not (event_dir / "event_metrics.parquet").exists()


def test_build_client_alert_streams_thresholds_scored_rows() -> None:
    streams = build_client_alert_streams(_observations(), threshold=0.5)
    assert len(streams) == 1
    assert streams[0].attacked_times.size == 16
    assert streams[0].benign_times.size == 8
    assert streams[0].attack_intervals[0].start == 10.0


def test_run_event_level_gate_fail_produces_no_claim(tmp_path: Path) -> None:
    paths = isolated_paths(tmp_path)
    config = _config()
    rows = run_event_level(
        config,
        paths,
        (0,),
        event_evidence(failing=EventCriterionId.INTERVAL_PROVENANCE),
        _observations(),
    )
    assert rows == ()


def test_run_event_level_gate_pass_produces_metrics(tmp_path: Path) -> None:
    paths = isolated_paths(tmp_path)
    config = _config()
    rows = run_event_level(
        config,
        paths,
        (0,),
        event_evidence(),
        _observations(),
    )
    assert rows
    assert all(row.attack_event_recall >= 0.0 for row in rows)
    event_metrics = paths.event_analysis_dir() / "event_metrics.parquet"
    assert event_metrics.exists()
    robustness = paths.event_analysis_dir() / "event_robustness.parquet"
    assert robustness.exists()
    event_budgets = paths.event_analysis_dir() / "event_budget.parquet"
    assert event_budgets.exists()


def test_collect_event_observations_joins_gotham_rows_to_scores(tmp_path: Path) -> None:
    from fabrid.detector.scoring import ScoreRecord
    from fabrid.experiments.event_level import collect_event_observations
    from tests.datasets.test_gotham import write_gotham_capture

    application = production_application()
    layout = application.datasets.gotham
    write_gotham_capture(
        tmp_path,
        "capture1.csv",
        [
            ("aa", layout.benign_label, "Jan 01, 2020 00:00:00.000000000 GMT"),
            ("aa", "attack-x", "Jan 01, 2020 00:00:02.000000000 GMT"),
        ],
    )
    scores = (
        ScoreRecord(
            sample_id="aa|0",
            source_file="capture1.csv",
            source_row=0,
            score=0.2,
            label=Label.BENIGN,
            attack_subtype=None,
            timestamp=1.0,
        ),
        ScoreRecord(
            sample_id="aa|1",
            source_file="capture1.csv",
            source_row=1,
            score=0.9,
            label=Label.ATTACK,
            attack_subtype="attack-x",
            timestamp=2.0,
        ),
    )
    observations = collect_event_observations(
        DatasetId.GOTHAM, tmp_path, application.datasets, scores
    )
    assert len(observations) == 2
    assert observations[0].client_id == "aa"
    assert observations[0].label is Label.BENIGN
    assert observations[1].label is Label.ATTACK
    streams = build_client_alert_streams(observations, threshold=0.5)
    assert streams[0].attacked_times.size == 1


def test_run_event_level_metrics_recall_and_rates() -> None:
    config = _config()
    row = run_event_level_metrics(
        config,
        0,
        config.budgets[0],
        _alert_streams()[0],
    )
    assert row is not None
    assert row.attack_event_recall == 1.0
    assert row.false_alert_events_per_hour > 0.0
    assert row.mean_time_to_detect >= 0.0


def test_time_to_detect_uses_earliest_timestamp_in_interval() -> None:
    stream = ClientAlertStreams(
        client_id="device_a",
        benign_times=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        attacked_times=np.asarray([15.0, 10.0, 12.0], dtype=np.float64),
        attack_intervals=(TimeInterval(start=10.0, end=20.0),),
        observation_duration=20.0,
    )
    row = run_event_level_metrics(_config(), 0, _config().budgets[0], stream)
    assert row is not None
    assert row.mean_time_to_detect == 0.0


def test_build_client_alert_streams_sorts_alert_times() -> None:
    observations = (
        EventObservation(
            client_id="device_a",
            timestamp=15.0,
            score=0.9,
            label=Label.ATTACK,
            source_file="capture.csv",
            source_row=0,
        ),
        EventObservation(
            client_id="device_a",
            timestamp=10.0,
            score=0.9,
            label=Label.ATTACK,
            source_file="capture.csv",
            source_row=1,
        ),
        EventObservation(
            client_id="device_a",
            timestamp=12.0,
            score=0.9,
            label=Label.ATTACK,
            source_file="capture.csv",
            source_row=2,
        ),
        EventObservation(
            client_id="device_a",
            timestamp=0.0,
            score=0.0,
            label=Label.BENIGN,
            source_file="capture.csv",
            source_row=3,
        ),
    )
    streams = build_client_alert_streams(observations, threshold=0.5)
    assert streams[0].attacked_times.tolist() == [10.0, 12.0, 15.0]


def _score_record(source_row: int, score: float) -> ScoreRecord:
    return ScoreRecord(
        sample_id=f"aa|{source_row}",
        source_file="capture1.csv",
        source_row=source_row,
        score=score,
        label=Label.BENIGN,
        attack_subtype=None,
        timestamp=None,
    )


def test_load_event_score_records_ignores_other_seeds(tmp_path: Path) -> None:
    paths = isolated_paths(tmp_path)
    first = ScoreCoordinate(dataset_id=DatasetId.GOTHAM, detector_seed=0, client_id="aa")
    second = ScoreCoordinate(dataset_id=DatasetId.GOTHAM, detector_seed=1, client_id="aa")
    persist_score_partition(
        paths.score_path(first, BenignSplit.TEST),
        ScorePartitionArtifact(
            coordinate=first,
            split=BenignSplit.TEST,
            records=(_score_record(0, 0.1),),
        ),
    )
    persist_score_partition(
        paths.score_path(second, BenignSplit.TEST),
        ScorePartitionArtifact(
            coordinate=second,
            split=BenignSplit.TEST,
            records=(_score_record(1, 0.9),),
        ),
    )
    loaded = load_event_score_records(paths, DatasetId.GOTHAM, 0)
    assert len(loaded) == 1
    assert loaded[0].source_row == 0
    assert loaded[0].score == 0.1


def test_gotham_seed_scores_materialized_requires_all_clients(tmp_path: Path) -> None:
    from fabrid.datasets.gotham import GothamNumericClient
    from fabrid.datasets.registry import FeatureMatrix
    from fabrid.experiments.event_level import gotham_seed_scores_materialized

    paths = isolated_paths(tmp_path)

    def client(client_id: str) -> GothamNumericClient:
        return GothamNumericClient(
            client_id=client_id,
            source_files=("capture.csv", "capture.csv"),
            source_rows=(0, 1),
            labels=(Label.BENIGN, Label.ATTACK),
            attack_subtypes=(None, "attack-x"),
            features=FeatureMatrix(np.zeros((2, 4), dtype=np.float64)),
        )

    clients = (client("aa"), client("bb"))
    assert not gotham_seed_scores_materialized(paths, DatasetId.GOTHAM, 0, clients)
    coordinate = ScoreCoordinate(dataset_id=DatasetId.GOTHAM, detector_seed=0, client_id="aa")
    persist_score_partition(
        paths.score_path(coordinate, BenignSplit.TEST),
        ScorePartitionArtifact(
            coordinate=coordinate,
            split=BenignSplit.TEST,
            records=(_score_record(0, 0.1),),
        ),
    )
    assert not gotham_seed_scores_materialized(paths, DatasetId.GOTHAM, 0, clients)
    coordinate = ScoreCoordinate(dataset_id=DatasetId.GOTHAM, detector_seed=0, client_id="bb")
    persist_score_partition(
        paths.score_path(coordinate, BenignSplit.TEST),
        ScorePartitionArtifact(
            coordinate=coordinate,
            split=BenignSplit.TEST,
            records=(_score_record(1, 0.9),),
        ),
    )
    assert gotham_seed_scores_materialized(paths, DatasetId.GOTHAM, 0, clients)


def test_collect_event_observations_skips_unmatched_join_keys(tmp_path: Path) -> None:
    from tests.datasets.test_gotham import write_gotham_capture

    application = production_application()
    layout = application.datasets.gotham
    write_gotham_capture(
        tmp_path,
        "capture1.csv",
        [
            ("aa", layout.benign_label, "Jan 01, 2020 00:00:00.000000000 GMT"),
            ("aa", "attack-x", "Jan 01, 2020 00:00:02.000000000 GMT"),
        ],
    )
    scores = (
        ScoreRecord(
            sample_id="aa|0",
            source_file="capture1.csv",
            source_row=0,
            score=0.2,
            label=Label.BENIGN,
            attack_subtype=None,
            timestamp=1.0,
        ),
        ScoreRecord(
            sample_id="aa|99",
            source_file="other.csv",
            source_row=1,
            score=0.9,
            label=Label.ATTACK,
            attack_subtype="attack-x",
            timestamp=2.0,
        ),
    )
    observations = collect_event_observations(
        DatasetId.GOTHAM, tmp_path, application.datasets, scores
    )
    assert len(observations) == 1
    assert observations[0].source_file == "capture1.csv"
    assert observations[0].source_row == 0


def test_prepare_and_run_event_level_gate_fail_on_missing_files(tmp_path: Path) -> None:
    context = isolated_context(production_application(), tmp_path)
    rows = prepare_and_run_event_level(context)
    assert rows == ()
    event_dir = context.paths.event_analysis_dir()
    assert not (event_dir / "event_metrics.parquet").exists()


def test_prepare_and_run_event_level_materializes_gotham_scores(tmp_path: Path) -> None:
    from tests.datasets.test_gotham import write_gotham_capture

    application = production_application()
    layout = application.datasets.gotham
    raw_root = tmp_path / "raw" / layout.directory_name
    rows: list[tuple[str, str, str]] = []
    feature_rows: list[tuple[str, ...]] = []
    for index in range(20):
        rows.append(
            (
                "aa",
                layout.benign_label,
                f"Jan 01, 2020 00:00:{index:02d}.000000000 GMT",
            )
        )
        feature_rows.append((f"{1.0 + 0.01 * index}",))
    for index in range(8):
        rows.append(
            (
                "aa",
                "attack-x",
                f"Jan 01, 2020 00:01:{index:02d}.000000000 GMT",
            )
        )
        feature_rows.append((f"{8.0 + 0.1 * index}",))
    write_gotham_capture(
        raw_root,
        "capture1.csv",
        rows,
        feature_rows=tuple(feature_rows),
    )
    tiny_detector = DetectorConfig(
        hidden_layers=(4, 2),
        learning_rate=0.01,
        local_epochs=1,
        rounds=1,
        batch_size=8,
        score_batch_size=16,
        seeds=(0,),
    )
    event_config = application.event_level.model_copy(update={"detector": tiny_detector})
    isolated = application.model_copy(update={"event_level": event_config})
    context = isolated_context(isolated, tmp_path)
    prepare_and_run_event_level(context)
    assert (context.paths.event_analysis_dir() / "seed_tables" / "seed-000" / ".complete").is_file()
    loaded = load_event_score_records(context.paths, DatasetId.GOTHAM, 0)
    assert loaded
    assert all(record.source_file == "capture1.csv" for record in loaded)
    assert all(record.source_file != "preprocessed" for record in loaded)
    assert {record.source_row for record in loaded} == set(range(28))
