from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from fabrid.config import (
    AllocationPolicy,
    AnalysisArtifactId,
    AttackSplit,
    BenignSplit,
    BudgetId,
    BudgetLevel,
    CampaignAnalysisId,
    ClientId,
    DatasetId,
    DetectorSeed,
    ExperimentId,
    ExperimentVariantId,
    FalsePositiveBudget,
    PathSettings,
    RelativePath,
    WeightMode,
)


class OutputDirectory(StrEnum):
    LOGS = "logs"
    CHECKPOINTS = "checkpoints"
    SCALERS = "scalers"
    SCORES = "scores"
    RUNS = "runs"
    FIGURES = "figures"
    ANALYSIS = "analysis"
    ALLOCATIONS = "allocations"
    EVENT_ANALYSIS = "event_analysis"
    TABLES = "tables"


class ResultsDirectory(StrEnum):
    METRICS = "metrics"
    STATISTICS = "statistics"
    TABLES = "tables"
    FIGURES = "figures"
    EXPERIMENTS = "experiments"


class ArtifactFileName(StrEnum):
    PROVENANCE = "provenance.json"
    FEATURES = "features.parquet"
    FEATURE_MANIFEST = "feature_manifest.json"
    FEATURE_MANIFEST_DIGEST = "feature_manifest.sha256"
    SPLIT_MANIFEST = "splits.json"
    MODEL = "model.safetensors"
    SCALER_SUFFIX = ".safetensors"
    METADATA = "metadata.json"
    TRAINING_DIGEST = "training.sha256"
    EVALUATION = "evaluation.json"
    RESULTS = "results.parquet"
    EVENT_METRICS = "event_metrics.parquet"
    EVENT_ROBUSTNESS = "event_robustness.parquet"
    EVENT_GINI = "event_gini.parquet"
    FIGURE_ARCHITECTURE = "architecture.png"
    FIGURE_BUDGET_MACRO_RECALL = "budget_macro_recall.png"
    FIGURE_BUDGET_WORST_CLIENT_RECALL = "budget_worst_client_recall.png"
    FIGURE_CLIENT_OPERATING_POINTS = "client_operating_points.png"
    FIGURE_ALLOCATION_STABILITY = "allocation_stability.png"
    FIGURE_CLIENT_UTILITY_CURVES = "client_utility_curves.png"
    FIGURE_GAIN_VS_HETEROGENEITY = "gain_vs_heterogeneity.png"
    FIGURE_REALIZED_FPR_LINES = "realized_fpr_lines.pdf"
    FIGURE_MACRO_ALLOCATION_HEATMAP = "macro_allocation_heatmap.pdf"
    EVENT_BUDGET = "event_budget.parquet"
    PROTOCOL = "protocol.json"
    REPRODUCIBILITY = "reproducibility.json"
    REPORT = "report.json"
    MANIFEST = "manifest.json"
    CONFIGURATION = "configuration.json"
    SUMMARY = "summary.json"
    PROGRESS_STATE = "progress.json"
    CONTRASTS = "contrasts.json"
    PRACTICAL_GATES = "practical_gates.json"


_SEED_DIRNAME_WIDTH = 3
_SEED_DIRNAME_PREFIX = "seed-"


@dataclass(frozen=True, slots=True)
class DetectorCoordinate:
    dataset_id: DatasetId
    detector_seed: DetectorSeed


@dataclass(frozen=True, slots=True)
class ScoreCoordinate:
    dataset_id: DatasetId
    detector_seed: DetectorSeed
    client_id: ClientId


@dataclass(frozen=True, slots=True)
class ExperimentCoordinate:
    experiment_id: ExperimentId
    variant_id: ExperimentVariantId
    dataset_id: DatasetId
    detector_seed: DetectorSeed
    budget_id: BudgetId
    budget: FalsePositiveBudget
    weight_mode: WeightMode


@dataclass(frozen=True, slots=True)
class AllocationCoordinate:
    experiment: ExperimentCoordinate
    policy: AllocationPolicy


def _seed_dirname(detector_seed: DetectorSeed) -> RelativePath:
    return f"{_SEED_DIRNAME_PREFIX}{detector_seed:0{_SEED_DIRNAME_WIDTH}d}"


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    outputs_root: Path
    preprocessed_root: Path
    results_root: Path
    raw_data_root: Path

    @classmethod
    def from_settings(cls, settings: PathSettings) -> ArtifactPaths:
        return cls(
            outputs_root=settings.outputs_root,
            preprocessed_root=settings.preprocessed_root,
            results_root=settings.results_root,
            raw_data_root=settings.raw_data_root,
        )

    def publication(self) -> ResultsPaths:
        return ResultsPaths(self.results_root)

    def logs_dir(self) -> Path:
        return self.outputs_root / OutputDirectory.LOGS

    def checkpoints_dir(self) -> Path:
        return self.outputs_root / OutputDirectory.CHECKPOINTS

    def scores_dir(self) -> Path:
        return self.outputs_root / OutputDirectory.SCORES

    def runs_dir(self) -> Path:
        return self.outputs_root / OutputDirectory.RUNS

    def figures_dir(self) -> Path:
        return self.outputs_root / OutputDirectory.FIGURES

    def campaign_log_path(self) -> Path:
        return self.logs_dir() / "campaign.log"

    def preprocessing_dir(self, dataset_id: DatasetId) -> Path:
        return self.preprocessed_root / dataset_id.value

    def preprocessing_provenance_path(self, dataset_id: DatasetId) -> Path:
        return self.preprocessing_dir(dataset_id) / ArtifactFileName.PROVENANCE

    def preprocessing_features_path(self, dataset_id: DatasetId) -> Path:
        return self.preprocessing_dir(dataset_id) / ArtifactFileName.FEATURES

    def preprocessing_manifest_path(self, dataset_id: DatasetId) -> Path:
        return self.preprocessing_dir(dataset_id) / ArtifactFileName.FEATURE_MANIFEST

    def preprocessing_manifest_digest_path(self, dataset_id: DatasetId) -> Path:
        return self.preprocessing_dir(dataset_id) / ArtifactFileName.FEATURE_MANIFEST_DIGEST

    def preprocessing_split_manifest_path(self, dataset_id: DatasetId) -> Path:
        return self.preprocessing_dir(dataset_id) / ArtifactFileName.SPLIT_MANIFEST

    def checkpoint_dir(self, coordinate: DetectorCoordinate) -> Path:
        return (
            self.checkpoints_dir()
            / coordinate.dataset_id.value
            / _seed_dirname(coordinate.detector_seed)
        )

    def checkpoint_model_path(self, coordinate: DetectorCoordinate) -> Path:
        return self.checkpoint_dir(coordinate) / ArtifactFileName.MODEL

    def checkpoint_metadata_path(self, coordinate: DetectorCoordinate) -> Path:
        return self.checkpoint_dir(coordinate) / ArtifactFileName.METADATA

    def checkpoint_training_digest_path(self, coordinate: DetectorCoordinate) -> Path:
        return self.checkpoint_dir(coordinate) / ArtifactFileName.TRAINING_DIGEST

    def checkpoint_client_scaler_path(
        self, coordinate: DetectorCoordinate, client_id: ClientId
    ) -> Path:
        return (
            self.checkpoint_dir(coordinate)
            / OutputDirectory.SCALERS
            / f"{client_id}{ArtifactFileName.SCALER_SUFFIX}"
        )

    def matched_budget_coordinate(
        self,
        dataset_id: DatasetId,
        seed: DetectorSeed,
        level: BudgetLevel,
    ) -> ExperimentCoordinate:
        return ExperimentCoordinate(
            experiment_id=ExperimentId.MATCHED_BUDGET,
            variant_id=ExperimentVariantId.PRIMARY,
            dataset_id=dataset_id,
            detector_seed=seed,
            budget_id=level.budget_id,
            budget=level.value,
            weight_mode=WeightMode.EQUAL_CLIENT,
        )

    def score_path(self, coordinate: ScoreCoordinate, split: BenignSplit | AttackSplit) -> Path:
        return (
            self.scores_dir()
            / coordinate.dataset_id.value
            / _seed_dirname(coordinate.detector_seed)
            / coordinate.client_id
            / f"{split.value}.parquet"
        )

    def score_provenance_path(self, dataset_id: DatasetId, seed: DetectorSeed) -> Path:
        return (
            self.scores_dir() / dataset_id.value / _seed_dirname(seed) / ArtifactFileName.PROVENANCE
        )

    def experiment_runs_root(
        self,
        experiment_id: ExperimentId,
        variant_id: ExperimentVariantId,
        dataset_id: DatasetId,
    ) -> Path:
        return self.runs_dir() / experiment_id.value / variant_id.value / dataset_id.value

    def run_root(self, coordinate: ExperimentCoordinate) -> Path:
        return (
            self.runs_dir()
            / coordinate.experiment_id.value
            / coordinate.variant_id.value
            / coordinate.dataset_id.value
            / _seed_dirname(coordinate.detector_seed)
            / coordinate.budget_id.value
            / coordinate.weight_mode.value
        )

    def allocation_path(self, coordinate: AllocationCoordinate) -> Path:
        return (
            self.run_root(coordinate.experiment)
            / OutputDirectory.ALLOCATIONS
            / f"{coordinate.policy.value}.json"
        )

    def result_path(self, coordinate: ExperimentCoordinate) -> Path:
        return self.run_root(coordinate) / ArtifactFileName.RESULTS

    def evaluation_summary_path(self, coordinate: ExperimentCoordinate) -> Path:
        return self.run_root(coordinate) / ArtifactFileName.EVALUATION

    def analysis_path(self, coordinate: ExperimentCoordinate, name: AnalysisArtifactId) -> Path:
        return self.run_root(coordinate) / OutputDirectory.ANALYSIS / f"{name.value}.parquet"

    def campaign_analysis_path(self, name: CampaignAnalysisId) -> Path:
        return self.runs_dir() / OutputDirectory.ANALYSIS / f"{name.value}.json"

    def campaign_protocol_path(self) -> Path:
        return self.runs_dir() / ArtifactFileName.PROTOCOL

    def progress_state_path(self) -> Path:
        return self.runs_dir() / ArtifactFileName.PROGRESS_STATE

    def event_analysis_dir(self) -> Path:
        return self.runs_dir() / ExperimentId.EVENT_LEVEL.value

    def event_training_progress_path(self, seed: DetectorSeed) -> Path:
        return (
            self.event_analysis_dir()
            / "progress"
            / _seed_dirname(seed)
            / ArtifactFileName.PROGRESS_STATE
        )

    def campaign_reproducibility_path(self) -> Path:
        return self.runs_dir() / ArtifactFileName.REPRODUCIBILITY

    def report_path(self) -> Path:
        return self.outputs_root / ArtifactFileName.REPORT

    def tables_dir(self) -> Path:
        return self.outputs_root / OutputDirectory.TABLES


@dataclass(frozen=True, slots=True)
class ResultsPaths:
    root: Path

    def bundle_dir(self) -> Path:
        return self.root

    def manifest_path(self) -> Path:
        return self.root / ArtifactFileName.MANIFEST

    def configuration_path(self) -> Path:
        return self.root / ArtifactFileName.CONFIGURATION

    def reproducibility_path(self) -> Path:
        return self.root / ArtifactFileName.REPRODUCIBILITY

    def summary_path(self) -> Path:
        return self.root / ArtifactFileName.SUMMARY

    def metrics_dir(self) -> Path:
        return self.root / ResultsDirectory.METRICS

    def statistics_dir(self) -> Path:
        return self.root / ResultsDirectory.STATISTICS

    def tables_dir(self) -> Path:
        return self.root / ResultsDirectory.TABLES

    def figures_dir(self) -> Path:
        return self.root / ResultsDirectory.FIGURES

    def experiments_dir(self) -> Path:
        return self.root / ResultsDirectory.EXPERIMENTS

    def experiment_dir(self, experiment_id: ExperimentId) -> Path:
        return self.experiments_dir() / experiment_id.value

    def report_json_path(self) -> Path:
        return self.root / ArtifactFileName.REPORT
