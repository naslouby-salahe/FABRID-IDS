from __future__ import annotations

import csv
import logging
import shutil
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, model_validator

from fabrid.artifacts.json import digest_file, digest_text, read_typed_json, write_typed_json
from fabrid.artifacts.paths import (
    ArtifactFileName,
    ArtifactPaths,
    ResultsDirectory,
    ResultsPaths,
)
from fabrid.config import (
    AllocationPolicy,
    AnalysisArtifactId,
    ArtifactDigest,
    ArtifactName,
    BalancedAccuracy,
    BudgetId,
    BudgetUsageRatio,
    CampaignAnalysisId,
    ColumnName,
    DetectorSeed,
    EnvironmentText,
    EvidenceAvailability,
    ExperimentId,
    F1Score,
    FabridConfig,
    FailureReason,
    FalsePositiveRate,
    FigureId,
    GitCommit,
    MacroRecall,
    MetricDifference,
    Probability,
    ProtocolVersion,
    RelativePath,
    ReportTableId,
    RowCount,
    SolverRuntimeMilliseconds,
    WorstClientRecall,
)
from fabrid.evaluation.gates import (
    PracticalGateAnalysis,
    UnavailableFabridMacroGate,
    UnavailableFabridMinimaxGate,
)
from fabrid.evaluation.inference import (
    BudgetContrast,
    HolmResult,
    HypothesisDecision,
    PrimaryInference,
)
from fabrid.evaluation.metrics import (
    CompletedPolicyEvaluation,
    MetricId,
    SeedBudgetEvaluation,
    mean,
)
from fabrid.validation.reproducibility import ReproducibilityMetadata, now_iso

logger = logging.getLogger(__name__)


class ArtifactKind(StrEnum):
    PNG = "png"


_BUNDLE_VERSION = "1.0"


class ReportTable(BaseModel):
    model_config = ConfigDict(frozen=True)
    table_id: ArtifactName
    title: EnvironmentText
    columns: tuple[ColumnName, ...]
    rows: tuple[tuple[EnvironmentText, ...], ...]

    @model_validator(mode="after")
    def _validate(self) -> ReportTable:
        if not self.columns:
            raise ValueError(f"table {self.table_id} requires at least one column")
        expected = len(self.columns)
        if any(len(row) != expected for row in self.rows):
            raise ValueError(f"table {self.table_id} rows must match the column count")
        return self


class BudgetPolicySummary(BaseModel):
    model_config = ConfigDict(frozen=True)
    budget_id: BudgetId
    policy: AllocationPolicy
    macro_recall: MacroRecall
    worst_client_recall: WorstClientRecall
    federation_fpr: FalsePositiveRate
    budget_usage: BudgetUsageRatio | None
    fallback_rate: Probability
    solver_runtime_ms: SolverRuntimeMilliseconds | None
    pooled_recall: Probability
    macro_f1: F1Score
    balanced_accuracy: BalancedAccuracy
    auroc: Probability
    auprc: Probability
    false_alert_gini: Probability
    budget_violation: BudgetUsageRatio | None


class ContrastSummary(BaseModel):
    model_config = ConfigDict(frozen=True)
    budget_id: BudgetId
    treatment: AllocationPolicy
    baseline: AllocationPolicy
    metric: MetricId
    mean_difference: MetricDifference
    median_difference: MetricDifference
    ci_low: MetricDifference
    ci_high: MetricDifference
    p_value: Probability
    holm_adjusted_p_value: Probability
    decision: HypothesisDecision
    included_seeds: tuple[DetectorSeed, ...]


class FigureEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    figure_id: FigureId
    title: EnvironmentText
    path: RelativePath
    kind: ArtifactKind


class ReportEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)
    generated_at: EnvironmentText
    summary: tuple[BudgetPolicySummary, ...]
    statistics: tuple[ContrastSummary, ...]
    gates: PracticalGateAnalysis
    tables: tuple[ReportTable, ...]
    figures: tuple[FigureEntry, ...]


class BundleKind(StrEnum):
    JSON = "json"
    PARQUET = "parquet"
    PNG = "png"
    CSV = "csv"


class ManifestEntry(BaseModel):
    model_config = ConfigDict(frozen=True)
    path: RelativePath
    digest: ArtifactDigest
    kind: BundleKind
    owning_experiment: ExperimentId | None


class KindCount(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: BundleKind
    count: RowCount


class ArtifactCounts(BaseModel):
    model_config = ConfigDict(frozen=True)
    files: RowCount
    by_kind: tuple[KindCount, ...]
    tables: RowCount
    figures: RowCount
    contrasts: RowCount
    budget_policy_rows: RowCount


class ResultsManifest(BaseModel):
    model_config = ConfigDict(frozen=True)
    bundle_version: ProtocolVersion
    generated_at: EnvironmentText
    protocol_version: ProtocolVersion
    git_commit: GitCommit
    evidence_digest: ArtifactDigest
    entries: tuple[ManifestEntry, ...]
    counts: ArtifactCounts


def load_evaluations(
    paths: ArtifactPaths,
    config: FabridConfig,
) -> tuple[SeedBudgetEvaluation, ...]:
    adapter = TypeAdapter(SeedBudgetEvaluation)
    dataset_id = config.dataset_id
    evaluations: list[SeedBudgetEvaluation] = []
    for seed in config.detector.seeds:
        for level in config.budgets:
            coordinate = paths.matched_budget_coordinate(dataset_id, seed, level)
            path = paths.evaluation_summary_path(coordinate)
            if path.is_file():
                evaluations.append(read_typed_json(path, adapter))
    return tuple(evaluations)


@dataclass(frozen=True, slots=True)
class LiteratureBoundaryEntry:
    strand: EnvironmentText
    distinction: EnvironmentText


def _budget_policy_summaries(
    evaluations: tuple[SeedBudgetEvaluation, ...],
    config: FabridConfig,
) -> tuple[BudgetPolicySummary, ...]:
    completed = tuple(
        (result.experiment.budget_id, policy_result)
        for result in evaluations
        for policy_result in result.policies
        if isinstance(policy_result, CompletedPolicyEvaluation)
        and policy_result.policy is not AllocationPolicy.TEST_ORACLE
    )
    keys = dict.fromkeys((budget_id, item.policy) for budget_id, item in completed)
    budget_order = {level.budget_id: index for index, level in enumerate(config.budgets)}
    policy_order = {policy: index for index, policy in enumerate(AllocationPolicy)}
    ordered_keys = sorted(
        keys,
        key=lambda key: (budget_order.get(key[0], len(budget_order)), policy_order[key[1]]),
    )
    summaries: list[BudgetPolicySummary] = []
    for budget_id, policy in ordered_keys:
        values = tuple(
            item
            for item_budget, item in completed
            if item_budget is budget_id and item.policy is policy
        )
        usages = tuple(value.budget_usage for value in values if value.budget_usage is not None)
        runtimes = tuple(
            value.solver_runtime_ms for value in values if value.solver_runtime_ms is not None
        )
        violations = tuple(
            value.budget_violation for value in values if value.budget_violation is not None
        )
        summaries.append(
            BudgetPolicySummary(
                budget_id=budget_id,
                policy=policy,
                macro_recall=mean(tuple(value.macro_recall for value in values)),
                worst_client_recall=mean(tuple(value.worst_client_recall for value in values)),
                federation_fpr=mean(tuple(value.federation_fpr for value in values)),
                budget_usage=mean(usages) if usages else None,
                fallback_rate=mean(tuple(value.fallback_rate for value in values)),
                solver_runtime_ms=mean(runtimes) if runtimes else None,
                pooled_recall=mean(tuple(value.pooled_recall for value in values)),
                macro_f1=mean(tuple(value.macro_f1 for value in values)),
                balanced_accuracy=mean(tuple(value.balanced_accuracy for value in values)),
                auroc=mean(tuple(value.auroc for value in values)),
                auprc=mean(tuple(value.auprc for value in values)),
                false_alert_gini=mean(tuple(value.false_alert_gini for value in values)),
                budget_violation=mean(violations) if violations else None,
            )
        )
    return tuple(summaries)


def _contrast_summary(contrast_item: BudgetContrast, holm: HolmResult) -> ContrastSummary:
    contrast = contrast_item.contrast
    return ContrastSummary(
        budget_id=contrast_item.budget_id,
        treatment=contrast.treatment,
        baseline=contrast.baseline,
        metric=contrast.metric,
        mean_difference=contrast.bootstrap.mean_difference,
        median_difference=contrast.bootstrap.median_difference,
        ci_low=contrast.bootstrap.confidence_interval_low,
        ci_high=contrast.bootstrap.confidence_interval_high,
        p_value=contrast.sign_flip.p_value,
        holm_adjusted_p_value=holm.adjusted_p_value,
        decision=holm.decision,
        included_seeds=contrast.included_seeds,
    )


def _contrast_summaries(
    primary: PrimaryInference,
    config: FabridConfig,
) -> tuple[ContrastSummary, ...]:
    macro_holm = primary.holm_macro(config.statistics.significance)
    worst_holm = primary.holm_worst_client(config.statistics.significance)
    macro_items = tuple(
        _contrast_summary(item, holm)
        for item, holm in zip(primary.macro_recall, macro_holm, strict=True)
    )
    worst_items = tuple(
        _contrast_summary(item, holm)
        for item, holm in zip(primary.worst_client_recall, worst_holm, strict=True)
    )
    return macro_items + worst_items


def format_ratio(ratio: Probability) -> EnvironmentText:
    return f"{ratio:.4f}"


def _format_usage(usage_ratio: BudgetUsageRatio) -> EnvironmentText:
    return f"{usage_ratio:.3f}"


def _format_runtime(runtime_ms: SolverRuntimeMilliseconds) -> EnvironmentText:
    return f"{runtime_ms:.1f}"


def _budget_policy_summary_table(
    summary: tuple[BudgetPolicySummary, ...], config: FabridConfig
) -> ReportTable:
    budget_value = {level.budget_id: level.value for level in config.budgets}
    rows = tuple(
        (
            f"{budget_value[entry.budget_id]:.4f}",
            entry.policy.value,
            format_ratio(entry.macro_recall),
            format_ratio(entry.worst_client_recall),
            _format_usage(entry.budget_usage) if entry.budget_usage is not None else "\u2014",
            format_ratio(entry.federation_fpr),
        )
        for entry in summary
    )
    return ReportTable(
        table_id=ReportTableId.BUDGET_POLICY_SUMMARY,
        title="Matched-budget primary results, mean across detector seeds",
        columns=(
            "budget",
            "policy",
            "macro recall",
            "worst-client recall",
            "budget usage",
            "federation fpr",
        ),
        rows=rows,
    )


def _secondary_metrics_table(
    summary: tuple[BudgetPolicySummary, ...], config: FabridConfig
) -> ReportTable:
    budget_value = {level.budget_id: level.value for level in config.budgets}
    rows = tuple(
        (
            f"{budget_value[entry.budget_id]:.4f}",
            entry.policy.value,
            format_ratio(entry.pooled_recall),
            format_ratio(entry.macro_f1),
            format_ratio(entry.balanced_accuracy),
            format_ratio(entry.auroc),
            format_ratio(entry.auprc),
            format_ratio(entry.false_alert_gini),
            _format_usage(entry.budget_violation)
            if entry.budget_violation is not None
            else "\u2014",
        )
        for entry in summary
    )
    return ReportTable(
        table_id="secondary_metrics",
        title=(
            "Secondary metrics and frozen-detector ranking, mean across detector seeds "
            "(AUROC/AUPRC are detector properties, not allocation endpoints)"
        ),
        columns=(
            "budget",
            "policy",
            "pooled recall",
            "macro f1",
            "balanced accuracy",
            "auroc",
            "auprc",
            "false-alert gini",
            "budget violation",
        ),
        rows=rows,
    )


def _generalization_rotations_table(config: FabridConfig) -> ReportTable:
    rotation_rows = tuple(
        (
            rotation.variant_id.value,
            rotation.validation_fold.value,
            ", ".join(fold.value for fold in rotation.test_folds),
        )
        for rotation in config.generalization.rotations
    )
    family_rows = tuple(
        (
            direction.variant_id.value,
            direction.validation_family.value,
            direction.test_family.value,
        )
        for direction in config.generalization.family_directions
    )
    return ReportTable(
        table_id=ReportTableId.GENERALIZATION_ROTATIONS,
        title=(
            "Generalization protocol: subtype-fold and botnet-family rotations (protocol structure)"
        ),
        columns=("variant", "validation scope", "test scope"),
        rows=rotation_rows + family_rows,
    )


_LITERATURE_BOUNDARY_ENTRIES: tuple[LiteratureBoundaryEntry, ...] = (
    LiteratureBoundaryEntry(
        "heterogeneous alert-rate regulation",
        "FABRID adds a federation-level resource formulation over per-client operating rates "
        "rather than per-client threshold tuning alone.",
    ),
    LiteratureBoundaryEntry(
        "constrained threshold optimization",
        "FABRID maximizes detection utility subject to a global false-alert budget instead of "
        "applying a threshold-selection heuristic.",
    ),
    LiteratureBoundaryEntry(
        "federated anomaly threshold calculation",
        "FABRID separates allocation from calibration: final thresholds are calibrated per "
        "client from frozen final-calibration data, never from allocation inputs.",
    ),
    LiteratureBoundaryEntry(
        "distributed collaborative IoT threshold finding",
        "FABRID shares only utility curves and weights; raw traffic and per-record scores stay "
        "local.",
    ),
    LiteratureBoundaryEntry(
        "federated finite-sample calibration",
        "FABRID uses order-statistic thresholds with an explicit finite-sample guarantee "
        "boundary; no infinite-sample coverage claims are made.",
    ),
    LiteratureBoundaryEntry(
        "alert-budget-aware intrusion detection",
        "FABRID formulates the alert budget at federation level with per-client budgets from a "
        "validated budget contract.",
    ),
    LiteratureBoundaryEntry(
        "event-level alarm-budget evaluation",
        "FABRID evaluates analyst workload on eventized alerts only after an explicit event "
        "data gate passes.",
    ),
    LiteratureBoundaryEntry(
        "cross-device IDS resource allocation",
        "FABRID targets per-client operating rates under a shared false-alert budget with "
        "explicit worst-client protection.",
    ),
)


def _literature_boundary_table() -> ReportTable:
    return ReportTable(
        table_id=ReportTableId.LITERATURE_BOUNDARY,
        title="Novelty boundary against the audited related-work strands",
        columns=("related-work strand", "FABRID boundary statement"),
        rows=tuple((entry.strand, entry.distinction) for entry in _LITERATURE_BOUNDARY_ENTRIES),
    )


def _dataset_populations_table(
    paths: ArtifactPaths,
    config: FabridConfig,
) -> ReportTable:
    from pydantic import TypeAdapter

    from fabrid.artifacts.json import read_typed_json
    from fabrid.validation.datasets import SplitManifest

    manifest_path = paths.preprocessing_split_manifest_path(config.dataset_id)
    if not manifest_path.exists():
        return ReportTable(
            table_id=ReportTableId.DATASET_POPULATIONS,
            title=f"Dataset populations ({config.dataset_id.value})",
            columns=("client", "benign rows", "attack rows per subtype"),
            rows=(),
        )
    manifest = read_typed_json(manifest_path, TypeAdapter(SplitManifest))
    rows = tuple(
        (
            client.client_id,
            str(client.benign_total_rows),
            "; ".join(f"{attack.subtype}:{attack.validation_end}" for attack in client.attacks),
        )
        for client in manifest.clients
    )
    return ReportTable(
        table_id=ReportTableId.DATASET_POPULATIONS,
        title=f"Dataset populations ({config.dataset_id.value})",
        columns=("client", "benign rows", "attack rows per subtype"),
        rows=rows,
    )


def _load_external_evaluations(
    paths: ArtifactPaths,
) -> tuple[SeedBudgetEvaluation, ...]:
    root = paths.runs_dir() / ExperimentId.EXTERNAL_REPLICATION.value
    if not root.is_dir():
        return ()
    adapter = TypeAdapter(SeedBudgetEvaluation)
    return tuple(
        read_typed_json(path, adapter)
        for path in sorted(root.rglob(ArtifactFileName.EVALUATION))
        if path.is_file()
    )


def _external_replication_table(
    paths: ArtifactPaths,
) -> ReportTable | None:
    evaluations = _load_external_evaluations(paths)
    completed = tuple(
        (result.experiment.budget_id, result.experiment.budget, policy_result)
        for result in evaluations
        for policy_result in result.policies
        if isinstance(policy_result, CompletedPolicyEvaluation)
        and policy_result.policy is AllocationPolicy.FABRID_MACRO
    )
    if not completed:
        return None
    budget_keys = tuple(dict.fromkeys((budget_id, budget) for budget_id, budget, _ in completed))
    rows = tuple(
        (
            f"{budget:.4f}",
            format_ratio(
                mean(
                    tuple(
                        policy_result.macro_recall
                        for item_budget, _budget, policy_result in completed
                        if item_budget is budget_id
                    )
                )
            ),
            format_ratio(
                mean(
                    tuple(
                        policy_result.worst_client_recall
                        for item_budget, _budget, policy_result in completed
                        if item_budget is budget_id
                    )
                )
            ),
            format_ratio(
                mean(
                    tuple(
                        policy_result.federation_fpr
                        for item_budget, _budget, policy_result in completed
                        if item_budget is budget_id
                    )
                )
            ),
        )
        for budget_id, budget in budget_keys
    )
    dataset_id = evaluations[0].experiment.dataset_id
    return ReportTable(
        table_id=ReportTableId.EXTERNAL_REPLICATION,
        title=f"External physical-device replication ({dataset_id.value})",
        columns=("budget", "macro recall", "worst-client recall", "federation fpr"),
        rows=rows,
    )


def _system_overhead_table(summary: tuple[BudgetPolicySummary, ...]) -> ReportTable:
    rows = tuple(
        (
            entry.budget_id.value,
            entry.policy.value,
            _format_runtime(entry.solver_runtime_ms)
            if entry.solver_runtime_ms is not None
            else "\u2014",
            format_ratio(entry.fallback_rate),
        )
        for entry in summary
    )
    return ReportTable(
        table_id=ReportTableId.SYSTEM_OVERHEAD,
        title="System overhead, mean across detector seeds: solver runtime and fallback rate",
        columns=("budget", "policy", "solver runtime (ms)", "fallback rate"),
        rows=rows,
    )


def _build_tables(
    config: FabridConfig,
    summary: tuple[BudgetPolicySummary, ...],
    paths: ArtifactPaths,
) -> tuple[ReportTable, ...]:
    prefix = (
        _budget_policy_summary_table(summary, config),
        _secondary_metrics_table(summary, config),
        _generalization_rotations_table(config),
        _dataset_populations_table(paths, config),
    )
    external = _external_replication_table(paths)
    suffix = (_system_overhead_table(summary), _literature_boundary_table())
    if external is None:
        return prefix + suffix
    return prefix + (external,) + suffix


def build_report_evidence(
    paths: ArtifactPaths,
    config: FabridConfig,
    evaluations: tuple[SeedBudgetEvaluation, ...],
    primary: PrimaryInference | None,
    gates: PracticalGateAnalysis,
) -> ReportEvidence:
    summary = _budget_policy_summaries(evaluations, config)
    statistics = _contrast_summaries(primary, config) if primary is not None else ()
    tables = _build_tables(config, summary, paths)
    evidence = ReportEvidence(
        generated_at=now_iso(),
        summary=summary,
        statistics=statistics,
        gates=gates,
        tables=tables,
        figures=(),
    )
    from fabrid.reporting.figures import render_figures

    figure_dir = paths.figures_dir()
    figure_entries = render_figures(evidence, config, figure_dir, paths=paths)
    return evidence.model_copy(update={"figures": figure_entries})


def _absent_practical_gates(reason: FailureReason) -> PracticalGateAnalysis:
    return PracticalGateAnalysis(
        fabrid_macro=UnavailableFabridMacroGate(
            availability=EvidenceAvailability.INSUFFICIENT_EVIDENCE,
            reason=reason,
        ),
        fabrid_minimax=UnavailableFabridMinimaxGate(
            availability=EvidenceAvailability.INSUFFICIENT_EVIDENCE,
            reason=reason,
        ),
        budget_compliance=(),
    )


def _load_primary_inference(
    paths: ArtifactPaths,
) -> PrimaryInference | None:
    path = paths.campaign_analysis_path(CampaignAnalysisId.PRIMARY_INFERENCE)
    if not path.is_file():
        return None
    return read_typed_json(path, TypeAdapter(PrimaryInference))


def _load_practical_gates(
    paths: ArtifactPaths,
) -> PracticalGateAnalysis:
    path = paths.campaign_analysis_path(CampaignAnalysisId.PRACTICAL_GATES)
    if not path.is_file():
        return _absent_practical_gates(
            "matched-budget evaluations are absent; practical gates were not written"
        )
    return read_typed_json(path, TypeAdapter(PracticalGateAnalysis))


def _write_table_csv(table: ReportTable, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(table.columns)
        writer.writerows(table.rows)


def build_campaign_report(
    paths: ArtifactPaths,
    config: FabridConfig,
) -> tuple[ReportEvidence, Path]:
    evaluations = load_evaluations(paths, config)
    evidence = build_report_evidence(
        paths,
        config,
        evaluations,
        _load_primary_inference(paths),
        _load_practical_gates(paths),
    )
    report_path = paths.report_path()
    write_typed_json(evidence, TypeAdapter(ReportEvidence), report_path)
    tables_dir = paths.tables_dir()
    for table in evidence.tables:
        write_typed_json(table, TypeAdapter(ReportTable), tables_dir / f"{table.table_id}.json")
        _write_table_csv(table, tables_dir / f"{table.table_id}.csv")
    return evidence, report_path


def _kind_for(path: Path) -> BundleKind:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return BundleKind.JSON
    if suffix == ".parquet":
        return BundleKind.PARQUET
    if suffix == ".png":
        return BundleKind.PNG
    if suffix == ".csv":
        return BundleKind.CSV
    raise ValueError(f"unsupported bundle file kind for {path}")


def _copy_figure(source: Path, destination: Path) -> Path | None:
    if not source.is_file():
        logger.warning("figure artifact %s is missing; skipping bundle copy", source)
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def _write_bundle_evidence(
    config: FabridConfig,
    results_paths: ResultsPaths,
    reproducibility: ReproducibilityMetadata,
    evidence: ReportEvidence,
) -> None:
    write_typed_json(config, TypeAdapter(FabridConfig), results_paths.configuration_path())
    write_typed_json(
        reproducibility,
        TypeAdapter(ReproducibilityMetadata),
        results_paths.reproducibility_path(),
    )
    write_typed_json(
        evidence.summary,
        TypeAdapter(tuple[BudgetPolicySummary, ...]),
        results_paths.summary_path(),
    )
    for entry in evidence.summary:
        write_typed_json(
            entry,
            TypeAdapter(BudgetPolicySummary),
            results_paths.metrics_dir() / f"{entry.budget_id.value}_{entry.policy.value}.json",
        )
    write_typed_json(
        evidence.statistics,
        TypeAdapter(tuple[ContrastSummary, ...]),
        results_paths.statistics_dir() / ArtifactFileName.CONTRASTS,
    )
    write_typed_json(
        evidence.gates,
        TypeAdapter(PracticalGateAnalysis),
        results_paths.statistics_dir() / ArtifactFileName.PRACTICAL_GATES,
    )
    for table in evidence.tables:
        table_path = results_paths.tables_dir() / f"{table.table_id}.json"
        write_typed_json(table, TypeAdapter(ReportTable), table_path)
        _write_table_csv(table, table_path.with_suffix(".csv"))


def _copy_bundle_figures(
    paths: ArtifactPaths,
    results_paths: ResultsPaths,
    evidence: ReportEvidence,
) -> tuple[RelativePath, ...]:
    copied: list[RelativePath] = []
    for figure in evidence.figures:
        source = paths.figures_dir() / figure.path
        destination = results_paths.figures_dir() / figure.path
        if _copy_figure(source, destination) is not None:
            copied.append(figure.path)
    return tuple(copied)


def _copy_campaign_run_artifacts(
    paths: ArtifactPaths,
    results_paths: ResultsPaths,
) -> None:
    source_root = paths.runs_dir()
    destination_root = results_paths.experiments_dir()
    for experiment_id in ExperimentId:
        source = source_root / experiment_id.value
        if source.is_dir():
            shutil.copytree(source, destination_root / experiment_id.value, dirs_exist_ok=True)


def _parquet_csv_rows(paths: tuple[Path, ...]) -> pl.DataFrame | None:
    frames = [pl.read_parquet(path) for path in paths]
    if not frames:
        return None
    return pl.concat(frames) if len(frames) > 1 else frames[0]


def _budget_cell_csv_frame(
    paths: ArtifactPaths,
    experiment_id: ExperimentId,
) -> pl.DataFrame | None:
    rows: list[
        tuple[
            DetectorSeed,
            EnvironmentText,
            EnvironmentText,
            MacroRecall,
            WorstClientRecall,
            FalsePositiveRate,
            BudgetUsageRatio | None,
            Probability,
        ]
    ] = []
    adapter = TypeAdapter(SeedBudgetEvaluation)
    for path in sorted((paths.runs_dir() / experiment_id.value).rglob(ArtifactFileName.EVALUATION)):
        evaluation = read_typed_json(path, adapter)
        for policy in evaluation.policies:
            if not isinstance(policy, CompletedPolicyEvaluation):
                continue
            rows.append(
                (
                    evaluation.experiment.detector_seed,
                    evaluation.experiment.budget_id.value,
                    policy.policy.value,
                    policy.macro_recall,
                    policy.worst_client_recall,
                    policy.federation_fpr,
                    policy.budget_usage,
                    policy.fallback_rate,
                )
            )
    if not rows:
        return None
    return pl.DataFrame(
        rows,
        schema={
            "seed": pl.Int64,
            "budget": pl.String,
            "policy": pl.String,
            "macro_recall": pl.Float64,
            "worst_client_recall": pl.Float64,
            "federation_fpr": pl.Float64,
            "budget_usage": pl.Float64,
            "fallback_rate": pl.Float64,
        },
        orient="row",
    )


def _experiment_summary_csv(
    paths: ArtifactPaths,
    results_paths: ResultsPaths,
    experiment_id: ExperimentId,
) -> None:
    root = paths.runs_dir() / experiment_id.value
    if not root.is_dir():
        return
    if experiment_id is ExperimentId.EVENT_LEVEL:
        frame = _parquet_csv_rows((root / ArtifactFileName.EVENT_METRICS,))
    elif experiment_id is ExperimentId.ALLOCATION_STABILITY:
        frame = _parquet_csv_rows(
            tuple(sorted(root.rglob(f"{AnalysisArtifactId.STABILITY_SUMMARY.value}.parquet")))
        )
    elif experiment_id is ExperimentId.UTILITY_HETEROGENEITY:
        frame = _parquet_csv_rows(
            tuple(sorted(root.rglob(f"{AnalysisArtifactId.UTILITY_HETEROGENEITY.value}.parquet")))
        )
    else:
        frame = _budget_cell_csv_frame(paths, experiment_id)
    if frame is None or frame.height == 0:
        return
    destination = results_paths.experiment_dir(experiment_id) / "summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.write_csv(destination)


def _write_bundle_experiment_csvs(paths: ArtifactPaths, results_paths: ResultsPaths) -> None:
    for experiment_id in ExperimentId:
        _experiment_summary_csv(paths, results_paths, experiment_id)


def _write_bundle_manifest(
    config: FabridConfig,
    results_paths: ResultsPaths,
    reproducibility: ReproducibilityMetadata,
    evidence: ReportEvidence,
    copied_figures: tuple[RelativePath, ...],
    bundle: Path,
) -> None:
    entries: list[ManifestEntry] = []
    experiments_prefix = f"{ResultsDirectory.EXPERIMENTS}/"
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path == results_paths.manifest_path():
            continue
        relative = path.relative_to(bundle).as_posix()
        owning_experiment = None
        if relative.startswith(experiments_prefix):
            owner = relative[len(experiments_prefix) :].split("/", 1)[0]
            owning_experiment = ExperimentId(owner)
        entries.append(
            ManifestEntry(
                path=relative,
                digest=digest_file(path),
                kind=_kind_for(path),
                owning_experiment=owning_experiment,
            )
        )
    by_kind = tuple(
        KindCount(kind=kind, count=sum(1 for entry in entries if entry.kind is kind))
        for kind in BundleKind
        if any(entry.kind is kind for entry in entries)
    )
    write_typed_json(
        ResultsManifest(
            bundle_version=_BUNDLE_VERSION,
            generated_at=now_iso(),
            protocol_version=config.protocol_version,
            git_commit=reproducibility.git_commit,
            evidence_digest=digest_text((evidence.model_dump_json(),)),
            entries=tuple(entries),
            counts=ArtifactCounts(
                files=len(entries),
                by_kind=by_kind,
                tables=len(evidence.tables),
                figures=len(copied_figures),
                contrasts=len(evidence.statistics),
                budget_policy_rows=len(evidence.summary),
            ),
        ),
        TypeAdapter(ResultsManifest),
        results_paths.manifest_path(),
    )


def package_results(
    config: FabridConfig,
    paths: ArtifactPaths,
    reproducibility: ReproducibilityMetadata,
    evidence: ReportEvidence,
) -> Path:
    results_paths = paths.publication()
    bundle = results_paths.bundle_dir()
    if bundle.exists():
        logger.info("replacing existing results bundle at %s", bundle)
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True, exist_ok=True)
    _write_bundle_evidence(config, results_paths, reproducibility, evidence)
    copied_figures = _copy_bundle_figures(paths, results_paths, evidence)
    _copy_campaign_run_artifacts(paths, results_paths)
    _write_bundle_experiment_csvs(paths, results_paths)
    write_typed_json(evidence, TypeAdapter(ReportEvidence), results_paths.report_json_path())
    _write_bundle_manifest(config, results_paths, reproducibility, evidence, copied_figures, bundle)
    return bundle


def _load_or_capture_reproducibility(
    paths: ArtifactPaths,
    config: FabridConfig,
) -> ReproducibilityMetadata:
    stored = paths.campaign_reproducibility_path()
    if stored.is_file():
        return read_typed_json(stored, TypeAdapter(ReproducibilityMetadata))
    from fabrid.validation.reproducibility import capture_reproducibility_metadata

    reproducibility = capture_reproducibility_metadata(
        config.seeds,
        (),
        command_line=" ".join(sys.argv),
        started_at=now_iso(),
        finished_at=now_iso(),
    )
    write_typed_json(reproducibility, TypeAdapter(ReproducibilityMetadata), stored)
    return reproducibility


def verify_results_bundle(
    paths: ArtifactPaths,
    expected_evidence_digest: ArtifactDigest | None = None,
) -> tuple[bool, tuple[EnvironmentText, ...]]:
    manifest_path = paths.publication().manifest_path()
    if not manifest_path.is_file():
        return False, ("results manifest.json missing",)
    try:
        manifest = read_typed_json(manifest_path, TypeAdapter(ResultsManifest))
    except (ValidationError, ValueError, OSError):
        return False, ("results manifest.json unparseable",)
    issues: list[EnvironmentText] = []
    if (
        expected_evidence_digest is not None
        and manifest.evidence_digest != expected_evidence_digest
    ):
        issues.append("manifest evidence digest does not match current evidence")
    for entry in manifest.entries:
        path = manifest_path.parent / entry.path
        if not path.is_file():
            issues.append(f"bundle entry missing: {entry.path}")
            continue
        try:
            actual = digest_file(path)
        except OSError:
            issues.append(f"bundle entry unreadable: {entry.path}")
            continue
        if actual != entry.digest:
            issues.append(f"bundle entry digest mismatch: {entry.path}")
    return not issues, tuple(issues)


def rebuild_results_bundle(
    paths: ArtifactPaths,
    config: FabridConfig,
    evidence: ReportEvidence,
) -> Path:
    reproducibility = _load_or_capture_reproducibility(paths, config)
    return package_results(config, paths, reproducibility, evidence)
