from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch
from pydantic import Field, TypeAdapter

from fabrid.artifacts.json import read_typed_json
from fabrid.artifacts.parquet import read_parquet_models
from fabrid.artifacts.paths import (
    ArtifactFileName,
    ArtifactPaths,
    OutputDirectory,
)
from fabrid.config import (
    AllocationPolicy,
    AnalysisArtifactId,
    BudgetId,
    BudgetLevel,
    ClientId,
    DetectorSeed,
    EnvironmentText,
    FabridConfig,
    FalsePositiveBudget,
    FigureId,
    Probability,
    TargetFalsePositiveRate,
)
from fabrid.evaluation.metrics import (
    ClientOperatingPoint,
    ClientResultRecord,
    CompletedPolicyEvaluation,
    MetricId,
    SeedBudgetEvaluation,
    StabilityReplicate,
    UtilityCurveRow,
    UtilityHeterogeneity,
    mean,
)
from fabrid.reporting.build import ArtifactKind, BudgetPolicySummary, FigureEntry, ReportEvidence

plt.switch_backend("Agg")
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _BudgetMetricPoint:
    budget: FalsePositiveBudget
    value: Probability


@dataclass(frozen=True, slots=True)
class _PolicyMetricCurve:
    policy: AllocationPolicy
    points: tuple[_BudgetMetricPoint, ...]


_STANDARD_FIGURE_WIDTH = 7
_WIDE_FIGURE_WIDTH = 8
_FIGURE_HEIGHT = 4.5
_FIGURE_DPI = 150
_SCHEMATIC_WIDTH = 10
_SCHEMATIC_HEIGHT = 4.2
_SCHEMATIC_ROW_Y = 2.6
_SCHEMATIC_LOWER_ROW_Y = 0.9
_SCHEMATIC_BOX_X = (0.3, 2.7, 5.1, 7.5)
_SCHEMATIC_ARROW_X_GAP = 0.7
_SCHEMATIC_ARROW_SPAN = 2.7
_SCHEMATIC_ARROW_Y_OFFSET = 0.55
_SCHEMATIC_DROP_ARROW_X = 8.5
_SCHEMATIC_ROW_GAP = 1.1
_BOX_HEIGHT = 1.1
_BOX_TITLE_FONT_SIZE = 9
_BOX_SUBTITLE_FONT_SIZE = 6.5
_LEGEND_FONT_SIZE = 8
_ARROW_LINE_WIDTH = 1.2
_GRID_ALPHA = 0.3


def _operating_budget(config: FabridConfig) -> BudgetLevel:
    return config.budgets[len(config.budgets) // 2]


AxisCoordinate = Annotated[float, Field(allow_inf_nan=False)]
_OPERATING_POLICY = AllocationPolicy.FABRID_MACRO
_POLICY_LABELS = {policy: policy.value.upper().replace("_", "-") for policy in AllocationPolicy}


def _policy_label(policy: AllocationPolicy) -> EnvironmentText:
    return _POLICY_LABELS[policy]


def _budget_value(config: FabridConfig, budget_id: BudgetId) -> FalsePositiveBudget | None:
    for level in config.budgets:
        if level.budget_id is budget_id:
            return level.value
    return None


def _entry(figure_id: FigureId, title: EnvironmentText, path: Path) -> FigureEntry:
    return FigureEntry(figure_id=figure_id, title=title, path=path.name, kind=ArtifactKind.PNG)


def _draw_box(
    axis: Axes,
    x: AxisCoordinate,
    y: AxisCoordinate,
    title: EnvironmentText,
    subtitle: EnvironmentText | None = None,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        2.0,
        _BOX_HEIGHT,
        boxstyle="round,pad=0.03",
        linewidth=1.0,
        edgecolor="0.15",
        facecolor="0.96",
    )
    axis.add_patch(patch)
    axis.text(
        x + 1.0,
        y + 0.72,
        title,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
    )
    if subtitle is not None:
        axis.text(
            x + 1.0,
            y + 0.3,
            subtitle,
            ha="center",
            va="center",
            fontsize=_BOX_SUBTITLE_FONT_SIZE,
            color="0.35",
        )


def _draw_arrow(
    axis: Axes,
    start: tuple[AxisCoordinate, AxisCoordinate],
    end: tuple[AxisCoordinate, AxisCoordinate],
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": "0.25", "lw": _ARROW_LINE_WIDTH},
    )


def _render_architecture_schematic(output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(_SCHEMATIC_WIDTH, _SCHEMATIC_HEIGHT))
    axis.set_xlim(0.0, float(_SCHEMATIC_WIDTH))
    axis.set_ylim(0.0, _SCHEMATIC_HEIGHT)
    axis.axis("off")
    row_y = _SCHEMATIC_ROW_Y
    boxes = (
        ("Frozen detector", "federated autoencoder per seed"),
        ("Local scores", "train / frontier / calibration / test"),
        ("Frontier + validation", "utility estimation u_k(alpha)"),
        ("Server allocation", "EQ-FPR, GREEDY, FABRID-MACRO, FABRID-MINIMAX"),
    )
    x_positions = _SCHEMATIC_BOX_X
    for (title, subtitle), x in zip(boxes, x_positions, strict=True):
        _draw_box(axis, x, row_y, title, subtitle)
        if x > 0.3:
            _draw_arrow(
                axis,
                (x - _SCHEMATIC_ARROW_X_GAP, row_y + _SCHEMATIC_ARROW_Y_OFFSET),
                (x, row_y + _SCHEMATIC_ARROW_Y_OFFSET),
            )
    lower_y = _SCHEMATIC_LOWER_ROW_Y
    lower_boxes = (
        ("Alerts", "score > threshold"),
        ("Thresholds", "per client, final calibration"),
        ("Final calibration", "independent held-out partition"),
        ("Target rates alpha_k", "per client"),
    )
    for (title, subtitle), x in zip(lower_boxes, x_positions, strict=True):
        _draw_box(axis, x, lower_y, title, subtitle)
        if x < _SCHEMATIC_BOX_X[-1]:
            _draw_arrow(
                axis,
                (x + _SCHEMATIC_ARROW_SPAN, lower_y + _SCHEMATIC_ARROW_Y_OFFSET),
                (x + 2.0, lower_y + _SCHEMATIC_ARROW_Y_OFFSET),
            )
    _draw_arrow(
        axis,
        (_SCHEMATIC_DROP_ARROW_X, row_y + _SCHEMATIC_ROW_GAP),
        (_SCHEMATIC_DROP_ARROW_X, lower_y + _SCHEMATIC_ROW_GAP),
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=_FIGURE_DPI)
    plt.close(figure)


def _metric_value(entry: BudgetPolicySummary, metric: MetricId) -> Probability:
    if metric is MetricId.MACRO_RECALL:
        return entry.macro_recall
    if metric is MetricId.WORST_CLIENT_RECALL:
        return entry.worst_client_recall
    raise ValueError(f"metric {metric.value} is not renderable as a budget curve")


def _policy_metric_points(
    evidence: ReportEvidence,
    config: FabridConfig,
    policy: AllocationPolicy,
    metric: MetricId,
) -> tuple[_BudgetMetricPoint, ...]:
    points: list[_BudgetMetricPoint] = []
    for entry in evidence.summary:
        if entry.policy is not policy:
            continue
        budget_value = _budget_value(config, entry.budget_id)
        if budget_value is None:
            continue
        points.append(_BudgetMetricPoint(budget=budget_value, value=_metric_value(entry, metric)))
    return tuple(points)


def _render_curves(
    evidence: ReportEvidence,
    config: FabridConfig,
    output_path: Path,
    metric: MetricId,
    y_label: EnvironmentText,
) -> bool:
    policies = tuple(dict.fromkeys(entry.policy for entry in evidence.summary))
    curves = tuple(
        _PolicyMetricCurve(
            policy=policy,
            points=_policy_metric_points(evidence, config, policy, metric),
        )
        for policy in policies
    )
    curves = tuple(curve for curve in curves if curve.points)
    if not curves:
        logger.warning("no policy budgets available for %s; skipping figure", metric.value)
        return False
    figure, axis = plt.subplots(figsize=(_STANDARD_FIGURE_WIDTH, _FIGURE_HEIGHT))
    for curve in curves:
        ordered = tuple(sorted(curve.points, key=lambda point: point.budget))
        axis.plot(
            [point.budget for point in ordered],
            [point.value for point in ordered],
            marker="o",
            label=_policy_label(curve.policy),
        )
    axis.set_xscale("log")
    axis.set_xlabel("Record-level false-positive budget")
    axis.set_ylabel(y_label)
    axis.legend(fontsize=_LEGEND_FONT_SIZE)
    axis.grid(True, which="both", alpha=_GRID_ALPHA)
    figure.tight_layout()
    figure.savefig(output_path, dpi=_FIGURE_DPI)
    plt.close(figure)
    return True


def _client_operating_points(
    paths: ArtifactPaths,
    config: FabridConfig,
) -> tuple[ClientOperatingPoint, ...] | None:
    level = _operating_budget(config)
    sources: list[Path] = []
    for seed in config.detector.seeds:
        coordinate = paths.matched_budget_coordinate(config.dataset_id, seed, level)
        source = paths.result_path(coordinate)
        if source.is_file():
            sources.append(source)
    if not sources:
        logger.warning(
            "no result parquet for %s; skipping per-client operating-point figure",
            level.budget_id.value,
        )
        return None
    collected: list[ClientOperatingPoint] = []
    for source in sources:
        records = tuple(
            record
            for record in read_parquet_models(source, ClientResultRecord)
            if record.policy is _OPERATING_POLICY
        )
        clients = tuple(dict.fromkeys(record.client_id for record in records))
        for client_id in clients:
            record = next(item for item in records if item.client_id == client_id)
            collected.append(
                ClientOperatingPoint(
                    client_id=client_id,
                    policy=_OPERATING_POLICY,
                    target_rate=record.alpha_selected,
                    threshold=record.threshold,
                    false_positive_rate=record.fpr,
                    macro_attack_recall=record.macro_attack_recall,
                )
            )
    if not collected:
        logger.warning(
            "no %s records found; skipping per-client operating-point figure",
            _OPERATING_POLICY.value,
        )
        return None
    points: list[ClientOperatingPoint] = []
    clients = tuple(sorted({point.client_id for point in collected}))
    for client_id in clients:
        entries = tuple(point for point in collected if point.client_id == client_id)
        points.append(
            ClientOperatingPoint(
                client_id=client_id,
                policy=_OPERATING_POLICY,
                target_rate=mean(tuple(point.target_rate for point in entries)),
                threshold=mean(tuple(point.threshold for point in entries)),
                false_positive_rate=mean(tuple(point.false_positive_rate for point in entries)),
                macro_attack_recall=mean(tuple(point.macro_attack_recall for point in entries)),
            )
        )
    return tuple(points)


def _render_operating_points(
    points: tuple[ClientOperatingPoint, ...],
    budget_value: FalsePositiveBudget,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(_STANDARD_FIGURE_WIDTH, _FIGURE_HEIGHT))
    scatter = axis.scatter(
        [point.target_rate for point in points],
        [point.macro_attack_recall for point in points],
        c=[point.false_positive_rate for point in points],
        cmap="viridis",
        s=40,
    )
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label("Client false-positive rate")
    axis.set_xlabel("Allocated target rate alpha_k")
    axis.set_ylabel("Client macro recall (mean across seeds)")
    axis.set_title(
        f"Per-client operating points at budget {budget_value:.4f} "
        f"({_policy_label(_OPERATING_POLICY)})"
    )
    axis.grid(True, alpha=_GRID_ALPHA)
    figure.tight_layout()
    figure.savefig(output_path, dpi=_FIGURE_DPI)
    plt.close(figure)


def _allocation_stability_alpha(
    paths: ArtifactPaths,
) -> tuple[tuple[ClientId, tuple[TargetFalsePositiveRate, ...]], ...] | None:
    candidates = sorted(
        paths.runs_dir().rglob(
            f"*/{OutputDirectory.ANALYSIS}/{AnalysisArtifactId.STABILITY_REPLICATES.value}.parquet"
        )
    )
    if not candidates:
        logger.warning("no allocation-stability parquet found; skipping alpha-distribution figure")
        return None
    replicates = tuple(
        replicate
        for candidate in candidates
        for replicate in read_parquet_models(candidate, StabilityReplicate)
    )
    if not replicates:
        logger.warning("allocation-stability parquet is empty; skipping figure")
        return None
    clients = tuple(sorted({replicate.client_id for replicate in replicates}))
    return tuple(
        (
            client_id,
            tuple(
                replicate.alpha_selected
                for replicate in replicates
                if replicate.client_id == client_id
            ),
        )
        for client_id in clients
    )


def _render_allocation_stability(
    per_client: tuple[tuple[ClientId, tuple[TargetFalsePositiveRate, ...]], ...],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(_WIDE_FIGURE_WIDTH, _FIGURE_HEIGHT))
    axis.boxplot(
        [values for _, values in per_client],
        vert=True,
    )
    axis.set_xticks(
        range(1, len(per_client) + 1), labels=[client_id for client_id, _ in per_client]
    )
    axis.set_xlabel("Client")
    axis.set_ylabel("Allocated target rate alpha")
    axis.set_title("Allocation sensitivity: allocated alpha distribution per client")
    axis.tick_params(axis="x", rotation=45)
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=_FIGURE_DPI)
    plt.close(figure)


def _utility_curves(
    paths: ArtifactPaths,
) -> (
    tuple[tuple[ClientId, tuple[TargetFalsePositiveRate, ...], tuple[Probability, ...]], ...] | None
):
    candidates = sorted(
        paths.runs_dir().rglob(
            f"*/{OutputDirectory.ANALYSIS}/{AnalysisArtifactId.CONSERVATIVE_UTILITY_CURVES.value}.parquet"
        )
    )
    if not candidates:
        logger.warning("no utility-curve parquet found; skipping utility-curve figure")
        return None
    rows = tuple(
        row
        for candidate in candidates
        for row in read_parquet_models(candidate, UtilityCurveRow)
        if row.target_rate > 0.0
    )
    if not rows:
        logger.warning("utility-curve parquet is empty; skipping figure")
        return None
    clients = tuple(sorted({row.client_id for row in rows}))
    return tuple(
        (
            client_id,
            tuple(row.target_rate for row in rows if row.client_id == client_id),
            tuple(row.utility for row in rows if row.client_id == client_id),
        )
        for client_id in clients
    )


def _render_utility_curves(
    curves: tuple[
        tuple[ClientId, tuple[TargetFalsePositiveRate, ...], tuple[Probability, ...]], ...
    ],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(_STANDARD_FIGURE_WIDTH, _FIGURE_HEIGHT))
    for client_id, rates, utilities in curves:
        axis.plot(rates, utilities, label=client_id, alpha=0.85)
    axis.set_xscale("log")
    axis.set_xlabel("Target false-positive rate alpha")
    axis.set_ylabel("Detection utility (subtype-averaged TPR)")
    axis.set_title("Client utility curves u_k(alpha)")
    axis.legend(fontsize=6)
    axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=_FIGURE_DPI)
    plt.close(figure)


def _per_seed_macro_gain(
    paths: ArtifactPaths,
    config: FabridConfig,
) -> tuple[tuple[DetectorSeed, Probability], ...] | None:
    differences: list[tuple[DetectorSeed, Probability]] = []
    for seed in config.detector.seeds:
        for level in config.budgets:
            summary_path = paths.evaluation_summary_path(
                paths.matched_budget_coordinate(config.dataset_id, seed, level)
            )
            if not summary_path.is_file():
                continue
            evaluation = read_typed_json(summary_path, TypeAdapter(SeedBudgetEvaluation))
            fabrid = evaluation.policy(AllocationPolicy.FABRID_MACRO)
            equal_fpr = evaluation.policy(AllocationPolicy.EQ_FPR)
            if not isinstance(fabrid, CompletedPolicyEvaluation):
                continue
            if not isinstance(equal_fpr, CompletedPolicyEvaluation):
                continue
            differences.append((seed, fabrid.macro_recall - equal_fpr.macro_recall))
    if not differences:
        logger.warning("no matched-budget evaluations found; skipping heterogeneity figure")
        return None
    seeds = tuple(sorted({seed for seed, _ in differences}))
    return tuple(
        (seed, mean(tuple(value for item_seed, value in differences if item_seed == seed)))
        for seed in seeds
    )


def _heterogeneity_analysis(
    paths: ArtifactPaths,
    config: FabridConfig,
) -> tuple[tuple[Probability, ...], tuple[Probability, ...]] | None:
    candidates = sorted(
        paths.runs_dir().rglob(
            f"*/{OutputDirectory.ANALYSIS}/{AnalysisArtifactId.UTILITY_HETEROGENEITY.value}.parquet"
        )
    )
    if not candidates:
        logger.warning("no utility-heterogeneity analysis found; skipping heterogeneity figure")
        return None
    aggregates = tuple(
        (row.seed, row.aggregate)
        for candidate in candidates
        for row in read_parquet_models(candidate, UtilityHeterogeneity)
    )
    if not aggregates:
        logger.warning("utility-heterogeneity analysis is empty; skipping figure")
        return None
    seeds = tuple(sorted({seed for seed, _ in aggregates}))
    heterogeneity = tuple(
        (seed, float(np.mean([value for item_seed, value in aggregates if item_seed == seed])))
        for seed in seeds
    )
    gains = _per_seed_macro_gain(paths, config)
    if gains is None:
        return None
    shared_seeds = tuple(
        seed for seed, _ in heterogeneity if any(item_seed == seed for item_seed, _ in gains)
    )
    if not shared_seeds:
        logger.warning("no seeds shared between heterogeneity and gain evidence; skipping figure")
        return None
    gain_values = tuple(
        next(value for item_seed, value in gains if item_seed == seed) for seed in shared_seeds
    )
    heterogeneity_values = tuple(
        next(value for item_seed, value in heterogeneity if item_seed == seed)
        for seed in shared_seeds
    )
    return gain_values, heterogeneity_values


def _render_gain_heterogeneity(
    gain: tuple[Probability, ...],
    heterogeneity: tuple[Probability, ...],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(_STANDARD_FIGURE_WIDTH, _FIGURE_HEIGHT))
    axis.scatter(heterogeneity, gain, s=36)
    axis.set_xlabel("Heterogeneity H_U")
    axis.set_ylabel("FABRID gain (percentage points)")
    axis.set_title("FABRID gain vs heterogeneity")
    axis.grid(True, alpha=_GRID_ALPHA)
    figure.tight_layout()
    figure.savefig(output_path, dpi=_FIGURE_DPI)
    plt.close(figure)


def render_figures(
    evidence: ReportEvidence,
    config: FabridConfig,
    output_dir: Path,
    paths: ArtifactPaths,
) -> tuple[FigureEntry, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[FigureEntry] = []
    architecture_path = output_dir / ArtifactFileName.FIGURE_ARCHITECTURE
    _render_architecture_schematic(architecture_path)
    entries.append(_entry(FigureId.ARCHITECTURE, "FABRID-IDS architecture", architecture_path))
    macro_recall_path = output_dir / ArtifactFileName.FIGURE_BUDGET_MACRO_RECALL
    if _render_curves(
        evidence,
        config,
        macro_recall_path,
        MetricId.MACRO_RECALL,
        "Macro recall (mean across seeds)",
    ):
        entries.append(
            _entry(
                FigureId.BUDGET_MACRO_RECALL,
                "Budget vs macro recall by policy",
                macro_recall_path,
            )
        )
    worst_client_recall_path = output_dir / ArtifactFileName.FIGURE_BUDGET_WORST_CLIENT_RECALL
    if _render_curves(
        evidence,
        config,
        worst_client_recall_path,
        MetricId.WORST_CLIENT_RECALL,
        "Worst-client recall (mean across seeds)",
    ):
        entries.append(
            _entry(
                FigureId.BUDGET_WORST_CLIENT_RECALL,
                "Budget vs worst-client recall by policy",
                worst_client_recall_path,
            )
        )
    operating_points = _client_operating_points(paths, config)
    if operating_points is not None:
        level = _operating_budget(config)
        operating_points_path = output_dir / ArtifactFileName.FIGURE_CLIENT_OPERATING_POINTS
        _render_operating_points(operating_points, level.value, operating_points_path)
        entries.append(
            _entry(
                FigureId.CLIENT_OPERATING_POINTS,
                "Per-client operating points at the matched operating budget",
                operating_points_path,
            )
        )
    stability = _allocation_stability_alpha(paths)
    if stability is not None:
        stability_path = output_dir / ArtifactFileName.FIGURE_ALLOCATION_STABILITY
        _render_allocation_stability(stability, stability_path)
        entries.append(
            _entry(
                FigureId.ALLOCATION_STABILITY,
                "Allocation sensitivity: allocated alpha distribution",
                stability_path,
            )
        )
    curves = _utility_curves(paths)
    if curves is not None:
        utility_curves_path = output_dir / ArtifactFileName.FIGURE_CLIENT_UTILITY_CURVES
        _render_utility_curves(curves, utility_curves_path)
        entries.append(
            _entry(
                FigureId.CLIENT_UTILITY_CURVES,
                "Client utility curves u_k(alpha)",
                utility_curves_path,
            )
        )
    heterogeneity = _heterogeneity_analysis(paths, config)
    if heterogeneity is not None:
        gain, heterogeneity_values = heterogeneity
        gain_heterogeneity_path = output_dir / ArtifactFileName.FIGURE_GAIN_VS_HETEROGENEITY
        _render_gain_heterogeneity(gain, heterogeneity_values, gain_heterogeneity_path)
        entries.append(
            _entry(
                FigureId.GAIN_VS_HETEROGENEITY,
                "FABRID gain vs heterogeneity H_U",
                gain_heterogeneity_path,
            )
        )
    return tuple(entries)
