from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
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
_FIGURE_DPI = 220
_SCHEMATIC_WIDTH = 11.4
_SCHEMATIC_HEIGHT = 4.6
_SCHEMATIC_ROW_Y = 2.85
_SCHEMATIC_LOWER_ROW_Y = 0.9
_SCHEMATIC_BOX_WIDTH = 2.3
_SCHEMATIC_BOX_X = (0.3, 3.06, 5.82, 8.58)
_SCHEMATIC_BOX_SPACING = _SCHEMATIC_BOX_X[1] - _SCHEMATIC_BOX_X[0]
_SCHEMATIC_ARROW_CLEARANCE = 0.08
_SCHEMATIC_ARROW_Y_OFFSET = 0.65
_BOX_HEIGHT = 1.35
_BOX_TITLE_FONT_SIZE = 12
_BOX_SUBTITLE_FONT_SIZE = 9.5
_LEGEND_FONT_SIZE = 12
_AXIS_LABEL_FONT_SIZE = 13
_TICK_LABEL_FONT_SIZE = 11
_CURVE_MARKER_SIZE = 7
_CURVE_LINE_WIDTH = 2.0
_ARROW_LINE_WIDTH = 1.4
_GRID_ALPHA = 0.3
_CURVE_FIGURE_HEIGHT = 7.6
_CURVE_LEGEND_NCOL = 3
_CURVE_LEGEND_ANCHOR_Y = -0.16
_CURVE_LAYOUT_BOTTOM_MARGIN = 0.26
_MANUSCRIPT_POLICIES: tuple[AllocationPolicy, ...] = (
    AllocationPolicy.EQ_FPR,
    AllocationPolicy.FABRID_MACRO,
    AllocationPolicy.FABRID_CVAR,
    AllocationPolicy.FABRID_MACRO_FINITE_SAFE,
    AllocationPolicy.FABRID_CVAR_FINITE_SAFE,
)
_MANUSCRIPT_POLICY_STYLE = {
    AllocationPolicy.EQ_FPR: ("Equal-FPR", "#1f4e79", "o", "-"),
    AllocationPolicy.FABRID_MACRO: ("Macro", "#b05f00", "s", "--"),
    AllocationPolicy.FABRID_CVAR: ("Tail (CVaR)", "#2f7d4a", "^", "-."),
    AllocationPolicy.FABRID_MACRO_FINITE_SAFE: ("Macro, cons.", "#a63d40", "D", ":"),
    AllocationPolicy.FABRID_CVAR_FINITE_SAFE: (
        "Tail, cons.",
        "#6d4c8d",
        "v",
        (0, (4, 1, 1, 1)),
    ),
}
_MANUSCRIPT_CLIENT_LABELS = {
    "Danmini_Doorbell": "Danmini Doorbell",
    "Ecobee_Thermostat": "Ecobee Thermostat",
    "Ennio_Doorbell": "Ennio Doorbell",
    "Philips_B120N10_Baby_Monitor": "Philips B120N10 Baby Monitor",
    "Provision_PT_737E_Security_Camera": "Provision PT-737E Security Camera",
    "Provision_PT_838_Security_Camera": "Provision PT-838 Security Camera",
    "Samsung_SNH_1011_N_Webcam": "Samsung SNH 1011 N Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera": "SimpleHome XCS7 1002 WHT Security Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera": "SimpleHome XCS7 1003 WHT Security Camera",
}


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
    kind = ArtifactKind.PDF if path.suffix.lower() == ".pdf" else ArtifactKind.PNG
    return FigureEntry(figure_id=figure_id, title=title, path=path.name, kind=kind)


def _draw_box(
    axis: Axes,
    x: AxisCoordinate,
    y: AxisCoordinate,
    title: EnvironmentText,
    subtitle: EnvironmentText | None = None,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        _SCHEMATIC_BOX_WIDTH,
        _BOX_HEIGHT,
        boxstyle="round,pad=0.03",
        linewidth=1.0,
        edgecolor="0.15",
        facecolor="0.96",
    )
    axis.add_patch(patch)
    center_x = x + _SCHEMATIC_BOX_WIDTH / 2
    axis.text(
        center_x,
        y + _BOX_HEIGHT * 0.72,
        title,
        ha="center",
        va="center",
        fontsize=_BOX_TITLE_FONT_SIZE,
        fontweight="bold",
    )
    if subtitle is not None:
        axis.text(
            center_x,
            y + _BOX_HEIGHT * 0.30,
            subtitle,
            ha="center",
            va="center",
            fontsize=_BOX_SUBTITLE_FONT_SIZE,
            color="0.35",
            linespacing=1.3,
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
        ("Frozen detector", "federated autoencoder\nper seed"),
        ("Local scores", "train / frontier /\ncalibration / test"),
        ("Utility estimation", "u_k(alpha) on the\nfrontier split"),
        ("Server allocation", "budget-constrained\npolicy selection"),
    )
    x_positions = _SCHEMATIC_BOX_X
    inter_box_gap = _SCHEMATIC_BOX_SPACING - _SCHEMATIC_BOX_WIDTH
    for (title, subtitle), x in zip(boxes, x_positions, strict=True):
        _draw_box(axis, x, row_y, title, subtitle)
        if x > x_positions[0]:
            _draw_arrow(
                axis,
                (x - inter_box_gap + _SCHEMATIC_ARROW_CLEARANCE, row_y + _SCHEMATIC_ARROW_Y_OFFSET),
                (x - _SCHEMATIC_ARROW_CLEARANCE, row_y + _SCHEMATIC_ARROW_Y_OFFSET),
            )
    lower_y = _SCHEMATIC_LOWER_ROW_Y
    lower_boxes = (
        ("Alerts", "score > threshold"),
        ("Thresholds", "per client,\nfinal value"),
        ("Final calibration", "independent\nheld-out partition"),
        ("Target rates", "alpha_k\nper client"),
    )
    for (title, subtitle), x in zip(lower_boxes, x_positions, strict=True):
        _draw_box(axis, x, lower_y, title, subtitle)
        if x < x_positions[-1]:
            _draw_arrow(
                axis,
                (
                    x + _SCHEMATIC_BOX_SPACING - _SCHEMATIC_ARROW_CLEARANCE,
                    lower_y + _SCHEMATIC_ARROW_Y_OFFSET,
                ),
                (
                    x + _SCHEMATIC_BOX_WIDTH + _SCHEMATIC_ARROW_CLEARANCE,
                    lower_y + _SCHEMATIC_ARROW_Y_OFFSET,
                ),
            )
    drop_arrow_x = x_positions[-1] + _SCHEMATIC_BOX_WIDTH / 2
    _draw_arrow(
        axis,
        (drop_arrow_x, row_y - _SCHEMATIC_ARROW_CLEARANCE),
        (drop_arrow_x, lower_y + _BOX_HEIGHT + _SCHEMATIC_ARROW_CLEARANCE),
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=_FIGURE_DPI, bbox_inches="tight", pad_inches=0.15)
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
    figure, axis = plt.subplots(figsize=(_WIDE_FIGURE_WIDTH, _CURVE_FIGURE_HEIGHT))
    for curve in curves:
        ordered = tuple(sorted(curve.points, key=lambda point: point.budget))
        axis.plot(
            [point.budget for point in ordered],
            [point.value for point in ordered],
            marker="o",
            markersize=_CURVE_MARKER_SIZE,
            linewidth=_CURVE_LINE_WIDTH,
            label=_policy_label(curve.policy),
        )
    axis.set_xscale("log")
    axis.set_xlabel("Record-level false-positive budget", fontsize=_AXIS_LABEL_FONT_SIZE)
    axis.set_ylabel(y_label, fontsize=_AXIS_LABEL_FONT_SIZE)
    axis.tick_params(axis="both", labelsize=_TICK_LABEL_FONT_SIZE)
    axis.legend(
        fontsize=_LEGEND_FONT_SIZE,
        loc="upper center",
        bbox_to_anchor=(0.5, _CURVE_LEGEND_ANCHOR_Y),
        ncol=_CURVE_LEGEND_NCOL,
        frameon=True,
    )
    axis.grid(True, which="both", alpha=_GRID_ALPHA)
    figure.tight_layout(rect=(0, _CURVE_LAYOUT_BOTTOM_MARGIN, 1, 1))
    figure.savefig(output_path, dpi=_FIGURE_DPI, bbox_inches="tight", pad_inches=0.15)
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
    figure.savefig(output_path, dpi=_FIGURE_DPI, bbox_inches="tight", pad_inches=0.15)
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
    figure, axis = plt.subplots(figsize=(_WIDE_FIGURE_WIDTH + 4, _FIGURE_HEIGHT + 2))
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
    axis.tick_params(axis="x", rotation=45, labelsize=8)
    for label in axis.get_xticklabels():
        label.set_horizontalalignment("right")
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=_FIGURE_DPI, bbox_inches="tight", pad_inches=0.15)
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
    figure.savefig(output_path, dpi=_FIGURE_DPI, bbox_inches="tight", pad_inches=0.15)
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
    figure.savefig(output_path, dpi=_FIGURE_DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)


def _render_realized_fpr_lines(
    evidence: ReportEvidence,
    config: FabridConfig,
    output_path: Path,
) -> bool:
    """Render the paper's nominal-versus-realized operating-point comparison."""
    budget_values = {level.budget_id: level.value for level in config.budgets}
    summary_by_policy = {
        policy: tuple(
            sorted(
                (
                    entry
                    for entry in evidence.summary
                    if entry.policy is policy and entry.budget_id in budget_values
                ),
                key=lambda entry: budget_values[entry.budget_id],
            )
        )
        for policy in _MANUSCRIPT_POLICIES
    }
    available = tuple(policy for policy in _MANUSCRIPT_POLICIES if summary_by_policy[policy])
    if not available:
        logger.warning("no primary summaries available; skipping realized-FPR manuscript figure")
        return False
    budgets = tuple(sorted({budget_values[entry.budget_id] for entry in evidence.summary}))
    with plt.rc_context(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10.2,
            "ytick.labelsize": 10.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        figure, axes = plt.subplots(
            1,
            2,
            figsize=(6.45, 2.82),
            gridspec_kw={"wspace": 0.34},
        )
        handles: list[Line2D] = []
        for policy in available:
            label, color, marker, linestyle = _MANUSCRIPT_POLICY_STYLE[policy]
            entries = summary_by_policy[policy]
            realized_fpr = [entry.federation_fpr for entry in entries]
            macro_recall = [entry.macro_recall for entry in entries]
            nominal_budget = [budget_values[entry.budget_id] for entry in entries]
            line = cast(
                Line2D,
                axes[0].plot(
                    nominal_budget,
                    realized_fpr,
                    label=label,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    lw=1.55,
                    ms=4.6,
                    mew=0.7,
                )[0],
            )
            axes[1].plot(
                realized_fpr,
                macro_recall,
                color=color,
                marker=marker,
                linestyle=linestyle,
                lw=1.55,
                ms=4.6,
                mew=0.7,
            )
            handles.append(line)
        if budgets:
            axes[0].plot(budgets, budgets, ":", color="0.15", lw=1.1, zorder=0)
        axes[0].annotate(
            "FPR = $B$",
            xy=(0.012, 0.012),
            xytext=(0.0047, 0.018),
            fontsize=9.2,
            color="0.18",
        )
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Nominal federation budget $B$")
        axes[0].set_ylabel("Realized federation FPR")
        axes[0].set_title("(a) Nominal versus realized FPR", loc="left", fontweight="bold")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Realized federation FPR")
        axes[1].set_ylabel("Macro recall")
        axes[1].set_title("(b) Recall at realized FPR", loc="left", fontweight="bold")
        for axis in axes:
            axis.grid(True, which="major", color="0.86", lw=0.55)
            axis.tick_params(which="major", length=3.2, width=0.7)
            axis.tick_params(which="minor", length=1.8, width=0.55)
        figure.legend(
            handles,
            [_MANUSCRIPT_POLICY_STYLE[policy][0] for policy in available],
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.03),
            frameon=False,
            fontsize=10.0,
            handlelength=2.2,
            columnspacing=1.25,
            handletextpad=0.45,
        )
        figure.subplots_adjust(left=0.105, right=0.99, top=0.74, bottom=0.20)
        figure.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(figure)
    return True


def _macro_allocations(
    paths: ArtifactPaths,
    config: FabridConfig,
) -> tuple[tuple[FalsePositiveBudget, ...], tuple[ClientId, ...], np.ndarray] | None:
    """Average one Macro allocation per client, budget, and detector seed."""
    allocations: dict[tuple[FalsePositiveBudget, ClientId], list[TargetFalsePositiveRate]] = {}
    for seed in config.detector.seeds:
        for level in config.budgets:
            source = paths.result_path(
                paths.matched_budget_coordinate(config.dataset_id, seed, level)
            )
            if not source.is_file():
                continue
            records = tuple(
                record
                for record in read_parquet_models(source, ClientResultRecord)
                if record.policy is AllocationPolicy.FABRID_MACRO
            )
            by_client = {record.client_id: record for record in records}
            for client_id, record in by_client.items():
                allocations.setdefault((level.value, client_id), []).append(record.alpha_selected)
    if not allocations:
        logger.warning(
            "no Macro allocation records available; skipping allocation manuscript figure"
        )
        return None
    budgets = tuple(level.value for level in config.budgets)
    clients = tuple(sorted({client_id for _, client_id in allocations}))
    if any((budget, client_id) not in allocations for budget in budgets for client_id in clients):
        logger.warning("incomplete Macro allocation records; skipping allocation manuscript figure")
        return None
    matrix = np.array(
        [
            [mean(tuple(allocations[(budget, client_id)])) for budget in budgets]
            for client_id in clients
        ]
    )
    return budgets, clients, matrix


def _render_macro_allocation_heatmap(
    allocations: tuple[tuple[FalsePositiveBudget, ...], tuple[ClientId, ...], np.ndarray],
    output_path: Path,
) -> None:
    """Render the paper's client-specific Macro nominal-rate allocation heatmap."""
    budgets, clients, matrix = allocations
    labels = [
        _MANUSCRIPT_CLIENT_LABELS.get(client_id, client_id.replace("_", " "))
        for client_id in clients
    ]
    with plt.rc_context({"font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42}):
        figure, axis = plt.subplots(figsize=(6.45, 3.25))
        image = axis.imshow(matrix, aspect="auto", cmap="YlOrBr", interpolation="nearest")
        axis.set_xticks(range(len(budgets)), [f"{budget:.4g}" for budget in budgets])
        axis.set_yticks(range(len(clients)), labels)
        axis.set_xlabel("Nominal federation budget $B$", fontsize=11)
        axis.set_ylabel("Client", fontsize=11)
        axis.tick_params(axis="x", labelsize=10.2, length=0, pad=4)
        axis.tick_params(axis="y", labelsize=10.0, length=0, pad=3)
        for spine in axis.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("0.35")
        colorbar = figure.colorbar(image, ax=axis, fraction=0.048, pad=0.025)
        colorbar.set_label("Mean allocated rate $\\alpha_k$", fontsize=10.4)
        colorbar.ax.tick_params(labelsize=10)
        figure.subplots_adjust(left=0.47, right=0.91, top=0.97, bottom=0.18)
        figure.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
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
    realized_fpr_path = output_dir / ArtifactFileName.FIGURE_REALIZED_FPR_LINES
    if _render_realized_fpr_lines(evidence, config, realized_fpr_path):
        entries.append(
            _entry(
                FigureId.REALIZED_FPR_LINES,
                "Nominal versus realized federation FPR and recall",
                realized_fpr_path,
            )
        )
    macro_allocations = _macro_allocations(paths, config)
    if macro_allocations is not None:
        allocation_heatmap_path = output_dir / ArtifactFileName.FIGURE_MACRO_ALLOCATION_HEATMAP
        _render_macro_allocation_heatmap(macro_allocations, allocation_heatmap_path)
        entries.append(
            _entry(
                FigureId.MACRO_ALLOCATION_HEATMAP,
                "Client-specific Macro nominal-rate allocations",
                allocation_heatmap_path,
            )
        )
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
