from __future__ import annotations

import logging
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from fabrid.config import ExperimentId, NonNegativeInt
from fabrid.errors import FabridError
from fabrid.execution.application import ApplicationContext, load_application_context
from fabrid.validation.completion import ExperimentVerification

app = typer.Typer(
    help="FABRID-IDS: Federated Alert-Budget Reallocation for Heterogeneous IoT Intrusion Detection"
)
console = Console()


def _exit_error(error: BaseException, code: NonNegativeInt) -> NoReturn:
    console.print(f"[red]{error}[/red]")
    raise typer.Exit(code=code) from error


@app.command(help="Validate the production configuration.")
def validate() -> None:
    try:
        context = load_application_context()
    except (FabridError, ValueError) as error:
        _exit_error(error, 1)
    loaded = context.config
    console.print(
        f"[green]valid[/green] protocol {loaded.protocol_version} "
        f"({len(loaded.protocol.seeds)} seeds, {len(loaded.protocol.budgets)} budgets)"
    )


@app.command(help="Resolve reusable preprocessing (REUSE/BUILD/REBUILD).")
def preprocess(
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Force preprocessing rebuild")
    ] = False,
) -> None:
    from fabrid.execution.prerequisites import prepare_federation

    context = load_application_context()
    prepared = prepare_federation(context, overwrite=overwrite)
    console.print(
        f"[green]preprocessing ready[/green] dataset={prepared.provenance.dataset_id.value} "
        f"features={len(prepared.feature_manifest.columns)}"
    )


@app.command(help="Run one configured experiment through the shared campaign runner.")
def experiment(
    experiment_id: Annotated[ExperimentId, typer.Argument(help="Experiment identity")],
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Delete this experiment's evidence and rebuild it")
    ] = False,
    seed_workers: Annotated[
        int | None,
        typer.Option(
            "--seed-workers",
            min=1,
            help="Override configured GPU/seed worker count",
        ),
    ] = None,
    evaluation_workers: Annotated[
        int | None,
        typer.Option(
            "--evaluation-workers",
            min=1,
            help="Override configured CPU evaluation-worker count",
        ),
    ] = None,
) -> None:
    from fabrid.execution.campaign import run_fabrid_campaign
    from fabrid.validation.completion import (
        ExperimentState,
        delete_experiment_evidence,
        verify_experiment,
    )

    context = load_application_context()
    if not overwrite:
        verification = verify_experiment(context.config, context.paths, experiment_id)
        if verification.state is ExperimentState.PASSED:
            _print_experiment_contract(context, experiment_id, verification)
            console.print(
                f"[green]already_passed[/green] {experiment_id.value} "
                f"({verification.valid_cells}/{verification.expected_cells} cells verified)"
            )
            return
    if overwrite:
        delete_experiment_evidence(context.paths, experiment_id)
    context = context.with_single_experiment(experiment_id)
    result = run_fabrid_campaign(
        context,
        overwrite=False,
        seed_workers=seed_workers,
        evaluation_workers=evaluation_workers,
    )
    verification = verify_experiment(context.config, context.paths, experiment_id)
    _print_experiment_contract(context, experiment_id, verification)
    console.print(
        f"[green]experiment complete[/green] {experiment_id.value} "
        f"status={verification.state.value} report={result.report_path}"
    )


def _print_experiment_contract(
    context: ApplicationContext,
    experiment_id: ExperimentId,
    verification: ExperimentVerification,
) -> None:
    paths = context.paths
    runs_root = paths.runs_dir() / experiment_id.value
    bundle = paths.publication().bundle_dir()
    experiment_csv = bundle / "experiments" / experiment_id.value / "summary.csv"
    console.print(f"experiment={experiment_id.value} status={verification.state.value}")
    console.print(f"  json results: {runs_root}")
    if experiment_csv.is_file():
        console.print(f"  csv results: {experiment_csv}")
    figures = paths.figures_dir()
    if figures.is_dir() and any(figures.iterdir()):
        console.print(f"  figures: {figures}")
    report_path = paths.report_path()
    if report_path.is_file():
        console.print(f"  report: {report_path}")
    if bundle.is_dir():
        console.print(f"  result bundle: {bundle}")
    console.print(f"  execution/output root: {paths.outputs_root}")
    for issue in verification.issues:
        console.print(f"  [red]{issue}[/red]")


@app.command(
    help="Run external replication, event-level, analysis, and reporting from persisted artifacts."
)
def finalize() -> None:
    from fabrid.execution.campaign import run_campaign_finalize

    context = load_application_context()
    result = run_campaign_finalize(context)
    console.print(
        f"[green]finalize complete[/green] "
        f"report={result.report_path} bundle={result.results_bundle_path}"
    )


@app.command(help="Rebuild report.json, tables, and figures from existing artifacts.")
def report() -> None:
    from fabrid.reporting.build import build_campaign_report
    from fabrid.validation.preflight import require_static_preflight

    try:
        context = load_application_context()
        require_static_preflight(context.config, repository_root=context.repository_root)
        _, report_path = build_campaign_report(context.paths, context.config.protocol)
    except (FabridError, ValueError) as error:
        _exit_error(error, 1)
    console.print(f"[green]report ready[/green] {report_path}")


@app.command(
    help=(
        "Build the publication bundle from verified experiment evidence. Refuses to "
        "package evidence that does not verify; reports already_generated when the "
        "existing bundle matches current evidence."
    )
)
def results(
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Delete the generated delivery bundle and rebuild it from verified evidence",
        ),
    ] = False,
) -> None:
    from fabrid.artifacts.json import digest_text
    from fabrid.reporting.build import (
        build_campaign_report,
        rebuild_results_bundle,
        verify_results_bundle,
    )
    from fabrid.validation.completion import (
        ExperimentState,
        delete_results_bundle,
        verify_campaign_evidence,
    )
    from fabrid.validation.preflight import require_static_preflight

    try:
        context = load_application_context()
        require_static_preflight(context.config, repository_root=context.repository_root)
        failing = tuple(
            verification
            for verification in verify_campaign_evidence(context.config, context.paths)
            if verification.state is not ExperimentState.PASSED
        )
        if failing:
            details = " ".join(
                f"{verification.experiment_id.value}={verification.state.value}"
                for verification in failing
            )
            raise FabridError(
                f"refusing to package results: mandatory evidence is missing or invalid: {details}"
            )
        evidence, _ = build_campaign_report(context.paths, context.config.protocol)
        evidence_digest = digest_text((evidence.model_dump_json(),))
        if not overwrite:
            current, _ = verify_results_bundle(context.paths, evidence_digest)
            if current:
                console.print(f"[green]already_generated[/green] {context.paths.results_root}")
                return
        if overwrite:
            delete_results_bundle(context.paths)
        bundle = rebuild_results_bundle(context.paths, context.config.protocol, evidence)
        verified, issues = verify_results_bundle(context.paths, evidence_digest)
        if not verified:
            raise FabridError("results bundle failed verification: " + "; ".join(issues))
        console.print(f"[green]results ready[/green] {bundle}")
    except (FabridError, ValueError) as error:
        _exit_error(error, 1)


@app.command(help="Show verified evidence state for the configured campaign.")
def status() -> None:
    from fabrid.execution.status import ComponentState, inspect_campaign_status
    from fabrid.validation.completion import ExperimentState

    context = load_application_context()
    rows = inspect_campaign_status(context)
    table = Table(title=f"campaign status: protocol {context.config.protocol_version}")
    table.add_column("component")
    table.add_column("state")
    table.add_column("valid")
    table.add_column("expected")
    table.add_column("detail")
    for row in rows:
        style = "green" if row.state in (ComponentState.VERIFIED, ExperimentState.PASSED) else "red"
        table.add_row(
            row.component,
            row.state.value,
            str(row.present_count),
            str(row.expected_count),
            row.detail,
        )
        table.rows[-1].style = style
    console.print(table)
    for row in rows:
        for issue in row.issues:
            console.print(f"[red]{row.component}: {issue}[/red]")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app()


if __name__ == "__main__":
    main()
