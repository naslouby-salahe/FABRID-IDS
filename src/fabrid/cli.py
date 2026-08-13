from __future__ import annotations

from pathlib import Path

import typer

from fabrid.domain.identifiers import CampaignId
from fabrid.pipeline.campaign import run_fabrid_campaign
from fabrid.pipeline.context import PipelinePaths
from fabrid.validation.datasets import CsvIntegrityStatus, validate_csv_tree

app = typer.Typer(
    name="fabrid",
    help="FABRID-IDS reproducible experiment pipeline.",
    no_args_is_help=True,
)


@app.command("campaign")
def run_campaign(
    campaign_id: str = typer.Argument(..., help="Unique campaign identifier."),
    raw_data_root: Path = typer.Option(
        ...,
        "--raw-data-root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    outputs_root: Path = typer.Option(
        ...,
        "--outputs-root",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
) -> None:
    run_fabrid_campaign(
        campaign_id=CampaignId(campaign_id),
        paths=PipelinePaths(
            raw_data_root=raw_data_root,
            outputs_root=outputs_root,
        ),
    )


@app.command("validate-data")
def validate_data(
    root: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    report = validate_csv_tree(root)
    typer.echo(
        f"{report.status.value}: checked {report.files_checked.value} CSV files"
    )
    for issue in report.issues:
        detail = issue.kind.value
        if issue.row_number is not None:
            detail = f"{detail} at row {issue.row_number.value}"
        typer.echo(f"{issue.path}: {detail}")
    if report.status is CsvIntegrityStatus.INVALID:
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
