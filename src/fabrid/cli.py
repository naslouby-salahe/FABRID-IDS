from __future__ import annotations

from pathlib import Path

import typer

from fabrid.domain.identifiers import CampaignId
from fabrid.pipeline.campaign import run_matched_budget_campaign
from fabrid.pipeline.context import PipelinePaths

app = typer.Typer(
    name="fabrid",
    help="FABRID-IDS reproducible experiment pipeline.",
    no_args_is_help=True,
)


@app.command("matched-budget")
def run_matched_budget(
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
    run_matched_budget_campaign(
        campaign_id=CampaignId(campaign_id),
        paths=PipelinePaths(
            raw_data_root=raw_data_root,
            outputs_root=outputs_root,
        ),
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
