from __future__ import annotations

import pytest
from typer.testing import CliRunner

from fabrid.cli import app

runner = CliRunner()
pytestmark = pytest.mark.integration


def test_validate_success() -> None:
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == 0
    assert "valid" in result.output


def test_status_shows_config_derived_expectations() -> None:
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "campaign status: protocol" in result.output
    assert "checkpoints" in result.output
    assert "matched_budget" in result.output


def test_help_lists_execution_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in (
        "validate",
        "preprocess",
        "experiment",
        "finalize",
        "report",
        "results",
        "status",
    ):
        assert command in result.output
    assert "--config" not in result.output
    assert "external-campaign" not in result.output


def test_campaign_command_no_longer_exists() -> None:
    result = runner.invoke(app, ["campaign", "--help"])
    assert result.exit_code != 0
    assert "campaign" not in [
        line.split()[0] for line in result.output.splitlines() if line.strip()
    ]


def test_experiment_help_requires_experiment_id() -> None:
    result = runner.invoke(app, ["experiment", "--help"])
    assert result.exit_code == 0
    assert "experiment_id" in result.output
    assert "matched_budget" in result.output
    assert "--overwrite" in result.output
    assert "--config" not in result.output


def test_experiment_help_has_no_seed_selection() -> None:
    result = runner.invoke(app, ["experiment", "--help"])
    assert result.exit_code == 0
    assert "--seeds" not in result.output


def test_no_production_command_exposes_seed_selection() -> None:
    for command in (
        "validate",
        "preprocess",
        "experiment",
        "finalize",
        "report",
        "results",
        "status",
    ):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
        assert "--seeds" not in result.output, f"{command} exposes --seeds"


def test_report_help_lists_rebuild_from_artifacts() -> None:
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code == 0
    assert "existing artifacts" in result.output
    assert "--config" not in result.output


def test_results_help_lists_publication_bundle() -> None:
    result = runner.invoke(app, ["results", "--help"])
    assert result.exit_code == 0
    assert "publication bundle" in result.output
    assert "--config" not in result.output
