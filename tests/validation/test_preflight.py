from __future__ import annotations

from pathlib import Path

import pytest

from fabrid.config import GateStatus, load_application_config
from fabrid.validation.preflight import (
    CampaignPreflightGate,
    evaluate_static_preflight,
    require_static_preflight,
)


def _production_findings():
    application = load_application_config()
    return evaluate_static_preflight(application, repository_root=Path.cwd())


def _finding(gate: CampaignPreflightGate):
    for finding in _production_findings():
        if finding.gate is gate:
            return finding
    raise AssertionError(f"missing preflight gate {gate}")


def test_g01_protocol_is_frozen() -> None:
    finding = _finding(CampaignPreflightGate.PROTOCOL)
    assert finding.status is GateStatus.PASS
    assert load_application_config().protocol.protocol_version


def test_g02_repo_commit_is_recorded() -> None:
    assert _finding(CampaignPreflightGate.REPO).status is GateStatus.PASS


def test_g03_primary_clients_are_configured() -> None:
    finding = _finding(CampaignPreflightGate.CLIENTS)
    assert finding.status is GateStatus.PASS
    assert load_application_config().datasets.nbaiot.devices


def test_g04_benign_splits_are_exclusive() -> None:
    assert _finding(CampaignPreflightGate.SPLITS).status is GateStatus.PASS


def test_g05_feature_manifest_prefix_is_configured() -> None:
    assert _finding(CampaignPreflightGate.FEATURES).status is GateStatus.PASS


def test_g06_detector_configuration_is_frozen() -> None:
    finding = _finding(CampaignPreflightGate.MODEL)
    assert finding.status is GateStatus.PASS
    assert load_application_config().protocol.detector.seeds


def test_g07_score_lock_is_required() -> None:
    assert _finding(CampaignPreflightGate.SCORE_LOCK).status is GateStatus.PASS


def test_g08_alpha_grid_is_frozen() -> None:
    finding = _finding(CampaignPreflightGate.ALPHA_GRID)
    assert finding.status is GateStatus.PASS
    protocol = load_application_config().protocol
    assert len(protocol.alpha_grid) == protocol.payload_sizing.alpha_candidate_count


def test_g09_primary_budgets_are_frozen() -> None:
    finding = _finding(CampaignPreflightGate.BUDGETS)
    assert finding.status is GateStatus.PASS
    assert load_application_config().protocol.budgets


def test_g10_test_blind_evaluation() -> None:
    assert _finding(CampaignPreflightGate.TEST_BLIND).status is GateStatus.PASS


def test_g11_solver_parity_gate_is_present() -> None:
    assert _finding(CampaignPreflightGate.SOLVER).status is GateStatus.PASS


def test_g12_determinism_gate_is_present() -> None:
    assert _finding(CampaignPreflightGate.DETERMINISM).status is GateStatus.PASS


def test_g13_metrics_gate_is_present() -> None:
    assert _finding(CampaignPreflightGate.METRICS).status is GateStatus.PASS


def test_g14_sign_flip_enumeration_is_configured() -> None:
    finding = _finding(CampaignPreflightGate.STATS)
    assert finding.status is GateStatus.PASS
    assert load_application_config().protocol.statistics.sign_flip_enumeration > 0


def test_g15_external_eligibility_is_frozen() -> None:
    assert _finding(CampaignPreflightGate.EXTERNAL).status is GateStatus.PASS


def test_g16_event_gate_is_configured() -> None:
    assert _finding(CampaignPreflightGate.EVENT).status is GateStatus.PASS


def test_g17_dependency_lockfile_exists() -> None:
    assert _finding(CampaignPreflightGate.DEPENDENCIES).status is GateStatus.PASS
    assert (Path.cwd() / "uv.lock").is_file()


def test_require_static_preflight_accepts_production() -> None:
    require_static_preflight(load_application_config(), repository_root=Path.cwd())


def test_require_static_preflight_rejects_reduced_smoke(tmp_path: Path) -> None:
    from tests.support import smoke_application

    application = smoke_application()
    with pytest.raises(ValueError, match="pre-execution audit gates failed"):
        require_static_preflight(application, repository_root=tmp_path)
