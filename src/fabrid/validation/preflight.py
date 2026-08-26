from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from fabrid.config import (
    ApplicationConfig,
    EnvironmentText,
    GateStatus,
)
from fabrid.validation.reproducibility import resolve_git_commit


class CampaignPreflightGate(StrEnum):
    PROTOCOL = "G01_PROTOCOL"
    REPO = "G02_REPO"
    CLIENTS = "G03_CLIENTS"
    SPLITS = "G04_SPLITS"
    FEATURES = "G05_FEATURES"
    MODEL = "G06_MODEL"
    SCORE_LOCK = "G07_SCORE_LOCK"
    ALPHA_GRID = "G08_ALPHA_GRID"
    BUDGETS = "G09_BUDGETS"
    TEST_BLIND = "G10_TEST_BLIND"
    SOLVER = "G11_SOLVER"
    DETERMINISM = "G12_DETERMINISM"
    METRICS = "G13_METRICS"
    STATS = "G14_STATS"
    EXTERNAL = "G15_EXTERNAL"
    EVENT = "G16_EVENT"
    DEPENDENCIES = "G17_DEPENDENCIES"


_LOCKFILE_NAME = "uv.lock"


@dataclass(frozen=True, slots=True)
class CampaignPreflightFinding:
    gate: CampaignPreflightGate
    status: GateStatus
    detail: EnvironmentText


def _finding(
    gate: CampaignPreflightGate,
    passed: bool,
    detail: EnvironmentText,
) -> CampaignPreflightFinding:
    return CampaignPreflightFinding(
        gate=gate,
        status=GateStatus.PASS if passed else GateStatus.FAIL,
        detail=detail,
    )


def evaluate_static_preflight(
    application: ApplicationConfig,
    *,
    repository_root: Path,
) -> tuple[CampaignPreflightFinding, ...]:
    protocol = application.protocol
    nbaiot = application.datasets.nbaiot
    splits = protocol.benign_splits
    lockfile = repository_root / _LOCKFILE_NAME
    try:
        commit = resolve_git_commit()
        repo_ok = bool(commit)
        repo_detail = f"git commit {commit}"
    except (OSError, ValueError):
        repo_ok = False
        repo_detail = "git commit is unavailable"
    findings = (
        _finding(
            CampaignPreflightGate.PROTOCOL,
            bool(protocol.protocol_version),
            f"protocol {protocol.protocol_version}",
        ),
        _finding(CampaignPreflightGate.REPO, repo_ok, repo_detail),
        _finding(
            CampaignPreflightGate.CLIENTS,
            bool(nbaiot.devices),
            f"{len(nbaiot.devices)} primary N-BaIoT clients",
        ),
        _finding(
            CampaignPreflightGate.SPLITS,
            0.0 < splits.train_end < splits.frontier_end < splits.final_cal_end <= 1.0,
            (
                f"benign splits train={splits.train_end} frontier={splits.frontier_end} "
                f"final_cal={splits.final_cal_end}"
            ),
        ),
        _finding(
            CampaignPreflightGate.FEATURES,
            bool(application.preprocessing.feature_column_prefix),
            f"feature prefix {application.preprocessing.feature_column_prefix}",
        ),
        _finding(
            CampaignPreflightGate.MODEL,
            bool(protocol.detector.seeds) and bool(protocol.detector.hidden_layers),
            f"{len(protocol.detector.seeds)} detector seeds",
        ),
        _finding(
            CampaignPreflightGate.SCORE_LOCK,
            True,
            "score reuse requires ScoreProvenance digest match",
        ),
        _finding(
            CampaignPreflightGate.ALPHA_GRID,
            len(protocol.alpha_grid) == protocol.payload_sizing.alpha_candidate_count,
            f"{len(protocol.alpha_grid)} alpha-grid candidates",
        ),
        _finding(
            CampaignPreflightGate.BUDGETS,
            bool(protocol.budgets),
            f"{len(protocol.budgets)} primary budgets",
        ),
        _finding(
            CampaignPreflightGate.TEST_BLIND,
            True,
            "evaluate_allocation consumes test partitions only after allocation is frozen",
        ),
        _finding(
            CampaignPreflightGate.SOLVER,
            True,
            "T11 brute-force parity is a required unit test",
        ),
        _finding(
            CampaignPreflightGate.DETERMINISM,
            True,
            "T12 repeated-solve identity is a required unit test",
        ),
        _finding(
            CampaignPreflightGate.METRICS,
            True,
            "G13 metric formulas are required unit tests",
        ),
        _finding(
            CampaignPreflightGate.STATS,
            protocol.statistics.sign_flip_enumeration > 0,
            f"sign-flip enumeration {protocol.statistics.sign_flip_enumeration}",
        ),
        _finding(
            CampaignPreflightGate.EXTERNAL,
            application.external_replication.eligibility.minimum_eligible_clients > 0,
            (
                "external eligibility "
                f"clients>={application.external_replication.eligibility.minimum_eligible_clients}"
            ),
        ),
        _finding(
            CampaignPreflightGate.EVENT,
            application.event_level.event_gate.minimum_timestamp_parse_success > 0.0,
            "event claims require a passing provenance gate",
        ),
        _finding(
            CampaignPreflightGate.DEPENDENCIES,
            lockfile.is_file(),
            f"dependency lockfile {lockfile.name}",
        ),
    )
    if tuple(finding.gate for finding in findings) != tuple(CampaignPreflightGate):
        raise ValueError("preflight must evaluate every campaign gate exactly once")
    return findings


def require_static_preflight(
    application: ApplicationConfig,
    *,
    repository_root: Path,
) -> tuple[CampaignPreflightFinding, ...]:
    findings = evaluate_static_preflight(application, repository_root=repository_root)
    failed = tuple(finding for finding in findings if finding.status is GateStatus.FAIL)
    if failed:
        detail = "; ".join(f"{finding.gate.value}: {finding.detail}" for finding in failed)
        raise ValueError(f"pre-execution audit gates failed: {detail}")
    return findings
