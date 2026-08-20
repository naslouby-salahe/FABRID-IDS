from __future__ import annotations

from pathlib import Path

from fabrid.config import CiciomtDatasetConfig, EventCriterionId
from fabrid.datasets.ciciomt import audit_ciciomt_event_provenance
from fabrid.datasets.registry import EventProvenanceEvidence
from tests.support import production_application


def _layout() -> CiciomtDatasetConfig:
    return production_application().datasets.ciciomt


def test_audit_ciciomt_event_provenance_no_timestamps(tmp_path: Path) -> None:
    layout = _layout()
    filename = layout.csv_glob.replace("*", "scan_train", 1)
    (tmp_path / filename).write_text("f1,f2\n1.0,2.0\n")
    evidence = audit_ciciomt_event_provenance(tmp_path, layout)
    assert isinstance(evidence, EventProvenanceEvidence)
    assert not evidence.passed(EventCriterionId.PACKET_TIMESTAMP)
    assert not evidence.passed(EventCriterionId.INTERVAL_PROVENANCE)
    assert not evidence.passed(EventCriterionId.DETERMINISTIC_SCORE_ASSOCIATION)


def test_audit_ciciomt_event_provenance_empty_tree(tmp_path: Path) -> None:
    evidence = audit_ciciomt_event_provenance(tmp_path, _layout())
    assert not evidence.passed(EventCriterionId.IMMUTABLE_CLIENT_ID)
    assert not evidence.passed(EventCriterionId.DETERMINISTIC_SCORE_ASSOCIATION)
