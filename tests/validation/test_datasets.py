from __future__ import annotations

from pathlib import Path

from fabrid.domain.values import RowCount
from fabrid.validation.datasets import (
    CsvIntegrityIssueKind,
    CsvIntegrityStatus,
    validate_csv_tree,
)


def test_valid_csv_tree_reports_explicit_valid_status(tmp_path: Path) -> None:
    (tmp_path / "valid.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    report = validate_csv_tree(tmp_path)

    assert report.status is CsvIntegrityStatus.VALID
    assert report.files_checked == RowCount(1)
    assert report.issues == ()


def test_field_mismatch_is_structured_evidence(tmp_path: Path) -> None:
    (tmp_path / "broken.csv").write_text("a,b\n1,2\n3\n", encoding="utf-8")

    report = validate_csv_tree(tmp_path)

    assert report.status is CsvIntegrityStatus.INVALID
    issue = report.issues[0]
    assert issue.kind is CsvIntegrityIssueKind.FIELD_COUNT_MISMATCH
    assert issue.row_number is not None and issue.row_number.value == 3
    assert issue.expected_fields is not None and issue.expected_fields.value == 2
    assert issue.actual_fields is not None and issue.actual_fields.value == 1


def test_header_only_and_missing_trailing_newline_are_reported(tmp_path: Path) -> None:
    (tmp_path / "header.csv").write_text("a,b", encoding="utf-8")

    report = validate_csv_tree(tmp_path)
    kinds = tuple(issue.kind for issue in report.issues)

    assert CsvIntegrityIssueKind.HEADER_ONLY in kinds
    assert CsvIntegrityIssueKind.MISSING_TRAILING_NEWLINE in kinds
