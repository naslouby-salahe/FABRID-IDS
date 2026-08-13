from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from fabrid.domain.values import RowCount


class CsvIntegrityStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"


class CsvIntegrityIssueKind(StrEnum):
    ZERO_BYTE_FILE = "zero_byte_file"
    HEADER_ONLY = "header_only"
    FIELD_COUNT_MISMATCH = "field_count_mismatch"
    MISSING_TRAILING_NEWLINE = "missing_trailing_newline"
    ENCODING_ERROR = "encoding_error"
    READ_ERROR = "read_error"


@dataclass(frozen=True, slots=True)
class FieldCount:
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("field count must be non-negative")


@dataclass(frozen=True, slots=True)
class CsvRowNumber:
    value: int

    def __post_init__(self) -> None:
        if self.value < 1:
            raise ValueError("CSV row number must be positive")


@dataclass(frozen=True, slots=True)
class CsvIntegrityIssue:
    path: Path
    kind: CsvIntegrityIssueKind
    row_number: CsvRowNumber | None = None
    expected_fields: FieldCount | None = None
    actual_fields: FieldCount | None = None


@dataclass(frozen=True, slots=True)
class CsvIntegrityReport:
    root: Path
    files_checked: RowCount
    issues: tuple[CsvIntegrityIssue, ...]

    @property
    def status(self) -> CsvIntegrityStatus:
        if self.issues:
            return CsvIntegrityStatus.INVALID
        return CsvIntegrityStatus.VALID


def _validate_csv(path: Path) -> tuple[CsvIntegrityIssue, ...]:
    issues: list[CsvIntegrityIssue] = []
    try:
        if path.stat().st_size == 0:
            return (CsvIntegrityIssue(path, CsvIntegrityIssueKind.ZERO_BYTE_FILE),)

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                return (CsvIntegrityIssue(path, CsvIntegrityIssueKind.ZERO_BYTE_FILE),)
            expected_fields = FieldCount(len(header))
            data_rows = RowCount(0)
            for row_number, row in enumerate(reader, start=2):
                if len(row) != expected_fields.value:
                    issues.append(
                        CsvIntegrityIssue(
                            path=path,
                            kind=CsvIntegrityIssueKind.FIELD_COUNT_MISMATCH,
                            row_number=CsvRowNumber(row_number),
                            expected_fields=expected_fields,
                            actual_fields=FieldCount(len(row)),
                        )
                    )
                    break
                data_rows = RowCount(data_rows.value + 1)
            if data_rows.value == 0 and not issues:
                issues.append(CsvIntegrityIssue(path, CsvIntegrityIssueKind.HEADER_ONLY))

        with path.open("rb") as raw:
            raw.seek(-1, 2)
            if raw.read(1) != b"\n":
                issues.append(
                    CsvIntegrityIssue(
                        path,
                        CsvIntegrityIssueKind.MISSING_TRAILING_NEWLINE,
                    )
                )
    except UnicodeDecodeError:
        issues.append(CsvIntegrityIssue(path, CsvIntegrityIssueKind.ENCODING_ERROR))
    except OSError:
        issues.append(CsvIntegrityIssue(path, CsvIntegrityIssueKind.READ_ERROR))
    return tuple(issues)


def validate_csv_tree(root: Path) -> CsvIntegrityReport:
    if not root.exists() or not root.is_dir():
        raise ValueError("CSV integrity root must be an existing directory")
    files = tuple(sorted(root.rglob("*.csv")))
    issues = tuple(issue for path in files for issue in _validate_csv(path))
    return CsvIntegrityReport(
        root=root,
        files_checked=RowCount(len(files)),
        issues=issues,
    )
