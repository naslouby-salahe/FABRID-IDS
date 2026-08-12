"""Verifies every downloaded CSV is structurally complete: opens and streams every row (not
just a sample), checking each row has the same field count as the header and that the file
doesn't end mid-row or with a truncated/unterminated last line. This is independent of the
crawler's `.partial`-rename download mechanism — it re-checks the bytes actually on disk,
so it also catches corruption from causes other than an interrupted download (disk errors,
manual copies, etc).

Usage: python3 verify_csv_integrity.py <directory>
"""

import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
csv_files = sorted(root.rglob("*.csv"))

print(f"Checking {len(csv_files)} CSV files under {root}...")

bad_files: list[tuple[Path, str]] = []
for path in csv_files:
    try:
        size = path.stat().st_size
        if size == 0:
            bad_files.append((path, "zero-byte file"))
            continue

        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            expected_fields = len(header)
            row_count = 0
            for row_number, row in enumerate(reader, start=2):
                if len(row) != expected_fields:
                    bad_files.append(
                        (
                            path,
                            f"row {row_number} has {len(row)} fields, expected {expected_fields}"
                            " (truncated/corrupt download)",
                        )
                    )
                    break
                row_count += 1
            else:
                if row_count == 0:
                    bad_files.append((path, "header only, zero data rows"))

        # Confirm the file ends with a newline (a download cut off mid-line would not).
        with path.open("rb") as raw:
            raw.seek(-1, 2)
            last_byte = raw.read(1)
            if last_byte != b"\n":
                bad_files.append((path, "file does not end with a newline (likely truncated)"))
    except UnicodeDecodeError as exc:
        bad_files.append((path, f"encoding error: {exc}"))
    except OSError as exc:
        bad_files.append((path, f"read error: {exc}"))

if bad_files:
    print(f"\n{len(bad_files)} PROBLEM FILE(S):")
    for path, reason in bad_files:
        print(f"  {path.relative_to(root)}: {reason}")
    sys.exit(1)

print(f"All {len(csv_files)} CSV files verified structurally complete.")
