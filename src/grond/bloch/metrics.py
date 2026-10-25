"""Explicit row-index loading."""

from __future__ import annotations

import csv
from pathlib import Path


def read_row_index_file(path: str | Path, *, row_count: int | None = None) -> list[int]:
    """Read unique zero-based row indices from text/CSV and reject invalid requested indices."""

    with open(path, newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        if "," in sample:
            reader = csv.DictReader(handle)
            field = (
                "row_index"
                if reader.fieldnames and "row_index" in reader.fieldnames
                else (reader.fieldnames[0] if reader.fieldnames else None)
            )
            if field is None:
                raise ValueError(f"No columns found in {path}")
            values = [int(row[field]) for row in reader]
        else:
            values = []
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                token = line.split()[0]
                if token.lower() in {"row_index", "index"}:
                    continue
                try:
                    values.append(int(token))
                except ValueError as exc:
                    raise ValueError(f"invalid row index {token!r} at {path}:{line_number}") from exc
    invalid = sorted({value for value in values if value < 0 or (row_count is not None and value >= row_count)})
    if invalid:
        allowed = "nonnegative" if row_count is None else f"in [0, {row_count})"
        raise ValueError(f"row indices in {path} must be {allowed}; invalid indices: {invalid[:8]}")
    return sorted(set(values))
