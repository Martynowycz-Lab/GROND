#!/usr/bin/env python3
"""Compare two native row tables with an alternate-hand diagnostic.

This is intentionally row-level. It does not pretend to be a SHELXL Flack
parameter or a merged absolute-structure statistic. It asks whether the same
signed observations have a lower scaled-residual sum for one native Bloch
prediction than for the alternate-hand check after each branch has had its
declared nuisance scales applied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

KEY_COLUMNS = ("row_index", "h", "k", "l")
REQUIRED_COLUMNS = (*KEY_COLUMNS, "dyn_residual_over_sigma")


def finite_float(value: object) -> float:
    """Return a finite float or NaN."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read a native row CSV with DictReader."""

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        return list(reader), fields


def missing_columns(fields: list[str]) -> list[str]:
    """Return required columns absent from a CSV header."""

    present = set(fields)
    return [name for name in REQUIRED_COLUMNS if name not in present]


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    """Return the string key used to identify a signed observation row."""

    return tuple(str(row.get(name, "")).strip() for name in KEY_COLUMNS)


def indexed_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, int], dict[str, str]]:
    """Index rows by key plus duplicate occurrence to avoid many-to-many joins."""

    seen: defaultdict[tuple[str, str, str, str], int] = defaultdict(int)
    out = {}
    for row in rows:
        key = row_key(row)
        occurrence = seen[key]
        seen[key] += 1
        out[(*key, occurrence)] = row
    return out


def duplicate_count(rows: list[dict[str, str]]) -> int:
    """Return the number of rows whose row key is duplicated."""

    counts = Counter(row_key(row) for row in rows)
    return int(sum(count - 1 for count in counts.values() if count > 1))


def joined_rows(best: list[dict[str, str]], other: list[dict[str, str]]) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Return one-to-one joined rows shared by both tables."""

    best_by_key = indexed_rows(best)
    other_by_key = indexed_rows(other)
    keys = sorted(set(best_by_key) & set(other_by_key))
    return [(best_by_key[key], other_by_key[key]) for key in keys]


def block(label: str, rows: list[tuple[dict[str, str], dict[str, str]]]) -> dict[str, Any]:
    """Compute paired row-level contrast statistics."""

    rb: list[float] = []
    ro: list[float] = []
    for best, other in rows:
        b = finite_float(best.get("dyn_residual_over_sigma"))
        o = finite_float(other.get("dyn_residual_over_sigma"))
        if math.isfinite(b) and math.isfinite(o):
            rb.append(b)
            ro.append(o)
    n = len(rb)
    if n == 0:
        return {"label": label, "n": 0}
    d = [o * o - b * b for b, o in zip(rb, ro, strict=True)]
    delta_chi2 = float(sum(d))
    mean_d = float(delta_chi2 / n)
    if n > 1:
        var = sum((x - mean_d) ** 2 for x in d) / (n - 1)
        sd = math.sqrt(max(var, 0.0))
    else:
        sd = float("nan")
    paired_t = float(mean_d / (sd / math.sqrt(n))) if n > 1 and sd > 0 else float("nan")
    return {
        "label": label,
        "n": n,
        "best_chi2": float(sum(x * x for x in rb)),
        "other_chi2": float(sum(x * x for x in ro)),
        "delta_chi2_other_minus_best": delta_chi2,
        "sqrt_positive_delta_chi2": float(math.sqrt(max(delta_chi2, 0.0))),
        "paired_row_t_delta_chi2": paired_t,
        "mean_row_delta_chi2": mean_d,
        "median_row_delta_chi2": float(
            sorted(d)[n // 2] if n % 2 else 0.5 * (sorted(d)[n // 2 - 1] + sorted(d)[n // 2])
        ),
        "best_lower_abs_residual_fraction": float(sum(abs(b) < abs(o) for b, o in zip(rb, ro, strict=True)) / n),
        "best_lower_squared_residual_fraction": float(sum(x > 0.0 for x in d) / n),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--best", required=True, help="Native row CSV for the active/best hand assignment.")
    ap.add_argument("--other", required=True, help="Native row CSV for the alternate-hand comparison.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    args = ap.parse_args()

    best_path = Path(args.best)
    other_path = Path(args.other)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing_files = [str(path) for path in (best_path, other_path) if not path.exists()]
    if missing_files:
        out = {
            "status": "missing_input_files",
            "best_csv": str(best_path.resolve()),
            "other_csv": str(other_path.resolve()),
            "missing_files": missing_files,
            "row_join": {
                "key": list(KEY_COLUMNS),
                "n_best": 0,
                "n_other": 0,
                "n_joined": 0,
                "duplicate_keys_best": 0,
                "duplicate_keys_other": 0,
                "missing_columns": {"best": [], "other": []},
            },
            "interpretation": {
                "diagnostic": "row-level alternate-hand check was skipped because one or more native row CSV files were absent",
                "recommendation": "Treat this as an incomplete diagnostic artifact, not as evidence for either hand.",
            },
            "all": {"label": "all", "n": 0},
            "work": {"label": "work", "n": 0},
            "free": {"label": "free", "n": 0},
        }
        out_path.write_text(json.dumps(out, indent=2, allow_nan=True) + "\n")
        return

    best_rows, best_fields = read_rows(best_path)
    other_rows, other_fields = read_rows(other_path)
    missing = {"best": missing_columns(best_fields), "other": missing_columns(other_fields)}
    if missing["best"] or missing["other"]:
        joined: list[tuple[dict[str, str], dict[str, str]]] = []
        status = "missing_required_columns"
    else:
        joined = joined_rows(best_rows, other_rows)
        status = "ok" if joined else "no_common_rows"

    work = [(best, other) for best, other in joined if best.get("split") == "work"]
    free = [(best, other) for best, other in joined if best.get("split") == "free"]
    out = {
        "status": status,
        "best_csv": str(best_path.resolve()),
        "other_csv": str(other_path.resolve()),
        "row_join": {
            "key": list(KEY_COLUMNS),
            "n_best": int(len(best_rows)),
            "n_other": int(len(other_rows)),
            "n_joined": int(len(joined)),
            "duplicate_keys_best": duplicate_count(best_rows),
            "duplicate_keys_other": duplicate_count(other_rows),
            "missing_columns": missing,
        },
        "interpretation": {
            "diagnostic": "row-level alternate-hand check; not a Flack parameter and not a merged absolute-structure statistic",
            "delta_chi2": "sum((other residual/sigma)^2 - (best residual/sigma)^2); positive favors best",
            "paired_row_t": "paired t-style contrast of per-row delta_chi2 values",
            "sqrt_positive_delta_chi2": "square root of positive delta_chi2; a scale indicator, not a probability or hand-assignment score",
        },
        "all": block("all", joined),
        "work": block("work", work),
        "free": block("free", free),
    }
    out_path.write_text(json.dumps(out, indent=2, allow_nan=True) + "\n")


if __name__ == "__main__":
    main()
