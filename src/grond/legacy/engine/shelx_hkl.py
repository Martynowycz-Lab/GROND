"""Validated SHELX HKLF4 fixed-width formatting helpers.

The two F8.2 observation fields are asymmetric: a positive value may use five
digits before the decimal point, whereas a negative value loses one digit to
its sign. Centralizing this rule prevents single-dataset and combined endpoint
writers from silently using incompatible limits.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from pathlib import Path

HKLF4_POSITIVE_WORKING_LIMIT = 90_000.0
HKLF4_NEGATIVE_ABS_WORKING_LIMIT = 9_000.0


def safe_fixed_width_factor(
    rows: Iterable[tuple[int, int, int, float, float]],
    *,
    positive_limit: float = HKLF4_POSITIVE_WORKING_LIMIT,
    negative_abs_limit: float = HKLF4_NEGATIVE_ABS_WORKING_LIMIT,
) -> float:
    """Return one multiplier that makes every HKLF4 F8.2 field printable.

    The same multiplier is applied to I and sigma(I), preserving I/sigma(I).
    The default limits retain a 10% margin relative to F8.2's extrema.
    """

    max_positive = 0.0
    max_negative_abs = 0.0
    for _h, _k, _l, intensity, sigma in rows:
        for value in (float(intensity), float(sigma)):
            if not math.isfinite(value):
                raise ValueError("HKLF4 observations must be finite")
            if value >= 0.0:
                max_positive = max(max_positive, value)
            else:
                max_negative_abs = max(max_negative_abs, -value)
    factors = [1.0]
    if max_positive > float(positive_limit):
        factors.append(float(positive_limit) / max_positive)
    if max_negative_abs > float(negative_abs_limit):
        factors.append(float(negative_abs_limit) / max_negative_abs)
    return min(factors)


def format_fixed_width_row(
    h: int,
    k: int,
    ell: int,
    intensity: float,
    sigma: float,
) -> str:
    """Format one HKLF4 row and reject non-finite or overflowing fields."""

    intensity_value = float(intensity)
    sigma_value = float(sigma)
    if not math.isfinite(intensity_value) or not math.isfinite(sigma_value):
        raise ValueError("HKLF4 observations must be finite")
    fields = (
        f"{int(h):4d}",
        f"{int(k):4d}",
        f"{int(ell):4d}",
        f"{intensity_value:8.2f}",
        f"{sigma_value:8.2f}",
    )
    for field, width in zip(fields, (4, 4, 4, 8, 8), strict=True):
        if len(field) != width:
            raise ValueError(
                f"HKLF4 field overflow: {field!r} needs {len(field)} "
                f"characters but the format allows {width}"
            )
    return "".join(fields) + "\n"


def write_fixed_width_hkl(
    path: Path,
    rows: Iterable[tuple[int, int, int, float, float]],
) -> None:
    """Write a validated fixed-width HKLF4 reflection file."""

    with path.open("w") as handle:
        for row in rows:
            handle.write(format_fixed_width_row(*row))
        handle.write(format_fixed_width_row(0, 0, 0, 0.0, 0.0))


def _fixed_width_hklf4_records(path: Path) -> tuple[list[bytes], dict[str, object]]:
    """Read strict fixed-width HKLF4 records without touching observation bytes."""

    lines = path.read_bytes().splitlines(keepends=True)
    reflection_count = 0
    terminator_count = 0
    seen_terminator = False
    observation_payload = hashlib.sha256()
    parsed: list[bytes] = []
    for line_number, line in enumerate(lines, start=1):
        body = line.rstrip(b"\r\n")
        if not body:
            parsed.append(line)
            continue
        if len(body) < 28:
            raise ValueError(
                f"HKLF4 row {line_number} is shorter than the required 28 fixed-width columns"
            )
        try:
            h = int(body[0:4])
            k = int(body[4:8])
            ell = int(body[8:12])
            float(body[12:20])
            float(body[20:28])
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"HKLF4 row {line_number} is not valid fixed-width h,k,l,I,sigma data"
            ) from exc
        if (h, k, ell) == (0, 0, 0):
            terminator_count += 1
            seen_terminator = True
        else:
            if seen_terminator:
                raise ValueError(
                    f"HKLF4 reflection row {line_number} occurs after the 0,0,0 terminator"
                )
            reflection_count += 1
            # This includes I, sigma, batch and any producer-specific trailing
            # fields exactly as recorded.
            observation_payload.update(body[12:])
            observation_payload.update(b"\n")
        parsed.append(line)
    if terminator_count != 1:
        raise ValueError(
            f"HKLF4 file must contain exactly one 0,0,0 terminator, found {terminator_count}"
        )
    return parsed, {
        "reflection_count": reflection_count,
        "terminator_count": terminator_count,
        "observation_payload_sha256": observation_payload.hexdigest(),
        "fixed_width_contract": "hkl_4_4_4_observation_tail_byte_preserved_v1",
    }


def audit_fixed_width_hklf4(path: Path) -> dict[str, object]:
    """Return a strict reflection/value-preservation audit for one HKLF4 file."""

    _lines, audit = _fixed_width_hklf4_records(path)
    return audit


def write_sign_paired_hklf4(source: Path, output: Path) -> dict[str, object]:
    """Write the exact gauge partner by negating h,k,l only.

    I, sigma(I), batch numbers, producer-specific trailing fields, line
    endings, and the terminator payload are retained byte-for-byte.
    """

    lines, source_audit = _fixed_width_hklf4_records(source)
    written: list[bytes] = []
    for line_number, line in enumerate(lines, start=1):
        body = line.rstrip(b"\r\n")
        if not body:
            written.append(line)
            continue
        h, k, ell = int(body[0:4]), int(body[4:8]), int(body[8:12])
        if (h, k, ell) == (0, 0, 0):
            written.append(line)
            continue
        index_bytes = f"{-h:4d}{-k:4d}{-ell:4d}".encode("ascii")
        if len(index_bytes) != 12:
            raise ValueError(
                f"HKLF4 index overflow after sign pairing on row {line_number}: "
                f"{(-h, -k, -ell)}"
            )
        written.append(index_bytes + line[12:])
    output.write_bytes(b"".join(written))
    output_audit = audit_fixed_width_hklf4(output)
    if output_audit != source_audit:
        raise ValueError(
            "sign-paired HKLF4 failed the reflection-count/observation-payload preservation audit"
        )
    return {
        "source": source_audit,
        "output": output_audit,
        "reflection_count_preserved": True,
        "observation_payload_preserved": True,
        "index_operation": "h,k,l -> -h,-k,-l",
    }


def audit_fixed_width_precision(path: Path) -> dict[str, object]:
    """Check the written observations, not the arbitrary units before export.

    Each F8.2 field rounds by at most 0.005. Expressing that bound in sigma
    units shows whether the printed precision matters to the observations.
    """
    try:
        lines, record = _fixed_width_hklf4_records(path)
        sigmas = []
        for line in lines:
            body = line.rstrip(b"\r\n")
            if not body or tuple(int(body[i:i + 4]) for i in (0, 4, 8)) == (0, 0, 0):
                continue
            intensity, sigma = float(body[12:20]), float(body[20:28])
            if not math.isfinite(intensity) or not math.isfinite(sigma):
                raise ValueError("written HKLF4 intensity or sigma is nonfinite")
            if sigma <= 0:
                raise ValueError("written HKLF4 sigma is nonpositive; the observation has no valid weight")
            sigmas.append(sigma)
        if not sigmas:
            raise ValueError("written HKLF4 contains no reflections")
    except (OSError, ValueError) as exc:
        return {"status": "invalid", "path": str(path), "reason": str(exc)}
    bounds = sorted(0.005 / sigma for sigma in sigmas)
    position = 0.95 * (len(bounds) - 1)
    lower = int(position)
    p95 = bounds[lower] + (position - lower) * (bounds[min(lower + 1, len(bounds) - 1)] - bounds[lower])
    return {"status": "ok", "path": str(path), "reflection_count": len(sigmas),
            "observation_payload_sha256": record["observation_payload_sha256"],
            "minimum_sigma": min(sigmas), "rounding_bound_sigma_units_p95": p95,
            "rounding_bound_sigma_units_max": bounds[-1],
            "definition": "F8.2 half-step 0.005 divided by each positive written sigma(I); a precision bound, not an uncertainty calibration"}
