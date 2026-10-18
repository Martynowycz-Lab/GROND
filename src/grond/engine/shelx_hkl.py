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


def parse_hkl_observation_line(line: str) -> tuple[int, int, int, float, float]:
    """Read a whitespace-separated or fixed-column observation, ignoring optional trailing columns.

    Full eight-character observation fields can touch each other or the last index. Try a complete free-format
    record first, then fixed columns; float-written integer indices are the final free-format fallback.
    """

    parts = line.split()
    if len(parts) >= 5:
        try:
            return int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4])
        except (ValueError, OverflowError):
            pass
    if len(line.rstrip("\r\n")) >= 28:
        try:
            return int(line[:4]), int(line[4:8]), int(line[8:12]), float(line[12:20]), float(line[20:28])
        except (ValueError, OverflowError):
            pass
    if len(parts) >= 5:
        indices = [float(value) for value in parts[:3]]
        if all(value.is_integer() for value in indices):
            h, k, ell = (int(value) for value in indices)
            return h, k, ell, float(parts[3]), float(parts[4])
    raise ValueError("not a valid free-format or fixed-column HKL observation")


def adaptive_observation_field(value: float) -> str:
    """Most accurate decimal or exponential number that fits one eight-character HKLF field.

    Keep an explicit decimal point, as required to avoid Fortran's implicit decimal placement. This opt-in
    format preserves small observations beside large uncertainties without forcing a common scale reduction.
    """

    value = float(value)
    if not math.isfinite(value):
        raise ValueError("HKLF4 observations must be finite")
    candidates = []
    for digits in range(7):
        fixed = f"{value:.{digits}f}"
        if digits == 0:
            fixed += "."
        if len(fixed) <= 8 and (value == 0.0 or float(fixed) != 0.0):
            candidates.append((abs(float(fixed) - value), 10.0**(-digits), fixed))
        if digits:
            mantissa, exponent = f"{value:.{digits}E}".split("E")
            exponent = int(exponent)
            scientific = f"{mantissa}E{exponent}"
            if len(scientific) <= 8 and math.isfinite(float(scientific)) and (value == 0.0 or float(scientific) != 0.0):
                candidates.append((abs(float(scientific) - value), 10.0**(exponent - digits), scientific))
    if not candidates:
        raise ValueError(f"HKLF4 field overflow for {value!r}")
    return min(candidates)[2].rjust(8)


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
    *,
    adaptive: bool = False,
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
        adaptive_observation_field(intensity_value) if adaptive else f"{intensity_value:8.2f}",
        adaptive_observation_field(sigma_value) if adaptive else f"{sigma_value:8.2f}",
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
    *,
    adaptive: bool = False,
) -> None:
    """Write a validated fixed-width HKLF4 reflection file."""

    with path.open("w") as handle:
        for row in rows:
            handle.write(format_fixed_width_row(*row, adaptive=adaptive))
        handle.write(format_fixed_width_row(0, 0, 0, 0.0, 0.0))


def write_fixed_width_batch_hkl(
    path: Path,
    rows: Iterable[tuple[int, int, int, float, float]],
    batches: Iterable[int],
) -> None:
    """Write an HKLF4 file with a batch number in columns 29-32 (SHELXL scales batch m > 1 by BASF(m-1))."""

    with path.open("w") as handle:
        for row, batch in zip(rows, batches, strict=True):
            value = int(batch)
            if not 1 <= value <= 9999:
                raise ValueError(f"HKLF4 batch number must be 1..9999, got {batch!r}")
            handle.write(format_fixed_width_row(*row).rstrip("\n") + f"{value:4d}\n")
        handle.write(format_fixed_width_row(0, 0, 0, 0.0, 0.0).rstrip("\n") + f"{0:4d}\n")


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


def audit_fixed_width_precision(path: Path) -> dict[str, object]:
    """Check the written observations, not the arbitrary units before export.

    Use each field's actual decimal/exponent precision (0.005 for ordinary F8.2).
    Expressing that bound in sigma units shows whether printed precision matters.
    """
    try:
        lines, record = _fixed_width_hklf4_records(path)
        sigmas = []
        bounds = []
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
            text = body[12:20].decode("ascii").strip().upper()
            mantissa, _, exponent = text.partition("E")
            decimals = len(mantissa.partition(".")[2]) if "." in mantissa else 2
            quantum = 10.0 ** (int(exponent or 0) - decimals)
            bounds.append(0.5 * quantum / sigma)
        if not sigmas:
            raise ValueError("written HKLF4 contains no reflections")
    except (OSError, ValueError) as exc:
        return {"status": "invalid", "path": str(path), "reason": str(exc)}
    bounds.sort()
    position = 0.95 * (len(bounds) - 1)
    lower = int(position)
    p95 = bounds[lower] + (position - lower) * (bounds[min(lower + 1, len(bounds) - 1)] - bounds[lower])
    return {"status": "ok", "path": str(path), "reflection_count": len(sigmas),
            "observation_payload_sha256": record["observation_payload_sha256"],
            "minimum_sigma": min(sigmas), "rounding_bound_sigma_units_p95": p95,
            "rounding_bound_sigma_units_max": bounds[-1],
            "definition": "Half of each written intensity field's last decimal/exponent unit divided by its positive sigma(I); a precision bound, not an uncertainty calibration"}
