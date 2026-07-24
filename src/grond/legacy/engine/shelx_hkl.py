"""Validated SHELX HKLF4 fixed-width formatting helpers.

The two F8.2 observation fields are asymmetric: a positive value may use five
digits before the decimal point, whereas a negative value loses one digit to
its sign. Centralizing this rule prevents single-dataset and combined endpoint
writers from silently using incompatible limits.
"""

from __future__ import annotations

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
