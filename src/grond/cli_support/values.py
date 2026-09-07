"""Small value conversions shared by commands, reports and workflows."""

from __future__ import annotations

import json
import math
from pathlib import Path

FULL_Q_LAMBDA = "1.0"


DEFAULT_ELECTRON_WAVELENGTH_A = 0.025079


DEFAULT_MAX_SUPPORT_SG = 0.01


XDS_TARGET_MODES = ("iobs", "iobs_over_rlp", "iobs_times_peak", "recorded_proxy")


XDS_TARGET_MODE_LABELS = {
    "iobs": "IOBS",
    "iobs_over_rlp": "IOBS/RLP",
    "iobs_times_peak": "IOBS*PEAK/100",
    "recorded_proxy": "IOBS*PEAK/(100*RLP)",
}


def parse_numeric_grid(text: str) -> list[float]:
    """Parse a comma/space separated numeric grid while preserving order."""

    values: list[float] = []
    seen: set[float] = set()
    for token in str(text or "").replace(",", " ").replace(";", " ").split():
        try:
            value = float(token)
        except ValueError:
            continue
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values


def parse_int_grid(text: str) -> list[int]:
    """Parse a comma/space separated integer grid while preserving order."""

    values: list[int] = []
    seen: set[int] = set()
    for token in str(text or "").replace(",", " ").replace(";", " ").split():
        try:
            value = int(float(token))
        except ValueError:
            continue
        if value not in seen:
            values.append(value)
            seen.add(value)
    return values


def parse_name_grid(text: str) -> list[str]:
    """Parse a comma/space separated label grid while preserving order."""

    values: list[str] = []
    seen: set[str] = set()
    for token in str(text or "").replace(",", " ").replace(";", " ").split():
        value = token.strip().lower()
        if value and value not in seen:
            values.append(value)
            seen.add(value)
    return values


def format_numeric_grid(values: list[float]) -> str:
    """Render a float list as a GROND engine grid string."""

    return ",".join(f"{float(v):.8g}" for v in values)


def centered_grid(center: float, *, window: float, step: float, lower: float, upper: float) -> list[float]:
    """Return a bounded local grid that always contains the center value."""

    if not math.isfinite(center):
        return []
    step = abs(float(step))
    window = abs(float(window))
    if step <= 0.0:
        step = max(window, 1.0)
    lo = max(float(lower), float(center) - window)
    hi = min(float(upper), float(center) + window)
    values = [float(center)]
    n_steps = int(math.ceil(max(0.0, hi - lo) / step))
    for idx in range(n_steps + 1):
        value = lo + idx * step
        if value <= hi + 1e-9:
            values.append(value)
    values = sorted({round(float(v), 8) for v in values if lower - 1e-9 <= v <= upper + 1e-9})
    return values


def read_json(path: Path) -> dict:
    """Read a JSON file, returning an empty dict when it does not exist."""

    return json.loads(path.read_text()) if path.exists() else {}


def try_read_json(path: Path) -> dict:
    """Read JSON without letting a missing or malformed file stop reporting."""

    try:
        return read_json(path)
    except Exception:
        return {}


def _json_contract_equal(left: object, right: object) -> bool:
    """Compare replay contracts deterministically, including NaN sentinels."""

    return json.dumps(
        left,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=True,
    ) == json.dumps(
        right,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=True,
    )
