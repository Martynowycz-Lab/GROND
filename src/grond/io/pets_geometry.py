"""PETS frame rotations, cell conventions and declared acquisition geometry."""

from __future__ import annotations

import math
import re

import gemmi
import numpy as np


def _number(text: str, *, label: str) -> float:
    """Parse a finite PETS number, dropping a trailing esd in parentheses such as ``1.234(5)``."""

    value_text = str(text).strip()
    if "(" in value_text:
        value_text = value_text.split("(", 1)[0]
    try:
        value = float(value_text)
    except ValueError as exc:
        raise ValueError(f"PETS field {label} is not numeric: {text!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"PETS field {label} must be finite")
    return value


def _required_pair(block: gemmi.cif.Block, tag: str) -> str:
    """Return a required CIF value as text; raise ValueError when it is missing, ``.`` or ``?``."""

    value = block.find_value(tag)
    if value is None or str(value).strip() in {"", ".", "?"}:
        raise ValueError(f"PETS file is missing required field {tag}")
    return str(value)


def _rotation(axis: tuple[float, float, float], angle_deg: float) -> np.ndarray:
    """Return the rotation matrix about ``axis`` by ``angle_deg`` degrees (Rodrigues formula)."""

    axis_array = np.asarray(axis, dtype=float)
    axis_array /= np.linalg.norm(axis_array)
    x, y, z = axis_array
    theta = math.radians(float(angle_deg))
    c, s = math.cos(theta), math.sin(theta)
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + s * cross + (1.0 - c) * (cross @ cross)


def _cell_matrix(parameters: np.ndarray) -> np.ndarray:
    """Return the real-space cell vectors a, b, c (A) as rows, with a along x and b in the xy plane."""

    a, b, c, alpha_deg, beta_deg, gamma_deg = np.asarray(parameters, dtype=float)
    alpha, beta, gamma = np.radians([alpha_deg, beta_deg, gamma_deg])
    ca, cb, cg = np.cos([alpha, beta, gamma])
    sg = np.sin(gamma)
    volume_factor = np.sqrt(1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg)
    return np.asarray(
        [
            [a, 0.0, 0.0],
            [b * cg, b * sg, 0.0],
            [c * cb, c * (ca - cb * cg) / sg, c * volume_factor / sg],
        ]
    )


def _busing_levy(parameters: np.ndarray) -> np.ndarray:
    """Return the reciprocal column basis inverse to the direct row basis.

    Its Gram matrix must equal the reciprocal-cell metric. In particular the
    (3, 2) term has the opposite sign to the direct-cell (3, 2) term; the wrong
    sign is hidden when alpha and gamma are both 90 degrees.
    """
    a, b, c, alpha_deg, beta_deg, gamma_deg = np.asarray(parameters, dtype=float)
    alpha, beta, gamma = np.radians([alpha_deg, beta_deg, gamma_deg])
    ca, cb, cg = np.cos([alpha, beta, gamma])
    sg = np.sin(gamma)
    volume = a * b * c * np.sqrt(1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg)
    return np.asarray(
        [
            [1.0 / a, 0.0, 0.0],
            [-cg / (a * sg), 1.0 / (b * sg), 0.0],
            [
                b * c / volume * (cg * (ca - cb * cg) / sg - cb * sg),
                -a * c / (volume * sg) * (ca - cb * cg),
                a * b * sg / volume,
            ],
        ]
    )


def _validate_cell_geometry(parameters: np.ndarray) -> None:
    """Reject nonphysical cells before inverse or pseudoinverse construction."""

    values = np.asarray(parameters, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError("PETS unit cell must contain six finite values")
    if np.any(values[:3] <= 0.0):
        raise ValueError("PETS unit-cell lengths must be positive")
    if np.any(values[3:] <= 0.0) or np.any(values[3:] >= 180.0):
        raise ValueError("PETS unit-cell angles must lie strictly between 0 and 180 degrees")
    ca, cb, cg = np.cos(np.radians(values[3:]))
    squared_volume_factor = 1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg
    if not math.isfinite(float(squared_volume_factor)) or squared_volume_factor <= 0.0:
        raise ValueError("PETS unit-cell angles do not define a positive cell volume")
    with np.errstate(over="ignore", invalid="ignore"):
        volume = float(np.prod(values[:3]) * np.sqrt(squared_volume_factor))
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("PETS unit-cell volume must be finite and positive")


def _validate_ub_metric(ub: np.ndarray, parameters: np.ndarray) -> None:
    """Require UB to carry the declared reciprocal metric, allowing rounding.

    PETS commonly prints UB entries to five decimals in inverse Angstroms.
    Propagate a half-last-digit uncertainty of 5e-6 per entry through UB.T@UB,
    then add 1e-4 of each reciprocal metric scale for cell/model serialization.
    This is a bounded input tolerance, not permission to fit a deformed cell.
    A left orientation rotation preserves this test exactly.
    """

    matrix = np.asarray(ub, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("PETS UB matrix must contain nine finite values")
    reciprocal = _busing_levy(parameters)
    reference = reciprocal.T @ reciprocal
    observed = matrix.T @ matrix
    uncertainty = np.full((3, 3), 5e-6, dtype=float)
    rounding_bound = np.abs(matrix).T @ uncertainty + uncertainty.T @ np.abs(matrix) + uncertainty.T @ uncertainty
    metric_scale = np.sqrt(np.outer(np.diag(reference), np.diag(reference)))
    tolerance = rounding_bound + 1e-4 * metric_scale
    error = np.abs(observed - reference)
    if not np.all(np.isfinite(observed)) or np.any(error > tolerance):
        maximum_ratio = float(np.max(error / tolerance))
        raise ValueError(
            "PETS UB reciprocal metric is inconsistent with the declared unit cell "
            f"(maximum error/tolerance {maximum_ratio:.6g}); check UB units, cell and setting"
        )


def _axis_position(block: gemmi.cif.Block) -> float:
    """Return the rotation (tilt) axis position angle in degrees from the PETS measurement text; raise if absent."""

    for tag in ("_diffrn_measurement_details", "_diffrn_reflns_reduction_process"):
        text = block.find_value(tag)
        if text is None:
            continue
        match = _AXIS_POSITION.search(str(text))
        if match:
            return _number(match.group(1), label="PETS rotation axis position")
    raise ValueError(
        "PETS file does not record 'rotation axis position:' or 'tilt axis position:'; "
        "the exact rocking axis cannot be assumed"
    )


def _continuous_rotation_contract(block: gemmi.cif.Block) -> dict[str, str]:
    """Fail closed unless PETS explicitly declares continuous rotation."""

    precession = str(block.find_value("_diffrn_measurement_method_precession") or "").strip()
    rotation_mode = str(block.find_value("_diffrn_measurement_rotation_mode") or "").strip()
    details = "\n".join(
        str(block.find_value(tag) or "")
        for tag in (
            "_diffrn_measurement_details",
            "_diffrn_reflns_reduction_process",
        )
    )
    precession_off = precession.lower() in {"no", "false", "0", "off"}
    continuous = (
        rotation_mode.lower()
        in {
            "rotation",
            "continuous_rotation",
            "continuous rotation",
        }
        or "continuous rotation" in details.lower()
    )
    if not precession_off or not continuous:
        raise ValueError(
            "PETS import currently supports explicitly declared continuous-rotation "
            "data only; precession/conical acquisition needs a different rocking path "
            f"(method_precession={precession!r}, rotation_mode={rotation_mode!r})"
        )
    return {
        "method_precession": precession,
        "rotation_mode": rotation_mode,
        "geometry": "continuous_rotation",
    }


_AXIS_POSITION = re.compile(
    r"(?:rotation|tilt)\s+axis\s+position:\s*"
    r"([-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?))",
    re.IGNORECASE,
)
