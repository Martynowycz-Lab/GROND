"""Small deterministic geometry and content-hash helpers shared by the row engine."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import gemmi
import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = np.radians(float(angle_deg))
    c, s = np.cos(theta), np.sin(theta)
    kx = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + s * kx + (1.0 - c) * (kx @ kx)


def parse_hkl_transform(spec: str | None) -> tuple[np.ndarray, str]:
    """Parse a signed axis permutation mapping XDS HKLs into model HKLs.

    The returned matrix T uses column-vector convention:
    h_model = T @ h_xds.  For row arrays this is hkl_model = hkl_xds @ T.T.
    """
    from grond.hkl_transform import parse_hkl_transform_info

    info = parse_hkl_transform_info(spec)
    return np.asarray(info.matrix, dtype=int), info.canonical


def cell_from_real_matrix(real_matrix: np.ndarray) -> gemmi.UnitCell:
    real = np.asarray(real_matrix, dtype=float)
    a_vec, b_vec, c_vec = real[:, 0], real[:, 1], real[:, 2]

    def length(v):
        return float(np.linalg.norm(v))

    def angle(u, v):
        denom = max(length(u) * length(v), 1e-30)
        cosang = float(np.dot(u, v) / denom)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosang))))

    return gemmi.UnitCell(
        length(a_vec),
        length(b_vec),
        length(c_vec),
        angle(b_vec, c_vec),
        angle(a_vec, c_vec),
        angle(a_vec, b_vec),
    )


def _rotation_matrix_axis_angle(axis, angle_deg):
    """Return a right-handed lab-frame rotation matrix."""

    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) or norm <= 0.0:
        raise ValueError("geometry-path rotation axis must be a finite nonzero 3-vector")
    axis = axis / norm
    x, y, z = axis
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    theta = math.radians(float(angle_deg))
    return np.eye(3, dtype=float) + math.sin(theta) * cross + (1.0 - math.cos(theta)) * (cross @ cross)


def load_geometry_path_contract(path):
    """Load one geometry-only smooth scan path with strict provenance checks."""

    text = str(path or "").strip()
    if not text:
        return None
    resolved = Path(text).expanduser().resolve()
    payload = json.loads(resolved.read_text())
    if int(payload.get("schema_version", -1)) != 1 or payload.get("kind") != "grond_geometry_path":
        raise ValueError("geometry path must be a schema-1 grond_geometry_path JSON file")
    if payload.get("basis") != "chebyshev":
        raise ValueError("geometry path currently requires Chebyshev coefficients")
    center = float(payload.get("frame_center"))
    half_range = float(payload.get("frame_half_range"))
    if not math.isfinite(center) or not math.isfinite(half_range) or half_range <= 0.0:
        raise ValueError("geometry path frame centre/range must be finite and positive")
    coefficients = payload.get("coefficients") or {}
    checked = {}
    for name in ("slew_frames", "tilt_x_deg", "tilt_y_deg"):
        values = np.asarray(coefficients.get(name, []), dtype=float)
        if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"geometry path {name} coefficients must be a finite vector")
        checked[name] = values
    axes = payload.get("tilt_axes_lab") or []
    if len(axes) != 2:
        raise ValueError("geometry path must define two lab-frame tilt axes")
    axes = np.asarray(axes, dtype=float)
    if axes.shape != (2, 3) or not np.all(np.isfinite(axes)):
        raise ValueError("geometry path tilt axes must be a finite 2 by 3 array")
    axis_norms = np.linalg.norm(axes, axis=1)
    if np.any(axis_norms <= 0.0):
        raise ValueError("geometry path tilt axes must be nonzero")
    axes /= axis_norms[:, None]
    bounds = payload.get("bounds") or {}
    max_slew = float(bounds.get("max_abs_slew_frames", float("nan")))
    max_tilt = float(bounds.get("max_abs_tilt_deg", float("nan")))
    if not math.isfinite(max_slew) or max_slew <= 0.0:
        raise ValueError("geometry path must record a positive finite slew bound")
    if not math.isfinite(max_tilt) or max_tilt <= 0.0:
        raise ValueError("geometry path must record a positive finite tilt bound")
    sampled_x = np.linspace(-1.0, 1.0, 1001)
    sampled = {name: np.polynomial.chebyshev.chebval(sampled_x, values) for name, values in checked.items()}
    if float(np.max(np.abs(sampled["slew_frames"]))) > max_slew + 1e-10:
        raise ValueError("geometry path slew exceeds its declared bound")
    for name in ("tilt_x_deg", "tilt_y_deg"):
        if float(np.max(np.abs(sampled[name]))) > max_tilt + 1e-10:
            raise ValueError(f"geometry path {name} exceeds its declared bound")
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "frame_center": center,
        "frame_half_range": half_range,
        "coefficients": checked,
        "tilt_axes_lab": axes,
        "source_hkl_sha256": str(payload.get("source_hkl_sha256", "") or ""),
        "bounds": {"max_abs_slew_frames": max_slew, "max_abs_tilt_deg": max_tilt},
        "fit_source": payload.get("fit_source"),
        "validation": payload.get("validation"),
    }
