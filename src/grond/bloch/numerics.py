"""Small deterministic geometry, content-hash and eigensolver helpers shared by the row engine."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import gemmi
import numpy as np
from scipy.linalg import eigh as _eigh


def sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in 1 MiB chunks."""

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hermitian_eigh(H):
    """Eigenvalues and eigenvectors of a Hermitian Bloch matrix.

    A centrosymmetric model with its inversion centre at the origin (SHELX LATT > 0) has real potential coefficients,
    so its elastic Bloch matrix is real symmetric. When the imaginary part is below the floating-point resolution of
    the matrix, the real symmetric solver returns the same spectrum about twice as fast. The eigenvectors keep the
    matrix's complex type, so callers see no difference.
    """

    H = np.asarray(H)
    if np.iscomplexobj(H) and H.size:
        real = H.real
        if np.max(np.abs(H.imag)) <= 8.0 * np.finfo(real.dtype).eps * np.max(np.abs(real)):
            vals, vecs = _eigh(np.ascontiguousarray(real))
            return vals, vecs.astype(H.dtype)
    return _eigh(H)


def validate_row_indices(indices, *, row_count: int) -> np.ndarray:
    """Return unique in-range integer row indices without truncating caller input."""

    values = np.asarray(indices)
    if values.ndim != 1 or values.dtype.kind not in {"i", "u", "f"}:
        raise ValueError("row indices must be a one-dimensional array of integers")
    fractional = values.dtype.kind == "f" and (
        not np.all(np.isfinite(values)) or np.any(values != np.floor(values))
    )
    if fractional or np.any(values < 0) or np.any(values >= row_count):
        raise ValueError(f"row indices must be finite integers in [0, {row_count})")
    result = values.astype(np.int64)
    if len(np.unique(result)) != len(result):
        raise ValueError("prediction row indices must be unique")
    return result


def _unit_axis(axis: np.ndarray) -> np.ndarray:
    """Normalize a finite axis without overflow or underflow in its squared norm."""

    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) or not np.any(axis):
        raise ValueError("rotation axis must be a finite nonzero 3-vector")
    axis = axis / np.max(np.abs(axis))
    return axis / np.linalg.norm(axis)


def rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Return the right-handed Rodrigues rotation; angle in degrees and axis normalised here."""

    axis = _unit_axis(axis)
    if not math.isfinite(float(angle_deg)):
        raise ValueError("rotation angle must be finite")
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
    """Return the gemmi UnitCell whose a, b and c vectors (Å) are the columns of ``real_matrix``."""

    real = np.asarray(real_matrix, dtype=float)
    a_vec, b_vec, c_vec = real[:, 0], real[:, 1], real[:, 2]

    def length(v):
        """Return the Euclidean length of a vector as a float."""

        return float(np.linalg.norm(v))

    def angle(u, v):
        """Return the angle between two vectors in degrees, with the cosine clipped to [-1, 1]."""

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
    """Compatibility entry point for geometry paths using the same Rodrigues rotation."""

    return rotation_matrix(axis, angle_deg)


def _chebyshev_range(coefficients):
    """Minimum and maximum p(x) on [-1,1], checking endpoints and derivative roots."""

    roots = np.polynomial.chebyshev.chebroots(np.polynomial.chebyshev.chebder(coefficients))
    # Real roots contain the extrema. Projections of any complex roots merely add valid check points and avoid
    # dropping an almost-real root because of eigensolver roundoff.
    interior = roots.real[(roots.real >= -1.0) & (roots.real <= 1.0)]
    points = np.r_[-1.0, interior, 1.0]
    values = np.polynomial.chebyshev.chebval(points, coefficients)
    if not np.all(np.isfinite(roots)) or not np.all(np.isfinite(values)):
        raise ValueError("geometry path polynomial is not finite on its declared interval")
    return float(np.min(values)), float(np.max(values))


def load_geometry_path_contract(path):
    """Load one geometry-only smooth scan path with strict provenance checks."""

    text = str(path or "").strip()
    if not text:
        return None
    resolved = Path(text).expanduser().resolve()
    source_bytes = resolved.read_bytes()
    payload = json.loads(source_bytes)
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
    if np.any(np.max(np.abs(axes), axis=1) == 0.0):
        raise ValueError("geometry path tilt axes must be nonzero")
    axes = np.asarray([_unit_axis(axis) for axis in axes])
    bounds = payload.get("bounds") or {}
    max_slew = float(bounds.get("max_abs_slew_frames", float("nan")))
    max_tilt = float(bounds.get("max_abs_tilt_deg", float("nan")))
    if not math.isfinite(max_slew) or max_slew <= 0.0:
        raise ValueError("geometry path must record a positive finite slew bound")
    if not math.isfinite(max_tilt) or max_tilt <= 0.0:
        raise ValueError("geometry path must record a positive finite tilt bound")
    maxima = {name: max(abs(v) for v in _chebyshev_range(values)) for name, values in checked.items()}
    if maxima["slew_frames"] > max_slew + 1e-10:
        raise ValueError("geometry path slew exceeds its declared bound")
    for name in ("tilt_x_deg", "tilt_y_deg"):
        if maxima[name] > max_tilt + 1e-10:
            raise ValueError(f"geometry path {name} exceeds its declared bound")
    # A corrected angle must identify one nominal frame. Outside this interval the correction is held constant,
    # so its slope is one. Inside, a nonnegative, nonzero polynomial derivative gives a strictly increasing map:
    # isolated stationary points are allowed, but reversal or a constant interval would make the inverse ambiguous.
    frame_derivative = np.polynomial.chebyshev.chebder(checked["slew_frames"]) / half_range
    frame_derivative[0] += 1.0
    if _chebyshev_range(frame_derivative)[0] < 0.0 or not np.any(frame_derivative):
        raise ValueError("geometry path corrected-frame mapping must be strictly increasing")
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "frame_center": center,
        "frame_half_range": half_range,
        "coefficients": checked,
        "tilt_axes_lab": axes,
        "source_hkl_sha256": str(payload.get("source_hkl_sha256", "") or ""),
        "bounds": {"max_abs_slew_frames": max_slew, "max_abs_tilt_deg": max_tilt},
        "fit_source": payload.get("fit_source"),
        "validation": payload.get("validation"),
    }
