"""Convert scan origins and orientations, then check the rows against the Ewald sphere."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Return the proper rotation matrix for a DIALS/XDS goniometer angle."""

    axis = np.asarray(axis, dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-30)
    x, y, z = axis
    theta = math.radians(float(angle_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=float,
    )


def _interp_scan_a(a_scan: np.ndarray, z: float) -> np.ndarray:
    """Match the Bloch engine's linear interpolation of DIALS scan-point A matrices."""

    if len(a_scan) == 1:
        return a_scan[0]
    if z <= 0:
        return a_scan[0]
    if z >= len(a_scan) - 1:
        return a_scan[-1]
    lo = int(math.floor(float(z)))
    frac = float(z) - lo
    return (1.0 - frac) * a_scan[lo] + frac * a_scan[lo + 1]


def _scan_array_range(metadata: dict[str, Any]) -> tuple[int, int]:
    """Return DIALS' zero-based, half-open scan array range."""

    scan = metadata.get("scan") or {}
    image_range = scan.get("image_range") or []
    array_range = scan.get("array_range") or []
    if array_range:
        if len(array_range) != 2:
            raise ValueError("DIALS scan array_range must contain exactly two integers")
        start, stop = (int(array_range[0]), int(array_range[1]))
    elif image_range:
        if len(image_range) != 2:
            raise ValueError("DIALS scan image_range must contain exactly two integers")
        start, stop = int(image_range[0]) - 1, int(image_range[1])
    else:
        start, stop = 0, 0
    if stop < start:
        raise ValueError(f"DIALS scan array_range is reversed: [{start}, {stop}]")
    if image_range:
        expected = (int(image_range[0]) - 1, int(image_range[1]))
        if (start, stop) != expected:
            raise ValueError(
                "DIALS scan image_range and array_range disagree: "
                f"image_range={list(image_range)} implies {list(expected)}, "
                f"array_range={[start, stop]}"
            )
    return start, stop


def _a_scan_points_for_current_scan(
    metadata: dict[str, Any],
    *,
    scorer=None,
    source_index_offset: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return scan-point A matrices in the cropped scan's local frame convention.

    DIALS can retain the full scan-varying crystal model when an experiment's
    ``Scan`` is cropped.  Reflection z and the scan origin remain in the
    original array coordinate, while GROND's sidecar is indexed locally after
    subtracting ``STARTING_FRAME``.  In that case select the inclusive
    ``[array_start, array_stop]`` A-matrix slice.  Already-local and static
    models are preserved unchanged.
    """

    raw = np.asarray(metadata["A_scan_points"], dtype=float)
    if raw.ndim != 3 or raw.shape[1:] != (3, 3) or len(raw) == 0:
        raise ValueError("DIALS A_scan_points must be a non-empty N x 3 x 3 array")
    start, stop = _scan_array_range(metadata)
    local_point_count = max(stop - start + 1, 1)
    source = "static" if len(raw) == 1 else "already_local"
    origin = 0
    selected = raw
    tested_offsets: list[dict[str, float | int]] = []
    if len(raw) > local_point_count:
        maximum_offset = len(raw) - local_point_count
        if source_index_offset is not None:
            origin = int(source_index_offset)
            if origin < 0 or origin > maximum_offset:
                raise ValueError(f"DIALS A-scan source offset {origin} is outside [0, {maximum_offset}]")
            selected = raw[origin : origin + local_point_count]
            source = "replayed_geometry_selected_offset"
        elif scorer is not None:
            preferred_offsets = []
            for candidate in (min(max(start, 0), maximum_offset), 0, maximum_offset):
                if candidate not in preferred_offsets:
                    preferred_offsets.append(candidate)
            scored: list[tuple[float, int]] = []
            for candidate in preferred_offsets:
                score = float(scorer(raw[candidate : candidate + local_point_count]))
                scored.append((score, candidate))
                tested_offsets.append({"source_index_offset": candidate, "p95_abs_Sg_inv_A": score})
            if min(score for score, _ in scored) > DIALS_SG_P95_ABS_LIMIT_INV_A:
                for candidate in range(maximum_offset + 1):
                    if candidate in preferred_offsets:
                        continue
                    score = float(scorer(raw[candidate : candidate + local_point_count]))
                    scored.append((score, candidate))
                    tested_offsets.append({"source_index_offset": candidate, "p95_abs_Sg_inv_A": score})
            _, origin = min(scored)
            selected = raw[origin : origin + local_point_count]
            source = "geometry_selected_cropped_scan_model"
        elif start > 0:
            origin = min(start, maximum_offset)
            selected = raw[origin : origin + local_point_count]
            source = "cropped_from_full_scan_model"
    return selected, {
        "source": source,
        "raw_scan_points": int(len(raw)),
        "written_scan_points": int(len(selected)),
        "source_scan_point_origin": int(origin),
        "scan_array_range": [int(start), int(stop)],
        "geometry_candidate_scores": tested_offsets,
    }


def dials_sg_statistics(arrays: dict[str, np.ndarray], metadata: dict[str, Any], keep: np.ndarray) -> dict[str, Any]:
    """Compute excitation-error diagnostics from the generated GROND geometry."""

    if not np.any(keep):
        return {"status": "error", "message": "no reflection rows survived DIALS bridge filtering", "rows": 0}
    hkl = np.asarray(arrays["hkl"], dtype=float)
    xyzcal = np.asarray(arrays["xyzcal"], dtype=float)
    goniometer = metadata.get("goniometer") or {}
    axis = np.asarray(
        goniometer.get("rotation_axis_datum", goniometer.get("rotation_axis", [1.0, 0.0, 0.0])),
        dtype=float,
    )
    fixed = np.asarray(goniometer.get("fixed_rotation", np.eye(3)), dtype=float).reshape(3, 3)
    setting = np.asarray(goniometer.get("setting_rotation", np.eye(3)), dtype=float).reshape(3, 3)
    s0 = np.asarray(metadata["beam"]["s0"], dtype=float)
    k = float(np.linalg.norm(s0))
    start_angle, osc_width = [float(v) for v in metadata["scan"]["oscillation"]]
    starting_frame, _ = _scan_array_range(metadata)

    def sg_values(a_scan: np.ndarray) -> np.ndarray:
        """Return the excitation error Sg (1/A) of every kept row for one set of scan-point A matrices.

        Sg = (k^2 - |s0 + g|^2) / (2k), where g is the rotated reciprocal-lattice vector at
        the row's calculated frame and k = |s0|.
        """

        sg = []
        for i in np.where(keep)[0]:
            scan_z = float(xyzcal[i, 2]) - float(starting_frame)
            omega = start_angle + scan_z * osc_width
            a_mat = _interp_scan_a(a_scan, scan_z)
            g = setting @ _rotation_matrix(axis, omega) @ fixed @ a_mat @ hkl[i]
            value = (k * k - float(np.sum((s0 + g) ** 2))) / (2.0 * k)
            sg.append(value)
        return np.asarray(sg, dtype=float)

    def p95_abs_sg(a_scan: np.ndarray) -> float:
        """Return the 95th percentile of abs(Sg) in 1/A, or infinity when no row gives a finite value."""

        values = sg_values(a_scan)
        finite = values[np.isfinite(values)]
        return float(np.percentile(np.abs(finite), 95)) if len(finite) else math.inf

    a_scan, a_scan_contract = _a_scan_points_for_current_scan(metadata, scorer=p95_abs_sg)
    sg_arr = sg_values(a_scan)
    finite_sg = sg_arr[np.isfinite(sg_arr)]
    median_sg = float(np.median(finite_sg)) if len(finite_sg) else None
    median_abs_sg = float(np.median(np.abs(finite_sg))) if len(finite_sg) else None
    p95_abs_sg = float(np.percentile(np.abs(finite_sg), 95)) if len(finite_sg) else None
    max_abs_sg = float(np.max(np.abs(finite_sg))) if len(finite_sg) else None
    violations = []
    if len(finite_sg) != len(sg_arr):
        violations.append(f"{len(sg_arr) - len(finite_sg)} rows produced nonfinite Sg")
    if p95_abs_sg is None:
        violations.append("no rows produced a finite Sg")
    elif p95_abs_sg > DIALS_SG_P95_ABS_LIMIT_INV_A:
        violations.append(f"p95 |Sg| {p95_abs_sg:.6g} exceeds {DIALS_SG_P95_ABS_LIMIT_INV_A:.6g} A^-1")
    return {
        "status": "error" if violations else "ok",
        "message": "; ".join(violations),
        "rows": int(len(sg_arr)),
        "finite_rows": int(len(finite_sg)),
        "starting_frame": int(starting_frame),
        "median_sg_inv_A": median_sg,
        "median_abs_sg_inv_A": median_abs_sg,
        "p95_abs_sg_inv_A": p95_abs_sg,
        "max_abs_sg_inv_A": max_abs_sg,
        "a_scan_contract": a_scan_contract,
        "limits_inv_A": {
            "p95_abs_sg": DIALS_SG_P95_ABS_LIMIT_INV_A,
        },
    }


def _require_valid_dials_geometry(stats: dict[str, Any]) -> None:
    """Refuse a bridge whose predicted rows are grossly off the Ewald sphere."""

    if stats.get("status") == "ok":
        return
    raise ValueError(
        "DIALS geometry validation failed after applying the scan frame origin: "
        f"{stats.get('message')}. Refusing to write GROND rows. Confirm that the .refl and .expt "
        "belong together and that the scan, goniometer, and scan-varying crystal model are supported."
    )


def _real_space_axes_from_a(a_static: np.ndarray) -> np.ndarray:
    """Return real-space axes as columns from a reciprocal A matrix."""

    return np.linalg.inv(np.asarray(a_static, dtype=float).T)


def _format_header_vector(values: list[float] | np.ndarray) -> str:
    """Format numbers as space-separated ``%.10g`` text for an INTEGRATE.HKL header line."""

    return " ".join(f"{float(v): .10g}" for v in values)


DIALS_SG_P95_ABS_LIMIT_INV_A = 5.0e-3
