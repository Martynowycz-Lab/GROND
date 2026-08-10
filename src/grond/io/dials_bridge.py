"""Convert DIALS integration output into the row contract used by GROND.

The GROND Bloch engine intentionally works from signed observation rows.  DIALS
stores those rows in a reflection table (`integrated.refl`) and stores the beam,
goniometer, scan, detector, crystal model, and scan-varying orientation in the
matching experiment list (`integrated.expt`).  This module reads that pair and
writes an XDS-like `INTEGRATE.HKL` plus a small sidecar containing the DIALS
scan-point A matrices.

The generated file is not pretending to be an XDS reduction.  It is a compact
compatibility contract for the existing GROND row-level engine: signed HKLs,
summation intensities, partiality as the `PEAK`-style row quality, and DIALS
scan-varying orientation matrices for excitation-error calculations.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls

DIALS_SG_P95_ABS_LIMIT_INV_A = 5.0e-3

DIALS_HELPER_CODE = r"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory


def bool_flags(table, name):
    flag = getattr(table.flags, name, None)
    if flag is None:
        return np.zeros(len(table), dtype=bool)
    return np.asarray(list(table.get_flags(flag)), dtype=bool)


def float_col(table, name, default=np.nan):
    if name not in table:
        return np.full(len(table), default, dtype=float)
    return np.asarray(list(table[name]), dtype=float)


def int_col(table, name, default=0):
    if name not in table:
        return np.full(len(table), default, dtype=np.int32)
    return np.asarray(list(table[name]), dtype=np.int32)


def vec3_col(table, name, default=np.nan):
    if name not in table:
        return np.full((len(table), 3), default, dtype=float)
    return np.asarray(list(table[name]), dtype=float)


def bbox_col(table):
    if "bbox" not in table:
        return np.full((len(table), 6), -1, dtype=np.int32)
    return np.asarray(list(table["bbox"]), dtype=np.int32)


def first_panel_metadata(detector):
    if detector is None or len(detector) == 0:
        return {}
    panel = detector[0]
    try:
        distance = float(panel.get_distance())
    except Exception:
        distance = None
    try:
        gain = float(panel.get_gain())
    except Exception:
        gain = None
    return {
        "n_panels": int(len(detector)),
        "panel0_name": str(panel.get_name()),
        "origin_mm": [float(v) for v in panel.get_origin()],
        "fast_axis": [float(v) for v in panel.get_fast_axis()],
        "slow_axis": [float(v) for v in panel.get_slow_axis()],
        "pixel_size_mm": [float(v) for v in panel.get_pixel_size()],
        "image_size_px": [int(v) for v in panel.get_image_size()],
        "distance_mm": distance,
        "gain": gain,
    }


def main():
    refl_path = Path(sys.argv[1])
    expt_path = Path(sys.argv[2])
    out_npz = Path(sys.argv[3])
    out_json = Path(sys.argv[4])

    table = flex.reflection_table.from_file(str(refl_path))
    experiments = ExperimentListFactory.from_json_file(str(expt_path), check_format=False)
    if len(experiments) != 1:
        raise SystemExit(
            f"GROND import-dials currently expects one experiment per dataset; found {len(experiments)}. "
            "Split multi-experiment DIALS outputs into one GROND dataset per experiment first."
        )

    exp = experiments[0]
    crystal = exp.crystal
    beam = exp.beam
    goniometer = exp.goniometer
    scan = exp.scan
    detector = exp.detector

    hkl = np.asarray(list(table["miller_index"]), dtype=np.int32)
    xyzcal = vec3_col(table, "xyzcal.px")
    xyzobs = vec3_col(table, "xyzobs.px.value")
    missing_xyzobs = ~np.all(np.isfinite(xyzobs), axis=1)
    xyzobs[missing_xyzobs] = xyzcal[missing_xyzobs]

    arrays = {
        "hkl": hkl,
        "i_sum": float_col(table, "intensity.sum.value"),
        "var_sum": float_col(table, "intensity.sum.variance"),
        "i_prf": float_col(table, "intensity.prf.value"),
        "var_prf": float_col(table, "intensity.prf.variance"),
        "i_scale": float_col(table, "intensity.scale.value"),
        "var_scale": float_col(table, "intensity.scale.variance"),
        "lp": float_col(table, "lp", 1.0),
        "qe": float_col(table, "qe", 1.0) if "qe" in table else float_col(table, "dqe", 1.0),
        "partiality": float_col(table, "partiality", 1.0),
        "zeta": float_col(table, "zeta"),
        "d": float_col(table, "d"),
        "bbox": bbox_col(table),
        "xyzcal": xyzcal,
        "xyzobs": xyzobs,
        "panel": int_col(table, "panel", 0),
        "id": int_col(table, "id", 0),
        "integrated_sum": bool_flags(table, "integrated_sum"),
        "integrated_prf": bool_flags(table, "integrated_prf"),
        "dont_integrate": bool_flags(table, "dont_integrate"),
        "overloaded": bool_flags(table, "overloaded"),
        "in_powder_ring": bool_flags(table, "in_powder_ring"),
        "excluded_for_scaling": bool_flags(table, "excluded_for_scaling"),
        "bad_for_scaling": bool_flags(table, "bad_for_scaling"),
        "outlier_in_scaling": bool_flags(table, "outlier_in_scaling"),
        "scaled": bool_flags(table, "scaled"),
        "background_includes_bad_pixels": bool_flags(table, "background_includes_bad_pixels"),
        "foreground_includes_bad_pixels": bool_flags(table, "foreground_includes_bad_pixels"),
        "includes_bad_pixels": bool_flags(table, "includes_bad_pixels"),
        "bad_shoebox": bool_flags(table, "bad_shoebox"),
        "failed_during_background_modelling": bool_flags(table, "failed_during_background_modelling"),
        "failed_during_summation": bool_flags(table, "failed_during_summation"),
    }
    np.savez_compressed(out_npz, **arrays)

    s0 = [float(v) for v in beam.get_s0()]
    wavelength = float(beam.get_wavelength())
    if scan is not None:
        image_range = [int(v) for v in scan.get_image_range()]
        if hasattr(scan, "get_array_range"):
            array_range = [int(v) for v in scan.get_array_range()]
        else:
            array_range = [int(image_range[0]) - 1, int(image_range[1])]
        oscillation = [float(v) for v in scan.get_oscillation()]
    else:
        image_range = [1, int(len(table))]
        array_range = [0, int(len(table))]
        oscillation = [0.0, 1.0]

    a_static = np.asarray(crystal.get_A(), dtype=float).reshape(3, 3)
    n_scan_points = int(getattr(crystal, "num_scan_points", 0) or 0)
    a_scan_points = []
    if n_scan_points > 0 and hasattr(crystal, "get_A_at_scan_point"):
        for i in range(n_scan_points):
            a_scan_points.append(np.asarray(crystal.get_A_at_scan_point(i), dtype=float).reshape(3, 3).tolist())
    if not a_scan_points:
        a_scan_points = [a_static.tolist()]

    identity = np.eye(3, dtype=float)
    if goniometer is None:
        rotation_axis = [1.0, 0.0, 0.0]
        rotation_axis_datum = rotation_axis
        fixed_rotation = identity
        setting_rotation = identity
        setting_scan_points = 0
    else:
        rotation_axis = [float(v) for v in goniometer.get_rotation_axis()]
        if hasattr(goniometer, "get_rotation_axis_datum"):
            rotation_axis_datum = [float(v) for v in goniometer.get_rotation_axis_datum()]
        else:
            rotation_axis_datum = rotation_axis
        fixed_rotation = np.asarray(goniometer.get_fixed_rotation(), dtype=float).reshape(3, 3)
        setting_rotation = np.asarray(goniometer.get_setting_rotation(), dtype=float).reshape(3, 3)
        setting_scan_points = int(goniometer.get_num_scan_points()) if hasattr(goniometer, "get_num_scan_points") else 0
    if setting_scan_points:
        raise SystemExit(
            "GROND import-dials does not yet support scan-varying goniometer setting rotations; "
            f"the experiment contains {setting_scan_points} setting-rotation scan points. "
            "Export a static setting rotation or split the experiment before conversion."
        )

    meta = {
        "source_refl": str(refl_path),
        "source_expt": str(expt_path),
        "row_count": int(len(table)),
        "experiment_count": int(len(experiments)),
        "space_group": str(crystal.get_space_group().info()),
        "unit_cell": [float(v) for v in crystal.get_unit_cell().parameters()],
        "A_static": a_static.tolist(),
        "A_scan_points": a_scan_points,
        "num_scan_points": int(len(a_scan_points)),
        "beam": {
            "wavelength_A": wavelength,
            "s0": s0,
            "sample_to_source_direction": [float(v) for v in beam.get_sample_to_source_direction()],
        },
        "goniometer": {
            "rotation_axis": rotation_axis,
            "rotation_axis_datum": rotation_axis_datum,
            "fixed_rotation": fixed_rotation.tolist(),
            "setting_rotation": setting_rotation.tolist(),
            "setting_rotation_scan_points": setting_scan_points,
        },
        "scan": {
            "image_range": image_range,
            "array_range": array_range,
            "oscillation": oscillation,
        },
        "detector": first_panel_metadata(detector),
    }
    out_json.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
"""


@dataclass
class DialsBridgeOutput:
    """Paths and summary information written by the DIALS bridge."""

    out_dir: Path
    hkl: Path
    scan_points: Path
    summary: Path
    prep: Path | None = None


def _run_dials_helper(refl: Path, expt: Path, dials_python: str = "") -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract DIALS reflection and experiment arrays using a DIALS-capable Python."""

    python = str(dials_python or sys.executable)
    with tempfile.TemporaryDirectory(prefix="grond_dials_") as tmp:
        tmp_path = Path(tmp)
        out_npz = tmp_path / "dials_arrays.npz"
        out_json = tmp_path / "dials_metadata.json"
        cmd = [python, "-c", DIALS_HELPER_CODE, str(refl), str(expt), str(out_npz), str(out_json)]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode != 0:
            hint = (
                "Could not import DIALS reflection data. Run GROND from a DIALS Python environment "
                "or pass --dials-python /path/to/dials.python or another Python executable that can "
                "import dials and dxtbx."
            )
            raise RuntimeError(f"{hint}\nCommand: {' '.join(cmd)}\nSTDERR:\n{proc.stderr.strip()}") from None
        with np.load(out_npz, allow_pickle=False) as loaded:
            arrays = {name: np.array(loaded[name]) for name in loaded.files}
        metadata = json.loads(out_json.read_text())
    return arrays, metadata


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


def _a_scan_points_for_current_scan(metadata: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
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
    if len(raw) > 1 and start > 0 and len(raw) >= stop + 1 and len(raw) != local_point_count:
        selected = raw[start : stop + 1]
        source = "cropped_from_full_scan_model"
        origin = start
    return selected, {
        "source": source,
        "raw_scan_points": int(len(raw)),
        "written_scan_points": int(len(selected)),
        "source_scan_point_origin": int(origin),
        "scan_array_range": [int(start), int(stop)],
    }


def dials_sg_statistics(arrays: dict[str, np.ndarray], metadata: dict[str, Any], keep: np.ndarray) -> dict[str, Any]:
    """Compute excitation-error diagnostics from the generated GROND geometry."""

    if not np.any(keep):
        return {"status": "error", "message": "no reflection rows survived DIALS bridge filtering", "rows": 0}
    hkl = np.asarray(arrays["hkl"], dtype=float)
    xyzcal = np.asarray(arrays["xyzcal"], dtype=float)
    a_scan, a_scan_contract = _a_scan_points_for_current_scan(metadata)
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

    sg = []
    for i in np.where(keep)[0]:
        scan_z = float(xyzcal[i, 2]) - float(starting_frame)
        omega = start_angle + scan_z * osc_width
        a_mat = _interp_scan_a(a_scan, scan_z)
        g = setting @ _rotation_matrix(axis, omega) @ fixed @ a_mat @ hkl[i]
        value = (k * k - float(np.sum((s0 + g) ** 2))) / (2.0 * k)
        sg.append(value)
    sg_arr = np.asarray(sg, dtype=float)
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
        violations.append(
            f"p95 |Sg| {p95_abs_sg:.6g} exceeds {DIALS_SG_P95_ABS_LIMIT_INV_A:.6g} A^-1"
        )
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
    return " ".join(f"{float(v): .10g}" for v in values)


def _choose_intensity(
    arrays: dict[str, np.ndarray],
    mode: str,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Return I, sigma, RLP-proxy, PEAK-proxy, and a plain-English mode note."""

    partiality = np.asarray(arrays["partiality"], dtype=float)
    lp = np.asarray(arrays["lp"], dtype=float)
    qe = np.asarray(arrays.get("qe", np.ones_like(lp)), dtype=float)
    if qe.shape != lp.shape:
        raise ValueError("DIALS qe column must have the same shape as lp")
    correction = np.full_like(lp, np.nan, dtype=float)
    valid_correction = np.isfinite(lp) & (lp > 0.0) & np.isfinite(qe) & (qe > 0.0)
    correction[valid_correction] = lp[valid_correction] / qe[valid_correction]
    peak = np.clip(partiality * 100.0, 0.0, 999.0)
    if mode == "sum_complete":
        factor = correction / np.maximum(partiality, eps)
        iobs = np.asarray(arrays["i_sum"], dtype=float) * factor
        sigma = np.sqrt(np.maximum(np.asarray(arrays["var_sum"], dtype=float), 0.0)) * np.abs(factor)
        note = "intensity.sum.value multiplied by (lp/qe)/partiality; PEAK=100*partiality and RLP=lp/qe"
        return iobs, sigma, correction, peak, note
    if mode == "sum_raw":
        iobs = np.asarray(arrays["i_sum"], dtype=float)
        sigma = np.sqrt(np.maximum(np.asarray(arrays["var_sum"], dtype=float), 0.0))
        note = "raw intensity.sum.value with no lp/partiality completion; PEAK=100*partiality"
        return iobs, sigma, np.ones_like(iobs), peak, note
    if mode == "scale":
        iobs = np.asarray(arrays["i_scale"], dtype=float)
        sigma = np.sqrt(np.maximum(np.asarray(arrays["var_scale"], dtype=float), 0.0))
        note = "DIALS intensity.scale.value from scaled.refl; PEAK still records 100*partiality"
        return iobs, sigma, np.ones_like(iobs), peak, note
    raise ValueError(f"unknown DIALS intensity mode {mode!r}")


def _detector_gain(metadata: dict[str, Any]) -> float:
    """Return the DIALS detector gain if it is available and physically usable."""

    detector = metadata.get("detector") or {}
    try:
        gain = float(detector.get("gain"))
    except Exception:
        return 1.0
    if not math.isfinite(gain) or gain <= 0:
        return 1.0
    return gain


def _finite_i_over_sigma(iobs: np.ndarray, sigma: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Return finite I/sigma values for summary and bridge policy decisions."""

    mask = keep & np.isfinite(iobs) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(mask):
        return np.asarray([], dtype=float)
    return np.asarray(iobs[mask], dtype=float) / np.asarray(sigma[mask], dtype=float)


def _bool_array(arrays: dict[str, np.ndarray], name: str, n: int, default: bool = False) -> np.ndarray:
    """Return a boolean DIALS flag array, tolerating small synthetic test arrays."""

    if name not in arrays:
        return np.full(n, default, dtype=bool)
    arr = np.asarray(arrays[name], dtype=bool)
    if len(arr) != n:
        return np.full(n, default, dtype=bool)
    return arr


def _dials_row_selection_keep(
    arrays: dict[str, np.ndarray],
    n: int,
    row_selection: str,
) -> np.ndarray:
    """Select integrated rows or the subset accepted by DIALS scaling.

    A ``scaled.refl`` table normally retains the original summation columns as
    well as the ``scaled`` flag.  ``row_selection='scaled'`` therefore permits
    GROND to use raw summation intensities while respecting the observations
    accepted by ``dials.scale`` (including delta-CC1/2 or other scaling
    exclusions).  This is intentionally independent of ``intensity_mode``.
    """

    selection = str(row_selection or "integrated").strip().lower()
    if selection not in {"integrated", "scaled"}:
        raise ValueError(f"unknown DIALS row selection {row_selection!r}")
    keep = np.ones(n, dtype=bool)
    integrated_sum = _bool_array(arrays, "integrated_sum", n, default=False)
    if np.any(integrated_sum):
        keep &= integrated_sum
    if selection == "scaled":
        scaled = _bool_array(arrays, "scaled", n, default=False)
        if not np.any(scaled):
            raise ValueError(
                "DIALS row-selection 'scaled' was requested, but the reflection "
                "table contains no rows flagged as accepted by scaling. Use a "
                "scaled.refl table or select 'integrated'."
            )
        keep &= scaled
    return keep


def _dials_sigma_preflight_keep(
    arrays: dict[str, np.ndarray],
    iobs: np.ndarray,
    sigma: np.ndarray,
    mode: str,
    row_selection: str = "integrated",
) -> np.ndarray:
    """Rows used to calibrate DIALS sigmas from equivalent-observation scatter."""

    n = len(iobs)
    partiality = np.asarray(arrays.get("partiality", np.ones(n, dtype=float)), dtype=float)
    if len(partiality) != n:
        partiality = np.ones(n, dtype=float)
    keep = np.isfinite(iobs) & np.isfinite(sigma) & (sigma > 0.0) & np.isfinite(partiality) & (partiality >= 0.90)
    effective_selection = "scaled" if str(mode).strip().lower() == "scale" else row_selection
    keep &= _dials_row_selection_keep(arrays, n, effective_selection)
    keep &= ~_bool_array(arrays, "dont_integrate", n)
    for flag in (
        "overloaded",
        "in_powder_ring",
        "background_includes_bad_pixels",
        "foreground_includes_bad_pixels",
        "includes_bad_pixels",
        "bad_shoebox",
        "failed_during_background_modelling",
        "failed_during_summation",
    ):
        keep &= ~_bool_array(arrays, flag, n)
    return keep


def _space_group_rotations(metadata: dict[str, Any]) -> list[np.ndarray]:
    """Return integer reciprocal-space rotation matrices from DIALS metadata."""

    try:
        import gemmi
    except Exception:
        return [np.eye(3, dtype=int)]

    name = str(metadata.get("space_group") or "").strip()
    sg = None
    if name:
        try:
            sg = gemmi.find_spacegroup_by_name(name)
        except Exception:
            sg = None
    if sg is None and name:
        match = re.search(r"No\.\s*(\d+)", name)
        if match:
            try:
                sg = gemmi.find_spacegroup_by_number(int(match.group(1)))
            except Exception:
                sg = None
    if sg is None:
        return [np.eye(3, dtype=int)]

    rotations: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for op in sg.operations():
        rot = np.rint(np.asarray(op.rot, dtype=float) / float(op.DEN)).astype(int)
        key = tuple(int(v) for v in rot.reshape(-1))
        if key not in seen:
            seen.add(key)
            rotations.append(rot)
    return rotations or [np.eye(3, dtype=int)]


def _canonical_reciprocal_hkl(hkl: np.ndarray, rotations: list[np.ndarray]) -> tuple[int, int, int]:
    """Canonical key for symmetry-equivalent observations, without adding Friedel mates."""

    row = np.asarray(hkl, dtype=int)
    keys: list[tuple[int, int, int]] = []
    for rot in rotations:
        for mat in (rot, np.linalg.inv(rot)):
            equiv = np.rint(row @ mat).astype(int)
            keys.append(tuple(int(v) for v in equiv))
    return min(set(keys))


def _fit_empirical_sigma_model(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    iobs: np.ndarray,
    sigma: np.ndarray,
    *,
    mode: str,
    row_selection: str = "integrated",
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Fit sigma_eff^2 = m^2 sigma_raw^2 + (a Iobs)^2 + c^2 from duplicate scatter."""

    hkl = np.asarray(arrays.get("hkl", np.empty((0, 3), dtype=int)), dtype=int)
    if hkl.ndim != 2 or hkl.shape[1] != 3 or len(hkl) != len(iobs):
        return None

    keep = _dials_sigma_preflight_keep(arrays, iobs, sigma, mode, row_selection=row_selection)
    if int(np.count_nonzero(keep)) < 50:
        return None

    rotations = _space_group_rotations(metadata)
    groups: dict[tuple[int, int, int], list[int]] = {}
    for idx in np.where(keep)[0]:
        groups.setdefault(_canonical_reciprocal_hkl(hkl[idx], rotations), []).append(int(idx))

    features: list[list[float]] = []
    target: list[float] = []
    usable_groups = 0
    for indices in groups.values():
        if len(indices) < 2:
            continue
        usable_groups += 1
        y = np.asarray([iobs[i] for i in indices], dtype=float)
        s = np.asarray([sigma[i] for i in indices], dtype=float)
        mean = float(np.mean(y))
        leverage = max(0.0, 1.0 - 1.0 / float(len(indices)))
        for yy, ss in zip(y, s, strict=True):
            features.append([float(ss * ss * leverage), float(yy * yy * leverage), float(leverage)])
            target.append(float((yy - mean) ** 2))

    dof = len(target) - usable_groups
    if usable_groups < 20 or dof < 30:
        return None

    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    coeff, _ = nnls(x, y)
    coeff[0] = max(float(coeff[0]), 1.0)
    coeff[1] = max(float(coeff[1]), 0.0)
    coeff[2] = max(float(coeff[2]), 0.0)

    fitted_var = coeff[0] * sigma * sigma + coeff[1] * iobs * iobs + coeff[2]
    sigma_eff = np.sqrt(np.maximum(fitted_var, sigma * sigma))

    fit_rows = np.any(x > 0, axis=1)
    equivalent_multiplier = math.sqrt(float(np.sum(x[fit_rows] @ coeff) / max(np.sum(x[fit_rows, 0]), 1e-30)))
    ratios = sigma_eff[keep] / np.maximum(sigma[keep], 1e-30)
    summary = {
        "status": "ok",
        "equation": "sigma_eff^2 = m^2*sigma_raw^2 + (a*Iobs)^2 + c^2",
        "source": "auto_empirical_replicate_variance",
        "groups": int(len(groups)),
        "usable_groups": int(usable_groups),
        "fit_rows": int(len(target)),
        "dof": int(dof),
        "space_group": str(metadata.get("space_group") or ""),
        "m": float(math.sqrt(coeff[0])),
        "a": float(math.sqrt(coeff[1])),
        "c": float(math.sqrt(coeff[2])),
        "coefficient_sigma2": float(coeff[0]),
        "coefficient_iobs2": float(coeff[1]),
        "coefficient_constant": float(coeff[2]),
        "equivalent_global_multiplier": float(equivalent_multiplier),
        "median_sigma_ratio": float(np.nanmedian(ratios)) if len(ratios) else None,
        "p90_sigma_ratio": float(np.nanpercentile(ratios, 90)) if len(ratios) else None,
    }
    return sigma_eff, summary


def _dials_sigma_multiplier(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    iobs: np.ndarray,
    sigma: np.ndarray,
    *,
    mode: str,
    row_selection: str = "integrated",
    sigma_mode: str,
    sigma_scale: float | None,
) -> tuple[float, dict[str, Any]]:
    """Choose a DIALS sigma multiplier and explain the decision.

    DIALS stores detector gain on the detector model, but rotation-data
    summation variances can still be too optimistic for electron diffraction.
    Auto mode first fits an empirical replicate-scatter variance model.  If
    there are not enough equivalent rows, it falls back to the older conservative
    detector-gain gross-overprecision gate.
    """

    requested = str(sigma_mode or "auto").strip().lower()
    if requested not in {"auto", "raw", "detector_gain"}:
        raise ValueError(f"unknown DIALS sigma mode {sigma_mode!r}")
    if sigma_scale is not None and float(sigma_scale) > 0:
        mult = float(sigma_scale)
        source = "explicit_sigma_scale"
    elif requested == "raw":
        mult = 1.0
        source = "raw"
    else:
        gain = _detector_gain(metadata)
        partiality = np.asarray(arrays["partiality"], dtype=float)
        keep = np.isfinite(partiality) & (partiality >= 0.90)
        ios_raw = _finite_i_over_sigma(iobs, sigma, keep)
        median_ios = float(np.nanmedian(ios_raw)) if len(ios_raw) else None
        p95_ios = float(np.nanpercentile(ios_raw, 95)) if len(ios_raw) else None
        if requested == "detector_gain":
            mult = math.sqrt(gain)
            source = "detector_gain"
        elif empirical := _fit_empirical_sigma_model(
            arrays, metadata, iobs, sigma, mode=mode, row_selection=row_selection
        ):
            sigma_eff, empirical_summary = empirical
            mult = float(empirical_summary["equivalent_global_multiplier"])
            source = str(empirical_summary["source"])
        elif gain > 1.01 and median_ios is not None and p95_ios is not None and median_ios > 50.0 and p95_ios > 250.0:
            mult = math.sqrt(gain)
            source = "auto_detector_gain_gross_overprecision"
        else:
            mult = 1.0
            source = "auto_raw"
    partiality = np.asarray(arrays.get("partiality", np.ones(len(iobs), dtype=float)), dtype=float)
    if len(partiality) != len(iobs):
        partiality = np.ones(len(iobs), dtype=float)
    keep = np.isfinite(partiality) & (partiality >= 0.90)
    sigma_after = sigma_eff if "sigma_eff" in locals() else sigma * mult
    ios_before = _finite_i_over_sigma(iobs, sigma, keep)
    ios_after = _finite_i_over_sigma(iobs, sigma_after, keep)
    summary = {
        "mode": requested,
        "source": source,
        "detector_gain": _detector_gain(metadata),
        "sigma_multiplier": float(mult),
        "explicit_sigma_scale": float(sigma_scale) if sigma_scale is not None and float(sigma_scale) > 0 else None,
        "i_over_sigma_before_median": float(np.nanmedian(ios_before)) if len(ios_before) else None,
        "i_over_sigma_after_median": float(np.nanmedian(ios_after)) if len(ios_after) else None,
        "i_over_sigma_before_p95": float(np.nanpercentile(ios_before, 95)) if len(ios_before) else None,
        "i_over_sigma_after_p95": float(np.nanpercentile(ios_after, 95)) if len(ios_after) else None,
        "note": (
            "DIALS sigma values are passed through unchanged; intensities are unchanged."
            if float(mult) == 1.0
            else "DIALS empirical sigma calibration is applied to bridge sigmas only; intensities are unchanged."
        ),
    }
    if "sigma_eff" in locals():
        summary["empirical_variance_model"] = empirical_summary
        summary["_sigma_effective"] = sigma_eff
    return float(mult), summary


def convert_dials_to_grond(
    refl: str | Path,
    expt: str | Path,
    out_dir: str | Path,
    *,
    intensity_mode: str = "sum_complete",
    row_selection: str = "integrated",
    sigma_mode: str = "raw",
    sigma_scale: float | None = None,
    dials_python: str = "",
    min_partiality_write: float = 1e-6,
    keep_dont_integrate: bool = False,
    drop_bad_flags: bool = True,
    structure: str | Path | None = None,
    baseline_hkl: str | Path | None = None,
    dataset_name: str = "dials",
    prep_out: str | Path | None = None,
) -> DialsBridgeOutput:
    """Write a GROND-ready XDS-like directory from DIALS `.refl` and `.expt` files.

    The default intensity mode uses DIALS summation intensities completed by
    `(lp/qe)/partiality`, while retaining partiality as the GROND `PEAK` quality
    field.  With GROND's default `--min-peak 90`, this keeps fully recorded rows
    for production but leaves lower-partiality rows available for diagnostics.
    ``row_selection='scaled'`` uses the raw summation values but retains only
    rows accepted by DIALS scaling. Production conversion drops severe DIALS
    bad-row flags by default and keeps
    the detector-model summation variance unchanged. Empirical or gain-based
    sigma inflation remains available as an explicit diagnostic policy.
    """

    refl_path = Path(refl).expanduser().resolve()
    expt_path = Path(expt).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    arrays, metadata = _run_dials_helper(refl_path, expt_path, dials_python=dials_python)
    mode = str(intensity_mode or "sum_complete").strip().lower()
    selection = str(row_selection or "integrated").strip().lower()
    iobs, sigma, rlp, peak, intensity_note = _choose_intensity(
        arrays, mode, eps=max(float(min_partiality_write), 1e-12)
    )
    sigma_multiplier, sigma_summary = _dials_sigma_multiplier(
        arrays,
        metadata,
        iobs,
        sigma,
        mode=mode,
        row_selection=selection,
        sigma_mode=sigma_mode,
        sigma_scale=sigma_scale,
    )
    sigma_effective = sigma_summary.pop("_sigma_effective", None)
    sigma = np.asarray(sigma_effective, dtype=float) if sigma_effective is not None else sigma * sigma_multiplier

    hkl = np.asarray(arrays["hkl"], dtype=int)
    xyzcal = np.asarray(arrays["xyzcal"], dtype=float)
    xyzobs = np.asarray(arrays["xyzobs"], dtype=float)
    partiality = np.asarray(arrays["partiality"], dtype=float)
    bbox = np.asarray(arrays.get("bbox", np.full((len(hkl), 6), -1, dtype=int)), dtype=float)
    bbox_z0 = bbox[:, 4] if bbox.ndim == 2 and bbox.shape[1] >= 6 else np.full(len(hkl), np.nan)
    bbox_z1 = bbox[:, 5] if bbox.ndim == 2 and bbox.shape[1] >= 6 else np.full(len(hkl), np.nan)
    bbox_z_width = bbox_z1 - bbox_z0
    panel = np.asarray(arrays["panel"], dtype=int)
    integrated_sum = np.asarray(arrays["integrated_sum"], dtype=bool)
    dont_integrate = np.asarray(arrays["dont_integrate"], dtype=bool)
    bad_flags = (
        np.asarray(arrays["overloaded"], dtype=bool)
        | np.asarray(arrays["in_powder_ring"], dtype=bool)
        | np.asarray(arrays["background_includes_bad_pixels"], dtype=bool)
        | np.asarray(arrays["foreground_includes_bad_pixels"], dtype=bool)
        | np.asarray(arrays["includes_bad_pixels"], dtype=bool)
        | np.asarray(arrays["bad_shoebox"], dtype=bool)
        | np.asarray(arrays["failed_during_background_modelling"], dtype=bool)
        | np.asarray(arrays["failed_during_summation"], dtype=bool)
    )
    finite = (
        np.all(np.isfinite(hkl), axis=1)
        & np.all(np.isfinite(xyzcal), axis=1)
        & np.all(np.isfinite(xyzobs), axis=1)
        & np.isfinite(iobs)
        & np.isfinite(sigma)
        & (sigma > 0)
        & np.isfinite(rlp)
        & (rlp > 0)
        & np.isfinite(partiality)
        & (partiality >= float(min_partiality_write))
    )
    effective_selection = "scaled" if mode == "scale" else selection
    finite &= _dials_row_selection_keep(arrays, len(hkl), effective_selection)
    if not keep_dont_integrate:
        finite &= ~dont_integrate
    if drop_bad_flags:
        finite &= ~bad_flags

    scan_array_range = _scan_array_range(metadata)
    starting_frame = int(scan_array_range[0])
    sg_validation = dials_sg_statistics(arrays, metadata, finite)
    summary_path = out / "dials_bridge_summary.json"
    if sg_validation.get("status") == "error":
        failure_summary = {
            "status": "failed",
            "source": "DIALS",
            "source_refl": str(refl_path),
            "source_expt": str(expt_path),
            "out_dir": str(out),
            "rows_read": int(len(hkl)),
            "rows_selected": int(np.count_nonzero(finite)),
            "geometry": {
                "scan_image_range": metadata.get("scan", {}).get("image_range", []),
                "scan_array_range": list(scan_array_range),
                "starting_frame": starting_frame,
            },
            "sg_validation": sg_validation,
        }
        summary_path.write_text(json.dumps(failure_summary, indent=2, allow_nan=False) + "\n")
    _require_valid_dials_geometry(sg_validation)

    a_scan, a_scan_contract = _a_scan_points_for_current_scan(metadata)
    scan_points_path = out / "DIALS_A_AT_SCAN_POINTS.txt"
    with scan_points_path.open("w") as f:
        for mat in a_scan:
            f.write(" ".join(f"{float(v): .12e}" for v in np.asarray(mat, dtype=float).reshape(-1)) + "\n")

    real = _real_space_axes_from_a(np.asarray(metadata["A_static"], dtype=float))
    s0 = np.asarray(metadata["beam"]["s0"], dtype=float)
    beam_dir = s0 / max(float(np.linalg.norm(s0)), 1e-30)
    scan_osc = metadata["scan"]["oscillation"]
    detector = metadata.get("detector", {})
    image_size = detector.get("image_size_px", [0, 0])
    pixel_size = detector.get("pixel_size_mm", [0.0, 0.0])

    hkl_path = out / "INTEGRATE.HKL"
    with hkl_path.open("w") as f:
        f.write("!OUTPUT_FILE=INTEGRATE.HKL\n")
        f.write("!GENERATED_BY=GROND import-dials\n")
        f.write("!GROND_SOURCE=DIALS\n")
        f.write(f"!DIALS_REFLECTIONS= {refl_path}\n")
        f.write(f"!DIALS_EXPERIMENTS= {expt_path}\n")
        f.write(f"!DIALS_INTENSITY_MODE= {mode}\n")
        f.write(f"!DIALS_INTENSITY_NOTE= {intensity_note}\n")
        f.write(f"!DIALS_SIGMA_MODE= {sigma_summary['mode']}\n")
        f.write(f"!DIALS_SIGMA_MULTIPLIER= {sigma_summary['sigma_multiplier']:.12g}\n")
        f.write(f"!DIALS_SIGMA_SOURCE= {sigma_summary['source']}\n")
        if sigma_summary.get("empirical_variance_model"):
            model_summary = sigma_summary["empirical_variance_model"]
            f.write(f"!DIALS_SIGMA_EQUATION= {model_summary['equation']}\n")
            f.write(
                "!DIALS_SIGMA_COEFFS= "
                f"m {model_summary['m']:.12g} "
                f"a {model_summary['a']:.12g} "
                f"c {model_summary['c']:.12g}\n"
            )
        f.write(f"!DIALS_ROWS_READ= {len(hkl)}\n")
        f.write(f"!DIALS_ROWS_WRITTEN= {int(np.count_nonzero(finite))}\n")
        f.write("!DIALS_PARTIALITY_AS_PEAK= TRUE\n")
        f.write("!DIALS_BBOX_Z_RANGE_AS_ALF0_BET0= TRUE\n")
        f.write(f"!DIALS_A_AT_SCAN_POINTS_FILE= {scan_points_path.name}\n")
        f.write(f"!DIALS_ROTATION_AXIS_DATUM= {_format_header_vector(metadata['goniometer']['rotation_axis_datum'])}\n")
        f.write(
            "!DIALS_FIXED_ROTATION= "
            f"{_format_header_vector(np.asarray(metadata['goniometer']['fixed_rotation']).reshape(-1))}\n"
        )
        f.write(
            "!DIALS_SETTING_ROTATION= "
            f"{_format_header_vector(np.asarray(metadata['goniometer']['setting_rotation']).reshape(-1))}\n"
        )
        image_range = metadata.get("scan", {}).get("image_range", [])
        if len(image_range) == 2:
            f.write(f"!DIALS_IMAGE_RANGE= {int(image_range[0])} {int(image_range[1])}\n")
        f.write(f"!DIALS_ARRAY_RANGE= {scan_array_range[0]} {scan_array_range[1]}\n")
        f.write(f"!X-RAY_WAVELENGTH= {float(metadata['beam']['wavelength_A']):.12g}\n")
        f.write(f"!STARTING_FRAME= {starting_frame}\n")
        f.write(f"!STARTING_ANGLE= {float(scan_osc[0]):.12g}\n")
        f.write(f"!OSCILLATION_RANGE= {float(scan_osc[1]):.12g}\n")
        f.write(f"!ROTATION_AXIS= {_format_header_vector(metadata['goniometer']['rotation_axis'])}\n")
        f.write(f"!INCIDENT_BEAM_DIRECTION= {_format_header_vector(beam_dir)}\n")
        f.write(f"!UNIT_CELL_CONSTANTS= {_format_header_vector(metadata['unit_cell'])}\n")
        f.write(f"!UNIT_CELL_A-AXIS= {_format_header_vector(real[:, 0])}\n")
        f.write(f"!UNIT_CELL_B-AXIS= {_format_header_vector(real[:, 1])}\n")
        f.write(f"!UNIT_CELL_C-AXIS= {_format_header_vector(real[:, 2])}\n")
        f.write(
            f"!NX= {int(image_size[0])} NY= {int(image_size[1])} QX= {float(pixel_size[0]):.12g} QY= {float(pixel_size[1]):.12g}\n"
        )
        f.write(
            "!COLUMN_CONTRACT= H K L IOBS SIGMA XCAL YCAL ZCAL RLP PEAK CORR MAXC XOBS YOBS ZOBS ALF0 BET0 ALF1 BET1 PSI ISEG\n"
        )
        f.write("!END_OF_HEADER\n")
        for i in np.where(finite)[0]:
            row = [
                int(hkl[i, 0]),
                int(hkl[i, 1]),
                int(hkl[i, 2]),
                float(iobs[i]),
                float(abs(sigma[i])),
                float(xyzcal[i, 0]),
                float(xyzcal[i, 1]),
                float(xyzcal[i, 2]),
                float(rlp[i]),
                float(peak[i]),
                100.0,
                float(bbox_z_width[i]) if np.isfinite(bbox_z_width[i]) else 0.0,
                float(xyzobs[i, 0]),
                float(xyzobs[i, 1]),
                float(xyzobs[i, 2]),
                float(bbox_z0[i]) if np.isfinite(bbox_z0[i]) else 0.0,
                float(bbox_z1[i]) if np.isfinite(bbox_z1[i]) else 0.0,
                0.0,
                0.0,
                0.0,
                int(panel[i]),
            ]
            f.write(
                f"{row[0]:5d} {row[1]:5d} {row[2]:5d} "
                f"{row[3]: .8e} {row[4]: .8e} "
                f"{row[5]: .5f} {row[6]: .5f} {row[7]: .5f} "
                f"{row[8]: .8e} {row[9]: .5f} {row[10]: .2f} {row[11]: .1f} "
                f"{row[12]: .5f} {row[13]: .5f} {row[14]: .5f} "
                f"{row[15]: .5f} {row[16]: .5f} {row[17]: .5f} {row[18]: .5f} "
                f"{row[19]: .5f} {row[20]:5d}\n"
            )

    summary = {
        "status": "ok",
        "source": "DIALS",
        "source_refl": str(refl_path),
        "source_expt": str(expt_path),
        "out_dir": str(out),
        "hkl": str(hkl_path),
        "scan_points": str(scan_points_path),
        "intensity_mode": mode,
        "row_selection": effective_selection,
        "intensity_note": intensity_note,
        "sigma_model": sigma_summary,
        "rows_read": int(len(hkl)),
        "rows_written": int(np.count_nonzero(finite)),
        "rows_excluded": int(len(hkl) - np.count_nonzero(finite)),
        "kept_dials_dont_integrate_rows": bool(keep_dont_integrate),
        "dropped_bad_dials_flags": bool(drop_bad_flags),
        "flags": {
            "integrated_sum": int(np.count_nonzero(integrated_sum)),
            "integrated_prf": int(np.count_nonzero(np.asarray(arrays["integrated_prf"], dtype=bool))),
            "dont_integrate": int(np.count_nonzero(dont_integrate)),
            "overloaded": int(np.count_nonzero(np.asarray(arrays["overloaded"], dtype=bool))),
            "in_powder_ring": int(np.count_nonzero(np.asarray(arrays["in_powder_ring"], dtype=bool))),
            "excluded_for_scaling": int(np.count_nonzero(np.asarray(arrays["excluded_for_scaling"], dtype=bool))),
            "bad_for_scaling": int(np.count_nonzero(np.asarray(arrays["bad_for_scaling"], dtype=bool))),
            "outlier_in_scaling": int(np.count_nonzero(np.asarray(arrays["outlier_in_scaling"], dtype=bool))),
            "scaled": int(np.count_nonzero(np.asarray(arrays["scaled"], dtype=bool))),
            "background_includes_bad_pixels": int(
                np.count_nonzero(np.asarray(arrays["background_includes_bad_pixels"], dtype=bool))
            ),
            "foreground_includes_bad_pixels": int(
                np.count_nonzero(np.asarray(arrays["foreground_includes_bad_pixels"], dtype=bool))
            ),
            "includes_bad_pixels": int(np.count_nonzero(np.asarray(arrays["includes_bad_pixels"], dtype=bool))),
            "bad_shoebox": int(np.count_nonzero(np.asarray(arrays["bad_shoebox"], dtype=bool))),
            "failed_during_background_modelling": int(
                np.count_nonzero(np.asarray(arrays["failed_during_background_modelling"], dtype=bool))
            ),
            "failed_during_summation": int(np.count_nonzero(np.asarray(arrays["failed_during_summation"], dtype=bool))),
            "bad_flags_dropped_by_default": int(np.count_nonzero(bad_flags)),
        },
        "partiality": {
            "min": float(np.nanmin(partiality)) if len(partiality) else None,
            "median": float(np.nanmedian(partiality)) if len(partiality) else None,
            "p90": float(np.nanpercentile(partiality, 90)) if len(partiality) else None,
            "rows_ge_0p90": int(np.count_nonzero(partiality >= 0.90)),
            "rows_ge_0p99": int(np.count_nonzero(partiality >= 0.99)),
        },
        "bbox_z": {
            "encoded_as": "ALF0=z_min_frame, BET0=z_max_frame, MAXC=z_width_frames",
            "available_rows": int(np.count_nonzero(np.isfinite(bbox_z_width) & (bbox_z_width > 0))),
            "width_min": float(np.nanmin(bbox_z_width)) if np.any(np.isfinite(bbox_z_width)) else None,
            "width_median": float(np.nanmedian(bbox_z_width)) if np.any(np.isfinite(bbox_z_width)) else None,
            "width_p90": float(np.nanpercentile(bbox_z_width, 90)) if np.any(np.isfinite(bbox_z_width)) else None,
            "width_max": float(np.nanmax(bbox_z_width)) if np.any(np.isfinite(bbox_z_width)) else None,
        },
        "geometry": {
            "space_group": metadata.get("space_group", ""),
            "unit_cell": metadata.get("unit_cell", []),
            "scan_points": int(len(a_scan)),
            "scan_oscillation": scan_osc,
            "scan_image_range": metadata.get("scan", {}).get("image_range", []),
            "scan_array_range": list(scan_array_range),
            "starting_frame": starting_frame,
            "a_scan_contract": a_scan_contract,
            "rotation_axis_lab": metadata.get("goniometer", {}).get("rotation_axis", []),
            "rotation_axis_datum": metadata.get("goniometer", {}).get("rotation_axis_datum", []),
            "fixed_rotation": metadata.get("goniometer", {}).get("fixed_rotation", []),
            "setting_rotation": metadata.get("goniometer", {}).get("setting_rotation", []),
        },
        "sg_validation": sg_validation,
        "recommended_grond_args": {
            "xds_dir": str(out),
            "hkl": str(hkl_path),
            "target_mode": "iobs",
            "min_peak": 90.0,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

    readme = out / "DIALS_BRIDGE_README.md"
    readme.write_text(
        "# GROND DIALS Bridge\n\n"
        "This directory was generated from DIALS `integrated.refl` and `integrated.expt` files. "
        "The `INTEGRATE.HKL` file is an XDS-like row contract for GROND, not a native XDS file.\n\n"
        f"- Source reflection table: `{refl_path}`\n"
        f"- Source experiment list: `{expt_path}`\n"
        f"- Intensity mode: `{mode}` ({intensity_note})\n"
        f"- Sigma mode: `{sigma_summary['mode']}`; multiplier `{sigma_summary['sigma_multiplier']:.6g}` ({sigma_summary['source']}).\n"
        f"- DIALS scan array range: `{list(scan_array_range)}`; GROND `STARTING_FRAME={starting_frame}`. Raw DIALS z coordinates are preserved.\n"
        "- `PEAK` is `100 * DIALS partiality`, so GROND's ordinary `--min-peak 90` means partiality >= 0.90.\n"
        "- `ALF0` and `BET0` store the DIALS reflection-table `bbox` z-range. GROND can use this as an auditable rocking-width estimate with `--sg-window-mode dials_bbox`.\n"
        "- Rows flagged `dont_integrate` or with severe DIALS bad-pixel/integration flags are dropped by default. Use the keep flags only for diagnostics.\n"
        "- `DIALS_A_AT_SCAN_POINTS.txt` carries scan-varying reciprocal A matrices. GROND interpolates these and applies the full DIALS `S R F A` goniometer geometry for Sg calculations.\n\n"
        "Run GROND with `--xds-dir` pointed at this generated directory, or use `GROND auto --dials-refl ... --dials-expt ...` to create it automatically under the run output directory.\n"
    )

    prep_path: Path | None = None
    if structure:
        from .config import DatasetSpec, PrepConfig, write_prep

        prep_path = Path(prep_out).expanduser().resolve() if prep_out else out / "grond_prep.json"
        prep = PrepConfig(
            structure=str(Path(structure).expanduser().resolve()),
            baseline_hkl=str(Path(baseline_hkl).expanduser().resolve()) if baseline_hkl else "",
            notes="Generated by GROND import-dials.",
            datasets=[
                DatasetSpec(
                    name=dataset_name,
                    hkl=str(hkl_path),
                    xds_dir=str(out),
                    min_peak=90.0,
                    target_mode="iobs",
                )
            ],
            metadata={"dials_bridge": summary},
        )
        write_prep(prep_path, prep)

    return DialsBridgeOutput(
        out_dir=out, hkl=hkl_path, scan_points=scan_points_path, summary=summary_path, prep=prep_path
    )
