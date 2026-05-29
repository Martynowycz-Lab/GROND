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
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
        "partiality": float_col(table, "partiality", 1.0),
        "zeta": float_col(table, "zeta"),
        "d": float_col(table, "d"),
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
        oscillation = [float(v) for v in scan.get_oscillation()]
    else:
        image_range = [1, int(len(table))]
        oscillation = [0.0, 1.0]

    a_static = np.asarray(crystal.get_A(), dtype=float).reshape(3, 3)
    n_scan_points = int(getattr(crystal, "num_scan_points", 0) or 0)
    a_scan_points = []
    if n_scan_points > 0 and hasattr(crystal, "get_A_at_scan_point"):
        for i in range(n_scan_points):
            a_scan_points.append(np.asarray(crystal.get_A_at_scan_point(i), dtype=float).reshape(3, 3).tolist())
    if not a_scan_points:
        a_scan_points = [a_static.tolist()]

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
            "rotation_axis": [float(v) for v in goniometer.get_rotation_axis()] if goniometer is not None else [1.0, 0.0, 0.0],
        },
        "scan": {
            "image_range": image_range,
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


def dials_sg_statistics(arrays: dict[str, np.ndarray], metadata: dict[str, Any], keep: np.ndarray) -> dict[str, Any]:
    """Compute excitation-error diagnostics from the generated GROND geometry."""

    if not np.any(keep):
        return {"status": "no_rows"}
    hkl = np.asarray(arrays["hkl"], dtype=float)
    xyzcal = np.asarray(arrays["xyzcal"], dtype=float)
    a_scan = np.asarray(metadata["A_scan_points"], dtype=float)
    axis = np.asarray(metadata["goniometer"]["rotation_axis"], dtype=float)
    s0 = np.asarray(metadata["beam"]["s0"], dtype=float)
    k = float(np.linalg.norm(s0))
    start_angle, osc_width = [float(v) for v in metadata["scan"]["oscillation"]]

    sg = []
    for i in np.where(keep)[0]:
        z = float(xyzcal[i, 2])
        omega = start_angle + z * osc_width
        a_mat = _interp_scan_a(a_scan, z)
        g = _rotation_matrix(axis, omega) @ a_mat @ hkl[i]
        value = (k * k - float(np.sum((s0 + g) ** 2))) / (2.0 * k)
        sg.append(value)
    sg_arr = np.asarray(sg, dtype=float)
    return {
        "status": "ok",
        "rows": int(len(sg_arr)),
        "median_sg_inv_A": float(np.nanmedian(sg_arr)),
        "median_abs_sg_inv_A": float(np.nanmedian(np.abs(sg_arr))),
        "p95_abs_sg_inv_A": float(np.nanpercentile(np.abs(sg_arr), 95)),
        "max_abs_sg_inv_A": float(np.nanmax(np.abs(sg_arr))),
    }


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
    peak = np.clip(partiality * 100.0, 0.0, 999.0)
    if mode == "sum_complete":
        factor = lp / np.maximum(partiality, eps)
        iobs = np.asarray(arrays["i_sum"], dtype=float) * factor
        sigma = np.sqrt(np.maximum(np.asarray(arrays["var_sum"], dtype=float), 0.0)) * np.abs(factor)
        note = "intensity.sum.value multiplied by lp/partiality; PEAK=100*partiality and RLP=lp"
        return iobs, sigma, lp, peak, note
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


def _dials_sigma_multiplier(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    iobs: np.ndarray,
    sigma: np.ndarray,
    *,
    mode: str,
    sigma_mode: str,
    sigma_scale: float | None,
) -> tuple[float, dict[str, Any]]:
    """Choose a DIALS sigma multiplier and explain the decision.

    DIALS stores detector gain on the detector model, while the summation
    variance path is controlled by the `integration.summation.detector_gain`
    integration parameter.  Electron-diffraction CBF reductions may therefore
    carry Poisson-like variances that have not been converted by the detector
    gain.  GROND keeps this as an explicit, auditable bridge policy rather than
    silently treating unrealistically precise DIALS rows as final weights.
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
        if requested == "detector_gain":
            mult = math.sqrt(gain)
            source = "detector_gain"
        elif gain > 1.01 and median_ios is not None and median_ios > 3.0:
            mult = math.sqrt(gain)
            source = "auto_detector_gain"
        else:
            mult = 1.0
            source = "auto_raw"
    partiality = np.asarray(arrays["partiality"], dtype=float)
    keep = np.isfinite(partiality) & (partiality >= 0.90)
    ios_before = _finite_i_over_sigma(iobs, sigma, keep)
    ios_after = _finite_i_over_sigma(iobs, sigma * mult, keep)
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
        "note": "DIALS detector-gain sigma scaling is applied to the bridge sigma only; intensities are unchanged.",
    }
    return float(mult), summary


def convert_dials_to_grond(
    refl: str | Path,
    expt: str | Path,
    out_dir: str | Path,
    *,
    intensity_mode: str = "sum_complete",
    sigma_mode: str = "auto",
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
    `lp/partiality`, while retaining partiality as the GROND `PEAK` quality
    field.  With GROND's default `--min-peak 90`, this keeps fully recorded rows
    for production but leaves lower-partiality rows available for diagnostics.
    Production conversion drops severe DIALS bad-row flags by default and uses
    an auditable sigma policy so detector-gain metadata can be applied to
    over-precise electron-diffraction summation variances.
    """

    refl_path = Path(refl).expanduser().resolve()
    expt_path = Path(expt).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    arrays, metadata = _run_dials_helper(refl_path, expt_path, dials_python=dials_python)
    mode = str(intensity_mode or "sum_complete").strip().lower()
    iobs, sigma, rlp, peak, intensity_note = _choose_intensity(arrays, mode, eps=max(float(min_partiality_write), 1e-12))
    sigma_multiplier, sigma_summary = _dials_sigma_multiplier(
        arrays,
        metadata,
        iobs,
        sigma,
        mode=mode,
        sigma_mode=sigma_mode,
        sigma_scale=sigma_scale,
    )
    sigma = sigma * sigma_multiplier

    hkl = np.asarray(arrays["hkl"], dtype=int)
    xyzcal = np.asarray(arrays["xyzcal"], dtype=float)
    xyzobs = np.asarray(arrays["xyzobs"], dtype=float)
    partiality = np.asarray(arrays["partiality"], dtype=float)
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
    if np.any(integrated_sum):
        finite &= integrated_sum
    if mode == "scale":
        finite &= np.asarray(arrays["scaled"], dtype=bool)
    if not keep_dont_integrate:
        finite &= ~dont_integrate
    if drop_bad_flags:
        finite &= ~bad_flags

    a_scan = np.asarray(metadata["A_scan_points"], dtype=float)
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
        f.write(f"!DIALS_ROWS_READ= {len(hkl)}\n")
        f.write(f"!DIALS_ROWS_WRITTEN= {int(np.count_nonzero(finite))}\n")
        f.write("!DIALS_PARTIALITY_AS_PEAK= TRUE\n")
        f.write(f"!DIALS_A_AT_SCAN_POINTS_FILE= {scan_points_path.name}\n")
        f.write(f"!X-RAY_WAVELENGTH= {float(metadata['beam']['wavelength_A']):.12g}\n")
        f.write("!STARTING_FRAME= 0\n")
        f.write(f"!STARTING_ANGLE= {float(scan_osc[0]):.12g}\n")
        f.write(f"!OSCILLATION_RANGE= {float(scan_osc[1]):.12g}\n")
        f.write(f"!ROTATION_AXIS= {_format_header_vector(metadata['goniometer']['rotation_axis'])}\n")
        f.write(f"!INCIDENT_BEAM_DIRECTION= {_format_header_vector(beam_dir)}\n")
        f.write(f"!UNIT_CELL_CONSTANTS= {_format_header_vector(metadata['unit_cell'])}\n")
        f.write(f"!UNIT_CELL_A-AXIS= {_format_header_vector(real[:, 0])}\n")
        f.write(f"!UNIT_CELL_B-AXIS= {_format_header_vector(real[:, 1])}\n")
        f.write(f"!UNIT_CELL_C-AXIS= {_format_header_vector(real[:, 2])}\n")
        f.write(f"!NX= {int(image_size[0])} NY= {int(image_size[1])} QX= {float(pixel_size[0]):.12g} QY= {float(pixel_size[1]):.12g}\n")
        f.write("!COLUMN_CONTRACT= H K L IOBS SIGMA XCAL YCAL ZCAL RLP PEAK CORR MAXC XOBS YOBS ZOBS ALF0 BET0 ALF1 BET1 PSI ISEG\n")
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
                0.0,
                float(xyzobs[i, 0]),
                float(xyzobs[i, 1]),
                float(xyzobs[i, 2]),
                0.0,
                0.0,
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
            "background_includes_bad_pixels": int(np.count_nonzero(np.asarray(arrays["background_includes_bad_pixels"], dtype=bool))),
            "foreground_includes_bad_pixels": int(np.count_nonzero(np.asarray(arrays["foreground_includes_bad_pixels"], dtype=bool))),
            "includes_bad_pixels": int(np.count_nonzero(np.asarray(arrays["includes_bad_pixels"], dtype=bool))),
            "bad_shoebox": int(np.count_nonzero(np.asarray(arrays["bad_shoebox"], dtype=bool))),
            "failed_during_background_modelling": int(np.count_nonzero(np.asarray(arrays["failed_during_background_modelling"], dtype=bool))),
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
        "geometry": {
            "space_group": metadata.get("space_group", ""),
            "unit_cell": metadata.get("unit_cell", []),
            "scan_points": int(len(a_scan)),
            "scan_oscillation": scan_osc,
            "scan_image_range": metadata.get("scan", {}).get("image_range", []),
        },
        "sg_validation": dials_sg_statistics(arrays, metadata, finite),
        "recommended_grond_args": {
            "xds_dir": str(out),
            "hkl": str(hkl_path),
            "target_mode": "iobs",
            "min_peak": 90.0,
        },
    }
    summary_path = out / "dials_bridge_summary.json"
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
        "- `PEAK` is `100 * DIALS partiality`, so GROND's ordinary `--min-peak 90` means partiality >= 0.90.\n"
        "- Rows flagged `dont_integrate` or with severe DIALS bad-pixel/integration flags are dropped by default. Use the keep flags only for diagnostics.\n"
        "- `DIALS_A_AT_SCAN_POINTS.txt` carries scan-varying reciprocal A matrices. GROND interpolates these and then applies the goniometer rotation for Sg calculations.\n\n"
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

    return DialsBridgeOutput(out_dir=out, hkl=hkl_path, scan_points=scan_points_path, summary=summary_path, prep=prep_path)
