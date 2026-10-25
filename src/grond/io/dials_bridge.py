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

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from grond.io.dials_geometry import (
    _a_scan_points_for_current_scan as _a_scan_points_for_current_scan,
    _format_header_vector as _format_header_vector,
    _real_space_axes_from_a as _real_space_axes_from_a,
    _require_valid_dials_geometry as _require_valid_dials_geometry,
    _rotation_matrix as _rotation_matrix,
    _scan_array_range as _scan_array_range,
    dials_sg_statistics as dials_sg_statistics,
)
from grond.io.dials_selection import (
    _choose_intensity as _choose_intensity,
    _dials_row_selection_keep as _dials_row_selection_keep,
)
from grond.io.dials_sigma import (
    _dials_sigma_multiplier as _dials_sigma_multiplier,
)

DIALS_HELPER_CODE = Path(__file__).with_name("_dials_reader.py").read_text()


@dataclass
class DialsBridgeOutput:
    """Paths and summary information written by the DIALS bridge."""

    out_dir: Path
    hkl: Path
    scan_points: Path
    summary: Path
    prep: Path | None = None


def _imports_dials(python: str) -> bool:
    """True when ``python`` can import dials and dxtbx."""

    try:
        return subprocess.run([python, "-c", "import dials, dxtbx"], capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def find_dials_python() -> str:
    """Find a Python that can import DIALS: this one, else the DIALS whose commands are on PATH.

    DIALS 3.30 and later put forwarder scripts in ``<install>/dialsbin`` (on PATH) and no longer ship
    ``dials.python``; their Python is ``<install>/bin/python``, and ``dialsbin/cctbx.python`` still exists.
    Older DIALS and CCP4 have ``dials.python`` and ``libtbx.python`` beside the ``dials.*`` commands.
    """

    if importlib.util.find_spec("dials") is not None and importlib.util.find_spec("dxtbx") is not None:
        return sys.executable
    candidates = []
    for command in ("dials.version", "dials.import", "dials.integrate"):
        found = shutil.which(command)
        if found:
            here = Path(found).parent
            candidates += [here / name for name in ("dials.python", "cctbx.python", "libtbx.python")]
            candidates += [here.parent / "bin" / "python", here / "python"]
            break
    candidates += [Path(found) for name in ("dials.python", "cctbx.python", "libtbx.python", "ccp4-python")
                   if (found := shutil.which(name))]
    seen = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        seen.add(candidate)
        if _imports_dials(str(candidate)):
            return str(candidate)
    return ""


def _run_dials_helper(refl: Path, expt: Path, dials_python: str = "") -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract DIALS reflection and experiment arrays using a DIALS-capable Python."""

    for label, path in (("reflection table", refl), ("experiment list", expt)):
        if not path.is_file():
            raise FileNotFoundError(f"DIALS {label} does not exist: {path}")
    dials_python = dials_python or find_dials_python()
    if not dials_python:
        raise FileNotFoundError(
            "GROND reads .refl and .expt files with DIALS, and it found no DIALS on PATH. Install DIALS and put its "
            "commands on PATH (source its setpaths.sh), or give --dials-python <DIALS>/bin/python."
        )
    requested_python = str(dials_python)
    explicit_path = Path(os.path.abspath(Path(requested_python).expanduser())) if os.sep in requested_python else None
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise FileNotFoundError(f"DIALS Python executable does not exist: {explicit_path}")
        if not os.access(explicit_path, os.X_OK):
            raise PermissionError(f"DIALS Python is not executable: {explicit_path}")
        # Preserve venv/DIALS launcher symlinks.  Resolving them can replace the
        # requested environment interpreter with its base Python and thereby
        # lose the environment's NumPy, DIALS, and dxtbx installations.
        python = str(explicit_path)
    else:
        resolved_python = shutil.which(requested_python)
        if not resolved_python:
            raise FileNotFoundError(
                f"DIALS Python executable was not found on PATH: {requested_python}. "
                "Pass --dials-python <DIALS>/bin/python."
            )
        python = resolved_python
    with tempfile.TemporaryDirectory(prefix="grond_dials_") as tmp:
        tmp_path = Path(tmp)
        out_npz = tmp_path / "dials_arrays.npz"
        out_json = tmp_path / "dials_metadata.json"
        cmd = [python, "-c", DIALS_HELPER_CODE, str(refl), str(expt), str(out_npz), str(out_json)]
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True)
        except OSError as exc:
            raise RuntimeError(f"Could not start DIALS Python {python}: {exc}") from None
        if proc.returncode != 0:
            hint = (
                "Could not read the DIALS files. Give --dials-python <DIALS>/bin/python, a Python that can "
                "import dials and dxtbx."
            )
            stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-30:])
            display = [python, "-c", "<embedded DIALS bridge helper>", str(refl), str(expt)]
            raise RuntimeError(
                f"{hint}\nCommand: {' '.join(display)}\nSTDERR (last {len(stderr_tail.splitlines())} lines):\n{stderr_tail}"
            ) from None
        with np.load(out_npz, allow_pickle=False) as loaded:
            arrays = {name: np.array(loaded[name]) for name in loaded.files}
        metadata = json.loads(out_json.read_text())
    return arrays, metadata


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
    `(lp/qe)/partiality`, with sigmas multiplied by the same factor. Partiality
    remains the GROND `PEAK` quality field. With GROND's default `--min-peak 90`, this keeps fully recorded rows
    for production but leaves lower-partiality rows available for diagnostics.
    ``row_selection='scaled'`` additionally requires DIALS scaling acceptance;
    it does not change the intensity mode. Production conversion drops severe
    DIALS bad-row flags and applies no additional variance recalibration. Empirical or gain-based
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

    a_scan_contract = dict(sg_validation.get("a_scan_contract") or {})
    a_scan, _ = _a_scan_points_for_current_scan(
        metadata,
        source_index_offset=int(a_scan_contract.get("source_scan_point_origin", 0)),
    )
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

    inverse_scale = np.asarray(arrays.get("inverse_scale_factor", np.full(len(hkl), np.nan)), dtype=float)
    inverse_scale_variance = np.asarray(
        arrays.get("inverse_scale_factor_variance", np.full(len(hkl), np.nan)),
        dtype=float,
    )
    inverse_scale_available = finite & np.isfinite(inverse_scale) & (inverse_scale > 0.0)
    inverse_scale_control_path = out / "DIALS_INVERSE_SCALE_CONTROL.npz"
    inverse_scale_control = {
        "status": "unavailable",
        "note": (
            "DIALS inverse-scale values are retained only as a fixed control or prior; "
            "GROND does not silently replace them with a fitted detector field."
        ),
        "available_rows": int(np.count_nonzero(inverse_scale_available)),
        "written_rows": int(np.count_nonzero(finite)),
    }
    if np.any(inverse_scale_available):
        selected = np.where(finite)[0]
        np.savez_compressed(
            inverse_scale_control_path,
            source_row_index=selected.astype(np.int64),
            hkl=np.asarray(hkl[selected], dtype=np.int32),
            inverse_scale_factor=np.asarray(inverse_scale[selected], dtype=float),
            inverse_scale_factor_variance=np.asarray(inverse_scale_variance[selected], dtype=float),
            xyzcal=np.asarray(xyzcal[selected], dtype=float),
            panel=np.asarray(panel[selected], dtype=np.int32),
        )
        available_values = inverse_scale[inverse_scale_available]
        inverse_scale_control.update(
            {
                "status": "recorded_fixed_control",
                "path": str(inverse_scale_control_path),
                "minimum": float(np.min(available_values)),
                "median": float(np.median(available_values)),
                "maximum": float(np.max(available_values)),
            }
        )

    hkl_path = out / "INTEGRATE.HKL"
    with hkl_path.open("w") as f:
        f.write("!OUTPUT_FILE=INTEGRATE.HKL\n")
        f.write("!GENERATED_BY=GROND import-dials\n")
        f.write("!GROND_SOURCE=DIALS\n")
        f.write(f"!DIALS_REFLECTIONS= {refl_path}\n")
        f.write(f"!DIALS_EXPERIMENTS= {expt_path}\n")
        f.write("!DIALS_INDEXING_BASIS= miller_index_from_source_reflection_table\n")
        if str(metadata.get("experiment_identifier") or ""):
            f.write(f"!DIALS_EXPERIMENT_IDENTIFIER= {str(metadata['experiment_identifier'])}\n")
        f.write(f"!DIALS_INTENSITY_MODE= {mode}\n")
        f.write(f"!DIALS_INTENSITY_NOTE= {intensity_note}\n")
        profile = metadata.get("profile") or {}
        if profile.get("model"):
            f.write(f"!DIALS_PROFILE_MODEL= {profile['model']}\n")
        if profile.get("parameters") is not None:
            compact_profile = json.dumps(profile["parameters"], sort_keys=True, separators=(",", ":"))
            f.write(f"!DIALS_PROFILE_PARAMETERS_JSON= {compact_profile}\n")
        frame_sliced = metadata.get("frame_sliced_shoebox") or {}
        f.write(f"!DIALS_FRAME_SLICED_SHOEBOX_AVAILABLE= {'TRUE' if frame_sliced.get('available') else 'FALSE'}\n")
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
        if inverse_scale_control.get("status") == "recorded_fixed_control":
            f.write(f"!DIALS_INVERSE_SCALE_CONTROL_FILE= {inverse_scale_control_path.name}\n")
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
        if metadata.get("space_group_number") is not None:
            f.write(f"!SPACE_GROUP_NUMBER= {int(metadata['space_group_number'])}\n")
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
        "dials_inverse_scale_control": inverse_scale_control,
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
            "space_group_number": metadata.get("space_group_number"),
            "experiment_identifier": metadata.get("experiment_identifier", ""),
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
            "profile": metadata.get("profile", {}),
            "frame_sliced_shoebox": metadata.get("frame_sliced_shoebox", {}),
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
        "- `ALF0` and `BET0` store the DIALS reflection-table `bbox` z-range. With the default `--sg-window-mode auto` GROND uses it as the per-row rocking-window width (capped at 0.03 1/A by default; XDS rows use the fixed 0.010 1/A window).\n"
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
