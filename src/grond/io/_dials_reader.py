"""Read one DIALS integrated.refl/integrated.expt pair inside the DIALS Python.

``dials_bridge`` runs this file's text with ``python -c``, passing the .refl, .expt, output
.npz and output .json paths. It writes the reflection columns and flags as NumPy arrays
(.npz) and the crystal, beam, goniometer, scan, detector and profile models as JSON. It
imports nothing from GROND, so the DIALS environment can differ from GROND's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from dials.array_family import flex
from dxtbx.model.experiment_list import ExperimentListFactory


def bool_flags(table, name):
    """Return a DIALS reflection flag as a boolean array, all False when DIALS has no such flag."""

    flag = getattr(table.flags, name, None)
    if flag is None:
        return np.zeros(len(table), dtype=bool)
    return np.asarray(list(table.get_flags(flag)), dtype=bool)


def float_col(table, name, default=np.nan):
    """Return a reflection-table column as a float array, filled with ``default`` when absent."""

    if name not in table:
        return np.full(len(table), default, dtype=float)
    return np.asarray(list(table[name]), dtype=float)


def int_col(table, name, default=0):
    """Return a reflection-table column as an int32 array, filled with ``default`` when absent."""

    if name not in table:
        return np.full(len(table), default, dtype=np.int32)
    return np.asarray(list(table[name]), dtype=np.int32)


def vec3_col(table, name, default=np.nan):
    """Return a 3-vector reflection-table column as an (n, 3) float array, filled with ``default`` when absent."""

    if name not in table:
        return np.full((len(table), 3), default, dtype=float)
    return np.asarray(list(table[name]), dtype=float)


def bbox_col(table):
    """Return the shoebox bounding boxes as an (n, 6) int32 array, or -1 everywhere when absent."""

    if "bbox" not in table:
        return np.full((len(table), 6), -1, dtype=np.int32)
    return np.asarray(list(table["bbox"]), dtype=np.int32)


def first_panel_metadata(detector):
    """Describe the panel count and the first panel's geometry (mm and pixels), distance and gain."""

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


def profile_model_metadata(profile):
    """Return the profile model's class name and its ``to_dict()`` parameters when available."""

    if profile is None:
        return {"model": None, "parameters": None}
    model = type(profile).__name__
    parameters = None
    if hasattr(profile, "to_dict"):
        try:
            parameters = profile.to_dict()
        except Exception:
            parameters = None
    return {"model": model, "parameters": parameters}


def main():
    """Convert the reflection table and single experiment named on the command line.

    Missing observed centroids (xyzobs.px.value) are replaced by the calculated ones
    (xyzcal.px). Several experiments or a scan-varying goniometer setting rotation stop the
    conversion.
    """

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
        "inverse_scale_factor": float_col(table, "inverse_scale_factor"),
        "inverse_scale_factor_variance": float_col(
            table, "inverse_scale_factor_variance"
        ),
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
        "experiment_identifier": str(getattr(exp, "identifier", "") or ""),
        "space_group": str(crystal.get_space_group().info()),
        "space_group_number": int(crystal.get_space_group().type().number()),
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
        "profile": profile_model_metadata(getattr(exp, "profile", None)),
        "frame_sliced_shoebox": {
            "available": bool("frame_sliced_shoebox" in table),
            "used_for_intensity": False,
            "status": "recorded_for_future_rocking_profile_import",
        },
    }
    out_json.write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
