"""DIALS-runtime helper for GROND's PETS interoperability exporter.

This file is launched as a script by a Python interpreter that can import
``dials`` and ``dxtbx``.  It deliberately has no imports from GROND so the
installed GROND environment and the selected DIALS environment may remain
independent.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace


def _bool_flag(table, name):
    from dials.array_family import flex

    flag = getattr(table.flags, name, None)
    if flag is None:
        return flex.bool(len(table), False)
    return table.get_flags(flag)


def _static_cell_projection(experiment, *, policy, max_length_fraction, max_angle_degrees):
    """Project scan-point orientations onto the experiment's static cell.

    DIALS' PETS writer supports scan-varying orientation but the PETS contract
    has only one unit cell.  Keeping U(scan) and replacing B(scan) by the
    experiment's static B is the least-lossy representable projection.
    """

    import numpy as np
    from scitbx import matrix

    crystal = experiment.crystal
    n_scan = int(getattr(crystal, "num_scan_points", 0) or 0)
    expected_scan_points = int(experiment.scan.get_num_images()) + 1
    source_indices = list(range(n_scan))
    if n_scan and n_scan != expected_scan_points:
        start, stop = experiment.scan.get_array_range()
        candidate = list(range(int(start), int(stop) + 1))
        if len(candidate) != expected_scan_points or max(candidate) >= n_scan:
            raise ValueError(
                "DIALS scan-varying crystal cannot be aligned to the selected scan: "
                f"crystal scan points={n_scan}, expected={expected_scan_points}, "
                f"scan array range=({start}, {stop})"
            )
        source_indices = candidate
    reference = np.asarray(crystal.get_unit_cell().parameters(), dtype=float)
    result = {
        "policy": str(policy),
        "scan_points": n_scan,
        "exported_scan_points": len(source_indices),
        "source_scan_point_range": (
            [source_indices[0], source_indices[-1]] if source_indices else None
        ),
        "projected": False,
        "reference_cell": reference.tolist(),
        "max_length_fraction": 0.0,
        "max_angle_degrees": 0.0,
    }
    if n_scan <= 0:
        return result

    cells = np.asarray(
        [crystal.get_unit_cell_at_scan_point(i).parameters() for i in source_indices],
        dtype=float,
    )
    length_fraction = float(
        np.max(np.abs(cells[:, :3] - reference[None, :3]) / reference[None, :3])
    )
    angle_degrees = float(np.max(np.abs(cells[:, 3:] - reference[None, 3:])))
    varying = not np.allclose(cells, reference[None, :], rtol=1e-7, atol=1e-7)
    result.update(
        {
            "varying": bool(varying),
            "max_length_fraction": length_fraction,
            "max_angle_degrees": angle_degrees,
            "first_cell": cells[0].tolist(),
            "last_cell": cells[-1].tolist(),
        }
    )
    if not varying:
        return result
    if policy == "error":
        raise ValueError(
            "DIALS experiment has a scan-varying unit cell, which cif_pets cannot encode; "
            "rerun with --scan-varying-cell project to retain U(scan) on one declared cell"
        )
    if length_fraction > float(max_length_fraction) or angle_degrees > float(max_angle_degrees):
        raise ValueError(
            "scan-varying cell exceeds the declared PETS projection limit: "
            f"max length change={length_fraction:.4%} (limit {max_length_fraction:.4%}), "
            f"max angle change={angle_degrees:.4f} deg (limit {max_angle_degrees:.4f} deg)"
        )

    static_b = matrix.sqr(crystal.get_B())
    scan_a = [
        matrix.sqr(crystal.get_U_at_scan_point(i)) * static_b for i in source_indices
    ]
    crystal.set_A_at_scan_points(scan_a)
    result["projected"] = True
    result["projection"] = "A_projected(scan)=U_source(scan)*B_static"
    return result


def _pets_params(prefix, *, intensity, partiality, excitation_error, n_merged, step):
    return SimpleNamespace(
        intensity=[str(intensity)],
        pets=SimpleNamespace(
            filename_prefix=str(prefix),
            id=None,
            partiality_cutoff=float(partiality),
            flag_filtering=False,
            virtual_frame=SimpleNamespace(
                excitation_error_cutoff=float(excitation_error),
                n_merged=int(n_merged),
                step=int(step),
            ),
        ),
    )


def _space_group_metadata(crystal):
    space_group = crystal.get_space_group()
    return {
        "source_space_group_symbol": str(space_group.info()),
        "source_space_group_number": int(space_group.type().number()),
    }


def _severe_bad_flags(table):
    from dials.array_family import flex

    bad = flex.bool(len(table), False)
    for name in (
        "dont_integrate",
        "overloaded",
        "in_powder_ring",
        "background_includes_bad_pixels",
        "foreground_includes_bad_pixels",
        "includes_bad_pixels",
        "bad_shoebox",
        "failed_during_background_modelling",
        "failed_during_summation",
    ):
        bad |= _bool_flag(table, name)
    return bad


def _export_dials(config):
    from dials.array_family import flex
    from dials.util.export_pets import PETSOutput
    from dxtbx.model.experiment_list import ExperimentList, ExperimentListFactory

    refl_path = Path(config["refl"])
    expt_path = Path(config["expt"])
    base_experiments = ExperimentListFactory.from_json_file(str(expt_path), check_format=False)
    selected_ids = config.get("experiment_ids")
    if selected_ids is None:
        selected_ids = list(range(len(base_experiments)))
    selected_ids = [int(value) for value in selected_ids]
    if not selected_ids:
        raise ValueError("no DIALS experiments selected")
    if min(selected_ids) < 0 or max(selected_ids) >= len(base_experiments):
        raise ValueError(
            f"DIALS experiment selection {selected_ids} is outside 0..{len(base_experiments) - 1}"
        )

    outputs = []
    for experiment_id in selected_ids:
        # PETSOutput rotates models and therefore must receive a fresh copy for
        # every exported experiment.
        experiments = ExperimentListFactory.from_json_file(str(expt_path), check_format=False)
        table = flex.reflection_table.from_file(str(refl_path))
        source_rows = int(len(table))
        experiment = experiments[experiment_id]
        keep = table["id"] == int(experiment_id)
        intensity = str(config["intensity"])
        integrated_flag = "integrated_sum" if intensity == "sum" else "integrated_prf"
        if config["row_selection"] == "scaled":
            scaled = _bool_flag(table, "scaled")
            if scaled.count(True) == 0:
                raise ValueError("--row-selection scaled requested but no DIALS rows are flagged scaled")
            keep &= scaled
        else:
            keep &= _bool_flag(table, integrated_flag)
        intensity_column = "intensity.sum" if intensity == "sum" else "intensity.prf"
        value_column = f"{intensity_column}.value"
        variance_column = f"{intensity_column}.variance"
        if value_column not in table or variance_column not in table:
            raise ValueError(
                f"DIALS experiment {experiment_id} is missing {value_column} or {variance_column}"
            )
        keep &= flex.bool([math.isfinite(float(value)) for value in table[value_column]])
        keep &= flex.bool(
            [math.isfinite(float(value)) for value in table[variance_column]]
        )
        keep &= table[variance_column] > 0.0
        if not bool(config["keep_bad_flags"]):
            keep &= ~_severe_bad_flags(table)
        table = table.select(keep)
        if len(table) == 0:
            raise ValueError(f"DIALS experiment {experiment_id} has no rows after selection")
        table["id"] = flex.int(len(table), 0)
        partiality_rows = int(
            (table["partiality"] >= float(config["min_partiality"])).count(True)
        )

        one_experiment = ExperimentList()
        one_experiment.append(experiment)
        cell_contract = _static_cell_projection(
            experiment,
            policy=str(config["scan_varying_cell"]),
            max_length_fraction=float(config["max_cell_length_fraction"]),
            max_angle_degrees=float(config["max_cell_angle_degrees"]),
        )
        # Rebuild s1 from the selected experiment and xyzcal centroids even
        # when the table carries a stale pre-refinement s1 column.  DIALS'
        # PETS writer verifies this vector before/after its coordinate-frame
        # rotation at 1e-10 and correctly rejects mixed geometry.
        table.map_centroids_to_reciprocal_space(one_experiment, calculated=True)

        suffix = "" if len(selected_ids) == 1 else f"_{experiment_id:03d}"
        prefix = Path(config["out_dir"]) / f"{config['dataset_name']}{suffix}"
        params = _pets_params(
            prefix,
            intensity=intensity,
            partiality=config["min_partiality"],
            excitation_error=config["excitation_error_cutoff"],
            n_merged=config["n_merged"],
            step=config["step"],
        )
        PETSOutput(one_experiment, [table], params).write_dyn_cif_pets()
        output = {
            "path": str(prefix) + ".cif_pets",
            "experiment_id": experiment_id,
            "experiment_identifier": str(getattr(experiment, "identifier", "") or ""),
            "source_rows": source_rows,
            "selected_rows_before_partiality": int(len(table)),
            "source_observations_at_partiality": partiality_rows,
            "static_cell_contract": cell_contract,
        }
        output.update(_space_group_metadata(experiment.crystal))
        outputs.append(output)
    return {
        "source_kind": "DIALS",
        "source_refl": str(refl_path.resolve()),
        "source_expt": str(expt_path.resolve()),
        "experiment_count": int(len(base_experiments)),
        "outputs": outputs,
        "observation_convention": (
            "DIALS intensity.sum.value and sqrt(intensity.sum.variance)"
            if config["intensity"] == "sum"
            else "DIALS intensity.prf.value and sqrt(intensity.prf.variance)"
        ),
    }


def _read_xds_profile_fitting(xds_inp, integrate_hkl):
    pattern = re.compile(r"(?:^|\s)PROFILE_FITTING\s*=\s*([^\s!]+)", re.IGNORECASE)
    values = []
    for path in (Path(xds_inp), Path(integrate_hkl)):
        for line in path.read_text(errors="replace").splitlines():
            line = line.split("!", 1)[0] if path == Path(xds_inp) else line.lstrip("!")
            match = pattern.search(line)
            if match:
                values.append(match.group(1).strip().upper())
    if not values:
        return None
    normalized = {value in {"FALSE", "F", "0", "NO"} for value in values}
    if len(normalized) != 1:
        raise ValueError(f"XDS PROFILE_FITTING declarations disagree: {values}")
    return not normalized.pop()


def _xds_geometry_experiment(xds_dir):
    """Read XDS geometry, working around old dxtbx zero-thickness EIGER input."""

    from dxtbx.model.experiment_list import ExperimentListFactory

    xds_dir = Path(xds_dir)
    xds_inp = xds_dir / "XDS.INP"
    choices = [xds_dir / name for name in ("GXPARM.XDS", "XDS_ASCII.HKL", "XPARM.XDS", "INTEGRATE.HKL")]
    geometry = next((path for path in choices if path.exists()), None)
    if geometry is None:
        raise ValueError("XDS export requires GXPARM.XDS, XDS_ASCII.HKL, XPARM.XDS, or INTEGRATE.HKL")

    text = xds_inp.read_text(errors="replace")
    sensor_pattern = re.compile(r"(^|\s)SENSOR_THICKNESS\s*=\s*[^\s!]+", re.IGNORECASE)
    workaround = False
    if re.search(r"(^|\s)DETECTOR\s*=\s*(EIGER|PILATUS)", text, re.IGNORECASE):
        match = sensor_pattern.search(text)
        if match is None:
            text += "\nSENSOR_THICKNESS=0.000001\n"
            workaround = True
        else:
            try:
                current = float(match.group(0).split("=", 1)[1])
            except ValueError:
                current = 0.0
            if current <= 0.0:
                text = sensor_pattern.sub(" SENSOR_THICKNESS=0.000001", text)
                workaround = True

    with tempfile.TemporaryDirectory(prefix="grond_xds_pets_") as tmp:
        temporary_inp = Path(tmp) / "XDS.INP"
        temporary_inp.write_text(text)
        experiments = ExperimentListFactory.from_xds(str(temporary_inp), str(geometry))
    if len(experiments) != 1:
        raise ValueError(f"XDS geometry produced {len(experiments)} experiments; expected one")
    return experiments, geometry, workaround


def _derive_xds_reindex(experiment, handle):
    import numpy as np
    from rstbx.cftbx.coordinate_frame_helpers import align_reference_frame
    from scitbx import matrix

    d_a = matrix.sqr(experiment.crystal.get_A())
    d_beam = matrix.col(experiment.beam.get_sample_to_source_direction())
    d_axis = matrix.col(experiment.goniometer.get_rotation_axis())
    normal = d_beam.cross(d_axis)
    x_beam = matrix.col(handle.beam_vector).normalize()
    x_axis = matrix.col(handle.rotation_axis).normalize()
    rotation = align_reference_frame(-x_beam, d_beam, x_axis, normal.cross(d_beam))
    x_a = matrix.sqr(
        handle.unit_cell_a_axis + handle.unit_cell_b_axis + handle.unit_cell_c_axis
    ).inverse()
    continuous = np.asarray((d_a.inverse() * (rotation * x_a)).elems, dtype=float).reshape(3, 3)
    rounded = np.rint(continuous).astype(int)
    error = float(np.max(np.abs(continuous - rounded)))
    determinant = int(round(np.linalg.det(rounded)))
    if error > 0.10 or abs(determinant) != 1:
        raise ValueError(
            "XDS-to-DIALS indexing transform is not a unimodular integer matrix: "
            f"max rounding error={error:.4f}, determinant={determinant}"
        )
    return rounded, error


def _read_xds_ascii_rows(path):
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if not line.strip() or line.startswith("!"):
            continue
        fields = line.split()
        if len(fields) < 12:
            continue
        try:
            rows.append([float(value) for value in fields[:12]])
        except ValueError:
            continue
    return rows


def _xds_correct_acceptance(handle, transformed_hkl, xds_ascii):
    """Match CORRECT's unmerged accepted rows back to INTEGRATE rows."""

    import numpy as np

    corrected = _read_xds_ascii_rows(xds_ascii)
    if not corrected:
        raise ValueError(f"no unmerged reflection rows found in {xds_ascii}")
    xyz = np.asarray(handle.xyzcal, dtype=float)
    lookup = {}
    for index, point in enumerate(xyz):
        lookup.setdefault(tuple(np.round(point, 1)), []).append(index)
    keep = np.zeros(len(xyz), dtype=bool)
    unmatched = 0
    mismatched = 0
    for row in corrected:
        key = tuple(np.round(np.asarray(row[5:8], dtype=float), 1))
        candidates = lookup.get(key, [])
        target = np.asarray(row[:3], dtype=int)
        exact = [index for index in candidates if np.array_equal(transformed_hkl[index], target)]
        if len(exact) == 1:
            if float(row[4]) > 0.0:
                keep[exact[0]] = True
        elif not candidates:
            unmatched += 1
        else:
            mismatched += 1
    if unmatched or mismatched:
        raise ValueError(
            "could not bind XDS_ASCII accepted rows exactly to INTEGRATE.HKL: "
            f"unmatched={unmatched}, indexing/duplicate mismatches={mismatched}"
        )
    return keep, len(corrected)


def _export_xds(config):
    import numpy as np
    from dials.array_family import flex
    from dials.util.export_pets import PETSOutput
    from dxtbx.model.experiment_list import ExperimentList
    from iotbx.xds import integrate_hkl

    xds_dir = Path(config["xds_dir"])
    xds_inp = xds_dir / "XDS.INP"
    integrate_path = xds_dir / "INTEGRATE.HKL"
    if not xds_inp.is_file() or not integrate_path.is_file():
        raise ValueError("XDS PETS export requires XDS.INP and INTEGRATE.HKL")
    profile_fitting = _read_xds_profile_fitting(xds_inp, integrate_path)
    if profile_fitting is not False and not bool(config["allow_profile_fitted_xds"]):
        state = "enabled" if profile_fitting else "not declared"
        raise ValueError(
            f"XDS PROFILE_FITTING is {state}; summation-only export requires PROFILE_FITTING=FALSE. "
            "Use --allow-profile-fitted-xds only when a labeled profile-estimate export is intended."
        )

    experiments, geometry_path, detector_workaround = _xds_geometry_experiment(xds_dir)
    experiment = experiments[0]
    handle = integrate_hkl.reader()
    handle.read_file(str(integrate_path))
    transform, transform_error = _derive_xds_reindex(experiment, handle)
    hkl_raw = np.asarray(handle.hkl, dtype=int)
    hkl = hkl_raw @ transform.T

    peak = np.asarray(handle.peak, dtype=float) * 0.01
    rlp = np.asarray(handle.rlp, dtype=float)
    iobs = np.asarray(handle.iobs, dtype=float)
    sigma = np.asarray(handle.sigma, dtype=float)
    factor = peak / np.maximum(rlp, 1e-30)
    recorded_i = iobs * factor
    recorded_var = np.square(sigma * factor)
    variance_policy = "XDS INTEGRATE sigma transformed by PEAK/RLP"
    nonpositive_after_undo = 0
    if getattr(handle, "variance_model", None):
        a, b = [float(value) for value in handle.variance_model[:2]]
        candidate = recorded_var / a - b * np.square(recorded_i)
        invalid = ~np.isfinite(candidate) | (candidate <= 0.0)
        nonpositive_after_undo = int(np.count_nonzero(invalid))
        candidate[invalid] = np.square(sigma[invalid])
        recorded_var = candidate
        variance_policy = (
            "DIALS import_xds variance undo: var_raw=var_recorded/a-b*I_recorded^2; "
            "nonpositive rows fall back to source SIGMA^2"
        )

    keep = (
        np.all(np.isfinite(hkl), axis=1)
        & np.isfinite(recorded_i)
        & np.isfinite(recorded_var)
        & (recorded_var > 0.0)
        & np.isfinite(peak)
        & (peak > 0.0)
        & np.isfinite(rlp)
        & (rlp > 0.0)
        & (sigma > 0.0)
    )
    xds_ascii = xds_dir / "XDS_ASCII.HKL"
    correct_rows = None
    if config["xds_row_selection"] == "correct":
        if not xds_ascii.is_file():
            raise ValueError("--xds-row-selection correct requires unmerged XDS_ASCII.HKL")
        correct_keep, correct_rows = _xds_correct_acceptance(handle, hkl, xds_ascii)
        keep &= correct_keep

    xyzcal = np.asarray(handle.xyzcal, dtype=float)
    if len(getattr(handle, "iseg", [])):
        panel = np.asarray(handle.iseg, dtype=int) - 1
    else:
        panel = np.zeros(len(hkl), dtype=int)
    if len(experiment.detector) > 1:
        for panel_id, detector_panel in enumerate(experiment.detector):
            selected = panel == panel_id
            offset = detector_panel.get_raw_image_offset()
            xyzcal[selected] -= np.asarray([offset[0], offset[1], 0.0])

    table = flex.reflection_table()
    table["id"] = flex.int(int(np.count_nonzero(keep)), 0)
    table["panel"] = flex.size_t(panel[keep].astype(int).tolist())
    table["miller_index"] = flex.miller_index([tuple(map(int, row)) for row in hkl[keep]])
    table["xyzcal.px"] = flex.vec3_double([tuple(map(float, row)) for row in xyzcal[keep]])
    table["partiality"] = flex.double(peak[keep].astype(float).tolist())
    table["intensity.sum.value"] = flex.double(recorded_i[keep].astype(float).tolist())
    table["intensity.sum.variance"] = flex.double(recorded_var[keep].astype(float).tolist())
    table.set_flags(flex.bool(len(table), True), table.flags.integrated_sum)
    partiality_rows = int(np.count_nonzero(peak[keep] >= float(config["min_partiality"])))

    one_experiment = ExperimentList()
    one_experiment.append(experiment)
    # dxtbx's public pixel-to-mm method operates on the observed-centroid
    # columns.  XDS supplies calculated detector positions (X/Y/ZCAL), so use
    # that method and copy the resulting geometry into the calculated column
    # expected by DIALS' PETS writer.
    table["xyzobs.px.value"] = table["xyzcal.px"].deep_copy()
    table["xyzobs.px.variance"] = flex.vec3_double(len(table), (1.0, 1.0, 1.0))
    table.centroid_px_to_mm(one_experiment)
    table["xyzcal.mm"] = table["xyzobs.mm.value"].deep_copy()
    table.map_centroids_to_reciprocal_space(one_experiment, calculated=True)
    prefix = Path(config["out_dir"]) / str(config["dataset_name"])
    params = _pets_params(
        prefix,
        intensity="sum",
        partiality=config["min_partiality"],
        excitation_error=config["excitation_error_cutoff"],
        n_merged=config["n_merged"],
        step=config["step"],
    )
    PETSOutput(one_experiment, [table], params).write_dyn_cif_pets()
    result = {
        "source_kind": "XDS",
        "source_dir": str(xds_dir.resolve()),
        "source_integrate_hkl": str(integrate_path.resolve()),
        "source_xds_ascii": str(xds_ascii.resolve()) if xds_ascii.exists() else None,
        "geometry_source": str(Path(geometry_path).resolve()),
        "profile_fitting": profile_fitting,
        "xds_row_selection": str(config["xds_row_selection"]),
        "source_rows": int(len(hkl)),
        "correct_rows": correct_rows,
        "selected_rows_before_partiality": int(len(table)),
        "source_observations_at_partiality": partiality_rows,
        "reindex_matrix_raw_to_export": transform.astype(int).tolist(),
        "reindex_determinant": int(round(np.linalg.det(transform))),
        "reindex_rounding_error": transform_error,
        "detector_sensor_thickness_workaround": bool(detector_workaround),
        "variance_model": [float(value) for value in getattr(handle, "variance_model", [])],
        "variance_policy": variance_policy,
        "variance_undo_nonpositive_fallback_rows": nonpositive_after_undo,
        "observation_convention": "I_recorded=IOBS*(PEAK/100)/RLP; sigma transformed identically",
        "outputs": [{"path": str(prefix) + ".cif_pets"}],
    }
    result.update(_space_group_metadata(experiment.crystal))
    return result


def main():
    from dials.util.version import dials_version

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    Path(config["out_dir"]).mkdir(parents=True, exist_ok=True)
    if config["source_kind"] == "DIALS":
        result = _export_dials(config)
    elif config["source_kind"] == "XDS":
        result = _export_xds(config)
    else:
        raise ValueError(f"unknown source kind {config['source_kind']!r}")
    result["dials_version"] = str(dials_version())
    Path(args.result).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
