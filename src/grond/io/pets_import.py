"""Import a native PETS2 ``cif_pets`` observation table into GROND.

This bridge deliberately consumes the ordinary PETS2 integration file: one
measured signed reflection assigned to one measured zone axis.  It refuses a
virtual-frame dynamical file when the same signed reflection has been copied
into more than one zone, because those copies are not independent
observations.

PETS writes the frame orientation as ``Rz(omega) Rx(alpha) Ry(beta) UB`` in a
coordinate system whose incident beam points along ``-z``.  GROND's existing
scan-varying-A path can represent that geometry without inventing detector
coordinates or virtual frames.  The nominal scan rotation is placed about the
beam direction (which cannot change excitation errors); the PETS frame A
matrices therefore remain the sole source of the physical orientation path.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gemmi
import numpy as np

from grond.hkl_transform import parse_hkl_transform_info
from grond.io.pets_contract import (
    _loop_columns as _loop_columns,
    _operation_key as _operation_key,
    _read_jana_reflistout as _read_jana_reflistout,
    _read_pets_hkl_sidecar as _read_pets_hkl_sidecar,
    _read_static_shelx_hkl as _read_static_shelx_hkl,
    _sha256 as _sha256,
    _validate_static_endpoint_hkl as _validate_static_endpoint_hkl,
    _validate_structure_setting as _validate_structure_setting,
    read_pets_contract as read_pets_contract,
)
from grond.io.pets_geometry import (
    _AXIS_POSITION as _AXIS_POSITION,
    _axis_position as _axis_position,
    _busing_levy as _busing_levy,
    _cell_matrix as _cell_matrix,
    _continuous_rotation_contract as _continuous_rotation_contract,
    _number as _number,
    _required_pair as _required_pair,
    _rotation as _rotation,
)


@dataclass(frozen=True)
class PetsImportOutput:
    out_dir: Path
    hkl: Path
    scan_points: Path
    summary: Path
    prep: Path | None = None


def import_pets_to_grond(
    cif_pets: Path,
    out_dir: Path,
    *,
    pets_hkl: Path | None = None,
    allow_virtual_frames: bool = False,
    klar_rsg: float | None = None,
    klar_dsg: float = 0.0,
    exclude_systematic_absences: bool = False,
    jana_reflistout: Path | None = None,
    dataset_name: str = "pets2",
    space_group_number: int,
    hkl_transform: str = "identity",
    structure: Path | None = None,
    baseline_hkl: Path | None = None,
    prep_out: Path | None = None,
) -> PetsImportOutput:
    """Write a PETS-owned, row-preserving GROND bridge directory."""

    if klar_rsg is not None and (not math.isfinite(float(klar_rsg)) or float(klar_rsg) <= 0.0):
        raise ValueError("--klar-rsg must be finite and positive when supplied")
    if not math.isfinite(float(klar_dsg)):
        raise ValueError("--klar-dsg must be finite")
    contract = read_pets_contract(
        cif_pets,
        pets_hkl=pets_hkl,
        allow_virtual_frames=allow_virtual_frames,
    )
    transform = parse_hkl_transform_info(hkl_transform)
    if transform.canonical != "identity":
        raise ValueError(
            "native PETS import currently requires hkl_transform=identity; a nonidentity "
            "setting needs symmetry conjugation that is not yet implemented. Hand remains "
            "a separate negative-thickness coordinate comparison."
        )
    structure_validation = (
        _validate_structure_setting(
            Path(structure),
            space_group_number=int(space_group_number),
            pets_cell=np.asarray(contract["cell"], dtype=float),
        )
        if structure is not None
        else None
    )
    destination = Path(out_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    all_observations = list(contract["observations"])
    static_endpoint_contract: dict[str, Any] = {
        "status": "unavailable",
        "reason": "no baseline HKLF4 supplied",
    }
    if baseline_hkl is not None:
        if bool(contract["virtual_frames"]):
            static_endpoint_contract = {
                "status": "unavailable",
                "reason": "PETS virtual-frame occurrences are not a static HKLF4 observation table",
            }
        elif klar_rsg is not None or bool(exclude_systematic_absences):
            static_endpoint_contract = {
                "status": "unavailable",
                "reason": "row filtering changed the ordinary PETS observation support",
            }
        else:
            static_endpoint_contract = _validate_static_endpoint_hkl(Path(baseline_hkl), all_observations)
    operations = gemmi.SpaceGroup(int(space_group_number)).operations()
    k_magnitude = 1.0 / float(contract["wavelength"])
    s0 = np.asarray([0.0, 0.0, -k_magnitude])
    selected_observations = []
    selection_ledger = []
    klar_kept = 0
    symmetry_kept = 0
    jana_categories = (
        _read_jana_reflistout(Path(jana_reflistout).expanduser().resolve()) if jana_reflistout is not None else {}
    )
    for source_index, row in enumerate(all_observations):
        hkl = np.asarray(row[:3], dtype=int)
        ordinal = int(row[6])
        source_zone_id = str(row[5])
        try:
            jana_zone_id = int(source_zone_id)
        except ValueError:
            if jana_categories:
                raise ValueError(
                    f"--jana-reflistout requires numeric PETS source zone IDs; found {source_zone_id!r}"
                ) from None
            jana_zone_id = ordinal
        matrix = np.asarray(contract["a_matrices"][ordinal - 1], dtype=float)
        g_vector = matrix @ hkl
        sg = abs((k_magnitude**2 - float(np.sum(np.square(s0 + g_vector)))) / (2.0 * k_magnitude))
        semiangle = float(contract["zones"][ordinal - 1]["integration_half_width_deg"])
        sg_max = float(np.linalg.norm(g_vector[1:]) * math.radians(semiangle))
        klar_ok = True
        if klar_rsg is not None:
            klar_ok = bool(sg_max > 0.0 and sg / sg_max < float(klar_rsg) and sg_max - sg > float(klar_dsg))
        symmetry_ok = not operations.is_systematically_absent(tuple(int(v) for v in hkl))
        if klar_ok:
            klar_kept += 1
        if symmetry_ok:
            symmetry_kept += 1
        keep = klar_ok and (symmetry_ok or not exclude_systematic_absences)
        if keep:
            selected_observations.append(row)
        selection_ledger.append(
            {
                "source_row_index": source_index,
                "h": int(hkl[0]),
                "k": int(hkl[1]),
                "l": int(hkl[2]),
                "zone_ordinal": ordinal,
                "sg_center_inv_A": sg,
                "sg_max_inv_A": sg_max,
                "klar_keep": klar_ok,
                "systematic_absence": not symmetry_ok,
                "jana_category": jana_categories.get((int(hkl[0]), int(hkl[1]), int(hkl[2]), jana_zone_id)),
                "selected": keep,
            }
        )
    contract["observations"] = selected_observations
    if not selected_observations:
        raise ValueError("PETS row selection retained no observations")
    selected_d_values = [
        gemmi.UnitCell(*[float(value) for value in contract["cell"]]).calculate_d(
            (int(row[0]), int(row[1]), int(row[2]))
        )
        for row in selected_observations
    ]
    selected_d_values = [
        float(value) for value in selected_d_values if math.isfinite(float(value)) and float(value) > 0.0
    ]
    if not selected_d_values:
        raise ValueError("PETS row selection has no finite positive d spacings")
    selected_d_min = float(min(selected_d_values))
    # The engine applies an inclusive d>=cutoff test. Move by one floating-point
    # representable value toward zero so a row exactly at the PETS edge survives
    # serialization/reload without weakening the declared support.
    engine_d_min = float(np.nextafter(selected_d_min, 0.0))

    scan_path = destination / "PETS_A_AT_ZONE_AXES.txt"
    with scan_path.open("w") as stream:
        for matrix in contract["a_matrices"]:
            stream.write(" ".join(f"{float(value): .12e}" for value in matrix.reshape(-1)) + "\n")

    geometry_path = destination / "PETS_EXACT_GEOMETRY.npz"
    np.savez_compressed(
        geometry_path,
        a_center=np.asarray(contract["a_matrices"], dtype=float),
        integration_semiangle_deg=np.asarray(
            [row["integration_half_width_deg"] for row in contract["zones"]],
            dtype=float,
        ),
        source_zone_id=np.arange(1, len(contract["zones"]) + 1, dtype=np.int64),
        source_zone_id_text=np.asarray([str(row["source_id"]) for row in contract["zones"]], dtype="U64"),
        rotation_axis_position_deg=np.asarray(contract["axis_position_deg"], dtype=float),
    )
    selection_path = destination / "PETS_ROW_SELECTION.json"
    selection_path.write_text(json.dumps(selection_ledger, indent=2) + "\n")

    cell = contract["cell"]
    real = contract["real_matrix"]
    step = float(contract["orientation_step_deg"])
    zone_by_ordinal = {int(row["ordinal"]): row for row in contract["zones"]}
    hkl_path = destination / "INTEGRATE.HKL"
    with hkl_path.open("w") as stream:
        stream.write("!OUTPUT_FILE=INTEGRATE.HKL\n")
        stream.write("!GENERATED_BY=GROND import-pets EXPERIMENTAL\n")
        stream.write("!GROND_SOURCE=PETS2\n")
        stream.write(f"!PETS_CIF= {contract['source']}\n")
        stream.write(f"!PETS_CIF_SHA256= {_sha256(contract['source'])}\n")
        stream.write(f"!PETS_EXACT_GEOMETRY_FILE= {geometry_path.name}\n")
        if contract["pets_hkl"] is not None:
            stream.write(f"!PETS_HKL= {contract['pets_hkl']}\n")
            stream.write(f"!PETS_HKL_SHA256= {_sha256(contract['pets_hkl'])}\n")
        stream.write(
            "!PETS_OBSERVATION_CONTRACT= "
            + (
                "PETS_owned_virtual_frame_occurrences\n"
                if contract["virtual_frames"]
                else "unique_signed_hkl_one_measured_frame\n"
            )
        )
        stream.write(f"!PETS_VIRTUAL_FRAMES= {'TRUE' if contract['virtual_frames'] else 'FALSE'}\n")
        stream.write(
            "!PETS_STATIC_HKLF4_VALIDATED= "
            f"{'TRUE' if static_endpoint_contract.get('status') == 'validated' else 'FALSE'}\n"
        )
        if static_endpoint_contract.get("status") == "validated":
            stream.write(f"!PETS_STATIC_HKLF4= {static_endpoint_contract['path']}\n")
            stream.write(f"!PETS_STATIC_HKLF4_SHA256= {static_endpoint_contract['sha256']}\n")
        stream.write("!PETS_INTENSITY_SOURCE= _refln_intensity_meas\n")
        stream.write("!PETS_SIGMA_SOURCE= _refln_intensity_sigma\n")
        stream.write("!PETS_PARTIALITY= unavailable; PEAK=100 is an inclusion sentinel only\n")
        stream.write(f"!X-RAY_WAVELENGTH= {contract['wavelength']:.12g}\n")
        stream.write(f"!SPACE_GROUP_NUMBER= {int(space_group_number)}\n")
        stream.write("!STARTING_FRAME= 1\n")
        stream.write("!STARTING_ANGLE= 0\n")
        stream.write(f"!OSCILLATION_RANGE= {step:.12g}\n")
        stream.write("!ROTATION_AXIS= 0 0 -1\n")
        stream.write("!GROND_ROTATION_AXIS_DATUM= 0 0 -1\n")
        stream.write("!INCIDENT_BEAM_DIRECTION= 0 0 -1\n")
        stream.write("!GROND_FIXED_ROTATION= 1 0 0 0 1 0 0 0 1\n")
        stream.write("!GROND_SETTING_ROTATION= 1 0 0 0 1 0 0 0 1\n")
        stream.write(f"!UNIT_CELL_CONSTANTS= {' '.join(f'{v:.10g}' for v in cell)}\n")
        for label, vector in zip("ABC", real.T, strict=True):
            stream.write(f"!UNIT_CELL_{label}-AXIS= {' '.join(f'{v:.12g}' for v in vector)}\n")
        stream.write(f"!DIALS_A_AT_SCAN_POINTS_FILE= {scan_path.name}\n")
        stream.write(
            "!COLUMN_CONTRACT= H K L IOBS SIGMA XCAL YCAL ZCAL RLP PEAK CORR MAXC XOBS YOBS ZOBS ALF0 BET0 ALF1 BET1 PSI ISEG\n"
        )
        stream.write("!END_OF_HEADER\n")
        for (
            h,
            k,
            ell,
            intensity,
            sigma,
            _frame_id,
            ordinal,
            _cif_zone_id,
            _incident,
            _diffracted,
        ) in contract["observations"]:
            half_width_deg = float(zone_by_ordinal[int(ordinal)]["integration_half_width_deg"])
            half_width_frames = half_width_deg / step
            stream.write(
                f"{h:5d} {k:5d} {ell:5d} {intensity: .8e} {sigma: .8e} "
                f"nan nan {float(ordinal):.5f} 1.00000000e+00 100.00000 100.00 0.0 "
                f"nan nan nan {float(ordinal) - half_width_frames:.5f} "
                f"{float(ordinal) + half_width_frames:.5f} nan nan 0.00000 0\n"
            )

    zone_ledger = destination / "PETS_ZONE_AXIS_LEDGER.json"
    zone_ledger.write_text(json.dumps(contract["zones"], indent=2) + "\n")
    summary_payload = {
        "schema": "grond.pets2_import.experimental.v1",
        "status": "ok",
        "source": str(contract["source"]),
        "source_sha256": _sha256(contract["source"]),
        "pets_hkl": str(contract["pets_hkl"]) if contract["pets_hkl"] else None,
        "pets_hkl_sha256": (_sha256(contract["pets_hkl"]) if contract["pets_hkl"] else None),
        "output_hkl": str(hkl_path),
        "scan_points": str(scan_path),
        "exact_geometry": str(geometry_path),
        "row_selection_ledger": str(selection_path),
        "zone_axis_ledger": str(zone_ledger),
        "dataset_name": dataset_name,
        "structure_validation": structure_validation,
        "space_group_number": int(space_group_number),
        "hkl_transform": transform.to_dict(),
        "rows_input": len(all_observations),
        "rows": len(contract["observations"]),
        "selected_min_d_spacing_A": selected_d_min,
        "engine_min_d_spacing_A": engine_d_min,
        "zone_axes": len(contract["zones"]),
        "unique_signed_hkls": len({(row[0], row[1], row[2]) for row in contract["observations"]}),
        "virtual_frames": contract["virtual_frames"],
        "repeated_signed_hkls": contract["repeated_signed_hkls"],
        "maximum_virtual_multiplicity": contract["maximum_virtual_multiplicity"],
        "row_selection": {
            "klar_rsg": klar_rsg,
            "klar_dsg": klar_dsg if klar_rsg is not None else None,
            "klar_kept": klar_kept,
            "space_group_number": int(space_group_number),
            "full_systematic_absence_filter": bool(exclude_systematic_absences),
            "symmetry_allowed": symmetry_kept,
            "selected": len(selected_observations),
            "selected_systematic_absences": int(
                sum(bool(row["selected"]) and bool(row["systematic_absence"]) for row in selection_ledger)
            ),
            "jana_reflistout": (
                str(Path(jana_reflistout).expanduser().resolve()) if jana_reflistout is not None else None
            ),
            "jana_category_counts": {
                str(category): int(sum(value == category for value in jana_categories.values()))
                for category in sorted(set(jana_categories.values()))
            },
            "klar_nonabsence_intersection_with_jana_category_5": int(
                sum(
                    bool(row["klar_keep"]) and not bool(row["systematic_absence"]) and row["jana_category"] == 5
                    for row in selection_ledger
                )
            ),
        },
        "wavelength_A": float(contract["wavelength"]),
        "cell": [float(value) for value in contract["cell"]],
        "orientation": {
            "formula": (
                "U=UB B(cell)^-1; O=Rz(-axis_position) Rz(omega) "
                "Rx(alpha) Ry(beta) U; A_zone=orientation_basis(cell,O)^T"
            ),
            "incident_beam": [0.0, 0.0, -1.0],
            "zone_step_deg_from_median_successive_orientation_rotation": step,
            "zone_step_range_deg": contract["orientation_step_range_deg"],
            "zone_step_source": contract["orientation_step_source"],
            "zone_axis_normalized_error_max": contract["zone_axis_error_max"],
            "zone_axis_normalized_error_rms": contract["zone_axis_error_rms"],
            "acquisition_geometry": contract["acquisition_geometry"],
        },
        "intensity_contract": {
            "intensity": "PETS _refln_intensity_meas unchanged",
            "sigma": "PETS _refln_intensity_sigma unchanged",
            "row_orientation": (
                "cif_pets virtual zone axis owned by PETS"
                if contract["virtual_frames"]
                else "companion PETS .hkl column 6 acquisition frame; cif_pets _refln_zone_axis_id retained only as provenance"
            ),
            "partiality": "not provided by cif_pets; no partiality filter applied",
            "virtual_frames": contract["virtual_frames"],
            "detector_coordinates": "not provided; detector scale fields must remain disabled",
        },
        "static_endpoint_contract": static_endpoint_contract,
        "recommended_run_contract": {
            "hkl_transform": "identity; nonidentity PETS/model setting conjugation is not yet implemented",
            "min_peak": 0.0,
            "detector_scale_bins": "0",
            "profile_geometry": "PETS_EXACT_GEOMETRY.npz direct-angle rocking path",
            "profile_accumulation": "sum",
            "beam_and_quadrature": "must be converged for the structure and PETS settings",
            "shelxl_endpoint": (
                "validated row-preserving PETS HKLF4; q correction and SHELXL feedback are available"
                if static_endpoint_contract.get("status") == "validated"
                else "not available without a row-preserving PETS static HKLF4 contract"
            ),
        },
    }
    summary_path = destination / "pets2_bridge_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")

    prep_path: Path | None = None
    if structure is not None:
        from .config import DatasetSpec, PrepConfig, write_prep

        structure_path = Path(structure).expanduser().resolve()
        prep_path = Path(prep_out).expanduser().resolve() if prep_out is not None else destination / "grond_prep.json"
        prep = PrepConfig(
            structure=str(structure_path),
            baseline_hkl=(str(Path(baseline_hkl).expanduser().resolve()) if baseline_hkl else ""),
            notes=(
                "Experimental native PETS2 row-space import; PETS-owned correlated virtual-frame occurrences."
                if contract["virtual_frames"]
                else "Experimental native PETS2 row-space import; one measured acquisition frame per signed row."
            ),
            datasets=[
                DatasetSpec(
                    name=dataset_name,
                    hkl=str(hkl_path),
                    xds_dir=str(destination),
                    hkl_transform=transform.canonical,
                    min_d_spacing=engine_d_min,
                    min_peak=0.0,
                    target_mode="iobs",
                    include_nonpositive=True,
                    geometry_percentile=100.0,
                )
            ],
            metadata={"pets2_bridge": summary_payload},
        )
        write_prep(prep_path, prep)

    return PetsImportOutput(destination, hkl_path, scan_path, summary_path, prep_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif-pets", type=Path, required=True)
    parser.add_argument("--pets-hkl", type=Path)
    parser.add_argument(
        "--allow-virtual-frames",
        action="store_true",
        help=(
            "Accept PETS-owned virtual-frame occurrences from *_dyn.cif_pets. "
            "These are correlated occurrences, not independent source observations."
        ),
    )
    parser.add_argument("--klar-rsg", type=float)
    parser.add_argument("--klar-dsg", type=float, default=0.0)
    parser.add_argument("--exclude-systematic-absences", action="store_true")
    parser.add_argument("--jana-reflistout", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="pets2")
    parser.add_argument("--space-group-number", type=int, required=True)
    parser.add_argument(
        "--hkl-transform",
        default="identity",
        help="Reserved for future setting conversion; the current importer requires identity.",
    )
    parser.add_argument("--structure", type=Path)
    parser.add_argument("--baseline-hkl", type=Path)
    parser.add_argument("--prep-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = import_pets_to_grond(
        args.cif_pets,
        args.out_dir,
        pets_hkl=args.pets_hkl,
        allow_virtual_frames=args.allow_virtual_frames,
        klar_rsg=args.klar_rsg,
        klar_dsg=args.klar_dsg,
        exclude_systematic_absences=args.exclude_systematic_absences,
        jana_reflistout=args.jana_reflistout,
        dataset_name=args.dataset_name,
        space_group_number=args.space_group_number,
        hkl_transform=args.hkl_transform,
        structure=args.structure,
        baseline_hkl=args.baseline_hkl,
        prep_out=args.prep_out,
    )
    print(
        json.dumps({key: str(value) if value is not None else None for key, value in asdict(output).items()}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
