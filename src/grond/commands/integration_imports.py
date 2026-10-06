"""Import or export integration contracts and apply supported external model updates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import read_json


def cmd_import_dials(args: argparse.Namespace) -> int:
    """CLI entrypoint for converting DIALS `.refl`/`.expt` files into GROND rows."""

    from grond.io.dials_bridge import convert_dials_to_grond

    try:
        bridge = convert_dials_to_grond(
            args.refl,
            args.expt,
            args.out_dir,
            intensity_mode=args.intensity_mode,
            row_selection=args.row_selection,
            sigma_mode=args.sigma_mode,
            sigma_scale=args.sigma_scale,
            dials_python=args.dials_python,
            min_partiality_write=args.min_partiality_write,
            keep_dont_integrate=bool(args.keep_dont_integrate) and not bool(args.drop_dont_integrate),
            drop_bad_flags=not bool(args.keep_bad_flags),
            structure=args.structure or None,
            baseline_hkl=args.baseline_hkl or None,
            dataset_name=args.dataset_name,
            prep_out=args.prep_out or None,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    payload = {
        "out_dir": str(bridge.out_dir),
        "hkl": str(bridge.hkl),
        "scan_points": str(bridge.scan_points),
        "summary": str(bridge.summary),
    }
    if bridge.prep:
        payload["prep"] = str(bridge.prep)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_import_pets(args: argparse.Namespace) -> int:
    """Import a native PETS2 observation/geometry contract into GROND rows."""

    from grond.io.pets_import import import_pets_to_grond

    try:
        bridge = import_pets_to_grond(
            args.cif_pets,
            args.out_dir,
            pets_hkl=args.pets_hkl or None,
            allow_virtual_frames=bool(args.allow_virtual_frames),
            klar_rsg=args.klar_rsg,
            klar_dsg=float(args.klar_dsg),
            exclude_systematic_absences=bool(args.exclude_systematic_absences),
            jana_reflistout=args.jana_reflistout or None,
            dataset_name=args.dataset_name,
            space_group_number=int(args.space_group_number),
            hkl_transform=args.hkl_transform,
            structure=args.structure or None,
            baseline_hkl=args.baseline_hkl or None,
            prep_out=args.prep_out or None,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    summary_payload = read_json(bridge.summary)
    static_endpoint = summary_payload.get("static_endpoint_contract")
    static_endpoint_ready = bool(isinstance(static_endpoint, Mapping) and static_endpoint.get("status") == "validated")
    payload = {
        "status": "complete",
        "out_dir": str(bridge.out_dir),
        "hkl": str(bridge.hkl),
        "scan_points": str(bridge.scan_points),
        "summary": str(bridge.summary),
        "scope": (
            "ordinary PETS row-space plus a validated SHELXL endpoint for AUTO"
            if static_endpoint_ready
            else "native PETS row-space calculation only; no validated static SHELXL endpoint"
        ),
    }
    if bridge.prep:
        payload["prep"] = str(bridge.prep)
    print(json.dumps(payload, indent=2))
    return 0


def cmd_export_pets(args: argparse.Namespace) -> int:
    """Export DIALS or XDS observations to PETS/Jana/diffBloch ``cif_pets``."""

    from grond.io.pets_export import export_pets

    try:
        result = export_pets(
            out_dir=args.out_dir,
            dataset_name=args.dataset_name,
            dials_python=args.dials_python,
            dials_refl=args.dials_refl or None,
            dials_expt=args.dials_expt or None,
            xds_dir=args.xds_dir or None,
            experiment_ids=args.experiment_id,
            intensity=args.intensity,
            row_selection=args.row_selection,
            xds_row_selection=args.xds_row_selection,
            xds_sigma_mode=args.xds_sigma_mode,
            min_partiality=args.min_partiality,
            excitation_error_cutoff=args.excitation_error_cutoff,
            frame_ownership=args.frame_ownership,
            n_merged=args.virtual_frame_n_merged,
            step=args.virtual_frame_step,
            scan_varying_cell=args.scan_varying_cell,
            max_cell_length_fraction=args.max_cell_length_deviation,
            max_cell_angle_degrees=args.max_cell_angle_deviation,
            keep_bad_flags=bool(args.keep_bad_flags),
            allow_profile_fitted_xds=bool(args.allow_profile_fitted_xds),
            row_contract=args.row_contract or None,
            split_selection=args.split_selection,
            overwrite=bool(args.overwrite),
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "out_dir": str(result.out_dir),
                "cif_pets": [str(path) for path in result.files],
                "manifest": str(result.manifest),
                "readme": str(result.readme),
            },
            indent=2,
        )
    )
    return 0


def cmd_apply_diffbloch_model(args: argparse.Namespace) -> int:
    """Transfer a refined diffBloch CIF into an existing SHELX instruction model."""

    from grond.diffbloch_endpoint import materialize_diffbloch_res

    audit_path = args.audit_json or str(Path(args.out_res).with_suffix(".diffbloch_handoff.json"))
    try:
        payload = materialize_diffbloch_res(
            args.source_res,
            args.refined_cif,
            args.out_res,
            audit_json=audit_path,
            max_cell_length_fraction=float(args.max_cell_length_deviation),
            max_cell_angle_degrees=float(args.max_cell_angle_deviation),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0 if payload.get("status") == "complete" or args.allow_npd else 3
