"""`grond import-dials` and `grond export-pets`."""

from __future__ import annotations

import argparse
import json


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
