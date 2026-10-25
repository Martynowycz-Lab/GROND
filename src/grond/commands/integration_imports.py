"""`grond import-dials`."""

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
