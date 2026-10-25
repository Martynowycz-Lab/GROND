"""Options of `grond import-dials`."""

from __future__ import annotations

import argparse


def register_integration(sub: argparse._SubParsersAction) -> None:
    """Register integration commands."""

    from grond.commands.integration_imports import cmd_import_dials

    p_dials = sub.add_parser(
        "import-dials", help="Convert a DIALS integrated.refl/integrated.expt pair for GROND."
    )
    p_dials.add_argument("--refl", required=True, help="DIALS integrated.refl or scaled.refl file.")
    p_dials.add_argument("--expt", required=True, help="DIALS integrated.expt or scaled.expt file.")
    p_dials.add_argument("--out-dir", required=True, help="Output directory.")
    p_dials.add_argument("--intensity-mode", choices=["sum_complete", "sum_raw", "scale"], default="sum_complete")
    p_dials.add_argument(
        "--row-selection",
        choices=["integrated", "scaled"],
        default="integrated",
        help="Use all integrated rows or only rows accepted by DIALS scaling. The latter can still use raw summation intensities.",
    )
    p_dials.add_argument("--sigma-mode", choices=["auto", "raw", "detector_gain"], default="raw")
    p_dials.add_argument("--sigma-scale", type=float, default=None)
    p_dials.add_argument("--min-partiality-write", type=float, default=1e-6)
    p_dials.add_argument(
        "--drop-dont-integrate",
        action="store_true",
        help="Deprecated alias for the default production behavior. Rows flagged dont_integrate are dropped unless --keep-dont-integrate is used.",
    )
    p_dials.add_argument(
        "--keep-dont-integrate", action="store_true", help="Diagnostic only: keep DIALS rows flagged dont_integrate."
    )
    p_dials.add_argument(
        "--keep-bad-flags",
        action="store_true",
        help="Diagnostic only: keep DIALS rows with severe bad-pixel, overloaded, powder-ring, failed-background, or failed-summation flags.",
    )
    p_dials.add_argument("--structure", default="", help="Optional RES/CIF path used to also write a prep JSON.")
    p_dials.add_argument("--baseline-hkl", default="", help="Optional solved-model HKL path stored in the prep JSON.")
    p_dials.add_argument("--dataset-name", default="dials")
    p_dials.add_argument(
        "--prep-out",
        default="",
        help="Optional prep JSON output path. Defaults to OUT/grond_prep.json when --structure is supplied.",
    )
    p_dials.set_defaults(func=cmd_import_dials)
