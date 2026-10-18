"""Options of `grond import-dials` and `grond export-pets`."""

from __future__ import annotations

import argparse


def register_integration(sub: argparse._SubParsersAction) -> None:
    """Register integration commands."""

    from grond.commands.integration_imports import (
        cmd_export_pets,
        cmd_import_dials,
    )

    p_dials = sub.add_parser(
        "import-dials", help="Convert a DIALS integrated.refl/integrated.expt pair for GROND."
    )
    p_dials.add_argument("--refl", required=True, help="DIALS integrated.refl or scaled.refl file.")
    p_dials.add_argument("--expt", required=True, help="DIALS integrated.expt or scaled.expt file.")
    p_dials.add_argument("--out-dir", required=True, help="Output directory.")
    p_dials.add_argument("--dials-python", default="", help="Python executable that can import dials and dxtbx.")
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

    p_pets = sub.add_parser(
        "export-pets",
        help="Write DIALS or XDS observations as PETS2 cif_pets files.",
    )
    p_pets.add_argument("--dials-refl", default="", help="DIALS integrated.refl or scaled.refl input.")
    p_pets.add_argument("--dials-expt", default="", help="DIALS integrated.expt or scaled.expt input.")
    p_pets.add_argument("--xds-dir", default="", help="XDS directory containing XDS.INP and INTEGRATE.HKL.")
    p_pets.add_argument("--out-dir", required=True, help="Output directory.")
    p_pets.add_argument("--dataset-name", default="dataset")
    p_pets.add_argument("--dials-python", default="", help="Python executable that can import dials and dxtbx.")
    p_pets.add_argument(
        "--experiment-id",
        action="append",
        type=int,
        default=None,
        help="DIALS experiment-list index to export. Repeat as needed; the default exports all experiments separately.",
    )
    p_pets.add_argument("--intensity", choices=["sum", "profile"], default="sum")
    p_pets.add_argument(
        "--row-selection",
        choices=["integrated", "scaled"],
        default="integrated",
        help="For DIALS, export integrated rows or only rows accepted by dials.scale while retaining the chosen observation column.",
    )
    p_pets.add_argument(
        "--xds-row-selection",
        choices=["correct", "integrate"],
        default="correct",
        help="For XDS, retain only rows accepted into unmerged XDS_ASCII.HKL, or all valid INTEGRATE.HKL rows.",
    )
    p_pets.add_argument(
        "--xds-sigma-mode",
        choices=["reported", "raw"],
        default="reported",
        help=(
            "Retain XDS's reported calibrated sigma after PEAK/RLP transformation, or invert "
            "the declared XDS variance model as a labeled raw-sigma sensitivity export."
        ),
    )
    p_pets.add_argument("--min-partiality", type=float, default=0.90)
    p_pets.add_argument("--excitation-error-cutoff", type=float, default=0.04)
    p_pets.add_argument(
        "--frame-ownership",
        choices=["overlap", "nearest-excitation"],
        default="overlap",
        help=(
            "Assign observations to all eligible PETS virtual frames (overlap), or exactly one "
            "minimum-|Sg| frame (nearest-excitation) for diffBloch refinement."
        ),
    )
    p_pets.add_argument("--virtual-frame-n-merged", type=int, default=1)
    p_pets.add_argument("--virtual-frame-step", type=int, default=1)
    p_pets.add_argument(
        "--scan-varying-cell",
        choices=["error", "project"],
        default="project",
        help="cif_pets has one cell. Project keeps DIALS U(scan) with the declared static B; error refuses varying-cell input.",
    )
    p_pets.add_argument("--max-cell-length-deviation", type=float, default=0.03)
    p_pets.add_argument("--max-cell-angle-deviation", type=float, default=1.0)
    p_pets.add_argument(
        "--keep-bad-flags", action="store_true", help="Diagnostic only: keep severe DIALS bad-flag rows."
    )
    p_pets.add_argument(
        "--allow-profile-fitted-xds",
        action="store_true",
        help="Diagnostic only: allow XDS input not explicitly produced with PROFILE_FITTING=FALSE.",
    )
    p_pets.add_argument(
        "--row-contract",
        default="",
        help=(
            "Optional GROND native_dyn_calc.csv whose immutable row_index/split columns filter an XDS export. "
            "Use with --split-selection work to keep GROND free rows out of an external optimizer."
        ),
    )
    p_pets.add_argument(
        "--split-selection",
        choices=["all", "work", "free"],
        default="all",
        help="Rows retained from --row-contract before PETS virtual-frame ownership.",
    )
    p_pets.add_argument("--overwrite", action="store_true")
    p_pets.set_defaults(func=cmd_export_pets)

