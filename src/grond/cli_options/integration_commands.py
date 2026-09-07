"""Build the public command grammar in its established display order."""

from __future__ import annotations

import argparse


def register_integration(sub: argparse._SubParsersAction) -> None:
    """Register integration commands."""

    from grond.cli_support.values import DEFAULT_MAX_SUPPORT_SG
    from grond.commands.integration_imports import (
        cmd_apply_diffbloch_model,
        cmd_export_pets,
        cmd_import_dials,
        cmd_import_pets,
    )
    from grond.commands.pets_evaluate import cmd_pets_evaluate

    p_dials = sub.add_parser(
        "import-dials", help="Convert DIALS integrated.refl/integrated.expt into a GROND row directory."
    )
    p_dials.add_argument("--refl", required=True, help="DIALS integrated.refl or scaled.refl file.")
    p_dials.add_argument("--expt", required=True, help="DIALS integrated.expt or scaled.expt file.")
    p_dials.add_argument("--out-dir", required=True)
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

    p_import_pets = sub.add_parser(
        "import-pets",
        help="Import a native PETS2 cif_pets observation table and exact rocking geometry.",
        description=(
            "Import PETS2 intensities, uncertainties, signed indices, UB/zone geometry, and "
            "rocking widths without inventing detector coordinates. Ordinary cif_pets input "
            "requires its companion 13-column .hkl file so the acquisition frame remains "
            "known. PETS-owned *_dyn.cif_pets virtual-frame occurrences require the explicit "
            "--allow-virtual-frames acknowledgement. An ordinary import becomes eligible "
            "for full q/LS1 AUTO only when --baseline-hkl is PETS' row-matched static SHELX "
            "export. PETS virtual-frame occurrences remain native row-space data and are "
            "never claimed as a conventional SHELXL HKLF4 endpoint."
        ),
    )
    p_import_pets.add_argument("--cif-pets", required=True)
    p_import_pets.add_argument(
        "--pets-hkl",
        default="",
        help="Companion 13-column PETS .hkl file; inferred beside ordinary cif_pets input.",
    )
    p_import_pets.add_argument(
        "--allow-virtual-frames",
        action="store_true",
        help=(
            "Accept correlated PETS-owned occurrences from *_dyn.cif_pets. Repeated signed "
            "HKLs remain distinct zone observations and must not be described as independent "
            "source integrations."
        ),
    )
    p_import_pets.add_argument(
        "--klar-rsg",
        type=float,
        default=None,
        help="Optional Klar/Palatinus |Sg|/Sgmax row threshold, for example 0.66.",
    )
    p_import_pets.add_argument(
        "--klar-dsg",
        type=float,
        default=0.0,
        help="Optional Klar/Palatinus Sgmax-|Sg| margin in inverse Angstrom.",
    )
    p_import_pets.add_argument(
        "--exclude-systematic-absences",
        action="store_true",
        help=(
            "Drop systematic absences for a static-comparison support. Leave them present for "
            "a full dynamical calculation because multiple scattering can populate them."
        ),
    )
    p_import_pets.add_argument(
        "--jana-reflistout",
        default="",
        help="Optional Jana eldyn_reflistout used only to audit category overlap.",
    )
    p_import_pets.add_argument("--out-dir", required=True)
    p_import_pets.add_argument("--dataset-name", default="pets2")
    p_import_pets.add_argument("--space-group-number", type=int, required=True)
    p_import_pets.add_argument(
        "--hkl-transform",
        default="identity",
        help=(
            "Reserved for a future setting bridge; the current native PETS importer "
            "requires identity and keeps hand in the separate negative-thickness calculation."
        ),
    )
    p_import_pets.add_argument("--structure", default="", help="Optional RES/CIF used to write a prep JSON.")
    p_import_pets.add_argument(
        "--baseline-hkl",
        default="",
        help="Optional conventional solved-model HKL recorded as a separate comparator.",
    )
    p_import_pets.add_argument(
        "--prep-out",
        default="",
        help="Prep JSON path; defaults to OUT/grond_prep.json when --structure is supplied.",
    )
    p_import_pets.set_defaults(func=cmd_import_pets)

    p_pets_evaluate = sub.add_parser(
        "pets-evaluate",
        help="Evaluate native PETS2 rows without invoking a fabricated SHELXL endpoint.",
        description=(
            "Run the exact PETS rocking geometry through matched Born/Bloch row calculations, "
            "a declared thickness grid, optional elastic negative-thickness hand comparison, "
            "and JANA/PETS-style row reporting. N, Q, reciprocal support, and thickness are "
            "explicit because they must be convergence-tested for each structure."
        ),
    )
    p_pets_evaluate.add_argument("--prep", required=True)
    p_pets_evaluate.add_argument(
        "--structure",
        default="",
        help="Optional audited model override, for example a validation-selected native update.",
    )
    p_pets_evaluate.add_argument("--out-dir", required=True)
    p_pets_evaluate.add_argument("--n-basis", type=int, required=True)
    p_pets_evaluate.add_argument("--n-points", type=int, required=True)
    p_pets_evaluate.add_argument("--topup-d-min", type=float, required=True)
    p_pets_evaluate.add_argument("--max-support-sg", type=float, default=DEFAULT_MAX_SUPPORT_SG)
    p_pets_evaluate.add_argument(
        "--thickness-grid",
        required=True,
        help="Comma-separated thicknesses in Angstrom, for example 800,1200,1500,1800.",
    )
    p_pets_evaluate.add_argument(
        "--paired-inverse-hand",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the exact elastic negative-thickness identity for the inverse-hand control.",
    )
    p_pets_evaluate.add_argument("--free-fraction", type=float, default=0.10)
    p_pets_evaluate.add_argument("--split-seed", type=int, default=42)
    p_pets_evaluate.add_argument("--instability-u", type=float, default=0.01)
    p_pets_evaluate.add_argument("--workers", type=int, default=0)
    p_pets_evaluate.add_argument("--chunksize", type=int, default=1)
    p_pets_evaluate.add_argument("--python", default="")
    p_pets_evaluate.add_argument("--resume", action="store_true")
    p_pets_evaluate.add_argument("--dry-run", action="store_true")
    p_pets_evaluate.set_defaults(func=cmd_pets_evaluate)

    p_pets = sub.add_parser(
        "export-pets",
        help="Export DIALS or XDS observations to PETS/Jana2020/Dyngo/diffBloch cif_pets files.",
    )
    p_pets.add_argument("--dials-refl", default="", help="DIALS integrated.refl or scaled.refl input.")
    p_pets.add_argument("--dials-expt", default="", help="DIALS integrated.expt or scaled.expt input.")
    p_pets.add_argument("--xds-dir", default="", help="XDS directory containing XDS.INP and INTEGRATE.HKL.")
    p_pets.add_argument("--out-dir", required=True)
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

    p_diffbloch_apply = sub.add_parser(
        "apply-diffbloch-model",
        help="Transfer a refined diffBloch CIF into a source RES while preserving SHELX instructions.",
    )
    p_diffbloch_apply.add_argument("--source-res", required=True)
    p_diffbloch_apply.add_argument("--refined-cif", required=True)
    p_diffbloch_apply.add_argument("--out-res", required=True)
    p_diffbloch_apply.add_argument("--audit-json", default="")
    p_diffbloch_apply.add_argument("--max-cell-length-deviation", type=float, default=0.01)
    p_diffbloch_apply.add_argument("--max-cell-angle-deviation", type=float, default=0.25)
    p_diffbloch_apply.add_argument(
        "--allow-npd",
        action="store_true",
        help="Write and return success for an NPD candidate. The audit remains labeled rejected_npd.",
    )
    p_diffbloch_apply.set_defaults(func=cmd_apply_diffbloch_model)
