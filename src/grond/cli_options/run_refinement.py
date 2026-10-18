"""Register options shared by automatic and expert run commands."""

from __future__ import annotations

import argparse


def add_scales_and_screening(ap: argparse.ArgumentParser) -> None:
    """Add scales and screening options in the established order."""

    ap.add_argument(
        "--production-scales",
        action="store_true",
        default=False,
        help="Fit nuisance scales on all rows for a final diagnostic; validation and automatic selection use work-only scales by default.",
    )
    ap.add_argument(
        "--validation-scales",
        dest="production_scales",
        action="store_false",
        help="Fit nuisance scales on work rows only (the default).",
    )
    ap.add_argument(
        "--scale-spline-n-basis-grid",
        default="32",
        help="Frame scales are a smooth P-spline in scan angle: basis sizes tried by the inner orbit CV; one value pins it.",
    )
    ap.add_argument(
        "--scale-spline-lambda-grid",
        default="0.3,1,3,10,30,100",
        help="P-spline smoothing penalties tried by the inner orbit CV; one value pins it.",
    )


def add_structure_updates(ap: argparse.ArgumentParser) -> None:
    """Add structure updates options in the established order."""

    ap.add_argument(
        "--structure-update",
        choices=["auto", "shelxl_feedback", "off"],
        default="auto",
        help="Structure-update engine. Auto uses the high-N/Q SHELXL feedback loop for full runs when SHELXL is available, otherwise off.",
    )


def add_resolution(ap: argparse.ArgumentParser) -> None:
    """Add resolution options in the established order."""

    ap.add_argument(
        "--endpoint-min-d-spacing",
        type=float,
        default=0.0,
        help=(
            "Resolution limit of the corrected HKL file and its statistics in Å (default 0: as --min-d-spacing)."
        ),
    )


def add_dataset_policy(ap: argparse.ArgumentParser) -> None:
    """Add dataset policy options in the established order."""

    ap.add_argument(
        "--dataset-policy",
        choices=["auto", "off"],
        default="auto",
        help="For multi-dataset jobs, automatically keep good crystals at full weight, downweight marginal crystals, and keep weak nonredundant crystals endpoint-only.",
    )
    ap.add_argument(
        "--multi-endpoint-merge-key",
        choices=["model_symmetry", "model_symmetry_friedel", "exact_signed_hkl"],
        default="model_symmetry",
        help=(
            "Multi-dataset corrected-HKL merge key before each shared SHELXL LS step. "
            "model_symmetry merges symmetry-equivalent corrected observations while preserving Friedel opposites in acentric groups."
        ),
    )
