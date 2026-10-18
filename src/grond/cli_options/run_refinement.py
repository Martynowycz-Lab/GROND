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
    ap.add_argument(
        "--resolution-limit-gate",
        dest="resolution_limit_gate",
        action="store_true",
        default=True,
        help="Promote per-dataset and merged resolution limits from held-out shell stability and GROND row/MR residuals.",
    )
    ap.add_argument("--no-resolution-limit-gate", dest="resolution_limit_gate", action="store_false")
    ap.add_argument("--resolution-shells", type=int, default=10)
    ap.add_argument("--resolution-strict-split-cc", type=float, default=0.30)
    ap.add_argument("--resolution-strict-mean-i-over-sigma", type=float, default=0.50)
    ap.add_argument("--resolution-exploratory-split-cc", type=float, default=0.20)
    ap.add_argument("--resolution-exploratory-mean-i-over-sigma", type=float, default=0.25)
    ap.add_argument("--resolution-min-common-pairs", type=int, default=8)
    ap.add_argument("--resolution-min-shell-rows", type=int, default=20)
    ap.add_argument("--resolution-min-strong-fraction", type=float, default=0.005)
    ap.add_argument("--resolution-max-shell-mwrall", type=float, default=0.60)
    ap.add_argument("--resolution-max-free-work-mwr-gap", type=float, default=0.12)
    ap.add_argument("--resolution-min-free-rows", type=int, default=5)
    ap.add_argument("--resolution-max-cumulative-mwr-penalty", type=float, default=0.03)


def add_dataset_policy(ap: argparse.ArgumentParser) -> None:
    """Add dataset policy options in the established order."""

    ap.add_argument(
        "--dataset-policy",
        choices=["auto", "off"],
        default="auto",
        help="For multi-dataset jobs, automatically keep good crystals at full weight, downweight marginal crystals, and keep weak nonredundant crystals endpoint-only.",
    )
    ap.add_argument("--dataset-policy-min-rows", type=int, default=100)
    ap.add_argument("--dataset-policy-min-weight", type=float, default=0.25)
    ap.add_argument("--dataset-policy-primary-min-quality", type=float, default=0.75)
    ap.add_argument("--dataset-policy-min-raw-quality", type=float, default=0.20)
    ap.add_argument("--dataset-policy-min-cc", type=float, default=0.70)
    ap.add_argument("--dataset-policy-max-wr2", type=float, default=0.75)
    ap.add_argument("--dataset-policy-max-r1", type=float, default=0.60)
    ap.add_argument("--dataset-policy-unique-gain-fraction", type=float, default=0.01)
    ap.add_argument("--dataset-policy-wr2-scale", type=float, default=0.08)
    ap.add_argument("--dataset-policy-r1-scale", type=float, default=0.08)
    ap.add_argument("--dataset-policy-cc-scale", type=float, default=0.04)
    ap.add_argument(
        "--multi-endpoint-merge-key",
        choices=["model_symmetry", "model_symmetry_friedel", "exact_signed_hkl"],
        default="model_symmetry",
        help=(
            "Multi-dataset corrected-HKL merge key before each shared SHELXL LS step. "
            "model_symmetry merges symmetry-equivalent corrected observations while preserving Friedel opposites in acentric groups."
        ),
    )
