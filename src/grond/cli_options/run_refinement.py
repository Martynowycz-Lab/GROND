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
        "--scale-model",
        choices=["pspline", "per-frame"],
        default="pspline",
        help=(
            "Frame-scale family fitted in every native stage. pspline (default): smooth P-spline in scan angle "
            "on the scale-fit rows, n_basis and lambda chosen per crystal by inner orbit CV; scale polish, "
            "compact scale-contract selection and endpoint-calibration replays are then off. per-frame: free "
            "per-frame scales with post-hoc compression."
        ),
    )
    ap.add_argument(
        "--scale-spline-n-basis-grid",
        default="32",
        help="P-spline basis sizes tried by the inner orbit CV; one value pins it.",
    )
    ap.add_argument(
        "--scale-spline-lambda-grid",
        default="0.3,1,3,10,30,100",
        help="P-spline smoothing penalties tried by the inner orbit CV; one value pins it.",
    )


def add_structure_updates(ap: argparse.ArgumentParser) -> None:
    """Add structure updates options in the established order."""

    ap.add_argument("--structure-n-basis", type=int, default=24)
    ap.add_argument(
        "--structure-n-points",
        type=int,
        default=None,
        help=(
            "Profile points of the native-gradient structure update. Default: 6 per 20 window-derived final "
            "points (6 for the legacy 0.004 1/A window, 15 for the 0.010 default, 18 for the DIALS bbox)."
        ),
    )
    ap.add_argument("--structure-objective-rows", type=int, default=400)
    ap.add_argument("--structure-max-nfev", type=int, default=5)
    ap.add_argument(
        "--structure-early-stop-window",
        type=int,
        default=64,
        help="Stop a structure-update block after this many residual calls without a meaningful data-RMS gain; 0 disables.",
    )
    ap.add_argument(
        "--structure-early-stop-min-evals",
        type=int,
        default=0,
        help="Minimum residual calls before a plateau stop can trigger; 0 uses the engine's parameter-count-aware floor.",
    )
    ap.add_argument(
        "--structure-early-stop-min-improvement",
        type=float,
        default=2e-5,
        help="Minimum RMS improvement counted as progress for plateau stopping.",
    )
    ap.add_argument(
        "--structure-early-stop-metric",
        choices=["total_rms", "data_rms"],
        default="data_rms",
        help="Trace column used by the structure-update plateau stop.",
    )
    ap.add_argument(
        "--structure-update",
        choices=["auto", "native_gradient", "shelxl_feedback", "off"],
        default="auto",
        help="Structure-update engine. Auto uses the high-N/Q SHELXL feedback loop for full runs when SHELXL is available; native_gradient is the slower finite-difference diagnostic path.",
    )
    ap.add_argument("--structure-blocks", default="adp_ueq")
    ap.add_argument("--structure-atom-selection", choices=["nonh", "all", "labels"], default="nonh")
    ap.add_argument("--restraints", choices=["none", "start_geometry", "cctbx_rigu"], default="cctbx_rigu")


def add_scale_polish(ap: argparse.ArgumentParser) -> None:
    """Add scale polish options in the established order."""

    ap.add_argument(
        "--scale-polish",
        choices=["auto", "on", "off"],
        default="auto",
        help="Run a post-hoc scale-window compression gate after the final Bloch row table.",
    )
    ap.add_argument(
        "--scale-polish-window-deg",
        type=float,
        default=5.0,
        help="Maximum physical frame-window width tested by the scale-polish gate.",
    )
    ap.add_argument(
        "--scale-polish-frame-models",
        default="adaptive,fixed",
        help="Frame-scale compression models tested by the scale-polish gate.",
    )
    ap.add_argument(
        "--scale-polish-adaptive-counts",
        default="auto",
        help="Adaptive neighboring-frame segment counts for scale polish; percentages such as 50%% are accepted.",
    )
    ap.add_argument(
        "--scale-polish-detector-bins",
        default="0,2,4,8",
        help="Detector flat-field grids tested on frozen predictions. Automatic promotion uses the smallest detector field that earns the validation gates; 8x8 remains diagnostic without independent calibration.",
    )
    ap.add_argument(
        "--scale-polish-objective",
        choices=["row_mr", "native", "sigma"],
        default="row_mr",
        help="Fit compact frame scales on work rows with the JANA-style row weight by default.",
    )
    ap.add_argument("--scale-polish-tolerance", type=float, default=0.01)
    ap.add_argument(
        "--scale-polish-selection-metrics",
        default="free_Rall,work_Rall",
        help=(
            "Metrics protected when selecting compact scales. Default: held-out and work JANA-style "
            "Rall; wRall is a secondary report/tie-break and endpoint/native wR2 are excluded."
        ),
    )
    ap.add_argument(
        "--scale-polish-scan-offsets",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Test all block-boundary offsets in the scale-polish gate.",
    )
    ap.add_argument(
        "--multi-scale-contract-selection",
        choices=["auto", "off"],
        default="auto",
        help=(
            "For multi-dataset runs, replay strict compact per-crystal scales through the complete combined "
            "endpoint and promote the smallest contract that passes physics, held-out, coverage, and SHELXL gates."
        ),
    )
    ap.add_argument(
        "--multi-scale-min-reduction",
        type=float,
        default=0.10,
        help="Minimum total scale-parameter reduction required for compact promotion.",
    )
    ap.add_argument(
        "--multi-scale-require-bic-improvement",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Deprecated diagnostic toggle; BIC is reported but does not select scale topology.",
    )
    ap.add_argument(
        "--multi-scale-max-row-rall-cost",
        "--multi-scale-max-row-wr2-cost",
        dest="multi_scale_max_row_rall_cost",
        type=float,
        default=0.01,
        help="Maximum absolute compact JANA-style row Rall cost on each of all, work, and free rows.",
    )
    ap.add_argument(
        "--multi-scale-max-endpoint-r1-cost",
        type=float,
        default=0.005,
        help="Maximum absolute compact R1all and R1gt endpoint cost.",
    )
    ap.add_argument(
        "--multi-scale-max-endpoint-wr2-cost",
        type=float,
        default=0.01,
        help="Maximum absolute compact endpoint wR2all cost.",
    )
    ap.add_argument("--multi-scale-goof-min", type=float, default=0.85)
    ap.add_argument("--multi-scale-goof-max", type=float, default=1.15)
    ap.add_argument(
        "--multi-scale-max-input-row-loss-fraction",
        type=float,
        default=0.001,
        help="Maximum corrected-row loss allowed when materializing compact scales.",
    )
    ap.add_argument(
        "--multi-scale-min-bloch-rall-gain",
        "--multi-scale-min-bloch-wr2-gain",
        dest="multi_scale_min_bloch_rall_gain",
        type=float,
        default=0.002,
        help="Minimum pooled Born-minus-Bloch row Rall gain and retained per-dataset gain.",
    )
    ap.add_argument(
        "--multi-scale-max-dataset-row-rall-cost",
        "--multi-scale-max-dataset-row-wr2-cost",
        dest="multi_scale_max_dataset_row_rall_cost",
        type=float,
        default=0.015,
        help="Maximum compact-versus-full row Rall cost on any individual dataset/split.",
    )
    ap.add_argument(
        "--multi-scale-max-preexisting-bloch-gain-loss",
        type=float,
        default=0.01,
        help="Maximum additional Born-minus-Bloch loss allowed where the full-scale control already lacks a split-level Bloch win.",
    )


def add_resolution(ap: argparse.ArgumentParser) -> None:
    """Add resolution options in the established order."""

    ap.add_argument(
        "--endpoint-min-d-spacing",
        type=float,
        default=0.0,
        help=(
            "Optional additional d-spacing cutoff for final combined row statistics and corrected-HKL endpoints. "
            "Zero keeps the row-selection cutoff; use an explicit value for an apples-to-apples reporting endpoint."
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
