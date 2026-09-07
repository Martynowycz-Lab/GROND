"""Register options shared by automatic and expert run commands."""

from __future__ import annotations

import argparse


def add_decision_controls(ap: argparse.ArgumentParser) -> None:
    """Add decision controls options in the established order."""

    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument(
        "--scout-max-rows", type=int, default=1000, help="Representative row limit for automatic transform/hand scouts."
    )
    ap.add_argument("--scout-n-basis", type=int, default=8, help="Basis size for cheap decision scouts.")
    ap.add_argument("--scout-n-points", type=int, default=3, help="Quadrature points for cheap decision scouts.")
    ap.add_argument(
        "--transform-rerank-top",
        type=int,
        default=3,
        help="Number of coarse transform-scout winners to rerun at better N/Q; 0 disables the second stage.",
    )
    ap.add_argument(
        "--transform-rerank-n-basis", type=int, default=24, help="Basis size for second-stage transform reranking."
    )
    ap.add_argument(
        "--transform-rerank-n-points",
        type=int,
        default=10,
        help="Quadrature points for second-stage transform reranking.",
    )
    ap.add_argument(
        "--transform-rerank-max-rows",
        type=int,
        default=0,
        help="Row limit for second-stage transform reranking; 0 reuses the preflight scout row budget.",
    )
    ap.add_argument(
        "--transform-candidates",
        default="auto",
        help=(
            "Semicolon-separated signed-axis transforms, or auto/common/off. "
            "Calibration-grade auto/common enumerates all 24 determinant-positive "
            "signed-axis permutations before the cell-metric filter."
        ),
    )
    ap.add_argument(
        "--transform-metric-tolerance",
        type=float,
        default=0.02,
        help=(
            "Maximum relative Frobenius mismatch between source and model "
            "reciprocal-cell metrics before a signed-axis transform is rejected. "
            "This filter uses cells only, never intensities, coordinates, chemistry, or hand."
        ),
    )
    ap.add_argument(
        "--transform-parity-policy",
        choices=["proper-only", "all-diagnostic"],
        default="proper-only",
        help=(
            "Use only determinant +1 transforms for automatic selection (default). "
            "all-diagnostic restores mixed-parity scouting for diagnostics, but cannot by itself "
            "support a named absolute-configuration claim."
        ),
    )
    ap.add_argument(
        "--transform-scout-jobs",
        type=int,
        default=0,
        help="Concurrent transform-scout subprocesses across candidates and datasets; 0 derives a core-safe value from --workers and --max-core-fraction.",
    )
    ap.add_argument(
        "--transform-scout-workers-per-job",
        type=int,
        default=1,
        help="Worker processes inside each transform-scout subprocess. Keep low when many scout jobs run concurrently.",
    )
    ap.add_argument(
        "--transform-metadata-prefilter",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "Use solved/XSCALE baseline-HKL overlap as a diagnostic ordering aid. "
            "Default auto never eliminates candidates; explicit on may eliminate candidates "
            "but is not eligible for parity calibration."
        ),
    )
    ap.add_argument(
        "--transform-metadata-max-candidates",
        type=int,
        default=3,
        help="Number of metadata-ranked candidates retained only by the explicit diagnostic --transform-metadata-prefilter on mode. Calibration-grade auto does not eliminate metric-compatible candidates.",
    )
    ap.add_argument(
        "--transform-metadata-min-pairs",
        type=int,
        default=100,
        help="Minimum paired positive-intensity rows required before trusting metadata transform ranking.",
    )
    ap.add_argument(
        "--transform-metadata-min-correlation",
        type=float,
        default=0.70,
        help="Minimum log-intensity correlation required before metadata can prefilter transform scouts.",
    )
    ap.add_argument(
        "--transform-metadata-min-correlation-gap",
        type=float,
        default=0.05,
        help="Correlation gap used only by the explicit diagnostic metadata-elimination mode; calibration-grade auto still Bloch-scouts every metric-compatible candidate.",
    )
    ap.add_argument(
        "--transform-metadata-max-rows",
        type=int,
        default=0,
        help="Maximum INTEGRATE/XDS rows used by metadata transform ranking; 0 uses all rows.",
    )
    ap.add_argument(
        "--trust-hkl-transform",
        action="store_true",
        help="Trust the supplied --hkl-transform or prep-file transform and skip Bloch transform scouting while keeping the rest of auto-decision enabled.",
    )
    ap.add_argument(
        "--hand-accept-delta",
        dest="hand_accept_delta",
        metavar="DELTA",
        type=float,
        default=0.015,
        help="Adaptive score margin required before switching to the alternate hand.",
    )
    ap.add_argument(
        "--hand-max-wR2-all",
        type=float,
        default=0.45,
        help="Maximum alternate-hand all-row wR2 allowed before automatic hand switching. Poor row contracts still report the hand diagnostic but keep the input hand.",
    )
    ap.add_argument(
        "--hand-max-free-wR2-all",
        type=float,
        default=0.50,
        help="Maximum alternate-hand free-row wR2 allowed before automatic hand switching.",
    )
    ap.add_argument(
        "--hand-min-free-wR2-gain",
        type=float,
        default=0.0,
        help="Minimum free-row wR2 improvement required before automatic scout-stage hand switching.",
    )
    ap.add_argument(
        "--hand-min-adjusted-z",
        type=float,
        default=2.0,
        help=(
            "Minimum absolute common-source-sigma Klar-Palatinus adjusted hand Z "
            "required before a matched hand decision may train or consume an "
            "external setup-parity calibration. Weaker decisions remain "
            "residual-preference diagnostics."
        ),
    )
    ap.add_argument(
        "--hand-flip-zkp-all",
        type=float,
        default=3.5,
        help=(
            "Invert the final acentric model once when the all-row Klar-Palatinus adjusted Z "
            "favors the inverse by at least this magnitude (default 3.5). The held-out Z is "
            "reported and may warn, but does not veto a decisive all-row result."
        ),
    )
    ap.add_argument("--hand-min-z-all", type=float, default=2.0, help=argparse.SUPPRESS)
    ap.add_argument("--hand-min-z-free", type=float, default=0.5, help=argparse.SUPPRESS)
    ap.add_argument("--hand-min-free-rows", type=int, default=10, help=argparse.SUPPRESS)
    ap.add_argument("--stop-on-bad-preflight", action="store_true", default=False)
    ap.add_argument("--stop-on-bad-fit", action="store_true", default=False)
    ap.add_argument(
        "--stop-if-wR2-above",
        type=float,
        default=0.75,
        help="Bad-data stop threshold for native all-row dynamic wR2 when --stop-on-bad-fit is enabled.",
    )
    ap.add_argument(
        "--min-bloch-wR2-gain",
        type=float,
        default=-0.01,
        help="Minimum wR2 gain over Born before structure update is considered reliable.",
    )
    ap.add_argument(
        "--feature-accept-delta",
        type=float,
        default=0.002,
        help="Minimum adaptive-score improvement before keeping a staged row-resolution/absorption/jitter/error/dose feature family.",
    )
