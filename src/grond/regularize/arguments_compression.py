"""Argument declarations for the scale compression polish workflow.

The adapter supplies runtime defaults explicitly. Keep option order, numerical
defaults, choices, and help text identical to the established command contract.
"""
from __future__ import annotations

import argparse


def build_parser(description: str) -> argparse.ArgumentParser:
    """Build the parser without reading data or starting a calculation."""

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--row-table", required=True)
    ap.add_argument("--summary", default="", help="Existing GROND run summary. Defaults to row-table parent/summary.json.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--title", default="GROND scale-compression polish")
    ap.add_argument("--raw-dyn-column", default="dyn_raw")
    ap.add_argument("--raw-born-column", default="born_raw")
    ap.add_argument("--baseline-dyn-column", default="dyn_scaled")
    ap.add_argument("--baseline-born-column", default="born_scaled")
    ap.add_argument("--fit-split", default="", help="Rows used for fitting scales. Defaults to summary scale_fit_split or all.")
    ap.add_argument("--frame-models", default="adaptive,fixed", help="Comma list of frame-scale compression families to test: adaptive,fixed.")
    ap.add_argument("--adaptive-frame-counts", default="auto", help="Adaptive neighboring-frame segment counts. Use auto, off, integers, or percentages such as 50%%.")
    ap.add_argument("--group-strides", default="1,2,4,8,16,32,64,128,global")
    ap.add_argument("--window-scan-deg", type=float, default=0.0, help="Also scan every integer group stride up to this physical window size.")
    ap.add_argument("--scan-offsets", action="store_true", help="For each non-global stride, test every possible block-boundary offset.")
    ap.add_argument("--detector-bins", default="0,2,4,8", help="Detector flat-field models. Automatic policy compares none, 2x2, 4x4, and diagnostic 8x8 fields on frozen predictions.")
    ap.add_argument("--detector-ridge", type=float, default=0.05, help="Dimensionless shrinkage of detector factors toward unity.")
    ap.add_argument("--detector-smoothness", type=float, default=0.10, help="Dimensionless nearest-neighbor detector-field smoothness penalty.")
    ap.add_argument("--detector-min-rows", type=int, default=20, help="Minimum work rows required for every independently supported detector tile.")
    ap.add_argument("--detector-min-holdout-improvement", type=float, default=0.002, help="Minimum absolute held-out wR2 improvement required on multiple deterministic splits.")
    ap.add_argument("--detector-required-improved-splits", type=int, default=2)
    ap.add_argument("--detector-max-bloch-gain-loss", type=float, default=0.002, help="Maximum loss of held-out Born-minus-Bloch wR2 advantage on any deterministic split.")
    ap.add_argument("--detector-min-field-correlation", type=float, default=0.50)
    ap.add_argument("--detector-max-field-log-rms", type=float, default=0.10)
    ap.add_argument("--detector-independent-calibration", action="store_true", help="Allow an 8x8 detector field to be promotable because an independent detector calibration supports it.")
    ap.add_argument("--scale-objective", choices=["row_mr", "native", "sigma"], default="row_mr")
    ap.add_argument("--tolerance", type=float, default=0.01)
    ap.add_argument(
        "--selection-metrics",
        default="free_Rall,work_Rall",
        help=(
            "Row-factor gates used to earn scale compression. The default protects held-out and work "
            "JANA-style Rall; wRall is a secondary tie-break/report, and SHELXL/native wR2 do not select topology."
        ),
    )
    ap.add_argument("--instability-u", type=float, default=0.01)
    ap.add_argument("--observed-sigma-cut", type=float, default=3.0)
    ap.add_argument("--native-wght-a", type=float, default=float("nan"))
    ap.add_argument("--native-wght-b", type=float, default=float("nan"))
    return ap
