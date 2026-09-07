"""Argument declarations for the continuous thickness polish workflow.

The adapter supplies runtime defaults explicitly. Keep option order, numerical
defaults, choices, and help text identical to the established command contract.
"""
from __future__ import annotations

import argparse


def build_parser(description: str, *, default_hkl, default_res, max_support_sg, native_regularizer, physical_grad_max) -> argparse.ArgumentParser:
    """Build the parser without reading data or starting a calculation."""

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--hkl", default=str(default_hkl))
    ap.add_argument("--res", default=str(default_res))
    ap.add_argument("--out-dir", default="outputs/continuous_thickness_polish")
    ap.add_argument("--hkl-transform", default="identity")
    ap.add_argument("--n-basis", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=10)
    ap.add_argument("--sg-window", type=float, default=0.004)
    ap.add_argument("--sg-window-mode", default="fixed")
    ap.add_argument("--sg-window-min", type=float, default=0.003)
    ap.add_argument("--sg-window-max", type=float, default=0.012)
    ap.add_argument("--sg-window-scale", type=float, default=1.0)
    ap.add_argument("--sg-window-frame-floor", type=float, default=0.5)
    ap.add_argument("--profile-quadrature", choices=["uniform", "trapezoid", "gauss_legendre"], default="uniform")
    ap.add_argument("--profile-accumulation", choices=["sum", "mean"], default="sum")
    ap.add_argument("--topup-d-min", type=float, default=0.8)
    ap.add_argument("--max-support-sg", type=float, default=max_support_sg)
    ap.add_argument("--min-d-spacing", type=float, default=0.7)
    ap.add_argument("--min-peak", type=float, default=90.0)
    ap.add_argument("--geometry-percentile", type=float, default=100.0)
    ap.add_argument("--target-mode", choices=list(native_regularizer.TARGET_MODES), default="iobs")
    ap.add_argument("--include-nonpositive", action="store_true")
    ap.add_argument("--start-thickness", type=float, default=1200.0)
    ap.add_argument("--start-grad", type=float, default=1.0)
    ap.add_argument("--thickness-points", type=int, default=5)
    ap.add_argument("--frame-offset", type=float, default=0.2)
    ap.add_argument("--group-width-deg", type=float, default=0.168968)
    ap.add_argument("--detector-scale-bins", type=int, default=8)
    ap.add_argument("--sigma-a", type=float, default=0.0)
    ap.add_argument("--sigma-b-corr", type=float, default=0.0)
    ap.add_argument("--sigma-c-peak", type=float, default=0.15)
    ap.add_argument("--row-resolution-b", type=float, default=0.0)
    ap.add_argument("--scale-smoothness", type=float, default=0.0)
    ap.add_argument("--matrix-convention", default="j_minus_i")
    ap.add_argument("--free-fraction", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--scale-fit-split", choices=["all", "work", "free"], default="work")
    ap.add_argument(
        "--selection-metric",
        choices=["work_wR2", "all_wR2", "free_wR2", "all_R1", "all_R1gt"],
        default="work_wR2",
    )
    ap.add_argument("--strong-rows", type=int, default=500)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--bracket-frac", type=float, default=0.35)
    ap.add_argument("--t-min", type=float, default=None, help="Optional lower clamp on the local thickness bracket.")
    ap.add_argument("--t-max", type=float, default=None, help="Optional upper clamp on the local thickness bracket.")
    ap.add_argument("--grad-min", type=float, default=0.0)
    ap.add_argument("--grad-max", type=float, default=physical_grad_max)
    ap.add_argument("--grid-thickness-step", type=float, default=100.0)
    ap.add_argument("--grid-grad-step", type=float, default=0.2)
    ap.add_argument("--powell-maxiter", type=int, default=24)
    ap.add_argument("--min-promotion-gain", type=float, default=5e-5, help="Minimum selected-objective gain required before replacing the grid value.")
    ap.add_argument("--max-free-wr2-loss", type=float, default=0.002, help="Maximum tolerated free-row wR2 loss for a promoted continuous thickness.")
    ap.add_argument("--max-all-r1-loss", type=float, default=0.001, help="Maximum tolerated all-row R1 loss for a promoted continuous thickness.")
    ap.add_argument("--progress-every", type=int, default=100)
    return ap
