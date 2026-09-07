"""Register optional forensic controls while respecting production q-exponent defaults."""

from __future__ import annotations

import argparse


def add_forensic_args(ap: argparse.ArgumentParser, *, expose_lambda_controls: bool = True) -> None:
    """Attach row-space forensic and observation-polish options to a parser."""

    ap.add_argument(
        "--forensic-gate",
        dest="forensic_gate",
        action="store_true",
        default=True,
        help="Run post-final row-space forensic audits.",
    )
    ap.add_argument("--no-forensic-gate", dest="forensic_gate", action="store_false")
    ap.add_argument(
        "--forensic-depth",
        choices=["quick", "standard", "full"],
        default="standard",
        help="quick runs residual/pre-merge audits; standard adds dose and two main observation-correction holdouts; full adds edge-frame and sigma audits.",
    )
    ap.add_argument("--forensic-bins", type=int, default=10)
    ap.add_argument("--forensic-top-rows", type=int, default=200)
    ap.add_argument("--forensic-top-groups", type=int, default=200)
    ap.add_argument(
        "--observation-polish",
        choices=["auto", "on", "off"],
        default="auto",
        help="Promote held-out observation corrections into corrected-HKL endpoints when both main gates pass.",
    )
    ap.add_argument("--observation-polish-combine", choices=["ensemble", "best_all", "best_free"], default="ensemble")
    if expose_lambda_controls:
        ap.add_argument(
            "--observation-polish-lambda-grid",
            default="",
            help="Diagnostic q-exponent grid for promoted observation-polish endpoints. Empty uses the full Bloch/Born correction.",
        )
    else:
        ap.set_defaults(observation_polish_lambda_grid="")
    ap.add_argument("--observation-polish-min-raw-wR2-gain", type=float, default=0.002)
    ap.add_argument("--observation-polish-endpoint-wR2-tolerance", type=float, default=0.0005)
    ap.add_argument(
        "--forensic-maps",
        action="store_true",
        help="Write maps for promoted observation-polish endpoints during standalone forensics.",
    )
