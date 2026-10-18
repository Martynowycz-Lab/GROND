"""Register the scout commands: scout, hand-check and thickness-scout (alias thickness)."""

from __future__ import annotations

import argparse


def register_scouts(sub: argparse._SubParsersAction) -> None:
    """Register scouts commands."""

    from grond.cli_support.tools import add_tool_config_args
    from grond.commands.hand import cmd_hand_check
    from grond.commands.scouting import cmd_scout

    p_scout = sub.add_parser(
        "scout",
        help="Run only preflight and hand-invariant proper HKL transform selection.",
    )
    p_scout.add_argument("--prep", required=True, help="Prep JSON written by GROND prep/import-dials.")
    p_scout.add_argument("--out-dir", required=True)
    p_scout.add_argument("--python", default="")
    add_tool_config_args(p_scout)
    p_scout.add_argument("--dry-run", action="store_true")
    p_scout.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p_scout.add_argument("--workers", type=int, default=0)
    p_scout.add_argument("--max-core-fraction", type=float, default=0.85)
    p_scout.add_argument("--chunksize", type=int, default=0)
    p_scout.add_argument("--scout-max-rows", type=int, default=1000)
    p_scout.add_argument("--scout-n-basis", type=int, default=8)
    p_scout.add_argument("--scout-n-points", type=int, default=3)
    p_scout.add_argument("--transform-rerank-top", type=int, default=3)
    p_scout.add_argument("--transform-rerank-n-basis", type=int, default=24)
    p_scout.add_argument("--transform-rerank-n-points", type=int, default=10)
    p_scout.add_argument("--transform-rerank-max-rows", type=int, default=0)
    p_scout.add_argument("--transform-candidates", default="auto")
    p_scout.add_argument("--transform-metric-tolerance", type=float, default=0.02)
    p_scout.add_argument(
        "--transform-parity-policy",
        choices=["proper-only", "all-diagnostic"],
        default="proper-only",
    )
    p_scout.add_argument("--transform-canonicalization", choices=["point-group", "off"], default="point-group")
    p_scout.add_argument("--split-candidate-set", choices=["shared-closure", "per-crystal"], default="shared-closure")
    p_scout.add_argument("--transform-scout-jobs", type=int, default=0)
    p_scout.add_argument("--transform-scout-workers-per-job", type=int, default=1)
    p_scout.add_argument(
        "--transform-metadata-prefilter",
        choices=["auto", "on", "off"],
        default="auto",
    )
    p_scout.add_argument("--transform-metadata-max-candidates", type=int, default=3)
    p_scout.add_argument("--transform-metadata-min-pairs", type=int, default=100)
    p_scout.add_argument("--transform-metadata-min-correlation", type=float, default=0.70)
    p_scout.add_argument("--transform-metadata-min-correlation-gap", type=float, default=0.05)
    p_scout.add_argument("--transform-metadata-max-rows", type=int, default=0)
    p_scout.add_argument(
        "--trust-hkl-transform",
        action="store_true",
        help="Skip scouting and freeze the proper transform already recorded in the prep file.",
    )
    p_scout.set_defaults(func=cmd_scout)

    p_hand = sub.add_parser(
        "hand-check",
        help=(
            "Run a matched fixed-nuisance hand comparison from a completed native summary. "
            "The default elastic negative-thickness method shares each eigendecomposition."
        ),
    )
    p_hand.add_argument("--out-dir", required=True)
    p_hand.add_argument("--python", default="")
    add_tool_config_args(p_hand)
    p_hand.add_argument("--dry-run", action="store_true")
    p_hand.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p_hand.add_argument("--workers", type=int, default=0)
    p_hand.add_argument("--max-core-fraction", type=float, default=0.85)
    p_hand.add_argument("--chunksize", type=int, default=0)
    p_hand.add_argument(
        "--row-plan-cache",
        choices=["off", "exact", "readwrite", "rebuild"],
        default="off",
    )
    p_hand.add_argument(
        "--prediction-cache-compression",
        choices=["compressed", "none"],
        default="none",
    )
    p_hand.add_argument(
        "--source-summary",
        required=True,
        help="summary.json from a completed fixed-model or final native row stage.",
    )
    p_hand.add_argument(
        "--hand-check-method",
        choices=["negative-thickness", "explicit"],
        default="negative-thickness",
        help=(
            "negative-thickness disables absorption and evaluates both hands from one elastic "
            "eigensystem; explicit rebuilds the coordinate-inverted model and can preserve absorption."
        ),
    )
    p_hand.add_argument(
        "--hand-check-absorption",
        choices=["elastic", "preserve"],
        default="elastic",
        help=(
            "elastic disables absorptive terms in both hands and is required by the "
            "negative-thickness identity. preserve is available only with explicit "
            "coordinate inversion."
        ),
    )
    p_hand.add_argument(
        "--hand-check-n-basis",
        type=int,
        default=0,
        help="Override the source N value. Zero reuses the exact source value.",
    )
    p_hand.add_argument(
        "--hand-check-n-points",
        type=int,
        default=0,
        help="Override the source Q value. Zero reuses the exact source value.",
    )
    p_hand.set_defaults(func=cmd_hand_check)


def register_thickness(sub: argparse._SubParsersAction) -> None:
    """Register thickness commands."""

    from grond.cli_options.common import add_common_run_args
    from grond.commands.thickness import cmd_thickness_scout

    p_thick = sub.add_parser(
        "thickness-scout",
        aliases=["thickness"],
        help="Audit effective thickness sensitivity to row contract and accepted nuisance terms.",
    )
    add_common_run_args(p_thick, expose_lambda_controls=False)
    p_thick.set_defaults(mode="regularize", n_points=10)
    p_thick.add_argument(
        "--n-basis-grid",
        default="24,48",
        help="Comma grid of basis sizes. The first value uses the broad grid; later values use a local confirmation grid by default.",
    )
    p_thick.add_argument(
        "--thickness-grid-nm",
        default="50,75,110,150,180,220,260,300",
        help="Broad central-thickness grid in nanometers.",
    )
    p_thick.add_argument(
        "--gradient-grid", default="0.0,0.4,0.8,1.0", help="Physical uniform-gradient fraction grid (0 through 1)."
    )
    p_thick.add_argument(
        "--feature-modes", default="bare,final-diagnostic", help="Comma grid: bare and/or final-diagnostic."
    )
    p_thick.add_argument(
        "--local-confirmation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use local thickness/gradient grids for basis values after the first broad scout.",
    )
    p_thick.add_argument("--local-window-nm", type=float, default=30.0)
    p_thick.add_argument("--local-step-nm", type=float, default=10.0)
    p_thick.add_argument("--local-gradient-window", type=float, default=0.2)
    p_thick.add_argument("--local-gradient-step", type=float, default=0.2)
    p_thick.add_argument("--thickness-min-nm", type=float, default=10.0)
    p_thick.add_argument("--thickness-max-nm", type=float, default=500.0)
    p_thick.add_argument("--gradient-min", type=float, default=0.0)
    p_thick.add_argument("--gradient-max", type=float, default=1.0)
    p_thick.add_argument(
        "--warning-shift-nm",
        type=float,
        default=75.0,
        help="Warn when a selected thickness is this far from the promoted reference.",
    )
    p_thick.add_argument(
        "--thickness-scout-max-rows",
        type=int,
        default=0,
        help="Optional row cap for scout runs. 0 uses the selected production row set.",
    )
    p_thick.add_argument(
        "--thickness-scout-thickness-points",
        type=int,
        default=5,
        help="Thickness quadrature points inside each candidate thickness model.",
    )
    p_thick.add_argument("--thickness-scout-final-sg-jitter-mode", default="gaussian")
    p_thick.add_argument("--thickness-scout-final-sg-jitter-sigma0", type=float, default=0.0015)
    p_thick.add_argument("--thickness-scout-final-sg-jitter-sigma-d-inv", type=float, default=0.0)
    p_thick.add_argument("--thickness-scout-final-sg-jitter-sigma-frame", type=float, default=0.0)
    p_thick.add_argument("--thickness-scout-final-sg-jitter-points", type=int, default=3)
    p_thick.add_argument("--thickness-scout-final-absorption-mode", default="midgley_like")
    p_thick.add_argument("--thickness-scout-final-absorption-mu", type=float, default=0.00005)
    p_thick.add_argument("--thickness-scout-final-absorption-frac", type=float, default=0.02)
    p_thick.add_argument("--thickness-scout-final-absorption-mu-g2", type=float, default=0.0)
    p_thick.add_argument("--thickness-scout-final-absorption-mu-theta2", type=float, default=0.0)
    p_thick.add_argument("--thickness-scout-final-absorption-tds-b", type=float, default=0.0)
    p_thick.set_defaults(func=cmd_thickness_scout)
