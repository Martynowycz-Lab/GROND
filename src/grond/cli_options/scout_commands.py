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
    p_scout.add_argument("--workers", type=int, default=0)
    p_scout.add_argument(
        "--transform-parity-policy",
        choices=["proper-only", "all-diagnostic"],
        default="proper-only",
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
    p_hand.add_argument("--workers", type=int, default=0)
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
    p_thick.set_defaults(func=cmd_thickness_scout)
