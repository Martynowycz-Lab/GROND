"""Register ``grond manual`` and its stages: setup, status, scout, thickness, hand, feedback, scales, finish."""

from __future__ import annotations

import argparse


def register_manual(sub: argparse._SubParsersAction) -> None:
    """Register manual commands."""

    from grond.cli_support.tools import add_tool_config_args
    from grond.commands.utilities import (
        cmd_manual,
    )

    p_manual = sub.add_parser(
        "manual",
        help="Run the calculation one stage at a time, for inspection.",
        description=(
            "Run GROND as explicit setup, transform-scout, thickness, hand, shared-feedback, "
            "scale-audit, and finish stages. Each stage reads the same manual_workflow.json "
            "and prints the next command. Use `GROND manual STAGE --help` for only that stage's options."
        ),
    )
    manual_stages = p_manual.add_subparsers(dest="stage", required=True, metavar="STAGE")

    def add_manual_workflow_arg(stage_parser: argparse.ArgumentParser) -> None:
        """Add the ``--workflow`` option that every manual stage uses."""

        stage_parser.add_argument(
            "--workflow",
            default="",
            help="Workflow state written by setup. Defaults to ./manual_workflow.json.",
        )

    def add_manual_compute_args(stage_parser: argparse.ArgumentParser) -> None:
        """Add the options of the manual stages that run child commands: workflow, workers,
        Python, output and tool config.
        """

        add_manual_workflow_arg(stage_parser)
        stage_parser.add_argument(
            "--workers",
            type=int,
            default=0,
            help="Worker processes (default 0: 85%% of the cores).",
        )
        stage_parser.add_argument(
            "--python",
            default="",
            help="Python for the calculation subprocesses (default: the one running grond).",
        )
        stage_parser.add_argument(
            "--verbose-child-output",
            action="store_true",
            help="Echo every underlying child line. By default the terminal shows milestones while the complete stream remains in manual_STAGE.log.",
        )
        add_tool_config_args(stage_parser)

    p_manual_setup = manual_stages.add_parser(
        "setup",
        help="Create the workflow state from a model, merged HKL, and one or more dataset directories.",
        description="Create 00_setup/prep.json and the manual_workflow.json used by every later stage.",
    )
    p_manual_setup.add_argument("--out-dir", required=True, help="Directory for the stages of this manual run.")
    p_manual_setup.add_argument("--structure", required=True, help="Starting SHELX RES or INS model.")
    p_manual_setup.add_argument(
        "--baseline-hkl",
        default="",
        help="Conventional merged HKL used for the shared endpoint and starting-model comparison.",
    )
    p_manual_setup.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset as NAME=PATH[,dmin=...,hkl_transform=...]. Repeat once per crystal.",
    )
    p_manual_setup.add_argument(
        "--hkl-transform",
        default="identity",
        help="Initial transform assertion. The scout normally replaces it independently for each crystal.",
    )
    p_manual_setup.add_argument("--geometry-percentile", type=float, default=99.0)
    p_manual_setup.add_argument("--parity-setup-id", default="")
    p_manual_setup.add_argument("--parity-setup-manifest-sha256", default="")
    p_manual_setup.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes (default 0: 85%% of the cores).",
    )
    p_manual_setup.add_argument(
        "--python",
        default="",
        help="Python for the calculation subprocesses (default: the one running grond).",
    )
    p_manual_setup.add_argument(
        "--config",
        default="",
        help="Config file with the SHELXL and shelx2map paths (default: $GROND_CONFIG or grond.config.json).",
    )
    p_manual_setup.add_argument(
        "--shelxl",
        default="shelxl",
        help="SHELXL executable (default: from --config or PATH).",
    )
    p_manual_setup.add_argument("--notes", default="", help="Free-text provenance note stored in prep.json.")
    p_manual_setup.set_defaults(func=cmd_manual)

    p_manual_status = manual_stages.add_parser(
        "status",
        help="Show completed stages, decisions, timings, outputs, and the exact next command.",
    )
    add_manual_workflow_arg(p_manual_status)
    p_manual_status.add_argument(
        "--json",
        action="store_true",
        help="Print the machine-readable status object instead of the concise human report.",
    )
    p_manual_status.set_defaults(func=cmd_manual)

    p_manual_scout = manual_stages.add_parser(
        "scout",
        help="Choose one chemistry-blind proper HKL transform per dataset.",
        description="Rank determinant-positive transform candidates without using known configuration or hand.",
    )
    add_manual_compute_args(p_manual_scout)
    p_manual_scout.add_argument(
        "--skip",
        action="store_true",
        help="Trust the transforms recorded by setup. This is recorded as a user assertion.",
    )
    p_manual_scout.set_defaults(func=cmd_manual)

    p_manual_thickness = manual_stages.add_parser(
        "thickness",
        help="Select and pin thickness, gradient, frame offset, and the primary nuisance contract.",
        description=(
            "Run the primary thickness experiment at the requested N/Q grid. "
            "This stage does not move coordinates."
        ),
    )
    add_manual_compute_args(p_manual_thickness)
    p_manual_thickness.add_argument(
        "--skip",
        action="store_true",
        help="Reuse audited summaries supplied with one --summary DATASET=summary.json per crystal.",
    )
    p_manual_thickness.add_argument(
        "--summary",
        action="append",
        default=[],
        help="Pinned DATASET=/path/to/summary.json binding required with --skip.",
    )
    p_manual_thickness.add_argument(
        "--expanded",
        action="store_true",
        help="Also calculate the final-diagnostic sensitivity branch; not needed for the primary run.",
    )
    p_manual_thickness.add_argument(
        "--n-basis-grid", default="24,48", help="Comma-separated basis sizes for scout and confirmation."
    )
    p_manual_thickness.add_argument(
        "--thickness-grid-nm",
        default="50,75,110,150,180,220,260,300",
        help="Comma-separated mean thickness candidates in nm. A single value pins the thickness at every basis size.",
    )
    p_manual_thickness.add_argument(
        "--gradient-grid",
        default="0.0,0.4,0.8,1.0",
        help="Comma-separated fractional uniform-gradient widths.",
    )
    p_manual_thickness.add_argument(
        "--n-points",
        type=int,
        default=20,
        help="Rocking points per observation; the thickness scout uses at most 10.",
    )
    p_manual_thickness.set_defaults(func=cmd_manual)

    p_manual_hand = manual_stages.add_parser(
        "hand",
        help="Evaluate the simplified pre-feedback elastic hand control, then record an explicit choice.",
        description=(
            "The first call computes per-crystal all/free Klar-Palatinus evidence on the thickness-stage model "
            "and pauses. Repeat with --choice input or --choice inverse; the evidence is reused. This simplified "
            "expert path is not auto's held-out hand test."
        ),
    )
    add_manual_compute_args(p_manual_hand)
    p_manual_hand.add_argument(
        "--choice",
        choices=["report", "input", "inverse", "skip"],
        default="report",
        help="report computes evidence; input/inverse records the coordinate choice; skip keeps input coordinates without a hand claim.",
    )
    p_manual_hand.add_argument("--skip", action="store_true", help="Alias for --choice skip.")
    p_manual_hand.add_argument(
        "--min-adjusted-z-all",
        type=float,
        default=3.5,
        help="All-row adjusted ZKP magnitude required for the input/inverse recommendation.",
    )
    p_manual_hand.add_argument(
        "--min-adjusted-z-free",
        type=float,
        default=3.5,
        help="Held-out adjusted ZKP reporting threshold; weak evidence warns but does not veto the recommendation.",
    )
    p_manual_hand.add_argument("--min-free-rows", type=int, default=10)
    p_manual_hand.set_defaults(func=cmd_manual)

    p_manual_feedback = manual_stages.add_parser(
        "feedback",
        help="Run shared Bloch/one-cycle-SHELXL feedback and select the earliest clean row-optimal cycle.",
        description=(
            "For each macrocycle GROND recomputes each crystal's rows and nuisance scales, merges the corrected "
            "observations with overlap and inverse-variance weighting, and performs exactly one shared SHELXL LS1. "
            "Per-crystal row jobs never refine separate coordinates."
        ),
    )
    add_manual_compute_args(p_manual_feedback)
    p_manual_feedback.add_argument(
        "--shelxl", default="shelxl", help="SHELXL executable (default: from --config or PATH)."
    )
    p_manual_feedback.add_argument("--cycles", type=int, default=5, help="Maximum Bloch/shared-LS1 macrocycles.")
    p_manual_feedback.add_argument(
        "--n-basis",
        type=int,
        default=0,
        help="Fixed production basis size; zero retains adaptive production selection.",
    )
    p_manual_feedback.add_argument("--n-points", type=int, default=20, help="Rocking points per observation (default %(default)s).")
    p_manual_feedback.set_defaults(func=cmd_manual)

    p_manual_scales = manual_stages.add_parser(
        "scales",
        help="Record that the full pinned frame scales stay applied.",
    )
    add_manual_compute_args(p_manual_scales)
    p_manual_scales.set_defaults(func=cmd_manual)

    p_manual_finish = manual_stages.add_parser(
        "finish",
        help="Write maps plus the top-level endpoint, hand, thickness, row, scale, and timing summary.",
    )
    add_manual_workflow_arg(p_manual_finish)
    p_manual_finish.add_argument(
        "--skip", action="store_true", help="Skip map generation but still write all final summaries."
    )
    p_manual_finish.add_argument("--map-mode", choices=["native", "off"], default="native")
    p_manual_finish.add_argument(
        "--native-map-spacing", type=float, default=0.1, help="Native map grid spacing in Angstrom (default 0.1)."
    )
    from .endpoint_commands import add_model_box_args

    add_model_box_args(p_manual_finish)
    p_manual_finish.set_defaults(func=cmd_manual)
