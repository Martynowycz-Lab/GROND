"""Options of `grond refine` and `grond finish`."""

from __future__ import annotations

import argparse

from grond.runtime import auto_worker_count


def register_push(sub: argparse._SubParsersAction) -> None:
    """Register push commands."""

    from grond.cli_options.run_physics import add_crystal_overrides, add_profile_coordinates
    from grond.cli_support.tools import add_tool_config_args, default_shelxl_executable
    from grond.cli_support.values import FULL_Q_LAMBDA
    from grond.commands.push import cmd_push_endpoint

    p_push = sub.add_parser(
        "refine",
        help="Recalculate the corrections for a model you refined yourself, with the settings of a run.",
    )
    p_push.add_argument("--checkpoint", default="", help="grond_continue_checkpoint.json of a finished `grond auto` run.")
    p_push.add_argument(
        "--source-summary",
        default="",
        help="Existing native_row_regularize summary whose selected nuisance values should be pinned.",
    )
    p_push.add_argument(
        "--start-res",
        default="",
        help="Model to continue from, for example one you refined in SHELXL or Olex2.",
    )
    p_push.add_argument("--out-dir", required=True, help="Output directory.")
    p_push.add_argument("--python", default="", help="Python for the calculation subprocesses (default: the one running grond).")
    add_tool_config_args(p_push)
    p_push.add_argument("--dry-run", action="store_true")
    p_push.add_argument("--hkl", default="", help="Override HKL path if the source summary does not record it.")
    p_push.add_argument(
        "--baseline-hkl", default="", help="Optional conventional HKL for baseline/reference endpoint comparisons."
    )
    p_push.add_argument(
        "--ignore-source-baseline-hkl",
        action="store_true",
        help=(
            "Do not inherit a conventional baseline HKL embedded in --source-summary. "
            "This prevents a larger-cohort baseline from entering a single-crystal leave-one-out endpoint."
        ),
    )
    p_push.add_argument(
        "--hkl-transform", default="", help="Override signed HKL transform. Defaults to the source summary."
    )
    p_push.add_argument("--cycles", type=int, default=3, help="Rounds of Bloch calculation and SHELXL refinement (default %(default)s).")
    p_push.add_argument(
        "--recompute-final-rows",
        action="store_true",
        help=(
            "With --cycles 0 on a multi-dataset checkpoint, recompute fixed-nuisance Bloch rows for --start-res "
            "without a SHELXL step. Use this to make maps/statistics for an already accepted endpoint model."
        ),
    )
    p_push.add_argument("--n-basis", type=int, default=None, help="Override the beam count for all continuation crystals; otherwise retain each saved count (single-dataset default: 108).")
    p_push.add_argument("--n-points", type=int, default=None, help="Rocking points per observation (default: the count saved by the run).")
    p_push.add_argument("--final-n-basis", type=int, default=0, help="Single-dataset final beam override; for multiple crystals use --dataset-n-basis or --n-basis.")
    p_push.add_argument("--final-n-points", type=int, default=20)
    add_crystal_overrides(p_push)
    add_profile_coordinates(p_push, continuation=True)
    p_push.add_argument("--profile-quadrature", choices=["uniform", "trapezoid", "gauss_legendre"], default=None)
    p_push.add_argument("--profile-accumulation", choices=["sum", "mean"], default=None)
    p_push.add_argument("--shelxl", default=default_shelxl_executable(), help="SHELXL executable (default: from --config or PATH).")
    p_push.add_argument("--shelxl-iterations", type=int, default=1, help="SHELXL executable calls per macrocycle.")
    p_push.add_argument(
        "--shelxl-feedback-xyz-tol",
        type=float,
        default=0.0,
        help="Stop once the max non-H coordinate shift between successive models is below this value in A. Default 0 runs every requested cycle; a shift test is only meaningful with converged SHELXL calls per cycle.",
    )
    p_push.add_argument(
        "--shelxl-feedback-keep",
        choices=["best-free", "last"],
        default="last",
        help="Continuation model to keep: last (default) or the output with the lowest free-row R_obs.",
    )
    p_push.add_argument(
        "--endpoint-ls-cycles",
        type=int,
        default=1,
        help="Diagnostic internal L.S./CGLS cycle count for each generated endpoint. Production keeps this at 1.",
    )
    p_push.add_argument("--workers", type=int, default=auto_worker_count(0, 0.85), help="Worker processes (default: 85%% of the cores).")
    p_push.add_argument("--chunksize", type=int, default=4, help="Observations per worker task (default %(default)s).")
    p_push.add_argument("--row-plan-cache", choices=["off", "exact", "readwrite", "rebuild"], default="off", help="Reuse of per-observation geometry: off (default), exact, readwrite or rebuild.")
    p_push.add_argument("--min-d-spacing", type=float, default=0.0)
    p_push.add_argument(
        "--endpoint-min-d-spacing",
        type=float,
        default=0.0,
        help="Resolution limit of the corrected HKL file and its statistics in Å (default 0: as --min-d-spacing).",
    )
    p_push.add_argument("--topup-d-min", type=float, default=None, help="Override the beam-pool resolution; otherwise use each checkpoint value (single-dataset default: 0.7 Angstrom).")
    p_push.add_argument("--min-peak", type=float, default=-1.0)
    p_push.add_argument("--geometry-percentile", type=float, default=0.0)
    p_push.add_argument("--max-rows", type=int, default=0)
    p_push.add_argument(
        "--dataset-row-contract",
        action="append",
        default=[],
        metavar="DATASET=CSV",
        help=(
            "Multi-dataset only: bind a continuation dataset to the exact policy-selected native_dyn_calc.csv "
            "membership. Supply once for every checkpoint dataset; GROND verifies row_index/HKL/target/sigma/d-spacing "
            "before calculation, reuses its dataset-role/raw/endpoint policy annotations, and refuses to rebuild "
            "membership from raw XDS filters."
        ),
    )
    p_push.add_argument("--post-bloch-sigma-policy", choices=["none", "uniform_z_rms"], default="none")
    p_push.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20)
    p_push.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.25)
    p_push.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0)
    p_push.add_argument(
        "--lambda-grid",
        default=FULL_Q_LAMBDA,
        help="Diagnostic q-exponent grid. Production auto uses the full Bloch/Born correction with lambda_q=1.0.",
    )
    # Continuation contract: the free split, the P-spline grids and the SHELXL handoff are replayed from the
    # checkpoint/source summary (the historical value when a summary predates the key).
    p_push.add_argument("--prediction-cache-compression", choices=["compressed", "none"], default="none")
    p_push.add_argument(
        "--model-prep",
        choices=[
            "as-is",
            "no-restraints",
            "free-h-isotropic-no-restraints",
            "free-h-anis-no-restraints",
            "adp-stabilized",
        ],
        default="as-is",
        help="Optional start-model preparation before the push. adp-stabilized adds RIGU/SIMU/targeted ISOR and converts known NPD atoms to isotropic fallback; free-H is a stress-test branch.",
    )
    p_push.add_argument(
        "--set-ls-cycles",
        type=int,
        default=1,
        help="Rewrite L.S. to this value in the prepared start model. Use 0 to preserve it.",
    )
    p_push.set_defaults(func=cmd_push_endpoint)




def add_model_box_args(parser: argparse.ArgumentParser) -> None:
    """Options for the model-covering box maps written beside every unit-cell map."""

    parser.add_argument(
        "--model-box-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write, beside every unit-cell map, a CCP4 map boxed around the whole model plus a margin "
        "(<map>_model_box.map; the unit-cell maps are kept). On by default.",
    )
    parser.add_argument(
        "--model-box-margin",
        type=float,
        default=3.0,
        help="Margin in Angstroms around the model for the model-box maps (default 3).",
    )


def register_finish(sub: argparse._SubParsersAction) -> None:
    """Register finish commands."""

    from grond.cli_support.tools import add_tool_config_args, default_shelxl_executable
    from grond.commands.finish import cmd_finish

    p_finish = sub.add_parser(
        "finish",
        help="Write the maps and summary tables of a finished run.",
    )
    p_finish.add_argument(
        "--publication-feedback-cycles",
        choices=["kept", "all"],
        default=None,
        help="Rounds the final model may come from: kept (default) or all.",
    )
    p_finish.add_argument(
        "--run-dir", default=".", help="Output directory of a finished run (default: the current directory)."
    )
    p_finish.add_argument(
        "--checkpoint", default="", help="grond_continue_checkpoint.json; its run directory is used."
    )
    p_finish.add_argument(
        "--model",
        default="",
        help="Model for the map phases (default: the final model of the run).",
    )
    p_finish.add_argument(
        "--shelxl",
        default=default_shelxl_executable(),
        help="SHELXL executable, for maps that need a zero-cycle refinement.",
    )
    add_tool_config_args(p_finish)
    p_finish.add_argument(
        "--make-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write maps (default on).",
    )
    p_finish.add_argument(
        "--map-mode",
        choices=["native", "shelx2map", "both", "off"],
        default="both",
        help="native (dynamical maps), shelx2map (SHELXL Fo-Fc maps), both (default) or off.",
    )
    p_finish.add_argument(
        "--native-map-spacing",
        type=float,
        default=0.1,
        help="Grid spacing of the maps in Å (default %(default)s).",
    )
    p_finish.add_argument(
        "--native-map-d-min",
        type=float,
        default=0.0,
        help="Optional low-pass d_min cutoff for native dynamical maps. Zero uses all row-table coefficients.",
    )
    p_finish.add_argument(
        "--native-map-min-i-over-sigma",
        type=float,
        default=0.0,
        help="Minimum row I/sigma used for native dynamical map amplitudes.",
    )
    p_finish.add_argument(
        "--native-map-max-grid-points",
        type=int,
        default=8_000_000,
        help="Safety cap on native dynamical map grid points.",
    )
    add_model_box_args(p_finish)
    p_finish.add_argument("--json", action="store_true", help="Print a JSON summary.")
    p_finish.set_defaults(func=cmd_finish)

