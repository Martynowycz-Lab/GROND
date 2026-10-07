"""Register the endpoint commands: push-endpoint (alias refine), check-shelx, validate-endpoint, finish, publish."""

from __future__ import annotations

import argparse

from grond.runtime import auto_worker_count


def register_push(sub: argparse._SubParsersAction) -> None:
    """Register push commands."""

    from grond.cli_options.run_endpoints import add_final_polish
    from grond.cli_options.run_physics import add_crystal_overrides, add_profile_coordinates
    from grond.cli_support.tools import add_tool_config_args, default_shelxl_executable
    from grond.cli_support.values import FULL_Q_LAMBDA
    from grond.commands.push import cmd_push_endpoint

    p_push = sub.add_parser(
        "push-endpoint",
        aliases=["refine"],
        help="Continue a pinned GROND endpoint from an existing summary and start model.",
    )
    p_push.add_argument("--checkpoint", default="", help="grond_continue_checkpoint.json from a completed auto run.")
    p_push.add_argument(
        "--source-summary",
        default="",
        help="Existing native_row_regularize summary whose selected nuisance values should be pinned.",
    )
    p_push.add_argument(
        "--start-res",
        default="",
        help="Starting RES model to continue from. Overrides the checkpoint model when provided.",
    )
    p_push.add_argument("--out-dir", required=True)
    p_push.add_argument("--python", default="")
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
    p_push.add_argument(
        "--multi-endpoint-merge-key",
        choices=["model_symmetry", "model_symmetry_friedel", "exact_signed_hkl"],
        default="",
        help="For multi-dataset checkpoints, corrected-HKL merge key before each shared SHELXL LS step. Defaults to the checkpoint value or model_symmetry.",
    )
    p_push.add_argument("--cycles", type=int, default=3, help="Additional Bloch/SHELXL macrocycles.")
    p_push.add_argument(
        "--recompute-final-rows",
        action="store_true",
        help=(
            "With --cycles 0 on a multi-dataset checkpoint, recompute fixed-nuisance Bloch rows for --start-res "
            "without a SHELXL step. Use this to make maps/statistics for an already accepted endpoint model."
        ),
    )
    p_push.add_argument("--n-basis", type=int, default=None, help="Override the beam count for all continuation crystals; otherwise retain each saved count (single-dataset default: 108).")
    p_push.add_argument("--n-points", type=int, default=None, help="Override rocking samples for all continuation crystals; otherwise retain each saved count (single-dataset default: 20).")
    p_push.add_argument("--final-n-basis", type=int, default=0, help="Single-dataset final beam override; for multiple crystals use --dataset-n-basis or --n-basis.")
    p_push.add_argument("--final-n-points", type=int, default=20)
    add_crystal_overrides(p_push)
    add_profile_coordinates(p_push, continuation=True)
    p_push.add_argument("--profile-quadrature", choices=["uniform", "trapezoid", "gauss_legendre"], default=None)
    p_push.add_argument("--profile-accumulation", choices=["sum", "mean"], default=None)
    p_push.add_argument("--shelxl", default=default_shelxl_executable())
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
    p_push.add_argument(
        "--native-wght-a",
        type=float,
        default=None,
        help="Override the checkpoint native row-weight a term for every continuation dataset.",
    )
    p_push.add_argument(
        "--native-wght-b",
        type=float,
        default=None,
        help="Override the checkpoint native row-weight b term for every continuation dataset.",
    )
    p_push.add_argument("--workers", type=int, default=auto_worker_count(0, 0.85))
    p_push.add_argument("--chunksize", type=int, default=4)
    p_push.add_argument(
        "--compute-backend",
        choices=[
            "scipy",
            "scipy_bethe",
            "scipy_expm",
            "scipy_expm_multiply",
            "jax",
            "jax_auto",
            "jax_cpu",
            "jax_mps",
            "jax_cuda",
            "jax_padded",
            "jax_auto_padded",
            "jax_cpu_padded",
            "jax_mps_padded",
            "jax_cuda_padded",
            "jax_cpu_real_expm",
            "jax_mps_real_expm",
            "jax_mps_real_expm_padded",
        ],
        default="scipy",
    )
    p_push.add_argument("--row-plan-cache", choices=["off", "exact", "readwrite", "rebuild"], default="off")
    p_push.add_argument(
        "--pinned-sg-jitter-fallback",
        choices=["none", "off"],
        default="none",
        help="Revalidate an infeasible pinned Sg-jitter model against an explicit no-jitter candidate.",
    )
    p_push.add_argument("--jax-batch-size", type=int, default=0)
    p_push.add_argument("--jax-matrix-bucket-size", type=int, default=0)
    p_push.add_argument("--bethe-strong-count", type=int, default=48)
    p_push.add_argument(
        "--bethe-variant", choices=["truncate", "schur0", "schur_target", "schur_mean"], default="schur_target"
    )
    p_push.add_argument("--bethe-promote-denom", type=float, default=0.005)
    p_push.add_argument("--bethe-denom-eps", type=float, default=1e-5)
    p_push.add_argument("--min-d-spacing", type=float, default=0.0)
    p_push.add_argument(
        "--endpoint-min-d-spacing",
        type=float,
        default=0.0,
        help="Optional cutoff applied again to each combined corrected-HKL endpoint; zero follows --min-d-spacing.",
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
    p_push.add_argument(
        "--dataset-scale-spec",
        action="append",
        default=[],
        metavar="DATASET=SPEC",
        help=(
            "Multi-dataset only: replay this compact scale topology after each fresh Bloch row calculation and before "
            "the shared endpoint. Supply once for every dataset; examples are adaptive:28:det8 or stride:6:det8."
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
    # Continuation contract: by default every value below is replayed from the checkpoint/source summary (the
    # historical value when a summary predates the key), never taken from the current AUTO defaults.
    p_push.add_argument(
        "--rescale",
        choices=["median-iobs", "none", "absolute"],
        default=None,
        help="Corrected-intensity scale of the endpoint HKL. Default: the source summary's value (median-iobs if unrecorded).",
    )
    p_push.add_argument(
        "--endpoint-include-free-rows",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write split=free rows into the corrected HKL. Default: the source summary's value (included if unrecorded).",
    )
    p_push.add_argument(
        "--endpoint-q-policy",
        choices=["clip", "exclude"],
        default=None,
        help="Pathological correction ratios: clip or exclude. Default: the source summary's value (clip if unrecorded).",
    )
    p_push.add_argument(
        "--shelxl-npd-recovery",
        choices=["damp", "isor"],
        default=None,
        help="NPD recovery of the SHELXL handoff. Default: the source summary's value (damp if unrecorded).",
    )
    p_push.add_argument(
        "--shelxl-convergence-shift-esd",
        type=float,
        default=None,
        help="Stop the SHELXL calls once |max shift/esd| is below this value. Default: the source summary's value (0 if unrecorded).",
    )
    p_push.add_argument(
        "--endpoint-wght-mode",
        choices=["freeze-zero", "inherit", "promote", "promote-gold", "converge"],
        default=None,
        help="SHELXL WGHT policy of the endpoints. Default: the source summary's value (freeze-zero if unrecorded).",
    )
    p_push.add_argument(
        "--keep-damp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Carry an active DAMP into later SHELXL calls. Default: the source summary's value (carried if unrecorded).",
    )
    p_push.add_argument(
        "--split-scheme",
        choices=["orbit-hash", "legacy"],
        default=None,
        help="Free-split scheme. Default: the source summaries' value (legacy if unrecorded). Single-dataset pushes only replay it.",
    )
    p_push.add_argument(
        "--split-salt", default=None, help="Orbit-hash free-set salt. Default: the source summaries' value."
    )
    p_push.add_argument(
        "--exclude-orbits-json",
        default=None,
        help="Lockbox file of orbits held back from every stage. Default: the source summaries' value (none if unrecorded).",
    )
    p_push.add_argument(
        "--scale-model",
        choices=["pspline", "per-frame"],
        default=None,
        help="Frame-scale family. Default: the source summaries' value (per-frame if unrecorded).",
    )
    p_push.add_argument(
        "--scale-spline-n-basis-grid", default=None, help="P-spline basis grid. Default: the source summaries' value."
    )
    p_push.add_argument(
        "--scale-spline-lambda-grid", default=None, help="P-spline penalty grid. Default: the source summaries' value."
    )
    p_push.add_argument("--prediction-cache-compression", choices=["compressed", "none"], default="none")
    p_push.add_argument("--make-maps-on-final", action="store_true")
    p_push.add_argument(
        "--best-cycle-tolerance", type=float, default=0.001, help="Safety report tolerance above the best row wR(I)all."
    )
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
    p_push.add_argument(
        "--run-dir",
        default="",
        help="Finished GROND auto/run directory to polish with --final-polish diffbloch (its grond_summary.json, final row "
        "tables and SHELXL publication model; --start-res overrides the model).",
    )
    add_final_polish(p_push)
    p_push.set_defaults(func=cmd_push_endpoint)


def register_check_shelx(sub: argparse._SubParsersAction) -> None:
    """Register the SHELXL endpoint readiness checker."""

    from grond.cli_support.tools import default_shelxl_executable
    from grond.commands.check_shelx import cmd_check_shelx

    p_check = sub.add_parser(
        "check-shelx",
        help=(
            "Run SHELXL once on a copy of an endpoint (.res/.hkl) and report WGHT convergence, GooF, NPD atoms, "
            "BASF sanity and .res/.hkl/.fcf consistency, as Olex2 and SHELXLE users will see them."
        ),
    )
    p_check.add_argument("path", help="Endpoint directory (one .res with a matching .hkl) or the .res file.")
    p_check.add_argument("--stem", default="", help="Endpoint file stem when the directory holds several.")
    p_check.add_argument("--shelxl", default=default_shelxl_executable(), help="SHELXL executable.")
    p_check.add_argument(
        "--out-dir",
        default="",
        help="Empty or new directory for the check run (default: PATH/check_shelx_<timestamp>); the endpoint is never modified.",
    )
    p_check.add_argument("--acta", action="store_true", help="Add ACTA so SHELXL also writes a CIF of the endpoint.")
    p_check.add_argument("--json", action="store_true", help="Print the JSON report instead of the summary.")
    p_check.set_defaults(func=cmd_check_shelx)


def register_validate_endpoint(sub: argparse._SubParsersAction) -> None:
    """Register the post-hoc held-out SHELXL validation of a finished run."""

    from grond.commands.validate_endpoint import cmd_validate_endpoint

    p_validate = sub.add_parser(
        "validate-endpoint",
        help=(
            "Held-out SHELXL validation of a finished AUTO run, written to a new directory (the run is only read): "
            "by default the endpoint the run published, scored at L.S. 0 on its rebuilt HKL without re-refinement "
            "(--endpoint final re-runs combine_final instead), and the leak-free conventional model scored on the "
            "free orbits; reporting only."
        ),
    )
    p_validate.add_argument("--run-dir", required=True, help="Finished AUTO run directory (read only).")
    p_validate.add_argument("--out-dir", required=True, help="New (or empty) directory outside the run for every output.")
    p_validate.add_argument(
        "--endpoint",
        choices=["published", "final"],
        default="published",
        help="published (default): the endpoint the run published (the endpoint chooser's best physical endpoint in "
        "grond_summary.json, e.g. the kept feedback cycle's combine). The combine command that wrote it is replayed "
        "without refining: its rows are rebuilt with the recorded recipe, the rebuilt work HKL must equal the "
        "published HKL byte for byte, and the published .res itself is scored at L.S. 0 with the WGHT its last "
        "refinement applied; merged endpoint only. final: re-run the last recorded combine_final and score its "
        "re-refined runs/combined endpoints, which must reproduce the recorded ones exactly (the behaviour before "
        "this option; runs/combined is the published endpoint only when the chooser published the final combine).",
    )
    p_validate.add_argument(
        "--shelxl", default="", help="SHELXL executable (default: the one the run recorded for combine_final)."
    )
    p_validate.add_argument("--python", default="", help="Python for the combine engine (default: this interpreter).")
    p_validate.add_argument(
        "--unmerged",
        choices=["recorded", "on"],
        default="recorded",
        help="--endpoint final only. recorded (default): validate the endpoints the run wrote. on: also write, refine "
        "and validate the unmerged batch endpoint when the run had none (the merged endpoint is unchanged by it).",
    )
    p_validate.set_defaults(func=cmd_validate_endpoint)


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
        help="Regenerate final summaries, native dynamical maps, and the GROND dynamic CIF from a completed run.",
    )
    p_finish.add_argument(
        "--publication-feedback-cycles",
        choices=["kept", "all"],
        default=None,
        help="SHELXL feedback cycles the refreshed publication chooser may pick: kept (default; also for runs whose "
        "run_config predates the setting) or all (every adopted cycle, ranked by in-sample R1).",
    )
    p_finish.add_argument(
        "--run-dir", default=".", help="Completed GROND auto/run output directory. Defaults to the current directory."
    )
    p_finish.add_argument(
        "--checkpoint", default="", help="Optional grond_continue_checkpoint.json; its run_dir is used."
    )
    p_finish.add_argument(
        "--model",
        default="",
        help="Optional model override for native dynamical map phases. Defaults to the selected publication endpoint RES.",
    )
    p_finish.add_argument(
        "--shelxl",
        default=default_shelxl_executable(),
        help="SHELXL executable used for zero-cycle LIST 6 endpoint-map replay when needed.",
    )
    add_tool_config_args(p_finish)
    p_finish.add_argument(
        "--make-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate finish maps. Native row-level dynamical maps are the default; use --no-make-maps to skip.",
    )
    p_finish.add_argument(
        "--map-mode",
        choices=["native", "shelx2map", "both", "off"],
        default="native",
        help="native writes row-level dynamical maps; shelx2map writes quick corrected-HKL endpoint maps; both writes both.",
    )
    p_finish.add_argument(
        "--native-map-spacing",
        type=float,
        default=0.25,
        help="Approximate CCP4 grid spacing in Angstroms for native dynamical maps.",
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
    p_finish.add_argument("--json", action="store_true", help="Print the compact JSON payload used by automation.")
    p_finish.set_defaults(func=cmd_finish)

    p_publish = sub.add_parser(
        "publish",
        help="Alias for finish: regenerate final summaries, native dynamical maps, and the GROND dynamic CIF.",
    )
    p_publish.add_argument(
        "--run-dir", default=".", help="Completed GROND auto/run output directory. Defaults to the current directory."
    )
    p_publish.add_argument(
        "--checkpoint", default="", help="Optional grond_continue_checkpoint.json; its run_dir is used."
    )
    p_publish.add_argument(
        "--model",
        default="",
        help="Optional model override for native dynamical map phases. Defaults to the selected publication endpoint RES.",
    )
    p_publish.add_argument(
        "--shelxl",
        default=default_shelxl_executable(),
        help="SHELXL executable used for zero-cycle LIST 6 endpoint-map replay when needed.",
    )
    add_tool_config_args(p_publish)
    p_publish.add_argument("--make-maps", action=argparse.BooleanOptionalAction, default=True)
    p_publish.add_argument("--map-mode", choices=["native", "shelx2map", "both", "off"], default="native")
    p_publish.add_argument("--native-map-spacing", type=float, default=0.25)
    p_publish.add_argument("--native-map-d-min", type=float, default=0.0)
    p_publish.add_argument("--native-map-min-i-over-sigma", type=float, default=0.0)
    p_publish.add_argument("--native-map-max-grid-points", type=int, default=8_000_000)
    add_model_box_args(p_publish)
    p_publish.add_argument("--json", action="store_true")
    p_publish.set_defaults(func=cmd_finish)
