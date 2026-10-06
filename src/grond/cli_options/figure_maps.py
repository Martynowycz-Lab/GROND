"""Options for ``grond figure-maps``."""

from __future__ import annotations

import argparse


def register_figure_maps(sub: argparse._SubParsersAction) -> None:
    """Register ``grond figure-maps`` and its options."""

    from grond.cli_support.tools import add_tool_config_args, default_shelxl_executable
    from grond.commands.figure_maps import cmd_figure_maps

    parser = sub.add_parser(
        "figure-maps",
        help=(
            "Box every map of a finished run (native dynamical, corrected-endpoint and conventional shelx2map maps) "
            "around the whole reference model plus a margin, after placing each map's model on the reference "
            "(space-group operation, lattice translations, allowed origin shift; never a hand change). The run is "
            "only read; missing maps are calculated in --out-dir."
        ),
    )
    parser.add_argument("--run-dir", required=True, help="Finished GROND run, or a grond never-h output directory.")
    parser.add_argument("--out-dir", required=True, help="New or empty directory outside the run for every output.")
    parser.add_argument(
        "--reference-model",
        default="",
        help="Model (.res/.cif) whose placement every box is cut around, e.g. the matching full-H model for a "
        "never-H run. Default: for a never-h directory its recorded full-H reference (reference_inputs/full_H.res, "
        "the model never-h boxes its own maps around); otherwise the run's own model (publication endpoint, else "
        "the native-map model).",
    )
    parser.add_argument("--margin-A", type=float, default=3.0, help="Box margin around the model in Angstroms.")
    parser.add_argument(
        "--rmsd-max-A",
        type=float,
        default=0.5,
        help="Refuse a placement whose common non-H atoms are further than this RMSD from the reference.",
    )
    parser.add_argument(
        "--allow-improper-space-group-operations",
        action="store_true",
        help="In a non-centrosymmetric group, allow an improper operation of the group itself (a glide or mirror). "
        "An inverted placement (hand change) is never applied.",
    )
    parser.add_argument(
        "--generate-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Calculate maps the run has not written (in --out-dir/source_maps). On by default.",
    )
    parser.add_argument(
        "--conventional",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also make conventional shelx2map maps from the run's conventional baseline .res/.fcf.",
    )
    parser.add_argument("--conventional-res", default="", help="Conventional model .res (default: the run's baseline).")
    parser.add_argument("--conventional-fcf", default="", help="Its LIST 6 .fcf (default: next to the .res).")
    parser.add_argument(
        "--shelxl", default=default_shelxl_executable(), help="SHELXL for a zero-cycle LIST 6 replay when needed."
    )
    parser.add_argument("--native-map-spacing", type=float, default=0.25, help="Native map grid spacing (A).")
    parser.add_argument("--native-map-d-min", type=float, default=0.0)
    parser.add_argument("--native-map-min-i-over-sigma", type=float, default=0.0)
    parser.add_argument("--native-map-max-grid-points", type=int, default=8_000_000)
    add_tool_config_args(parser)
    parser.add_argument("--json", action="store_true", help="Print the manifest as JSON.")
    parser.set_defaults(func=cmd_figure_maps)
