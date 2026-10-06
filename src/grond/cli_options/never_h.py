"""One command for a strict paired H-free refinement and map export."""

def register_never_h(sub):
    """Register ``grond never-h`` and its options."""

    from grond.cli_support.tools import add_tool_config_args, default_shelxl_executable
    from grond.commands.never_h import cmd_never_h

    parser = sub.add_parser("never-h", help="Refine an independent H-free model with all full-H crystal settings and fitted scales fixed.")
    parser.add_argument("--reference-run", required=True, help="Completed full-H multi-crystal auto directory.")
    parser.add_argument("--structure", required=True, help="Independently solved and conventionally refined H-free RES.")
    parser.add_argument("--baseline-hkl", default="", help="Conventional HKL in the input model's setting; defaults to its matching .hkl.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW", help="Relocate saved input path prefixes. Repeat as needed.")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--shelxl", default=default_shelxl_executable())
    parser.add_argument("--no-maps", dest="maps", action="store_false", default=True)
    parser.add_argument("--map-grid-spacing", type=float, default=0.1, help="Common map grid spacing in Angstroms.")
    parser.add_argument("--dry-run", action="store_true")
    from .endpoint_commands import add_model_box_args

    add_model_box_args(parser)
    add_tool_config_args(parser)
    parser.set_defaults(func=cmd_never_h)
