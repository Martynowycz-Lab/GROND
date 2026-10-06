"""Register ``GROND splits make`` and ``GROND splits inspect``."""

from __future__ import annotations

import argparse

from grond.validation.heldout import DEFAULT_FREE_SALT


def register_splits(sub: argparse._SubParsersAction) -> None:
    """Register the split tooling for new runs."""

    from grond.commands.splits import cmd_splits_inspect, cmd_splits_make

    p_splits = sub.add_parser(
        "splits",
        help=(
            "Write or inspect a run's split record: the orbit-hash free set AUTO holds out, and an optional lockbox, "
            "a set of orbits held back from every stage (GROND splits make | inspect)."
        ),
    )
    actions = p_splits.add_subparsers(dest="splits_command", required=True)
    p_make = actions.add_parser(
        "make",
        help=(
            "Write split_record.json (free set: salt, fraction, per-crystal and pooled counts, realised fraction) and, "
            "with --lockbox-fraction, lockbox.json to pass to AUTO with --exclude-orbits-json and score once afterwards."
        ),
    )
    p_make.add_argument("--prep", default="", help="Prep JSON written by GROND prep or import-dials.")
    p_make.add_argument(
        "--hkl",
        action="append",
        default=[],
        help="INTEGRATE.HKL, a GROND DIALS bridge HKL or their directory; repeat for several crystals (instead of --prep).",
    )
    p_make.add_argument("--dataset-name", action="append", default=[], help="One name per --hkl (default dataset_N).")
    p_make.add_argument("--structure", default="", help="Model RES/CIF (required with --hkl; overrides the prep model).")
    p_make.add_argument("--hkl-transform", default="identity", help="Declared transform of every --hkl file.")
    p_make.add_argument("--out-dir", required=True, help="New directory for split_record.json and lockbox.json.")
    p_make.add_argument("--free-fraction", type=float, default=0.10)
    p_make.add_argument("--split-salt", default=DEFAULT_FREE_SALT)
    p_make.add_argument(
        "--exclude-orbits-json", default="", help="An existing lockbox to honour (instead of making one)."
    )
    p_make.add_argument("--transform-candidates", default="auto")
    p_make.add_argument("--transform-metric-tolerance", type=float, default=0.02)
    p_make.add_argument("--transform-parity-policy", choices=["proper-only", "all-diagnostic"], default="proper-only")
    p_make.add_argument("--split-candidate-set", choices=["shared-closure", "per-crystal"], default="shared-closure")
    p_make.add_argument("--transform-canonicalization", choices=["point-group", "off"], default="point-group")
    p_make.add_argument(
        "--trust-hkl-transform",
        action="store_true",
        help="The run will use --trust-hkl-transform (no scout, no candidate union: plain Laue orbits).",
    )
    p_make.add_argument(
        "--lockbox-fraction",
        type=float,
        default=0.0,
        help="Fraction of orbit classes per resolution shell to lock away, for example 0.2 (default 0: no lockbox).",
    )
    p_make.add_argument(
        "--lockbox-salt",
        default="",
        help="Lockbox salt (default: GROND-LOCKBOX-<date>-<random 16 hex>, recorded in lockbox.json).",
    )
    p_make.add_argument("--lockbox-shells", type=int, default=5, help="Equal-count resolution shells (1/d^2).")
    p_make.add_argument("--lockbox-d-min", type=float, default=0.0, help="Leave orbits with d below this out (A).")
    p_make.set_defaults(func=cmd_splits_make)

    p_inspect = actions.add_parser(
        "inspect",
        help="Print and verify a split record, a lockbox, a split contract or an AUTO run directory.",
    )
    p_inspect.add_argument("path", help="split_record.json, lockbox.json, a split contract or a run directory.")
    p_inspect.add_argument(
        "--lockbox",
        default="",
        help="For a run directory: verify the run recorded this lockbox and that none of its rows lies in it.",
    )
    p_inspect.add_argument("--json", action="store_true")
    p_inspect.set_defaults(func=cmd_splits_inspect)
