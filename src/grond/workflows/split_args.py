"""Free-split engine options shared by the native, endpoint and structure-update command builders."""

from __future__ import annotations

import argparse
from pathlib import Path

from grond.validation.heldout import DEFAULT_FREE_SALT


def split_scheme_args(args: argparse.Namespace, *, split_contract: str = "") -> list[str]:
    """Free-split options for engines built outside native_stage_cmd (the structure updates).

    Namespaces without the split-scheme options keep the legacy split.  The
    immutable split contract is bound only under orbit-hash, as for the
    continuous thickness polish; legacy runs keep their historical split.
    """

    scheme = str(getattr(args, "split_scheme", "legacy"))
    cmd = [
        "--free-fraction",
        f"{float(getattr(args, 'free_fraction', 0.10)):g}",
        "--split-seed",
        str(int(getattr(args, "split_seed", 42))),
        "--split-scheme",
        scheme,
        "--split-salt",
        str(getattr(args, "split_salt", DEFAULT_FREE_SALT)),
    ]
    if scheme != "legacy" and str(split_contract or "").strip():
        cmd += ["--split-contract", str(split_contract)]
    if str(getattr(args, "exclude_orbits_json", "") or "").strip():
        cmd += ["--exclude-orbits-json", str(Path(args.exclude_orbits_json).expanduser().resolve())]
    return cmd
