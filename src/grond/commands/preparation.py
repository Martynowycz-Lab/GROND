"""Resolve ordinary integration inputs and write a reusable preparation file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grond.io.config import (
    DatasetSpec,
    PrepConfig,
    load_prep,
    parse_dataset_spec,
    resolve_hkl,
    write_datasets_csv,
    write_prep,
)

BASELINE_HINT = "Give the HKLF 4 file the model was refined against, or leave out --baseline-hkl."


def _existing_file(command: str, flag: str, value: str, hint: str) -> str:
    """The resolved path of a file named on the command line; stop with one line when it is not a file."""

    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    problem = "is a directory, not a file" if path.is_dir() else "does not exist"
    raise SystemExit(f"{command}: {flag} {value} {problem}. {hint}")


def _load_prep_file(command: str, value: str) -> PrepConfig:
    """Read the --prep file; stop with one line when it is missing or is not a prep file."""

    path = Path(value).expanduser()
    if not path.is_file():
        raise SystemExit(f"{command}: --prep {value} does not exist. Write it with `grond prep` (see grond prep -h).")
    try:
        return load_prep(path)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"{command}: --prep {value} is not a GROND prep file ({type(exc).__name__}: {exc}). "
                         "Write it again with `grond prep`.") from exc


def prep_from_args(args: argparse.Namespace) -> PrepConfig:
    """Create a preparation record from CLI flags or an existing prep file."""

    command = f"grond {getattr(args, 'command', '') or 'auto'}"
    if args.prep:
        prep = _load_prep_file(command, args.prep)
        # A new solution may have a different setting. Use its model/HKLF4 pair
        # for the baseline and indexing scout, not the old prep's pair.
        if args.structure:
            prep.structure = str(Path(args.structure).expanduser().resolve())
        if args.baseline_hkl:
            prep.baseline_hkl = _existing_file(command, "--baseline-hkl", args.baseline_hkl, BASELINE_HINT)
            for ds in prep.datasets:
                ds.baseline_hkl = prep.baseline_hkl
        elif str(prep.baseline_hkl or "").strip() and not Path(prep.baseline_hkl).expanduser().is_file():
            # the conventional-baseline stage would stop on it with a CalledProcessError traceback
            raise SystemExit(f"{command}: the baseline HKL {prep.baseline_hkl} named in --prep {args.prep} does not "
                             "exist. Put it back, give another with --baseline-hkl, or write the prep file again "
                             "with `grond prep`.")
    else:
        prep_metadata = {}
        if args.xds_dir or args.hkl:
            try:
                hkl, xds_dir = resolve_hkl(args.xds_dir or args.hkl)
            except FileNotFoundError as exc:
                raise SystemExit(str(exc)) from exc
        elif getattr(args, "dials_refl", ""):
            if not getattr(args, "dials_expt", ""):
                raise SystemExit("Provide --dials-expt with --dials-refl.")
            from grond.io.dials_bridge import convert_dials_to_grond

            bridge_dir = (
                Path(args.out_dir).expanduser().resolve() / "prepared" / "dials_dataset"
            )
            try:
                bridge = convert_dials_to_grond(
                    args.dials_refl,
                    args.dials_expt,
                    bridge_dir,
                    intensity_mode=args.dials_intensity_mode,
                    row_selection=args.dials_row_selection,
                    sigma_mode=args.dials_sigma_mode,
                    dataset_name="dataset",
                )
            except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
                raise SystemExit(f"DIALS import failed: {exc}") from exc
            hkl, xds_dir = str(bridge.hkl), str(bridge.out_dir)
            prep_metadata["dials_bridge_summary"] = str(bridge.summary)
        else:
            raise SystemExit("Provide --prep, --hkl/--xds-dir, or --dials-refl/--dials-expt.")
        if not args.structure:
            raise SystemExit("Provide --structure when not using --prep.")
        prep = PrepConfig(
            structure=str(Path(args.structure).expanduser().resolve()),
            baseline_hkl=(_existing_file(command, "--baseline-hkl", args.baseline_hkl, BASELINE_HINT)
                          if args.baseline_hkl else ""),
            default_hkl_transform=args.hkl_transform,
            geometry_percentile=args.geometry_percentile,
            datasets=[
                DatasetSpec(
                    name="dataset",
                    hkl=hkl,
                    xds_dir=xds_dir,
                    hkl_transform=args.hkl_transform,
                    min_d_spacing=args.min_d_spacing if args.min_d_spacing is not None else 0.70,
                    min_peak=args.min_peak,
                    geometry_percentile=args.geometry_percentile,
                )
            ],
            metadata=prep_metadata,
        )
    for ds in prep.datasets:
        if args.geometry_percentile is not None:
            ds.geometry_percentile = float(args.geometry_percentile)
    if getattr(args, "min_d_spacing", 0.70) is None:   # the combine and endpoint stages follow the prep file's crystals
        args.min_d_spacing = min(float(ds.min_d_spacing) for ds in prep.datasets)
    return prep


def cmd_prep(args: argparse.Namespace) -> int:
    """CLI entrypoint for writing a reusable prep JSON file."""

    try:
        datasets = [
            parse_dataset_spec(item, default_hkl_transform=args.hkl_transform, geometry_percentile=args.geometry_percentile)
            for item in args.dataset
        ]
    except (FileNotFoundError, ValueError) as exc:  # a malformed --dataset raises ValueError
        raise SystemExit(f"grond prep: {exc}") from exc
    prep = PrepConfig(
        structure=str(Path(args.structure).expanduser().resolve()),
        baseline_hkl=str(Path(args.baseline_hkl).expanduser().resolve()) if args.baseline_hkl else "",
        notes=args.notes,
        default_hkl_transform=args.hkl_transform,
        geometry_percentile=args.geometry_percentile,
        parity_setup_id=str(args.parity_setup_id).strip(),
        parity_setup_manifest_sha256=str(args.parity_setup_manifest_sha256).strip(),
        datasets=datasets,
    )
    write_prep(Path(args.out), prep)
    write_datasets_csv(Path(args.out).with_suffix(".datasets.csv"), prep)
    print(json.dumps({"prep": str(Path(args.out).resolve()), "datasets": len(datasets)}, indent=2))
    return 0
