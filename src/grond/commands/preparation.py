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


def prep_from_args(args: argparse.Namespace) -> PrepConfig:
    """Create a preparation record from CLI flags or an existing prep file."""

    if args.prep:
        prep = load_prep(Path(args.prep))
        # A new solution may have a different setting. Use its model/HKLF4 pair
        # for the baseline and indexing scout, not the old prep's pair.
        if args.structure:
            prep.structure = str(Path(args.structure).expanduser().resolve())
        if args.baseline_hkl:
            prep.baseline_hkl = str(Path(args.baseline_hkl).expanduser().resolve())
            for ds in prep.datasets:
                ds.baseline_hkl = prep.baseline_hkl
    else:
        prep_metadata = {}
        if args.xds_dir:
            hkl, xds_dir = resolve_hkl(args.xds_dir)
        elif args.hkl:
            hkl, xds_dir = resolve_hkl(args.hkl)
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
                    dials_python=args.dials_python,
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
            baseline_hkl=str(Path(args.baseline_hkl).expanduser().resolve()) if args.baseline_hkl else "",
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

    datasets = [
        parse_dataset_spec(item, default_hkl_transform=args.hkl_transform, geometry_percentile=args.geometry_percentile)
        for item in args.dataset
    ]
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
