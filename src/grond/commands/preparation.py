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
from grond.workflows.decision import safe_label
from grond.workflows.decision_bindings import _validated_sha256_text


def prep_from_args(args: argparse.Namespace) -> PrepConfig:
    """Create a preparation record from CLI flags or an existing prep file."""

    if args.prep:
        prep = load_prep(Path(args.prep))
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
                Path(args.out_dir).expanduser().resolve() / "prepared" / f"dials_{safe_label(args.dataset_name)}"
            )
            try:
                bridge = convert_dials_to_grond(
                    args.dials_refl,
                    args.dials_expt,
                    bridge_dir,
                    intensity_mode=args.dials_intensity_mode,
                    row_selection=args.dials_row_selection,
                    sigma_mode=args.dials_sigma_mode,
                    sigma_scale=args.dials_sigma_scale,
                    dials_python=args.dials_python,
                    min_partiality_write=args.dials_min_partiality_write,
                    keep_dont_integrate=bool(args.dials_keep_dont_integrate)
                    and not bool(args.dials_drop_dont_integrate),
                    drop_bad_flags=not bool(args.dials_keep_bad_flags),
                    dataset_name=args.dataset_name,
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
            parity_setup_id=str(getattr(args, "parity_setup_id", "") or "").strip(),
            parity_setup_manifest_sha256=str(getattr(args, "parity_setup_manifest_sha256", "") or "").strip(),
            datasets=[
                DatasetSpec(
                    name=args.dataset_name,
                    hkl=hkl,
                    xds_dir=xds_dir,
                    hkl_transform=args.hkl_transform,
                    min_d_spacing=args.min_d_spacing,
                    min_peak=args.min_peak,
                    target_mode=args.target_mode,
                    include_nonpositive=args.include_nonpositive,
                    geometry_percentile=args.geometry_percentile,
                    physical_crystal_id=str(getattr(args, "physical_crystal_id", "") or "").strip(),
                    acquisition_fingerprint=str(getattr(args, "acquisition_fingerprint", "") or "").strip(),
                )
            ],
            metadata=prep_metadata,
        )
    setup_id_override = str(getattr(args, "parity_setup_id", "") or "").strip()
    setup_hash_override = str(getattr(args, "parity_setup_manifest_sha256", "") or "").strip()
    physical_override = str(getattr(args, "physical_crystal_id", "") or "").strip()
    acquisition_override = str(getattr(args, "acquisition_fingerprint", "") or "").strip()
    if setup_id_override:
        prep.parity_setup_id = setup_id_override
    if setup_hash_override:
        prep.parity_setup_manifest_sha256 = _validated_sha256_text(
            setup_hash_override,
            label="parity setup manifest SHA-256",
        )
    if physical_override or acquisition_override:
        if len(prep.datasets) != 1:
            raise SystemExit(
                "--physical-crystal-id and --acquisition-fingerprint are "
                "single-dataset overrides; record them per --dataset entry "
                "when building a multi-dataset prep file"
            )
        if physical_override:
            prep.datasets[0].physical_crystal_id = physical_override
        if acquisition_override:
            prep.datasets[0].acquisition_fingerprint = _validated_sha256_text(
                acquisition_override,
                label="acquisition fingerprint",
            )
    for ds in prep.datasets:
        if args.geometry_percentile is not None:
            ds.geometry_percentile = float(args.geometry_percentile)
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
