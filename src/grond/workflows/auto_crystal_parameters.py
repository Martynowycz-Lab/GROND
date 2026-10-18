"""Fix selected crystal parameters while solving the new model's workflow afresh."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from grond.hashing import sha256_file
from grond.io.config import PrepConfig
from grond.workflows.crystal_parameters import dataset_values, resolve_crystal_parameters

THICKNESS_FIELDS = ("thickness_A", "thickness_model", "thickness_grad_frac", "thickness_sigma_frac")


def configure_auto_crystals(prep: PrepConfig, args: argparse.Namespace) -> None:
    """Copy numerical settings only; never take coordinates or indexing from a reference."""

    names = {ds.name for ds in prep.datasets}
    beams = dataset_values(args.dataset_n_basis, names, "--dataset-n-basis", integer=True)
    centers = dataset_values(args.dataset_thickness_nm, names, "--dataset-thickness-nm")
    source = str(args.crystal_parameters_from or "")
    if not (source or beams or centers):
        return
    if args.structure_update == "auto":
        args.structure_update = "shelxl_feedback"
    records = {}
    if source:
        print("INFO --crystal-parameters-from fixes beams, sampling and thickness only; absorption, jitter and scales can be refitted. Use GROND never-h for a fully fixed reference.", flush=True)
        if args.sg_window_scout != "off":
            raise SystemExit("--sg-window-scout must be off when copying fixed rocking settings from a reference checkpoint.")
        path = Path(source).expanduser().resolve()
        try:
            checkpoint = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Cannot read --crystal-parameters-from {path}: {exc}") from exc
        if not isinstance(checkpoint, dict) or checkpoint.get("kind") != "grond_continue_checkpoint" or checkpoint.get("status") != "ready":
            raise SystemExit("--crystal-parameters-from requires a completed, ready GROND checkpoint.")
        rows = checkpoint.get("datasets")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise SystemExit("Reference checkpoint has no valid datasets list.")
        by_name = {row.get("name"): row for row in rows}
        if len(by_name) != len(rows) or set(by_name) != names:
            raise SystemExit("Reference checkpoint and new run must have exactly the same unique dataset names.")
        for ds in prep.datasets:
            old = by_name[ds.name]
            if not isinstance(old.get("nuisance"), dict):
                raise SystemExit(f"Reference crystal {ds.name!r} has no valid nuisance parameters.")
            # Renamed copies are fine; a different integration is not the same crystal comparison.
            old_hkl = Path(str(old.get("hkl") or "")).expanduser()
            if not old_hkl.is_absolute():
                old_hkl = path.parent / old_hkl
            if not old_hkl.is_file() or not Path(ds.hkl).is_file():
                raise SystemExit(f"Cannot verify the reference integration for {ds.name!r}; both HKL files must be accessible.")
            if sha256_file(old_hkl) != sha256_file(Path(ds.hkl)):
                raise SystemExit(f"Reference integration differs for {ds.name!r}; use the same integration or supply explicit parameters.")
        inherit = argparse.Namespace(
            dataset_n_basis=args.dataset_n_basis, dataset_thickness_nm=args.dataset_thickness_nm,
            n_basis=None, n_points=None, topup_d_min=None,
        )
        records, nuisance = resolve_crystal_parameters(
            checkpoint, prep.datasets, {name: by_name[name].get("nuisance") or {} for name in names},
            inherit, checkpoint_dir=path.parent,
        )
        for name, record in records.items():
            nui = nuisance[name]
            if not set(THICKNESS_FIELDS).issubset(nui):
                raise SystemExit(f"Reference crystal {name!r} lacks a complete thickness distribution; supply explicit parameters instead.")
            if nui["thickness_model"] not in {"single", "uniform_gradient", "gaussian", "gaussian_gradient"}:
                raise SystemExit(f"Reference crystal {name!r} has unsupported thickness_model={nui['thickness_model']!r}.")
            for key in ("thickness_grad_frac", "thickness_sigma_frac"):
                try:
                    valid = math.isfinite(float(nui[key])) and float(nui[key]) >= 0
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    raise SystemExit(f"Reference crystal {name!r} has invalid {key}={nui[key]!r}.")
            if float(nui["thickness_grad_frac"]) > 1:
                raise SystemExit(f"Reference crystal {name!r} has thickness_grad_frac > 1.")
            record["fixed_thickness"] = {key: nui[key] for key in THICKNESS_FIELDS}
            record["reference_checkpoint"] = str(path)
            record["reference_sha256"] = sha256_file(path)
    else:
        for name in sorted(beams.keys() | centers.keys()):
            record = {"prediction": {}, "fixed_thickness": {}}
            if name in beams:
                record["prediction"]["n_basis"] = beams[name]
            if name in centers:
                if not math.isfinite(centers[name] * 10.0):
                    raise SystemExit(f"Thickness for {name!r} overflows when converted to Angstroms.")
                record["fixed_thickness"]["thickness_A"] = centers[name] * 10.0
            records[name] = record
    for record in records.values():
        record["fresh_auto"] = True
        record["unfixed"] = "indexing, frame offset, jitter, absorption, scale values, coordinates and endpoint scale selection follow the requested auto options"
    args.crystal_parameters = records
    (Path(args.out_dir) / "crystal_parameters.json").write_text(json.dumps(records, indent=2) + "\n")
