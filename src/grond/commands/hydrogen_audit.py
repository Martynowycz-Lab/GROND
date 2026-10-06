"""Build a hydrogen-omit diagnostic while preserving the selected endpoint contract."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import try_read_json
from grond.macromolecule import sha256_file
from grond.paths import engine_script
from grond.workflows.decision import write_json


def cmd_audit_hydrogen_omit(args: argparse.Namespace) -> int:
    """Quantify prospective never-H difference-map peaks at independent H sites."""

    import gemmi

    from grond.legacy.engine.bloch_engine import parse_model

    reference = Path(args.reference_res).expanduser().resolve()
    never_h = Path(args.never_h_res).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve()
    try:
        for label, path in (("full-H reference", reference), ("prospective never-H model", never_h)):
            if not path.is_file():
                raise ValueError(f"{label} does not exist: {path}")
        reference_model = parse_model(reference)
        never_h_model = parse_model(never_h)
        reference_h = [
            str(atom.get("name") or "")
            for atom in reference_model.get("atoms", [])
            if str(atom.get("element") or "").upper() == "H"
        ]
        never_h_atoms = [
            str(atom.get("name") or "")
            for atom in never_h_model.get("atoms", [])
            if str(atom.get("element") or "").upper() == "H"
        ]
        if not reference_h:
            raise ValueError("full-H reference contains no hydrogen atoms")
        if never_h_atoms:
            raise ValueError("prospective never-H model contains hydrogen atoms: " + ", ".join(never_h_atoms[:12]))
        reference_cell = tuple(float(value) for value in (reference_model.get("cell") or []))
        never_h_cell = tuple(float(value) for value in (never_h_model.get("cell") or []))
        if len(reference_cell) != 6 or len(never_h_cell) != 6:
            raise ValueError("full-H and never-H models must both define six unit-cell constants")
        cell_length_mismatch = max(
            abs(actual - expected) / max(abs(expected), 1e-12)
            for actual, expected in zip(never_h_cell[:3], reference_cell[:3], strict=True)
        )
        cell_angle_mismatch = max(
            abs(actual - expected) for actual, expected in zip(never_h_cell[3:], reference_cell[3:], strict=True)
        )
        if cell_length_mismatch > 0.01 or cell_angle_mismatch > 1.0:
            raise ValueError(
                "prospective never-H model cell is incompatible with the full-H reference "
                f"(max length mismatch {cell_length_mismatch:.4f}, "
                f"max angle mismatch {cell_angle_mismatch:.3f} deg)"
            )

        def normalized_symmetry(
            model: Mapping[str, object],
        ) -> tuple[int, tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]]:
            """Return a model's LATT code and its sorted symmetry operations, rounded, translations reduced modulo 1."""
            latt = int(model.get("shelx_latt", 0) or 0)
            operations = model.get("symmetry_ops")
            operations = operations if isinstance(operations, (list, tuple)) else []
            normalized: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
            for operation in operations:
                if not isinstance(operation, Mapping):
                    raise ValueError("model symmetry operation is not a numeric rotation/translation record")
                rotation = operation.get("rot")
                translation = operation.get("trans")
                if not isinstance(rotation, (list, tuple)) or not isinstance(translation, (list, tuple)):
                    raise ValueError("model symmetry operation lacks numeric rot/trans fields")
                flat_rotation = tuple(
                    0.0 if abs(float(value)) < 1e-10 else round(float(value), 10) for row in rotation for value in row
                )
                canonical_translation = []
                for value in translation:
                    modulo = float(value) % 1.0
                    if math.isclose(modulo, 0.0, abs_tol=1e-10) or math.isclose(modulo, 1.0, abs_tol=1e-10):
                        modulo = 0.0
                    canonical_translation.append(round(modulo, 10))
                if len(flat_rotation) != 9 or len(canonical_translation) != 3:
                    raise ValueError("model symmetry operation does not have a 3x3 rotation and 3-vector translation")
                normalized.append((flat_rotation, tuple(canonical_translation)))
            return latt, tuple(sorted(normalized))

        reference_symmetry = normalized_symmetry(reference_model)
        never_h_symmetry = normalized_symmetry(never_h_model)
        if never_h_symmetry != reference_symmetry:
            raise ValueError(
                "prospective never-H model does not share the full-H reference SHELX LATT/SYMM setting; "
                "align the models through common non-H atoms before auditing H peaks"
            )

        def non_h_sites(model: Mapping[str, object]) -> dict[str, tuple[str, tuple[float, float, float]]]:
            """Return each non-H atom's element and fractional position by label; raise on a blank or repeated label."""
            sites: dict[str, tuple[str, tuple[float, float, float]]] = {}
            for atom in model.get("atoms", []):
                if not isinstance(atom, Mapping):
                    continue
                element = str(atom.get("element") or "").upper()
                if element == "H":
                    continue
                name = str(atom.get("name") or "").upper()
                if not name or name in sites:
                    raise ValueError(
                        "full-H and never-H models need unique common non-H atom labels for an origin/alignment audit"
                    )
                sites[name] = (
                    element,
                    (float(atom["x"]), float(atom["y"]), float(atom["z"])),
                )
            return sites

        reference_non_h = non_h_sites(reference_model)
        never_h_non_h = non_h_sites(never_h_model)
        common_labels = sorted(set(reference_non_h) & set(never_h_non_h))
        minimum_common = max(
            3,
            int(math.ceil(float(args.min_common_nonh_fraction) * min(len(reference_non_h), len(never_h_non_h)))),
        )
        if len(common_labels) < minimum_common:
            raise ValueError(
                "full-H reference and prospective never-H model do not share enough uniquely labeled non-H atoms "
                f"for an origin/alignment audit ({len(common_labels)} found; {minimum_common} required)"
            )
        cell = gemmi.UnitCell(*reference_cell)
        common_distances: list[float] = []
        for label in common_labels:
            reference_element, reference_frac = reference_non_h[label]
            never_h_element, never_h_frac = never_h_non_h[label]
            if reference_element != never_h_element:
                raise ValueError(
                    f"common atom {label} changes element between full-H and never-H models: "
                    f"{reference_element} != {never_h_element}"
                )
            delta = [reference_frac[index] - never_h_frac[index] for index in range(3)]
            delta = [value - round(value) for value in delta]
            cart = cell.orthogonalize(gemmi.Fractional(*delta))
            common_distances.append(math.sqrt(cart.x * cart.x + cart.y * cart.y + cart.z * cart.z))
        common_nonh_rms = math.sqrt(
            math.fsum(distance * distance for distance in common_distances) / len(common_distances)
        )
        common_nonh_max = max(common_distances)
        if common_nonh_rms > float(args.max_common_nonh_rms_A) or common_nonh_max > float(args.max_common_nonh_max_A):
            raise ValueError(
                "full-H reference is not aligned to the prospective never-H coordinate origin/representation: "
                f"common non-H RMS {common_nonh_rms:.3f} A (limit {float(args.max_common_nonh_rms_A):.3f}), "
                f"max {common_nonh_max:.3f} A (limit {float(args.max_common_nonh_max_A):.3f}). "
                "Align from common non-H atoms without consulting H-map peaks, then rerun."
            )
        map_specs: list[str] = []
        map_contracts: list[dict[str, str]] = []
        for spec in args.map:
            name, separator, raw_path = str(spec).partition("=")
            if not separator:
                raw_path = name
                name = Path(raw_path).stem
            map_path = Path(raw_path).expanduser().resolve()
            if not map_path.is_file():
                raise ValueError(f"omit map does not exist: {map_path}")
            map_specs.append(f"{name}={map_path}")
            map_contracts.append({"name": name, "path": str(map_path), "sha256": sha256_file(map_path)})
        if not map_specs:
            raise ValueError("pass at least one --map name=/path/to/fofc.map")
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2))
        return 2

    out.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(engine_script("map_hydrogen_omit_audit.py")),
        "--reference-res",
        str(reference),
        "--out-dir",
        str(out),
        "--radius-A",
        str(float(args.radius_A)),
    ]
    for spec in map_specs:
        command.extend(["--map", spec])
    proc = subprocess.run(command, text=True, capture_output=True)
    if proc.returncode != 0:
        stderr = "\n".join(proc.stderr.strip().splitlines()[-30:])
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": "hydrogen omit-map audit failed",
                    "stderr_tail": stderr,
                },
                indent=2,
            )
        )
        return 2
    contract = {
        "schema_version": 1,
        "kind": "grond_prospective_never_h_map_audit_input",
        "reference_res": str(reference),
        "reference_res_sha256": sha256_file(reference),
        "reference_hydrogen_count": len(reference_h),
        "never_h_res": str(never_h),
        "never_h_res_sha256": sha256_file(never_h),
        "never_h_hydrogen_count": 0,
        "reference_cell": list(reference_cell),
        "never_h_cell": list(never_h_cell),
        "model_cell_max_length_relative_mismatch": cell_length_mismatch,
        "model_cell_max_angle_mismatch_deg": cell_angle_mismatch,
        "reference_shelx_latt": reference_symmetry[0],
        "reference_symmetry_ops": list(reference_symmetry[1]),
        "never_h_shelx_latt": never_h_symmetry[0],
        "never_h_symmetry_ops": list(never_h_symmetry[1]),
        "reference_non_h_count": len(reference_non_h),
        "never_h_non_h_count": len(never_h_non_h),
        "common_non_h_count": len(common_labels),
        "minimum_common_non_h_count": minimum_common,
        "common_non_h_labels": common_labels,
        "common_non_h_rms_A": common_nonh_rms,
        "common_non_h_max_A": common_nonh_max,
        "max_common_non_h_rms_A": float(args.max_common_nonh_rms_A),
        "max_common_non_h_max_A": float(args.max_common_nonh_max_A),
        "radius_A": float(args.radius_A),
        "maps": map_contracts,
        "coordinate_contract": (
            "reference and never-H maps must share the same cell, origin, proper HKL setting, and selected hand; "
            "the map reader verifies compatible cells but cannot infer an origin shift"
        ),
    }
    write_json(out / "input_contract.json", contract)
    summary_path = out / "summary.json"
    summary = try_read_json(summary_path)
    if summary:
        summary["input_contract"] = str((out / "input_contract.json").resolve())
        summary["never_h_res"] = str(never_h)
        summary["never_h_hydrogen_count"] = 0
        write_json(summary_path, summary)
    if proc.stdout.strip():
        print(proc.stdout.strip())
    print(json.dumps({"status": "ok", "input_contract": str((out / "input_contract.json").resolve())}, indent=2))
    return 0
