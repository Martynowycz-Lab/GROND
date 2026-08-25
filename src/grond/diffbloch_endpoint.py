"""Experimental diffBloch endpoint handoff helpers.

The external optimizer writes a crystallographic CIF.  GROND must transfer only
the declared coordinates and displacement parameters back into the exact source
SHELX instruction model; scattering factors, AFIX records, occupancies, free
variables, weighting, and every other instruction remain owned by the source
RES.  This module implements that narrow, auditable conversion.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import gemmi
import numpy as np

from .legacy.engine.bloch_engine import cif_number, loop_rows, parse_res, shelx_u_cart_matrix
from .legacy.engine.native_structure_update import update_res_coordinates


def _cif_cell(block: gemmi.cif.Block) -> tuple[float, float, float, float, float, float]:
    tags = (
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    )
    cell = tuple(cif_number(block.find_value(tag)) for tag in tags)
    if not all(math.isfinite(value) for value in cell):
        raise ValueError("diffBloch CIF has incomplete or non-finite cell constants")
    return cell  # type: ignore[return-value]


def _cif_atoms(path: Path) -> tuple[tuple[float, ...], dict[str, dict[str, Any]]]:
    block = gemmi.cif.read_file(str(path)).sole_block()
    cell = _cif_cell(block)
    _aniso_tags, aniso_rows = loop_rows(block, "_atom_site_aniso_label")
    aniso: dict[str, dict[str, float]] = {}
    for row in aniso_rows:
        values = {str(tag).lower(): value for tag, value in row.items()}
        label = str(values.get("_atom_site_aniso_label", "")).strip().strip("'\"")
        if not label:
            continue
        aniso[label] = {
            "u11": cif_number(values.get("_atom_site_aniso_u_11")),
            "u22": cif_number(values.get("_atom_site_aniso_u_22")),
            "u33": cif_number(values.get("_atom_site_aniso_u_33")),
            "u12": cif_number(values.get("_atom_site_aniso_u_12")),
            "u13": cif_number(values.get("_atom_site_aniso_u_13")),
            "u23": cif_number(values.get("_atom_site_aniso_u_23")),
        }
    _atom_tags, atom_rows = loop_rows(block, "_atom_site_label")
    atoms: dict[str, dict[str, Any]] = {}
    for row in atom_rows:
        values = {str(tag).lower(): value for tag, value in row.items()}
        label = str(values.get("_atom_site_label", "")).strip().strip("'\"")
        element = str(values.get("_atom_site_type_symbol", "")).strip().strip("'\"")
        if not label or not element:
            continue
        if label in atoms:
            raise ValueError(f"diffBloch CIF repeats atom label {label!r}")
        atom: dict[str, Any] = {
            "name": label,
            "element": element,
            "x": cif_number(values.get("_atom_site_fract_x")),
            "y": cif_number(values.get("_atom_site_fract_y")),
            "z": cif_number(values.get("_atom_site_fract_z")),
            "occ": cif_number(values.get("_atom_site_occupancy", 1.0), 1.0),
        }
        if label in aniso:
            atom.update({"adp_type": "anis", **aniso[label]})
        else:
            atom.update(
                {
                    "adp_type": "iso",
                    "uiso": cif_number(values.get("_atom_site_u_iso_or_equiv", 0.05), 0.05),
                }
            )
        numeric = [value for key, value in atom.items() if key not in {"name", "element", "adp_type"}]
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError(f"diffBloch CIF atom {label!r} has non-finite parameters")
        atoms[label] = atom
    if not atoms:
        raise ValueError(f"diffBloch CIF contains no atom-site rows: {path}")
    return cell, atoms


def materialize_diffbloch_res(
    source_res: str | Path,
    refined_cif: str | Path,
    output_res: str | Path,
    *,
    audit_json: str | Path | None = None,
    max_cell_length_fraction: float = 0.01,
    max_cell_angle_degrees: float = 0.25,
) -> dict[str, Any]:
    """Transfer a diffBloch CIF model into a source RES without changing instructions."""

    source_path = Path(source_res).expanduser().resolve()
    cif_path = Path(refined_cif).expanduser().resolve()
    output_path = Path(output_res).expanduser().resolve()
    source = parse_res(source_path)
    cif_cell, cif_atoms = _cif_atoms(cif_path)
    source_cell = tuple(float(value) for value in source["cell"])
    length_fraction = max(
        abs(float(cif_cell[i]) - source_cell[i]) / max(source_cell[i], 1e-12)
        for i in range(3)
    )
    angle_difference = max(abs(float(cif_cell[i]) - source_cell[i]) for i in range(3, 6))
    if length_fraction > float(max_cell_length_fraction) or angle_difference > float(max_cell_angle_degrees):
        raise ValueError(
            "diffBloch/source cell mismatch exceeds the endpoint handoff limit: "
            f"length fraction={length_fraction:.5g}, angle={angle_difference:.5g} deg"
        )

    candidate = copy.deepcopy(source)
    missing_non_h: list[str] = []
    extra = sorted(set(cif_atoms) - {str(atom["name"]) for atom in source["atoms"]})
    shifts: list[dict[str, Any]] = []
    npd_atoms: list[str] = []
    cell = gemmi.UnitCell(*source_cell)
    for atom in candidate["atoms"]:
        label = str(atom["name"])
        refined = cif_atoms.get(label)
        if refined is None:
            if str(atom.get("element", "")).upper() != "H":
                missing_non_h.append(label)
            continue
        if str(refined["element"]).upper() != str(atom.get("element", "")).upper():
            raise ValueError(
                f"diffBloch/source element mismatch for {label}: "
                f"{refined['element']} versus {atom.get('element')}"
            )
        delta_frac = np.asarray(
            [float(refined[key]) - float(atom[key]) for key in ("x", "y", "z")],
            dtype=float,
        )
        delta_frac -= np.rint(delta_frac)
        orth = cell.orthogonalize(gemmi.Fractional(*delta_frac.tolist()))
        shift_A = float(math.sqrt(orth.x * orth.x + orth.y * orth.y + orth.z * orth.z))
        for key in ("x", "y", "z"):
            atom[key] = float(refined[key])
        transferred_adp = False
        if atom.get("adp_type") == "anis" and refined.get("adp_type") == "anis":
            for key in ("u11", "u22", "u33", "u12", "u13", "u23"):
                atom[key] = float(refined[key])
            transferred_adp = True
            eigenvalues = np.linalg.eigvalsh(shelx_u_cart_matrix(atom, source_cell))
            if float(np.min(eigenvalues)) <= 0.0:
                npd_atoms.append(label)
        elif (
            atom.get("adp_type") == "iso"
            and refined.get("adp_type") == "iso"
            and str(atom.get("element", "")).upper() != "H"
        ):
            atom["uiso"] = float(refined["uiso"])
            transferred_adp = True
            if float(atom["uiso"]) <= 0.0:
                npd_atoms.append(label)
        shifts.append(
            {
                "atom": label,
                "element": atom.get("element", ""),
                "shift_A": shift_A,
                "adp_transferred": transferred_adp,
            }
        )
    if missing_non_h:
        raise ValueError(
            "diffBloch CIF omits source non-hydrogen atom(s): " + ", ".join(sorted(missing_non_h))
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    update_res_coordinates(source_path, output_path, candidate)
    heavy = [row["shift_A"] for row in shifts if str(row["element"]).upper() != "H"]
    payload = {
        "schema": "grond.diffbloch-res-handoff.v1",
        "status": "rejected_npd" if npd_atoms else "complete",
        "source_res": str(source_path),
        "refined_cif": str(cif_path),
        "output_res": str(output_path),
        "source_cell": source_cell,
        "cif_cell": cif_cell,
        "max_cell_length_fraction": length_fraction,
        "max_cell_angle_difference_degrees": angle_difference,
        "source_atoms": len(source["atoms"]),
        "cif_atoms": len(cif_atoms),
        "transferred_atoms": len(shifts),
        "extra_cif_labels": extra,
        "missing_hydrogens_preserved_from_source": sorted(
            str(atom["name"])
            for atom in source["atoms"]
            if str(atom.get("element", "")).upper() == "H" and str(atom["name"]) not in cif_atoms
        ),
        "hydrogen_adp_policy": "preserve source SHELX Uiso/multiplier and AFIX instructions",
        "heavy_atom_rms_shift_A": float(math.sqrt(np.mean(np.square(heavy)))) if heavy else 0.0,
        "heavy_atom_max_shift_A": float(max(heavy)) if heavy else 0.0,
        "npd_atoms": sorted(npd_atoms),
        "atom_shifts": shifts,
    }
    if audit_json is not None:
        audit_path = Path(audit_json).expanduser().resolve()
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload
