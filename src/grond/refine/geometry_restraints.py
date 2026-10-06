"""Periodic structure geometry and starting-model restraints. These restraints preserve the supplied geometry; they do not independently validate chemistry or discover missing atoms."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import gemmi
import numpy as np

from grond.bloch.model import join_shelx_continuations

from .parameters import atom_is_refinable, atom_ueq

COVALENT_RADII_A = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "P": 1.07,
    "F": 0.57,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
}


def frac_xyz(atom):
    """Return an atom's fractional coordinates as an array."""

    return np.asarray([float(atom["x"]), float(atom["y"]), float(atom["z"])], dtype=float)


def frac_delta_min_image(atom_i, atom_j, cell_tuple):
    """Return the closest periodic image under the full unit-cell metric."""

    delta = frac_xyz(atom_j) - frac_xyz(atom_i)
    basis = np.column_stack(
        [
            cart_delta(cell_tuple, np.asarray([1.0, 0.0, 0.0])),
            cart_delta(cell_tuple, np.asarray([0.0, 1.0, 0.0])),
            cart_delta(cell_tuple, np.asarray([0.0, 0.0, 1.0])),
        ]
    )
    _q, upper = np.linalg.qr(basis)
    target = upper @ delta
    if np.any(np.abs(np.diag(upper)) <= np.finfo(float).eps):
        raise ValueError("unit-cell basis is singular; no periodic closest image exists")

    # Babai's nearest-plane point supplies a finite initial radius. A short
    # three-dimensional sphere search then proves the closest lattice vector,
    # even when a highly skew, non-reduced cell needs a translation beyond +/-1.
    best_translation = np.zeros(3, dtype=int)
    for index in range(2, -1, -1):
        tail = float(upper[index, index + 1 :] @ best_translation[index + 1 :])
        best_translation[index] = int(np.rint((target[index] - tail) / upper[index, index]))
    best_distance2 = float(np.sum((target - upper @ best_translation) ** 2))
    candidate = np.zeros(3, dtype=int)

    def search(index: int, partial_distance2: float) -> None:
        """Depth-first search over integer translations for the lattice vector nearest the target (sphere decoding)."""

        nonlocal best_distance2, best_translation
        if index < 0:
            if partial_distance2 < best_distance2:
                best_distance2 = partial_distance2
                best_translation = candidate.copy()
            return
        remaining = max(best_distance2 - partial_distance2, 0.0)
        tail = float(upper[index, index + 1 :] @ candidate[index + 1 :])
        center = (target[index] - tail) / upper[index, index]
        radius = math.sqrt(remaining) / abs(float(upper[index, index]))
        lower = math.ceil(center - radius - 1e-12)
        upper_bound = math.floor(center + radius + 1e-12)
        values = sorted(range(lower, upper_bound + 1), key=lambda value: abs(value - center))
        for value in values:
            residual = target[index] - tail - upper[index, index] * value
            new_distance2 = partial_distance2 + float(residual * residual)
            if new_distance2 <= best_distance2 + 1e-15:
                candidate[index] = int(value)
                search(index - 1, new_distance2)

    search(2, 0.0)
    return np.asarray(delta - best_translation, dtype=float)


def cart_delta(cell_tuple, frac_delta):
    """Convert a fractional-coordinate difference to a Cartesian vector in A."""

    a, b, c, alpha, beta, gamma = (float(v) for v in cell_tuple)
    if abs(alpha - 90.0) < 1e-6 and abs(beta - 90.0) < 1e-6 and abs(gamma - 90.0) < 1e-6:
        return np.asarray([frac_delta[0] * a, frac_delta[1] * b, frac_delta[2] * c], dtype=float)
    cell = gemmi.UnitCell(a, b, c, alpha, beta, gamma)
    zero = cell.orthogonalize(gemmi.Fractional(0.0, 0.0, 0.0))
    vec = cell.orthogonalize(gemmi.Fractional(float(frac_delta[0]), float(frac_delta[1]), float(frac_delta[2])))
    return np.asarray([vec.x - zero.x, vec.y - zero.y, vec.z - zero.z], dtype=float)


def atom_distance_A(model, ai: int, aj: int) -> float:
    """Return the shortest periodic distance in A between atoms ``ai`` and ``aj``."""

    dfrac = frac_delta_min_image(model["atoms"][ai], model["atoms"][aj], model["cell"])
    return float(np.linalg.norm(cart_delta(model["cell"], dfrac)))


def build_start_restraints(model, selection: str, label_text: str, tolerance_A: float, max_distance_A: float):
    """Return bond restraints for atom pairs that are bonded in the starting model.

    A pair is bonded when its distance is between 0.45 A and the smaller of
    ``max_distance_A`` and the sum of covalent radii plus ``tolerance_A``, and at least one
    atom is refined. Q peaks are skipped. Each restraint keeps the starting distance and the
    starting Ueq difference.
    """

    labels = {x.strip() for x in re.split(r"[,\s]+", label_text or "") if x.strip()}
    atoms = model["atoms"]
    bonds = []
    for i, ai in enumerate(atoms):
        if ai["name"].upper().startswith("Q"):
            continue
        ei = ai.get("element", "").upper()
        ri = COVALENT_RADII_A.get(ei, 0.77)
        for j in range(i + 1, len(atoms)):
            aj = atoms[j]
            if aj["name"].upper().startswith("Q"):
                continue
            ej = aj.get("element", "").upper()
            rj = COVALENT_RADII_A.get(ej, 0.77)
            target = atom_distance_A(model, i, j)
            cutoff = min(float(max_distance_A), ri + rj + float(tolerance_A))
            if target < 0.45 or target > cutoff:
                continue
            update_i = atom_is_refinable(ai, selection, labels)
            update_j = atom_is_refinable(aj, selection, labels)
            if not (update_i or update_j):
                continue
            bonds.append(
                {
                    "i": i,
                    "j": j,
                    "atom_i": ai["name"],
                    "atom_j": aj["name"],
                    "element_i": ei,
                    "element_j": ej,
                    "target_A": float(target),
                    "start_ueq_diff": atom_ueq(ai, model["cell"]) - atom_ueq(aj, model["cell"]),
                    "updates_i": bool(update_i),
                    "updates_j": bool(update_j),
                }
            )
    return bonds


def bond_restraint_residuals(model, restraints, sigma_A: float):
    """Return (d - d_start) / sigma_A for each bond restraint, or an empty array when sigma_A <= 0."""

    if sigma_A <= 0.0 or not restraints:
        return np.zeros(0, dtype=float)
    out = []
    for r in restraints:
        d = atom_distance_A(model, int(r["i"]), int(r["j"]))
        out.append((d - float(r["target_A"])) / float(sigma_A))
    return np.asarray(out, dtype=float)


def adp_similarity_residuals(model, restraints, sigma_u: float):
    """Return the change in Ueq difference of each non-H bonded pair divided by ``sigma_u``
    (empty when sigma_u <= 0).
    """

    if sigma_u <= 0.0 or not restraints:
        return np.zeros(0, dtype=float)
    out = []
    atoms = model["atoms"]
    for r in restraints:
        ai = atoms[int(r["i"])]
        aj = atoms[int(r["j"])]
        if ai.get("element", "").upper() == "H" or aj.get("element", "").upper() == "H":
            continue
        diff = atom_ueq(ai, model["cell"]) - atom_ueq(aj, model["cell"])
        out.append((diff - float(r["start_ueq_diff"])) / float(sigma_u))
    return np.asarray(out, dtype=float)


def write_restraints_csv(path: Path, restraints):
    """Write the bond restraints as a CSV table."""

    with path.open("w", newline="") as f:
        fieldnames = ["atom_i", "atom_j", "element_i", "element_j", "target_A", "start_ueq_diff", "updates_i", "updates_j"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in restraints:
            w.writerow({k: r[k] for k in fieldnames})


def write_cctbx_restraint_view(src: Path, dst: Path) -> None:
    """Write a SHELX view cctbx can parse without losing the ED SFAC in src.

    cctbx's SHELX parser rejects explicit electron scattering coefficient SFAC
    rows, but the restraint/constraint grammar only needs element labels.  This
    view is therefore for cctbx/smtbx restraint and constraint discovery only;
    the Bloch calculator still reads the original ED SFAC rows.
    """
    elems = []
    lines = []
    for line in join_shelx_continuations(src.read_text(errors="replace").splitlines()):
        parts = line.split("!", 1)[0].split()
        if parts and parts[0].upper() == "SFAC":
            if len(parts) < 2:
                raise ValueError("SFAC record has no element labels")
            # A numeric record defines one indexed scattering factor; an
            # element-only record defines one per label, in the listed order.
            try:
                float(parts[2])
            except (IndexError, ValueError):
                elems.extend(parts[1:])
            else:
                elems.append(parts[1])
            continue
        lines.append(line)
    insert_at = 0
    for i, line in enumerate(lines):
        parts = line.split()
        if parts and parts[0].upper() == "LATT":
            insert_at = i + 1
            break
    lines.insert(insert_at, "SFAC  " + " ".join(elems))
    dst.write_text("\n".join(lines) + "\n")
