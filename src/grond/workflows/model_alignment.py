"""Put an independent heavy-atom model in a reference setting without copying atoms."""

from __future__ import annotations

import copy
import itertools
import math
from collections import Counter
from pathlib import Path

import gemmi
import numpy as np
from scipy.optimize import linear_sum_assignment

from grond.bloch.model import parse_model, reciprocal_lengths, shelx_u_matrix
from grond.refine.native_output import update_res_coordinates


def require_no_hydrogen(path: Path) -> dict:
    model = parse_model(path)
    hydrogens = [a["name"] for a in model["atoms"] if a["element"].upper() in {"H", "D"}]
    if hydrogens:
        raise ValueError(f"H-free model contains hydrogen atoms: {', '.join(hydrogens[:10])}")
    for line in path.read_text().splitlines():
        words = line.split()
        if words and words[0].upper() == "HFIX" and len(words) > 1 and float(words[1]) != 0:
            raise ValueError("Remove HFIX instructions that would generate H atoms before using never-h.")
    return model


def _metric(cell):
    orth = np.asarray(gemmi.UnitCell(*cell).orth.mat)
    return orth.T @ orth


def transformed_cell(cell, matrix):
    inverse = np.linalg.inv(matrix)
    metric = inverse.T @ _metric(cell) @ inverse
    lengths = np.sqrt(np.diag(metric))
    angles = [math.degrees(math.acos(np.clip(metric[i, j] / (lengths[i] * lengths[j]), -1, 1)))
              for i, j in ((1, 2), (0, 2), (0, 1))]
    return tuple([*lengths, *angles])


def setting_matrices(source, reference):
    """Small integer cell changes include axis permutations and common monoclinic settings."""
    source_metric, target_metric = _metric(source["cell"]), _metric(reference["cell"])
    candidates = np.array(list(itertools.product((-1, 0, 1), repeat=9))).reshape(-1, 3, 3)
    candidates = candidates[np.abs(np.linalg.det(candidates)).round() == 1]
    inv = np.linalg.inv(candidates)
    metrics = inv.transpose(0, 2, 1) @ source_metric @ inv
    scale = np.sqrt(np.outer(np.diag(target_metric), np.diag(target_metric)))
    valid = np.max(np.abs(metrics - target_metric) / scale, axis=(1, 2)) < 0.02
    return candidates[valid]


def _operations_match(source, reference, matrix, shift):
    inverse = np.linalg.inv(matrix)
    expected = reference["symmetry_ops"]
    if len(source["symmetry_ops"]) != len(expected):
        return False
    for op in source["symmetry_ops"]:
        rotation = matrix @ np.array(op["rot"]) @ inverse
        translation = matrix @ np.array(op["trans"]) + shift - rotation @ shift
        if not any(np.allclose(rotation, other["rot"], atol=1e-7) and
                   np.allclose((translation - other["trans"]) - np.rint(translation - other["trans"]), 0, atol=1e-6)
                   for other in expected):
            return False
    return True


def align_heavy_model(source: dict, reference: dict) -> tuple[dict, dict]:
    """Register using heavy atoms only. Coordinates are transformed, never replaced."""
    left = source["atoms"]
    right = [a for a in reference["atoms"] if a["element"].upper() not in {"H", "D"}]
    def composition(atoms):
        return Counter(a["element"].upper() for a in atoms)
    if composition(left) != composition(right):
        raise ValueError(f"Heavy-atom compositions differ: input {composition(left)}, reference {composition(right)}")
    x = np.array([[a[k] for k in ("x", "y", "z")] for a in left])
    y = np.array([[a[k] for k in ("x", "y", "z")] for a in right])
    compatible = np.array([[a["element"].upper() == b["element"].upper() for b in right] for a in left])
    orth = np.asarray(gemmi.UnitCell(*reference["cell"]).orth.mat)
    operations = reference["symmetry_ops"]
    mates = np.stack([y @ np.array(op["rot"]).T + op["trans"] for op in operations], axis=1)
    constraints = np.vstack([np.eye(3) - op["rot"] for op in operations])
    _, singular, vt = np.linalg.svd(constraints)
    free = vt[singular < 1e-7].T
    projector = free @ free.T
    images = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    best = None
    for matrix in setting_matrices(source, reference):
        transformed = x @ matrix.T
        for start in itertools.product((0, 0.25, 0.5, 0.75), repeat=3):
            shift = np.array(start, dtype=float)
            if not _operations_match(source, reference, matrix, shift):
                continue
            for _ in range(5 if free.size else 1):
                delta = mates[None, :, :, :] - transformed[:, None, None, :] - shift
                delta -= np.rint(delta)
                offsets = delta[:, :, :, None, :] + images
                squared = np.sum((offsets @ orth.T) ** 2, axis=-1)
                cost = squared.min(axis=(2, 3))
                cost[~compatible] = 1e12
                ri, ci = linear_sum_assignment(cost)
                flat = squared[ri, ci].reshape(len(ri), -1).argmin(axis=1)
                op_index, image_index = np.unravel_index(flat, squared.shape[2:])
                residual = offsets[ri, ci, op_index, image_index]
                update = projector @ np.mean(residual, axis=0)
                if np.linalg.norm(update) < 1e-10:
                    break
                shift += update
            distances = np.sqrt(cost[ri, ci])
            score = float(np.sqrt(np.mean(distances ** 2)))
            if best is None or score < best[0]:
                best = (score, float(max(distances)), matrix, shift, ri, ci)
    if best is None or best[0] > 0.35 or best[1] > 0.8:
        raise ValueError(f"No reliable heavy-atom setting match (RMS/max: {None if best is None else best[:2]} Angstrom). Supply a model in the reference setting.")
    rms, maximum, matrix, shift, ri, ci = best
    aligned = copy.deepcopy(source)
    aligned_cell = transformed_cell(source["cell"], matrix)
    source_recip = np.diag(reciprocal_lengths(source["cell"]))
    target_recip_inv = np.diag(1 / np.array(reciprocal_lengths(aligned_cell)))
    for atom in aligned["atoms"]:
        position = matrix @ np.array([atom[k] for k in ("x", "y", "z")]) + shift
        # Keep molecules contiguous for AFIX and distance restraints. Wrapping
        # atoms independently can turn a short bond into a cell-length vector.
        atom.update(zip(("x", "y", "z"), position, strict=True))
        if atom.get("adp_type") == "anis":
            change = target_recip_inv @ matrix @ source_recip
            tensor = change @ shelx_u_matrix(atom) @ change.T
            for key, (i, j) in zip(("u11", "u22", "u33", "u23", "u13", "u12"), ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)), strict=True):
                atom[key] = float(tensor[i, j])
    aligned["cell"] = aligned_cell
    aligned["symmetry_ops"] = copy.deepcopy(operations)
    return aligned, {"fractional_matrix": matrix.tolist(), "fractional_translation": shift.tolist(),
                     "heavy_atom_rms_A": rms, "heavy_atom_max_A": maximum,
                     "input_cell": list(source["cell"]), "aligned_cell": list(aligned_cell),
                     "reference_cell": list(reference["cell"]), "input_cell_metric_preserved": True,
                     "reference_hand_imposed": True, "coordinates_copied_from_reference": False,
                     "atom_mapping": {left[i]["name"]: right[j]["name"] for i, j in zip(ri, ci, strict=True)}}


def write_aligned_res(source: Path, reference: Path, destination: Path) -> dict:
    reference_model = parse_model(reference)
    model, report = align_heavy_model(require_no_hydrogen(source), reference_model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    update_res_coordinates(source, destination, model)
    reference_lines = reference.read_text().splitlines()
    symmetry = [line for line in reference_lines if line.split() and line.split()[0].upper() in {"LATT", "SYMM"}]
    wavelength = next(line.split()[1] for line in source.read_text().splitlines() if line.upper().startswith("CELL "))
    cell = "CELL " + wavelength + " " + " ".join(f"{value:.8g}" for value in model["cell"])
    if len(cell) > 79:
        words = cell.split()
        cell = " ".join(words[:5]) + " =\n " + " ".join(words[5:])
    zerr = []
    for line in source.read_text().splitlines():
        words = line.split()
        if words and words[0].upper() == "ZERR" and len(words) >= 8:
            initial = np.array(report["input_cell"])
            jacobian = np.zeros((6, 6))
            for axis in range(6):
                delta = np.zeros(6)
                delta[axis] = 1e-5
                plus = np.array(transformed_cell(initial + delta, report["fractional_matrix"]))
                minus = np.array(transformed_cell(initial - delta, report["fractional_matrix"]))
                jacobian[:, axis] = (plus - minus) / (2e-5)
            sigma = np.sqrt((jacobian ** 2) @ np.square([float(v) for v in words[2:8]]))
            zerr = ["ZERR " + words[1] + " " + " ".join(f"{s:.6g}" for s in sigma)]
            report["cell_uncertainties"] = "First-order propagation of the source ZERR, assuming uncorrelated cell parameters."
    output = []
    for line in destination.read_text().splitlines():
        key = line.split()[0].upper() if line.split() else ""
        if key == "CELL":
            output.extend([cell, *zerr, *symmetry])
        elif key in {"LATT", "SYMM", "ZERR"}:
            continue
        elif key == "EQIV":
            raise ValueError("Explicit EQIV restraints need manual setting conversion before never-h.")
        else:
            output.append(line)
        if key == "HKLF":
            output.append("END")
            break
    destination.write_text("\n".join(output) + "\n")
    # Use the same electron factors in both arms, even if their input files
    # rounded the coefficients differently. Atom types keep their input indices.
    factors = {}
    for sfac in reference_model["sfac"]:
        element = sfac["element"].upper()
        if element in factors and factors[element] != list(sfac["raw"]):
            raise ValueError(f"Reference has multiple scattering tables for {element}; an explicit atom-type mapping is needed.")
        factors[element] = list(sfac["raw"])
    rewritten, continuation = [], False
    for line in destination.read_text().splitlines():
        words = line.split()
        if continuation:
            continuation = line.rstrip().endswith("=")
            continue
        if words and words[0].upper() == "SFAC" and words[1].upper() in factors:
            continuation = line.rstrip().endswith("=")
            current = "SFAC " + words[1]
            for value in factors[words[1].upper()]:
                token = f"{value:.12g}"
                if len(current) + len(token) + 1 > 76:
                    rewritten.append(current + " =")
                    current = " "
                current += " " + token
            rewritten.append(current)
        else:
            rewritten.append(line)
    destination.write_text("\n".join(rewritten) + "\n")
    report["scattering_factors"] = "Reference electron coefficients by element; input atom-type indices retained."
    require_no_hydrogen(destination)
    return report


def reindex_hkl(source: Path, destination: Path, matrix) -> None:
    """Intensity and sigma stay unchanged under the reciprocal basis change."""
    reciprocal = np.linalg.inv(np.asarray(matrix)).T
    lines = []
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        try:
            hkl = np.array([int(line[j:j + 4]) for j in (0, 4, 8)])
        except ValueError as exc:
            raise ValueError(f"Expected fixed-width SHELX HKLF4 data in {source}") from exc
        transformed = reciprocal @ hkl
        if not np.allclose(transformed, np.rint(transformed), atol=1e-8):
            raise ValueError("Setting change produces fractional reflection indices")
        lines.append("".join(f"{int(round(h)):4d}" for h in transformed) + line[12:])
        if not hkl.any():
            break
    destination.write_text("\n".join(lines) + "\n")
