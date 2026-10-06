"""Native structure-update row and SHELX output serialization. Preserve the supplied instruction records and the exact objective-row ledger; output formatting must not change a physical model."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from grond.bloch.model import shelx_encoded
from grond.legacy.engine.bloch_engine import XDSData

from .parameters import atom_ueq


def write_objective_rows(path: Path, data: XDSData, rows, target, sig, masks) -> None:
    """Write the objective rows as CSV: HKL, d, Iobs, sigma, work/free split and XDS geometry columns."""

    fieldnames = [
        "row_index",
        "h",
        "k",
        "l",
        "d_A",
        "Iobs",
        "sigma",
        "I_over_sigma",
        "split",
        "zcal",
        "rlp",
        "peak",
        "corr",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in np.asarray(rows, dtype=int):
            h, k, l = (int(v) for v in data.hkl[i])
            split = "free" if masks["free"][i] else "work"
            sigma = float(sig[i])
            w.writerow(
                {
                    "row_index": int(i),
                    "h": h,
                    "k": k,
                    "l": l,
                    "d_A": float(data.cell.calculate_d((h, k, l))),
                    "Iobs": float(target[i]),
                    "sigma": sigma,
                    "I_over_sigma": float(target[i] / sigma) if sigma > 0 else float("nan"),
                    "split": split,
                    "zcal": float(data.zcal[i]),
                    "rlp": float(data.rlp[i]),
                    "peak": float(data.peak[i]),
                    "corr": float(data.corr[i]),
                }
            )


def _plain_uiso(token: str) -> bool:
    """True for a Uiso written as a plain positive value (not a riding multiplier, fixed value or free variable)."""

    try:
        value = float(token)
    except ValueError:
        return False
    return value > 0.0 and not shelx_encoded(value)


def update_res_coordinates(src: Path, dst: Path, model) -> None:
    """Copy a SHELX .res, replacing the coordinates and ADPs of atoms in ``model``.

    Anisotropic atoms are rewritten on two lines. Isotropic atoms, including free hydrogens, take their updated
    ``uiso`` over a plain positive value; riding multipliers (negative) and free-variable or fixed encodings
    (``10*m+p``) are kept, as are occupancies and other instruction records.
    """

    atom_by_name = {a["name"]: a for a in model["atoms"]}
    lines = src.read_text(errors="replace").splitlines()
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        parts = stripped.split()
        if len(parts) >= 6 and parts[0] in atom_by_name and parts[1].lstrip("-").isdigit():
            atom = atom_by_name[parts[0]]
            prefix_len = len(line) - len(line.lstrip())
            prefix = line[:prefix_len]
            if atom.get("adp_type") == "anis":
                occ = parts[5]
                out.append(
                    f"{prefix}{parts[0]:<4s} {parts[1]:>2s} "
                    f"{float(atom['x']):11.6f}{float(atom['y']):10.6f}{float(atom['z']):10.6f} "
                    f"{occ:>8s} {float(atom['u11']):8.5f} {float(atom['u22']):8.5f} ="
                )
                out.append(
                    f" {float(atom['u33']):8.5f} {float(atom['u23']):8.5f} {float(atom['u13']):8.5f} {float(atom['u12']):8.5f}"
                )
                if line.rstrip().endswith("=") and i + 1 < len(lines):
                    i += 1
            else:
                rest = parts[5:]
                if "uiso" in atom and len(rest) >= 2 and _plain_uiso(rest[1]):
                    rest = [rest[0], f"{float(atom['uiso']):.5f}"] + rest[2:]
                newline = (
                    f"{prefix}{parts[0]:<4s} {parts[1]:>2s} "
                    f"{float(atom['x']):11.6f}{float(atom['y']):10.6f}{float(atom['z']):10.6f} "
                    + " ".join(rest)
                )
                out.append(newline)
        else:
            out.append(line)
        i += 1
    dst.write_text("\n".join(out) + "\n")


def write_parameter_csv(path: Path, params, x0, x_final):
    """Write each refined parameter with its start value, final value and shift."""

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["atom", "kind", "coord", "start", "final", "shift"])
        w.writeheader()
        for p, a, b in zip(params, x0, x_final, strict=False):
            w.writerow({"atom": p["atom"], "kind": p.get("kind", "xyz"), "coord": p["coord"], "start": float(a), "final": float(b), "shift": float(b - a)})


def write_model_delta_csv(path: Path, start_model, final_model):
    """Write each atom's coordinate shifts (fractional) and Ueq change between two models."""

    start_by_name = {a["name"]: a for a in start_model["atoms"]}
    with path.open("w", newline="") as f:
        fieldnames = ["atom", "element", "dx", "dy", "dz", "start_ueq", "final_ueq", "delta_ueq"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for atom in final_model["atoms"]:
            start = start_by_name.get(atom["name"])
            if start is None:
                continue
            w.writerow(
                {
                    "atom": atom["name"],
                    "element": atom.get("element", ""),
                    "dx": float(atom["x"]) - float(start["x"]),
                    "dy": float(atom["y"]) - float(start["y"]),
                    "dz": float(atom["z"]) - float(start["z"]),
                    "start_ueq": atom_ueq(start, start_model["cell"]),
                    "final_ueq": atom_ueq(atom, final_model["cell"]),
                    "delta_ueq": atom_ueq(atom, final_model["cell"])
                    - atom_ueq(start, start_model["cell"]),
                }
            )
