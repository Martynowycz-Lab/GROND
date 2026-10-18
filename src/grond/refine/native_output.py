"""Write coordinate updates into a SHELX RES; the supplied instruction records are preserved and output formatting must not change a physical model."""
from __future__ import annotations

from pathlib import Path

from grond.bloch.model import shelx_encoded


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
