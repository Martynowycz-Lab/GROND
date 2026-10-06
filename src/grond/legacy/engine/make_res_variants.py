#!/usr/bin/env python3
"""Write explicit alternate-hand and no-H RES variants for sample Bloch endpoint tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import gemmi

if __package__:
    from .bloch_engine import alternate_hand_model, parse_model
    from .native_structure_update import update_res_coordinates
else:  # pragma: no cover - direct script execution
    from bloch_engine import alternate_hand_model, parse_model
    from native_structure_update import update_res_coordinates


def sfac_elements(lines: list[str]) -> dict[int, str]:
    """Map each SFAC card's 1-based index to its element symbol."""

    out: dict[int, str] = {}
    idx = 1
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].upper() == "SFAC":
            out[idx] = parts[1].upper()
            idx += 1
    return out


def is_atom_line(parts: list[str]) -> bool:
    """Return whether the split tokens of a RES line form an atom record rather than a control card."""

    if len(parts) < 6 or not parts[1].lstrip("-").isdigit():
        return False
    # SHELX control records such as UNIT can also have a numeric second token.
    # They must not be treated as atoms when the H SFAC index is present.
    control_cards = {
        "CELL",
        "ZERR",
        "LATT",
        "SYMM",
        "SFAC",
        "UNIT",
        "WGHT",
        "FVAR",
        "AFIX",
        "PART",
        "BOND",
        "CONF",
        "PLAN",
        "LIST",
        "FMAP",
        "MORE",
        "ACTA",
        "HKLF",
        "END",
    }
    return parts[0].upper() not in control_cards


def is_h_atom_line(line: str, sfac: dict[int, str]) -> bool:
    """Return whether a RES line is a hydrogen atom, by its SFAC element or an H label."""

    parts = line.strip().split()
    if not is_atom_line(parts):
        return False
    label = parts[0].upper()
    try:
        element = sfac.get(abs(int(parts[1])), "")
    except ValueError:
        element = ""
    return element == "H" or label.startswith("H")


def strip_hydrogens(src: Path, dst: Path) -> None:
    """Copy a RES file without hydrogen atoms, BOND $H cards or AFIX blocks that held only hydrogens."""

    lines = src.read_text(errors="replace").splitlines()
    sfac = sfac_elements(lines)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        parts = line.strip().split()
        if parts and parts[0].upper() == "BOND" and "$H" in [p.upper() for p in parts[1:]]:
            i += 1
            continue
        if parts and parts[0].upper() == "AFIX" and len(parts) > 1 and parts[1] != "0":
            block = [line]
            i += 1
            while i < len(lines):
                block.append(lines[i])
                p = lines[i].strip().split()
                if p and p[0].upper() == "AFIX" and len(p) > 1 and p[1] == "0":
                    i += 1
                    break
                i += 1
            kept = [b for b in block if not is_h_atom_line(b, sfac)]
            has_nonh_atom = any(is_atom_line(b.strip().split()) and not is_h_atom_line(b, sfac) for b in kept)
            if has_nonh_atom:
                out.extend(kept)
            continue
        if is_h_atom_line(line, sfac):
            i += 1
            continue
        out.append(line)
        i += 1
    dst.write_text("\n".join(out) + "\n")


def inverted_operation_text(text: str) -> str:
    """Return a SHELX symmetry operation with its translation inverted."""

    operation = gemmi.Op(str(text).strip())
    operation.tran = [-int(value) for value in operation.tran]
    return operation.triplet()


def rewrite_alternate_hand_symmetry(path: Path) -> None:
    """Invert explicit SYMM and EQIV translations in a written SHELX model."""

    out: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        parts = stripped.split(maxsplit=2)
        key = parts[0].upper() if parts else ""
        prefix = line[: len(line) - len(line.lstrip())]
        if key == "SYMM" and len(parts) >= 2:
            operation_text = stripped[len(parts[0]) :].strip()
            try:
                inverted = inverted_operation_text(operation_text)
            except Exception as exc:
                raise ValueError(f"could not invert SHELX SYMM operation {operation_text!r}") from exc
            out.append(f"{prefix}{parts[0]} {inverted}")
            continue
        if key == "EQIV" and len(parts) >= 3:
            try:
                inverted = inverted_operation_text(parts[2])
            except Exception as exc:
                raise ValueError(f"could not invert SHELX EQIV operation {parts[2]!r}") from exc
            out.append(f"{prefix}{parts[0]} {parts[1]} {inverted}")
            continue
        out.append(line)
    path.write_text("\n".join(out) + "\n")


def symmetry_signature(model: dict) -> tuple[tuple[float, ...], ...]:
    """Return an order-independent crystallographic symmetry signature."""

    records: list[tuple[float, ...]] = []
    for operation in model.get("symmetry_ops", []):
        rotation = tuple(round(float(value), 8) for row in operation.get("rot", []) for value in row)
        translation = tuple(round(float(value) % 1.0, 8) for value in operation.get("trans", []))
        records.append((*rotation, *translation))
    return tuple(sorted(records))


def validate_alternate_hand_materialization(expected: dict, written: dict) -> None:
    """Fail if a written variant does not encode the requested hand exactly."""

    if symmetry_signature(written) != symmetry_signature(expected):
        raise ValueError("written alternate-hand RES does not encode the inverted symmetry operations")
    expected_atoms = {str(atom["name"]): atom for atom in expected.get("atoms", [])}
    written_atoms = {str(atom["name"]): atom for atom in written.get("atoms", [])}
    if expected_atoms.keys() != written_atoms.keys():
        raise ValueError("written alternate-hand RES atom labels differ from the requested model")
    for name, expected_atom in expected_atoms.items():
        written_atom = written_atoms[name]
        for axis in ("x", "y", "z"):
            delta = abs((float(written_atom[axis]) - float(expected_atom[axis]) + 0.5) % 1.0 - 0.5)
            if delta > 1.0e-5:
                raise ValueError(f"written alternate-hand coordinate {name}.{axis} differs by {delta:.6g}")


def write_alternate_hand_variant(src: Path, dst: Path) -> dict:
    """Write atoms and explicit symmetry cards for the inverted SHELX hand."""

    chiv_lines = [
        line.strip()
        for line in src.read_text(errors="replace").splitlines()
        if line.strip().upper().startswith("CHIV ") or line.strip().upper() == "CHIV"
    ]
    if chiv_lines:
        raise ValueError(
            "cannot materialize an alternate-hand RES with CHIV restraints: "
            "signed CHIV targets are not transformed; retain the input hand or edit the restraints explicitly"
        )
    model = alternate_hand_model(parse_model(src))
    update_res_coordinates(src, dst, model)
    rewrite_alternate_hand_symmetry(dst)
    validate_alternate_hand_materialization(model, parse_model(dst))
    return model


def main() -> None:
    """Write the alternate-hand and/or hydrogen-free RES variant, or a plain copy, to --output."""

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Input RES path.")
    ap.add_argument("--output", required=True, help="Output RES path.")
    ap.add_argument("--alternate-hand", dest="alternate_hand", action="store_true", help="Use alternate fractional coordinates before writing.")
    ap.add_argument("--remove-h", action="store_true", help="Remove H atoms and H-only AFIX blocks after hand variant processing.")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if args.alternate_hand:
        write_alternate_hand_variant(src, dst)
    if args.remove_h:
        src_for_strip = dst if args.alternate_hand else src
        strip_hydrogens(src_for_strip, dst)
    if not args.alternate_hand and not args.remove_h:
        dst.write_text(src.read_text(errors="replace"))


if __name__ == "__main__":
    main()
