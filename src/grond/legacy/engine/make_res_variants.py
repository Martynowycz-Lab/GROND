#!/usr/bin/env python3
"""Write explicit polarity and no-H RES variants for sample Bloch endpoint tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from native_structure_update import update_res_coordinates
from bloch_engine import coordinate_polarity_model, parse_model


def sfac_elements(lines: list[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    idx = 1
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].upper() == "SFAC":
            out[idx] = parts[1].upper()
            idx += 1
    return out


def is_atom_line(parts: list[str]) -> bool:
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Input RES path.")
    ap.add_argument("--output", required=True, help="Output RES path.")
    ap.add_argument("--coordinate-polarity-alt", action="store_true", help="Use alternate fractional coordinates before writing.")
    ap.add_argument("--remove-h", action="store_true", help="Remove H atoms and H-only AFIX blocks after polarity variant processing.")
    args = ap.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst
    if args.coordinate_polarity_alt:
        model = coordinate_polarity_model(parse_model(src))
        update_res_coordinates(src, dst, model)
        tmp = dst
    if args.remove_h:
        if tmp == dst and args.coordinate_polarity_alt:
            src_for_strip = dst
        else:
            src_for_strip = src
        strip_hydrogens(src_for_strip, dst)
    if not args.coordinate_polarity_alt and not args.remove_h:
        dst.write_text(src.read_text(errors="replace"))


if __name__ == "__main__":
    main()
