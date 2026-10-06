"""Prepare the SHELX model used by an explicit endpoint push."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from grond.reports.shelxl import _quick_lst_diagnostics

SHELX_RESTRAINT_COMMANDS = {
    "RIGU",
    "DFIX",
    "DANG",
    "SADI",
    "FLAT",
    "CHIV",
    "SAME",
    "DELU",
    "SIMU",
    "ISOR",
    "EADP",
    "EXYZ",
}


def is_shelx_atom_line(line: str) -> bool:
    """Return true for ordinary SHELX atom records."""

    toks = line.split()
    if len(toks) < 7:
        return False
    if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", toks[0]):
        return False
    try:
        int(toks[1])
        [float(x) for x in toks[2:7]]
    except ValueError:
        return False
    return True


def shelx_label_element(label: str) -> str:
    """Return the leading element-like character from a SHELX atom label."""

    match = re.match(r"^[A-Za-z]+", label)
    return match.group(0).upper()[0] if match else ""


def prepare_push_endpoint_model(args: argparse.Namespace, out_dir: Path) -> tuple[Path, Path | None]:
    """Prepare the start model for an endpoint-push run."""

    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.start_res).expanduser().resolve()
    if args.model_prep == "as-is" and int(args.set_ls_cycles) <= 0:
        return source, None

    lines = source.read_text().splitlines()
    active: list[str] = []
    for line in lines:
        active.append(line)
        if line.strip().upper().startswith("END"):
            break

    cell = [1.0, 1.0, 1.0]
    sfac_elements: list[str] = []
    for line in active:
        toks = line.split()
        if toks and toks[0].upper() == "CELL" and len(toks) >= 8:
            cell = [float(x) for x in toks[2:5]]
        if toks and toks[0].upper() == "SFAC" and len(toks) >= 2:
            candidates = [str(value).strip().upper() for value in toks[1:]]
            element_tokens = [value for value in candidates if re.fullmatch(r"[A-Z][A-Z]?", value)]
            if len(element_tokens) == len(candidates):
                sfac_elements.extend(element_tokens)
            elif element_tokens:
                sfac_elements.append(element_tokens[0])

    def atom_element(toks: list[str]) -> str:
        """Return an atom line's element from its SFAC index, else from its label."""
        try:
            sfac_index = int(toks[1])
        except (IndexError, ValueError):
            sfac_index = 0
        if 1 <= sfac_index <= len(sfac_elements):
            return sfac_elements[sfac_index - 1]
        return shelx_label_element(toks[0])

    heavy: list[dict[str, object]] = []
    i = 0
    while i < len(active):
        line = active[i]
        if is_shelx_atom_line(line):
            toks = line.split()
            label = toks[0]
            if atom_element(toks) != "H" and line.rstrip().endswith("=") and i + 1 < len(active):
                cont = active[i + 1].split()
                try:
                    ueq = (float(toks[6]) + float(toks[7]) + float(cont[0])) / 3.0
                    heavy.append(
                        {
                            "label": label,
                            "xyz": tuple(float(x) % 1.0 for x in toks[2:5]),
                            "ueq": max(0.006, min(0.08, ueq)),
                        }
                    )
                except (ValueError, IndexError):
                    pass
        i += 1

    def distance2(p: tuple[float, float, float], q: tuple[float, float, float]) -> float:
        """Return the squared distance (A^2) between fractional positions, nearest lattice image, cell angles ignored."""
        dx = ((p[0] - q[0] + 0.5) % 1.0) - 0.5
        dy = ((p[1] - q[1] + 0.5) % 1.0) - 0.5
        dz = ((p[2] - q[2] + 0.5) % 1.0) - 0.5
        return (dx * cell[0]) ** 2 + (dy * cell[1]) ** 2 + (dz * cell[2]) ** 2

    def nearest_heavy_u(xyz: tuple[float, float, float]) -> tuple[float, str | None]:
        """Return the Ueq (A^2) and label of the nearest heavy atom; 0.03 and None when there are none."""
        if not heavy:
            return 0.03, None
        rec = min(heavy, key=lambda item: distance2(xyz, item["xyz"]))  # type: ignore[arg-type]
        return float(rec["ueq"]), str(rec["label"])

    remove_restraints = args.model_prep in {
        "no-restraints",
        "free-h-isotropic-no-restraints",
        "free-h-anis-no-restraints",
    }
    free_h_isotropic = args.model_prep == "free-h-isotropic-no-restraints"
    free_h_anis = args.model_prep == "free-h-anis-no-restraints"
    free_h = free_h_isotropic or free_h_anis
    adp_stabilized = args.model_prep == "adp-stabilized"
    npd_diagnostics = _quick_lst_diagnostics(source.with_suffix(".lst"))
    if not npd_diagnostics.get("exists"):
        npd_diagnostics = _quick_lst_diagnostics(source.with_suffix(".log"))
    npd_targets = {str(atom) for atom in (npd_diagnostics.get("npd_atoms") or [])}
    existing_restraints = {
        line.split()[0].upper()
        for line in active
        if line.split() and line.split()[0].upper() in SHELX_RESTRAINT_COMMANDS
    }
    new_lines: list[str] = []
    removed_restraints: list[str] = []
    removed_afix = 0
    converted_h: list[dict[str, object]] = []
    adp_isotropic: list[dict[str, object]] = []
    adp_restraints_added: list[str] = []
    inserted_rem = False
    inserted_adp_stabilizers = False

    def add_adp_stabilizers() -> None:
        """Append the ADP-stabilizing lines once: REM notes, RIGU, SIMU and ISOR on NPD atoms, unless present."""
        nonlocal inserted_adp_stabilizers
        if inserted_adp_stabilizers or not adp_stabilized:
            return
        new_lines.append("REM GROND endpoint-push ADP-stabilized replay branch")
        new_lines.append(
            "REM GROND endpoint-push keeps/adds ADP restraints and converts listed NPD atoms to isotropic fallback"
        )
        if npd_targets:
            new_lines.append("REM GROND endpoint-push NPD atoms: " + " ".join(sorted(npd_targets)))
        if "RIGU" not in existing_restraints:
            new_lines.append("RIGU")
            adp_restraints_added.append("RIGU")
        if "SIMU" not in existing_restraints:
            new_lines.append("SIMU 0.04 0.08")
            adp_restraints_added.append("SIMU 0.04 0.08")
        if npd_targets and "ISOR" not in existing_restraints:
            labels = sorted(npd_targets)
            for offset in range(0, len(labels), 8):
                command = "ISOR 0.02 0.04 " + " ".join(labels[offset : offset + 8])
                new_lines.append(command)
                adp_restraints_added.append(command)
        inserted_adp_stabilizers = True

    def isotropic_atom_line(toks: list[str], u_value: float) -> str:
        """Return a SHELX isotropic atom line with U clamped to [0.006, 0.08] A^2."""
        label = toks[0]
        sfac = int(toks[1])
        x, y, z = [float(v) for v in toks[2:5]]
        occ = float(toks[5])
        u = max(0.006, min(0.08, abs(float(u_value))))
        return f"{label:<4} {sfac:>2d} {x:11.6f} {y:11.6f} {z:11.6f} {occ:11.5f} {u:10.5f}"

    i = 0
    while i < len(active):
        line = active[i]
        toks = line.split()
        key = toks[0].upper() if toks else ""

        if not inserted_rem and key == "LATT":
            new_lines.append(f"REM GROND endpoint-push prepared model: {args.model_prep}")
            if int(args.set_ls_cycles) > 0:
                new_lines.append(f"REM GROND endpoint-push set L.S. {int(args.set_ls_cycles)}")
            inserted_rem = True

        if (
            adp_stabilized
            and not inserted_adp_stabilizers
            and (is_shelx_atom_line(line) or key == "HKLF" or key == "END")
        ):
            add_adp_stabilizers()
        if remove_restraints and key in SHELX_RESTRAINT_COMMANDS:
            removed_restraints.append(line)
            i += 1
            continue
        if free_h and key == "AFIX":
            removed_afix += 1
            i += 1
            continue
        if key in {"L.S.", "LS", "CGLS"} and int(args.set_ls_cycles) > 0:
            new_lines.append(f"{toks[0]} {int(args.set_ls_cycles)}")
            i += 1
            continue
        if key == "END":
            if adp_stabilized and not inserted_adp_stabilizers:
                add_adp_stabilizers()
            if free_h:
                mode = "anisotropic" if free_h_anis else "isotropic"
                new_lines.append(f"REM GROND endpoint-push free {mode} H count: {len(converted_h)}")
            new_lines.append("END")
            i += 1
            continue
        if adp_stabilized and is_shelx_atom_line(line) and toks[0] in npd_targets:
            label = toks[0]
            old_u = float(toks[6]) if len(toks) > 6 else 0.02
            ueq = old_u
            consumed = 1
            if line.rstrip().endswith("=") and i + 1 < len(active):
                cont = active[i + 1].split()
                try:
                    ueq = (float(toks[6]) + float(toks[7]) + float(cont[0])) / 3.0
                    consumed = 2
                except (ValueError, IndexError):
                    ueq = old_u
            new_lines.append(isotropic_atom_line(toks, ueq))
            adp_isotropic.append({"label": label, "old_u": old_u, "fallback_u": max(0.006, min(0.08, abs(float(ueq))))})
            i += consumed
            continue
        if free_h and is_shelx_atom_line(line) and atom_element(toks) == "H":
            label = toks[0]
            sfac = int(toks[1])
            x, y, z = [float(v) for v in toks[2:5]]
            occ = float(toks[5])
            old_u = float(toks[6]) if len(toks) > 6 else -1.2
            parent_ueq, parent = nearest_heavy_u((x % 1.0, y % 1.0, z % 1.0))
            multiplier = abs(old_u) if old_u < 0 else 1.2
            if multiplier < 0.5 or multiplier > 2.0:
                multiplier = 1.2
            u = max(0.010, min(0.080, parent_ueq * multiplier))
            if free_h_anis:
                new_lines.append(
                    f"{label:<4} {sfac:>2d} {x:11.6f} {y:11.6f} {z:11.6f} {occ:11.5f} {u:10.5f} {u:10.5f} ="
                )
                new_lines.append(f" {u:10.5f} {0.0:10.5f} {0.0:10.5f} {0.0:10.5f}")
            else:
                new_lines.append(f"{label:<4} {sfac:>2d} {x:11.6f} {y:11.6f} {z:11.6f} {occ:11.5f} {u:10.5f}")
            converted_h.append(
                {
                    "label": label,
                    "parent": parent,
                    "parent_ueq": parent_ueq,
                    "initial_adp_type": "anisotropic" if free_h_anis else "isotropic",
                    "init_u_diag": u,
                    "old_u": old_u,
                }
            )
            i += 1
            continue

        new_lines.append(line)
        i += 1

    prepared = out_dir / "start_prepared.res"
    audit = out_dir / "start_prepared_audit.json"
    prepared.write_text("\n".join(new_lines) + "\n")
    audit.write_text(
        json.dumps(
            {
                "source": str(source),
                "output": str(prepared),
                "model_prep": args.model_prep,
                "set_ls_cycles": int(args.set_ls_cycles),
                "discarded_post_end_lines": max(0, len(lines) - len(active)),
                "removed_restraints": removed_restraints,
                "removed_afix": removed_afix,
                "converted_hydrogens": converted_h,
                "adp_stabilized_npd_atoms": sorted(npd_targets),
                "adp_stabilized_restraints_added": adp_restraints_added,
                "adp_stabilized_isotropic_fallback": adp_isotropic,
            },
            indent=2,
        )
        + "\n"
    )
    return prepared, audit
