# ruff: noqa: E741
# h, k, l retain standard crystallographic Miller-index notation.
"""Electron scattering factors, crystallographic symmetry, occupancies and displacement tensors.

The internal occupancy is physical site occupancy; RES source SOFs retain their
separate multiplicity convention. No fitting or observation selection lives here."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

import gemmi
import numpy as np

from grond.bloch.constants import ELECTRON_COMPTON_WAVELENGTH_A, SITE_SYMMETRY_TOLERANCE_FRACTIONAL


def join_shelx_continuations(lines):
    out, pending = [], ""
    for raw in lines:
        line = raw.rstrip()
        pending = (pending + " " + line.strip()) if pending else line
        if pending.rstrip().endswith("="):
            pending = pending.rstrip()[:-1].strip()
            continue
        out.append(pending)
        pending = ""
    if pending:
        out.append(pending)
    return out


def parse_sfac(tokens):
    if len(tokens) < 10:
        raise ValueError(
            "SFAC records used for Bloch calculations must include an element and "
            "at least four explicit scattering-factor coefficient pairs"
        )
    element = tokens[1]
    try:
        vals = [float(v) for v in tokens[2:]]
    except ValueError as exc:
        raise ValueError(f"non-numeric SFAC coefficients for element {element}") from exc
    if not all(math.isfinite(value) for value in vals):
        raise ValueError(f"non-finite SFAC coefficients for element {element}")
    if not any(value != 0.0 for value in vals):
        raise ValueError(f"all-zero SFAC coefficients for element {element}")
    pairs = [(vals[0], vals[1]), (vals[2], vals[3]), (vals[4], vals[5]), (vals[6], vals[7])]
    c = vals[8] if len(vals) > 8 else 0.0
    return {"element": element, "pairs": pairs, "c": c, "raw": vals}


def gemmi_op_record(text: str):
    op = gemmi.Op(str(text).strip().replace(" ", "").lower())
    den = float(op.DEN)
    rot = (np.asarray(op.rot, dtype=float) / den).tolist()
    trans = (np.asarray(op.tran, dtype=float) / den).tolist()
    trans = [float(v % 1.0) for v in trans]
    return {"rot": rot, "trans": trans, "triplet": op.triplet()}


def lattice_translations_from_latt(latt_abs: int):
    # SHELX LATT codes. Most data here are primitive, but keeping centering
    # explicit prevents silent errors when this tool is reused.
    mapping = {
        1: [(0.0, 0.0, 0.0)],
        2: [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],  # I
        3: [(0.0, 0.0, 0.0), (2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), (1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0)],  # R
        4: [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],  # F
        5: [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5)],  # A
        6: [(0.0, 0.0, 0.0), (0.5, 0.0, 0.5)],  # B
        7: [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0)],  # C
    }
    return mapping.get(int(latt_abs), [(0.0, 0.0, 0.0)])


def unique_symmetry_records(records):
    out = []
    seen = set()
    for rec in records:
        rot = np.asarray(rec["rot"], dtype=float)
        trans = np.asarray(rec["trans"], dtype=float) % 1.0
        key = tuple(np.rint(rot * 24).astype(int).ravel().tolist() + np.rint(trans * 24).astype(int).tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append({"rot": rot.tolist(), "trans": trans.tolist(), "triplet": rec.get("triplet", "")})
    return out


def shelx_symmetry_records(latt: int, symm_texts):
    base = [gemmi_op_record("x,y,z")]
    base.extend(gemmi_op_record(text) for text in symm_texts)
    if int(latt) > 0:
        centrosym = []
        for rec in base:
            centrosym.append(rec)
            centrosym.append(
                {
                    "rot": (-np.asarray(rec["rot"], dtype=float)).tolist(),
                    "trans": (-np.asarray(rec["trans"], dtype=float)).tolist(),
                    "triplet": "",
                }
            )
        base = centrosym

    records = []
    for rec in base:
        rot = np.asarray(rec["rot"], dtype=float)
        trans = np.asarray(rec["trans"], dtype=float)
        for cent in lattice_translations_from_latt(abs(int(latt))):
            records.append(
                {
                    "rot": rot.tolist(),
                    "trans": (trans + np.asarray(cent, dtype=float)).tolist(),
                    "triplet": rec.get("triplet", ""),
                }
            )
    return unique_symmetry_records(records)


def cif_symmetry_records(block):
    rows = []
    for tag in ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"):
        _, rows = loop_rows(block, tag)
        if rows:
            break
    records = []
    for row in rows:
        values_by_tag = {str(tag).lower(): value for tag, value in row.items()}
        text = values_by_tag.get("_space_group_symop_operation_xyz") or values_by_tag.get("_symmetry_equiv_pos_as_xyz")
        if text:
            records.append(gemmi_op_record(str(text).strip().strip("'\"")))
    if records:
        return unique_symmetry_records(records)
    if rows:
        raise ValueError("CIF declares a symmetry-operation loop but contains no usable operations")

    declared_groups = []
    number = block.find_value("_space_group_IT_number") or block.find_value("_symmetry_Int_Tables_number")
    if number:
        parsed_number = cif_number(number)
        if not math.isfinite(parsed_number) or parsed_number != round(parsed_number):
            raise ValueError(f"CIF declares an invalid space-group number {number!r}")
        sg = gemmi.find_spacegroup_by_number(int(parsed_number))
        if sg is None:
            raise ValueError(f"CIF declares unknown space-group number {number!r}")
        declared_groups.append(("IT number", sg))

    for tag in (
        "_space_group_name_H-M_alt",
        "_symmetry_space_group_name_H-M",
        "_space_group_name_H-M_ref",
    ):
        raw = block.find_value(tag)
        if not raw:
            continue
        name = str(raw).strip().strip("'\"")
        sg = gemmi.find_spacegroup_by_name(name)
        if sg is None:
            raise ValueError(f"CIF declares an unrecognized space-group name {name!r} in {tag}")
        declared_groups.append((tag, sg))

    for tag in ("_space_group_name_Hall", "_symmetry_space_group_name_Hall"):
        raw = block.find_value(tag)
        if not raw:
            continue
        name = str(raw).strip().strip("'\"")
        sg = next((candidate for candidate in gemmi.spacegroup_table() if candidate.hall == name), None)
        if sg is None:
            raise ValueError(f"CIF declares an unrecognized Hall space-group name {name!r} in {tag}")
        declared_groups.append((tag, sg))

    if declared_groups:
        numbers = {int(group.number) for _source, group in declared_groups}
        if len(numbers) != 1:
            declarations = ", ".join(f"{source}={group.xhm()}" for source, group in declared_groups)
            raise ValueError(f"CIF space-group declarations disagree: {declarations}")
        sg = declared_groups[0][1]
        return unique_symmetry_records(gemmi_op_record(op.triplet()) for op in sg.operations())
    return [gemmi_op_record("x,y,z")]


SHELX_NON_ATOM_RECORDS = {
    "ACTA",
    "AFIX",
    "ANIS",
    "BASF",
    "BIND",
    "BLOC",
    "BOND",
    "BUMP",
    "CELL",
    "CGLS",
    "CHIV",
    "CONF",
    "CONN",
    "DAMP",
    "DANG",
    "DEFS",
    "DELU",
    "DFIX",
    "DISP",
    "EADP",
    "END",
    "EQIV",
    "EXTI",
    "EXYZ",
    "FEND",
    "FLAT",
    "FMAP",
    "FRAG",
    "FREE",
    "FVAR",
    "GRID",
    "HFIX",
    "HKLF",
    "HOPE",
    "HTAB",
    "ISOR",
    "L.S.",
    "LATT",
    "LAUE",
    "LIST",
    "MERG",
    "MOLE",
    "MORE",
    "MOVE",
    "MPLA",
    "NCSY",
    "NEUT",
    "OMIT",
    "PART",
    "PLAN",
    "PRIG",
    "REM",
    "RESI",
    "RIGU",
    "RTAB",
    "SADI",
    "SAME",
    "SFAC",
    "SHEL",
    "SIMU",
    "SIZE",
    "SPEC",
    "SQUEEZE",
    "STIR",
    "SUMP",
    "SWAT",
    "SYMM",
    "TEMP",
    "TIME",
    "TITL",
    "TWIN",
    "UNIT",
    "WGHT",
    "WPDB",
    "ZERR",
}


def shelx_parameter(value: float, free_variables: list[float], *, field: str) -> float:
    """Resolve SHELX's ``10*m+p`` parameter encoding.

    The first FVAR value is the overall scale.  Parameter references therefore
    start at ``m=2``.  A negative ``m`` denotes the complementary value.  Fixed
    values use ``m=1`` and are returned without the added ten.
    """

    raw = float(value)
    if not math.isfinite(raw):
        raise ValueError(f"non-finite SHELX {field}: {raw!r}")
    mode = math.trunc(raw / 10.0)
    multiplier = raw - 10.0 * mode
    if mode == 0 or abs(multiplier) >= 5.0:
        return raw
    if mode == 1:
        return multiplier
    if mode == -1:
        raise ValueError(f"invalid SHELX {field} free-variable reference {raw:g}: FVAR 1 is the scale")

    index = abs(mode) - 1
    if index >= len(free_variables):
        raise ValueError(
            f"SHELX {field} value {raw:g} references FVAR {abs(mode)}, "
            f"but only {len(free_variables)} FVAR values were supplied"
        )
    free_value = float(free_variables[index])
    if not math.isfinite(free_value):
        raise ValueError(f"SHELX FVAR {abs(mode)} is not finite")
    if mode > 1:
        return multiplier * free_value
    return multiplier * (free_value - 1.0)


def is_shelx_q_peak(label: str) -> bool:
    """Return whether a SHELX atom-like label is a residual-density Q peak."""

    return re.fullmatch(r"Q\d+[A-Z]?", str(label).strip(), flags=re.IGNORECASE) is not None


def cif_number(value, default=float("nan")):
    text = str(value).strip().strip("'\"")
    if not text or text in {".", "?"}:
        return float(default)
    if "(" in text:
        text = text.split("(", 1)[0]
    try:
        return float(text)
    except ValueError:
        return float(default)


def parse_res(path: Path):
    lines = join_shelx_continuations(path.read_text(errors="replace").splitlines())
    cell = None
    sfac = []
    atoms = []
    current_parent_ueq = None
    latt = 1
    symm_texts = []
    free_variables = []
    for line in lines:
        tokens = line.strip().split()
        if tokens and tokens[0].upper() == "FVAR":
            try:
                free_variables.extend(float(value) for value in tokens[1:])
            except ValueError as exc:
                raise ValueError(f"invalid FVAR instruction in {path}: {line.strip()}") from exc
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.upper().startswith("REM"):
            continue
        tokens = stripped.split()
        key = tokens[0].upper()
        if key == "CELL":
            if len(tokens) < 8:
                raise ValueError(f"CELL instruction in {path} does not contain six cell parameters")
            cell = tuple(float(v) for v in tokens[2:8])
            if (
                not all(math.isfinite(value) for value in cell)
                or any(value <= 0.0 for value in cell[:3])
                or any(not 0.0 < value < 180.0 for value in cell[3:])
            ):
                raise ValueError(f"CELL instruction in {path} contains invalid cell parameters")
        elif key == "SFAC":
            sfac.append(parse_sfac(tokens))
        elif key == "LATT" and len(tokens) > 1:
            latt = int(float(tokens[1]))
        elif key == "SYMM":
            symm_texts.append(stripped[4:].strip())
        elif key in {"END", "HKLF"}:
            if key == "HKLF":
                break
        elif key in SHELX_NON_ATOM_RECORDS:
            continue
        elif len(tokens) >= 7 and tokens[1].lstrip("-").isdigit():
            if is_shelx_q_peak(tokens[0]):
                continue
            sfac_index = int(tokens[1]) - 1
            if not 0 <= sfac_index < len(sfac):
                raise ValueError(
                    f"atom {tokens[0]} in {path} references SFAC {sfac_index + 1}, "
                    f"but {len(sfac)} scattering-factor records were read"
                )
            vals = [shelx_parameter(float(v), free_variables, field=f"{tokens[0]} ADP") for v in tokens[6:]]
            atom = {
                "name": tokens[0],
                "sfac_index": sfac_index,
                "element": sfac[sfac_index]["element"],
                "x": shelx_parameter(float(tokens[2]), free_variables, field=f"{tokens[0]} x"),
                "y": shelx_parameter(float(tokens[3]), free_variables, field=f"{tokens[0]} y"),
                "z": shelx_parameter(float(tokens[4]), free_variables, field=f"{tokens[0]} z"),
                "occ": shelx_parameter(float(tokens[5]), free_variables, field=f"{tokens[0]} occupancy"),
                "shelx_sof_encoded": float(tokens[5]),
                "raw_adp": vals,
            }
            numeric_atom_values = [
                atom["x"],
                atom["y"],
                atom["z"],
                atom["occ"],
                *vals,
            ]
            if not all(math.isfinite(float(value)) for value in numeric_atom_values):
                raise ValueError(f"atom {tokens[0]} in {path} contains non-finite parameters")
            if len(vals) >= 6:
                atom.update(
                    {
                        "adp_type": "anis",
                        "u11": vals[0],
                        "u22": vals[1],
                        "u33": vals[2],
                        "u23": vals[3],
                        "u13": vals[4],
                        "u12": vals[5],
                    }
                )
                if cell is None:
                    raise ValueError(f"CELL must precede anisotropic atom {tokens[0]} in {path}")
                current_parent_ueq = float(np.trace(shelx_u_cart_matrix(atom, cell))) / 3.0
            else:
                atom["adp_type"] = "iso"
                uiso = vals[0] if vals else 0.05
                if uiso < 0.0 and current_parent_ueq is not None:
                    uiso = abs(uiso) * current_parent_ueq
                elif uiso < 0.0:
                    uiso = abs(uiso) * 0.05
                atom["uiso"] = float(uiso)
                if atom["element"].upper() != "H" and uiso > 0.0:
                    current_parent_ueq = float(uiso)
            atoms.append(atom)
    if cell is None:
        raise ValueError(f"No CELL line in {path}")
    if not sfac:
        raise ValueError(f"No explicit electron-scattering SFAC records in {path}")
    if not atoms:
        raise ValueError(f"No atoms found before HKLF in {path}")
    model = {
        "cell": cell,
        "sfac": sfac,
        "atoms": atoms,
        "model_source": "shelx_res",
        "ignored_exti": None,
        "symmetry_ops": shelx_symmetry_records(latt, symm_texts),
        "shelx_latt": latt,
    }
    operation_count = len(model["symmetry_ops"])
    for atom in atoms:
        multiplicity = len(list(unique_symmetry_mates(atom, model)))
        if not multiplicity or operation_count % multiplicity:
            raise ValueError(
                f"inconsistent special-position multiplicity for {atom['name']} in {path}: "
                f"{multiplicity} sites from {operation_count} symmetry operations"
            )
        # SHELX's SOF already contains special/general multiplicity. The
        # internal model, like coreCIF, stores physical site occupancy because
        # fcalc expands unique sites. Retain both the encoded and decoded source
        # values so free-variable and partial-occupancy provenance is not lost.
        atom["shelx_sof"] = float(atom["occ"])
        atom["site_symmetry_order"] = operation_count // multiplicity
        atom["site_multiplicity"] = multiplicity
        atom["occ"] = atom["shelx_sof"] * atom["site_symmetry_order"]
    return model


def loop_rows(block, required_tag):
    col = block.find_loop(required_tag)
    if not col:
        return [], []
    loop = col.get_loop()
    tags = [str(t) for t in loop.tags]
    rows = []
    for i in range(loop.length()):
        rows.append({tag: loop[i, j] for j, tag in enumerate(tags)})
    return tags, rows


def parse_cif(path: Path):
    doc = gemmi.cif.read_file(str(path))
    block = doc.sole_block()
    cell = (
        cif_number(block.find_value("_cell_length_a")),
        cif_number(block.find_value("_cell_length_b")),
        cif_number(block.find_value("_cell_length_c")),
        cif_number(block.find_value("_cell_angle_alpha")),
        cif_number(block.find_value("_cell_angle_beta")),
        cif_number(block.find_value("_cell_angle_gamma")),
    )
    if (
        not all(np.isfinite(cell))
        or any(value <= 0.0 for value in cell[:3])
        or any(not 0.0 < value < 180.0 for value in cell[3:])
    ):
        raise ValueError(f"No usable cell constants in {path}")

    coefficient_kind = (
        str(block.find_value("_grond_scattering_factor_kind") or block.find_value("_atom_type_scat_source") or "")
        .strip()
        .strip("'\"")
        .lower()
    )
    if coefficient_kind not in {"electron", "electron_diffraction", "electron-diffraction"}:
        raise ValueError(
            f"CIF {path} does not identify its scattering coefficients as electron factors; "
            "add '_grond_scattering_factor_kind electron' or use a SHELX RES/INS model "
            "with explicit electron SFAC records"
        )

    sfac_tags, sfac_rows = loop_rows(block, "_atom_type_symbol")
    coefficient_tags = {
        *(f"_atom_type_scat_Cromer_Mann_a{i}" for i in range(1, 5)),
        *(f"_atom_type_scat_Cromer_Mann_b{i}" for i in range(1, 5)),
    }
    if sfac_rows and not {tag.lower() for tag in coefficient_tags}.issubset({tag.lower() for tag in sfac_tags}):
        raise ValueError(
            f"CIF {path} does not contain explicit electron scattering-factor coefficients; "
            "supply a CIF with coefficient columns or a SHELX RES/INS model with numeric SFAC records"
        )
    sfac = []
    symbol_to_index = {}
    for row in sfac_rows:
        values_by_tag = {str(tag).lower(): value for tag, value in row.items()}
        symbol = str(values_by_tag.get("_atom_type_symbol", "")).strip().strip("'\"")
        if not symbol:
            continue
        pairs = []
        for i in range(1, 5):
            a = cif_number(
                values_by_tag.get(f"_atom_type_scat_cromer_mann_a{i}", 0.0),
                0.0,
            )
            b = cif_number(
                values_by_tag.get(f"_atom_type_scat_cromer_mann_b{i}", 0.0),
                0.0,
            )
            pairs.append((a, b))
        c = cif_number(values_by_tag.get("_atom_type_scat_cromer_mann_c", 0.0), 0.0)
        coefficients = [value for pair in pairs for value in pair] + [c]
        if not all(math.isfinite(value) for value in coefficients) or not any(value != 0.0 for value in coefficients):
            raise ValueError(f"CIF {path} has invalid or all-zero scattering coefficients for {symbol}")
        symbol_to_index[symbol.upper()] = len(sfac)
        sfac.append({"element": symbol, "pairs": pairs, "c": c, "raw": [v for pair in pairs for v in pair] + [c]})
    if not sfac:
        raise ValueError(f"No _atom_type scattering-factor loop found in {path}")

    _, aniso_rows = loop_rows(block, "_atom_site_aniso_label")
    aniso = {}
    for row in aniso_rows:
        values_by_tag = {str(tag).lower(): value for tag, value in row.items()}
        label = str(values_by_tag.get("_atom_site_aniso_label", "")).strip().strip("'\"")
        if not label:
            continue
        forms = {}
        for kind, factor in (("u", 1.0), ("b", 1.0 / (8.0 * math.pi**2))):
            tags = {ij: f"_atom_site_aniso_{kind}_{ij}" for ij in ("11", "22", "33", "12", "13", "23")}
            if any(tag in values_by_tag for tag in tags.values()):
                values = {f"u{ij}": factor * cif_number(values_by_tag.get(tag)) for ij, tag in tags.items()}
                if not all(math.isfinite(value) for value in values.values()):
                    raise ValueError(f"CIF {path} has incomplete or non-finite {kind.upper()}ij for atom {label}")
                forms[kind] = {"adp_type": "anis", "adp_source": f"cif_{kind}ani", **values}
        if not forms:
            raise ValueError(f"CIF {path} has an anisotropic record without Uij or Bij for atom {label}")
        if (
            "u" in forms
            and "b" in forms
            and not np.allclose(
                [forms["u"][f"u{ij}"] for ij in ("11", "22", "33", "12", "13", "23")],
                [forms["b"][f"u{ij}"] for ij in ("11", "22", "33", "12", "13", "23")],
                rtol=1e-4,
                atol=1e-6,
            )
        ):
            raise ValueError(f"CIF {path} declares inconsistent Uij and Bij for atom {label}")
        aniso[label] = forms

    _, atom_rows = loop_rows(block, "_atom_site_label")
    atoms = []
    for row in atom_rows:
        values_by_tag = {str(tag).lower(): value for tag, value in row.items()}
        label = str(values_by_tag.get("_atom_site_label", "")).strip().strip("'\"")
        element = str(values_by_tag.get("_atom_site_type_symbol", "")).strip().strip("'\"")
        if not label or not element:
            continue
        sfac_index = symbol_to_index.get(element.upper())
        if sfac_index is None:
            raise ValueError(f"Atom {label} uses element {element}, but no matching _atom_type row was found in {path}")
        atom = {
            "name": label,
            "sfac_index": int(sfac_index),
            "element": element,
            "x": cif_number(values_by_tag.get("_atom_site_fract_x")),
            "y": cif_number(values_by_tag.get("_atom_site_fract_y")),
            "z": cif_number(values_by_tag.get("_atom_site_fract_z")),
            "occ": cif_number(values_by_tag.get("_atom_site_occupancy", 1.0), 1.0),
        }

        def adp_label(tag, values=values_by_tag):
            value = str(values.get(tag, "")).strip().strip("'\"").lower()
            return "" if value in {"", ".", "?"} else value

        declared = adp_label("_atom_site_adp_type")
        alternate = adp_label("_atom_site_thermal_displace_type")
        if declared and alternate and declared != alternate:
            raise ValueError(f"CIF {path} has conflicting ADP type labels for atom {label}")
        declared = declared or alternate
        if declared and declared not in {"uani", "bani", "uiso", "biso"}:
            raise ValueError(f"CIF {path} has unsupported ADP type {declared!r} for atom {label}")
        if label not in aniso:
            supplied_iso = {}
            for kind in ("u", "b"):
                tag = f"_atom_site_{kind}_iso_or_equiv"
                if tag in values_by_tag:
                    value = cif_number(values_by_tag[tag])
                    if not math.isfinite(value):
                        raise ValueError(
                            f"CIF {path} has a non-finite or malformed isotropic displacement value {tag} for atom {label}"
                        )
                    supplied_iso[kind] = value if kind == "u" else value / (8.0 * math.pi**2)
            if len(supplied_iso) == 2 and not math.isclose(
                supplied_iso["u"], supplied_iso["b"], rel_tol=1e-4, abs_tol=1e-6
            ):
                raise ValueError(f"CIF {path} declares inconsistent Uiso and Biso for atom {label}")
        if not declared:
            if label in aniso:
                declared = "uani" if "u" in aniso[label] else "bani"
            elif math.isfinite(cif_number(values_by_tag.get("_atom_site_u_iso_or_equiv"))):
                declared = "uiso"
            elif math.isfinite(cif_number(values_by_tag.get("_atom_site_b_iso_or_equiv"))):
                declared = "biso"
        if declared in {"uani", "bani"}:
            if declared[0] not in aniso.get(label, {}):
                raise ValueError(
                    f"CIF {path} declares {declared} but lacks matching anisotropic values for atom {label}"
                )
            atom.update(aniso[label][declared[0]])
        elif declared in {"uiso", "biso"}:
            value = cif_number(values_by_tag.get(f"_atom_site_{declared[0]}_iso_or_equiv"))
            if not math.isfinite(value):
                raise ValueError(f"CIF {path} declares {declared} but lacks its displacement value for atom {label}")
            atom.update(
                adp_type="iso",
                uiso=value if declared == "uiso" else value / (8.0 * math.pi**2),
                adp_source=f"cif_{declared}",
            )
        else:
            # Preserve the documented historical fallback only when no ADP
            # representation was supplied, never for an unsupported one.
            atom.update(adp_type="iso", uiso=0.05, adp_source="default_uiso")
        numeric_atom_values = [
            value
            for key, value in atom.items()
            if key in {"x", "y", "z", "occ", "uiso", "u11", "u22", "u33", "u12", "u13", "u23"}
        ]
        if not all(math.isfinite(float(value)) for value in numeric_atom_values):
            raise ValueError(f"atom {label} in CIF {path} contains non-finite parameters")
        atoms.append(atom)
    if not atoms:
        raise ValueError(f"No _atom_site loop found in {path}")

    exti = block.find_value("_ref" + "ine_ls_extinction_coef") or block.find_value("_shelx_extinction_method")
    return {
        "cell": cell,
        "sfac": sfac,
        "atoms": atoms,
        "model_source": "cif",
        "ignored_exti": str(exti) if exti else None,
        "symmetry_ops": cif_symmetry_records(block),
    }


def parse_model(path: Path):
    if path.suffix.lower() == ".cif":
        return parse_cif(path)
    return parse_res(path)


def alternate_hand_model(model):
    out = copy.deepcopy(model)
    for atom in out["atoms"]:
        atom["x"] = (-float(atom["x"])) % 1.0
        atom["y"] = (-float(atom["y"])) % 1.0
        atom["z"] = (-float(atom["z"])) % 1.0
    for operation in out.get("symmetry_ops", []):
        operation["trans"] = [(-float(value)) % 1.0 for value in operation.get("trans", (0.0, 0.0, 0.0))]
    return out


def model_content_hash(model):
    payload = {
        "cell": [round(float(v), 8) for v in model.get("cell", [])],
        "sfac": model.get("sfac", []),
        "atoms": [
            {
                "sfac_index": int(a["sfac_index"]),
                "x": round(float(a["x"]), 8),
                "y": round(float(a["y"]), 8),
                "z": round(float(a["z"]), 8),
                "occ": round(float(a["occ"]), 8),
                "adp_type": a.get("adp_type", ""),
                "uiso": round(float(a.get("uiso", 0.0)), 8),
                "u11": round(float(a.get("u11", 0.0)), 8),
                "u22": round(float(a.get("u22", 0.0)), 8),
                "u33": round(float(a.get("u33", 0.0)), 8),
                "u23": round(float(a.get("u23", 0.0)), 8),
                "u13": round(float(a.get("u13", 0.0)), 8),
                "u12": round(float(a.get("u12", 0.0)), 8),
            }
            for a in model.get("atoms", [])
        ],
        "symmetry_ops": [
            {
                "rot": [[round(float(v), 8) for v in row] for row in op.get("rot", [])],
                "trans": [round(float(v) % 1.0, 8) for v in op.get("trans", [])],
            }
            for op in model.get("symmetry_ops", [])
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def symmetry_ops_p212121():
    return [
        (np.diag([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0])),
        (np.diag([-1.0, -1.0, 1.0]), np.array([0.5, 0.0, 0.5])),
        (np.diag([-1.0, 1.0, -1.0]), np.array([0.0, 0.5, 0.5])),
        (np.diag([1.0, -1.0, -1.0]), np.array([0.5, 0.5, 0.0])),
    ]


def model_symmetry_ops(model):
    records = model.get("symmetry_ops") or []
    if not records:
        return [(np.eye(3), np.zeros(3))]
    return [(np.asarray(op["rot"], dtype=float), np.asarray(op["trans"], dtype=float)) for op in records]


def reciprocal_lengths(cell_tuple):
    uc = gemmi.UnitCell(*[float(v) for v in cell_tuple[:6]])
    rec = uc.reciprocal()
    return float(rec.a), float(rec.b), float(rec.c)


def inv_d2_from_hkl_array(h, k, l, cell_tuple):
    uc = gemmi.UnitCell(*[float(v) for v in cell_tuple[:6]])
    rec = uc.reciprocal()
    ast, bst, cst = float(rec.a), float(rec.b), float(rec.c)
    alst, best, gast = (math.radians(float(v)) for v in (rec.alpha, rec.beta, rec.gamma))
    return (
        (h * ast) ** 2
        + (k * bst) ** 2
        + (l * cst) ** 2
        + 2.0 * h * k * ast * bst * math.cos(gast)
        + 2.0 * h * l * ast * cst * math.cos(best)
        + 2.0 * k * l * bst * cst * math.cos(alst)
    )


def sfac_value(sfac, s2):
    return float(sum(a * math.exp(-b * s2) for a, b in sfac["pairs"]) + sfac.get("c", 0.0))


def sfac_value_array(sfac, s2):
    out = np.zeros_like(s2, dtype=float) + float(sfac.get("c", 0.0))
    for a, b in sfac["pairs"]:
        out += float(a) * np.exp(-float(b) * s2)
    return out


def cell_orthogonalization_matrix(cell_tuple):
    """Return the matrix that maps fractional coordinates to Cartesian Angstroms."""

    uc = gemmi.UnitCell(*[float(v) for v in cell_tuple[:6]])
    axes = (
        uc.orthogonalize(gemmi.Fractional(1, 0, 0)),
        uc.orthogonalize(gemmi.Fractional(0, 1, 0)),
        uc.orthogonalize(gemmi.Fractional(0, 0, 1)),
    )
    return np.asarray([[axis.x, axis.y, axis.z] for axis in axes], dtype=float).T


def shelx_u_matrix(atom):
    """Return a SHELX/CIF crystallographic U tensor in symmetric matrix form."""

    return np.asarray(
        [
            [float(atom["u11"]), float(atom["u12"]), float(atom["u13"])],
            [float(atom["u12"]), float(atom["u22"]), float(atom["u23"])],
            [float(atom["u13"]), float(atom["u23"]), float(atom["u33"])],
        ],
        dtype=float,
    )


def shelx_u_cart_matrix(atom, cell_tuple):
    """Convert crystallographic Uij values to a Cartesian displacement tensor.

    SHELX and coreCIF store Uij in the crystallographic convention used by the
    structure-factor exponent: the reciprocal-vector components are
    ``(h*a*, k*b*, l*c*)``.  Geometry restraints, by contrast, require a true
    Cartesian tensor.  This conversion is identity only for orthogonal cells.
    """

    orth = cell_orthogonalization_matrix(cell_tuple)
    reciprocal_scale = np.diag(reciprocal_lengths(cell_tuple))
    transform = orth @ reciprocal_scale
    return transform @ shelx_u_matrix(atom) @ transform.T


def atom_dw(atom, hkl, cell_tuple):
    h, k, l = (float(v) for v in hkl)
    if atom.get("adp_type") == "anis":
        ast, bst, cst = reciprocal_lengths(cell_tuple)
        scaled_hkl = np.asarray([h * ast, k * bst, l * cst], dtype=float)
        arg = float(scaled_hkl @ shelx_u_matrix(atom) @ scaled_hkl)
        return math.exp(-2.0 * math.pi * math.pi * arg)
    s2 = 0.25 * inv_d2_from_hkl_array(h, k, l, cell_tuple)
    return math.exp(-8.0 * math.pi * math.pi * float(atom.get("uiso", 0.05)) * s2)


def atom_dw_array(atom, h, k, l, s2, cell_tuple):
    if atom.get("adp_type") == "anis":
        ast, bst, cst = reciprocal_lengths(cell_tuple)
        u11, u22, u33 = float(atom["u11"]), float(atom["u22"]), float(atom["u33"])
        u12, u13, u23 = float(atom["u12"]), float(atom["u13"]), float(atom["u23"])
        sx = np.asarray(h, dtype=float) * ast
        sy = np.asarray(k, dtype=float) * bst
        sz = np.asarray(l, dtype=float) * cst
        arg = (
            sx * sx * u11
            + sy * sy * u22
            + sz * sz * u33
            + 2.0 * sx * sy * u12
            + 2.0 * sx * sz * u13
            + 2.0 * sy * sz * u23
        )
        return np.exp(-2.0 * math.pi * math.pi * arg)
    return np.exp(-8.0 * math.pi * math.pi * float(atom.get("uiso", 0.05)) * s2)


def unique_symmetry_mates(atom, model):
    """Yield unique sites with only a tight coordinate-rounding tolerance.

    Periodic differences handle positions written as 0.999999 versus zero.
    This is not a near-special-position constraint or coordinate adjustment.
    """

    xyz = np.asarray([atom["x"], atom["y"], atom["z"]], dtype=float)
    seen = []
    for rotation, translation in model_symmetry_ops(model):
        position = np.mod(rotation @ xyz + translation, 1.0)
        if any(
            np.max(np.abs((position - prior) - np.rint(position - prior))) <= SITE_SYMMETRY_TOLERANCE_FRACTIONAL
            for prior in seen
        ):
            continue
        seen.append(position)
        yield position, rotation


def fcalc_res(model, hkl):
    h, k, l = (int(v) for v in hkl)
    s2 = 0.25 * inv_d2_from_hkl_array(h, k, l, model["cell"])
    total = 0.0 + 0.0j
    for atom in model["atoms"]:
        sf = sfac_value(model["sfac"][atom["sfac_index"]], s2)
        for pos, rot in unique_symmetry_mates(atom, model):
            mate_hkl = np.asarray(rot, dtype=float).T @ np.asarray([h, k, l], dtype=float)
            dw = atom_dw(atom, mate_hkl, model["cell"])
            phase = 2.0 * math.pi * (h * pos[0] + k * pos[1] + l * pos[2])
            total += atom["occ"] * sf * dw * complex(math.cos(phase), math.sin(phase))
    return total


def fcalc_res_grid(model, h, k, l):
    h = np.asarray(h, dtype=float)
    k = np.asarray(k, dtype=float)
    l = np.asarray(l, dtype=float)
    s2 = 0.25 * inv_d2_from_hkl_array(h, k, l, model["cell"])
    total = np.zeros(h.shape, dtype=np.complex128)
    for atom in model["atoms"]:
        sf = sfac_value_array(model["sfac"][atom["sfac_index"]], s2)
        for pos, rot in unique_symmetry_mates(atom, model):
            mate_h = rot[0, 0] * h + rot[1, 0] * k + rot[2, 0] * l
            mate_k = rot[0, 1] * h + rot[1, 1] * k + rot[2, 1] * l
            mate_l = rot[0, 2] * h + rot[1, 2] * k + rot[2, 2] * l
            mate_s2 = 0.25 * inv_d2_from_hkl_array(mate_h, mate_k, mate_l, model["cell"])
            dw = atom_dw_array(atom, mate_h, mate_k, mate_l, mate_s2, model["cell"])
            amp = float(atom["occ"]) * sf * dw
            phase = 2.0 * math.pi * (h * pos[0] + k * pos[1] + l * pos[2])
            total += amp * np.exp(1j * phase)
    return total


def relativistic_gamma_from_wavelength(wavelength_A: float) -> float:
    """Return the electron Lorentz factor implied by its de Broglie wavelength."""

    wavelength_A = float(wavelength_A)
    if not math.isfinite(wavelength_A) or wavelength_A <= 0.0:
        raise ValueError("electron wavelength must be finite and positive")
    return math.sqrt(1.0 + (ELECTRON_COMPTON_WAVELENGTH_A / wavelength_A) ** 2)
