"""SHELX endpoint input and reflection formatting. This preserves the established one-cycle, full-correction endpoint contract; it does not make the frozen-q update exact dynamical least squares."""

from __future__ import annotations

import math
import re
from pathlib import Path

from grond.legacy.engine.shelx_hkl import safe_fixed_width_factor, write_fixed_width_hkl

DEFAULT_ELECTRON_WAVELENGTH_A = 0.025079


_SHELX_FLOAT_TOKEN_RE = re.compile(r"[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[Ee][-+]?\d+)?")


def parse_shelx_wght_terms(line: str) -> list[float]:
    """Return numeric WGHT terms from one SHELX line.

    SHELXL accepts a one-term WGHT record and treats the missing second term as
    zero.  A line-based parser is important here: using a whitespace regex can
    accidentally consume the ``E`` from a following EXTI line as a fake WGHT
    value.
    """

    parts = line.strip().split()
    if not parts or parts[0].upper() != "WGHT":
        return []
    terms: list[float] = []
    for token in parts[1:]:
        if not _SHELX_FLOAT_TOKEN_RE.fullmatch(token):
            break
        terms.append(float(token))
    return terms


def parse_shelx_wght_pair(line: str) -> tuple[float, float] | None:
    terms = parse_shelx_wght_terms(line)
    if not terms:
        return None
    return terms[0], terms[1] if len(terms) >= 2 else 0.0


def _iter_recommended_wght_terms(text: str):
    for line in text.splitlines():
        upper = line.upper()
        if "RECOMMENDED WEIGHTING SCHEME" not in upper or "WGHT" not in upper:
            continue
        terms = parse_shelx_wght_terms(line[upper.find("WGHT") :])
        if terms:
            yield terms


def parse_wght(res_path: Path) -> tuple[float, float]:
    for line in res_path.read_text(errors="replace").splitlines():
        pair = parse_shelx_wght_pair(line)
        if pair is not None:
            return pair
    return 0.05, 0.0


def write_hkl(path: Path, rows, float_format: str = "shelx_fixed") -> None:
    if float_format == "wide":
        fmt = "{h:4d}{k:4d}{l:4d}{I:14.4f}{sig:14.4f}\n"
    elif float_format == "shelx_fixed":
        write_fixed_width_hkl(path, rows)
        return
    else:
        raise ValueError(f"unknown hkl format {float_format!r}")
    with path.open("w") as f:
        for h, k, l, I, sig in rows:
            f.write(fmt.format(h=int(h), k=int(k), l=int(l), I=float(I), sig=float(sig)))
        f.write(fmt.format(h=0, k=0, l=0, I=0.0, sig=0.0))


def enforce_hkl_row_width(rows_merged, rows_unmerged, hkl_format: str):
    """Apply one common scale so both HKLF tables fit the requested format."""
    merged = list(rows_merged)
    unmerged = list(rows_unmerged)
    if hkl_format != "shelx_fixed":
        return merged, unmerged, 1.0
    factor = min(safe_fixed_width_factor(merged), safe_fixed_width_factor(unmerged))
    if factor < 1.0:
        merged = [
            (h, k, l, intensity * factor, sigma * factor)
            for h, k, l, intensity, sigma in merged
        ]
        unmerged = [
            (h, k, l, intensity * factor, sigma * factor)
            for h, k, l, intensity, sigma in unmerged
        ]
    return merged, unmerged, float(factor)


def shelx_wavelength_needs_electron_default(value: float) -> bool:
    """Return true when a SHELX CELL wavelength is clearly not an electron-diffraction value."""

    if not math.isfinite(float(value)) or float(value) <= 0.0:
        return True
    if abs(float(value) - 1.0) < 1e-6:
        return True
    # 200-300 keV electron wavelengths are near 0.025-0.020 A.  Values above
    # 0.1 A are almost always X-ray placeholders in small-molecule RES files.
    return float(value) >= 0.1


def normalize_shelx_cell_wavelength(line: str, default_wavelength: float = DEFAULT_ELECTRON_WAVELENGTH_A) -> str:
    """Normalize obvious placeholder CELL wavelengths for electron endpoints."""

    parts = line.split()
    if len(parts) < 8 or parts[0].upper() != "CELL":
        return line
    try:
        wavelength = float(parts[1])
    except ValueError:
        parts[1] = f"{default_wavelength:.6f}"
        return " ".join(parts)
    if shelx_wavelength_needs_electron_default(wavelength):
        parts[1] = f"{default_wavelength:.6f}"
        return " ".join(parts)
    return line


def force_shelxl_cycle_count(line: str, cycles: int = 1) -> tuple[str, bool]:
    """Force SHELXL LS/CGLS instruction lines to the requested internal cycle count."""

    match = re.match(r"^(\s*)(L\.S\.|LS|CGLS)(?=\s|$)(.*)$", line, flags=re.IGNORECASE)
    if not match:
        return line, False
    indent, command, tail = match.groups()
    count = max(1, int(cycles))
    rest = tail.split()
    if rest:
        rest[0] = str(count)
    else:
        rest = [str(count)]
    return f"{indent}{command} {' '.join(rest)}", True


def force_shelxl_one_cycle(line: str) -> tuple[str, bool]:
    """Force SHELXL LS/CGLS instruction lines to one internal cycle."""

    return force_shelxl_cycle_count(line, 1)


def _format_wght_line(a: float, b: float) -> str:
    return f"WGHT      {float(a):.4f}      {float(b):.4f}"


def set_active_wght_lines(lines: list[str], a: float, b: float) -> list[str]:
    """Return SHELX input lines with exactly one active pre-END WGHT line.

    SHELXL writes recommended WGHT records after END.  Those records are useful
    diagnostics, but they are not active input.  Endpoint files that users open
    in SHELXLE/Olex2 should therefore contain a single declared active WGHT in
    the input block and no stale tail after END.
    """

    out: list[str] = []
    inserted = False
    wght_line = _format_wght_line(a, b)
    for line in lines:
        upper = line.strip().upper()
        if upper.startswith("WGHT"):
            if not inserted:
                out.append(wght_line)
                inserted = True
            continue
        if not inserted and upper.startswith("FVAR"):
            out.append(wght_line)
            inserted = True
        out.append(line)
    if not inserted:
        insert_at = len(out)
        for i, line in enumerate(out):
            if line.strip().upper().startswith(("HKLF", "END")):
                insert_at = i
                break
        out.insert(insert_at, wght_line)
    return out


def _hklf_order(line: str) -> str:
    parts = line.split()
    return parts[1] if len(parts) > 1 else ""


def _unsupported_endpoint_reason(line: str) -> str:
    upper = line.strip().upper()
    head = upper.split()[0] if upper.split() else ""
    if head in {"BASF", "TWIN", "SWAT"}:
        return f"{head} is not supported by GROND corrected-HKLF4 endpoints yet"
    if head == "SQUEEZE" or "SQUEEZE" in upper:
        return "SQUEEZE/solvent-mask corrected SHELX models are not supported by GROND corrected-HKLF4 endpoints yet"
    if head == "HKLF":
        order = _hklf_order(line)
        if order not in {"3", "4"}:
            return f"HKLF {order or '?'} is not supported; GROND currently writes corrected intensity endpoints as HKLF 4"
    return ""


def write_endpoint_ins(
    src_res: Path,
    dst_ins: Path,
    hkl_name: str = "dyn_nuisance_corrected.hkl",
    keep_exti: bool = False,
    wght: tuple[float, float] | None = None,
    ls_cycles: int = 1,
) -> None:
    """Write the corrected-HKL endpoint input with GROND's endpoint contract."""

    out = []
    saw_cycle_instruction = False
    saw_hklf = False
    cycle_count = max(1, int(ls_cycles))
    for line in src_res.read_text(errors="replace").splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        unsupported = _unsupported_endpoint_reason(line)
        if unsupported:
            raise ValueError(unsupported)
        if upper.startswith("ACTA"):
            continue
        if not keep_exti and upper.startswith("EXTI"):
            continue
        if keep_exti and upper.startswith("EXTI") and "*" in stripped:
            # Large electron extinction parameters overflow SHELXL's RES field.
            # The parameter table has a wider field; never substitute zero.
            listing = src_res.with_suffix(".lst")
            values = []
            if listing.is_file():
                for record in listing.read_text(errors="replace").splitlines():
                    if record.rstrip().endswith("EXTI") and record[:6].strip().isdigit():
                        try:
                            value = float(record[6:17])
                        except ValueError:
                            continue
                        if math.isfinite(value):
                            values.append(value)
            if not values:
                raise ValueError(f"Overflowed EXTI in {src_res}; supply its SHELXL listing with the refined parameter")
            line = f"EXTI {values[-1]:.12g}"
        if upper.startswith("CELL "):
            line = normalize_shelx_cell_wavelength(line)
        line, forced_cycle = force_shelxl_cycle_count(line, cycle_count)
        saw_cycle_instruction = saw_cycle_instruction or forced_cycle
        if "<HklSrc" in line:
            line = f'REM <HklSrc "%./{hkl_name}">'
        if upper.startswith("HKLF"):
            line = "HKLF 4"
            saw_hklf = True
        out.append(line)
        if upper.startswith("END"):
            break
    if not saw_cycle_instruction:
        insert_at = len(out)
        for i, line in enumerate(out):
            if line.strip().upper().startswith(("HKLF", "END")):
                insert_at = i
                break
        out.insert(insert_at, f"L.S. {cycle_count}")
    if not saw_hklf:
        insert_at = len(out)
        for i, line in enumerate(out):
            if line.strip().upper().startswith("END"):
                insert_at = i
                break
        out.insert(insert_at, "HKLF 4")
    if wght is not None:
        out = set_active_wght_lines(out, float(wght[0]), float(wght[1]))
    dst_ins.write_text("\n".join(out) + "\n")


def write_no_exti_ins(src_res: Path, dst_ins: Path, hkl_name: str = "dyn_nuisance_corrected.hkl") -> None:
    write_endpoint_ins(src_res, dst_ins, hkl_name=hkl_name, keep_exti=False)
