"""SHELX endpoint input and reflection formatting. This preserves the established one-cycle, full-correction endpoint contract; it does not make the frozen-q update exact dynamical least squares.

Every generated input strips DAMP unless a caller asks for the legacy carry-over: a DAMP rescue in one
SHELXL call must not silently damp every later call, macro-cycle and the publication endpoint.  On the
absolute corrected-intensity scale (Fo^2 = kappa*|F|^2, ``ABSOLUTE_HKL_KAPPA``) the OSF is reset to
sqrt(kappa) at every handoff so that SHELXL never chases an arbitrary per-cycle intensity scale.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from grond.legacy.engine.shelx_hkl import safe_fixed_width_factor, write_fixed_width_hkl

DEFAULT_ELECTRON_WAVELENGTH_A = 0.025079

# Corrected intensities on the absolute scale are written as Fo^2 = kappa*|F|^2 (electron F in A), and the
# SHELXL OSF is set to sqrt(kappa).
ABSOLUTE_HKL_KAPPA = 10.0


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
    """Return (a, b) from a WGHT line, with b = 0 for a one-term WGHT, or None for another line."""

    terms = parse_shelx_wght_terms(line)
    if not terms:
        return None
    return terms[0], terms[1] if len(terms) >= 2 else 0.0


def _iter_recommended_wght_terms(text: str):
    """Yield the WGHT terms of every ``Recommended weighting scheme`` line in SHELXL output text."""

    for line in text.splitlines():
        upper = line.upper()
        if "RECOMMENDED WEIGHTING SCHEME" not in upper or "WGHT" not in upper:
            continue
        terms = parse_shelx_wght_terms(line[upper.find("WGHT") :])
        if terms:
            yield terms


def parse_wght(res_path: Path) -> tuple[float, float]:
    """Return the first WGHT (a, b) of a SHELX .res/.ins, or SHELXL's default (0.05, 0.0) when there is none."""

    for line in res_path.read_text(errors="replace").splitlines():
        pair = parse_shelx_wght_pair(line)
        if pair is not None:
            return pair
    return 0.05, 0.0


def write_hkl(path: Path, rows, float_format: str = "shelx_fixed") -> None:
    """Write (h, k, l, I, sigma) rows as a SHELX HKLF 4 file ending with a 0 0 0 row.

    ``shelx_fixed`` writes the validated fixed-width HKLF 4 layout; ``wide`` writes
    3I4,2F14.4 for inspection.
    """

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


def _format_wght_line(a: float, b: float) -> str:
    """Format a two-term WGHT instruction with four decimals."""

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


def set_fvar_osf(line: str, osf: float) -> str:
    """Return a FVAR line whose first free variable (the OSF) is ``osf``."""

    parts = line.split()
    if not parts or parts[0].upper() != "FVAR":
        return line
    return "FVAR " + " ".join([f"{float(osf):.5f}"] + parts[2:])


def _hklf_order(line: str) -> str:
    """Return the HKLF code of an HKLF instruction line (for example ``4``), or an empty string."""

    parts = line.split()
    return parts[1] if len(parts) > 1 else ""


def _unsupported_endpoint_reason(line: str, *, allow_basf: bool = False) -> str:
    """Return why a SHELX instruction line cannot go into a corrected-HKLF 4 endpoint, or an empty string."""

    upper = line.strip().upper()
    head = upper.split()[0] if upper.split() else ""
    if head == "BASF" and allow_basf:
        return ""
    if head in {"BASF", "TWIN", "SWAT"}:
        return f"{head} is not supported by GROND corrected-HKLF4 endpoints yet"
    if head == "SQUEEZE" or "SQUEEZE" in upper:
        return "SQUEEZE/solvent-mask corrected SHELX models are not supported by GROND corrected-HKLF4 endpoints yet"
    if head == "HKLF":
        order = _hklf_order(line)
        if order not in {"3", "4"}:
            return f"HKLF {order or '?'} is not supported; GROND currently writes corrected intensity endpoints as HKLF 4"
    return ""


Q_PEAK_RE = re.compile(r"^Q\d+\s", flags=re.IGNORECASE)
# SHELXL refuses instruction lines longer than 80 characters and writes at most seven batch scales per BASF
# record in its .res, repeating the BASF keyword; GROND writes and reads them the same way.
BASF_VALUES_PER_RECORD = 7
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")


def basf_records(values) -> list[str]:
    """BASF records of ``values`` in SHELXL's own layout (seven values per record)."""

    numbers = [float(value) for value in values]
    return [
        "BASF " + " ".join(f"{value:.5f}" for value in numbers[start : start + BASF_VALUES_PER_RECORD])
        for start in range(0, len(numbers), BASF_VALUES_PER_RECORD)
    ]


def requests_acta(path: Path) -> bool:
    """Whether a SHELX input's instruction block has an active ACTA (the endpoint asked SHELXL for a CIF).

    The SHELXL call loops rewrite the .ins from each .res and drop ACTA by default, so they check the first
    input and keep ACTA on every later call; the CIF then belongs to the call whose .res is kept.
    """

    if not Path(path).is_file():
        return False
    for line in Path(path).read_text(errors="replace").splitlines():
        head = line.split()[:1]
        if head and head[0].upper() in {"HKLF", "END"}:
            break
        if head and head[0].upper() == "ACTA":
            return True
    return False


def parse_basf_values(text: str) -> list[float]:
    """Every BASF value of a SHELX file's instruction block, in order (all BASF records, "=" continuations)."""

    values: list[float] = []
    continuing = False
    for line in text.splitlines():
        parts = line.split()
        head = parts[0].upper() if parts else ""
        if head in {"HKLF", "END"}:
            break
        if head == "BASF" or (continuing and parts):
            tokens = parts[1:] if head == "BASF" else parts
            continuing = bool(tokens) and tokens[-1] == "="
            values.extend(float(token) for token in tokens if _NUMBER_RE.fullmatch(token))
        else:
            continuing = False
    return values


def zerr_line(z: str, esds) -> str:
    """One ``ZERR Z esd(a) esd(b) esd(c) esd(alpha) esd(beta) esd(gamma)`` record."""

    values = [float(value) for value in esds]
    if len(values) != 6 or not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError(f"ZERR needs six finite nonnegative cell e.s.d.s, got {esds!r}")
    lengths = " ".join(f"{value:.4f}" for value in values[:3])
    angles = " ".join(f"{value:.3f}" for value in values[3:])
    return f"ZERR {z} {lengths} {angles}"


def write_endpoint_ins(
    src_res: Path,
    dst_ins: Path,
    hkl_name: str = "dyn_nuisance_corrected.hkl",
    keep_exti: bool = False,
    wght: tuple[float, float] | None = None,
    ls_cycles: int = 1,
    keep_damp: bool = False,
    osf: float | None = None,
    *,
    allow_basf: bool = False,
    basf: list[float] | None = None,
    merg: int | None = None,
    list_code: int | None = None,
    zerr_esds=None,
    strip_q_peaks: bool = False,
    acta: bool = False,
) -> None:
    """Write the corrected-HKL endpoint input with GROND's endpoint contract.

    DAMP is removed unless ``keep_damp`` (the legacy carry-over of a DAMP rescue into later calls and
    macro-cycles).  ``osf`` resets the first FVAR value, as the absolute-scale handoff requires.

    The keyword-only options serve the SHELXL-native outputs; their defaults leave those records unchanged.
    ``allow_basf`` keeps batch-scale BASF records (the unmerged multi-crystal endpoint)
    and ``basf`` writes the BASF records (seven values each) before FVAR; ``merg`` and ``list_code`` set MERG and LIST;
    ``zerr_esds`` replaces the six cell e.s.d.s of the ZERR record (Z is kept); ``strip_q_peaks`` drops the
    difference-map peaks Q1, Q2, ... of a previous run (SHELXL ignores them and writes fresh ones); ``acta``
    adds one ACTA instruction so SHELXL writes a CIF.
    """

    out = []
    saw_cycle_instruction = False
    saw_hklf = False
    saw_fvar = False
    saw_zerr = False
    cycle_count = max(1, int(ls_cycles))
    for line in src_res.read_text(errors="replace").splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        unsupported = _unsupported_endpoint_reason(line, allow_basf=allow_basf or basf is not None)
        if unsupported:
            raise ValueError(unsupported)
        if upper.startswith("ACTA"):
            continue
        if strip_q_peaks and Q_PEAK_RE.match(stripped):
            continue
        head = upper.split()[0] if upper.split() else ""
        if basf is not None and head == "BASF":
            continue
        if merg is not None and head == "MERG":
            continue
        if list_code is not None and head == "LIST":
            continue
        if zerr_esds is not None and head == "ZERR":
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"ZERR in {src_res} has no Z value")
            line = zerr_line(parts[1], zerr_esds)
            saw_zerr = True
        if not keep_exti and upper.startswith("EXTI"):
            continue
        if not keep_damp and upper.split()[:1] == ["DAMP"]:
            continue
        if osf is not None and not saw_fvar and upper.split()[:1] == ["FVAR"]:
            line = set_fvar_osf(line, float(osf))
            saw_fvar = True
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
    if osf is not None and not saw_fvar:
        raise ValueError(f"cannot reset the OSF: {src_res} has no FVAR instruction")
    if zerr_esds is not None and not saw_zerr:
        raise ValueError(f"cannot write cell e.s.d.s: {src_res} has no ZERR record (SHELXL needs ZERR with Z)")
    extra_instructions = []
    if merg is not None:
        extra_instructions.append(f"MERG {int(merg)}")
    if list_code is not None:
        extra_instructions.append(f"LIST {int(list_code)}")
    if acta:
        extra_instructions.append("ACTA")
    if extra_instructions:
        insert_at = next(
            (i for i, line in enumerate(out) if line.strip().upper().startswith(("L.S.", "LS ", "CGLS"))),
            None,
        )
        if insert_at is None:
            insert_at = next((i for i, line in enumerate(out) if line.strip().upper().startswith("FVAR")), len(out))
        else:
            insert_at += 1
        out[insert_at:insert_at] = extra_instructions
    if basf is not None:
        values = [float(value) for value in basf]
        if not values or not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError(f"BASF needs finite positive batch scales, got {basf!r}")
        insert_at = next((i for i, line in enumerate(out) if line.strip().upper().startswith("FVAR")), None)
        if insert_at is None:
            raise ValueError(f"cannot write BASF: {src_res} has no FVAR instruction")
        out[insert_at:insert_at] = basf_records(values)
    if wght is not None:
        out = set_active_wght_lines(out, float(wght[0]), float(wght[1]))
    dst_ins.write_text("\n".join(out) + "\n")


def write_no_exti_ins(
    src_res: Path, dst_ins: Path, hkl_name: str = "dyn_nuisance_corrected.hkl", *, keep_damp: bool = False
) -> None:
    """Write an endpoint .ins from a .res without EXTI (and without DAMP unless ``keep_damp``)."""

    write_endpoint_ins(src_res, dst_ins, hkl_name=hkl_name, keep_exti=False, keep_damp=keep_damp)
