"""Macromolecular helpers: FreeR flags, MTZ export of corrected rows and the protein endpoint.

FreeR flags are assigned once per unique (symmetry-merged) HKL, in resolution shells, so
every observation of a reflection shares its flag.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Formatter
from typing import Any

import gemmi

FREE_FLAG_LABEL = "FreeR_flag"


@dataclass(frozen=True)
class FreeFlagSummary:
    """Summary of one deterministic unique-HKL FreeR assignment."""

    space_group: str
    merge_anomalous: bool
    unique_reflections: int
    free_reflections: int
    free_fraction: float
    target_fraction: float
    min_free_count: int
    seed: int
    shells: int


@dataclass(frozen=True)
class MtzExportSummary:
    """Summary of a corrected-row MTZ export."""

    source_rows: str
    mtz: str
    space_group: str
    cell: list[float]
    row_count: int
    corrected_row_count: int
    merged_reflections: int
    free_reflections: int
    free_fraction: float
    merge_anomalous: bool
    sigma_policy: str
    q_policy: str
    q_off_d_A: float
    q_full_d_A: float


def _safe_float(value: object, default: float = float("nan")) -> float:
    """Return ``value`` as a finite float, or ``default`` when it is missing or not finite."""

    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: object) -> int:
    """Return ``value`` as an integer Miller index; raise ValueError if it is not a whole number."""

    number = _safe_float(value)
    if not math.isfinite(number) or not float(number).is_integer():
        raise ValueError(f"expected an integer Miller index, got {value!r}")
    return int(number)


def read_table(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV, TSV, or JSONL row table."""

    source = Path(path).expanduser()
    suffix = source.suffix.lower()
    rows: list[dict[str, str]] = []
    if suffix in {".jsonl", ".ndjson"}:
        for line_number, line in enumerate(source.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{source}:{line_number}: JSONL row must be an object")
            rows.append({str(k): str(v) for k, v in payload.items()})
    else:
        delimiter = "\t" if suffix in {".tsv", ".tab"} else ","
        with source.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if not reader.fieldnames:
                raise ValueError(f"{source} has no header row")
            for row in reader:
                rows.append({str(k): str(v) for k, v in row.items()})
    if not rows:
        raise ValueError(f"no rows found in {source}")
    return rows


def write_table(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write a CSV table with stable column order."""

    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return out.resolve()


def parse_cell(text: str) -> gemmi.UnitCell:
    """Parse six unit-cell constants from a comma- or space-separated string."""

    values = [float(v) for v in str(text).replace(",", " ").split()]
    if len(values) != 6:
        raise ValueError("--cell must contain six numbers: a b c alpha beta gamma")
    return gemmi.UnitCell(*values)


def infer_cell_and_spacegroup(
    *,
    cell_text: str = "",
    space_group_text: str = "",
    model: str | Path | None = None,
) -> tuple[gemmi.UnitCell, gemmi.SpaceGroup]:
    """Resolve a unit cell and space group from explicit options or a model file."""

    cell: gemmi.UnitCell | None = parse_cell(cell_text) if cell_text else None
    space_group_name = str(space_group_text or "").strip()
    if model and (cell is None or not space_group_name):
        structure = gemmi.read_structure(str(Path(model).expanduser()))
        if cell is None and structure.cell.a > 0 and structure.cell.b > 0 and structure.cell.c > 0:
            cell = structure.cell
        if not space_group_name:
            space_group_name = str(structure.spacegroup_hm or "").strip()
            if not space_group_name:
                sg = structure.find_spacegroup()
                if sg:
                    space_group_name = sg.hm
    if cell is None:
        raise ValueError("provide --cell or a model file with CRYST1/unit-cell metadata")
    if not space_group_name:
        raise ValueError("provide --space-group or a model file with space-group metadata")
    sg = gemmi.find_spacegroup_by_name(space_group_name)
    if sg is None:
        raise ValueError(f"could not parse space group: {space_group_name!r}")
    return cell, sg


def hkl_from_row(row: dict[str, Any]) -> tuple[int, int, int]:
    """Return the signed Miller index from a row-like mapping."""

    return (_safe_int(row.get("h")), _safe_int(row.get("k")), _safe_int(row.get("l")))


def canonical_hkl(hkl: tuple[int, int, int], sg: gemmi.SpaceGroup, *, merge_anomalous: bool = True) -> tuple[int, int, int]:
    """Return a deterministic unique-HKL key under symmetry and optional Friedel merging."""

    candidates: list[tuple[int, int, int]] = []
    for op in sg.operations().sym_ops:
        transformed = tuple(int(v) for v in op.apply_to_hkl(hkl))
        candidates.append(transformed)
        if merge_anomalous:
            candidates.append(tuple(-v for v in transformed))
    return min(candidates)


def _row_d(row: dict[str, Any], cell: gemmi.UnitCell | None, key: tuple[int, int, int]) -> float:
    """Return a row's resolution d in A from its own columns, else from the cell, else NaN."""

    for name in ("d_A", "d", "resolution_A", "res_A"):
        value = _safe_float(row.get(name))
        if math.isfinite(value) and value > 0:
            return value
    if cell is not None:
        try:
            return float(cell.calculate_d(key))
        except Exception:
            return float("nan")
    return float("nan")


def _unique_records(
    rows: list[dict[str, Any]],
    sg: gemmi.SpaceGroup,
    *,
    cell: gemmi.UnitCell | None = None,
    merge_anomalous: bool = True,
) -> dict[tuple[int, int, int], dict[str, Any]]:
    """Group rows by symmetry-unique HKL, with the row count and median d (A) of each."""

    records: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in rows:
        key = canonical_hkl(hkl_from_row(row), sg, merge_anomalous=merge_anomalous)
        rec = records.setdefault(key, {"h": key[0], "k": key[1], "l": key[2], "n_rows": 0, "d_values": []})
        rec["n_rows"] += 1
        d_value = _row_d(row, cell, key)
        if math.isfinite(d_value):
            rec["d_values"].append(d_value)
    for rec in records.values():
        d_values = rec.pop("d_values")
        rec["d_A"] = float(sorted(d_values)[len(d_values) // 2]) if d_values else float("nan")
    return records


def _shell_groups(records: dict[tuple[int, int, int], dict[str, Any]], shells: int) -> list[list[tuple[int, int, int]]]:
    """Split unique HKLs into ``shells`` groups of equal size ordered by 1/d^2.

    All keys form a single group when fewer than two shells are asked for or any d is
    missing.
    """

    keys = sorted(records)
    if shells <= 1 or not keys:
        return [keys]
    if not all(math.isfinite(_safe_float(records[key].get("d_A"))) and _safe_float(records[key].get("d_A")) > 0 for key in keys):
        return [keys]
    keys = sorted(keys, key=lambda key: 1.0 / max(_safe_float(records[key].get("d_A")), 1e-12) ** 2)
    n_shells = max(1, min(int(shells), len(keys)))
    return [keys[i * len(keys) // n_shells : (i + 1) * len(keys) // n_shells] for i in range(n_shells)]


def make_free_flags(
    rows: list[dict[str, Any]],
    *,
    sg: gemmi.SpaceGroup,
    cell: gemmi.UnitCell | None = None,
    fraction: float = 0.05,
    min_free_count: int = 500,
    seed: int = 20260622,
    shells: int = 10,
    merge_anomalous: bool = True,
    free_value: int = 1,
    work_value: int = 0,
) -> tuple[list[dict[str, Any]], FreeFlagSummary]:
    """Assign deterministic FreeR flags at unique-HKL level."""

    if not (0.0 < fraction < 0.5):
        raise ValueError("FreeR fraction must be between 0 and 0.5")
    records = _unique_records(rows, sg, cell=cell, merge_anomalous=merge_anomalous)
    unique_count = len(records)
    if unique_count == 0:
        raise ValueError("cannot make FreeR flags from an empty unique-HKL set")
    max_free = unique_count if unique_count == 1 else unique_count - 1
    target = int(round(fraction * unique_count))
    target = max(1, target)
    min_count = int(min_free_count)
    if min_count > 0 and min_count / max(unique_count, 1) <= 0.10:
        target = max(target, min(min_count, max_free))
    target = min(target, max_free)

    rng = random.Random(int(seed))
    selected: set[tuple[int, int, int]] = set()
    groups = _shell_groups(records, shells)
    for group in groups:
        if not group:
            continue
        shell_target = int(round(target * len(group) / max(unique_count, 1)))
        if target >= len(groups) and shell_target == 0:
            shell_target = 1
        shell_target = min(shell_target, len(group))
        selected.update(rng.sample(group, shell_target))
    all_keys = sorted(records)
    while len(selected) < target:
        candidates = [key for key in all_keys if key not in selected]
        if not candidates:
            break
        selected.add(rng.choice(candidates))
    while len(selected) > target:
        selected.remove(rng.choice(sorted(selected)))

    flag_rows: list[dict[str, Any]] = []
    for key in all_keys:
        rec = dict(records[key])
        rec["free_flag"] = int(free_value if key in selected else work_value)
        rec["free_value"] = int(free_value)
        rec["work_value"] = int(work_value)
        flag_rows.append(rec)
    summary = FreeFlagSummary(
        space_group=sg.hm,
        merge_anomalous=bool(merge_anomalous),
        unique_reflections=int(unique_count),
        free_reflections=int(len(selected)),
        free_fraction=float(len(selected) / max(unique_count, 1)),
        target_fraction=float(fraction),
        min_free_count=int(min_free_count),
        seed=int(seed),
        shells=int(shells),
    )
    return flag_rows, summary


def read_free_flags(
    path: str | Path,
    *,
    sg: gemmi.SpaceGroup,
    merge_anomalous: bool = True,
    flag_column: str = "free_flag",
) -> dict[tuple[int, int, int], int]:
    """Read a unique-HKL FreeR flag table."""

    rows = read_table(path)
    out: dict[tuple[int, int, int], int] = {}
    for row in rows:
        key = canonical_hkl(hkl_from_row(row), sg, merge_anomalous=merge_anomalous)
        out[key] = _safe_int(row.get(flag_column, row.get(FREE_FLAG_LABEL, 0)))
    if not out:
        raise ValueError(f"no FreeR flags read from {path}")
    return out


def propagate_free_flags(
    rows: list[dict[str, Any]],
    flags: dict[tuple[int, int, int], int],
    *,
    sg: gemmi.SpaceGroup,
    merge_anomalous: bool = True,
    flag_column: str = FREE_FLAG_LABEL,
    missing_value: int = 0,
) -> list[dict[str, Any]]:
    """Copy a unique-HKL FreeR assignment onto signed observation rows."""

    out: list[dict[str, Any]] = []
    for row in rows:
        key = canonical_hkl(hkl_from_row(row), sg, merge_anomalous=merge_anomalous)
        rr = dict(row)
        rr[flag_column] = int(flags.get(key, missing_value))
        out.append(rr)
    return out


def _first_finite(row: dict[str, Any], names: tuple[str, ...]) -> float:
    """Return the first finite value among the named row columns, or NaN."""

    for name in names:
        value = _safe_float(row.get(name))
        if math.isfinite(value):
            return value
    return float("nan")


def _q_ramp_weight(d_A: float, *, off_d_A: float, full_d_A: float) -> float:
    """Return the weight (0 to 1) of the dynamical q correction at resolution ``d_A``.

    The weight is 0 at or beyond ``off_d_A``, 1 at or inside ``full_d_A`` and linear in d in
    between. Invalid inputs give 1 (the full correction).
    """

    if not (math.isfinite(d_A) and d_A > 0):
        return 1.0
    if off_d_A <= 0 or full_d_A <= 0 or off_d_A <= full_d_A:
        return 1.0
    if d_A >= off_d_A:
        return 0.0
    if d_A <= full_d_A:
        return 1.0
    return float((off_d_A - d_A) / (off_d_A - full_d_A))


def corrected_row_values(
    row: dict[str, Any],
    *,
    q_policy: str = "full",
    q_off_d_A: float = 5.0,
    q_full_d_A: float = 3.0,
) -> tuple[float, float, float]:
    """Return corrected intensity, sigma, and effective q-ramp weight for one row."""

    direct_I = _first_finite(row, ("Icorr", "I_corr", "i_corr", "corrected_I", "IMEAN", "Imean"))
    direct_sig = _first_finite(row, ("SIGIcorr", "sigma_corr", "sig_corr", "corrected_sigma", "SIGIMEAN", "SIGImean"))
    if math.isfinite(direct_I) and math.isfinite(direct_sig) and direct_sig > 0:
        return direct_I, direct_sig, 1.0

    obs = _first_finite(row, ("target_I", "iobs", "Iobs", "IOBS"))
    sig = _first_finite(row, ("sigma_eff", "sigma", "sigma_Iobs", "SIGMA"))
    if not (math.isfinite(obs) and math.isfinite(sig) and sig > 0):
        raise ValueError("row lacks corrected intensity/sigma columns and cannot be reconstructed")

    q = _first_finite(row, ("q_dyn_over_born", "dyn_over_born"))
    if not math.isfinite(q):
        dyn_raw = _first_finite(row, ("dyn_raw",))
        born_raw = _first_finite(row, ("born_raw",))
        q = dyn_raw / max(born_raw, 1e-30) if math.isfinite(dyn_raw) and math.isfinite(born_raw) else 1.0
    q = min(max(q, 0.05), 20.0)

    scale = _first_finite(row, ("dyn_group_scale", "scale", "row_scale"))
    if not math.isfinite(scale):
        dyn_raw = _first_finite(row, ("dyn_raw",))
        dyn_scaled = _first_finite(row, ("dyn_scaled",))
        scale = dyn_scaled / max(dyn_raw, 1e-30) if math.isfinite(dyn_scaled) and math.isfinite(dyn_raw) else 1.0
    scale = max(scale, 1e-30)

    ramp = 1.0
    if q_policy == "low_resolution_damped":
        d_A = _first_finite(row, ("d_A", "d", "resolution_A", "res_A"))
        ramp = _q_ramp_weight(d_A, off_d_A=q_off_d_A, full_d_A=q_full_d_A)
    elif q_policy == "off":
        ramp = 0.0
    elif q_policy != "full":
        raise ValueError("q_policy must be 'full', 'low_resolution_damped', or 'off'")
    q_eff = math.exp(ramp * math.log(q))
    denom = max(scale * q_eff, 1e-30)
    return float(obs / denom), float(sig / denom), float(ramp)


def merge_corrected_rows(
    rows: list[dict[str, Any]],
    flags: dict[tuple[int, int, int], int],
    *,
    sg: gemmi.SpaceGroup,
    merge_anomalous: bool = True,
    sigma_policy: str = "formal",
    q_policy: str = "full",
    q_off_d_A: float = 5.0,
    q_full_d_A: float = 3.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge corrected signed rows into unique reflections with FreeR flags."""

    bins: dict[tuple[int, int, int], dict[str, Any]] = {}
    corrected_count = 0
    ramp_values: list[float] = []
    for row in rows:
        try:
            intensity, sigma, ramp = corrected_row_values(
                row,
                q_policy=q_policy,
                q_off_d_A=q_off_d_A,
                q_full_d_A=q_full_d_A,
            )
        except ValueError:
            continue
        if not (math.isfinite(intensity) and math.isfinite(sigma) and sigma > 0):
            continue
        key = canonical_hkl(hkl_from_row(row), sg, merge_anomalous=merge_anomalous)
        weight = 1.0 / max(sigma * sigma, 1e-30)
        bucket = bins.setdefault(
            key,
            {"w": 0.0, "wI": 0.0, "values": [], "weights": [], "n": 0, "flag": int(flags.get(key, 0))},
        )
        bucket["w"] += weight
        bucket["wI"] += weight * intensity
        bucket["values"].append(intensity)
        bucket["weights"].append(weight)
        bucket["n"] += 1
        corrected_count += 1
        ramp_values.append(ramp)

    merged: list[dict[str, Any]] = []
    inflations: list[float] = []
    for key in sorted(bins):
        bucket = bins[key]
        mean = bucket["wI"] / max(bucket["w"], 1e-30)
        sigma = math.sqrt(1.0 / max(bucket["w"], 1e-30))
        inflation = 1.0
        if sigma_policy not in {"formal", "chisq_overdispersion"}:
            raise ValueError("sigma_policy must be 'formal' or 'chisq_overdispersion'")
        if sigma_policy == "chisq_overdispersion" and bucket["n"] > 1:
            chi2 = sum(w * (v - mean) ** 2 for v, w in zip(bucket["values"], bucket["weights"], strict=True)) / max(
                bucket["n"] - 1, 1
            )
            if math.isfinite(chi2) and chi2 > 1.0:
                inflation = math.sqrt(chi2)
                sigma *= inflation
        inflations.append(inflation)
        merged.append(
            {
                "h": int(key[0]),
                "k": int(key[1]),
                "l": int(key[2]),
                "IMEAN": float(mean),
                "SIGIMEAN": float(sigma),
                "NOBS": int(bucket["n"]),
                FREE_FLAG_LABEL: int(bucket["flag"]),
            }
        )
    meta = {
        "input_rows": int(len(rows)),
        "corrected_rows": int(corrected_count),
        "merged_reflections": int(len(merged)),
        "free_reflections": int(sum(1 for row in merged if int(row[FREE_FLAG_LABEL]) != 0)),
        "sigma_policy": sigma_policy,
        "inflation_median": float(sorted(inflations)[len(inflations) // 2]) if inflations else float("nan"),
        "q_policy": q_policy,
        "q_ramp_median": float(sorted(ramp_values)[len(ramp_values) // 2]) if ramp_values else float("nan"),
    }
    return merged, meta


def write_mtz(
    reflections: list[dict[str, Any]],
    *,
    out_mtz: str | Path,
    cell: gemmi.UnitCell,
    sg: gemmi.SpaceGroup,
    dataset: str = "GROND",
) -> Path:
    """Write a macromolecular MTZ with IMEAN/SIGIMEAN/FreeR_flag/NOBS."""

    if not reflections:
        raise ValueError("cannot write an MTZ with no reflections")
    mtz = gemmi.Mtz(with_base=True)
    mtz.title = "GROND corrected intensities"
    mtz.spacegroup = sg
    mtz.set_cell_for_all(cell)
    mtz.add_dataset(dataset)
    mtz.add_column("IMEAN", "J")
    mtz.add_column("SIGIMEAN", "Q")
    mtz.add_column(FREE_FLAG_LABEL, "I")
    mtz.add_column("NOBS", "I")
    import numpy as np

    array = np.asarray(
        [
            [
                int(row["h"]),
                int(row["k"]),
                int(row["l"]),
                float(row["IMEAN"]),
                float(row["SIGIMEAN"]),
                int(row[FREE_FLAG_LABEL]),
                int(row["NOBS"]),
            ]
            for row in reflections
        ],
        dtype=np.float32,
    )
    mtz.set_data(array)
    mtz.switch_to_asu_hkl()
    mtz.sort()
    mtz.update_reso()
    mtz.history.append("GROND corrected row intensities merged to unique-HKL endpoint with fixed FreeR flags.")
    out = Path(out_mtz).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    mtz.write_to_file(str(out))
    return out.resolve()


def export_corrected_mtz(
    *,
    row_table: str | Path,
    free_flags: str | Path,
    out_mtz: str | Path,
    cell: gemmi.UnitCell,
    sg: gemmi.SpaceGroup,
    out_dir: str | Path | None = None,
    merge_anomalous: bool = True,
    sigma_policy: str = "formal",
    q_policy: str = "full",
    q_off_d_A: float = 5.0,
    q_full_d_A: float = 3.0,
    dataset: str = "GROND",
) -> MtzExportSummary:
    """Convert corrected signed rows into a macromolecular MTZ."""

    rows = read_table(row_table)
    flags = read_free_flags(free_flags, sg=sg, merge_anomalous=merge_anomalous)
    merged, merge_meta = merge_corrected_rows(
        rows,
        flags,
        sg=sg,
        merge_anomalous=merge_anomalous,
        sigma_policy=sigma_policy,
        q_policy=q_policy,
        q_off_d_A=q_off_d_A,
        q_full_d_A=q_full_d_A,
    )
    mtz_path = write_mtz(merged, out_mtz=out_mtz, cell=cell, sg=sg, dataset=dataset)
    target_dir = Path(out_dir).expanduser() if out_dir else mtz_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    merged_csv = write_table(target_dir / "merged_corrected_rows.csv", merged)
    summary = MtzExportSummary(
        source_rows=str(Path(row_table).expanduser().resolve()),
        mtz=str(mtz_path),
        space_group=sg.hm,
        cell=[cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma],
        row_count=int(len(rows)),
        corrected_row_count=int(merge_meta["corrected_rows"]),
        merged_reflections=int(len(merged)),
        free_reflections=int(merge_meta["free_reflections"]),
        free_fraction=float(merge_meta["free_reflections"] / max(len(merged), 1)),
        merge_anomalous=bool(merge_anomalous),
        sigma_policy=sigma_policy,
        q_policy=q_policy,
        q_off_d_A=float(q_off_d_A),
        q_full_d_A=float(q_full_d_A),
    )
    payload = {**asdict(summary), "free_flags": str(Path(free_flags).expanduser().resolve()), "merged_rows": str(merged_csv)}
    (target_dir / "export_mtz_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return summary


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_protein_endpoint_command(
    template: str,
    *,
    model: str | Path,
    mtz: str | Path,
    out_dir: str | Path,
    cycles: int,
    free_label: str = FREE_FLAG_LABEL,
    extra: dict[str, Any] | None = None,
) -> list[str]:
    """Render a macromolecular one-cycle endpoint command template."""

    values: dict[str, Any] = {
        "model": str(model),
        "mtz": str(mtz),
        "reflections": str(mtz),
        "out_dir": str(out_dir),
        "cycles": int(cycles),
        "free_label": free_label,
    }
    values.update(extra or {})
    needed = {field for _, field, _, _ in Formatter().parse(template) if field}
    missing = sorted(field for field in needed if field not in values)
    if missing:
        raise ValueError("protein endpoint command uses unknown placeholder(s): " + ", ".join(missing))
    return shlex.split(template.format(**values))


def run_protein_endpoint(
    *,
    command: list[str],
    out_dir: str | Path,
    output_model: str | Path,
    model: str | Path,
    mtz: str | Path,
    cycles: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run or dry-run one external macromolecular endpoint macrocycle."""

    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    payload = {
        "command": command,
        "out_dir": str(out.resolve()),
        "model": str(Path(model).expanduser().resolve()),
        "mtz": str(Path(mtz).expanduser().resolve()),
        "cycles": int(cycles),
        "dry_run": bool(dry_run),
        "output_model": str(Path(output_model).expanduser()),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "command.json").write_text(json.dumps(payload, indent=2) + "\n")
    if dry_run:
        payload.update({"returncode": 0, "status": "dry_run", "wall_seconds": 0.0})
        (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
        return payload

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    result = subprocess.run(command, cwd=out, text=True, capture_output=True, env=env)
    (out / "stdout.log").write_text(result.stdout)
    (out / "stderr.log").write_text(result.stderr)
    output_path = Path(output_model)
    if not output_path.is_absolute():
        output_path = out / output_path
    status = "ok" if result.returncode == 0 and output_path.exists() else "failed"
    payload.update(
        {
            "returncode": int(result.returncode),
            "status": status,
            "wall_seconds": float(time.time() - start),
            "output_model_resolved": str(output_path.resolve()) if output_path.exists() else str(output_path),
            "stdout": str((out / "stdout.log").resolve()),
            "stderr": str((out / "stderr.log").resolve()),
        }
    )
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    if status != "ok":
        raise RuntimeError(f"protein endpoint failed; see {out / 'summary.json'}")
    return payload
