"""Cell e.s.d.s for the ZERR record of GROND's SHELXL endpoints.

SHELXL propagates the ZERR e.s.d.s into every bond, angle and torsion e.s.d., so an all-zero ZERR tells
Olex2/SHELXLE users that the cell is exact.  AUTO therefore writes measured e.s.d.s, in this order:

1. ``--cell-esd`` given by the user (model setting);
2. two or more crystals: the sample standard deviation (ddof 1) of their cells in the model setting,
   a conservative measure that includes crystal-to-crystal calibration scatter;
3. one crystal: the e.s.d.s XDS printed in CORRECT.LP, or for a GROND DIALS bridge the standard deviation
   of the scan-varying cell (``DIALS_A_AT_SCAN_POINTS``).

Parameters fixed by the model's lattice symmetry (for example the 90 degree angles of an orthorhombic cell)
get exactly 0, as SHELXL expects; parameters tied by symmetry (a = b in a tetragonal cell) share their
mean e.s.d.  Without a measured source (one crystal with neither), or when a crystal's cell cannot be read, the
starting model's own ZERR e.s.d.s are used when they are non-zero, and otherwise the model's ZERR is kept unchanged;
both print a warning and are recorded.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from grond.bloch.model import parse_model
from grond.bloch.numerics import parse_hkl_transform

CELL_KEYS = ("a", "b", "c", "alpha", "beta", "gamma")
# Print precision of the ZERR record: a measured spread below it is written as one unit of the last digit.
ZERR_FLOOR = (1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3)
# A crystal whose model-setting cell differs from the model cell by more than this is not the same setting.
MAX_CELL_LENGTH_MISMATCH = 0.05
MAX_CELL_ANGLE_MISMATCH_DEG = 3.0


def cell_from_real_matrix(real: np.ndarray) -> list[float]:
    """a, b, c, alpha, beta, gamma of the direct axes in the columns of ``real``."""

    axes = [np.asarray(real[:, i], dtype=float) for i in range(3)]
    lengths = [float(np.linalg.norm(axis)) for axis in axes]

    def angle(i: int, j: int) -> float:
        """Return the angle in degrees between direct axes ``i`` and ``j``."""

        cosine = float(np.dot(axes[i], axes[j]) / (lengths[i] * lengths[j]))
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    return [*lengths, angle(1, 2), angle(0, 2), angle(0, 1)]


def _header(path: Path) -> dict[str, str]:
    """Return the ``!KEY=value`` entries of an INTEGRATE.HKL header."""

    header: dict[str, str] = {}
    with path.open(errors="replace") as handle:
        for line in handle:
            if not line.startswith("!"):
                break
            if line.startswith("!END_OF_HEADER"):
                break
            body = line[1:].rstrip("\n")
            if "=" in body:
                key, value = body.split("=", 1)
                header[key.strip()] = value.strip()
    return header


def _vector(text: str) -> np.ndarray:
    """Parse the first three numbers of a header value as a vector."""

    values = [float(value) for value in str(text).split()[:3]]
    if len(values) != 3:
        raise ValueError(f"expected three numbers, got {text!r}")
    return np.asarray(values, dtype=float)


def dataset_model_cell(hkl: str | Path, hkl_transform: str | None) -> dict[str, object]:
    """Cell of one XDS/DIALS-bridge row file in the model setting, and its scan-varying spread if recorded."""

    path = Path(hkl).expanduser()
    header = _header(path)
    try:
        real = np.stack(
            [_vector(header[key]) for key in ("UNIT_CELL_A-AXIS", "UNIT_CELL_B-AXIS", "UNIT_CELL_C-AXIS")], axis=1
        )
    except KeyError as exc:
        raise ValueError(f"{path} has no UNIT_CELL_[ABC]-AXIS header; cannot place its cell in the model setting") from exc
    transform, canonical = parse_hkl_transform(hkl_transform)
    inv_t = np.asarray(transform, dtype=float).T
    model_real = real @ inv_t
    record: dict[str, object] = {
        "hkl": str(path),
        "hkl_transform": canonical,
        "cell": cell_from_real_matrix(model_real),
    }
    scan_file = header.get("DIALS_A_AT_SCAN_POINTS_FILE", "").split()
    if scan_file:
        scan_path = Path(scan_file[0])
        if not scan_path.is_absolute():
            scan_path = path.parent / scan_path
        cells = []
        for line in scan_path.read_text().splitlines():
            values = [float(value) for value in line.split()]
            if len(values) != 9:
                continue
            a_matrix = np.asarray(values, dtype=float).reshape(3, 3) @ inv_t
            cells.append(cell_from_real_matrix(np.linalg.inv(a_matrix).T))
        if len(cells) >= 2:
            record["scan_varying_cell_sd"] = [float(v) for v in np.std(np.asarray(cells), axis=0, ddof=1)]
            record["scan_points"] = len(cells)
            record["scan_points_file"] = str(scan_path)
    lp = _correct_lp_esds(path)
    if lp is not None:
        record["xds_correct_lp"] = lp["path"]
        raw_cell = cell_from_real_matrix(real)
        lp_cell = lp.get("cell")
        if lp_cell is not None and not _same_cell(lp_cell, raw_cell):
            # CORRECT reindexed (another setting than INTEGRATE's axes): its e.s.d.s cannot be put on these axes.
            record["xds_correct_lp_setting_mismatch"] = {"correct_lp_cell": lp_cell, "row_file_cell": raw_cell}
        else:
            record["xds_correct_lp_esd"] = _permute_cell_esds(lp["esds"], transform)
    return record


def _same_cell(first: Sequence[float], second: Sequence[float]) -> bool:
    """Whether two cells agree axis by axis within the setting tolerances used for the model cell."""

    lengths = all(abs(float(a) / float(b) - 1.0) <= MAX_CELL_LENGTH_MISMATCH for a, b in zip(first[:3], second[:3], strict=True))
    angles = all(abs(float(a) - float(b)) <= MAX_CELL_ANGLE_MISMATCH_DEG for a, b in zip(first[3:], second[3:], strict=True))
    return lengths and angles


def _correct_lp_esds(hkl_path: Path) -> dict[str, object] | None:
    """The last "E.S.D. OF CELL PARAMETERS" line of an XDS CORRECT.LP next to the row file."""

    lp = hkl_path.parent / "CORRECT.LP"
    if not lp.is_file():
        return None
    text = lp.read_text(errors="replace")
    matches = re.findall(r"E\.S\.D\. OF CELL PARAMETERS\s+([^\n]+)", text)
    if not matches:
        return None
    values = [float(value) for value in matches[-1].split()[:6]]
    if len(values) != 6 or not all(math.isfinite(value) and value >= 0.0 for value in values):
        return None
    out: dict[str, object] = {"path": str(lp), "esds": values}
    # The refined cell printed with the e.s.d.s names their setting (CORRECT may reindex INTEGRATE's cell).
    cells = re.findall(r"UNIT CELL PARAMETERS\s+([^\n]+)", text)
    if cells:
        cell = [float(value) for value in cells[-1].split()[:6]]
        if len(cell) == 6 and all(math.isfinite(value) and value > 0.0 for value in cell):
            out["cell"] = cell
    return out


def _permute_cell_esds(esds: Sequence[float], transform: np.ndarray) -> list[float]:
    """E.s.d.s of a signed-permutation re-indexing: lengths and the angles between the mapped axes."""

    t = np.asarray(transform)
    source_axis = [int(np.flatnonzero(t[i])[0]) for i in range(3)]
    lengths = [float(esds[source_axis[i]]) for i in range(3)]
    pairs = {frozenset((1, 2)): 3, frozenset((0, 2)): 4, frozenset((0, 1)): 5}
    angles = []
    for i, j in ((1, 2), (0, 2), (0, 1)):
        angles.append(float(esds[pairs[frozenset((source_axis[i], source_axis[j]))]]))
    return lengths + angles


def _metric(cell: Sequence[float]) -> np.ndarray:
    """Return the metric tensor G (A^2) of a cell given as a, b, c, alpha, beta, gamma."""

    a, b, c, alpha, beta, gamma = (float(value) for value in cell)
    ca, cb, cg = (math.cos(math.radians(value)) for value in (alpha, beta, gamma))
    return np.asarray(
        [[a * a, a * b * cg, a * c * cb], [a * b * cg, b * b, b * c * ca], [a * c * cb, b * c * ca, c * c]],
        dtype=float,
    )


def _cell_from_metric(g: np.ndarray) -> list[float]:
    """Return a, b, c, alpha, beta, gamma from a metric tensor."""

    lengths = [math.sqrt(float(g[i, i])) for i in range(3)]

    def angle(i: int, j: int) -> float:
        """Return the angle in degrees between axes ``i`` and ``j`` of the metric."""

        return math.degrees(math.acos(max(-1.0, min(1.0, float(g[i, j]) / (lengths[i] * lengths[j])))))

    return [*lengths, angle(1, 2), angle(0, 2), angle(0, 1)]


def symmetry_constraints(model: Mapping[str, object]) -> dict[str, object]:
    """Which cell parameters the model's lattice symmetry fixes or ties.

    The metric G must satisfy R^T G R = G for every rotation R of the space group.  Random metrics in that
    invariant subspace (around the model's own metric) show which angles are constant (fixed) and which
    length ratios are constant (tied).  This works for every space group and setting.
    """

    rotations = [np.asarray(op["rot"], dtype=float) for op in model.get("symmetry_ops", []) or []]
    if not rotations:
        rotations = [np.eye(3)]
    basis_matrices = []
    for i in range(3):
        for j in range(i, 3):
            e = np.zeros((3, 3))
            e[i, j] = e[j, i] = 1.0
            basis_matrices.append(e)
    # One column per symmetric basis matrix: the stacked residuals R^T E R - E over every rotation.
    constraint = np.stack(
        [np.concatenate([(rot.T @ e @ rot - e).ravel() for rot in rotations]) for e in basis_matrices], axis=1
    )
    _u, singular, vt = np.linalg.svd(constraint)
    tolerance = 1e-9 * max(1.0, float(singular[0]) if singular.size else 1.0)
    rank = int(np.sum(singular > tolerance))
    null = vt[rank:].T  # 6 x d: the invariant metrics
    g0 = _metric(model["cell"])
    rng = np.random.default_rng(20260926)
    samples = []
    scale = float(np.trace(g0)) * 1e-3
    for _ in range(12):
        vector = null @ rng.normal(size=null.shape[1])
        g = g0 + scale * sum(float(v) * e for v, e in zip(vector, basis_matrices, strict=True))
        if np.all(np.linalg.eigvalsh(g) > 0.0):
            samples.append(_cell_from_metric(g))
    samples_arr = np.asarray(samples) if samples else np.asarray([_cell_from_metric(g0)])
    spread = np.std(samples_arr, axis=0)
    fixed_angles = [bool(spread[3 + i] < 1e-6) for i in range(3)]
    ties: list[list[int]] = []
    for i in range(3):
        for j in range(i + 1, 3):
            ratio = samples_arr[:, i] / samples_arr[:, j]
            if float(np.std(ratio)) < 1e-9:
                group = next((grp for grp in ties if i in grp or j in grp), None)
                if group is None:
                    ties.append([i, j])
                else:
                    group.extend(index for index in (i, j) if index not in group)
    angle_ties: list[list[int]] = []
    for i in range(3):
        for j in range(i + 1, 3):
            if fixed_angles[i] or fixed_angles[j]:
                continue
            if float(np.std(samples_arr[:, 3 + i] - samples_arr[:, 3 + j])) < 1e-9:
                group = next((grp for grp in angle_ties if i in grp or j in grp), None)
                if group is None:
                    angle_ties.append([i, j])
                else:
                    group.extend(index for index in (i, j) if index not in group)
    return {
        "invariant_metric_dimension": int(null.shape[1]),
        "fixed_angles": fixed_angles,
        "tied_lengths": ties,
        "tied_angles": angle_ties,
    }


def apply_symmetry(esds: Sequence[float], constraints: Mapping[str, object]) -> list[float]:
    """Zero the e.s.d.s of symmetry-fixed angles and give tied lengths their mean e.s.d."""

    out = [float(value) for value in esds]
    for index, fixed in enumerate(constraints.get("fixed_angles") or []):
        if fixed:
            out[3 + index] = 0.0
    for group in constraints.get("tied_lengths") or []:
        mean = float(np.mean([out[i] for i in group]))
        for i in group:
            out[i] = mean
    for group in constraints.get("tied_angles") or []:
        mean = float(np.mean([out[3 + i] for i in group]))
        for i in group:
            out[3 + i] = mean
    return out


def _informative(esds: Sequence[float], constraints: Mapping[str, object]) -> bool:
    """A scan-varying spread carries information only when a free cell length varies above print precision.

    DIALS refinements with a scan-varying orientation but a fixed cell give a spread of ~1e-12 A, which is
    not an e.s.d.; such a crystal has no e.s.d. source.
    """

    del constraints
    return any(float(esds[i]) >= ZERR_FLOOR[i] for i in range(3))


def _floor_free(esds: Sequence[float], constraints: Mapping[str, object]) -> tuple[list[float], list[str]]:
    """Apply the ZERR print floor to the e.s.d.s of free cell parameters.

    Angles fixed by symmetry get exactly 0. Returns the e.s.d.s and the names of the
    parameters that were raised to the floor.
    """

    out, floored = [], []
    fixed = [False, False, False, *list(constraints.get("fixed_angles") or [False, False, False])]
    for index, value in enumerate(esds):
        if fixed[index]:
            out.append(0.0)
            continue
        if not math.isfinite(float(value)):
            raise ValueError(f"cell e.s.d. of {CELL_KEYS[index]} is not finite")
        if float(value) < ZERR_FLOOR[index]:
            floored.append(CELL_KEYS[index])
            out.append(ZERR_FLOOR[index])
        else:
            out.append(float(value))
    return out, floored


def model_zerr_esds(structure: str | Path) -> list[float] | None:
    """The six e.s.d.s of the model's ``ZERR Z sa sb sc sal sbe sga`` record, or None when there is none."""

    for line in Path(structure).expanduser().read_text(errors="replace").splitlines():
        parts = line.split()
        if parts and parts[0].upper() == "ZERR" and len(parts) >= 8:
            try:
                values = [float(value) for value in parts[2:8]]
            except ValueError:
                return None
            return values if all(math.isfinite(value) and value >= 0.0 for value in values) else None
    return None


def _model_zerr_fallback(structure, constraints, record: dict, reason: str) -> tuple[list[float] | None, str]:
    """(raw e.s.d.s, source) from the model's ZERR, or (None, reason) when the model's ZERR carries no e.s.d."""

    model_esds = model_zerr_esds(structure)
    record["fallback"] = {"reason": reason, "model_zerr_esds": model_esds}
    if model_esds is not None and _informative(apply_symmetry(model_esds, constraints), constraints):
        return model_esds, f"the starting model's ZERR ({reason})"
    return None, reason


def _kept_model_zerr(record: dict, reason: str) -> dict:
    """No measured source and no e.s.d. in the model's ZERR: the model's ZERR is written unchanged."""

    record.update(
        {
            "status": "model_zerr_kept",
            "source": f"the starting model's ZERR, unchanged ({reason}; its e.s.d.s are zero or absent)",
            "esds": (record.get("fallback") or {}).get("model_zerr_esds") or [],
            "zerr_esds_arg": "",
            "definition": "no measured cell e.s.d.: SHELXL bond e.s.d.s will not include the cell uncertainty",
        }
    )
    return record


def resolve_cell_esds(
    structure: str | Path,
    datasets: Sequence[Mapping[str, str]],
    *,
    explicit: str = "",
) -> dict[str, object]:
    """Resolve the ZERR e.s.d.s of a run from the sources in the module docstring (fails closed)."""

    model = parse_model(Path(structure))
    model_cell = [float(value) for value in model["cell"]]
    constraints = symmetry_constraints(model)
    record: dict[str, object] = {
        "schema": "grond.cell-esd.v1",
        "model_cell": model_cell,
        "symmetry": constraints,
        "datasets": {},
    }
    if str(explicit or "").strip():
        values = [float(value) for value in str(explicit).split(",")]
        if len(values) != 6 or not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError(f"--cell-esd needs six finite nonnegative values a,b,c,alpha,beta,gamma, got {explicit!r}")
        raw, source = values, "explicit --cell-esd"
    else:
        cells = {}
        unreadable = []
        for ds in datasets:
            try:
                info = dataset_model_cell(ds["hkl"], ds.get("hkl_transform"))
            except (OSError, ValueError) as exc:
                unreadable.append(f"{ds['name']}: {exc}")
                continue
            cell = info["cell"]
            for index in range(3):
                if abs(cell[index] / model_cell[index] - 1.0) > MAX_CELL_LENGTH_MISMATCH:
                    raise ValueError(
                        f"dataset {ds['name']}: its cell {cell} in the model setting does not match the model cell "
                        f"{model_cell}; the HKL transform is not the model setting"
                    )
            for index in range(3, 6):
                if abs(cell[index] - model_cell[index]) > MAX_CELL_ANGLE_MISMATCH_DEG:
                    raise ValueError(
                        f"dataset {ds['name']}: its cell {cell} in the model setting does not match the model cell {model_cell}"
                    )
            cells[str(ds["name"])] = info
        record["datasets"] = cells
        if unreadable:
            raw, source = _model_zerr_fallback(
                structure, constraints, record, "the cell of " + "; ".join(unreadable) + " could not be read"
            )
            if raw is None:
                return _kept_model_zerr(record, source)
        elif len(cells) >= 2:
            raw = [float(v) for v in np.std(np.asarray([info["cell"] for info in cells.values()]), axis=0, ddof=1)]
            source = f"sample standard deviation of {len(cells)} crystal cells (model setting)"
        else:
            info = next(iter(cells.values()), {})
            if info.get("xds_correct_lp_esd") is not None:
                raw, source = list(info["xds_correct_lp_esd"]), f"XDS CORRECT.LP ({info.get('xds_correct_lp')})"
            elif info.get("scan_varying_cell_sd") is not None and _informative(
                apply_symmetry(info["scan_varying_cell_sd"], constraints), constraints
            ):
                raw = list(info["scan_varying_cell_sd"])
                source = f"standard deviation of the DIALS scan-varying cell ({info.get('scan_points')} scan points)"
            else:
                mismatch = info.get("xds_correct_lp_setting_mismatch")
                detail = (
                    f"; its CORRECT.LP cell {mismatch['correct_lp_cell']} is in another setting than the row file's "
                    f"axes {[round(v, 4) for v in mismatch['row_file_cell']]}"
                    if mismatch
                    else ""
                )
                raw, source = _model_zerr_fallback(
                    structure,
                    constraints,
                    record,
                    f"one crystal without an XDS CORRECT.LP or a DIALS scan-varying cell that varies{detail}",
                )
                if raw is None:
                    return _kept_model_zerr(record, source)
    symmetric = apply_symmetry(raw, constraints)
    esds, floored = _floor_free(symmetric, constraints)
    record.update(
        {
            "status": "ok_model_zerr_fallback" if record.get("fallback") else "ok",
            "source": source,
            "raw_esds": [float(v) for v in raw],
            "esds": esds,
            "floored_to_print_precision": floored,
            "zerr_esds_arg": ",".join(f"{value:.6g}" for value in esds),
            "definition": "ZERR e.s.d.s; symmetry-fixed parameters 0, tied lengths share their mean e.s.d.",
        }
    )
    return record
