"""Monitoring for the Bloch -> corrected-HKL -> SHELXL macro-cycle loop (a lagged fixed-point map).

Why the loop is not a fixed point of one objective
--------------------------------------------------
Each macro-cycle k predicts rows with the current model p_k, writes kinematically equivalent
intensities Icorr_i = y_i * K_i|F_i|^2 / pred_i(p_k) and lets SHELXL solve the kinematic least squares
against them.  The next model is therefore p_{k+1} = T(p_k), a lagged map whose stationary point p*
satisfies sum_i w_i (Icorr_i - K|F_i(p*)|^2) d|F_i|^2/dp = 0 with the dynamical ratio frozen at p*.
The dynamical least-squares condition is sum_i (r_i / sigma_i^2) d pred_i/dp = 0 with
d ln pred = d ln |F|^2 + d ln q, so at p* the true gradient is -2 sum_i (r_i/sigma_i^2) pred_i d ln q_i/dp,
which vanishes only if the residuals are orthogonal to the dynamical sensitivity d ln q/dp.  The map
converges only while the spectral radius of its Jacobian -(J_B' W J_B)^-1 J_B' W J_q stays below 1.

Single-step settings (one L.S. step per macro-cycle, no convergence) add moving parts on top of that bias.
Such a loop can contract towards an unphysical point, for example an atom whose smallest ADP eigenvalue
turns negative (NPD) while the row R of the entering model rises every cycle.  The same drift can appear
with undamped refinements converged to shift/esd < 0.01, so convergence inside a cycle does not make the
outer map exact.

What the loop monitors
----------------------
Every macro-cycle records the held-out (split == free) and work row amplitude R of the model that
entered it, the SHELXL statistics of the converged inner refinement, the maximum non-H coordinate shift
between successive models, the NPD/ISOR/DAMP state and the number of rows excluded for a pathological
correction ratio.  The loop stops when the coordinate shift falls below a tolerance (converged) or at the
cycle cap; an ADP pathology still stops it.  The kept model is the last adopted cycle
(--shelxl-feedback-keep last), so free rows never choose the model.
"""

from __future__ import annotations

import contextlib
import csv
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from grond.refine.shelxl_iterations import shelxl_model_restraint_audit
from grond.validation.heldout import r_metrics

KEEP_POLICIES = ("last",)
# How a loop that no unsafe model stopped ended, recorded as ``stop_reason``.
LOOP_STOP_REASONS = ("cap", "converged")
# Summary ``status`` values of such a loop (see completed_loop_status).
COMPLETED_LOOP_STATUSES = ("ok", *LOOP_STOP_REASONS)


def completed_loop_status(stop_reason: str, *, xyz_tol: float) -> str:
    """Summary ``status`` of a macro-cycle loop that ran without an unsafe stop.

    Under the legacy fixed-count rule (xyz tolerance 0) it is ``ok``; the monitored loop reports its stop
    reason (``cap`` or ``converged``).  The stop reason is always recorded separately as ``stop_reason``.
    """

    if stop_reason not in LOOP_STOP_REASONS:
        raise ValueError(f"unknown macro-cycle stop reason {stop_reason!r}; use one of {', '.join(LOOP_STOP_REASONS)}")
    return "ok" if float(xyz_tol) <= 0.0 else stop_reason

# Lines that are instructions, not atoms, inside the FVAR..HKLF block.
_INSTRUCTIONS = {
    "L.S.", "CGLS", "WGHT", "EXTI", "DAMP", "ACTA", "CONF", "FMAP", "PLAN", "LIST", "MORE", "BOND", "REM", "SHEL",
    "OMIT", "MERG", "HKLF", "END", "TWIN", "BASF", "SWAT", "HTAB", "TEMP", "SIZE", "ISOR", "RIGU", "SIMU", "DELU",
    "DFIX", "DANG", "SADI", "FLAT", "CHIV", "EQIV", "STIR", "WPDB", "BLOC", "FREE", "SPEC", "ANIS", "EADP", "EXYZ",
    "HFIX", "SUMP", "PART", "NCSY", "SAME",
}


def _finite_or_none(value) -> float | None:
    """Return ``value`` as a finite float, or None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _model_block(text: str) -> list[str]:
    """Lines from FVAR up to HKLF/END without REM/WGHT/EXTI/DAMP/L.S./CGLS (the model part of a RES file)."""

    model: list[str] = []
    in_model = False
    for raw in text.splitlines():
        token = raw.split()[0].upper() if raw.split() else ""
        if token in ("HKLF", "END"):
            break
        if token == "FVAR":
            in_model = True
        if not in_model or token in ("REM", "WGHT", "EXTI", "DAMP", "L.S.", "CGLS"):
            continue
        model.append(raw)
    return model


def atom_xyz(res_path: Path) -> dict[str, np.ndarray]:
    """Non-H atom fractional coordinates (key -> xyz mod 1) of a SHELX model.

    Keys are the atom names for atoms outside RESI/PART blocks.  Inside a residue or a
    disorder part, or for a repeated name, the key also carries the RESI and PART context and the
    occurrence, so a model with residues (N, CA, C, O in every residue) is compared atom by atom
    instead of the last residue overwriting all the others.
    """

    out: dict[str, np.ndarray] = {}
    residue = ""
    part = "0"
    occurrences: dict[str, int] = {}
    for line in _model_block(Path(res_path).read_text(errors="replace")):
        parts = line.split()
        name = parts[0].upper() if parts else ""
        if name == "RESI":
            residue = "" if parts[1:] in ([], ["0"]) else " ".join(parts[1:]).upper()
            continue
        if name == "PART":
            part = parts[1] if len(parts) > 1 else "0"
            continue
        if (
            len(parts) >= 5
            and parts[0][0].isalpha()
            and name not in _INSTRUCTIONS
            and name not in ("FVAR", "AFIX")
            and not name.startswith("H")
        ):
            with contextlib.suppress(ValueError):
                xyz = np.asarray([float(parts[2]), float(parts[3]), float(parts[4])]) % 1.0
                key = name if not residue and part == "0" else f"{name}|RESI {residue}|PART {part}"
                seen = occurrences.get(key, 0)
                occurrences[key] = seen + 1
                out[key if seen == 0 else f"{key}#{seen}"] = xyz
    return out


def cell_from_res(res_path: Path) -> list[float]:
    """a, b, c, alpha, beta, gamma from the CELL instruction."""

    for line in Path(res_path).read_text(errors="replace").splitlines():
        parts = line.split()
        if parts and parts[0].upper() == "CELL" and len(parts) >= 8:
            return [float(value) for value in parts[2:8]]
    raise ValueError(f"{res_path} has no CELL instruction")


def orthogonalization_matrix(cell) -> np.ndarray:
    """Fractional -> Cartesian matrix (a along x, b in the xy plane)."""

    a, b, c = (float(v) for v in cell[:3])
    alpha, beta, gamma = (math.radians(float(v)) for v in cell[3:6])
    cos_a, cos_b, cos_g = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sin_g = math.sin(gamma)
    volume_term = math.sqrt(max(1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2.0 * cos_a * cos_b * cos_g, 0.0))
    return np.asarray(
        [
            [a, b * cos_g, c * cos_b],
            [0.0, b * sin_g, c * (cos_a - cos_b * cos_g) / sin_g],
            [0.0, 0.0, c * volume_term / sin_g],
        ]
    )


def max_xyz_shift_A(res_a: Path, res_b: Path, cell=None) -> float:
    """Largest non-H atom displacement in A between two models (minimum image).

    The full orthogonalization matrix is used, so monoclinic and triclinic cells are handled as well as
    orthogonal ones.
    """

    first, second = atom_xyz(res_a), atom_xyz(res_b)
    matrix = orthogonalization_matrix(cell if cell is not None else cell_from_res(res_a))
    shifts = [
        float(np.linalg.norm(matrix @ ((first[name] - second[name]) - np.rint(first[name] - second[name]))))
        for name in first
        if name in second
    ]
    return float(max(shifts)) if shifts else float("nan")


def row_r_by_split(run_dirs: Iterable[Path | str]) -> dict[str, object]:
    """Pooled row amplitude R (percent) on work and free rows of native row tables.

    R_all uses every row, R_obs rows with target_I > 3 source_sigma (``heldout.r_metrics``).  The rows
    are those of the model that produced the tables; missing tables give None, never a fallback.
    """

    y: list[float] = []
    sig: list[float] = []
    pred: list[float] = []
    split: list[str] = []
    tables: list[str] = []
    for run_dir in run_dirs:
        table = Path(run_dir) / "native_dyn_calc.csv"
        if not table.is_file():
            return {"status": "missing_row_table", "row_table": str(table)}
        tables.append(str(table))
        with table.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    target = float(row["target_I"])
                    sigma = float(row.get("source_sigma") or row["sigma_eff"])
                    prediction = float(row["dyn_scaled"])
                except (KeyError, TypeError, ValueError):
                    continue
                y.append(target)
                sig.append(sigma)
                pred.append(prediction)
                split.append(str(row.get("split", "work")))
    out: dict[str, object] = {"status": "ok", "row_tables": tables}
    labels = np.asarray(split, dtype=object)
    for name in ("work", "free"):
        selected = labels == name
        metrics = r_metrics(y, sig, pred, selected) if len(y) else {"r_all": float("nan"), "r_obs": float("nan"), "n_all": 0}
        out[f"{name}_r_all"] = _finite_or_none(metrics["r_all"])
        out[f"{name}_r_obs"] = _finite_or_none(metrics["r_obs"])
        out[f"n_{name}_rows"] = int(metrics["n_all"])
    return out


def trajectory_entry(
    cycle: int,
    entering_rows: Mapping[str, object],
    shelxl: Mapping[str, object],
    *,
    input_model: str,
    output_model: str,
    n_excluded_q_rows=None,
) -> dict[str, object]:
    """One macro-cycle of the trajectory contract (row R of the entering model, SHELXL, shift, restraints)."""

    output = Path(output_model) if output_model else None
    restraints = shelxl_model_restraint_audit(output) if output is not None else {}
    shift = float("nan")
    if output is not None and output.is_file() and Path(input_model).is_file():
        try:
            shift = max_xyz_shift_A(Path(input_model), output)
        except (OSError, ValueError):
            shift = float("nan")
    try:
        n_npd = int(shelxl.get("npd_count", 0) or 0)
    except (TypeError, ValueError):
        n_npd = 0
    history = shelxl.get("iteration_history") if isinstance(shelxl.get("iteration_history"), list) else []
    return {
        "cycle": int(cycle),
        "free_r_obs": entering_rows.get("free_r_obs"),
        "free_r_all": entering_rows.get("free_r_all"),
        "work_r_obs": entering_rows.get("work_r_obs"),
        "work_r_all": entering_rows.get("work_r_all"),
        "n_free_rows": entering_rows.get("n_free_rows"),
        "n_work_rows": entering_rows.get("n_work_rows"),
        "r_units": "percent amplitude R of the model entering this cycle",
        "r1_gt": _finite_or_none(shelxl.get("R1_gt")),
        "r1_all": _finite_or_none(shelxl.get("R1_all")),
        "wr2": _finite_or_none(shelxl.get("wR2_all")),
        "goof": _finite_or_none(shelxl.get("GooF")),
        "max_shift_esd": _finite_or_none(shelxl.get("max_shift_esd")),
        "mean_shift_esd": _finite_or_none(shelxl.get("mean_shift_esd")),
        "max_shift_param": str(shelxl.get("max_shift_param") or ""),
        "shelxl_calls": len(history) if history else None,
        "shelxl_converged": bool(shelxl.get("converged", False)),
        "max_xyz_shift_A": _finite_or_none(shift),
        "n_npd": n_npd,
        "n_isor": int(restraints.get("n_isor", 0) or 0),
        "isor_atoms": list(restraints.get("isor_atoms") or []),
        "n_excluded_q_rows": None if n_excluded_q_rows is None else int(n_excluded_q_rows),
        "damp_present": bool(restraints.get("damp_present", False)),
        "input_model": str(input_model),
        "output_model": str(output_model or ""),
    }


def select_kept_cycle(scores: list[tuple[int, float | None]], policy: str) -> tuple[int, str]:
    """Pick the macro-cycle whose output model is kept: the last adopted cycle (``last``).

    ``scores`` are (cycle, free-row R_obs of that cycle's output model) for every adopted cycle in order.
    Returns (0, reason) when no cycle was adopted.
    """

    if policy not in KEEP_POLICIES:
        raise ValueError(f"unknown macro-cycle keep policy {policy!r}; use one of {', '.join(KEEP_POLICIES)}")
    if not scores:
        return 0, "no macro-cycle model was adopted; the start model is kept"
    return int(scores[-1][0]), "last adopted macro-cycle (legacy rule)"
