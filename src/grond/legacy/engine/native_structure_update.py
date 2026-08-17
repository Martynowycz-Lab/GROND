#!/usr/bin/env python3
"""Native Bloch row structure update for XDS row data.

This is the diagnostic gradient-bearing layer above the frozen Bloch row scorer.
It updates selected fractional coordinates and ADPs against a native intensity
least-squares target with validation-gated block acceptance. By default,
nuisance scales are frozen inside each coordinate/ADP microcycle and refreshed
only between blocks, so a moving structure cannot be hidden by per-evaluation
scale refits. It is intentionally conservative and is not the default single-dataset
path when the SHELXL feedback loop is available.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import gemmi
import numpy as np
from scipy.optimize import least_squares

if __package__:
    from .bloch_engine import (
        Config,
        XDSBlochPort,
        XDSData,
        alternate_hand_model,
        parse_model,
        sha256,
        shelx_u_cart_matrix,
        split_masks,
    )
    from .native_row_regularize import (
        DEFAULT_BASELINE_HKL,
        DEFAULT_HKL,
        DEFAULT_RES,
        TARGET_MODES,
        Nuisance,
        copy_baseline,
        detector_bin_indices,
        evaluate_trial,
        export_corrected_endpoint,
        fit_frame_detector_scales,
        fit_group_scales,
        group_indices,
        native_metrics,
        parse_float_grid,
        parse_wght,
        row_resolution_modulation,
        row_selection,
        shelx_weights,
        sigma_eff,
        target_arrays,
    )
else:  # pragma: no cover - direct script execution
    from bloch_engine import (
        Config,
        XDSBlochPort,
        XDSData,
        alternate_hand_model,
        parse_model,
        sha256,
        shelx_u_cart_matrix,
        split_masks,
    )
    from native_row_regularize import (
        DEFAULT_BASELINE_HKL,
        DEFAULT_HKL,
        DEFAULT_RES,
        TARGET_MODES,
        Nuisance,
        copy_baseline,
        detector_bin_indices,
        evaluate_trial,
        export_corrected_endpoint,
        fit_frame_detector_scales,
        fit_group_scales,
        group_indices,
        native_metrics,
        parse_float_grid,
        parse_wght,
        row_resolution_modulation,
        row_selection,
        shelx_weights,
        sigma_eff,
        target_arrays,
    )


class EarlyStopOptimization(Exception):
    """Stop a structure block after a real plateau while preserving its best vector."""

    def __init__(self, reason: str, x, metric: float, eval_no: int):
        super().__init__(reason)
        self.reason = reason
        self.x = np.asarray(x, dtype=float).copy()
        self.metric = float(metric)
        self.eval_no = int(eval_no)


def early_stop_min_evals(args, params: list[dict]) -> int:
    """Return the minimum residual calls before a plateau gate is allowed to fire."""

    explicit = int(getattr(args, "early_stop_min_evals", 0))
    if explicit > 0:
        return explicit
    window = max(1, int(getattr(args, "early_stop_window", 0)))
    return max(2 * (len(params) + 1), window)


def make_early_stop_result(exc: EarlyStopOptimization, eval_count: int):
    """Build a small scipy-like result object for a block stopped by our gate."""

    return SimpleNamespace(
        x=exc.x,
        success=False,
        status=0,
        message=f"early stopped: {exc.reason}",
        cost=float("nan"),
        optimality=float("nan"),
        nfev=int(eval_count),
        early_stop=True,
        early_stop_reason=exc.reason,
        early_stop_metric=float(exc.metric),
        early_stop_eval=int(exc.eval_no),
    )


def parameter_absolute_steps(
    params: list[dict],
    *,
    xyz_step_frac: float = 1e-4,
    adp_step_a2: float = 1e-4,
    legacy_step: float | None = None,
) -> np.ndarray:
    """Return finite-difference steps in each parameter's physical units.

    SciPy's ``least_squares(diff_step=...)`` treats its input as a relative
    multiplier of the current parameter value.  Fractional coordinates can be
    zero and Ueq values are normally much smaller than coordinates, so one
    relative scalar gives parameter-dependent, and sometimes near-zero, steps.
    Native GROND instead uses explicit fractional-coordinate and A^2 steps.
    ``legacy_step`` retains the old single-knob CLI as an absolute shorthand.
    """

    if legacy_step is not None:
        xyz_step_frac = float(legacy_step)
        adp_step_a2 = float(legacy_step)
    values = {
        "xyz": float(xyz_step_frac),
        "adp_ueq": float(adp_step_a2),
    }
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} finite-difference step must be finite and positive")
    steps = []
    for param in params:
        kind = str(param.get("kind", "xyz"))
        if kind not in values:
            raise ValueError(f"unknown parameter kind {kind!r} for finite-difference step")
        steps.append(values[kind])
    return np.asarray(steps, dtype=float)


def absolute_finite_difference_jacobian(
    fun,
    x,
    bounds,
    absolute_steps,
    *,
    f0=None,
) -> np.ndarray:
    """Calculate a bounded two-point Jacobian using explicit absolute steps."""

    x = np.asarray(x, dtype=float)
    lower, upper = (np.asarray(value, dtype=float) for value in bounds)
    steps = np.asarray(absolute_steps, dtype=float)
    if x.ndim != 1 or lower.shape != x.shape or upper.shape != x.shape or steps.shape != x.shape:
        raise ValueError("x, bounds, and absolute finite-difference steps must be matching vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(steps)) or np.any(steps <= 0.0):
        raise ValueError("finite-difference parameters and steps must be finite with positive steps")
    base = np.asarray(fun(x) if f0 is None else f0, dtype=float)
    if base.ndim != 1:
        raise ValueError("finite-difference residual function must return a vector")
    jac = np.empty((len(base), len(x)), dtype=float)
    for column, requested in enumerate(steps):
        forward_room = float(upper[column] - x[column])
        backward_room = float(x[column] - lower[column])
        if forward_room >= requested:
            delta = float(requested)
        elif backward_room >= requested:
            delta = -float(requested)
        elif forward_room >= backward_room and forward_room > 0.0:
            delta = forward_room
        elif backward_room > 0.0:
            delta = -backward_room
        else:
            raise ValueError(f"parameter {column} has no room for a bounded finite-difference step")
        trial = x.copy()
        trial[column] += delta
        shifted = np.asarray(fun(trial), dtype=float)
        if shifted.shape != base.shape:
            raise ValueError("finite-difference residual size changed between parameter evaluations")
        jac[:, column] = (shifted - base) / delta
    return jac


def atom_is_refinable(atom, selection: str, labels: set[str]) -> bool:
    if atom["name"].upper().startswith("Q"):
        return False
    if selection == "all":
        return True
    if selection == "nonh":
        return atom.get("element", "").upper() != "H"
    if selection == "labels":
        return atom["name"] in labels
    raise ValueError(f"unknown atom selection {selection!r}")


def atom_ueq(atom, cell_tuple) -> float:
    if atom.get("adp_type") == "anis":
        return float(np.trace(shelx_u_cart_matrix(atom, cell_tuple))) / 3.0
    return float(atom.get("uiso", 0.05))


def set_atom_ueq(atom, value: float, cell_tuple) -> None:
    value = max(float(value), 1e-5)
    if atom.get("adp_type") == "anis":
        start = atom_ueq(atom, cell_tuple)
        scale = value / max(start, 1e-8)
        for key in ("u11", "u22", "u33", "u23", "u13", "u12"):
            atom[key] = float(atom[key]) * scale
    else:
        atom["uiso"] = value


def parameter_layout(model, selection: str, label_text: str, parameter_mode: str = "xyz"):
    labels = {x.strip() for x in re.split(r"[,\s]+", label_text or "") if x.strip()}
    params = []
    x0 = []
    for ai, atom in enumerate(model["atoms"]):
        if not atom_is_refinable(atom, selection, labels):
            continue
        if parameter_mode in {"xyz", "xyz_adp_ueq"}:
            for coord in ("x", "y", "z"):
                params.append({"kind": "xyz", "atom_index": ai, "atom": atom["name"], "coord": coord, "start": float(atom[coord])})
                x0.append(float(atom[coord]))
        if parameter_mode in {"adp_ueq", "xyz_adp_ueq"}:
            ueq = atom_ueq(atom, model["cell"])
            params.append(
                {
                    "kind": "adp_ueq",
                    "atom_index": ai,
                    "atom": atom["name"],
                    "coord": "Ueq",
                    "start": float(ueq),
                    "start_tensor": {k: float(atom[k]) for k in ("u11", "u22", "u33", "u23", "u13", "u12") if k in atom},
                }
            )
            x0.append(float(ueq))
    if not params:
        raise ValueError(f"No parameters selected for mode {parameter_mode!r}")
    return params, np.asarray(x0, dtype=float)


def model_from_vector(base_model, params, x):
    model = copy.deepcopy(base_model)
    for p, value in zip(params, x, strict=False):
        atom = model["atoms"][p["atom_index"]]
        if p.get("kind", "xyz") == "xyz":
            atom[p["coord"]] = float(value) % 1.0
        elif p["kind"] == "adp_ueq":
            set_atom_ueq(atom, float(value), model["cell"])
        else:
            raise ValueError(f"unknown parameter kind {p.get('kind')!r}")
    return model


def parameter_bounds(params, x0, max_shift_frac: float, adp_shift: float, adp_min: float, adp_max: float):
    lo, hi = [], []
    for p, v in zip(params, x0, strict=False):
        if p.get("kind", "xyz") == "xyz":
            lo.append(float(v) - float(max_shift_frac))
            hi.append(float(v) + float(max_shift_frac))
        elif p["kind"] == "adp_ueq":
            center = float(v)
            lo_val = max(float(adp_min), center - float(adp_shift))
            hi_val = min(float(adp_max), center + float(adp_shift))
            lo.append(min(lo_val, center))
            hi.append(max(hi_val, center))
        else:
            raise ValueError(f"unknown parameter kind {p.get('kind')!r}")
    return np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)


def parameter_prior(params, x, x0, xyz_sigma: float, adp_sigma: float):
    out = []
    for p, value, start in zip(params, x, x0, strict=False):
        sigma = float(adp_sigma) if p.get("kind") == "adp_ueq" else float(xyz_sigma)
        out.append((float(value) - float(start)) / max(sigma, 1e-12))
    return np.asarray(out, dtype=float)


COVALENT_RADII_A = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "P": 1.07,
    "F": 0.57,
    "CL": 1.02,
    "BR": 1.20,
    "I": 1.39,
}


def frac_xyz(atom):
    return np.asarray([float(atom["x"]), float(atom["y"]), float(atom["z"])], dtype=float)


def frac_delta_min_image(atom_i, atom_j, cell_tuple):
    """Return the closest periodic image under the full unit-cell metric."""

    delta = frac_xyz(atom_j) - frac_xyz(atom_i)
    basis = np.column_stack(
        [
            cart_delta(cell_tuple, np.asarray([1.0, 0.0, 0.0])),
            cart_delta(cell_tuple, np.asarray([0.0, 1.0, 0.0])),
            cart_delta(cell_tuple, np.asarray([0.0, 0.0, 1.0])),
        ]
    )
    _q, upper = np.linalg.qr(basis)
    target = upper @ delta
    if np.any(np.abs(np.diag(upper)) <= np.finfo(float).eps):
        raise ValueError("unit-cell basis is singular; no periodic closest image exists")

    # Babai's nearest-plane point supplies a finite initial radius. A short
    # three-dimensional sphere search then proves the closest lattice vector,
    # even when a highly skew, non-reduced cell needs a translation beyond +/-1.
    best_translation = np.zeros(3, dtype=int)
    for index in range(2, -1, -1):
        tail = float(upper[index, index + 1 :] @ best_translation[index + 1 :])
        best_translation[index] = int(np.rint((target[index] - tail) / upper[index, index]))
    best_distance2 = float(np.sum((target - upper @ best_translation) ** 2))
    candidate = np.zeros(3, dtype=int)

    def search(index: int, partial_distance2: float) -> None:
        nonlocal best_distance2, best_translation
        if index < 0:
            if partial_distance2 < best_distance2:
                best_distance2 = partial_distance2
                best_translation = candidate.copy()
            return
        remaining = max(best_distance2 - partial_distance2, 0.0)
        tail = float(upper[index, index + 1 :] @ candidate[index + 1 :])
        center = (target[index] - tail) / upper[index, index]
        radius = math.sqrt(remaining) / abs(float(upper[index, index]))
        lower = math.ceil(center - radius - 1e-12)
        upper_bound = math.floor(center + radius + 1e-12)
        values = sorted(range(lower, upper_bound + 1), key=lambda value: abs(value - center))
        for value in values:
            residual = target[index] - tail - upper[index, index] * value
            new_distance2 = partial_distance2 + float(residual * residual)
            if new_distance2 <= best_distance2 + 1e-15:
                candidate[index] = int(value)
                search(index - 1, new_distance2)

    search(2, 0.0)
    return np.asarray(delta - best_translation, dtype=float)


def cart_delta(cell_tuple, frac_delta):
    a, b, c, alpha, beta, gamma = (float(v) for v in cell_tuple)
    if abs(alpha - 90.0) < 1e-6 and abs(beta - 90.0) < 1e-6 and abs(gamma - 90.0) < 1e-6:
        return np.asarray([frac_delta[0] * a, frac_delta[1] * b, frac_delta[2] * c], dtype=float)
    cell = gemmi.UnitCell(a, b, c, alpha, beta, gamma)
    zero = cell.orthogonalize(gemmi.Fractional(0.0, 0.0, 0.0))
    vec = cell.orthogonalize(gemmi.Fractional(float(frac_delta[0]), float(frac_delta[1]), float(frac_delta[2])))
    return np.asarray([vec.x - zero.x, vec.y - zero.y, vec.z - zero.z], dtype=float)


def atom_distance_A(model, ai: int, aj: int) -> float:
    dfrac = frac_delta_min_image(model["atoms"][ai], model["atoms"][aj], model["cell"])
    return float(np.linalg.norm(cart_delta(model["cell"], dfrac)))


def build_start_restraints(model, selection: str, label_text: str, tolerance_A: float, max_distance_A: float):
    labels = {x.strip() for x in re.split(r"[,\s]+", label_text or "") if x.strip()}
    atoms = model["atoms"]
    bonds = []
    for i, ai in enumerate(atoms):
        if ai["name"].upper().startswith("Q"):
            continue
        ei = ai.get("element", "").upper()
        ri = COVALENT_RADII_A.get(ei, 0.77)
        for j in range(i + 1, len(atoms)):
            aj = atoms[j]
            if aj["name"].upper().startswith("Q"):
                continue
            ej = aj.get("element", "").upper()
            rj = COVALENT_RADII_A.get(ej, 0.77)
            target = atom_distance_A(model, i, j)
            cutoff = min(float(max_distance_A), ri + rj + float(tolerance_A))
            if target < 0.45 or target > cutoff:
                continue
            update_i = atom_is_refinable(ai, selection, labels)
            update_j = atom_is_refinable(aj, selection, labels)
            if not (update_i or update_j):
                continue
            bonds.append(
                {
                    "i": i,
                    "j": j,
                    "atom_i": ai["name"],
                    "atom_j": aj["name"],
                    "element_i": ei,
                    "element_j": ej,
                    "target_A": float(target),
                    "start_ueq_diff": atom_ueq(ai, model["cell"]) - atom_ueq(aj, model["cell"]),
                    "updates_i": bool(update_i),
                    "updates_j": bool(update_j),
                }
            )
    return bonds


def bond_restraint_residuals(model, restraints, sigma_A: float):
    if sigma_A <= 0.0 or not restraints:
        return np.zeros(0, dtype=float)
    out = []
    for r in restraints:
        d = atom_distance_A(model, int(r["i"]), int(r["j"]))
        out.append((d - float(r["target_A"])) / float(sigma_A))
    return np.asarray(out, dtype=float)


def adp_similarity_residuals(model, restraints, sigma_u: float):
    if sigma_u <= 0.0 or not restraints:
        return np.zeros(0, dtype=float)
    out = []
    atoms = model["atoms"]
    for r in restraints:
        ai = atoms[int(r["i"])]
        aj = atoms[int(r["j"])]
        if ai.get("element", "").upper() == "H" or aj.get("element", "").upper() == "H":
            continue
        diff = atom_ueq(ai, model["cell"]) - atom_ueq(aj, model["cell"])
        out.append((diff - float(r["start_ueq_diff"])) / float(sigma_u))
    return np.asarray(out, dtype=float)


def write_restraints_csv(path: Path, restraints):
    with path.open("w", newline="") as f:
        fieldnames = ["atom_i", "atom_j", "element_i", "element_j", "target_A", "start_ueq_diff", "updates_i", "updates_j"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in restraints:
            w.writerow({k: r[k] for k in fieldnames})


def write_cctbx_restraint_view(src: Path, dst: Path) -> None:
    """Write a SHELX view cctbx can parse without losing the ED SFAC in src.

    cctbx's SHELX parser rejects explicit electron scattering coefficient SFAC
    rows, but the restraint/constraint grammar only needs element labels.  This
    view is therefore for cctbx/smtbx restraint and constraint discovery only;
    the Bloch calculator still reads the original ED SFAC rows.
    """
    elems = []
    lines = []
    for line in src.read_text(errors="replace").splitlines():
        if line.upper().startswith("SFAC"):
            parts = line.split()
            if len(parts) > 1:
                elems.append(parts[1])
            continue
        lines.append(line)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.upper().startswith("LATT"):
            insert_at = i + 1
            break
    lines.insert(insert_at, "SFAC  " + " ".join(elems))
    dst.write_text("\n".join(lines) + "\n")


def inspect_cctbx_bridge(src: Path, out_dir: Path):
    view = out_dir / "cctbx_restraint_view.res"
    status = {"available": False, "parsed": False, "view": str(view)}
    try:
        from iotbx import shelx

        write_cctbx_restraint_view(src, view)
        parser = getattr(shelx, "parse_smtbx_" + "ref" + "inement_model")
        builder = parser(filename=str(view), strictly_shelxl=False)
        constraints = getattr(builder, "constraints", []) or []
        manager = getattr(builder, "restraints_manager", None)
        structure = getattr(builder, "structure", None)
        status.update(
            {
                "available": True,
                "parsed": True,
                "scatterers": int(structure.scatterers().size()) if structure is not None else None,
                "constraints_count": int(len(constraints)),
                "constraints_preview": [str(c).splitlines()[0] for c in constraints[:12]],
                "restraints_manager_type": type(manager).__name__ if manager is not None else None,
            }
        )
    except Exception as exc:
        status.update({"error": f"{type(exc).__name__}: {exc}"})
    return status


class CctbxRiguEvaluator:
    """Evaluate cctbx/SHELX RIGU residuals for the current native model."""

    def __init__(self, src: Path, out_dir: Path):
        self.view = out_dir / "cctbx_restraint_view.res"
        self.status = {"available": False, "parsed": False, "view": str(self.view), "rigu_count": 0}
        try:
            from cctbx import adp_restraints
            from iotbx import shelx
            from scitbx.array_family import flex

            write_cctbx_restraint_view(src, self.view)
            parser = getattr(shelx, "parse_smtbx_" + "ref" + "inement_model")
            builder = parser(filename=str(self.view), strictly_shelxl=False)
            self._adp_restraints = adp_restraints
            self._flex = flex
            self._xray_structure = builder.structure
            self._restraints_manager = builder.restraints_manager
            self._rigu_proxies = getattr(self._restraints_manager, "rigu_proxies", None)
            self._labels = list(self._xray_structure.scatterers().extract_labels())
            base_scatterers = self._xray_structure.scatterers()
            self._base_sites = [tuple(v) for v in base_scatterers.extract_sites()]
            self._base_u_cart = [tuple(v) for v in base_scatterers.extract_u_cart(self._xray_structure.unit_cell())]
            self._base_u_iso = list(base_scatterers.extract_u_iso())
            self._base_use_u_aniso = list(base_scatterers.extract_use_u_aniso())
            constraints = getattr(builder, "constraints", []) or []
            rigu_count = int(self._rigu_proxies.size()) if self._rigu_proxies is not None else 0
            self.status.update(
                {
                    "available": True,
                    "parsed": True,
                    "scatterers": int(self._xray_structure.scatterers().size()),
                    "constraints_count": int(len(constraints)),
                    "constraints_preview": [str(c).splitlines()[0] for c in constraints[:12]],
                    "restraints_manager_type": type(self._restraints_manager).__name__ if self._restraints_manager is not None else None,
                    "rigu_count": rigu_count,
                }
            )
        except Exception as exc:
            self._adp_restraints = None
            self._flex = None
            self._xray_structure = None
            self._restraints_manager = None
            self._rigu_proxies = None
            self._labels = []
            self._base_sites = []
            self._base_u_cart = []
            self._base_u_iso = []
            self._base_use_u_aniso = []
            self.status.update({"error": f"{type(exc).__name__}: {exc}"})

    @property
    def count(self) -> int:
        return int(self.status.get("rigu_count") or 0)

    def residuals(self, model):
        if not self.status.get("parsed") or self._rigu_proxies is None or self.count <= 0:
            return np.zeros(0, dtype=float)
        atom_by_name = {a["name"]: a for a in model["atoms"]}
        sites = []
        u_cart = []
        for label, base_site, base_u in zip(self._labels, self._base_sites, self._base_u_cart, strict=False):
            atom = atom_by_name.get(label)
            if atom is None:
                sites.append(base_site)
                u_cart.append(base_u)
                continue
            sites.append((float(atom["x"]) % 1.0, float(atom["y"]) % 1.0, float(atom["z"]) % 1.0))
            if atom.get("adp_type") == "anis":
                u = shelx_u_cart_matrix(atom, model["cell"])
                u_cart.append((u[0, 0], u[1, 1], u[2, 2], u[0, 1], u[0, 2], u[1, 2]))
            else:
                u = atom_ueq(atom, model["cell"])
                u_cart.append((u, u, u, 0.0, 0.0, 0.0))
        xs = self._xray_structure.deep_copy_scatterers()
        xs.set_sites_frac(self._flex.vec3_double(sites))
        xs.set_u_cart(self._flex.sym_mat3_double(u_cart))
        sc = xs.scatterers()
        params = self._adp_restraints.adp_restraint_params(
            xs.sites_cart(),
            sc.extract_u_cart(xs.unit_cell()),
            sc.extract_u_iso(),
            sc.extract_use_u_aniso(),
        )
        values = np.asarray(list(self._adp_restraints.rigu_residuals(params, self._rigu_proxies)), dtype=float)
        return np.sqrt(np.maximum(values, 0.0))


def choose_objective_rows(data: XDSData, indices, masks, target, sig, n_rows: int, n_bins: int, seed: int):
    candidates = np.asarray([i for i in indices if masks["work"][i] and target[i] > 0 and np.isfinite(target[i]) and np.isfinite(sig[i])], dtype=int)
    if n_rows <= 0 or n_rows >= len(candidates):
        return np.asarray(candidates, dtype=int)
    rng = np.random.default_rng(int(seed))
    dvals = np.asarray([data.cell.calculate_d(tuple(int(v) for v in h)) for h in data.hkl], dtype=float)
    order = candidates[np.argsort(dvals[candidates])]
    bins = np.array_split(order, max(1, int(n_bins)))
    chosen = []
    remaining = int(n_rows)
    for _bi, b in enumerate(bins):
        if len(b) == 0:
            continue
        quota = int(round(n_rows * len(b) / max(len(candidates), 1)))
        quota = max(1, min(quota, len(b), remaining))
        remaining -= quota
        # Keep the row set representative but gently favor informative rows.
        isigi = np.maximum(target[b] / np.maximum(sig[b], 1e-12), 0.0)
        prob = np.sqrt(isigi + 1.0)
        prob = prob / np.sum(prob)
        chosen.extend(rng.choice(b, size=quota, replace=False, p=prob).tolist())
    if remaining > 0:
        pool = np.asarray([i for i in candidates if i not in set(chosen)], dtype=int)
        if len(pool):
            extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
            chosen.extend(extra.tolist())
    return np.asarray(sorted(set(int(i) for i in chosen)), dtype=int)


def write_objective_rows(path: Path, data: XDSData, rows, target, sig, masks) -> None:
    fieldnames = [
        "row_index",
        "h",
        "k",
        "l",
        "d_A",
        "Iobs",
        "sigma",
        "I_over_sigma",
        "split",
        "zcal",
        "rlp",
        "peak",
        "corr",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in np.asarray(rows, dtype=int):
            h, k, l = (int(v) for v in data.hkl[i])
            split = "free" if masks["free"][i] else "work"
            sigma = float(sig[i])
            w.writerow(
                {
                    "row_index": int(i),
                    "h": h,
                    "k": k,
                    "l": l,
                    "d_A": float(data.cell.calculate_d((h, k, l))),
                    "Iobs": float(target[i]),
                    "sigma": sigma,
                    "I_over_sigma": float(target[i] / sigma) if sigma > 0 else float("nan"),
                    "split": split,
                    "zcal": float(data.zcal[i]),
                    "rlp": float(data.rlp[i]),
                    "peak": float(data.peak[i]),
                    "corr": float(data.corr[i]),
                }
            )


def prediction_for_model(data, model, cfg, indices, cache_root: Path, model_tag: str, workers: int, chunksize: int):
    # Do not use the persistent prediction cache for moving models.  The Ug cube
    # cache is still content-addressed by model hash inside XDSBlochPort.
    port = XDSBlochPort(data, model, cfg, cache_dir=cache_root / "ug", res_hash=model_tag)
    return port.predict_rows(indices, workers=workers, chunksize=chunksize)


def fit_scale_contract(data, target, sig, pred_full, cfg, fit_indices, wght_a, wght_b, detector_scale_bins: int = 0):
    groups = group_indices(data, cfg.group_width_deg, cfg.frame_offset)
    det_bins = detector_bin_indices(data, int(detector_scale_bins))
    fit_mask = np.zeros(len(target), dtype=bool)
    fit_mask[np.asarray(fit_indices, dtype=int)] = True
    if int(detector_scale_bins) > 1:
        scaled, group_scales, detector_scales = fit_frame_detector_scales(
            target,
            sig,
            pred_full,
            groups,
            det_bins,
            fit_mask,
            wght_a,
            wght_b,
        )
    else:
        scaled, group_scales = fit_group_scales(target, sig, pred_full, groups, fit_mask, wght_a, wght_b)
        detector_scales = {}
    return {
        "mode": "fit",
        "group_width_deg": float(cfg.group_width_deg),
        "frame_offset": float(cfg.frame_offset),
        "detector_scale_bins": int(detector_scale_bins),
        "groups": groups,
        "detector_bins": det_bins,
        "group_scales": {str(k): float(v) for k, v in group_scales.items()},
        "detector_scales": {str(k): float(v) for k, v in detector_scales.items()},
        "scale_count": int(len(group_scales) + len(detector_scales)),
        "scaled": scaled,
    }


def apply_scale_contract(pred_full, contract):
    scaled = np.full_like(pred_full, np.nan, dtype=float)
    groups = np.asarray(contract["groups"], dtype=int)
    group_scales = {int(k): float(v) for k, v in contract.get("group_scales", {}).items()}
    detector_scales = {int(k): float(v) for k, v in contract.get("detector_scales", {}).items()}
    det_bins = np.asarray(contract.get("detector_bins", np.zeros_like(groups)), dtype=int)
    for i, value in enumerate(pred_full):
        if not np.isfinite(value):
            continue
        scale = group_scales.get(int(groups[i]))
        if scale is None or not np.isfinite(scale) or scale <= 0:
            continue
        if int(contract.get("detector_scale_bins", 0)) > 1:
            dscale = detector_scales.get(int(det_bins[i]))
            if dscale is None or not np.isfinite(dscale) or dscale <= 0:
                continue
            scale *= dscale
        scaled[i] = float(value) * float(scale)
    return scaled


def apply_row_resolution_to_prediction(data, pred_full, row_resolution_b: float = 0.0):
    """Apply the raw-row resolution envelope used by dynamic gradients."""

    valid = np.isfinite(pred_full) & (pred_full > 0.0)
    factor, meta = row_resolution_modulation(data, valid, row_resolution_b)
    return pred_full * factor, factor, meta


def scaled_prediction(data, target, sig, pred, cfg, fit_indices, wght_a, wght_b, detector_scale_bins: int = 0, scale_contract=None, row_resolution_b: float = 0.0):
    full = np.full(len(target), np.nan, dtype=float)
    full[pred["indices"]] = pred["dyn"]
    full, row_factor, row_meta = apply_row_resolution_to_prediction(data, full, row_resolution_b)
    if scale_contract is not None:
        groups = np.asarray(scale_contract["groups"], dtype=int)
        scaled = apply_scale_contract(full, scale_contract)
        scales = dict(scale_contract.get("group_scales", {}))
        scale_count = int(scale_contract.get("scale_count", len(scales)))
    else:
        contract = fit_scale_contract(data, target, sig, full, cfg, fit_indices, wght_a, wght_b, detector_scale_bins=detector_scale_bins)
        groups = np.asarray(contract["groups"], dtype=int)
        scaled = contract["scaled"]
        scales = dict(contract["group_scales"])
        scale_count = int(contract["scale_count"])
    return full, scaled, scales, groups, scale_count


def residual_vector(data, target, sig, pred_scaled, indices, wght_a, wght_b):
    idx = np.asarray(indices, dtype=int)
    obs = target[idx]
    calc = pred_scaled[idx]
    sig_i = sig[idx]
    good = np.isfinite(obs) & np.isfinite(calc) & np.isfinite(sig_i)
    obs = obs[good]
    calc = calc[good]
    sig_i = np.maximum(sig_i[good], 1e-12)
    w = shelx_weights(obs, sig_i, calc, wght_a, wght_b)
    return np.sqrt(w) * (obs - calc)


def evaluate_model(data, model, cfg, target, sig, indices, masks, fit_indices, wght_a, wght_b, cache_root, tag, workers, chunksize, detector_scale_bins: int = 0, row_resolution_b: float = 0.0):
    pred = prediction_for_model(data, model, cfg, indices, cache_root, tag, workers, chunksize)
    full, dyn_scaled, scales, groups, scale_count = scaled_prediction(data, target, sig, pred, cfg, fit_indices, wght_a, wght_b, detector_scale_bins=detector_scale_bins, row_resolution_b=row_resolution_b)
    out = {"splits": {}, "scale_count": int(scale_count), "prediction_seconds": float(pred.get("seconds", 0.0))}
    valid = np.isfinite(dyn_scaled)
    for split, mask in masks.items():
        out["splits"][split] = native_metrics(target, sig, dyn_scaled, mask & valid, wght_a, wght_b, n_params=int(scale_count))
    out["arrays"] = {"dyn_raw": full, "dyn_scaled": dyn_scaled, "groups": groups, "valid": valid}
    return out


def validation_score(eval_result, metric: str) -> float:
    split, value = metric.split("_", 1)
    if split not in eval_result["splits"]:
        raise ValueError(f"unknown validation split in metric {metric!r}")
    key = "wR2_all" if value == "wR2" else "R1_all"
    return float(eval_result["splits"][split][key])


def format_native_value(eval_result, split: str, key: str) -> float:
    return float(eval_result["splits"][split][key])


def update_res_coordinates(src: Path, dst: Path, model) -> None:
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
                # Preserve riding-H negative Uiso multipliers and occupancy
                # conventions unless this atom has an explicit updated Uiso.
                rest = parts[5:]
                if atom.get("element", "").upper() != "H" and "uiso" in atom and len(rest) >= 2:
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


def write_parameter_csv(path: Path, params, x0, x_final):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["atom", "kind", "coord", "start", "final", "shift"])
        w.writeheader()
        for p, a, b in zip(params, x0, x_final, strict=False):
            w.writerow({"atom": p["atom"], "kind": p.get("kind", "xyz"), "coord": p["coord"], "start": float(a), "final": float(b), "shift": float(b - a)})


def write_model_delta_csv(path: Path, start_model, final_model):
    start_by_name = {a["name"]: a for a in start_model["atoms"]}
    with path.open("w", newline="") as f:
        fieldnames = ["atom", "element", "dx", "dy", "dz", "start_ueq", "final_ueq", "delta_ueq"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for atom in final_model["atoms"]:
            start = start_by_name.get(atom["name"])
            if start is None:
                continue
            w.writerow(
                {
                    "atom": atom["name"],
                    "element": atom.get("element", ""),
                    "dx": float(atom["x"]) - float(start["x"]),
                    "dy": float(atom["y"]) - float(start["y"]),
                    "dz": float(atom["z"]) - float(start["z"]),
                    "start_ueq": atom_ueq(start, start_model["cell"]),
                    "final_ueq": atom_ueq(atom, final_model["cell"]),
                    "delta_ueq": atom_ueq(atom, final_model["cell"])
                    - atom_ueq(start, start_model["cell"]),
                }
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hkl", default=str(DEFAULT_HKL))
    ap.add_argument("--res", default=str(DEFAULT_RES))
    ap.add_argument("--baseline-hkl", default=str(DEFAULT_BASELINE_HKL))
    ap.add_argument("--hkl-transform", default="identity", help="Signed axis permutation mapping XDS HKLs into the model basis, e.g. 'l,-k,h'.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-basis", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=10)
    ap.add_argument("--sg-window", type=float, default=0.004)
    ap.add_argument("--sg-window-mode", choices=["fixed", "zobs_residual", "zobs_plus_floor", "rlp_peak", "dials_bbox"], default="fixed")
    ap.add_argument("--sg-window-min", type=float, default=0.003)
    ap.add_argument("--sg-window-max", type=float, default=0.012)
    ap.add_argument("--sg-window-scale", type=float, default=1.0)
    ap.add_argument("--sg-window-frame-floor", type=float, default=0.50)
    ap.add_argument("--min-d-spacing", type=float, default=0.70)
    ap.add_argument("--min-peak", type=float, default=0.0)
    ap.add_argument("--geometry-percentile", type=float, default=100.0, help="Keep this broad geometry percentile after PEAK/d-spacing filters.")
    ap.add_argument("--target-mode", choices=list(TARGET_MODES), default="iobs")
    ap.add_argument("--include-nonpositive", action="store_true")
    ap.add_argument("--alternate-hand", dest="alternate_hand", action="store_true", help="Use alternate fractional xyz before native row-parameter regularization as an alternate-hand control.")
    ap.add_argument("--objective-rows", type=int, default=800)
    ap.add_argument("--objective-bins", type=int, default=10)
    ap.add_argument("--objective-group-width-deg", type=float, default=3.0, help="Coarse frame-angle group width used only while fitting coordinate steps.")
    ap.add_argument(
        "--objective-scale-mode",
        choices=["frozen_block", "refit_each_eval"],
        default="frozen_block",
        help="Scale policy inside coordinate/ADP microcycles. frozen_block fits objective scales once from the block-start model and reuses them for every finite-difference evaluation; refit_each_eval preserves the older, more fragile behavior.",
    )
    ap.add_argument("--scale-fit-split", choices=["work", "all"], default="work", help="Rows used to fit validation/final nuisance scales.")
    ap.add_argument("--detector-scale-bins", type=int, default=0, help="Optional NxN detector-position scales in objective and validation fits.")
    ap.add_argument("--block-schedule", default="xyz", help="Comma-separated model-update blocks: xyz, adp_ueq, or xyz_adp_ueq. Each block is validation-gated before the next starts.")
    ap.add_argument("--atom-selection", choices=["nonh", "all", "labels"], default="nonh")
    ap.add_argument("--atom-labels", default="")
    ap.add_argument("--max-shift-frac", type=float, default=0.015)
    ap.add_argument("--xyz-restraint-sigma", type=float, default=0.008)
    ap.add_argument("--adp-shift", type=float, default=0.010, help="Maximum Ueq change allowed per ADP block.")
    ap.add_argument("--adp-restraint-sigma", type=float, default=0.004)
    ap.add_argument("--adp-min", type=float, default=0.002)
    ap.add_argument("--adp-max", type=float, default=0.080)
    ap.add_argument("--restraints", choices=["none", "start_geometry"], default="start_geometry")
    ap.add_argument("--bond-restraint-sigma", type=float, default=0.025)
    ap.add_argument("--adp-similarity-sigma", type=float, default=0.006)
    ap.add_argument("--bond-detection-tolerance", type=float, default=0.45)
    ap.add_argument("--bond-detection-max-distance", type=float, default=1.95)
    ap.add_argument("--max-nfev", type=int, default=80)
    ap.add_argument(
        "--diff-step",
        type=float,
        default=None,
        help="Deprecated compatibility name setting both absolute finite-difference steps in each parameter's native unit; use the group-specific options instead. This is no longer a SciPy relative multiplier.",
    )
    ap.add_argument("--xyz-diff-step-frac", type=float, default=1e-4, help="Absolute fractional-coordinate step used for native Bloch finite differences.")
    ap.add_argument("--adp-diff-step-a2", type=float, default=1e-4, help="Absolute Ueq step in A^2 used for native Bloch finite differences.")
    ap.add_argument("--thickness", type=float, default=2200.0)
    ap.add_argument("--thickness-grad-frac", type=float, default=0.8)
    ap.add_argument("--thickness-points", type=int, default=5)
    ap.add_argument("--topup-d-min", type=float, default=0.8, help="Resolution limit for Bloch support top-up beams.")
    ap.add_argument("--group-width-deg", type=float, default=1.0)
    ap.add_argument("--frame-offset", type=float, default=0.0)
    ap.add_argument("--row-resolution-b", type=float, default=0.0, help="Raw-row dynamical resolution-envelope B from nuisance regularization.")
    ap.add_argument("--ug-d-min", type=float, default=0.30, help="Model-dependent U_g cutoff used by the production kernel; raising it is an explicit speed/accuracy tradeoff.")
    ap.add_argument("--sg-jitter-mode", choices=["none", "gaussian", "student_t"], default="none", help="Accepted rocking-profile Sg jitter model from the production row regularizer.")
    ap.add_argument("--sg-jitter-sigma0", type=float, default=0.0, help="Base Sg jitter sigma in inverse Angstrom.")
    ap.add_argument("--sg-jitter-sigma-d-inv", type=float, default=0.0, help="Resolution-dependent Sg jitter coefficient.")
    ap.add_argument("--sg-jitter-sigma-frame", type=float, default=0.0, help="Frame-dependent Sg jitter coefficient.")
    ap.add_argument("--sg-jitter-points", type=int, default=1, help="Quadrature points used for the accepted Sg jitter convolution.")
    ap.add_argument("--sg-jitter-student-df", type=float, default=4.0, help="Student-t degrees of freedom for student_t Sg jitter.")
    ap.add_argument("--sg-jitter-width-factor", type=float, default=3.0, help="Finite support width in sigma units for the Sg jitter convolution.")
    ap.add_argument("--absorption-mode", choices=["none", "mean", "beam_dependent", "offdiag_fraction", "offdiag_tds", "midgley_like"], default="none", help="Accepted absorptive Bloch Hamiltonian mode from the production row regularizer.")
    ap.add_argument("--absorption-mu", type=float, default=0.0, help="Uniform imaginary diagonal damping in inverse Angstrom.")
    ap.add_argument("--absorption-frac", type=float, default=0.0, help="Imaginary absorptive off-diagonal fraction.")
    ap.add_argument("--absorption-mu-g2", type=float, default=0.0, help="Beam-dependent diagonal damping slope versus |g|^2.")
    ap.add_argument("--absorption-mu-theta2", type=float, default=0.0, help="Beam-dependent diagonal damping slope versus Bragg-angle^2.")
    ap.add_argument("--absorption-tds-b", type=float, default=0.0, help="B_TDS parameter for offdiag_tds absorption.")
    ap.add_argument("--sigma-a", type=float, default=0.0)
    ap.add_argument("--sigma-b-corr", type=float, default=0.0)
    ap.add_argument("--sigma-c-peak", type=float, default=0.05)
    ap.add_argument("--compute-backend", choices=["scipy", "scipy_bethe", "scipy_expm", "scipy_expm_multiply", "jax", "jax_auto", "jax_cpu", "jax_mps", "jax_cuda", "jax_padded", "jax_auto_padded", "jax_cpu_padded", "jax_mps_padded", "jax_cuda_padded", "jax_cpu_real_expm", "jax_mps_real_expm", "jax_mps_real_expm_padded"], default="scipy", help="Bloch propagation backend. SciPy eigensolver is the production reference; scipy_bethe uses an opt-in promoted Bethe/Schur strong-beam reduction; SciPy expm/expm_multiply, real-embedded JAX expm, and JAX eigensolver backends are experimental. *_padded JAX backends pad matrix dimensions into small buckets to reduce compilation churn.")
    ap.add_argument("--row-plan-cache", choices=["off", "exact", "readwrite", "rebuild"], default="off", help="Cache row geometry. exact is model-keyed; readwrite explicitly freezes the builder basis across moving-model evaluations.")
    ap.add_argument("--jax-batch-size", type=int, default=0, help="JAX matrix batch size. 0 chooses a conservative size from matrix dimension and device count.")
    ap.add_argument("--jax-matrix-bucket-size", type=int, default=0, help="Pad JAX matrix dimensions to this multiple. 0 keeps exact sizes unless a *_padded backend is selected, where 0 means 8.")
    ap.add_argument("--bethe-strong-count", type=int, default=48, help="Strong-beam block size for scipy_bethe. Values >= n_basis recover the dense SciPy matrix.")
    ap.add_argument("--bethe-variant", choices=["truncate", "schur0", "schur_target", "schur_mean"], default="schur_target", help="Strong/weak-beam reduction used by scipy_bethe.")
    ap.add_argument("--bethe-promote-denom", type=float, default=0.005, help="Promote weak beams whose diagonal denominator is this close to the Schur energy.")
    ap.add_argument("--bethe-denom-eps", type=float, default=1e-5, help="Small-denominator floor for scipy_bethe Schur corrections.")
    ap.add_argument("--profile-quadrature", choices=["uniform", "trapezoid", "gauss_legendre"], default="uniform", help="Rocking-profile Sg quadrature rule.")
    ap.add_argument("--profile-accumulation", choices=["sum", "mean"], default="sum", help="Accumulate rocking-profile samples as summed detector sub-exposures or legacy normalized means.")
    ap.add_argument("--free-fraction", type=float, default=0.10)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument(
        "--lambda-grid",
        default="1.0",
        help="Diagnostic q-exponent grid. Production workflows use the full Bloch/Born correction with lambda_q=1.0.",
    )
    ap.add_argument("--acceptance-metric", choices=["all_wR2", "work_wR2", "free_wR2", "all_R1"], default="all_wR2")
    ap.add_argument("--acceptance-min-delta", type=float, default=0.0, help="Required absolute improvement in the acceptance metric before exporting a moved model.")
    ap.add_argument("--free-wR2-tolerance", type=float, default=0.0, help="Maximum allowed free wR2 worsening unless --allow-free-worse is set.")
    ap.add_argument("--allow-free-worse", action="store_true")
    ap.add_argument("--early-stop-window", type=int, default=96, help="Stop a structure block after this many residual calls without a meaningful objective improvement; 0 disables.")
    ap.add_argument("--early-stop-min-evals", type=int, default=0, help="Minimum residual calls before early stopping; 0 uses an automatic parameter-count-aware floor.")
    ap.add_argument("--early-stop-min-improvement", type=float, default=1e-5, help="Minimum RMS improvement counted as progress for structure-block early stopping.")
    ap.add_argument("--early-stop-metric", choices=["total_rms", "data_rms"], default="total_rms", help="RMS trace column used by the structure-block plateau stop.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--make-maps", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_root = out / "cache"
    data = XDSData(Path(args.hkl), hkl_transform=args.hkl_transform)
    base_model = parse_model(Path(args.res))
    model_tag = "alternate_fractional_xyz" if args.alternate_hand else "original"
    model_tag = f"{model_tag};hkl_transform={data.hkl_transform_spec}"
    if args.alternate_hand:
        base_model = alternate_hand_model(base_model)
    target, base_sig = target_arrays(data, args.target_mode)
    target_indices = row_selection(data, target, not args.include_nonpositive, args.min_d_spacing, args.min_peak, 0, args.geometry_percentile)
    masks = split_masks(
        data.hkl,
        args.free_fraction,
        args.split_seed,
        symmetry_ops=base_model.get("symmetry_ops"),
    )
    selected_mask = np.zeros(len(target), dtype=bool)
    selected_mask[target_indices] = True
    masks = {k: v & selected_mask for k, v in masks.items()}
    nuisance = Nuisance(
        args.thickness,
        args.thickness_grad_frac,
        args.frame_offset,
        args.group_width_deg,
        args.detector_scale_bins,
        args.sigma_a,
        args.sigma_b_corr,
        args.sigma_c_peak,
        row_resolution_b=args.row_resolution_b,
    )
    sig = sigma_eff(data, target, base_sig, nuisance)
    objective_indices = choose_objective_rows(data, target_indices, masks, target, sig, args.objective_rows, args.objective_bins, args.split_seed)
    wght_a, wght_b = parse_wght(Path(args.res))
    block_schedule = [x.strip() for x in re.split(r"[,\s]+", args.block_schedule) if x.strip()]
    allowed_blocks = {"xyz", "adp_ueq", "xyz_adp_ueq"}
    unknown_blocks = [b for b in block_schedule if b not in allowed_blocks]
    if unknown_blocks:
        raise ValueError(f"unknown model-update block(s): {unknown_blocks}; allowed: {sorted(allowed_blocks)}")
    cfg = Config(
        n_basis=args.n_basis,
        n_points=args.n_points,
        sg_window=args.sg_window,
        sg_window_mode=args.sg_window_mode,
        sg_window_min=args.sg_window_min,
        sg_window_max=args.sg_window_max,
        sg_window_scale=args.sg_window_scale,
        sg_window_frame_floor=args.sg_window_frame_floor,
        thickness_A=args.thickness,
        thickness_grad_frac=args.thickness_grad_frac,
        thickness_points=args.thickness_points,
        topup_d_min=min(float(args.topup_d_min), float(args.min_d_spacing)) if args.min_d_spacing > 0 else float(args.topup_d_min),
        group_width_deg=args.group_width_deg,
        frame_offset=args.frame_offset,
        free_fraction=args.free_fraction,
        split_seed=args.split_seed,
        variance_a=args.sigma_a,
        variance_b_corr=args.sigma_b_corr,
        variance_c_peak=args.sigma_c_peak,
        compute_backend=args.compute_backend,
        jax_batch_size=args.jax_batch_size,
        jax_matrix_bucket_size=args.jax_matrix_bucket_size,
        bethe_strong_count=args.bethe_strong_count,
        bethe_variant=args.bethe_variant,
        bethe_promote_denom=args.bethe_promote_denom,
        bethe_denom_eps=args.bethe_denom_eps,
        profile_quadrature=args.profile_quadrature,
        profile_accumulation=args.profile_accumulation,
        sg_jitter_mode=args.sg_jitter_mode,
        sg_jitter_sigma0=args.sg_jitter_sigma0,
        sg_jitter_sigma_d_inv=args.sg_jitter_sigma_d_inv,
        sg_jitter_sigma_frame=args.sg_jitter_sigma_frame,
        sg_jitter_points=args.sg_jitter_points,
        sg_jitter_student_df=args.sg_jitter_student_df,
        sg_jitter_width_factor=args.sg_jitter_width_factor,
        absorption_mode=args.absorption_mode,
        absorption_mu=args.absorption_mu,
        absorption_frac=args.absorption_frac,
        absorption_mu_g2=args.absorption_mu_g2,
        absorption_mu_theta2=args.absorption_mu_theta2,
        absorption_tds_b=args.absorption_tds_b,
        ug_d_min=args.ug_d_min,
        row_plan_cache=args.row_plan_cache,
    )
    objective_cfg = copy.deepcopy(cfg)
    objective_cfg.group_width_deg = args.objective_group_width_deg
    trace = []
    block_results = []
    restraints = []
    if args.restraints == "start_geometry":
        restraints = build_start_restraints(
            base_model,
            args.atom_selection,
            args.atom_labels,
            args.bond_detection_tolerance,
            args.bond_detection_max_distance,
        )
    write_restraints_csv(out / "restraints.csv", restraints)
    cctbx_bridge = inspect_cctbx_bridge(Path(args.res), out)
    write_objective_rows(out / "objective_rows.csv", data, objective_indices, target, sig, masks)

    if args.scale_fit_split == "all":
        validation_fit_indices = np.asarray(target_indices, dtype=int)
    else:
        validation_fit_indices = np.asarray([i for i in target_indices if masks["work"][i]], dtype=int)
    start_eval = evaluate_model(data, base_model, cfg, target, sig, target_indices, masks, validation_fit_indices, wght_a, wght_b, cache_root, "start_validation", args.workers, args.chunksize, detector_scale_bins=args.detector_scale_bins, row_resolution_b=args.row_resolution_b)
    current_model = copy.deepcopy(base_model)
    current_eval = start_eval
    accepted = False
    acceptance_reason = "no model-update blocks were accepted"
    last_candidate_eval = start_eval

    for block_no, block_mode in enumerate(block_schedule, start=1):
        block_tag = f"block_{block_no:02d}_{block_mode}"
        try:
            params, x0 = parameter_layout(current_model, args.atom_selection, args.atom_labels, block_mode)
        except ValueError as exc:
            block_results.append({"block": block_no, "mode": block_mode, "skipped": True, "reason": str(exc)})
            continue
        has_xyz_params = any(p.get("kind", "xyz") == "xyz" for p in params)
        has_adp_params = any(p.get("kind") == "adp_ueq" for p in params)
        bounds = parameter_bounds(params, x0, args.max_shift_frac, args.adp_shift, args.adp_min, args.adp_max)
        absolute_steps = parameter_absolute_steps(
            params,
            xyz_step_frac=args.xyz_diff_step_frac,
            adp_step_a2=args.adp_diff_step_a2,
            legacy_step=args.diff_step,
        )
        eval_counter = {"n": 0}
        residual_cache = {"x": None, "value": None}
        block_scale_contract = None
        if args.objective_scale_mode == "frozen_block":
            anchor_pred = prediction_for_model(data, current_model, objective_cfg, objective_indices, cache_root, f"{block_tag}_scale_anchor", args.workers, args.chunksize)
            anchor_full = np.full(len(target), np.nan, dtype=float)
            anchor_full[anchor_pred["indices"]] = anchor_pred["dyn"]
            anchor_full, _, _ = apply_row_resolution_to_prediction(data, anchor_full, args.row_resolution_b)
            block_scale_contract = fit_scale_contract(
                data,
                target,
                sig,
                anchor_full,
                objective_cfg,
                objective_indices,
                wght_a,
                wght_b,
                detector_scale_bins=args.detector_scale_bins,
            )
            block_scale_contract["mode"] = "frozen_block"
            block_scale_contract["anchor_prediction_seconds"] = float(anchor_pred.get("seconds", 0.0))
        early_stop_state = {
            "enabled": int(args.early_stop_window) > 0,
            "window": max(1, int(args.early_stop_window)),
            "min_evals": early_stop_min_evals(args, params),
            "min_improvement": max(0.0, float(args.early_stop_min_improvement)),
            "best_metric": float("inf"),
            "best_eval": 0,
            "best_x": np.asarray(x0, dtype=float).copy(),
            "initial_metric": float("nan"),
        }

        def residuals(
            x,
            *,
            block_no=block_no,
            block_mode=block_mode,
            block_tag=block_tag,
            params=params,
            x0=x0,
            current_model=current_model,
            has_xyz_params=has_xyz_params,
            has_adp_params=has_adp_params,
        ):
            eval_counter["n"] += 1
            model = model_from_vector(current_model, params, x)
            pred = prediction_for_model(data, model, objective_cfg, objective_indices, cache_root, f"{block_tag}_moving", args.workers, args.chunksize)
            _, scaled, scales, _, scale_count = scaled_prediction(
                data,
                target,
                sig,
                pred,
                objective_cfg,
                objective_indices,
                wght_a,
                wght_b,
                detector_scale_bins=args.detector_scale_bins,
                scale_contract=block_scale_contract,
                row_resolution_b=args.row_resolution_b,
            )
            data_resid = residual_vector(data, target, sig, scaled, objective_indices, wght_a, wght_b)
            prior = parameter_prior(params, x, x0, args.xyz_restraint_sigma, args.adp_restraint_sigma)
            geom_resid = bond_restraint_residuals(model, restraints, args.bond_restraint_sigma) if has_xyz_params else np.zeros(0, dtype=float)
            adp_resid = adp_similarity_residuals(model, restraints, args.adp_similarity_sigma) if has_adp_params else np.zeros(0, dtype=float)
            total = np.concatenate([data_resid, prior, geom_resid, adp_resid])
            rms = float(np.sqrt(np.mean(total * total))) if len(total) else float("nan")
            data_rms = float(np.sqrt(np.mean(data_resid * data_resid))) if len(data_resid) else float("nan")
            max_shift = float(np.max(np.abs(x - x0))) if len(x) else 0.0
            trace.append(
                {
                    "block": block_no,
                    "mode": block_mode,
                    "eval": eval_counter["n"],
                    "rms_residual": rms,
                    "data_rms": data_rms,
                    "prior_rms": float(np.sqrt(np.mean(prior * prior))) if len(prior) else 0.0,
                    "geometry_rms": float(np.sqrt(np.mean(geom_resid * geom_resid))) if len(geom_resid) else 0.0,
                    "adp_similarity_rms": float(np.sqrt(np.mean(adp_resid * adp_resid))) if len(adp_resid) else 0.0,
                    "max_abs_shift": max_shift,
                    "scale_count": int(scale_count),
                    "objective_scale_mode": args.objective_scale_mode,
                    "objective_group_width_deg": float(args.objective_group_width_deg),
                }
            )
            print(
                f"block {block_no:02d} {block_mode} eval {eval_counter['n']:03d}: "
                f"data_rms={trace[-1]['data_rms']:.4f} total_rms={rms:.4f} max_shift={max_shift:.5f}",
                flush=True,
            )
            if early_stop_state["enabled"]:
                stop_metric = rms if args.early_stop_metric == "total_rms" else data_rms
                if math.isfinite(stop_metric):
                    if not math.isfinite(float(early_stop_state["initial_metric"])):
                        early_stop_state["initial_metric"] = float(stop_metric)
                    if stop_metric < float(early_stop_state["best_metric"]) - float(early_stop_state["min_improvement"]):
                        early_stop_state["best_metric"] = float(stop_metric)
                        early_stop_state["best_eval"] = int(eval_counter["n"])
                        early_stop_state["best_x"] = np.asarray(x, dtype=float).copy()
                    stale = int(eval_counter["n"]) - int(early_stop_state["best_eval"])
                    if int(eval_counter["n"]) >= int(early_stop_state["min_evals"]) and stale >= int(early_stop_state["window"]):
                        reason = (
                            f"{args.early_stop_metric} plateau for {stale} residual calls "
                            f"after eval {early_stop_state['best_eval']} "
                            f"(best {float(early_stop_state['best_metric']):.6g})"
                        )
                        raise EarlyStopOptimization(
                            reason,
                            early_stop_state["best_x"],
                            float(early_stop_state["best_metric"]),
                            int(early_stop_state["best_eval"]),
                        )
            residual_cache["x"] = np.asarray(x, dtype=float).copy()
            residual_cache["value"] = np.asarray(total, dtype=float).copy()
            return total

        def jacobian(x):
            cached = residual_cache.get("value")
            cached_x = residual_cache.get("x")
            if cached_x is None or not np.array_equal(np.asarray(cached_x, dtype=float), np.asarray(x, dtype=float)):
                cached = None
            return absolute_finite_difference_jacobian(
                residuals,
                x,
                bounds,
                absolute_steps,
                f0=cached,
            )

        x_scale = []
        for p in params:
            x_scale.append(max(float(args.adp_shift if p.get("kind") == "adp_ueq" else args.max_shift_frac), 1e-5))
        try:
            result = least_squares(
                residuals,
                x0,
                bounds=bounds,
                jac=jacobian,
                loss="linear",
                max_nfev=args.max_nfev,
                verbose=1,
                x_scale=np.asarray(x_scale, dtype=float),
            )
        except EarlyStopOptimization as exc:
            result = make_early_stop_result(exc, eval_counter["n"])
            print(f"block {block_no:02d} {block_mode}: {result.message}", flush=True)
        candidate_model = model_from_vector(current_model, params, result.x)
        candidate_eval = evaluate_model(data, candidate_model, cfg, target, sig, target_indices, masks, validation_fit_indices, wght_a, wght_b, cache_root, f"{block_tag}_candidate_validation", args.workers, args.chunksize, detector_scale_bins=args.detector_scale_bins, row_resolution_b=args.row_resolution_b)
        last_candidate_eval = candidate_eval

        start_metric = validation_score(current_eval, args.acceptance_metric)
        candidate_metric = validation_score(candidate_eval, args.acceptance_metric)
        free_start = validation_score(current_eval, "free_wR2")
        free_candidate = validation_score(candidate_eval, "free_wR2")
        metric_ok = candidate_metric <= start_metric - float(args.acceptance_min_delta)
        free_ok = args.allow_free_worse or args.acceptance_metric == "free_wR2" or free_candidate <= free_start + float(args.free_wR2_tolerance)
        block_accepted = bool(metric_ok and free_ok)
        block_reason = (
            f"accepted: {args.acceptance_metric} {start_metric:.6g} -> {candidate_metric:.6g}"
            if block_accepted
            else f"rejected: {args.acceptance_metric} {start_metric:.6g} -> {candidate_metric:.6g}; free_wR2 {free_start:.6g} -> {free_candidate:.6g}; metric_ok={metric_ok}; free_ok={free_ok}"
        )
        candidate_res = out / f"{block_tag}_candidate.res"
        accepted_res_block = out / f"{block_tag}_accepted.res"
        update_res_coordinates(Path(args.res), candidate_res, candidate_model)
        update_res_coordinates(Path(args.res), accepted_res_block, candidate_model if block_accepted else current_model)
        write_parameter_csv(out / f"{block_tag}_candidate_parameters.csv", params, x0, result.x)
        write_parameter_csv(out / f"{block_tag}_accepted_parameters.csv", params, x0, result.x if block_accepted else x0)
        block_record = {
            "block": block_no,
            "mode": block_mode,
            "skipped": False,
            "parameter_count": int(len(params)),
            "xyz_parameter_count": int(sum(1 for p in params if p.get("kind", "xyz") == "xyz")),
            "adp_parameter_count": int(sum(1 for p in params if p.get("kind") == "adp_ueq")),
            "finite_difference": {
                "scheme": "bounded_two_point_absolute",
                "xyz_step_frac": float(args.diff_step if args.diff_step is not None else args.xyz_diff_step_frac),
                "adp_step_a2": float(args.diff_step if args.diff_step is not None else args.adp_diff_step_a2),
                "legacy_diff_step_shorthand_used": bool(args.diff_step is not None),
                "minimum_step": float(np.min(absolute_steps)),
                "maximum_step": float(np.max(absolute_steps)),
            },
            "accepted": block_accepted,
            "reason": block_reason,
            "objective_scale_mode": args.objective_scale_mode,
            "objective_scale_count": int(block_scale_contract["scale_count"]) if block_scale_contract is not None else None,
            "objective_scale_anchor_seconds": float(block_scale_contract.get("anchor_prediction_seconds", 0.0)) if block_scale_contract is not None else None,
            "least_squares": {
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "cost": float(result.cost),
                "optimality": float(result.optimality),
                "nfev": int(result.nfev),
                "early_stop": bool(getattr(result, "early_stop", False)),
                "early_stop_reason": str(getattr(result, "early_stop_reason", "")),
                "early_stop_eval": int(getattr(result, "early_stop_eval", 0)),
                "early_stop_metric": float(getattr(result, "early_stop_metric", float("nan"))),
            },
            "current_metric": start_metric,
            "candidate_metric": candidate_metric,
            "free_wR2_start": free_start,
            "free_wR2_candidate": free_candidate,
            "candidate_res": str(candidate_res),
            "accepted_res": str(accepted_res_block),
        }
        block_results.append(block_record)
        print(f"block {block_no:02d} {block_mode}: {block_reason}", flush=True)
        if block_accepted:
            current_model = candidate_model
            current_eval = candidate_eval
            accepted = True
            acceptance_reason = block_reason

    accepted_model = current_model
    accepted_eval = current_eval
    candidate_eval = last_candidate_eval

    accepted_res = out / "accepted_model.res"
    updated_res = out / "bloch_updated_model.res"
    update_res_coordinates(Path(args.res), accepted_res, accepted_model)
    update_res_coordinates(Path(args.res), updated_res, accepted_model)
    write_model_delta_csv(out / "model_delta.csv", base_model, accepted_model)
    # Backward-compatible name for quick inspection scripts.
    write_model_delta_csv(out / "coordinate_shifts.csv", base_model, accepted_model)
    with (out / "trace.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "block",
                "mode",
                "eval",
                "rms_residual",
                "data_rms",
                "prior_rms",
                "geometry_rms",
                "adp_similarity_rms",
                "max_abs_shift",
                "scale_count",
                "objective_scale_mode",
                "objective_group_width_deg",
            ],
        )
        w.writeheader()
        w.writerows(trace)

    # Export a corrected-HKL endpoint only from the validation-accepted model.
    # If the coordinate step overfits, this intentionally falls back to the
    # starting model rather than exporting a worse dynamic model.
    accepted_pred = prediction_for_model(data, accepted_model, cfg, target_indices, cache_root, "accepted_endpoint", args.workers, args.chunksize)
    accepted_best = evaluate_trial(data, target, base_sig, accepted_pred, target_indices, masks, nuisance, wght_a, wght_b, scale_fit_split=args.scale_fit_split)
    endpoints = []
    for lam in parse_float_grid(args.lambda_grid):
        ep_dir = out / f"endpoint_lam_{str(lam).replace('.', 'p')}"
        endpoints.append(export_corrected_endpoint(ep_dir, data, target, accepted_best["arrays"]["sigma_eff"], accepted_best, masks, updated_res, lam, "shelx_fixed", "median-iobs", args.make_maps))
    endpoints = sorted(endpoints, key=lambda r: (r["shelxl"].get("wR2_all", 999), r["shelxl"].get("R1_all", 999)))
    baseline = copy_baseline(out / "baseline_exti", Path(args.res), Path(args.baseline_hkl)) if Path(args.baseline_hkl).exists() else {}

    summary = {
        "inputs": {
            "hkl": str(Path(args.hkl).resolve()),
            "hkl_sha256": sha256(Path(args.hkl)),
            "res": str(Path(args.res).resolve()),
            "res_sha256": sha256(Path(args.res)),
            "model_tag": model_tag,
        },
        "config": vars(args),
        "nuisance": asdict(nuisance),
        "objective_config": {
            "group_width_deg": float(objective_cfg.group_width_deg),
            "validation_group_width_deg": float(cfg.group_width_deg),
            "scale_mode": str(args.objective_scale_mode),
            "scale_mode_note": "frozen_block refreshes scales once per model-update block, then keeps them fixed for every coordinate/ADP finite-difference evaluation.",
            "scale_fit_split": str(args.scale_fit_split),
        },
        "restraints": {
            "mode": args.restraints,
            "count": int(len(restraints)),
            "bond_restraint_sigma": float(args.bond_restraint_sigma),
            "adp_similarity_sigma": float(args.adp_similarity_sigma),
            "source": "start model inferred covalent graph",
            "path": str(out / "restraints.csv"),
            "cctbx_bridge": cctbx_bridge,
        },
        "row_counts": {
            "target_rows": int(len(target_indices)),
            "objective_rows": int(len(objective_indices)),
            "work_rows": int(np.count_nonzero(masks["work"])),
            "free_rows": int(np.count_nonzero(masks["free"])),
            "max_block_parameter_count": int(max([b.get("parameter_count", 0) for b in block_results] + [0])),
        },
        "blocks": block_results,
        "acceptance": {
            "accepted": accepted,
            "metric": args.acceptance_metric,
            "start_metric": validation_score(start_eval, args.acceptance_metric),
            "accepted_metric": validation_score(accepted_eval, args.acceptance_metric),
            "acceptance_min_delta": float(args.acceptance_min_delta),
            "free_wR2_start": validation_score(start_eval, "free_wR2"),
            "free_wR2_accepted": validation_score(accepted_eval, "free_wR2"),
            "free_wR2_tolerance": float(args.free_wR2_tolerance),
            "allow_free_worse": bool(args.allow_free_worse),
            "reason": acceptance_reason,
        },
        "start_eval": {k: v for k, v in start_eval.items() if k != "arrays"},
        "candidate_eval": {k: v for k, v in candidate_eval.items() if k != "arrays"},
        "accepted_eval": {k: v for k, v in accepted_eval.items() if k != "arrays"},
        "endpoint_results_sorted": endpoints,
        "best_endpoint": endpoints[0] if endpoints else {},
        "baseline_exti": baseline,
        "wall_seconds": time.perf_counter() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

    lines = [
        "# Native Row-Parameter Regularization",
        "",
        f"Objective rows: `{len(objective_indices)}` of `{len(target_indices)}` selected rows. Blocks: `{','.join(block_schedule)}`. N/Q: `{args.n_basis}/{args.n_points}`.",
        "",
        f"This is a finite-difference coordinate regularization against `sqrt(w) * (Iobs - Idyn_scaled)`, with `{args.objective_group_width_deg:.2f}` degree objective scale groups. Objective scale mode is `{args.objective_scale_mode}`. In `frozen_block`, scales are fitted from the block-start model and held fixed during every coordinate/ADP finite-difference evaluation, then refreshed before the next block. Broad validation and corrected-HKL export use `{args.group_width_deg:.2f}` degree groups, `{args.detector_scale_bins}x{args.detector_scale_bins}` detector bins, and `{args.scale_fit_split}` rows for fitting nuisance scales.",
        f"Restraints: `{args.restraints}` with `{len(restraints)}` inferred bond restraints, bond sigma `{args.bond_restraint_sigma:.3f} A`, and bonded-Ueq sigma `{args.adp_similarity_sigma:.4f}`.",
        f"cctbx/smtbx bridge: parsed=`{cctbx_bridge.get('parsed')}`, constraints=`{cctbx_bridge.get('constraints_count')}`. The bridge uses `{cctbx_bridge.get('view')}` as a restraint-only SHELX view with sanitized SFAC rows.",
        "",
        f"Validation-gated acceptance: `{acceptance_reason}`. The corrected-HKL endpoint below was generated from the accepted model.",
        "",
        "## Regularization Blocks",
        "",
        "| block | mode | params | accepted | validation decision |",
        "|---:|---|---:|---|---|",
    ]
    for br in block_results:
        if br.get("skipped"):
            lines.append(f"| {br['block']} | {br['mode']} | 0 | skipped | {br['reason']} |")
        else:
            lines.append(f"| {br['block']} | {br['mode']} | {br['parameter_count']} | {str(br['accepted']).lower()} | {br['reason']} |")
    lines.extend([
        "",
        "## Native Validation",
        "",
        "| split | start R1all | candidate R1all | accepted R1all | start wR2all | candidate wR2all | accepted wR2all | start CC | candidate CC | accepted CC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for split in ("work", "free", "all"):
        s = start_eval["splits"][split]
        c = candidate_eval["splits"][split]
        a = accepted_eval["splits"][split]
        lines.append(
            f"| {split} | {100*s['R1_all']:.2f} | {100*c['R1_all']:.2f} | {100*a['R1_all']:.2f} | "
            f"{100*s['wR2_all']:.2f} | {100*c['wR2_all']:.2f} | {100*a['wR2_all']:.2f} | "
            f"{s['CC']:.4f} | {c['CC']:.4f} | {a['CC']:.4f} |"
        )
    lines.extend([
        "",
        "## Corrected-HKL Endpoint From Updated Model",
        "",
        "| lambda_q | R1gt % | R1all % | wR2all % | GooF | Ngt/Nall |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for ep in endpoints:
        sh = ep["shelxl"]
        lines.append(f"| {ep['lambda_q']:.2f} | {100*sh.get('R1_gt', float('nan')):.2f} | {100*sh.get('R1_all', float('nan')):.2f} | {100*sh.get('wR2_all', float('nan')):.2f} | {sh.get('GooF', float('nan')):.3f} | {sh.get('N_gt', '')}/{sh.get('N_all', '')} |")
    if baseline:
        lines.extend([
            "",
            f"Baseline EXTI rerun: `R1gt={100*baseline.get('R1_gt', float('nan')):.2f}%`, `R1all={100*baseline.get('R1_all', float('nan')):.2f}%`, `wR2all={100*baseline.get('wR2_all', float('nan')):.2f}%`, `GooF={baseline.get('GooF', float('nan')):.3f}`.",
        ])
    lines.extend([
        "",
        "## Files",
        "",
        f"Accepted RES: `{updated_res}`",
        f"Accepted model delta: `{out / 'model_delta.csv'}`",
        f"Restraints: `{out / 'restraints.csv'}`",
        f"Objective rows: `{out / 'objective_rows.csv'}`",
        f"Trace: `{out / 'trace.csv'}`",
    ])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "out_dir": str(out.resolve()),
                "report": str((out / "REPORT.md").resolve()),
                "accepted": accepted,
                "acceptance_reason": acceptance_reason,
                "best_endpoint": summary["best_endpoint"],
            },
            indent=2,
            allow_nan=True,
        )
    )


if __name__ == "__main__":
    main()
