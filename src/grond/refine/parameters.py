"""Native coordinate/Ueq parameterization and bounded finite differences. Special-site XYZ remains frozen until a symmetry-tangent parameterization exists; scalar Ueq is not a full ADP refinement."""

from __future__ import annotations

import copy
import math
import re

import numpy as np

from grond.legacy.engine.bloch_engine import (
    model_symmetry_ops,
    shelx_u_cart_matrix,
    unique_symmetry_mates,
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


def special_position_coordinate_policy(model, selection: str, label_text: str, parameter_mode: str = "xyz"):
    """Describe conservative XYZ freezing until a site-tangent basis exists."""

    labels = {x.strip() for x in re.split(r"[,\s]+", label_text or "") if x.strip()}
    frozen = []
    if parameter_mode in {"xyz", "xyz_adp_ueq"}:
        order = len(model_symmetry_ops(model))
        for ai, atom in enumerate(model["atoms"]):
            if not atom_is_refinable(atom, selection, labels):
                continue
            multiplicity = len(list(unique_symmetry_mates(atom, model)))
            if multiplicity < order:
                frozen.append({"atom_index": ai, "atom": atom["name"], "site_multiplicity": multiplicity, "symmetry_operation_count": order})
    return {
        "policy": "freeze_special_site_xyz",
        "frozen_atoms": frozen,
        "warning": (
            "Special-position XYZ coordinates are held fixed in native updates because "
            "a symmetry-constrained tangent parameterization is not implemented. "
            "Scalar Ueq updates remain available."
        ) if frozen else "",
    }


def parameter_layout(model, selection: str, label_text: str, parameter_mode: str = "xyz"):
    labels = {x.strip() for x in re.split(r"[,\s]+", label_text or "") if x.strip()}
    frozen_xyz = {item["atom_index"] for item in special_position_coordinate_policy(model, selection, label_text, parameter_mode)["frozen_atoms"]}
    params = []
    x0 = []
    for ai, atom in enumerate(model["atoms"]):
        if not atom_is_refinable(atom, selection, labels):
            continue
        if parameter_mode in {"xyz", "xyz_adp_ueq"} and ai not in frozen_xyz:
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
        reason = "; special-position XYZ coordinates are held fixed" if frozen_xyz else ""
        raise ValueError(f"No parameters selected for mode {parameter_mode!r}{reason}")
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
