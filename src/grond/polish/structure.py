# ruff: noqa: E741
# h, k, l retain standard crystallographic Miller-index notation.
"""SHELX model parameterization and the differentiable Fourier-potential table U_g of each crystal.

By default, refined atoms are the non-hydrogen atoms whose coordinate and displacement tokens are plain numbers.  Each
moves only along its site-symmetry tangent space (a general position has three coordinate and six Uij
directions; a special position fewer), so every space group is handled without special cases.  Tokens that
SHELX encodes as fixed or free-variable values (10*m + p) keep their value.  Hydrogen atoms ride on the
preceding non-hydrogen atom: they translate with it, and a negative SHELX Uiso keeps its multiple of the
parent Ueq. The explicit ``independent`` hydrogen mode instead gives unconstrained
H atoms their own coordinate and ADP parameters. Occupancies are fixed.

For one crystal U_g = gamma / (pi V) * sum_atoms occ f(s) sum_unique_mates T(R^T g) exp(2 pi i g.(R x + t)),
exactly the engine's ``fcalc_res_grid`` with the crystal's own cell.  The Jacobian dU_g/dp is computed by
torch forward-mode autograd (dual numbers), one atom and parameter direction at a time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from grond.bloch.constants import SITE_SYMMETRY_TOLERANCE_FRACTIONAL
from grond.bloch.model import (
    SHELX_NON_ATOM_RECORDS,
    cell_orthogonalization_matrix,
    inv_d2_from_hkl_array,
    is_shelx_q_peak,
    join_shelx_continuations,
    model_symmetry_ops,
    parse_res,
    reciprocal_lengths,
    sfac_value_array,
    shelx_encoded as _encoded,
    shelx_u_cart_matrix,
)

U6_KEYS = ("u11", "u22", "u33", "u23", "u13", "u12")
HYDROGEN_ELEMENTS = {"H", "D"}
# Instructions that tie coordinates or displacement parameters together; the polish cannot represent them and
# refuses the model.  (SUMP only ties occupancies, which the polish keeps fixed.)
UNSUPPORTED_CONSTRAINTS = {"EXYZ", "EADP", "FRAG", "FEND"}
# Restraints are not terms of the dynamical objective; they are listed in the summary and stay in the SHELXL
# re-refinement's instruction file.
RESTRAINTS = {"DFIX", "DANG", "SADI", "SAME", "FLAT", "CHIV", "DELU", "SIMU", "RIGU", "ISOR", "BUMP", "NCSY"}


def raw_atom_tokens(res_path: Path) -> dict[str, list[float]]:
    """Atom label -> raw numeric tokens [x, y, z, sof, adp...] exactly as written in the RES."""

    out: dict[str, list[float]] = {}
    lines = join_shelx_continuations(Path(res_path).read_text(errors="replace").splitlines())
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.upper().startswith("REM"):
            continue
        tokens = stripped.split()
        key = tokens[0].upper()
        if key == "HKLF":
            break
        if key in SHELX_NON_ATOM_RECORDS or key in {"SFAC", "UNIT", "END"}:
            continue
        if len(tokens) >= 7 and tokens[1].lstrip("-").isdigit() and not is_shelx_q_peak(tokens[0]):
            out[tokens[0]] = [float(v) for v in tokens[2:]]
    return out


def instruction_audit(res_path: Path) -> dict[str, list[str]]:
    """Constraint and restraint instructions present in the RES.

    A non-hydrogen atom inside an AFIX group with a nonzero code is part of a SHELXL rigid or idealised group
    (AFIX 66, 106, 6, 9, ...), which the free-atom parameterization cannot represent.
    """

    constraints, restraints, hydrogen_constraints = [], [], []
    elements: list[str] = []
    afix = 0
    for line in join_shelx_continuations(Path(res_path).read_text(errors="replace").splitlines()):
        tokens = line.strip().split()
        if not tokens:
            continue
        key = tokens[0].upper()
        if key == "HKLF":
            break
        if key == "SFAC" and len(tokens) > 1:
            # GROND models carry one explicit electron SFAC record per element (parse_res indexes them in order).
            elements.append(tokens[1].upper())
        elif key == "AFIX":
            try:
                afix = int(float(tokens[1])) if len(tokens) > 1 else 0
            except ValueError:
                afix = 0
        elif key == "HFIX" and len(tokens) > 1 and float(tokens[1]) != 0:
            hydrogen_constraints.append(line.strip())
        elif key in UNSUPPORTED_CONSTRAINTS:
            constraints.append(line.strip())
        elif key in RESTRAINTS:
            restraints.append(line.strip())
        elif (
            afix
            and len(tokens) >= 7
            and tokens[1].lstrip("-").isdigit()
            and key not in SHELX_NON_ATOM_RECORDS
            and not is_shelx_q_peak(tokens[0])
        ):
            index = int(tokens[1]) - 1
            element = elements[index] if 0 <= index < len(elements) else ""
            if element not in HYDROGEN_ELEMENTS:
                constraints.append(f"AFIX {afix} on non-hydrogen atom {tokens[0]}")
            else:
                hydrogen_constraints.append(f"AFIX {afix} on hydrogen atom {tokens[0]}")
    return {"unsupported_constraints": constraints, "restraints": restraints, "hydrogen_constraints": hydrogen_constraints}



def _null_space(rows: np.ndarray, dim: int, tol: float = 1e-8) -> np.ndarray:
    """Return an orthonormal column basis of the null space of ``rows``; the identity when there are no rows."""

    if rows.size == 0:
        return np.eye(dim)
    _u, s, vt = np.linalg.svd(rows)
    rank = int(np.sum(s > tol * max(1.0, float(s[0]) if len(s) else 1.0)))
    return vt[rank:].T.copy()


def _u6_to_matrix(u6: np.ndarray) -> np.ndarray:
    """Return the symmetric 3x3 U matrix of six components in U6_KEYS order (u11, u22, u33, u23, u13, u12)."""

    u11, u22, u33, u23, u13, u12 = u6
    return np.asarray([[u11, u12, u13], [u12, u22, u23], [u13, u23, u33]], dtype=float)


def _matrix_to_u6(m: np.ndarray) -> np.ndarray:
    """Return the six U components of a symmetric 3x3 matrix in U6_KEYS order."""

    return np.asarray([m[0, 0], m[1, 1], m[2, 2], m[1, 2], m[0, 2], m[0, 1]], dtype=float)


@dataclass
class AtomState:
    """Parameter state of one atom: start values, symmetry mates, role and the directions it may move along.

    ``role`` is refined (own parameters along xyz_basis and adp_basis, from param_start), riding (moves with ``parent``
    and/or takes uiso_multiplier times the Ueq of ``ueq_parent``) or fixed. xyz_basis columns are fractional directions
    of unit Cartesian length (1 Å); u0 holds six Uij or [Uiso] in Å².
    """

    index: int
    name: str
    element: str
    sfac_index: int
    occ: float
    xyz0: np.ndarray
    adp_type: str
    u0: np.ndarray  # six U components (anis) or [Uiso]
    mate_ops: list[int]
    role: str  # refined | riding | fixed
    xyz_basis: np.ndarray = field(default_factory=lambda: np.zeros((3, 0)))
    adp_basis: np.ndarray = field(default_factory=lambda: np.zeros((6, 0)))
    param_start: int = 0
    parent: int | None = None  # position parent of a riding atom
    ueq_parent: int | None = None
    uiso_multiplier: float | None = None
    frozen_tokens: list[str] = field(default_factory=list)

    @property
    def n_params(self) -> int:
        """Number of this atom's own parameters (coordinate plus displacement directions)."""

        return int(self.xyz_basis.shape[1] + self.adp_basis.shape[1])


class StructureModel:
    """The polished SHELX model: parameter layout, per-atom states and export back to SHELX values."""

    def __init__(self, res_path: str | Path, *, hydrogen_mode: str = "translate"):
        """Parse the start .res, audit its instructions and give each atom its role and parameter slice.

        Raises ValueError for constraints the polish cannot represent (EXYZ, EADP, FRAG, AFIX groups on non-hydrogen
        atoms) and for repeated atom names.
        """

        if hydrogen_mode not in {"translate", "independent"}:
            raise ValueError(f"unknown hydrogen mode {hydrogen_mode!r}; use translate or independent")
        self.hydrogen_mode = hydrogen_mode
        self.res_path = Path(res_path).expanduser().resolve()
        audit = instruction_audit(self.res_path)
        if hydrogen_mode == "independent" and audit["hydrogen_constraints"]:
            raise ValueError("independent hydrogen refinement cannot retain AFIX/HFIX constraints: " + "; ".join(audit["hydrogen_constraints"]))
        if audit["unsupported_constraints"]:
            raise ValueError(
                "dynamical polish cannot represent these SHELX constraints: "
                + "; ".join(audit["unsupported_constraints"])
                + ". Refine without them or keep the SHELXL endpoint."
            )
        self.instructions = audit
        self.model = parse_res(self.res_path)
        names = [str(atom["name"]) for atom in self.model["atoms"]]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            # Tokens and polished values are matched to atoms by name (RESI residues may repeat names).
            raise ValueError(f"dynamical polish needs unique atom names; {res_path} repeats {', '.join(duplicates)}")
        self.cell = tuple(float(v) for v in self.model["cell"])
        self.ops = model_symmetry_ops(self.model)
        tokens = raw_atom_tokens(self.res_path)
        orth = cell_orthogonalization_matrix(self.cell)
        self._orth = orth
        self._frac_from_cart = np.linalg.inv(orth)
        self.atoms: list[AtomState] = []
        last_non_h: int | None = None
        last_ueq: int | None = None
        for index, atom in enumerate(self.model["atoms"]):
            element = str(atom.get("element", "")).upper()
            is_h = element in HYDROGEN_ELEMENTS
            raw = tokens.get(str(atom["name"]))
            if raw is None:
                raise ValueError(f"RES atom {atom['name']} was parsed but its tokens could not be re-read")
            xyz0 = np.asarray([float(atom["x"]), float(atom["y"]), float(atom["z"])], dtype=float)
            if atom.get("adp_type") == "anis":
                u0 = np.asarray([float(atom[k]) for k in U6_KEYS], dtype=float)
            else:
                u0 = np.asarray([float(atom.get("uiso", 0.05))], dtype=float)
            state = AtomState(
                index=index,
                name=str(atom["name"]),
                element=element,
                sfac_index=int(atom["sfac_index"]),
                occ=float(atom["occ"]),
                xyz0=xyz0,
                adp_type=str(atom.get("adp_type", "iso")),
                u0=u0,
                mate_ops=self._unique_mate_ops(xyz0),
                role="fixed",
            )
            if state.adp_type == "iso" and len(raw) >= 5 and not _encoded(raw[4]) and float(raw[4]) < 0.0 and last_ueq is not None:
                # A negative SHELX Uiso is |U| x Ueq of the preceding atom (parse_res semantics, for H and non-H).
                state.ueq_parent = last_ueq
                state.uiso_multiplier = abs(float(raw[4]))
            if is_h and hydrogen_mode == "independent" and state.adp_type == "iso" and float(raw[4]) < 0.0:
                raise ValueError(f"independent hydrogen refinement needs an explicit non-riding Uiso for {state.name}")
            if is_h and hydrogen_mode == "translate":
                state.parent = last_non_h
            else:
                self._set_tangent_space(state, raw)
            self.atoms.append(state)
            if not is_h:
                last_non_h = index
                if state.adp_type == "anis" or float(u0[0]) > 0.0:
                    last_ueq = index
            elif state.adp_type == "anis":
                last_ueq = index
        start = 0
        for state in self.atoms:
            if state.role == "refined":
                state.param_start = start
                start += state.n_params
        self.n_params = start
        for state in self.atoms:
            if state.role == "refined":
                continue
            parent = self.atoms[state.parent] if state.parent is not None else None
            moves = state.element in HYDROGEN_ELEMENTS and parent is not None and parent.role == "refined"
            if moves or self._rides_varying_u(state):
                state.role = "riding"
        self._ueq_coef = self._ueq_coefficients()

    def _rides_varying_u(self, state: AtomState) -> bool:
        """True when a riding-U atom's parent Ueq changes with the parameters (possibly through a riding chain)."""

        if not state.uiso_multiplier or state.ueq_parent is None:
            return False
        parent = self.atoms[state.ueq_parent]
        return (parent.role == "refined" and parent.adp_basis.shape[1] > 0) or self._rides_varying_u(parent)

    # ------------------------------------------------------------------ layout
    def _unique_mate_ops(self, xyz: np.ndarray) -> list[int]:
        """Return the indices of the symmetry operations that map ``xyz`` to distinct sites."""

        seen: list[np.ndarray] = []
        keep: list[int] = []
        for i, (rot, trans) in enumerate(self.ops):
            position = np.mod(rot @ xyz + trans, 1.0)
            if any(
                np.max(np.abs((position - prior) - np.rint(position - prior))) <= SITE_SYMMETRY_TOLERANCE_FRACTIONAL
                for prior in seen
            ):
                continue
            seen.append(position)
            keep.append(i)
        return keep

    def _stabilizer(self, xyz: np.ndarray) -> list[np.ndarray]:
        """Return the non-identity rotations of the operations that map ``xyz`` onto itself (its site symmetry)."""

        out = []
        for rot, trans in self.ops:
            delta = rot @ xyz + trans - xyz
            if np.max(np.abs(delta - np.rint(delta))) <= SITE_SYMMETRY_TOLERANCE_FRACTIONAL and not np.allclose(rot, np.eye(3)):
                out.append(np.asarray(rot, dtype=float))
        return out

    def _set_tangent_space(self, state: AtomState, raw: list[float]) -> None:
        """Set an atom's coordinate and ADP directions allowed by its site symmetry and its SHELX tokens.

        Coordinate directions are an orthonormal Cartesian basis (1 unit = 1 Å) expressed in fractional coordinates.
        Tokens with the fixed/free-variable encoding and a riding (negative) Uiso stay fixed. The role becomes refined
        when any direction remains, else fixed.
        """

        stabilizer = self._stabilizer(state.xyz0)
        rows = [rot - np.eye(3) for rot in stabilizer]
        for axis, name in enumerate(("x", "y", "z")):
            if _encoded(raw[axis]):
                row = np.zeros((1, 3))
                row[0, axis] = 1.0
                rows.append(row)
                state.frozen_tokens.append(name)
        # Allowed fractional displacements, re-expressed as an orthonormal Cartesian basis (1 unit = 1 A).
        frac_null = _null_space(np.vstack(rows) if rows else np.zeros((0, 3)), 3)
        if frac_null.shape[1]:
            cart, _r = np.linalg.qr(self._orth @ frac_null)
            state.xyz_basis = self._frac_from_cart @ cart
        else:
            state.xyz_basis = np.zeros((3, 0))
        adp_tokens = raw[4:]
        if state.adp_type == "anis":
            recip = np.diag(reciprocal_lengths(self.cell))
            inv = np.diag(1.0 / np.diag(recip))
            maps = []
            for rot in stabilizer:
                cols = []
                for k in range(6):
                    unit = np.zeros(6)
                    unit[k] = 1.0
                    beta = recip @ _u6_to_matrix(unit) @ recip
                    cols.append(_matrix_to_u6(inv @ (rot @ beta @ rot.T) @ inv) - unit)
                maps.append(np.asarray(cols).T)
            for k, key in enumerate(("u11", "u22", "u33", "u23", "u13", "u12")):
                if k < len(adp_tokens) and _encoded(adp_tokens[k]):
                    row = np.zeros((1, 6))
                    row[0, k] = 1.0
                    maps.append(row)
                    state.frozen_tokens.append(key)
            state.adp_basis = _null_space(np.vstack(maps) if maps else np.zeros((0, 6)), 6)
        else:
            if adp_tokens and _encoded(adp_tokens[0]):
                state.adp_basis = np.zeros((1, 0))
                state.frozen_tokens.append("uiso")
            elif adp_tokens and float(adp_tokens[0]) < 0.0:
                # A non-hydrogen atom with a riding Uiso keeps it; its coordinates still refine.
                state.adp_basis = np.zeros((1, 0))
                state.frozen_tokens.append("uiso_multiplier")
            else:
                state.adp_basis = np.ones((1, 1))
        state.role = "refined" if state.n_params else "fixed"

    def _ueq_coefficients(self) -> np.ndarray:
        """Return c with Ueq = c . u6 for this cell (a third of the trace of the Cartesian U tensor)."""

        coef = np.zeros(6)
        for k in range(6):
            unit = np.zeros(6)
            unit[k] = 1.0
            atom = dict(zip(U6_KEYS, unit, strict=True))
            coef[k] = float(np.trace(shelx_u_cart_matrix(atom, self.cell))) / 3.0
        return coef

    def parameter_labels(self) -> list[dict]:
        """Return one label per structure parameter, in order: atom, kind (xyz, uij or uiso), component, direction."""

        labels = []
        for state in self.atoms:
            if state.role != "refined":
                continue
            for j in range(state.xyz_basis.shape[1]):
                labels.append({"atom": state.name, "kind": "xyz", "component": j, "frac_direction": state.xyz_basis[:, j].tolist()})
            for j in range(state.adp_basis.shape[1]):
                kind = "uij" if state.adp_type == "anis" else "uiso"
                labels.append({"atom": state.name, "kind": kind, "component": j, "u_direction": state.adp_basis[:, j].tolist()})
        return labels

    def dependencies(self, state: AtomState) -> list[int]:
        """Parameter indices one atom's contribution depends on."""

        deps: set[int] = set()
        if state.role == "refined":
            deps.update(range(state.param_start, state.param_start + state.n_params))
        if state.role == "riding" and state.parent is not None:
            deps.update(self.dependencies(self.atoms[state.parent]))
        if state.role in {"refined", "riding"} and state.uiso_multiplier and state.ueq_parent is not None:
            deps.update(self.dependencies(self.atoms[state.ueq_parent]))
        return sorted(deps)

    # ------------------------------------------------------------------ values
    def _refined_values(self, state: AtomState, local, lib):
        """(xyz, u) of a refined atom for its local parameter vector (numpy or torch)."""

        kx = state.xyz_basis.shape[1]
        xyz = lib.asarray(state.xyz0) if lib is np else _t(state.xyz0)
        u = lib.asarray(state.u0) if lib is np else _t(state.u0)
        if kx:
            xyz = xyz + (_t(state.xyz_basis) if lib is not np else state.xyz_basis) @ local[:kx]
        if state.adp_basis.shape[1]:
            u = u + (_t(state.adp_basis) if lib is not np else state.adp_basis) @ local[kx:]
        return xyz, u

    def atom_values(self, state: AtomState, params, lib=np):
        """(xyz, u) of any atom from the full parameter vector."""

        if state.role == "refined":
            local = params[state.param_start : state.param_start + state.n_params]
            xyz, u = self._refined_values(state, local, lib)
        else:
            xyz = np.asarray(state.xyz0) if lib is np else _t(state.xyz0)
            u = np.asarray(state.u0) if lib is np else _t(state.u0)
        if state.role == "riding" and state.parent is not None:
            parent = self.atoms[state.parent]
            pxyz, _pu = self.atom_values(parent, params, lib)
            xyz = xyz + (pxyz - (np.asarray(parent.xyz0) if lib is np else _t(parent.xyz0)))
        if state.role in {"refined", "riding"} and state.uiso_multiplier and state.ueq_parent is not None:
            # A negative SHELX Uiso keeps its multiple of the preceding atom's Ueq (hydrogen or not).
            ueq_parent = self.atoms[state.ueq_parent]
            _qxyz, qu = self.atom_values(ueq_parent, params, lib)
            if ueq_parent.adp_type == "anis":
                coef = self._ueq_coef if lib is np else _t(self._ueq_coef)
                ueq = (coef * qu).sum()
            else:
                ueq = qu[0]
            u = state.uiso_multiplier * ueq * (np.ones(1) if lib is np else _t(np.ones(1)))
        return xyz, u

    def model_at(self, params: np.ndarray) -> dict:
        """A parse_res-style model dict with every refined and riding atom moved to ``params``."""

        import copy

        params = np.asarray(params, dtype=float)
        model = copy.deepcopy(self.model)
        for state in self.atoms:
            if state.role == "fixed":
                continue
            xyz, u = self.atom_values(state, params)
            atom = model["atoms"][state.index]
            atom["x"], atom["y"], atom["z"] = (float(v) for v in xyz)
            if state.adp_type == "anis":
                for key, value in zip(U6_KEYS, u, strict=True):
                    atom[key] = float(value)
            else:
                atom["uiso"] = float(u[0])
        return model

    def parameters_from_model(self, model: dict, *, xyz_tolerance_A: float = 1e-7,
                              adp_tolerance_A2: float = 1e-8) -> np.ndarray:
        """Express another model in this unchanged parameterization.

        Used when a SHELXL update must be evaluated on existing frozen row plans. Atom names, cell, symmetry,
        occupancies and scattering factors must match. Fixed values and riding/site constraints are checked by
        reconstructing the complete model. Coordinates are used as written: no origin, hand or lattice shift is
        silently applied. The tolerances cover serialization only, not a change of physical model.
        """

        if not all(np.isfinite(v) and v >= 0 for v in (xyz_tolerance_A, adp_tolerance_A2)):
            raise ValueError("model-transfer tolerances must be finite and nonnegative")
        if not np.allclose(model["cell"], self.cell, rtol=0, atol=1e-10):
            raise ValueError("model transfer cannot change the unit cell")
        if model["sfac"] != self.model["sfac"]:
            raise ValueError("model transfer cannot change the scattering factors")

        def symmetry_key(ops):
            return sorted(tuple(np.round(np.r_[r.ravel(), np.mod(t, 1)], 12)) for r, t in ops)

        if symmetry_key(model_symmetry_ops(model)) != symmetry_key(self.ops):
            raise ValueError("model transfer cannot change the symmetry operations")
        atoms = {str(atom["name"]): atom for atom in model["atoms"]}
        if len(atoms) != len(model["atoms"]) or set(atoms) != {s.name for s in self.atoms}:
            raise ValueError("model transfer needs the same unique atom names")
        params = np.zeros(self.n_params)
        for state in self.atoms:
            atom = atoms[state.name]
            if (str(atom.get("element", "")).upper() != state.element
                    or int(atom["sfac_index"]) != state.sfac_index
                    or not np.isclose(atom["occ"], state.occ, rtol=0, atol=1e-12)
                    or atom.get("adp_type", "iso") != state.adp_type):
                raise ValueError(f"{state.name}: element, occupancy or ADP type changed")
            xyz = np.asarray([atom[k] for k in ("x", "y", "z")], dtype=float)
            u = np.asarray([atom[k] for k in (U6_KEYS if state.adp_type == "anis" else ("uiso",))])
            if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(u)):
                raise ValueError(f"{state.name}: non-finite coordinates or ADPs")
            if len(self._unique_mate_ops(xyz)) != len(state.mate_ops):
                raise ValueError(f"{state.name}: site multiplicity changed")
            if state.role == "refined":
                kx = state.xyz_basis.shape[1]
                start = state.param_start
                params[start:start + kx] = np.linalg.lstsq(state.xyz_basis, xyz - state.xyz0, rcond=None)[0]
                params[start + kx:start + state.n_params] = np.linalg.lstsq(state.adp_basis, u - state.u0, rcond=None)[0]

        for expected in self.model_at(params)["atoms"]:
            atom = atoms[expected["name"]]
            delta = np.asarray([atom[k] - expected[k] for k in ("x", "y", "z")])
            keys = U6_KEYS if expected.get("adp_type") == "anis" else ("uiso",)
            u_error = max(abs(atom[k] - expected[k]) for k in keys)
            if np.linalg.norm(self._orth @ delta) > xyz_tolerance_A or u_error > adp_tolerance_A2:
                raise ValueError(f"{expected['name']}: update violates fixed, riding or site constraints")
        return params

    def moved_atoms(self) -> list[str]:
        """Return the names of the refined and riding atoms, whose values the polished .res rewrites."""

        return [s.name for s in self.atoms if s.role in {"refined", "riding"}]

    def npd_atoms(self, params: np.ndarray) -> list[str]:
        """Return the names of atoms whose U is not positive definite at ``params``."""

        model = self.model_at(params)
        bad = []
        for atom in model["atoms"]:
            if atom.get("adp_type") == "anis":
                if float(np.min(np.linalg.eigvalsh(shelx_u_cart_matrix(atom, self.cell)))) <= 0.0:
                    bad.append(str(atom["name"]))
            elif float(atom.get("uiso", 0.0)) <= 0.0:
                bad.append(str(atom["name"]))
        return bad

    def shifts(self, params: np.ndarray) -> list[dict]:
        """Return, for each moved atom, its coordinate shift (Å) and Ueq change (Å²) at ``params``."""

        out = []
        for state in self.atoms:
            if state.role == "fixed":
                continue
            xyz, u = self.atom_values(state, np.asarray(params, dtype=float))
            delta = xyz - state.xyz0
            out.append(
                {
                    "atom": state.name,
                    "element": state.element,
                    "role": state.role,
                    "shift_A": float(np.linalg.norm(self._orth @ delta)),
                    "delta_ueq_A2": float(self._ueq(state, u) - self._ueq(state, state.u0)),
                }
            )
        return out

    def _ueq(self, state: AtomState, u: np.ndarray) -> float:
        """Return the Ueq (Å²) of an atom's U values (Uiso itself for an isotropic atom)."""

        return float(self._ueq_coef @ u) if state.adp_type == "anis" else float(u[0])


def _t(array):
    """Return an array as a float64 torch tensor."""

    import torch

    return torch.as_tensor(np.asarray(array, dtype=float), dtype=torch.float64)


class CrystalPotential:
    """U_g on a fixed table of reciprocal-lattice vectors for one crystal (its own cell and wavelength)."""

    def __init__(self, structure: StructureModel, g_table: np.ndarray, cell: tuple, volume: float, gamma: float):
        """Precompute scattering factors and, per symmetry operation, R^T g, ADP quadratic forms and phases."""

        self.structure = structure
        self.g = np.asarray(g_table, dtype=float)
        self.cell = tuple(float(v) for v in cell)
        self.factor = float(gamma) / (math.pi * float(volume))
        h, k, l = self.g[:, 0], self.g[:, 1], self.g[:, 2]
        s2 = 0.25 * inv_d2_from_hkl_array(h, k, l, self.cell)
        self.form = {
            idx: np.asarray(sfac_value_array(sf, s2), dtype=float) for idx, sf in enumerate(structure.model["sfac"])
        }
        astar = np.asarray(reciprocal_lengths(self.cell), dtype=float)
        self.form_t = {idx: _t(values) for idx, values in self.form.items()}
        self.q_t, self.quad_t, self.mate_s2_t, self.phase0_t = [], [], [], []
        for rot, trans in structure.ops:
            q = self.g @ rot  # rows are R^T g, the engine's mate indices
            qs = q * astar
            quad = np.column_stack(
                [qs[:, 0] ** 2, qs[:, 1] ** 2, qs[:, 2] ** 2, 2 * qs[:, 1] * qs[:, 2], 2 * qs[:, 0] * qs[:, 2], 2 * qs[:, 0] * qs[:, 1]]
            )
            self.q_t.append(_t(q))
            self.quad_t.append(_t(quad))
            self.mate_s2_t.append(_t(0.25 * inv_d2_from_hkl_array(q[:, 0], q[:, 1], q[:, 2], self.cell)))
            self.phase0_t.append(_t(2.0 * math.pi * (self.g @ np.asarray(trans, dtype=float))))

    def _atom_contribution(self, state: AtomState, xyz, u):
        """Complex contribution of one atom (without gamma/(pi V)) as a torch complex128 tensor of length n_g."""

        import torch

        total = torch.zeros(len(self.g), dtype=torch.complex128)
        amp0 = float(state.occ) * self.form_t[state.sfac_index]
        for op in state.mate_ops:
            phase = 2.0 * math.pi * (self.q_t[op] @ xyz) + self.phase0_t[op]
            if state.adp_type == "anis":
                dw = torch.exp(-2.0 * math.pi * math.pi * (self.quad_t[op] @ u))
            else:
                dw = torch.exp(-8.0 * math.pi * math.pi * u[0] * self.mate_s2_t[op])
            total = total + torch.polar(amp0 * dw, phase)
        return total

    def values_and_jacobian(self, params: np.ndarray, *, jacobian: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
        """U_g (n_g complex) and dU_g/dp (n_g, n_params complex) by forward-mode autograd (dual numbers)."""

        import torch
        import torch.autograd.forward_ad as fwad

        structure = self.structure
        p_full = _t(params)
        with torch.no_grad():
            values = torch.zeros(len(self.g), dtype=torch.complex128)
            for state in structure.atoms:
                xyz, u = structure.atom_values(state, p_full, lib=torch)
                values = values + self._atom_contribution(state, xyz, u)
        out = values.numpy() * self.factor
        if not jacobian:
            return out, None
        jac = np.zeros((len(self.g), structure.n_params), dtype=np.complex128)
        for state in structure.atoms:
            for column in structure.dependencies(state):
                tangent = torch.zeros_like(p_full)
                tangent[column] = 1.0
                with fwad.dual_level():
                    dual = fwad.make_dual(p_full, tangent)
                    xyz, u = structure.atom_values(state, dual, lib=torch)
                    jac[:, column] += fwad.unpack_dual(self._atom_contribution(state, xyz, u)).tangent.detach().numpy()
        return out, jac * self.factor


def reciprocal_ball(cell, radius_inv_A: float) -> np.ndarray:
    """Every integer (h, k, l) != 0 with |g| <= radius (1/A) in the given cell."""

    import gemmi

    uc = gemmi.UnitCell(*[float(v) for v in cell])
    hmax, kmax, lmax = (int(v) + 1 for v in uc.get_hkl_limits(1.0 / float(radius_inv_A)))
    h, k, l = np.meshgrid(
        np.arange(-hmax, hmax + 1), np.arange(-kmax, kmax + 1), np.arange(-lmax, lmax + 1), indexing="ij"
    )
    hkl = np.column_stack([h.ravel(), k.ravel(), l.ravel()])
    inv_d2 = inv_d2_from_hkl_array(hkl[:, 0].astype(float), hkl[:, 1].astype(float), hkl[:, 2].astype(float), tuple(cell))
    keep = (inv_d2 <= float(radius_inv_A) ** 2 * (1.0 + 1e-9)) & np.any(hkl != 0, axis=1)
    return hkl[keep].astype(np.int32)


class GIndex:
    """Dense lookup from (h, k, l) to the row of a reciprocal-vector table."""

    def __init__(self, g_table: np.ndarray):
        """Build the dense lookup over [-lim, lim] on each axis, -1 where a vector is not in the table."""

        g = np.asarray(g_table, dtype=np.int64)
        self.lim = np.max(np.abs(g), axis=0) if len(g) else np.zeros(3, dtype=np.int64)
        shape = tuple(int(2 * v + 1) for v in self.lim)
        self.lookup = np.full(shape, -1, dtype=np.int32)
        self.lookup[tuple((g + self.lim).T)] = np.arange(len(g), dtype=np.int32)

    def __call__(self, hkl: np.ndarray) -> np.ndarray:
        """Return the table row of each HKL along the last axis, -1 for a vector outside the table."""

        hkl = np.asarray(hkl, dtype=np.int64)
        shifted = hkl + self.lim
        inside = np.all((shifted >= 0) & (shifted <= 2 * self.lim), axis=-1)
        out = np.full(hkl.shape[:-1], -1, dtype=np.int32)
        out[inside] = self.lookup[tuple(shifted[inside].T)]
        return out
