"""Gauss-Newton/Levenberg-Marquardt dynamical polish of a finished GROND run.

Objective (work rows only; free rows are predicted and reported, never fitted):

    F = sum_crystals [ sum_work w_r (y_r - s_r f_r I_r(p, t))^2 + kappa c^T P c ]

I_r is the Bloch row intensity of the run's physics, f_r the recorded row-resolution factor, s_r the frame
scale of the stage's own family (a P-spline in scan angle with the stage's fitted lambda, or free per-frame
scales) whose coefficients c are profiled exactly at every evaluation, and w_r the stage's scale-fit weights
times the dataset-policy row weight.  p are the structure parameters (structure.StructureModel) and t each
crystal's thickness centre (plus the gradient fraction of the uniform-gradient model).  The Gauss-Newton
matrix uses the Kaufman projection of this variable-projection problem. The gradient includes active scale
bounds and the prediction dependence of their floor; the Gauss-Newton Hessian remains an approximation.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grond.polish import crystals as C
from grond.polish.structure import CrystalPotential, GIndex, StructureModel, reciprocal_ball
from grond.regularize.spline_scales import PSplineScaleFitter, calculate_weights, construct_spline_basis
from grond.validation import heldout as H

SCALE_FLOOR = 1e-12
GRAD_BOUNDS = (0.0, 1.0)  # the engine bound of thickness_grad_frac
# SHELX-weight iterations of the engine's per-frame scale fit (regularize.scales.fit_group_scales default).
ENGINE_SCALE_IRLS_ITERATIONS = 6


@dataclass
class PolishOptions:
    """Optimizer and resource settings of the dynamical polish (set from the --final-polish-* options).

    Step limits are max_xyz_step_A (Å), max_u_step_A2 (Å²) and max_thickness_step (fraction of the current thickness).
    Convergence requires a small projected gradient. Small shifts/esds (including thickness) together with a
    small relative decrease can stop a stalled fit, but do not by themselves establish convergence.
    """

    workers: int = 1
    max_iterations: int = 8
    rows_per_task: int = 16
    max_xyz_step_A: float = 0.05
    max_u_step_A2: float = 0.005
    max_thickness_step: float = 0.1
    shift_esd_tol: float = 0.01
    ftol: float = 1e-6
    gradient_tol: float = 1e-6
    refine_thickness: bool = True
    jacobian_memory_gib: float = 8.0
    n_boot: int = H.N_BOOT
    hydrogen_mode: str = "translate"
    checkpoint_evaluations: bool = False


def _atomic_save(path: str | Path, array: np.ndarray) -> None:
    """Save an array as .npy through a temporary file and a rename, so a worker never reads a partial file."""

    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- frame scales

class ScaleBlock:
    """The stage's frame-scale family on one crystal, profiled exactly for a fixed unscaled prediction.

    ``w`` are the crystal's own scale-fit weights; the dataset-policy weight multiplies the crystal's whole
    objective term (``objective_weight``), which leaves its scales unchanged, and is zero for an endpoint-only
    crystal whose scales are still fitted for its predictions.
    """

    def __init__(self, crystal: C.PolishCrystal):
        """Set up a P-spline, an explicit nonnegative design, or per-frame groups.

        Per-frame groups need at least two fit rows for their own scale. Raises ValueError when fewer than 10 work rows
        can fit the scales.
        """

        self.kind = crystal.scale["kind"]
        self.crystal = crystal
        self.objective_weight = float(crystal.fit_weight)
        fit = crystal.fit_mask
        if self.kind in {"pspline", "fixed_design"}:
            self.w = calculate_weights(crystal.y, crystal.sigma_source, crystal.scale.get("objective", "row_mr"))
            if self.kind == "fixed_design":
                # A declared design can share exactly the same nuisance freedoms across fitting algorithms.
                # Only its topology is supplied; coefficients are fitted from this crystal's work observations.
                extra = getattr(crystal, "extra", {})
                self.basis = np.asarray(extra.get("scale_basis", []), dtype=float).copy()
                if (self.basis.ndim != 2 or self.basis.shape[0] != len(crystal.y) or self.basis.shape[1] == 0
                        or not np.all(np.isfinite(self.basis)) or np.any(self.basis < 0)
                        or np.any(np.sum(self.basis, axis=1) <= 0)):
                    raise ValueError("fixed scale design needs a finite nonnegative basis covering every row")
                if np.linalg.matrix_rank(self.basis[fit]) < self.basis.shape[1]:
                    raise ValueError("fixed scale design is not identifiable on the working rows")
                self.penalty = np.asarray(extra.get("scale_penalty", np.zeros((0, self.basis.shape[1]))), dtype=float).copy()
                if (self.penalty.ndim != 2 or self.penalty.shape[1] != self.basis.shape[1]
                        or not np.all(np.isfinite(self.penalty))):
                    raise ValueError("fixed scale penalty must have one column per coefficient")
            else:
                angle = np.asarray(crystal.angle, dtype=float)
                if crystal.scale.get("clamp", True):
                    angle = np.clip(angle, np.min(angle[fit]), np.max(angle[fit]))
                self.basis, self.penalty, _cuts = construct_spline_basis(angle, fit, n_basis_target=int(crystal.scale["n_basis"]))
            self.lam = float(crystal.scale["lambda"])
            self.kappa = 0.0
            self.fit = fit.copy()
        elif self.kind == "per-frame":
            sig = np.asarray(crystal.sigma_eff, dtype=float)
            self.sig = sig
            self.wght = (float(crystal.scale["wght_a"]), float(crystal.scale["wght_b"]))
            # 1/sigma^2 until freeze_penalty() fixes the stage's iterated SHELX weights at the start prediction.
            self.w = np.where(np.isfinite(sig) & (sig > 0.0), 1.0 / np.maximum(sig, 1e-300) ** 2, 0.0)
            self.w_global = self.w.copy()
            groups, inverse = np.unique(crystal.group, return_inverse=True)
            counts = np.bincount(inverse[fit], minlength=len(groups))
            # Groups with fewer than two work rows take the global scale over every work row, as the engine's
            # per-frame fit (regularize.scales.fit_group_scales) does.
            self.fit = fit & (counts[inverse] >= 2)
            self.global_fit = fit.copy()
            self.group_of = inverse
            self.group_ids = groups
            fitted = np.flatnonzero(counts >= 2)
            self.column = np.full(len(groups), -1)
            self.column[fitted] = np.arange(len(fitted))
            self.n_groups = int(len(fitted))
            self.basis = None
            self.penalty = np.zeros((0, self.n_groups))
            self.kappa = 0.0
        else:
            raise ValueError(f"unsupported scale family {self.kind!r}")
        if int(np.count_nonzero(self.fit)) < 10:
            raise ValueError(f"{crystal.name}: fewer than 10 work rows can fit the frame scales")

    @property
    def n_coef(self) -> int:
        """Number of scale coefficients: P-spline basis functions or fitted frame groups."""

        return int(self.basis.shape[1]) if self.basis is not None else self.n_groups

    def design(self, pred_u: np.ndarray) -> np.ndarray:
        """Return the matrix that maps scale coefficients to scaled predictions for the unscaled prediction ``pred_u``.

        Per-frame: rows of frames without their own scale (fewer than two fit rows) are zero; scales() gives them the
        global scale.
        """

        if self.basis is not None:
            return pred_u[:, None] * self.basis
        a = np.zeros((len(pred_u), self.n_coef))
        rows = np.flatnonzero(self.column[self.group_of] >= 0)
        a[rows, self.column[self.group_of[rows]]] = pred_u[rows]
        return a

    def freeze_penalty(self, pred_u: np.ndarray) -> None:
        """Fix the prediction-dependent parts of the stage's scale fit at ``pred_u`` (the start prediction).

        P-spline: kappa = lambda * penalty_scale / raw_scale^2 of PSplineScaleFitter.  Per-frame: the engine's
        iterated SHELX weights w = 1/(sigma^2 + (a P)^2 + b P) at its converged group and global scales.
        """

        if self.basis is None:
            from grond.regularize.scales import _fit_global_scale, fit_group_scales, shelx_weights

            a, b = self.wght
            y = self.crystal.y
            # The engine's last least-squares step uses the weights of its previous iterate, so weights frozen at
            # the (n - 1)-iteration scales reproduce its n-iteration scales.
            n_iter = ENGINE_SCALE_IRLS_ITERATIONS - 1
            _scaled, group_scales = fit_group_scales(y, self.sig, pred_u, self.crystal.group, self.global_fit, a, b, n_iter=n_iter)
            scale = np.asarray([group_scales[int(g)] for g in self.group_ids], dtype=float)[self.group_of]
            ok = np.isfinite(self.sig) & (self.sig > 0.0)
            self.w = np.where(ok, shelx_weights(y, np.where(ok, self.sig, 1.0), scale * pred_u, a, b), 0.0)
            global_scale = _fit_global_scale(y, self.sig, pred_u, self.global_fit, a, b, n_iter)
            self.w_global = np.where(ok, shelx_weights(y, np.where(ok, self.sig, 1.0), global_scale * pred_u, a, b), 0.0)
            return
        if self.lam == 0 or self.penalty.size == 0:
            self.kappa = 0.0
            return
        fitter = PSplineScaleFitter(n_basis=self.n_coef, objective=self.crystal.scale.get("objective", "row_mr"))
        system = fitter._prepare_system(self.crystal.y, self.crystal.sigma_source, pred_u, self.basis, self.penalty, self.fit)
        self.kappa = self.lam * float(system["penalty_scale"]) / float(system["raw_scale"]) ** 2

    def floor(self, pred_u: np.ndarray) -> float:
        """Lower bound of a scale coefficient: PSplineScaleFitter's floor_factor on its normalised coefficients."""

        if self.basis is None:
            return SCALE_FLOOR
        fit = self.fit
        num = float(np.sum(self.w[fit] * self.crystal.y[fit] ** 2))
        den = max(float(np.sum(self.w[fit] * pred_u[fit] ** 2)), 1e-300)
        return SCALE_FLOOR * max(math.sqrt(num / den), np.finfo(float).tiny)

    def augmented(self, pred_u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(A_aug, b_aug) with residual = b_aug - A_aug c: fitted rows, then penalty rows."""

        sw = np.sqrt(self.w[self.fit])
        a = self.design(pred_u)[self.fit] * sw[:, None]
        b = self.crystal.y[self.fit] * sw
        if self.basis is not None:
            a = np.vstack([a, math.sqrt(self.kappa) * self.penalty])
            b = np.concatenate([b, np.zeros(self.penalty.shape[0])])
        return a, b

    def solve(self, pred_u: np.ndarray) -> tuple[np.ndarray, int]:
        """Return (coefficients, number held at the floor) of the least-squares scale fit.

        A coefficient below the floor switches to bounded least squares (BVLS) with that floor as the lower bound.
        """

        a, b = self.augmented(pred_u)
        coef, *_ = np.linalg.lstsq(a, b, rcond=None)
        floor = self.floor(pred_u)
        if np.min(coef) < floor:
            from scipy.optimize import lsq_linear

            coef = lsq_linear(a, b, bounds=(floor, np.inf), method="bvls", tol=1e-14).x
        active = int(np.count_nonzero(~self.free_coefficients(coef, pred_u)))
        return coef, active

    def free_coefficients(self, coef: np.ndarray, pred_u: np.ndarray) -> np.ndarray:
        """Coefficients that can vary locally, rather than sitting at the positivity floor."""

        return coef > self.floor(pred_u) * (1.0 + 1e-9)

    def floor_gradient(self, pred_u: np.ndarray, prediction_jacobian_fit: np.ndarray) -> np.ndarray:
        """Derivative of the P-spline's normalization-dependent coefficient floor."""

        if self.basis is None:
            return np.zeros(prediction_jacobian_fit.shape[1])
        weighted = self.w[self.fit] * pred_u[self.fit]
        denominator = float(weighted @ pred_u[self.fit])
        if denominator <= 1e-300:
            return np.zeros(prediction_jacobian_fit.shape[1])
        return -self.floor(pred_u) * (weighted @ prediction_jacobian_fit) / denominator

    def scales(self, coef: np.ndarray, pred_u: np.ndarray) -> np.ndarray:
        """Return the per-row scale for ``coef``; rows of sparse frames take the weighted global scale."""

        if self.basis is not None:
            return self.basis @ coef
        s = np.full(len(self.group_of), np.nan)
        rows = self.column[self.group_of] >= 0
        s[rows] = coef[self.column[self.group_of[rows]]]
        if np.any(~rows):
            g = self.global_fit
            x = pred_u[g]
            y = self.crystal.y[g]
            global_scale = float(np.sum(self.w_global[g] * x * y) / max(float(np.sum(self.w_global[g] * x * x)), 1e-300))
            s[~rows] = max(global_scale, SCALE_FLOOR)
        return s

    def edf(self, pred_u: np.ndarray, coef: np.ndarray | None = None) -> float:
        """Effective degrees of freedom conditional on the active bounds and floor."""

        a, _b = self.augmented(pred_u)
        if coef is None:
            coef, _active = self.solve(pred_u)
        a = a[:, self.free_coefficients(coef, pred_u)]
        if a.shape[1] == 0:
            return 0.0
        n_fit = int(np.count_nonzero(self.fit))
        m = a.T @ a
        data = a[:n_fit]
        return float(np.trace(np.linalg.lstsq(m, data.T @ data, rcond=None)[0]))


# ----------------------------------------------------------------------------- the polish problem

class PolishProblem:
    """The dynamical polish of one structure against the row tables of one or more crystals.

    Holds each crystal's Fourier-coefficient table, geometry, thickness quadrature and scale block. The parameter vector
    x is the structure parameters followed by each fitted crystal's thickness centre t0 (Å) and, for a uniform-gradient
    model, its gradient fraction.
    """

    def __init__(self, crystals: list[C.PolishCrystal], structure: StructureModel, work_dir: Path, options: PolishOptions):
        """Build per-crystal Fourier-potential tables, geometry, quadratures and scale blocks; index parameters.

        Raises ValueError when a crystal's structure Jacobian would need more than options.jacobian_memory_gib.
        """

        from grond.legacy.engine.bloch_engine import XDSData, relativistic_gamma_from_wavelength

        self.crystals = crystals
        self.structure = structure
        self.options = options
        self.work = Path(work_dir)
        self.state_dir = self.work / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.potentials: dict[str, CrystalPotential] = {}
        self.geometry: dict[str, dict] = {}
        self.quad: dict[str, dict] = {}
        self.thickness_bounds: dict[str, tuple[float, float]] = {}
        self.scales: dict[str, ScaleBlock] = {}
        for crystal in crystals:
            data = XDSData(crystal.hkl, hkl_transform=crystal.hkl_transform)
            cell = (data.cell.a, data.cell.b, data.cell.c, data.cell.alpha, data.cell.beta, data.cell.gamma)
            gamma = relativistic_gamma_from_wavelength(data.wavelength) if crystal.cfg.gamma is None else float(crystal.cfg.gamma)
            g_table = reciprocal_ball(cell, C.ball_radius(crystal))
            need = len(g_table) * max(structure.n_params, 1) * 16 / 2**30
            if need > float(options.jacobian_memory_gib):
                raise ValueError(
                    f"{crystal.name}: the structure Jacobian needs {need:.1f} GiB ({len(g_table)} Fourier coefficients x "
                    f"{structure.n_params} parameters), above --final-polish-jacobian-memory-gib "
                    f"{options.jacobian_memory_gib:g}"
                )
            index = GIndex(g_table)
            lookup_path = self.state_dir / f"{crystal.name}_lookup.npy"
            _atomic_save(lookup_path, index.lookup)
            self.potentials[crystal.name] = CrystalPotential(structure, g_table, cell, data.cell.volume, gamma)
            self.geometry[crystal.name] = {
                "k": float(data.k),
                "a_matrix": np.asarray(data.A, dtype=float).tolist(),
                "lookup": str(lookup_path),
                "lookup_lim": index.lim.tolist(),
                "n_g": int(len(g_table)),
                "cell": [float(v) for v in cell],
                "gamma": float(gamma),
            }
            self.quad[crystal.name] = C.thickness_quadrature(crystal.cfg)
            initial = self.quad[crystal.name]["t0"]
            bounds = crystal.extra.get("thickness_bounds_A", (0.2 * initial, None))
            lower, upper = float(bounds[0]), float(bounds[1]) if bounds[1] is not None else float("inf")
            if not (np.isfinite(lower) and 0 < lower < upper and lower <= initial <= upper):
                raise ValueError(f"{crystal.name}: thickness bounds must be positive, ordered and contain the start")
            self.thickness_bounds[crystal.name] = (lower, upper)
            self.scales[crystal.name] = ScaleBlock(crystal)
        self.n_struct = structure.n_params
        self.thick_index: dict[str, dict[str, int]] = {}
        next_index = self.n_struct
        for crystal in crystals:
            entry = {}
            # An endpoint-only crystal is not in the objective, so its thickness stays at the stage value.
            if options.refine_thickness and crystal.in_objective:
                entry["t0"] = next_index
                next_index += 1
                if self.quad[crystal.name]["refine_grad"]:
                    entry["grad"] = next_index
                    next_index += 1
            self.thick_index[crystal.name] = entry
        self.n_x = next_index
        self.labels = structure.parameter_labels() + [
            {"crystal": name, "kind": key} for name, entry in self.thick_index.items() for key in entry
        ]
        self.chunks: dict[str, list[Path]] = {}
        self.evaluations = 0
        self.eval_seconds = 0.0

    # ------------------------------------------------------------------ parameters
    def x0(self) -> np.ndarray:
        """Return the start vector: structure parameters at zero (the start model), stage thickness values."""

        x = np.zeros(self.n_x)
        for name, entry in self.thick_index.items():
            if "t0" in entry:
                x[entry["t0"]] = self.quad[name]["t0"]
            if "grad" in entry:
                x[entry["grad"]] = self.quad[name]["grad"]
        return x

    def thickness(self, x: np.ndarray, name: str) -> tuple[float, float]:
        """Return a crystal's (t0 in Å, gradient fraction) at x; unrefined values come from its quadrature."""

        q = self.quad[name]
        entry = self.thick_index[name]
        return (
            float(x[entry["t0"]]) if "t0" in entry else float(q["t0"]),
            float(x[entry["grad"]]) if "grad" in entry else float(q["grad"]),
        )

    # ------------------------------------------------------------------ plans
    def build_plans(self, start_res: Path) -> dict:
        """Build and save every crystal's frozen row plans in worker tasks; return counts and timing.

        Each chunk also computes the engine and stage intensities of its sampled check rows.
        """

        tasks = []
        for crystal in self.crystals:
            polish_cfg = C.config_dict(crystal.cfg)
            polish_cfg["double_precision"] = True
            polish_spec = {"hkl": str(crystal.hkl), "hkl_transform": crystal.hkl_transform, "res": str(start_res), "cfg": polish_cfg}
            stage_spec = {
                "hkl": str(crystal.hkl),
                "hkl_transform": crystal.hkl_transform,
                "res": str(crystal.stage_res),
                "alternate_hand": crystal.stage_alternate_hand,
                "cfg": C.config_dict(crystal.cfg),
            }
            checks = set(C.sample_rows(len(crystal.rows)).tolist())
            size = max(1, int(self.options.rows_per_task))
            self.chunks[crystal.name] = []
            for start in range(0, len(crystal.rows), size):
                pos = np.arange(start, min(start + size, len(crystal.rows)))
                out = self.work / "plans" / crystal.name / f"chunk_{start // size:05d}.npz"
                self.chunks[crystal.name].append(out)
                sampled = [int(crystal.rows[i]) for i in pos if int(i) in checks]
                tasks.append(
                    {
                        "polish_spec": polish_spec,
                        "stage_spec": stage_spec,
                        "rows": crystal.rows[pos].tolist(),
                        "positions": pos.tolist(),
                        "engine_check_rows": sampled,
                        "table_check_rows": sampled,
                        "out": str(out),
                    }
                )
        t0 = time.perf_counter()
        results = C.run_tasks(C.build_plan_chunk, tasks, self.options.workers)
        return {
            "tasks": len(tasks),
            "rows": int(sum(r["rows"] for r in results)),
            "max_beams": int(max(r["max_beams"] for r in results)) if results else 0,
            "seconds": time.perf_counter() - t0,
        }

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, x: np.ndarray, *, jacobian: bool = True, thickness_response: bool = False) -> dict[str, dict]:
        """Evaluate every crystal's row intensities, Born controls and thickness derivatives at x.

        With ``jacobian`` the structure Jacobian of each fitted crystal's work rows is added. U_g and its derivatives
        are written to the state directory for the workers. Raises ValueError for non-finite or non-positive
        intensities. With thickness_response, also return the profile-averaged intensity at every support node.
        For a fixed structure and geometry this table permits distribution fitting without repeated propagation.
        """

        t_start = time.perf_counter()
        if thickness_response and any(q.get("evaluator") != "table" for q in self.quad.values()):
            raise ValueError("reusable thickness responses require fixed support-table nodes")
        p = np.asarray(x[: self.n_struct], dtype=float)
        tasks = []
        for crystal in self.crystals:
            # An endpoint-only crystal is predicted but not fitted, so it needs no structure Jacobian.
            want = jacobian and crystal.in_objective
            u, d = self.potentials[crystal.name].values_and_jacobian(p, jacobian=want)
            base = self.state_dir / crystal.name
            u_path, d_re_path, d_im_path = f"{base}_U.npy", f"{base}_Dre.npy", f"{base}_Dim.npy"
            _atomic_save(u_path, u)
            if want:
                _atomic_save(d_re_path, np.ascontiguousarray(d.real))
                _atomic_save(d_im_path, np.ascontiguousarray(d.imag))
            t0, grad = self.thickness(x, crystal.name)
            nodes, dt_dt0, dt_dgrad = C.thickness_nodes(self.quad[crystal.name], t0, grad)
            weights, dw_dt0, dw_dgrad = C.thickness_weights(self.quad[crystal.name], t0, grad)
            geo = self.geometry[crystal.name]
            grad_positions = np.flatnonzero(self.scales[crystal.name].fit).tolist() if want else []
            for chunk in self.chunks[crystal.name]:
                tasks.append(
                    {
                        "crystal": crystal.name,
                        "chunk": str(chunk),
                        "u": u_path,
                        "d_re": d_re_path if want else "",
                        "d_im": d_im_path if want else "",
                        "lookup": geo["lookup"],
                        "lookup_lim": geo["lookup_lim"],
                        "a_matrix": geo["a_matrix"],
                        "k": geo["k"],
                        "matrix_convention": crystal.cfg.matrix_convention,
                        "t_nodes": nodes.tolist(),
                        "t_weights": weights.tolist(),
                        "dt_dt0": dt_dt0.tolist(),
                        "dt_dgrad": dt_dgrad.tolist(),
                        "dw_dt0": dw_dt0.tolist(),
                        "dw_dgrad": dw_dgrad.tolist(),
                        "grad_positions": grad_positions,
                        "return_thickness_response": thickness_response,
                    }
                )
        evaluator = C.evaluate_chunk
        if self.options.checkpoint_evaluations:
            from grond.polish.checkpoints import evaluate_chunk_cached

            evaluator = evaluate_chunk_cached
            for task in tasks:
                task["cache_directory"] = str(self.work / "evaluations")
        results = C.run_tasks(evaluator, tasks, self.options.workers)
        out: dict[str, dict] = {}
        for crystal in self.crystals:
            n = len(crystal.rows)
            out[crystal.name] = {
                "intensity": np.full(n, np.nan),
                "born": np.full(n, np.nan),
                "d_t0": np.zeros(n),
                "d_grad": np.zeros(n),
                "jacobian": np.zeros((n, self.n_struct)) if jacobian else None,
                "engine_dyn": np.full(n, np.nan),
                "stage_dyn": np.full(n, np.nan),
                "thick": None,
                "thick_w": None,
            }
            if thickness_response:
                nodes = self.quad[crystal.name]["nodes"]
                out[crystal.name]["thickness_response"] = np.zeros((n, len(nodes)))
                out[crystal.name]["thickness_response_nodes"] = nodes.copy()
        for res in results:
            block = out[res["crystal"]]
            pos = np.asarray(res["positions"], dtype=int)
            for key in ("intensity", "born", "d_t0", "d_grad", "engine_dyn", "stage_dyn"):
                block[key][pos] = res[key]
            if jacobian and res["jacobian"] is not None and len(res["grad_positions"]):
                block["jacobian"][np.asarray(res["grad_positions"], dtype=int)] = res["jacobian"]
            block["thick"], block["thick_w"] = res["thick"], res["thick_w"]
            if thickness_response:
                if not np.array_equal(block["thickness_response_nodes"], res["thickness_response_nodes"]):
                    raise ValueError("worker thickness-response nodes differ from the declared support")
                block["thickness_response"][pos] = res["thickness_response"]
        for name, block in out.items():
            if not np.all(np.isfinite(block["intensity"])) or np.any(block["intensity"] <= 0.0):
                raise ValueError(f"{name}: the polish forward model returned non-finite or non-positive row intensities")
        self.evaluations += 1
        self.eval_seconds += time.perf_counter() - t_start
        return out

    # ------------------------------------------------------------------ objective
    def profile(self, x: np.ndarray, ev: dict[str, dict], *, with_jacobian: bool = True) -> dict:
        """Profiled objective, scales and the Kaufman-projected Gauss-Newton system at x."""

        total = 0.0
        rows_k, res_k = [], []
        parts = {}
        for crystal in self.crystals:
            name = crystal.name
            block = ev[name]
            sb = self.scales[name]
            pred_u = crystal.row_factor * block["intensity"]
            coef, active = sb.solve(pred_u)
            s = sb.scales(coef, pred_u)
            a_aug, b_aug = sb.augmented(pred_u)
            r_aug = b_aug - a_aug @ coef
            # The dataset-policy weight multiplies the crystal's whole term; an endpoint-only crystal has none.
            weight = sb.objective_weight
            total += weight * float(r_aug @ r_aug)
            parts[name] = {"coef": coef, "scales": s, "pred": s * pred_u, "active_floor": active, "edf": sb.edf(pred_u, coef)}
            if not with_jacobian or weight <= 0.0:
                continue
            fit = sb.fit
            sw = np.sqrt(weight * sb.w[fit])
            prediction_jacobian = np.zeros((int(np.count_nonzero(fit)), self.n_x))
            factor = crystal.row_factor[fit]
            prediction_jacobian[:, : self.n_struct] = factor[:, None] * block["jacobian"][fit]
            entry = self.thick_index[name]
            if "t0" in entry:
                prediction_jacobian[:, entry["t0"]] = factor * block["d_t0"][fit]
            if "grad" in entry:
                prediction_jacobian[:, entry["grad"]] = factor * block["d_grad"][fit]
            jac = -(sw * s[fit])[:, None] * prediction_jacobian
            j_aug = np.vstack([jac, np.zeros((a_aug.shape[0] - jac.shape[0], self.n_x))])
            free = sb.free_coefficients(coef, pred_u)
            if np.any(~free):
                floor_gradient = sb.floor_gradient(pred_u, prediction_jacobian)
                j_aug -= math.sqrt(weight) * a_aug[:, ~free].sum(axis=1)[:, None] * floor_gradient
            # Only unconstrained scale directions can absorb a model change.
            a_free = a_aug[:, free]
            m = a_free.T @ a_free
            proj = a_free @ np.linalg.lstsq(m, a_free.T @ j_aug, rcond=None)[0]
            rows_k.append(j_aug - proj)
            res_k.append(math.sqrt(weight) * r_aug)
        out = {"objective": total, "parts": parts}
        if with_jacobian:
            out["jacobian"] = np.vstack(rows_k)
            out["residual"] = np.concatenate(res_k)
        return out

    def step_limit(self, x: np.ndarray, delta: np.ndarray) -> float:
        """Return the factor (at most 1) that keeps a step within the xyz (Å), U (Å²) and relative t0 limits."""

        factor = 1.0
        for i, label in enumerate(self.labels):
            value = abs(float(delta[i]))
            if value == 0.0:
                continue
            kind = label["kind"]
            if kind == "xyz":
                factor = min(factor, float(self.options.max_xyz_step_A) / value)
            elif kind in {"uij", "uiso"}:
                factor = min(factor, float(self.options.max_u_step_A2) / value)
            elif kind == "t0":
                factor = min(factor, float(self.options.max_thickness_step) * abs(float(x[i])) / value)
        return factor

    def clip(self, x: np.ndarray) -> np.ndarray:
        """Clip thickness parameters to their declared bounds; by default t0 >= 0.2 times its start."""

        x = x.copy()
        for name, entry in self.thick_index.items():
            if "grad" in entry:
                x[entry["grad"]] = min(max(x[entry["grad"]], GRAD_BOUNDS[0]), GRAD_BOUNDS[1])
            if "t0" in entry:
                lower, upper = self.thickness_bounds[name]
                x[entry["t0"]] = np.clip(x[entry["t0"]], lower, upper)
        return x


# ----------------------------------------------------------------------------- optimizer

def projected_gradient_cosine(problem: PolishProblem, x: np.ndarray, prof: dict) -> float:
    """Largest residual/Jacobian-column cosine after removing outward directions at parameter bounds.

    This dimensionless stationarity check is independent of LM damping and step clipping. A small accepted
    step caused by high damping therefore cannot manufacture a small gradient. Exactly zero residuals pass.
    """

    jac, residual = prof["jacobian"], prof["residual"]
    gradient = jac.T @ residual
    if not (np.all(np.isfinite(gradient)) and np.all(np.isfinite(residual))):
        return float("inf")
    feasible = problem.clip(x - np.sign(gradient)) - x
    gradient = np.where(feasible * gradient < 0.0, gradient, 0.0)
    denominator = np.linalg.norm(jac, axis=0) * np.linalg.norm(residual)
    cosine = np.divide(np.abs(gradient), denominator, out=np.zeros_like(gradient), where=denominator > 0)
    return float(np.max(cosine, initial=0.0))


def levenberg_marquardt(problem: PolishProblem, x0: np.ndarray, ev0: dict, log=print, *,
                       optimizer_state: dict | None = None, on_accept=None) -> dict:
    """Minimise the profiled objective from x0 by Levenberg-Marquardt; return the final state, trajectory and esds.

    Each iteration tries up to eight damped steps, limited by step_limit and clip; a step that makes a positive-definite
    atom NPD is refused. Only a small projected gradient is labelled converged. No descent, small changes without
    stationarity, the NPD guard and the iteration cap have distinct nonconvergence statuses.
    optimizer_state restores damping and the original NPD guard for the unchanged problem. on_accept receives a
    copied parameter vector and this small state after each accepted step, allowing durable outer checkpoints.
    """

    opts = problem.options
    x = np.asarray(x0, dtype=float).copy()
    ev = ev0
    prof = problem.profile(x, ev)
    mu = float(optimizer_state["mu"]) if optimizer_state is not None else 1e-3
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError("LM damping must be finite and positive")
    npd_start = (set(optimizer_state["npd_start"]) if optimizer_state is not None
                 else set(problem.structure.npd_atoms(x[: problem.n_struct])))
    npd_rejections = 0
    trajectory = [{"iteration": 0, "objective": prof["objective"], "accepted": True, "mu": None, "max_shift_esd": None}]
    status = "cap"
    esd = np.full(problem.n_x, np.nan)
    for iteration in range(1, int(opts.max_iterations) + 1):
        stationarity = projected_gradient_cosine(problem, x, prof)
        trajectory[-1]["projected_gradient_cosine"] = stationarity
        if stationarity <= float(opts.gradient_tol):
            status = "converged"
            break
        jac, res = prof["jacobian"], prof["residual"]
        hess = jac.T @ jac
        grad = jac.T @ res
        diag = np.diag(hess).copy()
        diag[diag <= 0.0] = max(float(np.max(diag)), 1.0) * 1e-12
        accepted = False
        guarded = 0
        for _attempt in range(8):
            try:
                delta = np.linalg.solve(hess + mu * np.diag(diag), -grad)
            except np.linalg.LinAlgError:
                mu *= 10.0
                continue
            limit = problem.step_limit(x, delta)
            delta = delta * min(1.0, limit)
            x_try = problem.clip(x + delta)
            new_npd = set(problem.structure.npd_atoms(x_try[: problem.n_struct])) - npd_start
            if new_npd:
                # A step may not make a positive-definite atom non-positive definite; shorten it instead.
                npd_rejections += 1
                guarded += 1
                log(f"polish iteration {iteration}: step rejected, it makes {', '.join(sorted(new_npd))} NPD (mu {mu:.3g})")
                mu *= 4.0
                continue
            ev_try = problem.evaluate(x_try)
            prof_try = problem.profile(x_try, ev_try)
            log(
                f"polish iteration {iteration}: objective {prof['objective']:.8g} -> {prof_try['objective']:.8g} "
                f"(mu {mu:.3g}, step limit {min(1.0, limit):.3g})"
            )
            if prof_try["objective"] < prof["objective"]:
                accepted = True
                break
            mu *= 4.0
        if not accepted:
            # stopped_npd_guard: every shortened step would have made a positive-definite atom NPD.
            status = "stopped_npd_guard" if guarded == 8 else "stopped_no_descent"
            trajectory.append({"iteration": iteration, "objective": prof["objective"], "accepted": False, "mu": mu})
            break
        step = x_try - x
        relative = (prof["objective"] - prof_try["objective"]) / max(prof["objective"], 1e-300)
        x, ev, prof = x_try, ev_try, prof_try
        mu = max(mu / 3.0, 1e-9)
        esd = parameter_esds(problem, prof)
        with np.errstate(invalid="ignore", divide="ignore"):
            shift_esd = np.where(step == 0, 0.0, np.abs(step) / esd)
        shift_esd = np.where(np.isfinite(shift_esd), shift_esd, np.inf)
        max_shift_esd = float(np.max(shift_esd[:problem.n_struct], initial=0.0))
        max_all_shift_esd = float(np.max(shift_esd, initial=0.0))
        stationarity = projected_gradient_cosine(problem, x, prof)
        trajectory.append(
            {
                "iteration": iteration,
                "objective": prof["objective"],
                "relative_decrease": relative,
                "accepted": True,
                "mu": mu,
                "max_shift_esd": max_shift_esd,
                "max_all_parameter_shift_esd": max_all_shift_esd,
                "projected_gradient_cosine": stationarity,
                "max_xyz_step_A": _max_kind(problem, step, "xyz"),
                "max_u_step_A2": max(_max_kind(problem, step, "uij"), _max_kind(problem, step, "uiso")),
            }
        )
        if on_accept is not None:
            on_accept({"x": x.copy(), "objective": float(prof["objective"]), "iteration": iteration,
                       "optimizer_state": {"mu": float(mu), "npd_start": sorted(npd_start)}})
        if stationarity <= float(opts.gradient_tol):
            status = "converged"
            break
        if max_all_shift_esd < float(opts.shift_esd_tol) and relative < float(opts.ftol):
            status = "stopped_small_step"
            break
    if not np.all(np.isfinite(esd)):
        esd = parameter_esds(problem, prof)
    return {"x": x, "evaluation": ev, "profile": prof, "trajectory": trajectory, "status": status,
            "converged": status == "converged", "projected_gradient_cosine": projected_gradient_cosine(problem, x, prof), "esd": esd,
            "optimizer_state": {"mu": float(mu), "npd_start": sorted(npd_start)},
            "npd_rejections": npd_rejections, "npd_start": sorted(npd_start)}


def _max_kind(problem: PolishProblem, step: np.ndarray, kind: str) -> float:
    """Return the largest |step| over the parameters of one kind (xyz, uij or uiso), or 0.0 when there are none."""

    values = [abs(float(step[i])) for i, label in enumerate(problem.labels) if label["kind"] == kind]
    return max(values) if values else 0.0


def parameter_esds(problem: PolishProblem, prof: dict) -> np.ndarray:
    """Return parameter esds: pseudo-inverse normal matrix times chi2 = objective / (n_fit - n_x - scale edf)."""

    jac = prof["jacobian"]
    hess = jac.T @ jac
    fitted = [c.name for c in problem.crystals if c.in_objective]
    n_fit = sum(int(np.count_nonzero(problem.scales[name].fit)) for name in fitted)
    edf = sum(float(prof["parts"][name]["edf"]) for name in fitted)
    dof = max(n_fit - problem.n_x - edf, 1.0)
    chi2 = float(prof["objective"]) / dof
    try:
        cov = np.linalg.pinv(hess, rcond=1e-12) * chi2
    except np.linalg.LinAlgError:
        return np.full(problem.n_x, np.nan)
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


# ----------------------------------------------------------------------------- held-out report

def heldout_report(problem: PolishProblem, start: dict, final: dict, n_boot: int) -> dict:
    """Free- and work-row R_all/R_obs (amplitude R1, %) with orbit-bootstrap CIs, start versus polished."""

    symmetry = problem.structure.model.get("symmetry_ops")
    ys, sigs, keys, ds, sets = [], [], [], [], {"free": [], "work": []}
    pred_start, pred_final, pred_table, crystal_of = [], [], [], []
    # Endpoint-only crystals are outside the raw-row metrics, as in the run's own held-out report.
    for crystal in (c for c in problem.crystals if c.in_objective):
        ys.append(crystal.y)
        sigs.append(crystal.sigma_source)
        keys.append(H.orbit_keys(crystal.hkl_rows, symmetry, include_friedel=True))
        ds.append(crystal.d)
        sets["free"].append(crystal.free_mask)
        sets["work"].append(problem.scales[crystal.name].fit)
        pred_start.append(start["parts"][crystal.name]["pred"])
        pred_final.append(final["parts"][crystal.name]["pred"])
        pred_table.append(crystal.table_pred)
        crystal_of.append(np.full(len(crystal.y), crystal.name, dtype=object))
    y = np.concatenate(ys)
    sig = np.concatenate(sigs)
    orbit = np.concatenate(keys)
    shell = H.shell_index(np.concatenate(ds))
    a, b, t = np.concatenate(pred_final), np.concatenate(pred_start), np.concatenate(pred_table)
    owner = np.concatenate(crystal_of)
    report = {"definition": "amplitude R1 in %, R_obs on Io > 3 sigma(source); paired shell-stratified merged-orbit bootstrap",
              "n_boot": int(n_boot), "seed": H.BOOTSTRAP_SEED}
    for split, masks in sets.items():
        sel = np.concatenate(masks)
        if not np.any(sel):
            report[split] = {"status": "no_rows"}
            continue
        block = {
            "polished": H.paired_orbit_bootstrap(y, sig, a, None, orbit, shell, sel, n_boot=n_boot),
            "start": H.paired_orbit_bootstrap(y, sig, b, None, orbit, shell, sel, n_boot=n_boot),
            "polished_minus_start": H.paired_orbit_bootstrap(y, sig, a, b, orbit, shell, sel, n_boot=n_boot),
            "run_row_table": H.r_metrics(y, sig, t, sel),
            "per_crystal": {},
        }
        for name in dict.fromkeys(owner):
            m = sel & (owner == name)
            if np.any(m):
                block["per_crystal"][str(name)] = {
                    "polished": H.r_metrics(y, sig, a, m),
                    "start": H.r_metrics(y, sig, b, m),
                    "run_row_table": H.r_metrics(y, sig, t, m),
                }
        report[split] = block
    return report


def scale_reproduction_checks(crystals: list[C.PolishCrystal]) -> dict:
    """The polish's frame-scale family, fitted to the stage's own prediction, must return the stage's scaled rows.

    Binds the scale contract (family, basis, lambda and its normalisation, weights, floor, the global scale of
    sparse frames) to the row table before anything is fitted; fails closed otherwise.
    """

    out = {}
    for crystal in crystals:
        block = ScaleBlock(crystal)
        pred_u = np.asarray(crystal.table_dyn_raw, dtype=float)
        block.freeze_penalty(pred_u)
        coef, _active = block.solve(pred_u)
        mismatch = C.relative_mismatch(block.scales(coef, pred_u) * pred_u, crystal.table_pred)
        out[crystal.name] = mismatch
        if not (math.isfinite(mismatch) and mismatch <= C.SCALE_CHECK_RTOL):
            raise ValueError(
                f"{crystal.name}: the polish's {crystal.scale['kind']} frame scales, fitted to the stage's own rows, reproduce "
                f"its dyn_scaled only to {mismatch:.3g} (relative); the stage used a scale contract the polish cannot represent"
            )
    return out


def leak_flags(problem: PolishProblem) -> dict:
    """Whether any free row entered a fitted quantity, computed from the masks actually used."""

    free_in_fit = any(bool(np.any(problem.scales[c.name].fit & c.free_mask)) for c in problem.crystals)
    return {"free_rows_in_objective": free_in_fit, "free_rows_in_scales": free_in_fit, "free_rows_in_stopping_rule": free_in_fit}


def forward_checks(problem: PolishProblem, ev0: dict) -> dict:
    """Engine identity (start model) and table reproduction (stage model) on the sampled rows; fail closed."""

    out = {}
    for crystal in problem.crystals:
        block = ev0[crystal.name]
        engine = C.relative_mismatch(block["intensity"], block["engine_dyn"])
        table = C.relative_mismatch(block["stage_dyn"], crystal.table_dyn)
        quad = problem.quad[crystal.name]
        nodes, _a, _b = C.thickness_nodes(quad, quad["t0"], quad["grad"])
        weights, _dw0, _dwg = C.thickness_weights(quad, quad["t0"], quad["grad"])
        # A fixed-model native calculation stores only nodes carrying positive mass. Compare measures,
        # while keeping all support nodes in the optimizer so later distributions can widen or move.
        if quad.get("evaluator") == "table":
            positive = weights > 0.0
            nodes, weights = nodes[positive], weights[positive]
        weight_difference = (
            float(np.max(np.abs(np.asarray(block["thick_w"]) - weights)))
            if len(block["thick_w"]) == len(weights) else float("inf")
        )
        thick = (
            float(np.max(np.abs(np.asarray(block["thick"]) - nodes))) if len(block["thick"]) == len(nodes) else float("inf")
        )
        out[crystal.name] = {
            "engine_identity_max_relative_difference": engine,
            "table_reproduction_max_relative_difference": table,
            "thickness_grid_max_difference_A": thick,
            "thickness_weight_max_difference": weight_difference,
            "sampled_rows": int(np.count_nonzero(np.isfinite(block["engine_dyn"]))),
        }
        if not (math.isfinite(engine) and engine <= C.IDENTITY_RTOL):
            raise ValueError(
                f"{crystal.name}: the polish forward model differs from the engine by {engine:.3g} (relative) on the "
                "sampled rows; the Hamiltonian reconstruction does not match this stage's physics"
            )
        if not (math.isfinite(table) and table <= C.TABLE_CHECK_RTOL):
            raise ValueError(
                f"{crystal.name}: the rebuilt Bloch configuration reproduces the row table only to {table:.3g} (relative); "
                "the stage summary does not describe the physics of its own rows"
            )
        if thick > 1e-9 * max(quad["t0"], 1.0):
            raise ValueError(f"{crystal.name}: the polish thickness quadrature differs from the engine grid")
        if weight_difference > 1e-12:
            raise ValueError(f"{crystal.name}: the polish thickness weights differ from the engine weights")
    return out


def solve_polish(problem: PolishProblem, start_res: Path, log=print) -> dict:
    """Plans, checks, optimization and held-out report (no files other than the working state)."""

    t0 = time.perf_counter()
    scale_checks = scale_reproduction_checks(problem.crystals)
    plans = problem.build_plans(start_res)
    log(f"polish plans: {plans['rows']} rows in {plans['tasks']} tasks, up to {plans['max_beams']} beams ({plans['seconds']:.1f} s)")
    x0 = problem.x0()
    ev0 = problem.evaluate(x0)
    checks = forward_checks(problem, ev0)
    for name, mismatch in scale_checks.items():
        checks[name]["scale_reproduction_max_relative_difference"] = mismatch
    for crystal in problem.crystals:
        problem.scales[crystal.name].freeze_penalty(crystal.row_factor * ev0[crystal.name]["intensity"])
    start = problem.profile(x0, ev0)
    fit = levenberg_marquardt(problem, x0, ev0, log=log)
    final = fit["profile"]
    heldout = heldout_report(problem, start, final, int(problem.options.n_boot))
    return {
        "plans": plans,
        "checks": checks,
        "start": start,
        "start_evaluation": ev0,
        "fit": fit,
        "heldout": heldout,
        "seconds": time.perf_counter() - t0,
    }
