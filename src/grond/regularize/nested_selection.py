"""Nested nuisance selection with physical thickness distributions.

API note (for the AUTO nuisance stage built on this module)
-----------------------------------------------------------
Inputs

* Spectra, one dict per physics variant, in the packed format (tests/reference/reference_lib.py pack_spectra) that
  ``grond.legacy.engine.spectra_stage.compute_spectra`` writes: ``row_ids, n_p, n_b, w, lam, c, h10, delta, mu``.
  Every variant must hold the same rows in the same order (the rows of the arrays below).
* ``Variant(label, spectra, params, config)``: ``params`` are the physics values of the variant used for edge
  flags (e.g. ``{"dose_b_total_A2": 2.0, "mosaicity_sigma_deg": 0.2}``); ``config`` are the Config overrides
  that produced its spectra (returned in the pick so the engine can reproduce it). ``Variant(label, None, params,
  config, table=...)`` gives a thickness response table instead (grond.legacy.engine.response_stage); the selection
  then needs ``evaluator="table"`` and ``score_with="table"``.
* Row arrays of one crystal: ``y`` (target intensity), ``sig`` (its sigma), ``ang`` (scan angle, deg), ``keys``
  (model Laue-orbit keys, grond.validation.heldout.orbit_keys) and ``work`` (bool). Only work rows are ever
  fitted, scored or selected on; free and excluded rows must be False in ``work``. The engine definitions are
  ``y, sig = target_arrays(data, target_mode)[row]``, ``ang = port.omega_from_z(data.zcal[row])`` and
  ``keys = orbit_keys(data.hkl[rows], model["symmetry_ops"], include_friedel=True)``; dose variants use
  ``dose_b_per_frame = dB_total / dataset_frame_span(hkl)`` (grond.workflows.nuisance_policy).
* Multi-crystal runs call select_nested once per crystal with the shared (crystal-independent) orbit keys.
  ``compute_spectra`` with workers > 1 starts spawned workers on macOS, so a calling script needs the usual
  ``if __name__ == "__main__":`` guard (a worker that cannot start raises BrokenProcessPool).
* ``ThicknessSupport(t_min_A=100, t_max_A=10000, dt_A=2)`` (``from_nm(10, 1000)``): physical support; every
  distribution is renormalised on it. ``SelectionGrids``: families (``gamma``, ``wedge``), their parameter grids,
  uniform absorption ``mu`` and the P-spline ``n_basis`` x ``lam`` grid (defaults: 32 knots, lambda 0.3-100).

Functions

* ``select_nested(variants, y, sig, ang, keys, work, grids=, support=, evaluator="auto", hands=(1,), workers=1,
  clamp=True, objective="row_mr", salt=INNER_CV_SALT, score_with="table", physical_bounds=None)``
  -> ``{hand: NestedSelection}``. For each hand (+1 model, -1 exact inverse) it picks one
  (variant, family, params, mu, scale) by inner 3-fold orbit CV R_all on work rows (salt GROND-INNER-20260925),
  then evaluates the pick with its resolved evaluator.
* ``NestedSelection``: ``pick`` (JSON-safe dict: variant, thickness, mu, scale, evaluator, scores, edge and
  physical-bound flags, support, grids), ``dyn``/``born`` (raw row intensities of the pick, every row),
  ``oof`` (the pick's inner-CV out-of-fold predictions, work rows only, NaN elsewhere), ``prediction`` (the pick's
  P-spline fitted on all work rows, predicting every row), ``scale_edf``, ``scores``/``candidates`` (the grid).
* ``row_intensities(spec, thickness, mu=0.0, hand=1, evaluator="auto", support=, table=None, workers=1)``
  -> ``{"dyn", "born", "evaluator", "mass_outside_support"}``; ``thickness_table(spec, support, hands, workers)``,
  ``table_row_intensities`` and ``cf_row_intensities`` are the two exact evaluators it dispatches to.
* ``engine_config_fields(pick, support)`` -> Config overrides (variant config + thickness + uniform absorption)
  whose ``XDSBlochPort`` rows equal the pick's ``dyn``/``born`` (tests/test_nested_selection.py).

Evaluators and scores

* ``table``: exact hat-function weights on the support grid (grond.bloch.thickness.distribution_weights).
* ``cf``: exact untruncated gamma average (characteristic function); gamma only.
* ``auto``: ``cf`` for a gamma pick whose mass outside the support is below 1e-6 (then it equals the truncated,
  renormalised average to that mass), else ``table``. With ``score_with="table"`` (default) every candidate is
  scored on the table, which evaluates the same support model to its 2 A interpolation error; the pick is then
  re-evaluated with its resolved evaluator and both scores are recorded (``inner_r_all_selection`` and
  ``inner_r_all``). ``score_with="resolved"`` scores each candidate with its own resolved evaluator (exact, slow).

Physical bounds are not grid edges: gradient 0 and 1, mu 0, dose 0, cone width 0 and the support bounds for the
mean / t0. mu > 0 is refused for gamma (it only rescales theta).
"""

from __future__ import annotations

import math
import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from grond.bloch.thickness import (
    CF_MASS_TOLERANCE,
    DEFAULT_TABLE_DT_A,
    DEFAULT_THICKNESS_MAX_A,
    DEFAULT_THICKNESS_MIN_A,
    canonical_family,
    distribution_weights,
    gamma_cf_born,
    gamma_cf_profile,
    gamma_parameters,
    mass_outside_support,
    resolve_evaluator,
    support_grid,
    support_mass,
    uniform_grid_amplitudes,
    uniform_grid_spec,
    validate_support,
)
from grond.regularize.spline_scales import PSplineScaleFitter, construct_spline_basis
from grond.validation.heldout import INNER_CV_SALT, hash_fold, r1

SPECTRA_KEYS = ("row_ids", "n_p", "n_b", "w", "lam", "c", "h10", "delta", "mu")
INNER_FOLDS = 3
DYN_FLOOR = 1e-30
TABLE_BLOCK = 256
EVALUATORS = ("table", "cf", "auto")
SCORE_WITH = ("table", "resolved")

DEFAULT_GAMMA_MEAN_GRID = tuple(float(v) for v in range(100, 3401, 100))
DEFAULT_GAMMA_SHAPE_GRID = (0.25, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 40.0)
DEFAULT_WEDGE_T0_GRID = tuple(float(v) for v in range(200, 5001, 200))
DEFAULT_WEDGE_GRAD_GRID = (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
DEFAULT_LAMBDA_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
# Values of a parameter that are physical bounds rather than grid edges; support bounds are added
# for the gamma mean and the wedge t0.
DEFAULT_PHYSICAL_BOUNDS = {
    "grad": (0.0, 1.0),
    "mu": (0.0,),
    "dose_b_total_A2": (0.0,),
    "dose_b_per_frame": (0.0,),
    "mosaicity_sigma_deg": (0.0,),
}
FAMILY_PARAMS = {"gamma": ("mean", "shape"), "wedge": ("t0", "grad")}


# ----------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class ThicknessSupport:
    """Physical thickness support [t_min_A, t_max_A] and table step dt_A (A)."""

    t_min_A: float = DEFAULT_THICKNESS_MIN_A
    t_max_A: float = DEFAULT_THICKNESS_MAX_A
    dt_A: float = DEFAULT_TABLE_DT_A

    def __post_init__(self) -> None:
        """Refuse a support that grond.bloch.thickness.validate_support rejects."""
        validate_support(self.t_min_A, self.t_max_A, self.dt_A)

    @classmethod
    def from_nm(cls, t_min_nm: float = 10.0, t_max_nm: float = 1000.0, dt_A: float = DEFAULT_TABLE_DT_A):
        """Build a support from bounds in nm; the step dt_A stays in A."""
        return cls(10.0 * float(t_min_nm), 10.0 * float(t_max_nm), float(dt_A))

    def grid(self) -> np.ndarray:
        """Return the thickness nodes from t_min_A to t_max_A at step dt_A (A)."""
        return support_grid(self.t_min_A, self.t_max_A, self.dt_A)

    def to_dict(self) -> dict:
        """Return the support as a JSON-safe dict (A)."""
        return {"t_min_A": float(self.t_min_A), "t_max_A": float(self.t_max_A), "dt_A": float(self.dt_A)}


@dataclass(frozen=True)
class SelectionGrids:
    """Candidate grids of the nested selection (defaults: the gamma family and the AUTO P-spline scales)."""

    families: tuple[str, ...] = ("gamma",)
    gamma_mean: tuple[float, ...] = DEFAULT_GAMMA_MEAN_GRID
    gamma_shape: tuple[float, ...] = DEFAULT_GAMMA_SHAPE_GRID
    wedge_t0: tuple[float, ...] = DEFAULT_WEDGE_T0_GRID
    wedge_grad: tuple[float, ...] = DEFAULT_WEDGE_GRAD_GRID
    mu: tuple[float, ...] = (0.0,)
    n_basis: tuple[int, ...] = (32,)
    lam: tuple[float, ...] = DEFAULT_LAMBDA_GRID

    def __post_init__(self) -> None:
        """Refuse unknown families, empty, repeated, non-finite or unphysical grids, and mu > 0 with gamma."""
        if not self.families:
            raise ValueError("nested selection needs at least one thickness family")
        for fam in self.families:
            if fam not in FAMILY_PARAMS:
                raise ValueError(f"unknown thickness family {fam!r}; expected gamma or wedge")
        grids = {"gamma_mean": self.gamma_mean, "gamma_shape": self.gamma_shape, "wedge_t0": self.wedge_t0,
                 "wedge_grad": self.wedge_grad, "mu": self.mu, "n_basis": self.n_basis, "lam": self.lam}
        for name, values in grids.items():
            if not values:
                raise ValueError(f"nested selection grid {name} is empty")
            if len(set(float(v) for v in values)) != len(values):
                raise ValueError(f"nested selection grid {name} has repeated values")
            if any(not math.isfinite(float(v)) for v in values):
                raise ValueError(f"nested selection grid {name} has non-finite values")
        if any(float(v) <= 0.0 for v in (*self.gamma_mean, *self.gamma_shape, *self.wedge_t0, *self.lam)):
            raise ValueError("gamma mean/shape, wedge t0 and lambda grids must be positive")
        if any(not 0.0 <= float(v) <= 1.0 for v in self.wedge_grad):
            raise ValueError("wedge gradient grid must lie in the physical range [0, 1]")
        if any(float(v) < 0.0 for v in self.mu):
            raise ValueError("absorption mu grid must be non-negative")
        if "gamma" in self.families and any(float(v) > 0.0 for v in self.mu):
            raise ValueError(
                "mu > 0 with the gamma family is degenerate with the gamma scale (exp(-4 pi mu t) only rescales "
                "theta); use mu grid (0,) for gamma"
            )
        if any(int(v) != v or int(v) < 4 for v in self.n_basis):
            raise ValueError("P-spline n_basis values must be integers of at least 4")

    def thickness_candidates(self) -> list[tuple[str, dict]]:
        """Return every (family, parameters) pair of the family grids, for example (gamma, {mean, shape})."""
        out = []
        for fam in self.families:
            a_name, b_name = FAMILY_PARAMS[fam]
            a_grid, b_grid = (self.gamma_mean, self.gamma_shape) if fam == "gamma" else (self.wedge_t0, self.wedge_grad)
            out += [(fam, {a_name: float(a), b_name: float(b)}) for a in a_grid for b in b_grid]
        return out

    def scale_candidates(self, objective: str) -> list[dict]:
        """Return every P-spline scale option (n_basis, lambda) with the given objective."""
        return [{"n_basis": int(nb), "lambda": float(lam), "objective": str(objective)} for nb in self.n_basis for lam in self.lam]

    def param_grids(self) -> dict[str, list[float]]:
        """Return the parameter grids of the selected families (mean and shape, t0 and grad), used for edge flags."""
        out = {}
        if "gamma" in self.families:
            out.update(mean=[float(v) for v in self.gamma_mean], shape=[float(v) for v in self.gamma_shape])
        if "wedge" in self.families:
            out.update(t0=[float(v) for v in self.wedge_t0], grad=[float(v) for v in self.wedge_grad])
        return out

    def to_dict(self) -> dict:
        """Return the grids as a JSON-safe dict; lam is written as lambda."""
        return {"families": list(self.families), "gamma_mean": [float(v) for v in self.gamma_mean],
                "gamma_shape": [float(v) for v in self.gamma_shape], "wedge_t0": [float(v) for v in self.wedge_t0],
                "wedge_grad": [float(v) for v in self.wedge_grad], "mu": [float(v) for v in self.mu],
                "n_basis": [int(v) for v in self.n_basis], "lambda": [float(v) for v in self.lam]}


@dataclass
class Variant:
    """One physics variant (dose, cone, window, ...) with its spectra or its thickness response table.

    ``spectra`` is the packed dict or a zero-argument loader returning it (for example from the spectra-stage
    cache), so that select_nested holds one variant's spectra at a time instead of every variant's. A variant may
    instead give ``table``, a thickness table (or its loader) as thickness_table or
    grond.legacy.engine.response_stage returns it, with ``spectra`` None; the selection then needs
    evaluator="table" and score_with="table".
    """

    label: str
    spectra: dict | Callable[[], dict] | None
    params: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    table: dict | Callable[[], dict] | None = None

    def load(self) -> dict:
        """Return the packed spectra, calling the loader when one was given."""
        if self.spectra is None:
            raise ValueError(f"variant {self.label!r} has a response table and no spectra")
        return self.spectra() if callable(self.spectra) else self.spectra

    def load_table(self) -> dict | None:
        """Return the response table (None when the variant has spectra), calling the loader when one was given."""
        return self.table() if callable(self.table) else self.table


@dataclass
class NestedSelection:
    """Result of select_nested for one hand; the fields are described in the module docstring."""

    pick: dict
    dyn: np.ndarray
    born: np.ndarray
    oof: np.ndarray
    prediction: np.ndarray
    scale_edf: float
    scores: np.ndarray
    candidates: list


# ----------------------------------------------------------------------------- spectra


def validate_spectra(spec: dict, label: str = "spectra") -> int:
    """Number of rows of a packed spectra dict; inconsistent or non-finite spectra fail closed."""
    missing = [k for k in SPECTRA_KEYS if k not in spec]
    if missing:
        raise ValueError(f"{label} lacks {missing}")
    n_p = np.asarray(spec["n_p"], dtype=int)
    n_b = np.asarray(spec["n_b"], dtype=int)
    n = len(np.asarray(spec["row_ids"]))
    if not (len(n_p) == len(n_b) == len(np.asarray(spec["mu"])) == n):
        raise ValueError(f"{label}: per-row arrays have different lengths")
    if np.any(n_p < 1) or np.any(n_b < 2):
        raise ValueError(f"{label}: every row needs at least one profile point and two beams")
    points = int(n_p.sum())
    coeffs = int(np.sum(n_p * n_b))
    for key, size in (("w", points), ("h10", points), ("delta", points), ("lam", coeffs), ("c", coeffs)):
        if len(np.asarray(spec[key])) != size:
            raise ValueError(f"{label}: {key} has {len(np.asarray(spec[key]))} values, expected {size}")
    for key in ("w", "lam", "h10", "delta", "mu"):
        if not np.all(np.isfinite(np.asarray(spec[key], dtype=float))):
            raise ValueError(f"{label}: non-finite {key}")
    if not np.all(np.isfinite(np.asarray(spec["c"]))):
        raise ValueError(f"{label}: non-finite couplings")
    return n


def validate_table(table: dict, support: ThicknessSupport, hands=(1,), label: str = "response table") -> int:
    """Number of rows of a thickness response table for this support and these hands; anything else fails closed.

    The table needs ``row_ids`` and the per-row ``mu`` besides the thickness_table arrays. Intensities are checked
    in row blocks, so the check needs no full-size temporary arrays.
    """
    for key in ("t", "support", "row_ids", "mu", "born"):
        if key not in table:
            raise ValueError(f"{label} lacks {key!r}")
    if dict(table["support"]) != support.to_dict():
        raise ValueError(f"{label} was built on support {dict(table['support'])}, not {support.to_dict()}")
    t = np.asarray(table["t"])
    if not np.array_equal(t, support.grid()):
        raise ValueError(f"{label}: thickness nodes differ from the support grid")
    missing = [h for h in hands if h not in table]
    if missing:
        raise ValueError(f"{label} has no hand {missing}")
    n = len(np.asarray(table["row_ids"]))
    if len(np.asarray(table["mu"])) != n:
        raise ValueError(f"{label}: mu has {len(np.asarray(table['mu']))} values, expected {n}")
    for key in [*hands, "born"]:
        values = table[key]
        if tuple(values.shape) != (n, len(t)):
            raise ValueError(f"{label}: {key} has shape {tuple(values.shape)}, expected {(n, len(t))}")
        for lo in range(0, n, 4096):
            block = np.asarray(values[lo:lo + 4096])
            if not np.all(np.isfinite(block)) or np.any(block < 0.0):
                raise ValueError(f"{label}: {key} has non-finite or negative intensities")
    return n


def _offsets(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return each row's start offsets into the per-point and per-coefficient arrays (length n_rows + 1)."""
    n_p = np.asarray(spec["n_p"], dtype=int)
    n_b = np.asarray(spec["n_b"], dtype=int)
    return np.concatenate([[0], np.cumsum(n_p)]), np.concatenate([[0], np.cumsum(n_p * n_b)])


def spectra_views(spec: dict, lo: int = 0, hi: int | None = None):
    """Yield per-row (w[P], lam[P, N], c[P, N], h10[P], delta[P], mu) for rows lo..hi."""
    op, ob = _offsets(spec)
    hi = len(op) - 1 if hi is None else int(hi)
    for i in range(int(lo), hi):
        p, n = int(spec["n_p"][i]), int(spec["n_b"][i])
        yield (np.asarray(spec["w"][op[i]:op[i + 1]], dtype=float),
               np.asarray(spec["lam"][ob[i]:ob[i + 1]], dtype=float).reshape(p, n),
               np.asarray(spec["c"][ob[i]:ob[i + 1]]).reshape(p, n),
               np.asarray(spec["h10"][op[i]:op[i + 1]], dtype=float),
               np.asarray(spec["delta"][op[i]:op[i + 1]], dtype=float),
               float(spec["mu"][i]))


# ----------------------------------------------------------------------------- parallel map

_SHARED: dict = {}


def _chunks(n: int, workers: int) -> list[tuple[int, int]]:
    """Split n rows into up to 4 x workers contiguous (lo, hi) ranges."""
    edges = np.linspace(0, n, max(2, min(n, 4 * max(1, int(workers)))) + 1).astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a]


def _parallel_map(fn, items: list, workers: int, shared: dict) -> list:
    """fn(item) for every item, in order; workers > 1 forks workers sharing ``shared`` copy-on-write.

    Without the fork start method the map runs serially (identical results). A worker that dies raises
    BrokenProcessPool instead of hanging.
    """
    _SHARED.clear()
    _SHARED.update(shared)
    try:
        if int(workers) > 1 and len(items) > 1 and "fork" in mp.get_all_start_methods():
            with ProcessPoolExecutor(max_workers=min(int(workers), len(items)), mp_context=mp.get_context("fork")) as pool:
                return list(pool.map(fn, items))
        return [fn(item) for item in items]
    finally:
        _SHARED.clear()


# ----------------------------------------------------------------------------- evaluators


def _table_worker(bounds):
    """Compute rows lo..hi of the thickness table from _SHARED; returns (lo, {hand: intensities, born: intensities}).

    Each array has one row per spectra row in lo..hi and one column per support node.
    """
    lo, hi = bounds
    spec, t, hands = _SHARED["spec"], _SHARED["t"], _SHARED["hands"]
    out = {h: np.empty((hi - lo, len(t))) for h in hands}
    out["born"] = np.empty((hi - lo, len(t)))
    grid = uniform_grid_spec(t)
    for i, (w, lam, c, h10, delta, _mu) in enumerate(spectra_views(spec, lo, hi)):
        if grid is not None:
            # Exact factorised phases (grond.bloch.thickness.uniform_grid_amplitudes), one GEMM per profile point.
            for h in hands:
                amp = uniform_grid_amplitudes(lam, c, *grid, inverse=h < 0)
                out[h][i] = w @ (np.abs(amp) ** 2)
        else:
            for t_lo in range(0, len(t), TABLE_BLOCK):
                tb = t[t_lo:t_lo + TABLE_BLOCK]
                E = np.exp(2j * np.pi * lam[:, :, None] * tb[None, None, :])
                for h in hands:
                    amp = np.einsum("pj,pjt->pt", c, E if h > 0 else E.conj())
                    out[h][i, t_lo:t_lo + len(tb)] = w @ (np.abs(amp) ** 2)
        sinc = np.sinc(delta[:, None] * t[None, :])
        out["born"][i] = w @ ((2.0 * np.pi) ** 2 * h10[:, None] * (t[None, :] ** 2) * sinc ** 2)
    return lo, out


def thickness_table(spec: dict, support: ThicknessSupport, hands=(1,), workers: int = 1) -> dict:
    """Profile-summed row intensity on the support grid for each hand, and the Born term (no absorption).

    Any distribution on the support is then a weighted sum over the grid (table_row_intensities), exact for the
    piecewise-linear interpolant of I(t) at step dt (as reference_lib.thickness_table, here on [t_min, t_max]).
    """
    n = validate_spectra(spec)
    hands = tuple(int(h) for h in hands)
    if any(h not in (1, -1) for h in hands):
        raise ValueError("hands must be +1 (model) and/or -1 (inverse)")
    t = support.grid()
    parts = _parallel_map(_table_worker, _chunks(n, workers), workers, {"spec": spec, "t": t, "hands": hands})
    out = {"t": t, "support": support.to_dict(), "hands": hands}
    for key in list(hands) + ["born"]:
        out[key] = np.empty((n, len(t)))
        for lo, part in parts:
            out[key][lo:lo + len(part[key])] = part[key]
    return out


def table_row_intensities(table: dict, thickness: dict, mu: float = 0.0, hand: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """(dyn, born) of one distribution from a thickness table; dyn carries exp(-4 pi mu t) at the nodes."""
    fam, params = _split_thickness(thickness)
    if int(hand) not in table:
        raise ValueError(f"thickness table has no hand {hand}")
    t = table["t"]
    w = distribution_weights(fam, params, t, mu=float(mu))
    w0 = w if float(mu) == 0.0 else distribution_weights(fam, params, t, mu=0.0)
    dyn = np.maximum(table[int(hand)] @ w, DYN_FLOOR)
    born = np.maximum(table["born"] @ w0, DYN_FLOOR)
    return dyn, born


def _cf_worker(bounds):
    """Compute rows lo..hi of the gamma characteristic-function average from _SHARED.

    Returns (lo, {hand: dyn, born: born}).
    """
    lo, hi = bounds
    spec, k, theta, alpha, hands = (_SHARED[key] for key in ("spec", "k", "theta", "alpha", "hands"))
    out = {h: np.empty(hi - lo) for h in hands}
    out["born"] = np.empty(hi - lo)
    for i, (w, lam, c, h10, delta, _mu) in enumerate(spectra_views(spec, lo, hi)):
        model, inverse = gamma_cf_profile(lam, c, k, theta, alpha)
        for h in hands:
            out[h][i] = float(w @ (model if h > 0 else inverse))
        out["born"][i] = float(w @ gamma_cf_born(h10, delta, k, theta))
    return lo, out


def cf_row_intensities(spec: dict, thickness: dict, mu: float = 0.0, hands=(1,), workers: int = 1) -> dict:
    """Exact untruncated gamma average for every row: {hand: dyn, "born": born}."""
    fam, params = _split_thickness(thickness)
    if fam != "gamma":
        raise ValueError("the cf evaluator (gamma characteristic function) needs the gamma family")
    n = validate_spectra(spec)
    k, theta = gamma_parameters(params)
    hands = tuple(int(h) for h in hands)
    parts = _parallel_map(_cf_worker, _chunks(n, workers), workers,
                          {"spec": spec, "k": k, "theta": theta, "alpha": 4.0 * np.pi * float(mu), "hands": hands})
    out = {}
    for key in list(hands) + ["born"]:
        out[key] = np.empty(n)
        for lo, part in parts:
            out[key][lo:lo + len(part[key])] = part[key]
        out[key] = np.maximum(out[key], DYN_FLOOR)
    return out


def _split_thickness(thickness: dict) -> tuple[str, dict]:
    """Return the canonical family and its two parameters (floats) of a thickness dict; missing parameters raise."""
    fam = str(thickness.get("family", ""))
    canonical = canonical_family(fam)
    names = FAMILY_PARAMS["gamma" if canonical == "gamma" else "wedge"]
    missing = [k for k in names if k not in thickness]
    if missing:
        raise ValueError(f"thickness {thickness} lacks {missing}")
    return canonical, {k: float(thickness[k]) for k in names}


# ----------------------------------------------------------------------------- inner CV scoring


def row_intensities(spec: dict, thickness: dict, mu: float = 0.0, hand: int = 1, evaluator: str = "auto",
                    support: ThicknessSupport | None = None, table: dict | None = None, workers: int = 1) -> dict:
    """Row intensities of one thickness distribution with the requested evaluator (table, cf or auto)."""
    support = support or ThicknessSupport()
    fam, params = _split_thickness(thickness)
    resolved = resolve_evaluator(fam, params, evaluator, support.t_min_A, support.t_max_A)
    if resolved == "cf":
        res = cf_row_intensities(spec, thickness, mu, hands=(int(hand),), workers=workers)
        dyn, born = res[int(hand)], res["born"]
    else:
        if table is None:
            table = thickness_table(spec, support, hands=(int(hand),), workers=workers)
        elif table.get("support") != support.to_dict():
            raise ValueError("thickness table was built on a different support")
        dyn, born = table_row_intensities(table, thickness, mu, hand)
    return {"dyn": dyn, "born": born, "evaluator": resolved,
            "mass_outside_support": mass_outside_support(fam, params, support.t_min_A, support.t_max_A)}


def _fold_contexts(ang, work, inner, n_basis_grid, clamp: bool) -> list[dict]:
    """Per inner fold: training/validation masks and the P-spline basis of each n_basis (dyn-independent)."""
    folds = []
    for h in range(INNER_FOLDS):
        tr = work & (inner != h)
        va = work & (inner == h)
        if not tr.any() or not va.any():
            raise ValueError(
                f"inner orbit CV needs work rows in every one of the {INNER_FOLDS} folds; fold {h} has "
                f"{int(tr.sum())} training and {int(va.sum())} validation rows"
            )
        a = np.clip(ang, np.min(ang[tr]), np.max(ang[tr])) if clamp else ang
        bases = {}
        for nb in n_basis_grid:
            B, D, _cuts = construct_spline_basis(a, tr, n_basis_target=int(nb), breakpoints=[])
            bases[int(nb)] = (B, D)
        folds.append({"tr": tr, "va": va, "bases": bases})
    return folds


def inner_cv_oof(dyn, y, sig, work, folds, scale_options, objective: str = "row_mr", floor_factor: float = 1e-12):
    """Out-of-fold predictions of every scale option (one P-spline fit per inner fold, n_basis and lambda).

    Identical to PSplineScaleFitter.fit(..., lambda_smooth=lam) on the clamped angles of each inner fold
    (as reference_lib._inner_cv_r_all); the basis and the normal equations are shared across lambda.
    """
    fitter = PSplineScaleFitter(objective=objective, floor_factor=floor_factor)
    oof = np.full((len(scale_options), len(y)), np.nan)
    for fold in folds:
        tr, va = fold["tr"], fold["va"]
        systems = {}
        for si, opt in enumerate(scale_options):
            nb = int(opt["n_basis"])
            B, D = fold["bases"][nb]
            if nb not in systems:
                systems[nb] = fitter._prepare_system(y, sig, dyn, B, D, tr)
            sol = fitter._solve_for_lambda(systems[nb], B, float(opt["lambda"]), tr, y)
            oof[si, va] = sol["prediction"][va]
    return oof


def _r_all(y, pred, work) -> float:
    """Return amplitude R_all in percent of pred against y on the work rows where pred is finite."""
    fo = np.sqrt(np.maximum(y, 0.0))
    m = work & np.isfinite(pred)
    return r1(fo, np.sqrt(np.maximum(np.nan_to_num(pred), 0.0)), m)


def _score_dyn(dyn) -> np.ndarray:
    """Return the inner-CV R_all (percent) of every scale option in _SHARED for one set of raw row intensities."""
    S = _SHARED
    oof = inner_cv_oof(dyn, S["y"], S["sig"], S["work"], S["folds"], S["scale_options"], S["objective"])
    return np.asarray([_r_all(S["y"], oof[si], S["work"]) for si in range(len(S["scale_options"]))])


def _table_score_worker(qi: int):
    """Score thickness candidate qi on the shared thickness table; returns {hand: R_all per scale option}."""
    S = _SHARED
    fam, params, mu = S["thick"][qi]
    w = distribution_weights(fam, params, S["table"]["t"], mu=mu)
    return {h: _score_dyn(np.maximum(S["table"][h] @ w, DYN_FLOOR)) for h in S["hands"]}


# ----------------------------------------------------------------------------- selection


def _variant_param_grids(variants: list[Variant]) -> dict[str, list[float]]:
    """Collect the numeric physics parameters of all variants into sorted value grids, used for edge flags."""
    grids: dict[str, set] = {}
    for v in variants:
        for k, val in (v.params or {}).items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                grids.setdefault(k, set()).add(float(val))
    return {k: sorted(vals) for k, vals in grids.items()}


def _edge_flags(value_by_param: dict, grids: dict, bounds: dict) -> tuple[list[str], list[str]]:
    """(grid-edge flags, parameters at a physical bound) for the chosen values."""
    edges, physical = [], []
    for name, value in value_by_param.items():
        grid = grids.get(name) or []
        value = float(value)
        if value in bounds.get(name, ()):
            physical.append(name)
            continue
        if len(grid) > 1 and value in (min(grid), max(grid)):
            edges.append(name)
    return edges, physical


def select_nested(variants: list[Variant], y, sig, ang, keys, work, *, grids: SelectionGrids | None = None,
                  support: ThicknessSupport | None = None, evaluator: str = "auto", hands=(1,), workers: int = 1,
                  clamp: bool = True, objective: str = "row_mr", salt: str = INNER_CV_SALT,
                  score_with: str = "table", physical_bounds: dict | None = None) -> dict[int, NestedSelection]:
    """Pick (variant, thickness family and parameters, mu, P-spline scale) per hand by inner orbit CV on work rows.

    See the module docstring for the contract. Free and excluded rows (work False) never enter a fit, a score or
    a selection; they only receive predictions.
    """
    grids = grids or SelectionGrids()
    support = support or ThicknessSupport()
    if evaluator not in EVALUATORS:
        raise ValueError(f"unknown thickness evaluator {evaluator!r}; expected one of {EVALUATORS}")
    if score_with not in SCORE_WITH:
        raise ValueError(f"unknown score_with {score_with!r}; expected one of {SCORE_WITH}")
    if not variants:
        raise ValueError("nested selection needs at least one physics variant")
    labels = [str(v.label) for v in variants]
    if len(set(labels)) != len(labels):
        raise ValueError(f"physics variant labels must be unique, got {labels}")
    hands = tuple(dict.fromkeys(int(h) for h in hands))
    if not hands or any(h not in (1, -1) for h in hands):
        raise ValueError("hands must be +1 (model) and/or -1 (inverse)")
    y = np.asarray(y, dtype=float)
    sig = np.asarray(sig, dtype=float)
    ang = np.asarray(ang, dtype=float)
    work = np.asarray(work, dtype=bool)
    keys = np.asarray(keys, dtype=object)
    n = len(y)
    if not (len(sig) == len(ang) == len(work) == len(keys) == n):
        raise ValueError("y, sig, ang, keys and work must describe the same rows")
    if not work.any():
        raise ValueError("nested selection needs work rows")
    if not np.all(np.isfinite(y[work]) & np.isfinite(sig[work]) & np.isfinite(ang[work])):
        raise ValueError("work rows have non-finite y, sigma or angle")
    if any(v.table is not None for v in variants):
        if any(v.table is not None and v.spectra is not None for v in variants):
            raise ValueError("a physics variant gives spectra or a response table, not both")
        if evaluator != "table" or score_with != "table":
            raise ValueError("response-table variants need evaluator='table' and score_with='table' "
                             "(the cf evaluator needs the spectra)")
    row_ids = None
    for v in variants:
        if v.table is not None:
            kind, data = "response table", v.load_table()
            count = validate_table(data, support, hands, f"variant {v.label!r} response table")
        else:
            kind, data = "spectra", v.load()
            count = validate_spectra(data, f"variant {v.label!r} spectra")
        if count != n:
            raise ValueError(f"variant {v.label!r} {kind} have a different number of rows than the row arrays")
        ids = np.asarray(data["row_ids"])
        del data
        if row_ids is None:
            row_ids = ids
        elif not np.array_equal(ids, row_ids):
            raise ValueError(f"variant {v.label!r} {kind} hold different rows from variant {variants[0].label!r}")

    thick = grids.thickness_candidates()
    thick_mu = [(canonical_family(fam), prm, float(mu)) for fam, prm in thick for mu in grids.mu]
    no_mass = [(fam, prm) for fam, prm in thick if support_mass(fam, prm, support.t_min_A, support.t_max_A) <= 0.0]
    if no_mass:
        raise ValueError(f"thickness candidates without mass on the support {support.to_dict()}: {no_mass[:5]}")
    resolved = [resolve_evaluator(fam, prm, evaluator, support.t_min_A, support.t_max_A) for fam, prm, _mu in thick_mu]
    scale_options = grids.scale_candidates(objective)
    inner = hash_fold(list(keys), salt, INNER_FOLDS)
    folds = _fold_contexts(ang, work, inner, sorted({int(o["n_basis"]) for o in scale_options}), clamp)
    base_shared = {"y": y, "sig": sig, "work": work, "folds": folds, "scale_options": scale_options,
                   "objective": objective}

    n_q, n_s = len(thick_mu), len(scale_options)
    scores = {h: np.full((len(variants), n_q, n_s), np.nan) for h in hands}
    cf_scored = [qi for qi, r in enumerate(resolved) if r == "cf" and (score_with == "resolved" or evaluator == "cf")]
    table_scored = [qi for qi in range(n_q) if qi not in set(cf_scored)]
    best_table = {}
    for vi, v in enumerate(variants):
        table = v.load_table()
        spec = v.load() if table is None else None
        if table_scored:
            if table is None:
                table = thickness_table(spec, support, hands=hands, workers=workers)
            res = _parallel_map(_table_score_worker, table_scored, workers,
                                {**base_shared, "table": table, "thick": thick_mu, "hands": hands})
            for qi, by_hand in zip(table_scored, res, strict=True):
                for h in hands:
                    scores[h][vi, qi] = by_hand[h]
        for qi in cf_scored:
            fam, prm, mu = thick_mu[qi]
            cf = cf_row_intensities(spec, {"family": fam, **prm}, mu, hands=hands, workers=workers)
            _SHARED.clear()
            _SHARED.update(base_shared)
            for h in hands:
                scores[h][vi, qi] = _score_dyn(cf[h])
            _SHARED.clear()
        for h in hands:  # keep the table of the variant holding each hand's running best
            if table is not None and (h not in best_table or np.nanmin(scores[h][vi]) < best_table[h][1]):
                best_table[h] = (vi, float(np.nanmin(scores[h][vi])), table)
        del table, spec

    bounds = {k: tuple(float(x) for x in v) for k, v in {**DEFAULT_PHYSICAL_BOUNDS, **(physical_bounds or {})}.items()}
    for name in ("mean", "t0"):
        bounds[name] = tuple(sorted(set(bounds.get(name, ())) | {float(support.t_min_A), float(support.t_max_A)}))
    param_grids = {**grids.param_grids(), "mu": [float(v) for v in grids.mu],
                   "scale.n_basis": [float(v) for v in grids.n_basis], "scale.lambda": [float(v) for v in grids.lam]}
    variant_grids = _variant_param_grids(variants)
    candidates = [{"variant": labels[vi], "family": thick_mu[qi][0] if thick_mu[qi][0] == "gamma" else "wedge",
                   **thick_mu[qi][1], "mu": thick_mu[qi][2], "n_basis": scale_options[si]["n_basis"],
                   "lambda": scale_options[si]["lambda"]}
                  for vi in range(len(variants)) for qi in range(n_q) for si in range(n_s)]

    out = {}
    for h in hands:
        flat = scores[h].reshape(-1)
        if not np.any(np.isfinite(flat)):
            raise RuntimeError("nested selection produced no finite inner-CV score")
        ci = int(np.nanargmin(flat))
        vi, rest = divmod(ci, n_q * n_s)
        qi, si = divmod(rest, n_s)
        fam, prm, mu = thick_mu[qi]
        family_label = "gamma" if fam == "gamma" else "wedge"
        thickness = {"family": family_label, **prm}
        pick_eval = resolved[qi]
        score_eval = "cf" if qi in set(cf_scored) else "table"
        given = variants[vi].load_table()
        spec = variants[vi].load() if given is None else given  # a response table also carries the per-row mu
        if pick_eval == "cf":
            cf = cf_row_intensities(spec, thickness, mu, hands=(h,), workers=workers)
            dyn, born = cf[h], cf["born"]
        else:
            if best_table.get(h, (None,))[0] == vi:
                tab = best_table[h][2]
            else:
                tab = given if given is not None else thickness_table(spec, support, (h,), workers)
            dyn, born = table_row_intensities(tab, thickness, mu, h)
        oof_all = inner_cv_oof(dyn, y, sig, work, folds, [scale_options[si]], objective)
        oof = np.where(work, oof_all[0], np.nan)
        inner_final = _r_all(y, oof, work)
        a_all = np.clip(ang, np.min(ang[work]), np.max(ang[work])) if clamp else ang
        sol = PSplineScaleFitter(n_basis=int(scale_options[si]["n_basis"]), objective=objective).fit(
            y, sig, dyn, a_all, fit_mask=work, lambda_smooth=float(scale_options[si]["lambda"]))
        values = {**prm, "mu": mu, "scale.n_basis": float(scale_options[si]["n_basis"]),
                  "scale.lambda": float(scale_options[si]["lambda"])}
        edges, physical = _edge_flags(values, param_grids, bounds)
        v_params = variants[vi].params or {}
        v_edges, v_physical = _edge_flags({k: v_params[k] for k in variant_grids if k in v_params}, variant_grids, bounds)
        if not variant_grids and len(variants) > 2 and vi in (0, len(variants) - 1):
            v_edges.append("variant")  # the outermost of three or more unlabelled variants is an edge
        pick = {
            "hand": int(h),
            "variant": labels[vi], "variant_index": int(vi), "variant_params": dict(v_params),
            "variant_config": dict(variants[vi].config or {}),
            "thickness": thickness, "mu": float(mu), "scale": dict(scale_options[si]),
            "evaluator": pick_eval, "evaluator_requested": evaluator, "score_evaluator": score_eval,
            "mass_outside_support": mass_outside_support(fam, prm, support.t_min_A, support.t_max_A),
            "cf_mass_tolerance": CF_MASS_TOLERANCE, "support": support.to_dict(),
            "inner_r_all": float(inner_final), "inner_r_all_selection": float(flat[ci]),
            "inner_cv": {"salt": salt, "n_folds": INNER_FOLDS, "n_work": int(work.sum()), "n_rows": int(n),
                         "work_rows_per_fold": [int((work & (inner == f)).sum()) for f in range(INNER_FOLDS)],
                         "clamp": bool(clamp), "objective": objective},
            "edge": edges + v_edges, "at_physical_bound": physical + v_physical,
            "any_edge": bool(edges or v_edges), "grid_size": int(flat.size), "grids": grids.to_dict(),
            "variants": labels, "scale_edf": float(sol["edf"]),
            "spectra_mu_max": float(np.max(np.abs(np.asarray(spec["mu"], dtype=float)))),
        }
        out[h] = NestedSelection(pick=pick, dyn=dyn, born=born, oof=oof, prediction=np.asarray(sol["prediction"]),
                                 scale_edf=float(sol["edf"]), scores=flat.copy(), candidates=candidates)
    return out


# ----------------------------------------------------------------------------- engine mapping


def engine_config_fields(pick: dict, support: ThicknessSupport | None = None) -> dict:
    """Config overrides that make XDSBlochPort reproduce a pick's rows (variant config, thickness, absorption).

    The spectra were split as H = H0 + i*mu_spectra*I and the selection used its own mu, so the absorption fields
    are set to the pick's uniform mu (mean mode) or none.
    """
    sup = support or ThicknessSupport(**pick.get("support", {}))
    fam, params = _split_thickness(pick["thickness"])
    fields = dict(pick.get("variant_config") or {})
    fields.update(thickness_evaluator=str(pick.get("evaluator", "table")), thickness_min_A=float(sup.t_min_A),
                  thickness_max_A=float(sup.t_max_A), thickness_table_dt_A=float(sup.dt_A))
    if fam == "gamma":
        fields.update(thickness_model="gamma", thickness_A=params["mean"], thickness_gamma_shape=params["shape"])
    else:
        fields.update(thickness_model="uniform_gradient", thickness_A=params["t0"], thickness_grad_frac=params["grad"])
    mu = float(pick.get("mu", 0.0))
    fields.update(absorption_mode="mean" if mu > 0.0 else "none", absorption_mu=mu, absorption_frac=0.0,
                  absorption_mu_g2=0.0, absorption_mu_theta2=0.0, absorption_tds_b=0.0)
    return fields
