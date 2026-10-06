"""Physical thickness distributions on a bounded support and their exact evaluators.

Families (engine ``thickness_model`` name first, the reference-implementation name second):

* ``uniform_gradient`` / ``wedge`` (t0, grad): uniform on t0*(1 - grad) .. t0*(1 + grad); grad 0 is one thickness.
* ``gamma`` (mean, shape): p(t) ~ t^(k-1) exp(-t/theta), k = shape, theta = mean / shape.

Every distribution lives on the physical support [t_min, t_max] (default 100 A to 10,000 A, i.e. 10 nm to 1 um)
and is renormalised to unit mass there. Uniform absorption multiplies each node by exp(-4*pi*mu*t) and is not
renormalised (the frame scale absorbs the overall factor), exactly as every engine quadrature does.

Evaluators:

* ``table``: exact hat-function weights on the uniform grid t_min, t_min + dt, ..., t_max (default dt = 2 A), so
  the average is exact for the piecewise-linear interpolant of I(t) (tests/reference/reference_lib.py
  distribution_weights).
* ``cf``: the exact untruncated gamma average through the characteristic function,
  sum_jk c_j conj(c_k) (1 + theta*(alpha - 2*pi*i*s*(lam_j - lam_k)))^(-k), alpha = 4*pi*mu, s = +1 / -1 for the model
  / inverse hand (reference_lib.gamma_cf_intensities). It ignores the support.
* ``auto``: ``cf`` when the gamma mass outside the support is below CF_MASS_TOLERANCE (the untruncated and the
  truncated-renormalised averages then differ by at most that mass, in units of the incident beam), otherwise
  ``table``. A wedge always resolves to ``table``.

The legacy ``quadrature`` evaluator (Gauss-Legendre/Hermite nodes, no support) is not handled here.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_THICKNESS_MIN_A = 100.0
DEFAULT_THICKNESS_MAX_A = 10000.0
DEFAULT_TABLE_DT_A = 2.0
CF_MASS_TOLERANCE = 1e-6
MAX_TABLE_NODES = 1_000_001
THICKNESS_EVALUATORS = ("quadrature", "table", "cf", "auto")
FAMILY_ALIASES = {"wedge": "uniform_gradient", "uniform_gradient": "uniform_gradient", "gamma": "gamma"}


def canonical_family(family: str) -> str:
    """Engine family name for a reference or engine family label; unknown families fail closed."""
    key = str(family).strip().lower()
    if key not in FAMILY_ALIASES:
        raise ValueError(f"unknown thickness family {family!r}; expected gamma, wedge or uniform_gradient")
    return FAMILY_ALIASES[key]


def validate_support(t_min: float, t_max: float, dt: float) -> int:
    """Number of table cells of [t_min, t_max] at step dt; the span must be an integer number of steps."""
    t_min, t_max, dt = float(t_min), float(t_max), float(dt)
    if not all(math.isfinite(v) for v in (t_min, t_max, dt)):
        raise ValueError("thickness support bounds and table step must be finite")
    if t_min < 0.0:
        raise ValueError("thickness support minimum must be non-negative")
    if t_max <= t_min:
        raise ValueError("thickness support maximum must exceed its minimum")
    if dt <= 0.0:
        raise ValueError("thickness table step must be positive")
    cells = (t_max - t_min) / dt
    n = int(round(cells))
    if n < 1 or abs(cells - n) > 1e-9 * max(1.0, cells):
        raise ValueError(
            f"thickness support [{t_min:g}, {t_max:g}] A is not an integer number of {dt:g} A table steps"
        )
    if n + 1 > MAX_TABLE_NODES:
        raise ValueError(f"thickness table would have {n + 1} nodes (limit {MAX_TABLE_NODES}); use a coarser step")
    return n


def support_grid(t_min: float, t_max: float, dt: float) -> np.ndarray:
    """Uniform table nodes t_min, t_min + dt, ..., t_max (last node exactly t_max)."""
    n = validate_support(t_min, t_max, dt)
    t = float(t_min) + float(dt) * np.arange(n + 1, dtype=float)
    t[-1] = float(t_max)
    return t


def _gamma_moments(a, b, k: float, theta: float):
    """Integral of p and of t*p over [a, b] for a gamma(k, theta) density (vectorised)."""
    from scipy.special import gammainc

    fa, fb = gammainc(k, np.maximum(a, 0.0) / theta), gammainc(k, np.maximum(b, 0.0) / theta)
    ga, gb = gammainc(k + 1.0, np.maximum(a, 0.0) / theta), gammainc(k + 1.0, np.maximum(b, 0.0) / theta)
    return fb - fa, k * theta * (gb - ga)


def _hat_weights_from_moments(t: np.ndarray, m0: np.ndarray, m1: np.ndarray) -> np.ndarray:
    """w_j = int p(t) phi_j(t) dt from per-cell integrals of p (m0) and t*p (m1); phi_j are the hat functions."""
    dt = t[1:] - t[:-1]
    right = (t[1:] * m0 - m1) / dt      # contribution to the left node of each cell
    left = (m1 - t[:-1] * m0) / dt      # contribution to the right node of each cell
    w = np.zeros(len(t), dtype=float)
    w[:-1] += right
    w[1:] += left
    return w


def gamma_parameters(params: dict) -> tuple[float, float]:
    """(shape k, scale theta) of a gamma(mean, shape) distribution; fail closed on non-physical values."""
    mean, shape = float(params["mean"]), float(params["shape"])
    if not (math.isfinite(mean) and mean > 0.0):
        raise ValueError(f"gamma thickness mean must be finite and positive, got {mean!r}")
    if not (math.isfinite(shape) and shape > 0.0):
        raise ValueError(f"gamma thickness shape must be finite and positive, got {shape!r}")
    return shape, mean / shape


def wedge_bounds(params: dict) -> tuple[float, float]:
    """Return the wedge support (t0*(1 - grad), t0*(1 + grad)) in Å; t0 must be positive and grad in [0, 1]."""
    t0, grad = float(params["t0"]), float(params["grad"])
    if not (math.isfinite(t0) and t0 > 0.0):
        raise ValueError(f"wedge thickness t0 must be finite and positive, got {t0!r}")
    if not (math.isfinite(grad) and 0.0 <= grad <= 1.0):
        raise ValueError(f"wedge thickness gradient must lie in [0, 1], got {grad!r}")
    return t0 * (1.0 - grad), t0 * (1.0 + grad)


def support_mass(family: str, params: dict, t_min: float, t_max: float) -> float:
    """Probability mass of the (untruncated) distribution inside [t_min, t_max]."""
    fam = canonical_family(family)
    if fam == "gamma":
        from scipy.special import gammainc, gammaincc

        k, theta = gamma_parameters(params)
        return float(max(0.0, 1.0 - gammainc(k, float(t_min) / theta) - gammaincc(k, float(t_max) / theta)))
    lo, hi = wedge_bounds(params)
    if hi == lo:
        return float(t_min <= lo <= t_max)
    return float(max(0.0, min(hi, t_max) - max(lo, t_min)) / (hi - lo))


def mass_outside_support(family: str, params: dict, t_min: float, t_max: float) -> float:
    """Probability mass outside [t_min, t_max]; for gamma computed from both tails without cancellation."""
    fam = canonical_family(family)
    if fam == "gamma":
        from scipy.special import gammainc, gammaincc

        k, theta = gamma_parameters(params)
        return float(gammainc(k, float(t_min) / theta) + gammaincc(k, float(t_max) / theta))
    return float(1.0 - support_mass(fam, params, t_min, t_max))


def resolve_evaluator(family: str, params: dict, requested: str, t_min: float, t_max: float) -> str:
    """'table' or 'cf' for a support evaluator request ('table', 'cf', 'auto'); fail closed otherwise."""
    fam = canonical_family(family)
    req = str(requested).strip().lower()
    if req == "table":
        return "table"
    if req == "cf":
        if fam != "gamma":
            raise ValueError("thickness evaluator cf (gamma characteristic function) is defined for the gamma family only")
        return "cf"
    if req == "auto":
        if fam == "gamma" and mass_outside_support(fam, params, t_min, t_max) < CF_MASS_TOLERANCE:
            return "cf"
        return "table"
    raise ValueError(f"unknown support thickness evaluator {requested!r}; expected table, cf or auto")


def distribution_weights(family: str, params: dict, t: np.ndarray, mu: float = 0.0) -> np.ndarray:
    """Exact hat-function weights on the support grid t, renormalised to unit mass, times exp(-4*pi*mu*t_j).

    t must be a uniform grid (support_grid). A wedge with grad 0 is one thickness: its weights are the linear
    interpolation weights of that thickness. Zero mass on the support fails closed.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("thickness table grid needs at least two nodes")
    a, b = t[:-1], t[1:]
    fam = canonical_family(family)
    if fam == "gamma":
        k, theta = gamma_parameters(params)
        m0, m1 = _gamma_moments(a, b, k, theta)
        w = _hat_weights_from_moments(t, m0, m1)
    else:
        lo, hi = wedge_bounds(params)
        if hi == lo:
            w = np.zeros(len(t), dtype=float)
            if t[0] <= lo <= t[-1]:
                j = min(int(np.searchsorted(t, lo, side="right")) - 1, len(t) - 2)
                frac = (lo - t[j]) / (t[j + 1] - t[j])
                w[j] += 1.0 - frac
                w[j + 1] += frac
        else:
            x1 = np.clip(a, lo, hi)
            x2 = np.clip(b, lo, hi)
            width = hi - lo
            m0 = (x2 - x1) / width
            m1 = 0.5 * (x2 ** 2 - x1 ** 2) / width
            w = _hat_weights_from_moments(t, m0, m1)
    w = np.maximum(w, 0.0)  # hat weights are non-negative; this only removes rounding below zero
    total = float(np.sum(w))
    if not (math.isfinite(total) and total > 0.0):
        raise ValueError(
            f"thickness distribution {fam} {dict(params)} has no mass on the support [{t[0]:g}, {t[-1]:g}] A"
        )
    w = w / total
    if float(mu) != 0.0:
        w = w * np.exp(-4.0 * np.pi * float(mu) * t)
    return w


def distribution_weight_derivatives(family: str, params: dict, t: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Table weights and derivatives of the support-renormalised distribution.

    Gamma returns the derivative with respect to its nominal mean at fixed
    shape. A wedge returns derivatives for its centre and gradient fraction.
    Nodes and support stay fixed. Uniform absorption is applied by propagation,
    not included here. At the zero-width wedge limit, gradient stays fixed and
    the centre uses the local linear-interpolation slope (right side at a knot).
    """
    t = np.asarray(t, dtype=float)
    weights = distribution_weights(family, params, t)
    if canonical_family(family) == "gamma":
        shape, theta = gamma_parameters(params)
        mean = float(params["mean"])
        moment0, moment1 = _gamma_moments(t[:-1], t[1:], shape, theta)
        moment2 = shape * theta * _gamma_moments(t[:-1], t[1:], shape + 1, theta)[1]
        raw = _hat_weights_from_moments(t, moment0, moment1)
        derivative = _hat_weights_from_moments(
            t, shape * (moment1 / mean**2 - moment0 / mean),
            shape * (moment2 / mean**2 - moment1 / mean),
        )
        # Match the forward evaluator's removal of negative rounding residue.
        derivative[raw <= 0] = 0
        mass = np.maximum(raw, 0).sum()
        derivative = (derivative - weights * derivative.sum()) / mass
        return weights, {"mean": derivative}

    low, high = wedge_bounds(params)
    centre, gradient = float(params["t0"]), float(params["grad"])
    if low == high:
        index = min(int(np.searchsorted(t, centre, side="right")) - 1, len(t) - 2)
        derivative = np.zeros_like(t)
        derivative[index:index + 2] = [-1.0, 1.0]
        derivative /= t[index + 1] - t[index]
        return weights, {"t0": derivative, "grad": np.zeros_like(t)}

    lower, upper = max(low, t[0]), min(high, t[-1])
    if lower == 0:
        lower_hat = np.zeros_like(t)
        lower_hat[0] = 1.0
    else:
        lower_hat = distribution_weights("wedge", {"t0": lower, "grad": 0.0}, t)
    upper_hat = distribution_weights("wedge", {"t0": upper, "grad": 0.0}, t)
    lower_moves, upper_moves = float(low > t[0]), float(high < t[-1])

    def derivative(lower_step, upper_step):
        return (upper_hat * upper_step - lower_hat * lower_step - weights * (upper_step - lower_step)) / (upper - lower)

    return weights, {
        "t0": derivative((1 - gradient) * lower_moves, (1 + gradient) * upper_moves),
        "grad": derivative(-centre * lower_moves, centre * upper_moves),
    }


def gamma_cf_profile(lam, coupling, shape: float, theta: float, alpha: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Exact gamma(shape, theta) average of |sum_j c_j exp(2*pi*i*s*lam_j*t)|^2 exp(-alpha*t), s = +1 and -1.

    lam (P, N) are the Hermitian eigenvalues and coupling (P, N) c_j = V[1, j] * conj(V[0, j]) of P profile-point
    matrices. The double sum sum_jk c_j conj(c_k) g(lam_j - lam_k) with g(x) = (1 + theta*(alpha - 2*pi*i*x))^(-k)
    is folded onto j < k (g(-x) = conj(g(x))), which halves the complex powers. The inverse hand uses conj(g),
    exact because 1 + theta*alpha > 0. Returns (model, inverse), each of shape (P,).
    """
    lam = np.atleast_2d(np.asarray(lam, dtype=float))
    c = np.atleast_2d(np.asarray(coupling, dtype=np.complex128))
    if lam.shape != c.shape:
        raise ValueError("gamma characteristic function needs eigenvalues and couplings of one shape")
    base = 1.0 + float(theta) * float(alpha)
    diagonal = np.sum(np.abs(c) ** 2, axis=1) * base ** (-float(shape))
    iu, ju = np.triu_indices(lam.shape[1], k=1)
    g = (base - 2j * np.pi * float(theta) * (lam[:, iu] - lam[:, ju])) ** (-float(shape))
    cc = c[:, iu] * np.conj(c[:, ju])
    model = diagonal + 2.0 * np.sum(np.real(cc * g), axis=1)
    inverse = diagonal + 2.0 * np.sum(np.real(cc * np.conj(g)), axis=1)
    return model, inverse


def gamma_cf_point(lam, coupling, shape: float, theta: float, alpha: float = 0.0) -> tuple[float, float]:
    """gamma_cf_profile for one profile-point matrix: (model, inverse) intensities."""
    model, inverse = gamma_cf_profile(np.asarray(lam)[None, :], np.asarray(coupling)[None, :], shape, theta, alpha)
    return float(model[0]), float(inverse[0])


def uniform_grid_spec(t, rtol: float = 1e-9) -> tuple[float, float, int] | None:
    """(t0, dt, n) when the nodes t form a uniform grid of at least two nodes (step within rtol), else None."""
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or len(t) < 2:
        return None
    dt = float(t[-1] - t[0]) / float(len(t) - 1)
    if not (math.isfinite(dt) and dt > 0.0) or not np.allclose(np.diff(t), dt, rtol=rtol, atol=0.0):
        return None
    return float(t[0]), dt, int(len(t))


def uniform_grid_phase_factors(lam, t0: float, dt: float, n: int, *, inverse: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Within-block and block-offset phases for t=t0+(b*B+r)*dt, with B near sqrt(n).

    Shapes are (profiles, beams, B) and (profiles, blocks, beams). Their product represents the original
    exponential exactly to rounding; no interpolation, binning or change of thickness weights is involved.
    """

    lam = np.atleast_2d(np.asarray(lam, dtype=float))
    n = int(n)
    if n < 1:
        raise ValueError("uniform-grid amplitudes need at least one node")
    sign = -1.0 if inverse else 1.0
    block = max(1, math.isqrt(n))
    n_blocks = -(-n // block)
    within = float(t0) + float(dt) * np.arange(block, dtype=float)
    offsets = float(dt) * float(block) * np.arange(n_blocks, dtype=float)
    E = np.exp((sign * 2j * np.pi) * lam[:, :, None] * within[None, None, :])        # (P, N, B)
    shift = np.exp((sign * 2j * np.pi) * lam[:, None, :] * offsets[None, :, None])   # (P, nb, N)
    return E, shift


def uniform_grid_amplitudes(lam, coupling, t0: float, dt: float, n: int, *, inverse: bool = False) -> np.ndarray:
    """sum_j c_j exp(2*pi*i*s*lam_j*(t0+m*dt)), shape (profiles, n); s=-1 for the inverse hand.

    Factor the phases into about sqrt(n) blocks. One matrix product per profile replaces beams*n complex
    exponentials, while preserving the same finite-grid sum to rounding.
    """

    lam = np.atleast_2d(np.asarray(lam, dtype=float))
    c = np.atleast_2d(np.asarray(coupling, dtype=np.complex128))
    if lam.shape != c.shape:
        raise ValueError("uniform-grid amplitudes need eigenvalues and couplings of one shape")
    within, shift = uniform_grid_phase_factors(lam, t0, dt, n, inverse=inverse)
    amplitude = np.matmul(shift * c[:, None, :], within)
    return amplitude.reshape(lam.shape[0], shift.shape[1] * within.shape[2])[:, :int(n)]


def gamma_cf_born(h10_abs2, delta, shape: float, theta: float) -> np.ndarray:
    """Exact untruncated gamma average of the Born row term (2*pi)^2 |H10|^2 t^2 sinc^2(delta*t).

    With x = 2*pi*theta*delta the average is 2|H10|^2 (1 - Re (1 - i*x)^(-k)) / delta^2, evaluated without
    cancellation; |x|*(k + 3) < 1e-3 uses the series (2*pi)^2 |H10|^2 k(k+1) theta^2 (1 - x^2 (k+2)(k+3)/12).
    """
    h10_abs2 = np.asarray(h10_abs2, dtype=float)
    delta = np.asarray(delta, dtype=float)
    k, theta = float(shape), float(theta)
    x = 2.0 * np.pi * theta * delta
    small = np.abs(x) * (k + 3.0) < 1e-3
    series = (2.0 * np.pi) ** 2 * h10_abs2 * k * (k + 1.0) * theta ** 2 * (1.0 - x ** 2 * (k + 2.0) * (k + 3.0) / 12.0)
    xs = np.where(small, 1.0, x)
    ds = np.where(small, 1.0, delta)
    amp = np.exp(-0.5 * k * np.log1p(xs ** 2))
    bracket = -np.expm1(-0.5 * k * np.log1p(xs ** 2)) + 2.0 * amp * np.sin(0.5 * k * np.arctan(xs)) ** 2
    exact = 2.0 * h10_abs2 * bracket / ds ** 2
    return np.where(small, series, exact)


def config_distribution(cfg) -> tuple[str, dict]:
    """(engine family, params) of a Config whose thickness evaluator uses the support."""
    model = str(getattr(cfg, "thickness_model", "uniform_gradient") or "uniform_gradient").lower()
    if model == "gamma":
        return "gamma", {"mean": float(cfg.thickness_A), "shape": float(getattr(cfg, "thickness_gamma_shape", 1.0))}
    if model == "uniform_gradient":
        return "uniform_gradient", {"t0": float(cfg.thickness_A), "grad": float(cfg.thickness_grad_frac)}
    raise ValueError(
        f"thickness_model={model!r} has no support evaluator; the table/cf/auto evaluators need uniform_gradient or gamma"
    )


def config_support(cfg) -> tuple[float, float, float]:
    """Return a Config's thickness support (t_min, t_max, dt) in Å, with the module defaults for missing fields."""
    return (
        float(getattr(cfg, "thickness_min_A", DEFAULT_THICKNESS_MIN_A)),
        float(getattr(cfg, "thickness_max_A", DEFAULT_THICKNESS_MAX_A)),
        float(getattr(cfg, "thickness_table_dt_A", DEFAULT_TABLE_DT_A)),
    )


def resolved_config_evaluator(cfg) -> str:
    """'quadrature', 'table' or 'cf' for a Config (the legacy quadrature needs no support)."""
    requested = str(getattr(cfg, "thickness_evaluator", "quadrature") or "quadrature").lower()
    model = str(getattr(cfg, "thickness_model", "uniform_gradient") or "uniform_gradient").lower()
    if requested == "quadrature":
        if model == "gamma":
            raise ValueError("thickness_model=gamma needs thickness_evaluator table, cf or auto (it has no quadrature)")
        return "quadrature"
    family, params = config_distribution(cfg)
    t_min, t_max, _dt = config_support(cfg)
    return resolve_evaluator(family, params, requested, t_min, t_max)


def config_table_nodes(cfg) -> tuple[np.ndarray, np.ndarray]:
    """Support-grid nodes with non-zero hat weights and those weights (unit mass, no absorption) for a Config."""
    family, params = config_distribution(cfg)
    t = support_grid(*config_support(cfg))
    w = distribution_weights(family, params, t)
    keep = w > 0.0
    return t[keep], w[keep]


def config_cf_parameters(cfg) -> dict | None:
    """{'shape', 'theta'} when a Config resolves to the cf evaluator, else None."""
    if resolved_config_evaluator(cfg) != "cf":
        return None
    _family, params = config_distribution(cfg)
    k, theta = gamma_parameters(params)
    return {"shape": k, "theta": theta}


def config_thickness_contract(cfg) -> dict:
    """Machine-readable record of the thickness distribution and the evaluator a Config resolves to."""
    requested = str(getattr(cfg, "thickness_evaluator", "quadrature") or "quadrature").lower()
    resolved = resolved_config_evaluator(cfg)
    out = {
        "thickness_model": str(getattr(cfg, "thickness_model", "uniform_gradient")),
        "evaluator_requested": requested,
        "evaluator": resolved,
    }
    if resolved == "quadrature":
        return out
    family, params = config_distribution(cfg)
    t_min, t_max, dt = config_support(cfg)
    out.update(
        family=family,
        params=params,
        support_A=[t_min, t_max],
        table_dt_A=dt,
        mass_outside_support=mass_outside_support(family, params, t_min, t_max),
        cf_mass_tolerance=CF_MASS_TOLERANCE,
    )
    return out
