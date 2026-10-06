"""Thickness-domain checks and exact reuse of fixed row/profile eigensystems. This is a local polish within the selected basin; a stable fit does not imply an identifiable physical thickness."""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss

from grond.bloch.numerics import hermitian_eigh
from grond.legacy.engine.bloch_engine import (
    Config,
    XDSBlochPort,
)

PHYSICAL_GRAD_MIN = 0.0


PHYSICAL_GRAD_MAX = 1.0


PACKED_PROPAGATION_CHUNK = 2048


class ResponseRecords(list):
    """List-compatible response cache with an exact packed propagation view."""

    def __init__(self, records=(), *, packed: dict | None = None):
        """Hold the records and the optional packed view (pack_response_records); packed_status starts not_built."""

        super().__init__(records)
        self.packed = packed
        self.packed_status: dict[str, object] = {"status": "not_built"}


def estimated_packed_response_bytes(records: list[dict]) -> int:
    """Estimate additional packed-array bytes before allocating them."""

    total = 0
    for record in records:
        total += np.asarray(record["profile_weights"], dtype=float).nbytes
        for dynamic, _born in zip(record["dyn_specs"], record["born_specs"], strict=True):
            if "matrix" in dynamic:
                total += np.asarray(dynamic["matrix"], dtype=np.complex128).nbytes
            else:
                total += np.asarray(dynamic["vals"], dtype=np.complex128).nbytes
                total += np.asarray(dynamic["coupling"], dtype=np.complex128).nbytes
            total += 16
    return int(total + 32 * len(records))


def inclusive_bounded_axis(lower: float, upper: float, step: float) -> np.ndarray:
    """Return a deterministic inclusive grid that never crosses its bounds."""

    lower = float(lower)
    upper = float(upper)
    step = float(step)
    if not all(math.isfinite(value) for value in (lower, upper, step)):
        raise ValueError("bounded grid limits and step must be finite")
    if upper < lower:
        raise ValueError("bounded grid upper limit must be at least its lower limit")
    if step <= 0.0:
        raise ValueError("bounded grid step must be positive")
    if upper == lower:
        return np.asarray([lower], dtype=float)
    count = int(math.floor((upper - lower) / step))
    values = lower + step * np.arange(count + 1, dtype=float)
    tolerance = max(1e-12, 1e-12 * max(abs(lower), abs(upper), 1.0))
    values = values[values <= upper + tolerance]
    if not len(values) or values[-1] < upper - tolerance:
        values = np.append(values, upper)
    else:
        values[-1] = upper
    return np.asarray(values, dtype=float)


def local_powell_thickness_bounds(
    sampled_thicknesses: np.ndarray,
    winner: float,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    """Bracket Powell by the immediate sampled neighbors of the guard-grid winner."""

    samples = np.asarray(sorted(set(float(value) for value in sampled_thicknesses)), dtype=float)
    if not len(samples):
        return float(lower), float(upper)
    index = int(np.argmin(np.abs(samples - float(winner))))
    local_lower = float(samples[index - 1]) if index > 0 else float(lower)
    local_upper = float(samples[index + 1]) if index + 1 < len(samples) else float(upper)
    if local_upper <= local_lower:
        return float(lower), float(upper)
    return local_lower, local_upper


def local_thickness_search_bounds(
    start_thickness: float,
    bracket_frac: float,
    *,
    t_min: float | None = None,
    t_max: float | None = None,
) -> tuple[float, float]:
    """Return the local thickness basin established by the coarse-grid winner.

    Dynamical intensities can have separated thickness aliases.  This stage is
    therefore a local polish: the fractional bracket selects the basin, while
    optional hard bounds may only make that basin narrower.  The old
    400--2600 A defaults were numerical limits rather than physical limits and
    must not reject an otherwise valid coarse-grid winner.
    """

    start = float(start_thickness)
    bracket = float(bracket_frac)
    if not math.isfinite(start) or start <= 0.0:
        raise ValueError("start thickness must be finite and positive")
    if not math.isfinite(bracket) or not 0.0 < bracket < 1.0:
        raise ValueError("thickness bracket fraction must satisfy 0 < bracket_frac < 1")

    lo = start * (1.0 - bracket)
    hi = start * (1.0 + bracket)
    if t_min is not None:
        hard_min = float(t_min)
        if not math.isfinite(hard_min) or hard_min <= 0.0:
            raise ValueError("t_min must be finite and positive")
        lo = max(lo, hard_min)
    if t_max is not None:
        hard_max = float(t_max)
        if not math.isfinite(hard_max) or hard_max <= 0.0:
            raise ValueError("t_max must be finite and positive")
        hi = min(hi, hard_max)
    if not lo <= start <= hi:
        raise ValueError("explicit thickness bounds must include the starting thickness")
    if not lo < hi:
        raise ValueError("local thickness search bounds must span a nonzero interval")
    return float(lo), float(hi)


def thickness_candidate_domain_reasons(
    t0: float,
    grad: float,
    *,
    t_min: float = 0.0,
    t_max: float = math.inf,
    grad_min: float = PHYSICAL_GRAD_MIN,
    grad_max: float = PHYSICAL_GRAD_MAX,
) -> list[str]:
    """Return fail-closed reasons why a continuous-thickness point is invalid.

    A uniform-gradient fraction above one permits non-positive crystal
    thickness at the edge of the distribution and is rejected by the Bloch
    engine.  Keep that physical domain independent of user search bounds so an
    optimizer result can never be promoted merely because a wider numerical
    bound was requested.
    """

    reasons: list[str] = []
    t0 = float(t0)
    grad = float(grad)
    t_min = float(t_min)
    t_max = float(t_max)
    grad_min = float(grad_min)
    grad_max = float(grad_max)
    if not math.isfinite(t0):
        reasons.append("nonfinite_thickness")
    elif t0 <= 0.0:
        reasons.append("nonpositive_thickness")
    elif not t_min <= t0 <= t_max:
        reasons.append("thickness_outside_search_bounds")
    if not math.isfinite(grad):
        reasons.append("nonfinite_thickness_gradient")
    else:
        if not PHYSICAL_GRAD_MIN <= grad <= PHYSICAL_GRAD_MAX:
            reasons.append("thickness_gradient_outside_physical_domain")
        if not grad_min <= grad <= grad_max:
            reasons.append("thickness_gradient_outside_search_bounds")
    return reasons


def require_physical_thickness_candidate(t0: float, grad: float) -> None:
    """Raise before constructing a nuisance model outside the engine domain."""

    reasons = thickness_candidate_domain_reasons(t0, grad)
    if reasons:
        raise ValueError(
            f"invalid continuous-thickness candidate t={float(t0)!r}, grad={float(grad)!r}: "
            + ", ".join(reasons)
        )


def thickness_grid(t0: float, grad: float, points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return GROND's uniform-gradient thickness quadrature for arbitrary t0."""

    t0 = float(t0)
    grad = float(grad)
    require_physical_thickness_candidate(t0, grad)
    points = max(1, int(points))
    if points <= 1 or abs(grad) <= 0.0:
        return np.asarray([t0], dtype=float), np.asarray([1.0], dtype=float)
    x, w = leggauss(points)
    t = t0 * (1.0 + grad * x)
    ok = t > 0.0
    if not np.any(ok):
        return np.asarray([t0], dtype=float), np.asarray([1.0], dtype=float)
    t = np.asarray(t[ok], dtype=float)
    w = np.asarray(w[ok], dtype=float)
    return t, w / np.sum(w)


def build_base_config(args) -> Config:
    """Create a GROND Bloch config whose non-thickness terms define the contract."""

    return Config(
        n_basis=int(args.n_basis),
        n_points=int(args.n_points),
        sg_window=float(args.sg_window),
        sg_window_mode=str(args.sg_window_mode),
        sg_window_min=float(args.sg_window_min),
        sg_window_max=float(args.sg_window_max),
        sg_window_scale=float(args.sg_window_scale),
        sg_window_frame_floor=float(args.sg_window_frame_floor),
        profile_quadrature=str(args.profile_quadrature),
        profile_accumulation=str(args.profile_accumulation),
        topup_d_min=float(args.topup_d_min),
        max_support_sg=float(args.max_support_sg),
        thickness_A=float(args.start_thickness),
        thickness_model="uniform_gradient",
        thickness_grad_frac=float(args.start_grad),
        thickness_points=int(args.thickness_points),
        group_width_deg=float(args.group_width_deg),
        frame_offset=float(args.frame_offset),
        matrix_convention=str(args.matrix_convention),
        free_fraction=float(args.free_fraction),
        split_seed=int(args.split_seed),
        double_precision=bool(getattr(args, "double_precision", False)),
    )


def propagation_spec(H: np.ndarray, absorptive: bool) -> dict[str, np.ndarray]:
    """Cache a stable direct-to-target propagator for one Hamiltonian."""

    if absorptive:
        spectral = XDSBlochPort._nonhermitian_spectral_target_coupling(H)
        if spectral is None:
            return {
                "matrix": np.asarray(H),
                "absorptive": True,
            }
        vals, coupling = spectral
    else:
        vals, vecs = hermitian_eigh(H)
        coupling = vecs[1, :] * np.conj(vecs[0, :])
    return {
        "vals": np.asarray(vals),
        "coupling": np.asarray(coupling),
        "absorptive": bool(absorptive),
    }


def dyn_from_spec(spec: dict[str, np.ndarray], thick: np.ndarray, weights: np.ndarray) -> float:
    """Return the thickness-weighted dynamical intensity of one profile point.

    The intensity is sum over t of weights |sum_j c_j exp(2 pi i lambda_j t)|^2, with t (thick) in A. A spec holding
    a non-Hermitian matrix is propagated directly instead.
    """

    if "matrix" in spec:
        per_t = XDSBlochPort._nonhermitian_target_intensities(spec["matrix"], thick)
        return float(np.sum(weights * per_t))
    vals = spec["vals"]
    coupling = spec["coupling"]
    phase = np.exp((2j * np.pi) * vals[:, None] * thick[None, :])
    per_t = np.abs(np.sum(coupling[:, None] * phase, axis=0)) ** 2
    return float(np.sum(weights * per_t))


def born_from_spec(h10_abs2: float, delta: float, thick: np.ndarray, weights: np.ndarray) -> float:
    """Return the thickness-weighted kinematic (Born) intensity of one profile point.

    The intensity is sum over t of weights (2 pi)^2 |H10|^2 t^2 (sin x / x)^2, with x = pi delta t.
    """

    arg = np.pi * float(delta) * thick
    sinc = np.ones_like(arg, dtype=float)
    nz = np.abs(arg) >= 1e-12
    sinc[nz] = np.sin(arg[nz]) / arg[nz]
    per_t = (2.0 * math.pi) ** 2 * float(h10_abs2) * (thick**2) * (sinc**2)
    return float(np.sum(weights * per_t))


def pack_response_records(records: list[dict]) -> dict:
    """Pack spectral profile records for chunked phase-only propagation."""

    phase_buckets: dict[int, dict[str, list]] = {}
    fallback_specs: list[tuple[int, dict]] = []
    profile_weights: list[float] = []
    born_h10_abs2: list[float] = []
    born_delta: list[float] = []
    offsets = [0]
    row_indices = []
    basis_sizes = []
    for record in records:
        weights = np.asarray(record["profile_weights"], dtype=float)
        dyn_specs = list(record["dyn_specs"])
        born_specs = list(record["born_specs"])
        if not (len(weights) == len(dyn_specs) == len(born_specs)):
            raise ValueError("response profile weights and dynamical/Born spectra have different lengths")
        row_indices.append(int(record["row_index"]))
        basis_sizes.append(int(record["basis_size"]))
        for weight, dyn_spec, born_spec in zip(weights, dyn_specs, born_specs, strict=True):
            flat_position = len(profile_weights)
            profile_weights.append(float(weight))
            born_h10_abs2.append(float(born_spec["h10_abs2"]))
            born_delta.append(float(born_spec["delta"]))
            if "matrix" in dyn_spec:
                fallback_specs.append((flat_position, dyn_spec))
                continue
            values = np.asarray(dyn_spec["vals"])
            coupling = np.asarray(dyn_spec["coupling"])
            if values.ndim != 1 or coupling.ndim != 1 or values.shape != coupling.shape:
                raise ValueError("spectral response values and coupling must be matching vectors")
            bucket = phase_buckets.setdefault(
                len(values), {"positions": [], "phase_rates": [], "couplings": []}
            )
            bucket["positions"].append(flat_position)
            bucket["phase_rates"].append((2j * np.pi) * values)
            bucket["couplings"].append(coupling)
        offsets.append(len(profile_weights))
    packed_buckets = {
        int(size): {
            "positions": np.asarray(bucket["positions"], dtype=int),
            "phase_rates": np.stack(bucket["phase_rates"]),
            "couplings": np.stack(bucket["couplings"]),
        }
        for size, bucket in phase_buckets.items()
    }
    return {
        "schema_version": 1,
        "phase_buckets": packed_buckets,
        "fallback_specs": fallback_specs,
        "profile_weights": np.asarray(profile_weights, dtype=float),
        "born_h10_abs2": np.asarray(born_h10_abs2, dtype=float),
        "born_delta": np.asarray(born_delta, dtype=float),
        "profile_offsets": np.asarray(offsets, dtype=int),
        "row_indices": np.asarray(row_indices, dtype=int),
        "basis_sizes": np.asarray(basis_sizes, dtype=int),
        "chunk_size": PACKED_PROPAGATION_CHUNK,
    }


def _reduce_packed_profiles(values: np.ndarray, packed: dict) -> np.ndarray:
    """Sum per-profile-point values into per-row values with the packed profile weights."""

    weights = np.asarray(packed["profile_weights"], dtype=float)
    offsets = np.asarray(packed["profile_offsets"], dtype=int)
    reduced = np.zeros(len(offsets) - 1, dtype=float)
    for row_position, (lower, upper) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        reduced[row_position] = float(np.sum(weights[lower:upper] * values[lower:upper]))
    return reduced


def predict_from_packed_response(packed: dict, n_rows: int, t0: float, grad: float, points: int) -> dict:
    """Evaluate exactly the same spectral response in bounded NumPy chunks."""

    thick, thick_w = thickness_grid(float(t0), float(grad), int(points))
    profile_count = len(np.asarray(packed["profile_weights"]))
    dyn_profiles = np.zeros(profile_count, dtype=float)
    chunk_size = max(1, int(packed.get("chunk_size", PACKED_PROPAGATION_CHUNK)))
    for basis_size in sorted(packed["phase_buckets"]):
        bucket = packed["phase_buckets"][basis_size]
        positions = np.asarray(bucket["positions"], dtype=int)
        rates = np.asarray(bucket["phase_rates"])
        couplings = np.asarray(bucket["couplings"])
        for lower in range(0, len(positions), chunk_size):
            upper = min(lower + chunk_size, len(positions))
            phase = np.exp(rates[lower:upper, :, None] * thick[None, None, :])
            amplitude = np.sum(couplings[lower:upper, :, None] * phase, axis=1)
            dyn_profiles[positions[lower:upper]] = np.sum(
                thick_w[None, :] * np.abs(amplitude) ** 2, axis=1
            )
    for position, spec in packed["fallback_specs"]:
        dyn_profiles[int(position)] = dyn_from_spec(spec, thick, thick_w)

    delta = np.asarray(packed["born_delta"], dtype=float)
    h10_abs2 = np.asarray(packed["born_h10_abs2"], dtype=float)
    born_profiles = np.zeros(profile_count, dtype=float)
    for lower in range(0, profile_count, chunk_size):
        upper = min(lower + chunk_size, profile_count)
        arg = np.pi * delta[lower:upper, None] * thick[None, :]
        sinc = np.ones_like(arg, dtype=float)
        nonzero = np.abs(arg) >= 1e-12
        sinc[nonzero] = np.sin(arg[nonzero]) / arg[nonzero]
        per_t = (
            (2.0 * math.pi) ** 2
            * h10_abs2[lower:upper, None]
            * thick[None, :] ** 2
            * sinc**2
        )
        born_profiles[lower:upper] = np.sum(thick_w[None, :] * per_t, axis=1)
    return {
        "indices": np.asarray(packed["row_indices"], dtype=int),
        "dyn": _reduce_packed_profiles(dyn_profiles, packed),
        "born": _reduce_packed_profiles(born_profiles, packed),
        "basis_size": np.asarray(packed["basis_sizes"], dtype=int),
        "response_rows_total": int(n_rows),
    }


def predict_from_response_scalar(records: list[dict], n_rows: int, t0: float, grad: float, points: int) -> dict:
    """Reference scalar-loop response evaluation retained for exact fallback."""

    thick, thick_w = thickness_grid(float(t0), float(grad), int(points))
    indices = np.asarray([r["row_index"] for r in records], dtype=int)
    dyn = np.zeros(len(records), dtype=float)
    born = np.zeros(len(records), dtype=float)
    basis = np.zeros(len(records), dtype=int)
    for i, rec in enumerate(records):
        pw = np.asarray(rec["profile_weights"], dtype=float)
        dyn_points = [
            dyn_from_spec(s, thick, thick_w)
            for s in rec["dyn_specs"]
        ]
        born_points = [
            born_from_spec(s["h10_abs2"], s["delta"], thick, thick_w)
            for s in rec["born_specs"]
        ]
        dyn[i] = float(np.sum(pw * np.asarray(dyn_points, dtype=float)))
        born[i] = float(np.sum(pw * np.asarray(born_points, dtype=float)))
        basis[i] = int(rec["basis_size"])
    return {
        "indices": indices,
        "dyn": dyn,
        "born": born,
        "basis_size": basis,
        "response_rows_total": int(n_rows),
    }


def predict_from_response(records: list[dict], n_rows: int, t0: float, grad: float, points: int) -> dict:
    """Return a native prediction, using exact packed propagation when available."""

    packed = getattr(records, "packed", None)
    if packed is not None:
        return predict_from_packed_response(packed, n_rows, t0, grad, points)
    return predict_from_response_scalar(records, n_rows, t0, grad, points)


def predict_candidates_from_response(
    records: list[dict],
    n_rows: int,
    thicknesses: np.ndarray,
    gradients: np.ndarray,
    points: int,
) -> list[dict]:
    """Vectorize an exact uniform-gradient response over many candidates.

    The eigensystems and the row/profile support are unchanged.  Only the
    thickness propagation phases are broadcast over the candidate axis.  An
    absorptive response cache falls back to the scalar implementation because
    its direct matrix exponential does not share the Hermitian spectral form.
    """

    thicknesses = np.asarray(thicknesses, dtype=float)
    gradients = np.asarray(gradients, dtype=float)
    if thicknesses.shape != gradients.shape or thicknesses.ndim != 1:
        raise ValueError(
            "thickness and gradient candidate arrays must be one-dimensional "
            "and have identical shapes"
        )
    if any(
        bool(spec.get("absorptive", False))
        for record in records
        for spec in record["dyn_specs"]
    ):
        return [
            predict_from_response(records, n_rows, float(t0), float(grad), int(points))
            for t0, grad in zip(thicknesses, gradients, strict=True)
        ]

    nodes, thickness_weights = leggauss(int(points))
    thickness_weights = thickness_weights / np.sum(thickness_weights)
    quadrature = thicknesses[:, None] * (
        1.0 + gradients[:, None] * nodes[None, :]
    )
    dynamic = np.zeros((len(thicknesses), len(records)), dtype=float)
    born = np.zeros_like(dynamic)
    for row_number, record in enumerate(records):
        profile_weights = np.asarray(record["profile_weights"], dtype=float)
        dynamic_orientation = np.zeros(
            (len(thicknesses), len(record["dyn_specs"])), dtype=float
        )
        born_orientation = np.zeros_like(dynamic_orientation)
        for orientation_number, (spec, born_spec) in enumerate(
            zip(record["dyn_specs"], record["born_specs"], strict=True)
        ):
            values = np.asarray(spec["vals"], dtype=np.complex128)
            coupling = np.asarray(spec["coupling"], dtype=np.complex128)
            phase = np.exp(
                (2j * np.pi)
                * values[:, None, None]
                * quadrature[None, :, :]
            )
            per_node = np.abs(
                np.sum(coupling[:, None, None] * phase, axis=0)
            ) ** 2
            dynamic_orientation[:, orientation_number] = (
                per_node @ thickness_weights
            )

            argument = np.pi * float(born_spec["delta"]) * quadrature
            sinc = np.ones_like(argument)
            nonzero = np.abs(argument) >= 1e-12
            sinc[nonzero] = np.sin(argument[nonzero]) / argument[nonzero]
            born_per_node = (
                (2.0 * np.pi) ** 2
                * float(born_spec["h10_abs2"])
                * quadrature**2
                * sinc**2
            )
            born_orientation[:, orientation_number] = (
                born_per_node @ thickness_weights
            )
        dynamic[:, row_number] = dynamic_orientation @ profile_weights
        born[:, row_number] = born_orientation @ profile_weights

    indices = np.asarray([record["row_index"] for record in records], dtype=int)
    basis_sizes = np.asarray([record["basis_size"] for record in records], dtype=int)
    return [
        {
            "indices": indices,
            "dyn": dynamic[candidate],
            "born": born[candidate],
            "basis_size": basis_sizes,
            "response_rows_total": int(n_rows),
        }
        for candidate in range(len(thicknesses))
    ]


def metric_value(ev: dict, metric: str) -> float:
    """Return one Bloch metric of an evaluated trial: work_wR2, all_wR2, free_wR2, all_R1 or all_R1gt (fractions)."""

    if metric == "work_wR2":
        return float(ev["metrics"]["work"]["dyn"]["wR2_all"])
    if metric == "all_wR2":
        return float(ev["metrics"]["all"]["dyn"]["wR2_all"])
    if metric == "free_wR2":
        return float(ev["metrics"]["free"]["dyn"]["wR2_all"])
    if metric == "all_R1":
        return float(ev["metrics"]["all"]["dyn"]["R1_all"])
    if metric == "all_R1gt":
        return float(ev["metrics"]["all"]["dyn"]["R1_gt"])
    raise ValueError(f"unknown metric {metric!r}")
