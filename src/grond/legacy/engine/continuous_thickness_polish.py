#!/usr/bin/env python3
"""Continuous crystal-thickness regularization for GROND.

This script tests the key implementation idea needed to move beyond a coarse
thickness grid: for a fixed structure, geometry, beam basis, and absorption
model, each Bloch Hamiltonian is independent of crystal thickness.  We can
therefore build and eigensolve each row/profile Hamiltonian once, then re-score
many continuous thickness and thickness-gradient values by changing only the
propagation phases.

The implementation deliberately reuses GROND's native nuisance objective and
scale fitting.  It is run after a cheap grid screen: the grid finds a defensible
bracket, and this script turns that bracket into a continuous thickness value
without repeating all of the eigensolver work for each trial point.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh
from scipy.optimize import minimize, minimize_scalar

if __package__:
    from . import native_row_regularize as nrf
    from .bloch_engine import Config, XDSBlochPort, XDSData, parse_model, sha256, split_masks
else:  # pragma: no cover - direct script execution
    import native_row_regularize as nrf
    from bloch_engine import Config, XDSBlochPort, XDSData, parse_model, sha256, split_masks


DEFAULT_HKL = Path("INTEGRATE.HKL")
DEFAULT_RES = Path("model.res")
PHYSICAL_GRAD_MIN = 0.0
PHYSICAL_GRAD_MAX = 1.0
PACKED_PROPAGATION_CHUNK = 2048


class ResponseRecords(list):
    """List-compatible response cache with an exact packed propagation view."""

    def __init__(self, records=(), *, packed: dict | None = None):
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
    )


def selected_indices(data: XDSData, target: np.ndarray, args) -> np.ndarray:
    """Choose either all eligible rows or the strongest eligible subset."""

    idx = nrf.row_selection(
        data,
        target,
        positive_only=not bool(args.include_nonpositive),
        min_d=float(args.min_d_spacing),
        min_peak=float(args.min_peak),
        max_rows=0,
        geometry_percentile=float(args.geometry_percentile),
    )
    if int(args.strong_rows) > 0 and len(idx) > int(args.strong_rows):
        sig = np.asarray(data.sigma, dtype=float)
        score = np.asarray(target, dtype=float) / np.maximum(sig, 1e-12)
        order = np.argsort(score[idx])[::-1]
        idx = idx[order[: int(args.strong_rows)]]
        idx = np.sort(idx)
    if int(args.max_rows) > 0 and len(idx) > int(args.max_rows):
        idx = idx[: int(args.max_rows)]
    return np.asarray(idx, dtype=int)


def propagation_spec(H: np.ndarray, absorptive: bool) -> dict[str, np.ndarray]:
    """Cache a stable direct-to-target propagator for one Hamiltonian."""

    if absorptive:
        spectral = XDSBlochPort._nonhermitian_spectral_target_coupling(H)
        if spectral is None:
            return {
                "matrix": np.asarray(H, dtype=np.complex128),
                "absorptive": True,
            }
        vals, coupling = spectral
    else:
        vals, vecs = eigh(H)
        coupling = vecs[1, :] * np.conj(vecs[0, :])
    return {
        "vals": np.asarray(vals, dtype=np.complex128),
        "coupling": np.asarray(coupling, dtype=np.complex128),
        "absorptive": bool(absorptive),
    }


def build_response_cache(
    data: XDSData,
    model,
    cfg: Config,
    indices: np.ndarray,
    res_hash: str,
    cache_dir: Path,
) -> list[dict]:
    """Build row/profile eigensystems that can be re-scored for any thickness."""

    port = XDSBlochPort(data, model, cfg, cache_dir=cache_dir, res_hash=res_hash)
    absorptive = port.absorption_enabled()
    records: list[dict] = []
    t0 = time.perf_counter()
    for n, row_idx in enumerate(indices, start=1):
        plan = port.row_prediction_plan(int(row_idx))
        dyn_specs = [
            propagation_spec(matrix, absorptive=absorptive)
            for matrix in plan["_dyn_matrices"]
        ]
        born_specs = [
            {
                "h10_abs2": float(plan["_born_h10_abs2"]),
                "delta": float(delta),
            }
            for delta in plan["_born_deltas"]
        ]
        metadata = {key: value for key, value in plan.items() if not key.startswith("_")}
        sg_resid = np.asarray(plan["_sg_resid"], dtype=float)

        records.append(
            {
                "row_index": int(row_idx),
                "profile_weights": np.asarray(plan["_profile_weights"], dtype=float),
                "dyn_specs": dyn_specs,
                "born_specs": born_specs,
                "max_sg_solve_residual": float(np.max(np.abs(sg_resid))) if sg_resid.size else 0.0,
                **metadata,
            }
        )
        if n % int(args_progress_every) == 0 or n == len(indices):
            elapsed = time.perf_counter() - t0
            rate = n / max(elapsed, 1e-12)
            eta = (len(indices) - n) / max(rate, 1e-12)
            print(
                f"response cache {n}/{len(indices)} rows | {rate:.2f} rows/s | ETA {eta/60:.1f}m",
                flush=True,
            )
    response_records = ResponseRecords(records)
    estimated_bytes = estimated_packed_response_bytes(response_records)
    max_gib = float(os.environ.get("GROND_MAX_PACKED_THICKNESS_GIB", "2.0"))
    max_bytes = max(0, int(max_gib * 1024**3))
    if estimated_bytes > max_bytes:
        response_records.packed_status = {
            "status": "skipped_memory_limit",
            "estimated_bytes": estimated_bytes,
            "maximum_bytes": max_bytes,
        }
    else:
        try:
            response_records.packed = pack_response_records(response_records)
            response_records.packed_status = {
                "status": "built",
                "estimated_bytes": estimated_bytes,
                "maximum_bytes": max_bytes,
            }
        except MemoryError:
            response_records.packed = None
            response_records.packed_status = {
                "status": "skipped_memory_error",
                "estimated_bytes": estimated_bytes,
                "maximum_bytes": max_bytes,
            }
    return response_records


def dyn_from_spec(spec: dict[str, np.ndarray], thick: np.ndarray, weights: np.ndarray) -> float:
    if "matrix" in spec:
        per_t = XDSBlochPort._nonhermitian_target_intensities(spec["matrix"], thick)
        return float(np.sum(weights * per_t))
    vals = spec["vals"]
    coupling = spec["coupling"]
    phase = np.exp((2j * np.pi) * vals[:, None] * thick[None, :])
    per_t = np.abs(np.sum(coupling[:, None] * phase, axis=0)) ** 2
    return float(np.sum(weights * per_t))


def born_from_spec(h10_abs2: float, delta: float, thick: np.ndarray, weights: np.ndarray) -> float:
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
            values = np.asarray(dyn_spec["vals"], dtype=np.complex128)
            coupling = np.asarray(dyn_spec["coupling"], dtype=np.complex128)
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
            "phase_rates": np.stack(bucket["phase_rates"]).astype(np.complex128, copy=False),
            "couplings": np.stack(bucket["couplings"]).astype(np.complex128, copy=False),
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
        rates = np.asarray(bucket["phase_rates"], dtype=np.complex128)
        couplings = np.asarray(bucket["couplings"], dtype=np.complex128)
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


def make_nuisance(args, t0: float, grad: float) -> nrf.Nuisance:
    require_physical_thickness_candidate(t0, grad)
    return nrf.Nuisance(
        thickness_A=float(t0),
        thickness_grad_frac=float(grad),
        frame_offset=float(args.frame_offset),
        group_width_deg=float(args.group_width_deg),
        detector_scale_bins=int(args.detector_scale_bins),
        sigma_a=float(args.sigma_a),
        sigma_b_corr=float(args.sigma_b_corr),
        sigma_c_peak=float(args.sigma_c_peak),
        thickness_model="uniform_gradient",
        thickness_sigma_frac=0.0,
        matrix_convention=str(args.matrix_convention),
        row_resolution_b=float(args.row_resolution_b),
        native_wght_a=float("nan"),
        native_wght_b=float("nan"),
        scale_smoothness=float(args.scale_smoothness),
    )


def metric_value(ev: dict, metric: str) -> float:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hkl", default=str(DEFAULT_HKL))
    ap.add_argument("--res", default=str(DEFAULT_RES))
    ap.add_argument("--out-dir", default="outputs/continuous_thickness_polish")
    ap.add_argument("--hkl-transform", default="identity")
    ap.add_argument("--n-basis", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=10)
    ap.add_argument("--sg-window", type=float, default=0.004)
    ap.add_argument("--sg-window-mode", default="fixed")
    ap.add_argument("--sg-window-min", type=float, default=0.003)
    ap.add_argument("--sg-window-max", type=float, default=0.012)
    ap.add_argument("--sg-window-scale", type=float, default=1.0)
    ap.add_argument("--sg-window-frame-floor", type=float, default=0.5)
    ap.add_argument("--profile-quadrature", choices=["uniform", "trapezoid", "gauss_legendre"], default="uniform")
    ap.add_argument("--profile-accumulation", choices=["sum", "mean"], default="sum")
    ap.add_argument("--topup-d-min", type=float, default=0.8)
    ap.add_argument("--max-support-sg", type=float, default=0.005)
    ap.add_argument("--min-d-spacing", type=float, default=0.7)
    ap.add_argument("--min-peak", type=float, default=90.0)
    ap.add_argument("--geometry-percentile", type=float, default=100.0)
    ap.add_argument("--target-mode", choices=list(nrf.TARGET_MODES), default="iobs")
    ap.add_argument("--include-nonpositive", action="store_true")
    ap.add_argument("--start-thickness", type=float, default=1200.0)
    ap.add_argument("--start-grad", type=float, default=1.0)
    ap.add_argument("--thickness-points", type=int, default=5)
    ap.add_argument("--frame-offset", type=float, default=0.2)
    ap.add_argument("--group-width-deg", type=float, default=0.168968)
    ap.add_argument("--detector-scale-bins", type=int, default=8)
    ap.add_argument("--sigma-a", type=float, default=0.0)
    ap.add_argument("--sigma-b-corr", type=float, default=0.0)
    ap.add_argument("--sigma-c-peak", type=float, default=0.15)
    ap.add_argument("--row-resolution-b", type=float, default=0.0)
    ap.add_argument("--scale-smoothness", type=float, default=0.0)
    ap.add_argument("--matrix-convention", default="j_minus_i")
    ap.add_argument("--free-fraction", type=float, default=0.1)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--scale-fit-split", choices=["all", "work", "free"], default="work")
    ap.add_argument(
        "--selection-metric",
        choices=["work_wR2", "all_wR2", "free_wR2", "all_R1", "all_R1gt"],
        default="work_wR2",
    )
    ap.add_argument("--strong-rows", type=int, default=500)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--bracket-frac", type=float, default=0.35)
    ap.add_argument("--t-min", type=float, default=None, help="Optional lower clamp on the local thickness bracket.")
    ap.add_argument("--t-max", type=float, default=None, help="Optional upper clamp on the local thickness bracket.")
    ap.add_argument("--grad-min", type=float, default=0.0)
    ap.add_argument("--grad-max", type=float, default=PHYSICAL_GRAD_MAX)
    ap.add_argument("--grid-thickness-step", type=float, default=100.0)
    ap.add_argument("--grid-grad-step", type=float, default=0.2)
    ap.add_argument("--powell-maxiter", type=int, default=24)
    ap.add_argument("--min-promotion-gain", type=float, default=5e-5, help="Minimum selected-objective gain required before replacing the grid value.")
    ap.add_argument("--max-free-wr2-loss", type=float, default=0.002, help="Maximum tolerated free-row wR2 loss for a promoted continuous thickness.")
    ap.add_argument("--max-all-r1-loss", type=float, default=0.001, help="Maximum tolerated all-row R1 loss for a promoted continuous thickness.")
    ap.add_argument("--progress-every", type=int, default=100)
    args = ap.parse_args()

    try:
        args.t_min, args.t_max = local_thickness_search_bounds(
            float(args.start_thickness),
            float(args.bracket_frac),
            t_min=args.t_min,
            t_max=args.t_max,
        )
    except ValueError as exc:
        ap.error(str(exc))
    search_values = {
        "t_min": float(args.t_min),
        "t_max": float(args.t_max),
        "grad_min": float(args.grad_min),
        "grad_max": float(args.grad_max),
        "grid_thickness_step": float(args.grid_thickness_step),
        "grid_grad_step": float(args.grid_grad_step),
    }
    if not all(math.isfinite(value) for value in search_values.values()):
        ap.error("continuous-thickness search bounds and steps must be finite")
    if not PHYSICAL_GRAD_MIN <= float(args.grad_min) <= float(args.grad_max) <= PHYSICAL_GRAD_MAX:
        ap.error(
            "continuous-thickness gradient bounds must satisfy "
            f"{PHYSICAL_GRAD_MIN:g} <= grad_min <= grad_max <= {PHYSICAL_GRAD_MAX:g}"
        )
    if float(args.grid_thickness_step) <= 0.0 or float(args.grid_grad_step) <= 0.0:
        ap.error("continuous-thickness grid steps must be positive")
    if int(args.powell_maxiter) <= 0:
        ap.error("continuous-thickness Powell iterations must be positive")
    start_reasons = thickness_candidate_domain_reasons(
        float(args.start_thickness),
        float(args.start_grad),
        t_min=float(args.t_min),
        t_max=float(args.t_max),
        grad_min=float(args.grad_min),
        grad_max=float(args.grad_max),
    )
    if start_reasons:
        ap.error("invalid starting continuous-thickness point: " + ", ".join(start_reasons))

    global args_progress_every
    args_progress_every = max(1, int(args.progress_every))

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    data = XDSData(Path(args.hkl), hkl_transform=args.hkl_transform)
    target, base_sig = nrf.target_arrays(data, args.target_mode)
    indices = selected_indices(data, target, args)
    model = parse_model(Path(args.res))
    masks_all = split_masks(
        data.hkl,
        float(args.free_fraction),
        int(args.split_seed),
        symmetry_ops=model.get("symmetry_ops"),
    )
    selected = np.zeros(len(target), dtype=bool)
    selected[indices] = True
    masks = {k: v & selected for k, v in masks_all.items()}
    cfg = build_base_config(args)
    wght_a, wght_b = nrf.parse_wght(Path(args.res))
    print(
        f"selected {len(indices)} rows for continuous thickness polish | "
        f"N={args.n_basis} Q={args.n_points} start={args.start_thickness:.1f} A grad={args.start_grad:.3f}",
        flush=True,
    )

    cache_t0 = time.perf_counter()
    records = build_response_cache(
        data,
        model,
        cfg,
        indices,
        res_hash=sha256(Path(args.res)),
        cache_dir=out / "ug_cache",
    )
    cache_seconds = time.perf_counter() - cache_t0

    eval_history: list[dict] = []

    def evaluate(t0: float, grad: float, label: str = "eval") -> tuple[float, dict]:
        if thickness_candidate_domain_reasons(
            t0,
            grad,
            t_min=float(args.t_min),
            t_max=float(args.t_max),
            grad_min=float(args.grad_min),
            grad_max=float(args.grad_max),
        ):
            return 1.0e9, {}
        pred = predict_from_response(records, len(target), float(t0), float(grad), int(args.thickness_points))
        nui = make_nuisance(args, float(t0), float(grad))
        ev = nrf.evaluate_trial(
            data,
            target,
            base_sig,
            pred,
            indices,
            masks,
            nui,
            wght_a,
            wght_b,
            scale_fit_split=str(args.scale_fit_split),
        )
        val = metric_value(ev, str(args.selection_metric))
        row = {
            "label": label,
            "thickness_A": float(t0),
            "thickness_nm": float(t0) / 10.0,
            "thickness_grad_frac": float(grad),
            "objective": float(val),
            "all_wR2": float(ev["metrics"]["all"]["dyn"]["wR2_all"]),
            "all_R1": float(ev["metrics"]["all"]["dyn"]["R1_all"]),
            "all_R1gt": float(ev["metrics"]["all"]["dyn"]["R1_gt"]),
            "free_wR2": float(ev["metrics"]["free"]["dyn"]["wR2_all"]),
            "scale_count": int(ev["scale_count"]),
            "valid_rows": int(ev["valid_rows"]),
        }
        eval_history.append(row)
        return val, ev

    # 1. Baseline at the grid-promoted point.
    baseline_val, baseline_ev = evaluate(float(args.start_thickness), float(args.start_grad), "baseline")
    print(f"baseline objective={baseline_val:.6f}", flush=True)

    # 2. Local scalar bounded regularization of thickness at fixed gradient.
    t0 = float(args.start_thickness)
    lo = float(args.t_min)
    hi = float(args.t_max)
    scalar_t0 = time.perf_counter()
    scalar_res = minimize_scalar(
        lambda x: evaluate(float(x), float(args.start_grad), "scalar_t")[0],
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 0.05, "maxiter": 32},
    )
    scalar_seconds = time.perf_counter() - scalar_t0
    scalar_val, scalar_ev = evaluate(float(scalar_res.x), float(args.start_grad), "scalar_t_final")
    print(
        f"scalar thickness optimum t={scalar_res.x:.3f} A objective={scalar_val:.6f} "
        f"in {scalar_seconds:.2f}s",
        flush=True,
    )

    # 3. Parabolic/Newton-like one-step from three local points.
    step = max(10.0, 0.05 * t0)
    xs = np.unique(np.asarray([max(lo, t0 - step), t0, min(hi, t0 + step)], dtype=float))
    ys = np.asarray([evaluate(float(x), float(args.start_grad), "parabolic_probe")[0] for x in xs], dtype=float)
    if len(xs) == 3:
        coeff = np.polyfit(xs, ys, 2)
        if coeff[0] > 0:  # noqa: SIM108 - keep the fitted and fallback branches explicit
            t_parab = float(np.clip(-coeff[1] / (2.0 * coeff[0]), lo, hi))
        else:
            t_parab = float(xs[int(np.argmin(ys))])
    else:
        t_parab = float(xs[int(np.argmin(ys))])
    parab_val, parab_ev = evaluate(t_parab, float(args.start_grad), "parabolic_newton")
    print(f"parabolic step t={t_parab:.3f} A objective={parab_val:.6f}", flush=True)

    # 4. Small direct grid over t and gradient to guard against local minima.
    grid_t0 = time.perf_counter()
    t_values = inclusive_bounded_axis(lo, hi, float(args.grid_thickness_step))
    g_values = inclusive_bounded_axis(
        float(args.grad_min), float(args.grad_max), float(args.grid_grad_step)
    )
    grid_candidates = [
        (float(tv), float(gv)) for tv in t_values for gv in g_values
    ]
    grid_predictions = predict_candidates_from_response(
        records,
        len(target),
        np.asarray([candidate[0] for candidate in grid_candidates]),
        np.asarray([candidate[1] for candidate in grid_candidates]),
        int(args.thickness_points),
    )
    grid_best = None
    for (tv, gv), pred in zip(
        grid_candidates, grid_predictions, strict=True
    ):
        nui = make_nuisance(args, tv, gv)
        ev = nrf.evaluate_trial(
            data,
            target,
            base_sig,
            pred,
            indices,
            masks,
            nui,
            wght_a,
            wght_b,
            scale_fit_split=str(args.scale_fit_split),
        )
        val = metric_value(ev, str(args.selection_metric))
        eval_history.append(
            {
                "label": "guard_grid_batched",
                "thickness_A": tv,
                "thickness_nm": tv / 10.0,
                "thickness_grad_frac": gv,
                "objective": float(val),
                "all_wR2": float(ev["metrics"]["all"]["dyn"]["wR2_all"]),
                "all_R1": float(ev["metrics"]["all"]["dyn"]["R1_all"]),
                "all_R1gt": float(ev["metrics"]["all"]["dyn"]["R1_gt"]),
                "free_wR2": float(ev["metrics"]["free"]["dyn"]["wR2_all"]),
                "scale_count": int(ev["scale_count"]),
                "valid_rows": int(ev["valid_rows"]),
            }
        )
        if grid_best is None or val < grid_best[0]:
            grid_best = (val, tv, gv, ev)
    grid_seconds = time.perf_counter() - grid_t0
    print(
        f"guard grid best t={grid_best[1]:.3f} A grad={grid_best[2]:.3f} "
        f"objective={grid_best[0]:.6f} in {grid_seconds:.2f}s",
        flush=True,
    )

    # 5. Bounded Powell in thickness and gradient, seeded from the guard grid.
    powell_t0 = time.perf_counter()
    x0 = np.asarray([grid_best[1], grid_best[2]], dtype=float)
    powell_t_lower, powell_t_upper = local_powell_thickness_bounds(
        t_values, float(grid_best[1]), lo, hi
    )
    powell_res = minimize(
        lambda x: evaluate(float(x[0]), float(x[1]), "powell")[0],
        x0,
        method="Powell",
        bounds=[(powell_t_lower, powell_t_upper), (float(args.grad_min), float(args.grad_max))],
        options={"maxiter": int(args.powell_maxiter), "xtol": 0.05, "ftol": 1e-5},
    )
    powell_seconds = time.perf_counter() - powell_t0
    powell_t = float(powell_res.x[0])
    powell_g = float(powell_res.x[1])
    powell_domain_reasons = thickness_candidate_domain_reasons(
        powell_t,
        powell_g,
        t_min=float(args.t_min),
        t_max=float(args.t_max),
        grad_min=float(args.grad_min),
        grad_max=float(args.grad_max),
    )
    if powell_domain_reasons:
        powell_val, powell_ev = 1.0e9, {}
        print(
            f"Powell result rejected before promotion: t={powell_t!r} A grad={powell_g!r}; "
            + ", ".join(powell_domain_reasons),
            flush=True,
        )
    else:
        powell_val, powell_ev = evaluate(powell_t, powell_g, "powell_final")
    print(
        f"Powell optimum t={powell_t:.3f} A grad={powell_g:.4f} "
        f"objective={powell_val:.6f} in {powell_seconds:.2f}s",
        flush=True,
    )

    candidates = [
        ("baseline", baseline_val, float(args.start_thickness), float(args.start_grad), baseline_ev),
        ("scalar_t", scalar_val, float(scalar_res.x), float(args.start_grad), scalar_ev),
        ("parabolic_newton", parab_val, t_parab, float(args.start_grad), parab_ev),
        ("guard_grid", grid_best[0], grid_best[1], grid_best[2], grid_best[3]),
        ("powell", powell_val, powell_t, powell_g, powell_ev),
    ]
    baseline_free = float(baseline_ev["metrics"]["free"]["dyn"]["wR2_all"])
    baseline_r1 = float(baseline_ev["metrics"]["all"]["dyn"]["R1_all"])
    eligible = []
    rejected = []
    for name, val, tv, gv, ev in candidates:
        reasons = thickness_candidate_domain_reasons(
            tv,
            gv,
            t_min=float(args.t_min),
            t_max=float(args.t_max),
            grad_min=float(args.grad_min),
            grad_max=float(args.grad_max),
        )
        if not math.isfinite(float(val)):
            reasons.append("nonfinite_objective")
        if not ev:
            reasons.append("missing_valid_evaluation")
        if reasons:
            rejected.append(
                {
                    "strategy": name,
                    "gain": None,
                    "free_wR2_loss": None,
                    "all_R1_loss": None,
                    "reasons": list(dict.fromkeys(reasons)),
                }
            )
            continue
        gain = float(baseline_val - val)
        free_loss = float(ev["metrics"]["free"]["dyn"]["wR2_all"] - baseline_free)
        all_r1_loss = float(ev["metrics"]["all"]["dyn"]["R1_all"] - baseline_r1)
        reasons = []
        if name != "baseline" and gain < float(args.min_promotion_gain):
            reasons.append("gain_below_minimum")
        if free_loss > float(args.max_free_wr2_loss):
            reasons.append("free_wR2_loss")
        if all_r1_loss > float(args.max_all_r1_loss):
            reasons.append("all_R1_loss")
        rec = {
            "strategy": name,
            "gain": gain,
            "free_wR2_loss": free_loss,
            "all_R1_loss": all_r1_loss,
            "reasons": reasons,
        }
        if reasons:
            rejected.append(rec)
        else:
            eligible.append((name, val, tv, gv, ev, rec))
    if not eligible:
        eligible = [("baseline", baseline_val, float(args.start_thickness), float(args.start_grad), baseline_ev, {
            "strategy": "baseline",
            "gain": 0.0,
            "free_wR2_loss": 0.0,
            "all_R1_loss": 0.0,
            "reasons": [],
        })]
    best_label, best_val, best_t, best_g, best_ev, best_decision = min(eligible, key=lambda x: x[1])
    best_nui = make_nuisance(args, best_t, best_g)
    boundary_tolerance = max(0.1, 0.01 * float(args.grid_thickness_step))
    thickness_at_search_boundary = bool(
        math.isclose(best_t, lo, rel_tol=0.0, abs_tol=boundary_tolerance)
        or math.isclose(best_t, hi, rel_tol=0.0, abs_tol=boundary_tolerance)
    )
    gradient_at_physical_boundary = bool(
        math.isclose(best_g, PHYSICAL_GRAD_MAX, rel_tol=0.0, abs_tol=1e-6)
    )
    identifiability_warnings = []
    if thickness_at_search_boundary:
        identifiability_warnings.append(
            "selected thickness is on the local search boundary; the mean is not bracketed by this run"
        )
    if gradient_at_physical_boundary:
        identifiability_warnings.append(
            "selected gradient is at the widest physical uniform-gradient boundary; interpret it as effective dephasing unless independently calibrated"
        )
    elif best_g >= 0.8:
        identifiability_warnings.append(
            "selected thickness distribution is broad enough to suppress coherent thickness oscillations; the mean may be weakly identified"
        )
    best_native = {k: v for k, v in best_ev.items() if k != "arrays"}
    best_native["nuisance"] = asdict(best_nui)
    best_native["continuous_thickness_strategy"] = best_label

    with (out / "continuous_thickness_trials.csv").open("w", newline="") as f:
        fields = list(eval_history[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(eval_history)

    summary = {
        "inputs": {
            "hkl": str(Path(args.hkl).resolve()),
            "res": str(Path(args.res).resolve()),
            "hkl_transform": str(args.hkl_transform),
        },
        "config": vars(args),
        "contract": {
            "n_basis": int(args.n_basis),
            "n_points": int(args.n_points),
            "profile_quadrature": str(args.profile_quadrature),
            "profile_accumulation": str(args.profile_accumulation),
            "thickness_points": int(args.thickness_points),
            "topup_d_min": float(args.topup_d_min),
            "max_support_sg": float(args.max_support_sg),
            "frame_offset": float(args.frame_offset),
            "group_width_deg": float(args.group_width_deg),
            "detector_scale_bins": int(args.detector_scale_bins),
            "sigma_c_peak": float(args.sigma_c_peak),
            "selection_metric": str(args.selection_metric),
            "scale_fit_split": str(args.scale_fit_split),
            "target_mode": str(args.target_mode),
        },
        "row_selection": {
            "eligible_rows": int(len(indices)),
            "strong_rows": int(args.strong_rows),
            "max_rows": int(args.max_rows),
        },
        "timings": {
            "response_cache_seconds": float(cache_seconds),
            "scalar_seconds": float(scalar_seconds),
            "guard_grid_seconds": float(grid_seconds),
            "powell_seconds": float(powell_seconds),
            "total_seconds": float(time.perf_counter() - t_start),
            "evaluations": int(len(eval_history)),
            "packed_response": dict(getattr(records, "packed_status", {})),
        },
        "strategies": {
            name: {
                "status": "ok" if ev else "rejected_invalid_domain",
                "objective": float(val) if ev else None,
                "thickness_A": float(tv),
                "thickness_nm": float(tv) / 10.0,
                "thickness_grad_frac": float(gv),
                "all_wR2": float(ev["metrics"]["all"]["dyn"]["wR2_all"]) if ev else None,
                "all_R1": float(ev["metrics"]["all"]["dyn"]["R1_all"]) if ev else None,
                "all_R1gt": float(ev["metrics"]["all"]["dyn"]["R1_gt"]) if ev else None,
                "free_wR2": float(ev["metrics"]["free"]["dyn"]["wR2_all"]) if ev else None,
                "scale_count": int(ev["scale_count"]) if ev else None,
                "valid_rows": int(ev["valid_rows"]) if ev else None,
                "domain_reasons": thickness_candidate_domain_reasons(
                    tv,
                    gv,
                    t_min=float(args.t_min),
                    t_max=float(args.t_max),
                    grad_min=float(args.grad_min),
                    grad_max=float(args.grad_max),
                ),
            }
            for name, val, tv, gv, ev in candidates
        },
        "best": {
            "strategy": best_label,
            "objective": float(best_val),
            "thickness_A": float(best_t),
            "thickness_nm": float(best_t) / 10.0,
            "thickness_grad_frac": float(best_g),
            "all_wR2": float(best_ev["metrics"]["all"]["dyn"]["wR2_all"]),
            "all_R1": float(best_ev["metrics"]["all"]["dyn"]["R1_all"]),
            "all_R1gt": float(best_ev["metrics"]["all"]["dyn"]["R1_gt"]),
            "free_wR2": float(best_ev["metrics"]["free"]["dyn"]["wR2_all"]),
            "scale_count": int(best_ev["scale_count"]),
            "valid_rows": int(best_ev["valid_rows"]),
        },
        "thickness_identifiability": {
            "status": "warning" if identifiability_warnings else "locally_bracketed",
            "mean_thickness_A": float(best_t),
            "absolute_half_width_A": float(best_t * best_g),
            "nominal_support_A": [float(best_t * (1.0 - best_g)), float(best_t * (1.0 + best_g))],
            "thickness_at_local_search_boundary": thickness_at_search_boundary,
            "gradient_at_physical_boundary": gradient_at_physical_boundary,
            "warnings": identifiability_warnings,
        },
        "promotion_decision": {
            "selected": best_decision,
            "rejected": rejected,
            "min_promotion_gain": float(args.min_promotion_gain),
            "max_free_wR2_loss": float(args.max_free_wr2_loss),
            "max_all_R1_loss": float(args.max_all_r1_loss),
        },
        "best_native": best_native,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    (out / "best_native_fit.json").write_text(json.dumps(best_native, indent=2, allow_nan=True) + "\n")
    report = [
        "# Continuous Thickness Regularization",
        "",
        f"Rows: {len(indices)} selected; N={args.n_basis}; Q={args.n_points}; thickness quadrature={args.thickness_points}.",
        f"Response cache build: {cache_seconds:.2f} s. Total: {summary['timings']['total_seconds']:.2f} s for {len(eval_history)} objective evaluations.",
        "",
        "| strategy | thickness A | thickness nm | grad | all wR2 | all R1 | R1gt | free wR2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _val, tv, gv, ev in candidates:
        if ev:
            report.append(
                f"| {name} | {tv:.3f} | {tv / 10.0:.3f} | {gv:.4f} | "
                f"{ev['metrics']['all']['dyn']['wR2_all']:.6f} | "
                f"{ev['metrics']['all']['dyn']['R1_all']:.6f} | "
                f"{ev['metrics']['all']['dyn']['R1_gt']:.6f} | "
                f"{ev['metrics']['free']['dyn']['wR2_all']:.6f} |"
            )
        else:
            report.append(
                f"| {name} | {tv:.3f} | {tv / 10.0:.3f} | {gv:.4f} | rejected | rejected | rejected | rejected |"
            )
    report.extend(
        [
            "",
            f"Selected result: `{best_label}` at `{best_t:.3f} A` (`{best_t / 10.0:.3f} nm`) with gradient `{best_g:.4f}`.",
            f"Promotion gate: gain `{best_decision['gain']:.6f}`, free-wR2 loss `{best_decision['free_wR2_loss']:.6f}`, all-R1 loss `{best_decision['all_R1_loss']:.6f}`.",
            f"Thickness identifiability: `{summary['thickness_identifiability']['status']}`; absolute half-width `{summary['thickness_identifiability']['absolute_half_width_A']:.3f} A`.",
            *(f"Warning: {warning}." for warning in summary["thickness_identifiability"]["warnings"]),
            "",
            "The important speed result is not the subset metric itself; it is that once the response cache is built, scalar, parabolic, grid, and Powell searches reuse the same eigensystems and only change propagation phases.",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    print(f"wrote {out / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
