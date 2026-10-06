#!/usr/bin/env python3
"""Standalone XDS Bloch-wave port.

This keeps the audited row-contract conventions explicit in the XDS-facing
calculator:

* signed observation HKLs are preserved;
* off-diagonal coupling uses U(h_j - h_i), not U(h_i - h_j);
* the rocking path is sampled in target Sg and converted back to XDS omega;
* all basis-beam Sg values are recomputed at each omega;
* Born and Bloch predictions are scaled and scored on identical row masks.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import multiprocessing as multiprocessing
import os
import sys
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

if not __package__:
    # Direct legacy scripts still resolve the same installed/source package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from scipy.linalg import eig, expm
from scipy.sparse.linalg import expm_multiply

try:
    from grond.paths import runtime_cache_dir
except ImportError:  # Direct script execution from the legacy engine directory.

    def runtime_cache_dir(*parts: str) -> Path:
        """Return the GROND cache directory ($GROND_CACHE_DIR or ~/.cache/grond) joined with parts."""

        override = os.environ.get("GROND_CACHE_DIR", "").strip()
        root = Path(override).expanduser() if override else Path.home() / ".cache" / "grond"
        return root.joinpath(*parts)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HKL = Path("INTEGRATE.HKL")
DEFAULT_RES = Path("model.res")


from grond.bloch.backends import (
    _jax_backend_uses_real_expm as _jax_backend_uses_real_expm,
    _jax_matrix_bucket_size as _jax_matrix_bucket_size,
    _jax_module_key as _jax_module_key,
    _jax_modules as _jax_modules,
    _jax_propagate_batch as _jax_propagate_batch,
    _jax_real_expm_propagate_batch as _jax_real_expm_propagate_batch,
    _real_embedding_generator_batch as _real_embedding_generator_batch,
    _stack_square_matrices as _stack_square_matrices,
    promoted_bethe_effective_hamiltonian as promoted_bethe_effective_hamiltonian,
)
from grond.bloch.basis import (
    BasisMixin as BasisMixin,
)
from grond.bloch.cache import (
    dense_ug_memory_plan as dense_ug_memory_plan,
    enforce_dense_ug_memory_limit as enforce_dense_ug_memory_limit,
    row_plan_cache_validation_errors as row_plan_cache_validation_errors,
)
from grond.bloch.config import (
    Config as Config,
    frozen_kernel_contract as frozen_kernel_contract,
)
from grond.bloch.constants import (
    ALL_JAX_BACKENDS as ALL_JAX_BACKENDS,
    BETHE_BACKENDS as BETHE_BACKENDS,
    COMPUTE_BACKEND_HELP as COMPUTE_BACKEND_HELP,
    COMPUTE_BACKENDS as COMPUTE_BACKENDS,
    DEFAULT_MAX_SUPPORT_SG as DEFAULT_MAX_SUPPORT_SG,
    DENSE_UG_BUILD_BYTES_PER_VOXEL as DENSE_UG_BUILD_BYTES_PER_VOXEL,
    DENSE_UG_CACHE_LOAD_BYTES_PER_VOXEL as DENSE_UG_CACHE_LOAD_BYTES_PER_VOXEL,
    UG_CACHE_SCHEMA as UG_CACHE_SCHEMA,
)
from grond.bloch.errors import (
    InfeasibleSgJitterProfileError as InfeasibleSgJitterProfileError,
    LocalSgCrossingError as LocalSgCrossingError,
    NonPassiveAbsorptionError as NonPassiveAbsorptionError,
)
from grond.bloch.geometry import (
    GeometryMixin as GeometryMixin,
)
from grond.bloch.hamiltonian import (
    HamiltonianMixin as HamiltonianMixin,
    validate_passive_absorption as validate_passive_absorption,
)
from grond.bloch.metrics import (
    effective_sigma as effective_sigma,
    evaluate_predictions as evaluate_predictions,
    read_row_index_file as read_row_index_file,
    scale_by_groups as scale_by_groups,
)
from grond.bloch.model import (
    alternate_hand_model as alternate_hand_model,
    atom_dw as atom_dw,
    atom_dw_array as atom_dw_array,
    cell_orthogonalization_matrix as cell_orthogonalization_matrix,
    cif_number as cif_number,
    cif_symmetry_records as cif_symmetry_records,
    fcalc_res as fcalc_res,
    fcalc_res_grid as fcalc_res_grid,
    inv_d2_from_hkl_array as inv_d2_from_hkl_array,
    loop_rows as loop_rows,
    model_symmetry_ops as model_symmetry_ops,
    parse_cif as parse_cif,
    parse_model as parse_model,
    parse_res as parse_res,
    relativistic_gamma_from_wavelength as relativistic_gamma_from_wavelength,
    shelx_u_cart_matrix as shelx_u_cart_matrix,
    unique_symmetry_mates as unique_symmetry_mates,
)
from grond.bloch.numerics import (
    _rotation_matrix_axis_angle as _rotation_matrix_axis_angle,
    cell_from_real_matrix as cell_from_real_matrix,
    hermitian_eigh as hermitian_eigh,
    load_geometry_path_contract as load_geometry_path_contract,
    parse_hkl_transform as parse_hkl_transform,
    rotation_matrix as rotation_matrix,
    sha256 as sha256,
)
from grond.bloch.potential import (
    PotentialMixin as PotentialMixin,
)
from grond.bloch.prediction import (
    PredictionMixin as PredictionMixin,
)
from grond.bloch.profiles import (
    ProfileMixin as ProfileMixin,
)
from grond.bloch.row_cache import (
    RowCacheMixin as RowCacheMixin,
)
from grond.bloch.row_plans import (
    RowPlanMixin as RowPlanMixin,
)
from grond.bloch.splits import (
    build_transform_split_contract as build_transform_split_contract,
    candidate_invariant_split_membership as candidate_invariant_split_membership,
    metric_filter_signed_axis_transforms as metric_filter_signed_axis_transforms,
    split_masks as split_masks,
    validate_transform_split_contract as validate_transform_split_contract,
)
from grond.bloch.thickness import gamma_cf_point as gamma_cf_point, uniform_grid_amplitudes as uniform_grid_amplitudes
from grond.io.xds import (
    XDSData as XDSData,
)


class XDSBlochPort(
    PotentialMixin,
    GeometryMixin,
    ProfileMixin,
    BasisMixin,
    HamiltonianMixin,
    RowPlanMixin,
    RowCacheMixin,
    PredictionMixin,
):
    """Public compatibility facade for the signed-row Bloch forward model.

    Mixins own disjoint responsibilities and share the state initialized here.
    Dense propagation remains here to keep its explicit class-level fallbacks
    and existing standalone script semantics unchanged.
    """

    def __init__(
        self, data: XDSData, res_model, cfg: Config, cache_dir: Path | None = None, res_hash: str | None = None
    ):
        """Bind one reflection table, model and configuration, and build the shared Bloch state.

        Loads or builds the Ug cube (skipped in sparse_active storage), the beam pool, the thickness
        quadrature, the frame scale groups and the ZCAL range used by frame-dependent terms. A geometry-path
        contract must carry the hash of the current HKL file.
        """

        self.data = data
        source_hashes = self.data.verify_source_files()
        self.model = copy.deepcopy(res_model)
        self.cfg = cfg
        self.geometry_path = load_geometry_path_contract(getattr(cfg, "geometry_path_file", ""))
        if self.geometry_path is not None:
            expected_hkl_hash = str(self.geometry_path.get("source_hkl_sha256", ""))
            actual_hkl_hash = source_hashes["hkl"]
            if not expected_hkl_hash:
                raise ValueError("geometry path is missing its source HKL hash")
            if expected_hkl_hash != actual_hkl_hash:
                raise ValueError("geometry path source HKL hash does not match the current reflection table")
        self.gamma = relativistic_gamma_from_wavelength(data.wavelength) if cfg.gamma is None else float(cfg.gamma)
        self.res_hash = res_hash or "unknown"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model["cell"] = (data.cell.a, data.cell.b, data.cell.c, data.cell.alpha, data.cell.beta, data.cell.gamma)
        self.h_lim, self.k_lim, self.l_lim = data.cell.get_hkl_limits(float(self.cfg.ug_d_min))
        requested_ug_storage = str(getattr(self.cfg, "ug_storage_mode", "auto") or "auto").lower()
        row_plan_mode = str(getattr(self.cfg, "row_plan_cache", "off") or "off").lower()
        self.ug_storage_mode = (
            "sparse_active"
            if requested_ug_storage == "auto" and row_plan_mode != "off"
            else ("dense" if requested_ug_storage == "auto" else requested_ug_storage)
        )
        if self.ug_storage_mode == "sparse_active":
            self.ug_cube = None
            self.ug_cache_status = {
                "status": "sparse_active",
                "allocation": {
                    "dense_cube_skipped": True,
                    "reason": "row-plan cache evaluates only its exact active HKL differences",
                },
            }
        else:
            self.ug_cube = self.load_or_build_ug_cube()
        self.pool_hkl, self.pool_ug = self.build_pool()
        self.pool_inv_d2 = self.inv_d2_for_hkls(self.pool_hkl)
        self.thick, self.thick_w = self.thickness_grid()
        self.group_indices = self.make_groups()
        valid_rlp = np.asarray(self.data.rlp, dtype=float)
        valid_rlp = valid_rlp[np.isfinite(valid_rlp) & (valid_rlp > 0.0)]
        self.median_rlp = float(np.median(valid_rlp)) if valid_rlp.size else 1.0
        z_valid = np.asarray(self.data.zcal, dtype=float)
        z_valid = z_valid[np.isfinite(z_valid)]
        self.zcal_min = float(np.min(z_valid)) if z_valid.size else 0.0
        self.zcal_max = float(np.max(z_valid)) if z_valid.size else self.zcal_min
        self.zcal_span = max(self.zcal_max - self.zcal_min, 1e-12)
        self.dose_ref_frame = float(getattr(self.cfg, "dose_b_ref_frame", float("nan")))
        if not np.isfinite(self.dose_ref_frame):
            self.dose_ref_frame = float(np.min(z_valid)) if z_valid.size else 0.0

    @staticmethod
    def _nonhermitian_spectral_target_coupling(H):
        """Return a stable eigensystem coupling, or ``None`` when direct expm is safer."""

        H = np.asarray(H, dtype=np.complex128)
        try:
            vals, vecs = eig(H)
            condition = float(np.linalg.cond(vecs))
            if not math.isfinite(condition) or condition > 1e10:
                return None
            inv_vecs = np.linalg.inv(vecs)
            reconstruction = vecs @ np.diag(vals) @ inv_vecs
            relative_error = float(np.linalg.norm(reconstruction - H) / max(np.linalg.norm(H), np.finfo(float).eps))
            if not math.isfinite(relative_error) or relative_error > 1e-8:
                return None
        except np.linalg.LinAlgError:
            return None
        coupling = vecs[1, :] * inv_vecs[:, 0]
        return np.asarray(vals, dtype=np.complex128), np.asarray(coupling, dtype=np.complex128)

    @staticmethod
    def _uniform_scalar_absorption_decomposition(H, rtol=None):
        """Return ``(Hermitian H0, mu)`` when ``H = H0 + i*mu*I`` exactly.

        A uniform imaginary potential commutes with the elastic Hamiltonian.
        Its wave-amplitude effect is the scalar ``exp(-2*pi*mu*t)`` and its
        intensity effect is ``exp(-4*pi*mu*t)``.  Recognizing this case avoids
        a general complex eigensolve without approximating the physics.

        With ``rtol`` the anti-Hermitian part must equal ``i*mu*I`` within
        ``rtol*max|H|`` (single-precision matrices round U(g) and U(-g)
        independently) and the exact Hermitian part ``(H + H^H)/2`` is
        returned, as in the reference ``split_uniform_absorption`` (tests/reference/reference_lib.py).
        """

        matrix = np.asarray(H, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return None
        diagonal_imaginary = np.imag(np.diag(matrix))
        if not np.all(np.isfinite(diagonal_imaginary)):
            return None
        if rtol is not None:
            anti = 0.5 * (matrix - matrix.conj().T)
            mu_diag = np.imag(np.diag(anti))
            mu = float(np.mean(mu_diag))
            bound = float(rtol) * max(float(np.max(np.abs(matrix))), 1e-30)
            off_diagonal = anti - np.diag(np.diag(anti))
            if float(np.max(np.abs(off_diagonal), initial=0.0)) > bound or float(np.max(np.abs(mu_diag - mu))) > bound:
                return None
            return 0.5 * (matrix + matrix.conj().T), mu
        mu = float(np.mean(diagonal_imaginary))
        elastic = matrix.copy()
        elastic[np.diag_indices(len(elastic))] -= 1j * mu
        if not np.allclose(elastic, elastic.conj().T, atol=1e-12, rtol=0.0):
            return None
        return elastic, mu

    @staticmethod
    def _nonhermitian_target_intensities(H, thicknesses):
        """Propagate a non-Hermitian matrix, falling back for defective eigensystems."""

        H = np.asarray(H, dtype=np.complex128)
        thicknesses = np.atleast_1d(np.asarray(thicknesses, dtype=float))
        spectral = XDSBlochPort._nonhermitian_spectral_target_coupling(H)

        if spectral is None:
            psi0 = np.zeros(H.shape[0], dtype=np.complex128)
            psi0[0] = 1.0 + 0.0j
            return np.asarray(
                [abs((expm((2j * np.pi * float(t)) * H) @ psi0)[1]) ** 2 for t in thicknesses],
                dtype=float,
            )

        vals, coupling = spectral
        phase = np.exp((2j * np.pi) * vals[:, None] * thicknesses[None, :])
        return np.asarray(np.abs(np.sum(coupling[:, None] * phase, axis=0)) ** 2, dtype=float)

    def propagate_direct_to_target(self, H, thickness):
        """Return the target-beam intensity after propagating the direct beam through one thickness.

        Beam 0 is the direct beam and beam 1 the target; thickness is in Å and H in Å⁻¹. The configured
        compute backend chooses the eigensolver, expm or JAX path.
        """

        backend = str(getattr(self.cfg, "compute_backend", "scipy") or "scipy").lower()
        if backend in BETHE_BACKENDS:
            H = promoted_bethe_effective_hamiltonian(H, self.cfg)
            if self.absorption_enabled():
                validate_passive_absorption(H, mode=f"Bethe {self.cfg.bethe_variant} approximation")
            backend = "scipy"
        if backend in {"scipy", "numpy_scipy_cpu"}:
            if self.absorption_enabled():
                uniform = self._uniform_scalar_absorption_decomposition(H)
                if uniform is None:
                    return float(self._nonhermitian_target_intensities(H, [thickness])[0])
                elastic, mu = uniform
                vals, vecs = hermitian_eigh(elastic)
                coupling = vecs[1, :] * np.conj(vecs[0, :])
                phase = np.exp(2j * np.pi * vals * thickness)
                return float(abs(np.sum(coupling * phase)) ** 2 * math.exp(-4.0 * math.pi * mu * float(thickness)))
            else:
                vals, vecs = hermitian_eigh(H)
                coupling = vecs[1, :] * np.conj(vecs[0, :])
            phase = np.exp(2j * np.pi * vals * thickness)
            return abs(np.sum(coupling * phase)) ** 2
        if backend in {"scipy_expm", "expm"}:
            psi0 = np.zeros(H.shape[0], dtype=np.complex128)
            psi0[0] = 1.0 + 0.0j
            psi = expm((2j * np.pi * float(thickness)) * H) @ psi0
            return float(abs(psi[1]) ** 2)
        if backend in {"scipy_expm_multiply", "expm_multiply"}:
            psi0 = np.zeros(H.shape[0], dtype=np.complex128)
            psi0[0] = 1.0 + 0.0j
            psi = expm_multiply((2j * np.pi * float(thickness)) * H, psi0)
            return float(abs(psi[1]) ** 2)
        if backend in ALL_JAX_BACKENDS:
            values = _jax_propagate_batch(
                np.asarray(H, dtype=np.complex128)[None, :, :],
                np.asarray([float(thickness)]), np.asarray([1.0]), backend,
                self.absorption_enabled(), int(getattr(self.cfg, "jax_batch_size", 0)),
            )
            return float(values[0])
        raise NotImplementedError(f"unknown compute_backend={backend!r}; expected one of {COMPUTE_BACKEND_HELP}")

    def propagate_direct_to_target_gamma_cf(self, H, cf):
        """Exact untruncated gamma thickness average of both hands from one Hermitian eigensystem.

        The gamma characteristic function replaces the thickness nodes (grond.bloch.thickness.gamma_cf_point).
        It needs H = H0 + i*mu*I; any other absorptive potential is refused rather than approximated. H0 is the
        exact Hermitian part (H + H^H)/2, as in tests/reference/reference_lib.py matrix_spectrum.
        """

        matrix = np.asarray(H, dtype=np.complex128)
        mu = 0.0
        if self.absorption_enabled():
            uniform = self._uniform_scalar_absorption_decomposition(matrix, rtol=2e-6)
            if uniform is None:
                raise ValueError(
                    "thickness_evaluator cf needs a uniform scalar absorptive potential H = H0 + i*mu*I; this "
                    "Hamiltonian has a non-uniform or off-diagonal absorptive part. Use thickness_evaluator table"
                )
            elastic, mu = uniform
        else:
            elastic = 0.5 * (matrix + matrix.conj().T)
        vals, vecs = hermitian_eigh(elastic)
        coupling = vecs[1, :] * np.conj(vecs[0, :])
        return gamma_cf_point(vals, coupling, cf["shape"], cf["theta"], alpha=4.0 * np.pi * float(mu))

    def _uniform_table_intensities(self, vals, coupling, inverse=False):
        """|amplitude|^2 at the table evaluator's uniform nodes by exact factorised phases; None for other nodes.

        grond.bloch.thickness.uniform_grid_amplitudes equals the direct exp(2*pi*i*lam*t) sum to rounding and makes
        the 4,951-node support table cost one small matrix product per profile point. The legacy quadrature and any
        non-uniform node set keep the direct expression.
        """

        grid = self.thickness_uniform_grid()
        if grid is None:
            return None
        return np.abs(uniform_grid_amplitudes(vals, coupling, *grid, inverse=inverse)[0]) ** 2

    def propagate_direct_to_target_thickness_average(self, H):
        """Propagate one Bloch matrix and average over the configured thickness quadrature."""

        cf = self.thickness_cf_parameters()
        if cf is not None:
            return self.propagate_direct_to_target_gamma_cf(H, cf)[0]
        backend = str(getattr(self.cfg, "compute_backend", "scipy") or "scipy").lower()
        if backend in BETHE_BACKENDS:
            H = promoted_bethe_effective_hamiltonian(H, self.cfg)
            if self.absorption_enabled():
                validate_passive_absorption(H, mode=f"Bethe {self.cfg.bethe_variant} approximation")
            backend = "scipy"
        if backend in {"scipy", "numpy_scipy_cpu"}:
            if self.absorption_enabled():
                uniform = self._uniform_scalar_absorption_decomposition(H)
                if uniform is None:
                    per_t = self._nonhermitian_target_intensities(H, self.thick)
                    return float(np.sum(self.thick_w * per_t))
                elastic, mu = uniform
                vals, vecs = hermitian_eigh(elastic)
                coupling = vecs[1, :] * np.conj(vecs[0, :])
                per_t = self._uniform_table_intensities(vals, coupling)
                if per_t is None:
                    phase = np.exp((2j * np.pi) * vals[:, None] * self.thick[None, :])
                    per_t = np.abs(np.sum(coupling[:, None] * phase, axis=0)) ** 2
                per_t *= np.exp(-4.0 * np.pi * mu * self.thick)
                return float(np.sum(self.thick_w * per_t))
            else:
                vals, vecs = hermitian_eigh(H)
                coupling = vecs[1, :] * np.conj(vecs[0, :])
            per_t = self._uniform_table_intensities(vals, coupling)
            if per_t is None:
                phase = np.exp((2j * np.pi) * vals[:, None] * self.thick[None, :])
                per_t = np.abs(np.sum(coupling[:, None] * phase, axis=0)) ** 2
            return float(np.sum(self.thick_w * per_t))
        if backend in {"scipy_expm", "expm", "scipy_expm_multiply", "expm_multiply"}:
            psi0 = np.zeros(H.shape[0], dtype=np.complex128)
            psi0[0] = 1.0 + 0.0j
            per_t = []
            for t in self.thick:
                if backend in {"scipy_expm", "expm"}:
                    psi = expm((2j * np.pi * float(t)) * H) @ psi0
                else:
                    psi = expm_multiply((2j * np.pi * float(t)) * H, psi0)
                per_t.append(float(abs(psi[1]) ** 2))
            return float(np.sum(self.thick_w * np.asarray(per_t, dtype=float)))
        if backend in ALL_JAX_BACKENDS:
            vals = _jax_propagate_batch(
                np.asarray(H, dtype=np.complex128)[None, :, :],
                self.thick,
                self.thick_w,
                backend,
                self.absorption_enabled(),
                int(getattr(self.cfg, "jax_batch_size", 0)),
            )
            return float(vals[0])
        raise NotImplementedError(f"unknown compute_backend={backend!r}; expected one of {COMPUTE_BACKEND_HELP}")

    def propagate_direct_to_target_thickness_average_pair(self, H):
        """Return original- and inverse-hand intensities from one eigensystem.

        For elastic scattering the inverted-coordinate Hamiltonian is ``H*``.
        With a real incident direct-beam vector, propagation through ``H*`` at
        positive thickness has the same intensity as propagation through ``H``
        at negative thickness.  The two signs therefore share one Hermitian
        eigendecomposition.  A uniform absorptive potential ``H = H0 + i*mu*I``
        keeps the identity exactly: the inverse hand is ``conj(H0) + i*mu*I``
        and both hands carry the same scalar ``exp(-4*pi*mu*t)``.  Any other
        absorptive potential is rejected rather than silently approximated.
        """

        backend = str(getattr(self.cfg, "compute_backend", "scipy") or "scipy").lower()
        if backend not in {"scipy", "numpy_scipy_cpu"}:
            raise NotImplementedError(
                "paired inverse-hand propagation requires the full audited SciPy Hermitian "
                "eigensolver; choose --compute-backend scipy"
            )
        cf = self.thickness_cf_parameters()
        if cf is not None:
            return self.propagate_direct_to_target_gamma_cf(H, cf)
        elastic = np.asarray(H, dtype=np.complex128)
        mu = 0.0
        if self.absorption_enabled():
            # complex64 matrices: compare the anti-Hermitian part to i*mu*I at
            # single-precision tolerance.
            uniform = self._uniform_scalar_absorption_decomposition(elastic, rtol=2e-6)
            if uniform is None:
                raise ValueError(
                    "paired inverse-hand propagation requires a uniform scalar absorptive potential "
                    "H = H0 + i*mu*I; this Hamiltonian has a non-uniform or off-diagonal absorptive part. "
                    "Run explicit coordinate-inverted branches for that absorption model"
                )
            elastic, mu = uniform
        vals, vecs = hermitian_eigh(elastic)
        coupling = vecs[1, :] * np.conj(vecs[0, :])
        attenuation = np.exp(-4.0 * np.pi * mu * self.thick)
        forward = self._uniform_table_intensities(vals, coupling)
        if forward is not None:
            forward = forward * attenuation
            reverse = self._uniform_table_intensities(vals, coupling, inverse=True) * attenuation
        else:
            phase = np.exp((2j * np.pi) * vals[:, None] * self.thick[None, :])
            forward = np.abs(np.sum(coupling[:, None] * phase, axis=0)) ** 2 * attenuation
            reverse = np.abs(np.sum(coupling[:, None] * np.conj(phase), axis=0)) ** 2 * attenuation
        return (
            float(np.sum(self.thick_w * forward)),
            float(np.sum(self.thick_w * reverse)),
        )


def main():
    """Run the XDS Bloch forward model from the command line and write the summary JSON to --out."""

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hkl", default=str(DEFAULT_HKL))
    ap.add_argument("--res", default=str(DEFAULT_RES))
    ap.add_argument(
        "--hkl-transform",
        default="identity",
        help="Signed axis permutation mapping XDS HKLs into the model basis, e.g. 'l,-k,h'.",
    )
    ap.add_argument("--out", default=str(ROOT / "outputs/engine_summary.json"))
    ap.add_argument(
        "--cache-dir",
        default=str(runtime_cache_dir("legacy_engine")),
        help="Local Ug cube cache directory. Use 'none' to disable.",
    )
    ap.add_argument("--n-basis", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=20)
    ap.add_argument("--sg-window", type=float, default=0.006)
    ap.add_argument(
        "--sg-window-mode",
        choices=["fixed", "zobs_residual", "zobs_plus_floor", "rlp_peak", "dials_bbox"],
        default="fixed",
    )
    ap.add_argument("--sg-window-min", type=float, default=0.003)
    ap.add_argument("--sg-window-max", type=float, default=0.012)
    ap.add_argument("--sg-window-scale", type=float, default=1.0)
    ap.add_argument(
        "--sg-window-frame-floor",
        type=float,
        default=0.50,
        help="Minimum half-width in frames for ZOBS/ZCAL adaptive Sg-window modes.",
    )
    ap.add_argument(
        "--sg-jitter-mode",
        choices=["none", "gaussian", "student_t"],
        default="none",
        help="Optional target-Sg orientation-distribution convolution.",
    )
    ap.add_argument(
        "--sg-jitter-sigma0", type=float, default=0.0, help="Base target-Sg jitter sigma in inverse Angstrom."
    )
    ap.add_argument("--sg-jitter-sigma-d-inv", type=float, default=0.0, help="Target-Sg jitter sigma slope versus 1/d.")
    ap.add_argument(
        "--sg-jitter-sigma-frame",
        type=float,
        default=0.0,
        help="Target-Sg jitter sigma slope versus normalized frame number.",
    )
    ap.add_argument(
        "--sg-jitter-points", type=int, default=5, help="Kernel quadrature points for Sg-jitter convolution."
    )
    ap.add_argument(
        "--sg-jitter-student-df", type=float, default=4.0, help="Student-t degrees of freedom for Sg-jitter mode."
    )
    ap.add_argument(
        "--sg-jitter-width-factor",
        type=float,
        default=3.0,
        help="Truncation half-width in sigma units for Student-t jitter mode.",
    )
    ap.add_argument(
        "--mosaicity-mode",
        choices=["none", "gaussian_omega", "gaussian_cone", "transverse_target"],
        default="none",
        help=(
            "Optional projected one-axis or incoherent two-axis Gaussian mosaic convolution; transverse_target is the "
            "re-centred transverse cone (requires --profile-coordinate target_sg)."
        ),
    )
    ap.add_argument(
        "--mosaicity-sigma-deg",
        type=float,
        default=0.0,
        help="Projected Gaussian mosaicity sigma in degrees; never inferred from the XDS reflecting range automatically.",
    )
    ap.add_argument(
        "--mosaicity-max-sigma-deg", type=float, default=0.5, help="Hard upper bound for the projected mosaicity sigma."
    )
    ap.add_argument(
        "--mosaicity-points", type=int, default=3, help="Gauss-Hermite points for direct angular mosaicity quadrature."
    )
    ap.add_argument(
        "--mosaicity-support-sigma-deg",
        type=float,
        default=0.0,
        help="Largest tested two-axis cone sigma. Every candidate uses the beam basis selected over this same envelope.",
    )
    ap.add_argument(
        "--mosaicity-support-points",
        type=int,
        default=5,
        help="Points per scan/cone axis used only to construct the common enclosing beam basis.",
    )
    ap.add_argument(
        "--mosaicity-support-width-sigma",
        type=float,
        default=4.0,
        help="Half-width of the common basis cone in support-sigma units.",
    )
    ap.add_argument(
        "--geometry-path",
        default="",
        help="Optional geometry-only Chebyshev slew/two-axis tilt JSON produced independently of intensities.",
    )
    ap.add_argument(
        "--profile-basis-mode",
        choices=["center", "common_profile"],
        default="center",
        help="Use the row-center beam basis or a fixed beam union over the complete rocking-profile support.",
    )
    ap.add_argument(
        "--profile-coordinate",
        choices=["target_sg", "direct_angle"],
        default="target_sg",
        help="Integrate through solved target-Sg points or directly in recorded scan angle.",
    )
    ap.add_argument(
        "--profile-angle-max-half-width-deg",
        type=float,
        default=2.0,
        help="Hard half-width bound for direct-angle integration.",
    )
    ap.add_argument(
        "--profile-quadrature",
        choices=["uniform", "trapezoid", "gauss_legendre"],
        default="uniform",
        help="Rocking-profile Sg quadrature rule.",
    )
    ap.add_argument(
        "--profile-accumulation",
        choices=["sum", "mean"],
        default="sum",
        help="Accumulate rocking-profile samples as summed detector sub-exposures or legacy normalized means.",
    )
    ap.add_argument("--topup-d-min", type=float, default=0.8)
    ap.add_argument(
        "--max-support-sg",
        type=float,
        default=DEFAULT_MAX_SUPPORT_SG,
        help="Maximum absolute excitation error for beam-basis support (default 0.01 A^-1).",
    )
    ap.add_argument("--thickness", type=float, default=1240.0)
    ap.add_argument(
        "--thickness-model",
        choices=["single", "uniform_gradient", "gaussian", "gaussian_gradient", "gamma"],
        default="uniform_gradient",
    )
    ap.add_argument("--thickness-grad-frac", type=float, default=0.25)
    ap.add_argument("--thickness-sigma-frac", type=float, default=0.0)
    ap.add_argument("--thickness-points", type=int, default=7)
    ap.add_argument(
        "--thickness-gamma-shape", type=float, default=1.0, help="Gamma thickness shape k (mean is --thickness)."
    )
    ap.add_argument(
        "--thickness-evaluator",
        choices=["quadrature", "table", "cf", "auto"],
        default="quadrature",
        help="quadrature: legacy nodes. table/cf/auto: support evaluators of grond.bloch.thickness.",
    )
    ap.add_argument("--thickness-min-nm", type=float, default=10.0, help="Support minimum for table/cf/auto (nm).")
    ap.add_argument("--thickness-max-nm", type=float, default=1000.0, help="Support maximum for table/cf/auto (nm).")
    ap.add_argument("--thickness-table-dt", type=float, default=2.0, help="Table step for table/auto (A).")
    ap.add_argument("--group-width-deg", type=float, default=3.0)
    ap.add_argument("--variance-a", type=float, default=0.0)
    ap.add_argument("--variance-b-corr", type=float, default=0.0)
    ap.add_argument("--variance-c-peak", type=float, default=0.0)
    ap.add_argument(
        "--frame-offset", type=float, default=0.0, help="Global frame offset added to ZCAL before omega replay."
    )
    ap.add_argument(
        "--ug-d-min",
        type=float,
        default=0.30,
        help="Resolution limit for model-dependent U_g cache. Keep 0.30 for endpoint/audit runs; larger values are diagnostic speedups.",
    )
    ap.add_argument(
        "--absorption-mode",
        choices=["none", "mean", "beam_dependent", "offdiag_fraction", "offdiag_tds", "midgley_like"],
        default="none",
    )
    ap.add_argument(
        "--absorption-mu", type=float, default=0.0, help="Uniform imaginary diagonal damping in inverse Angstrom."
    )
    ap.add_argument(
        "--absorption-frac", type=float, default=0.0, help="Imaginary absorptive off-diagonal fraction relative to U_g."
    )
    ap.add_argument(
        "--absorption-mu-g2", type=float, default=0.0, help="Beam-dependent diagonal damping slope versus |g|^2."
    )
    ap.add_argument(
        "--absorption-mu-theta2",
        type=float,
        default=0.0,
        help="Beam-dependent diagonal damping slope versus Bragg-angle^2.",
    )
    ap.add_argument(
        "--absorption-tds-b",
        type=float,
        default=0.0,
        help="B_TDS in eta*U_g*(1-exp(-B_TDS*|g|^2/4)) for offdiag_tds mode.",
    )
    ap.add_argument("--dose-b-mode", choices=["none", "linear_frame"], default="none")
    ap.add_argument(
        "--dose-b-per-frame",
        type=float,
        default=0.0,
        help="Extra isotropic B added per XDS frame after the reference frame.",
    )
    ap.add_argument(
        "--dose-b-ref-frame",
        type=float,
        default=float("nan"),
        help="Reference ZCAL frame for dose B. Default is the minimum selected/data frame.",
    )
    ap.add_argument(
        "--matrix-convention",
        choices=["j_minus_i", "i_minus_j"],
        default="j_minus_i",
        help="Off-diagonal U_g indexing convention. j_minus_i is the production signed-row default; i_minus_j is a diagnostic.",
    )
    ap.add_argument(
        "--alternate-hand",
        dest="alternate_hand",
        action="store_true",
        help="Use alternate fractional coordinates as an alternate-hand control run.",
    )
    ap.add_argument("--free-fraction", type=float, default=0.10)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="Debug limit. 0 means all rows.")
    ap.add_argument("--top-isigma", type=int, default=0, help="Keep the strongest I/sigma rows after other filters.")
    ap.add_argument(
        "--row-index-file", default="", help="Optional text/CSV file of zero-based XDS row indices to predict."
    )
    ap.add_argument(
        "--save-row-table", default="", help="Optional CSV with row-wise metadata, predictions, and residuals."
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--compute-backend", choices=COMPUTE_BACKENDS, default="scipy")
    ap.add_argument(
        "--jax-batch-size",
        type=int,
        default=0,
        help="JAX matrix batch size. 0 chooses a conservative size from matrix dimension and device count.",
    )
    ap.add_argument(
        "--jax-matrix-bucket-size",
        type=int,
        default=0,
        help="Pad JAX matrix dimensions to this multiple. 0 keeps exact sizes unless a *_padded backend is selected, where 0 means 8.",
    )
    ap.add_argument(
        "--bethe-strong-count",
        type=int,
        default=48,
        help="Strong-beam block size for scipy_bethe. Values >= n_basis recover the dense SciPy matrix.",
    )
    ap.add_argument(
        "--bethe-variant",
        choices=["truncate", "schur0", "schur_target", "schur_mean"],
        default="schur_target",
        help="Strong/weak-beam reduction used by scipy_bethe.",
    )
    ap.add_argument(
        "--bethe-promote-denom",
        type=float,
        default=0.005,
        help="Promote weak beams whose diagonal denominator is this close to the Schur energy.",
    )
    ap.add_argument(
        "--bethe-denom-eps", type=float, default=1e-5, help="Small-denominator floor for scipy_bethe Schur corrections."
    )
    ap.add_argument(
        "--row-plan-cache",
        choices=["off", "exact", "readwrite", "rebuild"],
        default="off",
        help=(
            "Cache row geometry, basis, Sg profiles, and h_j-h_i indices. "
            "exact keys the basis to the current model; readwrite is the "
            "opt-in frozen-basis mode across model updates."
        ),
    )
    ap.add_argument(
        "--positive-only", action="store_true", help="Ignore non-positive XDS IOBS rows for fitting/evaluation."
    )
    ap.add_argument(
        "--min-peak",
        type=float,
        default=90.0,
        help="Default PEAK/partiality cutoff for XDS row selection. Use 0 for a true full-row diagnostic.",
    )
    ap.add_argument("--min-corr", type=float, default=-999.0)
    ap.add_argument("--max-resolution", type=float, default=0.0, help="Minimum d spacing in Angstrom; 0 disables.")
    ap.add_argument(
        "--double",
        "--float64",
        dest="double_precision",
        action="store_true",
        default=False,
        help="Use exact complex128/float64 double precision (default: complex64 single precision).",
    )
    args = ap.parse_args()

    cfg = Config(
        double_precision=bool(args.double_precision),
        n_basis=args.n_basis,
        n_points=args.n_points,
        sg_window=args.sg_window,
        sg_window_mode=args.sg_window_mode,
        sg_window_min=args.sg_window_min,
        sg_window_max=args.sg_window_max,
        sg_window_scale=args.sg_window_scale,
        sg_window_frame_floor=args.sg_window_frame_floor,
        sg_jitter_mode=args.sg_jitter_mode,
        sg_jitter_sigma0=args.sg_jitter_sigma0,
        sg_jitter_sigma_d_inv=args.sg_jitter_sigma_d_inv,
        sg_jitter_sigma_frame=args.sg_jitter_sigma_frame,
        sg_jitter_points=args.sg_jitter_points,
        sg_jitter_student_df=args.sg_jitter_student_df,
        sg_jitter_width_factor=args.sg_jitter_width_factor,
        mosaicity_mode=args.mosaicity_mode,
        mosaicity_sigma_deg=args.mosaicity_sigma_deg,
        mosaicity_max_sigma_deg=args.mosaicity_max_sigma_deg,
        mosaicity_points=args.mosaicity_points,
        mosaicity_support_sigma_deg=args.mosaicity_support_sigma_deg,
        mosaicity_support_points=args.mosaicity_support_points,
        mosaicity_support_width_sigma=args.mosaicity_support_width_sigma,
        geometry_path_file=args.geometry_path,
        profile_basis_mode=args.profile_basis_mode,
        profile_coordinate=args.profile_coordinate,
        profile_angle_max_half_width_deg=args.profile_angle_max_half_width_deg,
        profile_quadrature=args.profile_quadrature,
        profile_accumulation=args.profile_accumulation,
        topup_d_min=args.topup_d_min,
        max_support_sg=args.max_support_sg,
        thickness_A=args.thickness,
        thickness_model=args.thickness_model,
        thickness_grad_frac=args.thickness_grad_frac,
        thickness_sigma_frac=args.thickness_sigma_frac,
        thickness_points=args.thickness_points,
        thickness_gamma_shape=args.thickness_gamma_shape,
        thickness_evaluator=args.thickness_evaluator,
        thickness_min_A=10.0 * args.thickness_min_nm,
        thickness_max_A=10.0 * args.thickness_max_nm,
        thickness_table_dt_A=args.thickness_table_dt,
        group_width_deg=args.group_width_deg,
        variance_a=args.variance_a,
        variance_b_corr=args.variance_b_corr,
        variance_c_peak=args.variance_c_peak,
        frame_offset=args.frame_offset,
        ug_d_min=args.ug_d_min,
        absorption_mode=args.absorption_mode,
        absorption_mu=args.absorption_mu,
        absorption_frac=args.absorption_frac,
        absorption_mu_g2=args.absorption_mu_g2,
        absorption_mu_theta2=args.absorption_mu_theta2,
        absorption_tds_b=args.absorption_tds_b,
        dose_b_mode=args.dose_b_mode,
        dose_b_per_frame=args.dose_b_per_frame,
        dose_b_ref_frame=args.dose_b_ref_frame,
        compute_backend=args.compute_backend,
        jax_batch_size=args.jax_batch_size,
        jax_matrix_bucket_size=args.jax_matrix_bucket_size,
        bethe_strong_count=args.bethe_strong_count,
        bethe_variant=args.bethe_variant,
        bethe_promote_denom=args.bethe_promote_denom,
        bethe_denom_eps=args.bethe_denom_eps,
        matrix_convention=args.matrix_convention,
        row_plan_cache=args.row_plan_cache,
        free_fraction=args.free_fraction,
        split_seed=args.split_seed,
    )
    data_all = XDSData(Path(args.hkl), hkl_transform=args.hkl_transform)

    if args.row_index_file:
        row_indices = read_row_index_file(args.row_index_file, row_count=len(data_all.iobs))
        indices = np.asarray(row_indices, dtype=int)
    else:
        mask = np.isfinite(data_all.sigma) & (data_all.sigma > 0.0) & np.isfinite(data_all.zcal)
        if args.positive_only:
            mask &= data_all.iobs > 0.0
        if args.min_peak > 0:
            mask &= data_all.peak >= args.min_peak
        if args.min_corr > -999:
            mask &= data_all.corr >= args.min_corr
        if args.max_resolution > 0:
            dvals = np.asarray([data_all.cell.calculate_d(tuple(int(v) for v in h)) for h in data_all.hkl])
            mask &= dvals >= args.max_resolution
        indices = np.where(mask)[0]
    if args.top_isigma and args.top_isigma > 0 and len(indices) > args.top_isigma:
        isigi = data_all.iobs[indices] / np.maximum(data_all.sigma[indices], 1e-12)
        indices = indices[np.argsort(isigi)[::-1][: int(args.top_isigma)]]
    if args.limit and args.limit > 0:
        indices = indices[: int(args.limit)]

    model = parse_model(Path(args.res))
    if args.alternate_hand:
        model = alternate_hand_model(model)
    res_hash = sha256(Path(args.res))
    cache_dir = None if str(args.cache_dir).lower() == "none" else Path(args.cache_dir)
    wall0 = time.perf_counter()
    port = XDSBlochPort(data_all, model, cfg, cache_dir=cache_dir, res_hash=res_hash)
    pred_subset = port.predict_rows(indices, workers=args.workers, chunksize=args.chunksize)
    wall_seconds = time.perf_counter() - wall0
    full_dyn = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_born = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_basis_size = np.zeros(len(data_all.iobs), dtype=int)
    full_sg_resid = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_center_sg = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_omega_bragg = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_omega_delta = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_omega_bragg_root_available = np.zeros(len(data_all.iobs), dtype=bool)
    full_sg_window = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_sg_domega = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_sg_dframe = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_sg_window_raw = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_sg_window_half_frames = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_sg_window_clipped = np.zeros(len(data_all.iobs), dtype=bool)
    full_sg_jitter_sigma = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_sg_jitter_d_inv = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_sg_jitter_frame_fraction = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_sg_jitter_kernel_points = np.zeros(len(data_all.iobs), dtype=int)
    full_sg_profile_points_total = np.zeros(len(data_all.iobs), dtype=int)
    full_sg_profile_weight_sum = np.full(len(data_all.iobs), np.nan, dtype=float)
    full_sg_profile_same_branch_continuation_used = np.zeros(len(data_all.iobs), dtype=bool)
    full_sg_profile_same_branch_continuation_segments = np.ones(len(data_all.iobs), dtype=int)
    full_dyn[pred_subset["indices"]] = pred_subset["dyn"]
    full_born[pred_subset["indices"]] = pred_subset["born"]
    full_basis_size[pred_subset["indices"]] = pred_subset["basis_size"]
    full_sg_resid[pred_subset["indices"]] = pred_subset["sg_residual"]
    full_center_sg[pred_subset["indices"]] = pred_subset["center_sg"]
    full_omega_bragg[pred_subset["indices"]] = pred_subset["omega_bragg"]
    full_omega_delta[pred_subset["indices"]] = pred_subset["omega_delta"]
    full_omega_bragg_root_available[pred_subset["indices"]] = pred_subset.get(
        "omega_bragg_root_available",
        np.isfinite(pred_subset["omega_bragg"]),
    )
    full_sg_window[pred_subset["indices"]] = pred_subset["sg_window"]
    full_sg_domega[pred_subset["indices"]] = pred_subset["sg_dSg_domega"]
    full_sg_dframe[pred_subset["indices"]] = pred_subset["sg_dSg_dframe"]
    full_sg_window_raw[pred_subset["indices"]] = pred_subset["sg_window_raw"]
    full_sg_window_half_frames[pred_subset["indices"]] = pred_subset["sg_window_half_frames"]
    full_sg_window_clipped[pred_subset["indices"]] = pred_subset["sg_window_clipped"]
    full_sg_jitter_sigma[pred_subset["indices"]] = pred_subset["sg_jitter_sigma"]
    full_sg_jitter_d_inv[pred_subset["indices"]] = pred_subset["sg_jitter_d_inv"]
    full_sg_jitter_frame_fraction[pred_subset["indices"]] = pred_subset["sg_jitter_frame_fraction"]
    full_sg_jitter_kernel_points[pred_subset["indices"]] = pred_subset["sg_jitter_kernel_points"]
    full_sg_profile_points_total[pred_subset["indices"]] = pred_subset["sg_profile_points_total"]
    full_sg_profile_weight_sum[pred_subset["indices"]] = pred_subset["sg_profile_weight_sum"]
    full_sg_profile_same_branch_continuation_used[pred_subset["indices"]] = pred_subset[
        "sg_profile_same_branch_continuation_used"
    ]
    full_sg_profile_same_branch_continuation_segments[pred_subset["indices"]] = pred_subset[
        "sg_profile_same_branch_continuation_segments"
    ]
    support_by_idx = {
        int(idx): basis for idx, basis in zip(pred_subset["indices"], pred_subset["support_hkls"], strict=True)
    }
    pred = {"dyn": full_dyn, "born": full_born, "groups": port.group_indices}
    valid = np.isfinite(full_dyn) & np.isfinite(full_born)
    eval_data = data_all
    eval_out = evaluate_predictions(eval_data, cfg, pred, fit_scales_on="work", model=model)
    row_table_path = ""
    if args.save_row_table:
        row_table_path = str(Path(args.save_row_table).resolve())
        sig_eff = effective_sigma(data_all, cfg)
        work_mask = split_masks(
            data_all.hkl,
            cfg.free_fraction,
            cfg.split_seed,
            symmetry_ops=model.get("symmetry_ops"),
        )["work"]
        dyn_scaled, dyn_scales = scale_by_groups(data_all.iobs, sig_eff, full_dyn, port.group_indices, work_mask)
        born_scaled, born_scales = scale_by_groups(data_all.iobs, sig_eff, full_born, port.group_indices, work_mask)
        dvals = np.asarray([data_all.cell.calculate_d(tuple(int(v) for v in h)) for h in data_all.hkl])
        Path(args.save_row_table).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_row_table, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "row_index",
                    "h",
                    "k",
                    "l",
                    "d_A",
                    "iobs",
                    "sigma",
                    "isigi",
                    "xcal",
                    "ycal",
                    "zcal",
                    "xobs",
                    "yobs",
                    "zobs",
                    "dx_obs_cal",
                    "dy_obs_cal",
                    "dz_obs_cal",
                    "rlp",
                    "peak",
                    "corr",
                    "maxc",
                    "psi",
                    "omega_deg",
                    "omega_bragg_deg",
                    "omega_bragg_minus_center_deg",
                    "omega_bragg_root_available",
                    "center_sg_inv_A",
                    "max_sg_solve_residual",
                    "sg_window_inv_A",
                    "sg_window_raw_inv_A",
                    "sg_dSg_domega_inv_A_per_deg",
                    "sg_dSg_dframe_inv_A",
                    "sg_window_half_frames",
                    "sg_window_clipped",
                    "sg_jitter_sigma_inv_A",
                    "sg_jitter_d_inv_A",
                    "sg_jitter_frame_fraction",
                    "sg_jitter_kernel_points",
                    "sg_profile_points_total",
                    "sg_profile_weight_sum",
                    "sg_profile_same_branch_continuation_used",
                    "sg_profile_same_branch_continuation_segments",
                    "group",
                    "basis_size",
                    "support_hkls_json",
                    "dyn_raw",
                    "born_raw",
                    "dyn_scaled",
                    "born_scaled",
                    "dyn_minus_iobs_over_sigma",
                    "born_minus_iobs_over_sigma",
                    "dyn_over_born",
                    "dyn_group_scale",
                    "born_group_scale",
                ]
            )
            for idx in indices:
                g = int(port.group_indices[idx])
                writer.writerow(
                    [
                        int(idx),
                        *[int(v) for v in data_all.hkl[idx]],
                        float(dvals[idx]),
                        float(data_all.iobs[idx]),
                        float(sig_eff[idx]),
                        float(data_all.iobs[idx] / max(sig_eff[idx], 1e-12)),
                        float(data_all.xcal[idx]),
                        float(data_all.ycal[idx]),
                        float(data_all.zcal[idx]),
                        float(data_all.xobs[idx]),
                        float(data_all.yobs[idx]),
                        float(data_all.zobs[idx]),
                        float(data_all.xobs[idx] - data_all.xcal[idx]),
                        float(data_all.yobs[idx] - data_all.ycal[idx]),
                        float(data_all.zobs[idx] - data_all.zcal[idx]),
                        float(data_all.rlp[idx]),
                        float(data_all.peak[idx]),
                        float(data_all.corr[idx]),
                        float(data_all.maxc[idx]),
                        float(data_all.psi[idx]),
                        float(port.omega_from_z(data_all.zcal[idx])),
                        float(full_omega_bragg[idx]),
                        float(full_omega_delta[idx]),
                        int(bool(full_omega_bragg_root_available[idx])),
                        float(full_center_sg[idx]),
                        float(full_sg_resid[idx]),
                        float(full_sg_window[idx]),
                        float(full_sg_window_raw[idx]),
                        float(full_sg_domega[idx]),
                        float(full_sg_dframe[idx]),
                        float(full_sg_window_half_frames[idx]),
                        bool(full_sg_window_clipped[idx]),
                        float(full_sg_jitter_sigma[idx]),
                        float(full_sg_jitter_d_inv[idx]),
                        float(full_sg_jitter_frame_fraction[idx]),
                        int(full_sg_jitter_kernel_points[idx]),
                        int(full_sg_profile_points_total[idx]),
                        float(full_sg_profile_weight_sum[idx]),
                        int(full_sg_profile_same_branch_continuation_used[idx]),
                        int(full_sg_profile_same_branch_continuation_segments[idx]),
                        g,
                        int(full_basis_size[idx]),
                        json.dumps(support_by_idx.get(int(idx), []), separators=(",", ":")),
                        float(full_dyn[idx]),
                        float(full_born[idx]),
                        float(dyn_scaled[idx]),
                        float(born_scaled[idx]),
                        float((dyn_scaled[idx] - data_all.iobs[idx]) / max(sig_eff[idx], 1e-12))
                        if np.isfinite(dyn_scaled[idx])
                        else float("nan"),
                        float((born_scaled[idx] - data_all.iobs[idx]) / max(sig_eff[idx], 1e-12))
                        if np.isfinite(born_scaled[idx])
                        else float("nan"),
                        float(full_dyn[idx] / max(full_born[idx], 1e-30))
                        if np.isfinite(full_dyn[idx]) and np.isfinite(full_born[idx])
                        else float("nan"),
                        float(dyn_scales.get(g, float("nan"))),
                        float(born_scales.get(g, float("nan"))),
                    ]
                )
    out = {
        "inputs": {
            "hkl": str(Path(args.hkl).resolve()),
            "hkl_sha256": sha256(Path(args.hkl)),
            "hkl_transform": data_all.hkl_transform_spec,
            "raw_xds_cell": [
                data_all.raw_cell.a,
                data_all.raw_cell.b,
                data_all.raw_cell.c,
                data_all.raw_cell.alpha,
                data_all.raw_cell.beta,
                data_all.raw_cell.gamma,
            ],
            "axis_derived_source_cell": [
                data_all.axis_derived_source_cell.a,
                data_all.axis_derived_source_cell.b,
                data_all.axis_derived_source_cell.c,
                data_all.axis_derived_source_cell.alpha,
                data_all.axis_derived_source_cell.beta,
                data_all.axis_derived_source_cell.gamma,
            ],
            "axis_derived_source_cell_basis": data_all.axis_derived_source_cell_basis,
            "simulation_cell": [
                data_all.cell.a,
                data_all.cell.b,
                data_all.cell.c,
                data_all.cell.alpha,
                data_all.cell.beta,
                data_all.cell.gamma,
            ],
            "simulation_cell_basis": data_all.working_cell_basis,
            "integration_profile_provenance": data_all.integration_profile_provenance(),
            "res": str(Path(args.res).resolve()),
            "res_sha256": res_hash,
        },
        "bloch_kernel_contract": frozen_kernel_contract(cfg),
        "config": cfg.__dict__,
        "model_hand": "alternate_fractional_xyz" if args.alternate_hand else "original",
        "filters": {
            "positive_only": bool(args.positive_only),
            "min_peak": args.min_peak,
            "min_corr": args.min_corr,
            "max_resolution": args.max_resolution,
            "limit": args.limit,
            "top_isigma": args.top_isigma,
            "row_index_file": args.row_index_file,
            "selected_rows": int(len(indices)),
            "valid_prediction_rows": int(np.count_nonzero(valid)),
        },
        "row_table": row_table_path,
        "timing": {"prediction_seconds": pred_subset["seconds"], "wall_seconds": wall_seconds},
        "diagnostics": {
            "ug_cache": getattr(port, "ug_cache_status", {"status": "disabled"}),
            "row_plan_cache": pred_subset.get("row_plan_cache_status", {"status": "off"}),
            "row_plan_cache_unique_diffs": int(pred_subset.get("row_plan_cache_unique_diffs", 0)),
            "basis_size_min": int(np.min(pred_subset["basis_size"])) if len(indices) else 0,
            "basis_size_median": float(np.median(pred_subset["basis_size"])) if len(indices) else 0.0,
            "basis_size_max": int(np.max(pred_subset["basis_size"])) if len(indices) else 0,
            "max_sg_solve_residual": float(np.max(pred_subset["sg_residual"])) if len(indices) else 0.0,
            "sg_window_mode": cfg.sg_window_mode,
            "sg_window_min_used": float(np.nanmin(pred_subset["sg_window"])) if len(indices) else float("nan"),
            "sg_window_median_used": float(np.nanmedian(pred_subset["sg_window"])) if len(indices) else float("nan"),
            "sg_window_max_used": float(np.nanmax(pred_subset["sg_window"])) if len(indices) else float("nan"),
            "sg_window_clipped_count": int(np.count_nonzero(pred_subset["sg_window_clipped"])) if len(indices) else 0,
            "sg_jitter_mode": cfg.sg_jitter_mode,
            "mosaicity_mode": cfg.mosaicity_mode,
            "mosaicity_sigma_deg": cfg.mosaicity_sigma_deg,
            "mosaicity_max_sigma_deg": cfg.mosaicity_max_sigma_deg,
            "mosaicity_points": cfg.mosaicity_points,
            "geometry_path": (
                None
                if port.geometry_path is None
                else {
                    "path": port.geometry_path["path"],
                    "sha256": port.geometry_path["sha256"],
                    "frame_center": port.geometry_path["frame_center"],
                    "frame_half_range": port.geometry_path["frame_half_range"],
                    "coefficients": {
                        name: values.tolist() for name, values in port.geometry_path["coefficients"].items()
                    },
                    "tilt_axes_lab": port.geometry_path["tilt_axes_lab"].tolist(),
                    "source_hkl_sha256": port.geometry_path["source_hkl_sha256"],
                    "bounds": port.geometry_path["bounds"],
                    "fit_source": port.geometry_path.get("fit_source"),
                    "validation": port.geometry_path.get("validation"),
                }
            ),
            "profile_quadrature": cfg.profile_quadrature,
            "profile_accumulation": cfg.profile_accumulation,
            "sg_jitter_sigma_median_used": float(np.nanmedian(pred_subset["sg_jitter_sigma"]))
            if len(indices)
            else float("nan"),
            "sg_jitter_sigma_max_used": float(np.nanmax(pred_subset["sg_jitter_sigma"]))
            if len(indices)
            else float("nan"),
            "sg_profile_points_median_total": float(np.median(pred_subset["sg_profile_points_total"]))
            if len(indices)
            else 0.0,
            "sg_profile_weight_sum_median": float(np.nanmedian(pred_subset["sg_profile_weight_sum"]))
            if len(indices)
            else 0.0,
            "sg_profile_same_branch_continuation_rows": int(
                np.count_nonzero(pred_subset["sg_profile_same_branch_continuation_used"])
            )
            if len(indices)
            else 0,
            "sg_profile_same_branch_continuation_max_segments": int(
                np.max(
                    pred_subset["sg_profile_same_branch_continuation_segments"][
                        pred_subset["sg_profile_same_branch_continuation_used"]
                    ]
                )
            )
            if len(indices) and np.any(pred_subset["sg_profile_same_branch_continuation_used"])
            else 1,
            "median_abs_dSg_dframe": float(np.nanmedian(np.abs(pred_subset["sg_dSg_dframe"])))
            if len(indices)
            else float("nan"),
            "median_dyn_over_born": float(np.nanmedian(full_dyn[valid] / np.clip(full_born[valid], 1e-30, None)))
            if np.any(valid)
            else float("nan"),
        },
        "metrics": eval_out,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, allow_nan=True))
    print(json.dumps(out["metrics"], indent=2, allow_nan=True))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
