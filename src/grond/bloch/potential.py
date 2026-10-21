# ruff: noqa: E741
# h, k, l retain standard crystallographic Miller-index notation.
"""Model-dependent Fourier-potential lookup, beam pool and thickness distribution."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss

from grond.bloch.cache import (
    dense_ug_memory_plan,
    enforce_dense_ug_memory_limit,
    load_validated_ug_cache,
    validate_ug_cube,
    write_ug_cache_atomic,
)
from grond.bloch.constants import UG_CACHE_SCHEMA
from grond.bloch.model import fcalc_res_grid, inv_d2_from_hkl_array, lattice_translations_from_latt, model_content_hash
from grond.bloch.thickness import (
    config_cf_parameters,
    config_table_nodes,
    gamma_cf_born,
    resolved_config_evaluator,
    uniform_grid_spec,
)


def centring_coset_mask(hkls, latt) -> np.ndarray:
    """True for the HKLs in the lattice-centring coset of the direct beam: g . t is an integer for every centring
    translation t of SHELX LATT code ``latt`` (all True for a primitive lattice)."""

    hkls = np.asarray(hkls, dtype=float).reshape(-1, 3)
    keep = np.ones(len(hkls), dtype=bool)
    for translation in lattice_translations_from_latt(abs(int(latt))):
        phase = hkls @ np.asarray(translation, dtype=float)
        keep &= np.abs(phase - np.round(phase)) < 1e-9
    return keep


_UG_BUILD: dict = {}


def _ug_build_workers() -> int:
    """Processes for the dense U_g cube build: GROND_UG_BUILD_WORKERS (default 1, serial); fork platforms only."""
    import multiprocessing as mp
    import os

    value = int(os.environ.get("GROND_UG_BUILD_WORKERS", "1") or 1)
    return max(1, value) if "fork" in mp.get_all_start_methods() else 1


def _ug_slab(h_vals):
    """fcalc_res_grid on one slab of h values (the U_g cube builder's worker)."""
    hh, kk, ll = np.meshgrid(h_vals, _UG_BUILD["k"], _UG_BUILD["l"], indexing="ij")
    return fcalc_res_grid(_UG_BUILD["model"], hh, kk, ll)


class PotentialMixin:
    """One cohesive part of XDSBlochPort; initialized by the public facade."""

    def ug_cache_key(self):
        """Return (16-hex key, payload) naming the dense U_g cube and every input that changes it.

        The payload holds the cache schema, .res and model hashes, cell, HKL limits, gamma and ug_d_min.
        """
        cell = self.model["cell"]
        payload = {
            "cache_schema": UG_CACHE_SCHEMA,
            "res_sha256": self.res_hash,
            "model_content_sha256": model_content_hash(self.model),
            "cell": [round(float(v), 6) for v in cell],
            "lims": [int(self.h_lim), int(self.k_lim), int(self.l_lim)],
            "gamma": float(self.gamma),
            "gamma_source": "wavelength" if self.cfg.gamma is None else "explicit_config",
            "ug_d_min_A": float(self.cfg.ug_d_min),
            "formula": "gamma*Fe/(pi*V); explicit electron SFAC; parsed model symmetry",
        }
        text = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:16], payload

    def load_or_build_ug_cube(self):
        """Return the dense U_g cube, from a validated cache file when possible, else built and written to the cache.

        Sets ``ug_cache_status`` to disabled, hit, built or rebuilt; a rebuild records why the existing file was
        rejected.
        """
        if self.cache_dir is None:
            cube = self.build_ug_cube()
            self.ug_cache_status = {
                "status": "disabled",
                "allocation": getattr(self, "ug_allocation_status", {}),
            }
            return cube
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        key, meta = self.ug_cache_key()
        path = self.cache_dir / f"ug_cube_{key}.npz"
        lims = (int(self.h_lim), int(self.k_lim), int(self.l_lim))
        plan = dense_ug_memory_plan(*lims)
        cache_meta = {**meta, "key": key}
        rebuild_reason = ""
        if path.exists():
            self.ug_allocation_status = enforce_dense_ug_memory_limit(plan, operation="cache load")
            try:
                cube = load_validated_ug_cache(
                    path,
                    expected_lims=lims,
                    expected_meta=cache_meta,
                )
            except Exception as exc:
                rebuild_reason = f"{type(exc).__name__}: {str(exc)[:240]}"
            else:
                self.ug_cache_status = {
                    "status": "hit",
                    "path": str(path),
                    "key": key,
                    "meta": meta,
                    "allocation": self.ug_allocation_status,
                }
                return cube
        cube = self.build_ug_cube()
        write_ug_cache_atomic(path, cube=cube, lims=lims, meta=cache_meta)
        self.ug_cache_status = {
            "status": "rebuilt" if rebuild_reason else "built",
            "path": str(path),
            "key": key,
            "meta": meta,
            "allocation": getattr(self, "ug_allocation_status", {}),
        }
        if rebuild_reason:
            self.ug_cache_status["rebuild_reason"] = rebuild_reason
        return cube

    def build_ug_cube(self):
        """Build the dense U_g cube for h, k, l in [-lim, lim] (complex128) after the memory-limit check.

        U_g = gamma * F(g) / (pi * V), with gamma the relativistic factor, F the model's electron structure factor
        (fcalc_res_grid) and V the cell volume in Å³; index [h + h_lim, k + k_lim, l + l_lim].
        """
        plan = dense_ug_memory_plan(self.h_lim, self.k_lim, self.l_lim)
        self.ug_allocation_status = enforce_dense_ug_memory_limit(plan, operation="build")
        h_vals = np.arange(-self.h_lim, self.h_lim + 1, dtype=float)
        k_vals = np.arange(-self.k_lim, self.k_lim + 1, dtype=float)
        l_vals = np.arange(-self.l_lim, self.l_lim + 1, dtype=float)
        factor = float(self.gamma) / (math.pi * self.data.cell.volume)
        workers = _ug_build_workers()
        if workers > 1 and len(h_vals) > 1:
            # Same values as the serial build: fcalc_res_grid is elementwise in (h, k, l), split here over h slabs.
            import multiprocessing as mp

            slabs = [h_vals[i::workers] for i in range(workers)]
            _UG_BUILD.update(model=self.model, k=k_vals, l=l_vals)
            try:
                with mp.get_context("fork").Pool(min(workers, len(slabs))) as pool:
                    parts = pool.map(_ug_slab, slabs)
            finally:
                _UG_BUILD.clear()
            cube = np.empty((len(h_vals), len(k_vals), len(l_vals)), dtype=np.complex128)
            for i, part in enumerate(parts):
                cube[i::workers] = part
            cube *= factor
        else:
            hh, kk, ll = np.meshgrid(h_vals, k_vals, l_vals, indexing="ij")
            cube = factor * fcalc_res_grid(self.model, hh, kk, ll)
        return validate_ug_cube(cube, tuple(int(value) for value in plan["shape"]))

    def calculate_ug_array(self, hkls):
        """Evaluate each distinct integer HKL once, restoring the requested shape and order.

        A dense Hamiltonian repeats many pair differences. This is an exact
        lookup within one call; no coefficients persist across model changes.
        """

        hkls = np.asarray(hkls, dtype=int)
        shape = hkls.shape[:-1]
        flat = hkls.reshape(-1, 3)
        if not len(flat):
            return np.empty(shape, dtype=np.complex128)
        unique_hkls, inverse = np.unique(flat, axis=0, return_inverse=True)
        factor = float(self.gamma) / (math.pi * self.data.cell.volume)
        values = factor * fcalc_res_grid(self.model, unique_hkls[:, 0], unique_hkls[:, 1], unique_hkls[:, 2])
        return np.asarray(values, dtype=np.complex128)[inverse.reshape(-1)].reshape(shape)

    def ug(self, hkl):
        """Return U_g of one HKL, from the dense cube inside its limits, else computed directly from the model."""
        h, k, l = (int(v) for v in hkl)
        i, j, m = h + self.h_lim, k + self.k_lim, l + self.l_lim
        if self.ug_cube is not None and (
            0 <= i <= 2 * self.h_lim and 0 <= j <= 2 * self.k_lim and 0 <= m <= 2 * self.l_lim
        ):
            return self.ug_cube[i, j, m]
        return complex(self.calculate_ug_array(np.asarray([[h, k, l]], dtype=int))[0])

    def ug_array(self, hkls):
        """Return U_g for HKLs along the last axis of ``hkls``: cube values where covered, else computed directly."""
        hkls = np.asarray(hkls, dtype=int)
        if self.ug_cube is None:
            return self.calculate_ug_array(hkls)
        shape = hkls.shape[:-1]
        flat = hkls.reshape(-1, 3)
        i = flat[:, 0] + self.h_lim
        j = flat[:, 1] + self.k_lim
        m = flat[:, 2] + self.l_lim
        ok = (i >= 0) & (j >= 0) & (m >= 0) & (i <= 2 * self.h_lim) & (j <= 2 * self.k_lim) & (m <= 2 * self.l_lim)
        out = np.empty(len(flat), dtype=np.complex128)
        out[ok] = self.ug_cube[i[ok], j[ok], m[ok]]
        # The cache boundary is a storage limit. It must not zero a real coupling.
        if np.any(~ok):
            out[~ok] = self.calculate_ug_array(flat[~ok])
        return out.reshape(shape)

    def inv_d2_for_hkls(self, hkls):
        """Return 1/d^2 = |A h|^2 (Å⁻²) for one HKL (float) or HKL rows (array), A the orientation matrix."""
        hkls = np.asarray(hkls, dtype=float)
        if hkls.ndim == 1:
            g = self.data.A @ hkls
            return float(np.dot(g, g))
        g = (self.data.A @ hkls.T).T
        return np.sum(g * g, axis=1)

    def row_delta_b(self, row_idx):
        """Return a row's dose-damage B increase (Å²): dose_b_per_frame times frames past the reference frame."""
        mode = str(getattr(self.cfg, "dose_b_mode", "none") or "none").lower()
        if mode == "none" or abs(float(getattr(self.cfg, "dose_b_per_frame", 0.0))) <= 0.0:
            return 0.0
        if mode != "linear_frame":
            raise ValueError(f"unknown dose_b_mode={self.cfg.dose_b_mode!r}")
        dz = max(0.0, float(self.data.zcal[int(row_idx)]) - self.dose_ref_frame)
        return float(getattr(self.cfg, "dose_b_per_frame", 0.0)) * dz


    def dose_attenuation_for_inv_d2(self, inv_d2, row_idx):
        """Return exp(-DeltaB * inv_d2 / 4) elementwise for precomputed 1/d^2 values (Å⁻²) and one row."""
        delta_b = self.row_delta_b(row_idx)
        if delta_b == 0.0:
            return np.ones_like(inv_d2, dtype=float)
        return np.exp(-0.25 * delta_b * np.asarray(inv_d2, dtype=float))


    def ug_for_row_array(self, hkls, row_idx):
        """Return ug_array(hkls) with the row's dose attenuation exp(-DeltaB/(4 d^2)) applied."""
        vals = self.ug_array(hkls)
        delta_b = self.row_delta_b(row_idx)
        if delta_b == 0.0:
            return vals
        inv_d2 = self.inv_d2_for_hkls(np.asarray(hkls, dtype=int).reshape(-1, 3)).reshape(vals.shape)
        return vals * np.exp(-0.25 * delta_b * inv_d2)

    def build_pool(self):
        """Return the top-up beam pool: every nonzero HKL with d >= topup_d_min (Å) and its complex128 U_g."""
        h_lim, k_lim, l_lim = self.data.cell.get_hkl_limits(float(self.cfg.topup_d_min))
        k_grid, l_grid = np.meshgrid(
            np.arange(-k_lim, k_lim + 1, dtype=int),
            np.arange(-l_lim, l_lim + 1, dtype=int),
            indexing="ij",
        )
        hkls = []
        uvals = []
        inv_d2_limit = 1.0 / (float(self.cfg.topup_d_min) ** 2)
        for h in range(-h_lim, h_lim + 1):
            h_grid = np.full_like(k_grid, h)
            inv_d2 = inv_d2_from_hkl_array(h_grid, k_grid, l_grid, self.model["cell"])
            keep = (inv_d2 > 0.0) & (inv_d2 <= inv_d2_limit * (1.0 + 1e-12))
            if not np.any(keep):
                continue
            block = np.column_stack((h_grid[keep], k_grid[keep], l_grid[keep])).astype(int)
            hkls.append(block)
            uvals.append(np.asarray(self.ug_array(block), dtype=np.complex128))
        if not hkls:
            return np.zeros((0, 3), dtype=int), np.zeros(0, dtype=np.complex128)
        pool, ug = np.concatenate(hkls, axis=0), np.concatenate(uvals)
        if bool(getattr(self.cfg, "basis_centring_coset", False)):
            keep = centring_coset_mask(pool, self.model.get("shelx_latt", 1))
            pool, ug = pool[keep], ug[keep]
        return pool, ug

    def thickness_grid(self):
        """Return thickness nodes (Å) and weights for the configured thickness model and evaluator.

        ``table`` gives the support-table nodes and ``cf`` gives (None, None), since the gamma average is then closed
        form. The legacy quadrature uses Gauss-Legendre nodes (uniform_gradient), Gauss-Hermite nodes (gaussian) or
        their product (gaussian_gradient), drops nonpositive thicknesses and renormalises the weights to sum 1.
        """
        evaluator = resolved_config_evaluator(self.cfg)
        if evaluator == "table":
            # Exact hat-function weights of the support-renormalised distribution (grond.bloch.thickness).
            return config_table_nodes(self.cfg)
        if evaluator == "cf":
            # No nodes: every propagation and Born average uses the closed-form gamma average instead.
            return None, None
        model = str(getattr(self.cfg, "thickness_model", "uniform_gradient") or "uniform_gradient").lower()
        n = max(1, int(self.cfg.thickness_points))
        t0 = float(self.cfg.thickness_A)
        if model == "single" or n <= 1:
            return np.array([t0], dtype=float), np.array([1.0], dtype=float)
        if model == "uniform_gradient":
            if self.cfg.thickness_grad_frac <= 0.0:
                return np.array([t0], dtype=float), np.array([1.0], dtype=float)
            x, w = leggauss(n)
            t = t0 * (1.0 + float(self.cfg.thickness_grad_frac) * x)
        elif model == "gaussian":
            sigma = abs(float(getattr(self.cfg, "thickness_sigma_frac", 0.0))) * t0
            if sigma <= 0.0:
                return np.array([t0], dtype=float), np.array([1.0], dtype=float)
            x, w = hermgauss(n)
            t = t0 + math.sqrt(2.0) * sigma * x
            w = w / math.sqrt(math.pi)
        elif model == "gaussian_gradient":
            sigma = abs(float(getattr(self.cfg, "thickness_sigma_frac", 0.0))) * t0
            grad = float(self.cfg.thickness_grad_frac)
            if sigma <= 0.0 and grad <= 0.0:
                return np.array([t0], dtype=float), np.array([1.0], dtype=float)
            xg, wg = leggauss(n)
            xh, wh = hermgauss(n)
            centers = t0 * (1.0 + grad * xg)
            wh = wh / math.sqrt(math.pi)
            t = (centers[:, None] + math.sqrt(2.0) * sigma * xh[None, :]).reshape(-1)
            w = (wg[:, None] * wh[None, :]).reshape(-1)
        else:
            raise ValueError(f"unknown thickness_model={self.cfg.thickness_model!r}")
        ok = np.asarray(t, dtype=float) > 0.0
        if not np.any(ok):
            return np.array([self.cfg.thickness_A], dtype=float), np.array([1.0], dtype=float)
        t = np.asarray(t, dtype=float)[ok]
        w = np.asarray(w, dtype=float)[ok]
        return t, w / np.sum(w)

    def thickness_cf_parameters(self):
        """Gamma (shape, theta) when the thickness evaluator resolves to cf, else None (resolved once per port)."""
        try:
            return self._thickness_cf
        except AttributeError:
            self._thickness_cf = config_cf_parameters(self.cfg)
            return self._thickness_cf

    def thickness_uniform_grid(self):
        """(t0, dt, n) of the table evaluator's uniform support nodes, else None (legacy quadrature, cf)."""
        try:
            return self._thickness_uniform
        except AttributeError:
            self._thickness_uniform = (
                uniform_grid_spec(self.thick) if resolved_config_evaluator(self.cfg) == "table" else None
            )
            return self._thickness_uniform

    def born_thickness_average(self, h10_born, delta):
        """Thickness-averaged Born row term (2*pi)^2 |H10|^2 t^2 sinc^2(delta*t) for one profile point."""
        cf = self.thickness_cf_parameters()
        if cf is not None:
            return float(gamma_cf_born(abs(h10_born) ** 2, float(delta), cf["shape"], cf["theta"]))
        sinc = np.sinc(delta * self.thick)
        born_t = (2.0 * math.pi) ** 2 * (abs(h10_born) ** 2) * (self.thick**2) * (sinc**2)
        return float(np.sum(self.thick_w * born_t))

    def make_groups(self):
        """Return a scale-group index per row: its omega bin.

        Omega bins are group_width_deg degrees wide, counted from the lowest omega.
        """
        omegas = np.array([self.omega_from_z(z) for z in self.data.zcal])
        return ((omegas - np.min(omegas)) // float(self.cfg.group_width_deg)).astype(int)
