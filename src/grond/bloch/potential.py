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
from grond.bloch.model import fcalc_res_grid, inv_d2_from_hkl_array, model_content_hash


class PotentialMixin:
    """One cohesive part of XDSBlochPort; initialized by the public facade."""

    def ug_cache_key(self):
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
        plan = dense_ug_memory_plan(self.h_lim, self.k_lim, self.l_lim)
        self.ug_allocation_status = enforce_dense_ug_memory_limit(plan, operation="build")
        h_vals = np.arange(-self.h_lim, self.h_lim + 1, dtype=float)
        k_vals = np.arange(-self.k_lim, self.k_lim + 1, dtype=float)
        l_vals = np.arange(-self.l_lim, self.l_lim + 1, dtype=float)
        hh, kk, ll = np.meshgrid(h_vals, k_vals, l_vals, indexing="ij")
        factor = float(self.gamma) / (math.pi * self.data.cell.volume)
        cube = factor * fcalc_res_grid(self.model, hh, kk, ll)
        return validate_ug_cube(cube, tuple(int(value) for value in plan["shape"]))

    def ug(self, hkl):
        h, k, l = (int(v) for v in hkl)
        i, j, m = h + self.h_lim, k + self.k_lim, l + self.l_lim
        if i < 0 or j < 0 or m < 0 or i > 2 * self.h_lim or j > 2 * self.k_lim or m > 2 * self.l_lim:
            return 0.0 + 0.0j
        if self.ug_cube is None:
            factor = float(self.gamma) / (math.pi * self.data.cell.volume)
            value = fcalc_res_grid(
                self.model,
                np.asarray([h], dtype=float),
                np.asarray([k], dtype=float),
                np.asarray([l], dtype=float),
            )
            return complex(factor * np.asarray(value).reshape(-1)[0])
        return self.ug_cube[i, j, m]

    def ug_array(self, hkls):
        hkls = np.asarray(hkls, dtype=int)
        shape = hkls.shape[:-1]
        flat = hkls.reshape(-1, 3)
        i = flat[:, 0] + self.h_lim
        j = flat[:, 1] + self.k_lim
        m = flat[:, 2] + self.l_lim
        ok = (i >= 0) & (j >= 0) & (m >= 0) & (i <= 2 * self.h_lim) & (j <= 2 * self.k_lim) & (m <= 2 * self.l_lim)
        out = np.zeros(len(flat), dtype=np.complex128)
        if self.ug_cube is None:
            if np.any(ok):
                factor = float(self.gamma) / (math.pi * self.data.cell.volume)
                out[ok] = factor * fcalc_res_grid(
                    self.model,
                    flat[ok, 0],
                    flat[ok, 1],
                    flat[ok, 2],
                )
        else:
            out[ok] = self.ug_cube[i[ok], j[ok], m[ok]]
        return out.reshape(shape)

    def inv_d2_for_hkls(self, hkls):
        hkls = np.asarray(hkls, dtype=float)
        if hkls.ndim == 1:
            g = self.data.A @ hkls
            return float(np.dot(g, g))
        g = (self.data.A @ hkls.T).T
        return np.sum(g * g, axis=1)

    def row_delta_b(self, row_idx):
        mode = str(getattr(self.cfg, "dose_b_mode", "none") or "none").lower()
        if mode == "none" or abs(float(getattr(self.cfg, "dose_b_per_frame", 0.0))) <= 0.0:
            return 0.0
        if mode != "linear_frame":
            raise ValueError(f"unknown dose_b_mode={self.cfg.dose_b_mode!r}")
        dz = max(0.0, float(self.data.zcal[int(row_idx)]) - self.dose_ref_frame)
        return float(getattr(self.cfg, "dose_b_per_frame", 0.0)) * dz

    def dose_attenuation(self, hkl, row_idx):
        delta_b = self.row_delta_b(row_idx)
        if delta_b == 0.0:
            return 1.0
        inv_d2 = self.inv_d2_for_hkls(hkl)
        return float(math.exp(-0.25 * delta_b * inv_d2))

    def dose_attenuation_for_inv_d2(self, inv_d2, row_idx):
        delta_b = self.row_delta_b(row_idx)
        if delta_b == 0.0:
            return np.ones_like(inv_d2, dtype=float)
        return np.exp(-0.25 * delta_b * np.asarray(inv_d2, dtype=float))

    def ug_for_row(self, hkl, row_idx):
        return self.ug(hkl) * self.dose_attenuation(hkl, row_idx)

    def ug_for_row_array(self, hkls, row_idx):
        vals = self.ug_array(hkls)
        delta_b = self.row_delta_b(row_idx)
        if delta_b == 0.0:
            return vals
        inv_d2 = self.inv_d2_for_hkls(np.asarray(hkls, dtype=int).reshape(-1, 3)).reshape(vals.shape)
        return vals * np.exp(-0.25 * delta_b * inv_d2)

    def build_pool(self):
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
        return np.concatenate(hkls, axis=0), np.concatenate(uvals)

    def thickness_grid(self):
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

    def make_groups(self):
        if getattr(self.data, "pets_exact_a_center", None) is not None:
            return np.rint(np.asarray(self.data.zcal, dtype=float)).astype(int) - 1
        omegas = np.array([self.omega_from_z(z) for z in self.data.zcal])
        return ((omegas - np.min(omegas)) // float(self.cfg.group_width_deg)).astype(int)
