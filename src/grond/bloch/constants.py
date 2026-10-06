"""Numerical names and fixed defaults shared by the Bloch implementations."""

from __future__ import annotations

JAX_BACKENDS = {"jax", "jax_auto", "jax_cpu", "jax_mps", "jax_cuda", "jax_gpu"}

JAX_PADDED_BACKENDS = {
    "jax_padded",
    "jax_auto_padded",
    "jax_cpu_padded",
    "jax_mps_padded",
    "jax_cuda_padded",
    "jax_gpu_padded",
}

JAX_REAL_EXPM_BACKENDS = {"jax_cpu_real_expm", "jax_mps_real_expm"}

JAX_REAL_EXPM_PADDED_BACKENDS = {"jax_mps_real_expm_padded"}

ALL_JAX_BACKENDS = JAX_BACKENDS | JAX_PADDED_BACKENDS | JAX_REAL_EXPM_BACKENDS | JAX_REAL_EXPM_PADDED_BACKENDS

COMPUTE_BACKENDS = [
    "scipy",
    "scipy_bethe",
    "scipy_expm",
    "scipy_expm_multiply",
    "jax",
    "jax_auto",
    "jax_cpu",
    "jax_mps",
    "jax_cuda",
    "jax_padded",
    "jax_auto_padded",
    "jax_cpu_padded",
    "jax_mps_padded",
    "jax_cuda_padded",
    "jax_cpu_real_expm",
    "jax_mps_real_expm",
    "jax_mps_real_expm_padded",
]

COMPUTE_BACKEND_HELP = ", ".join(COMPUTE_BACKENDS)

# Version 8 binds the parsed input snapshots and rejects source replacement.
ROW_PLAN_CACHE_SCHEMA = 8

UG_CACHE_SCHEMA = 4

# Only text-rounding tolerance (including six-decimal thirds), not the much
# wider SHELXL SPEC constraint-search radius. Shared by parsing and expansion.
SITE_SYMMETRY_TOLERANCE_FRACTIONAL = 2e-6

# Electron Compton wavelength in angstroms; used with de Broglie wavelength
# to calculate the Lorentz factor, without a fixed accelerating-voltage guess.
ELECTRON_COMPTON_WAVELENGTH_A = 0.024263102

DEFAULT_MAX_DENSE_UG_GIB = 2.0

# The stored complex128 cube needs 16 bytes/voxel, but vectorized construction
# also holds temporary arrays (measured near 170 bytes/voxel for anisotropic
# models). Reserve explicit headroom before allocation and decompression.
DENSE_UG_BUILD_BYTES_PER_VOXEL = 256

DENSE_UG_CACHE_LOAD_BYTES_PER_VOXEL = 64

DEFAULT_MAX_SUPPORT_SG = 0.01

BETHE_BACKENDS = {"scipy_bethe", "bethe", "scipy_bethe_promoted"}
