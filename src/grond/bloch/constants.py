"""Numerical names and fixed defaults shared by the Bloch implementations."""

from __future__ import annotations

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
