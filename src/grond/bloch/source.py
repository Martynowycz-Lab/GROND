"""Explicit source manifest for invalidating predictions after physics edits.

Paths are package-relative so the caller can hash identical code consistently
on a workstation and a compute server. No module execution is needed to discover
implementation files; extend this list whenever a numerical helper is added.
"""

from pathlib import Path

PHYSICS_SOURCE_PATHS = (
    # Imported sign/row contracts are part of physics even when unchanged here.
    "hkl_transform.py",
    "io/row_contract.py",
    "paths.py",
    "legacy/engine/bloch_engine.py",
    "bloch/source.py",
    "bloch/constants.py",
    "bloch/errors.py",
    "bloch/model.py",
    "bloch/numerics.py",
    "bloch/config.py",
    "bloch/thickness.py",
    "bloch/backends.py",
    "bloch/cache.py",
    "io/xds.py",
    "bloch/splits.py",
    "bloch/metrics.py",
    "bloch/potential.py",
    "bloch/geometry.py",
    "bloch/profiles.py",
    "bloch/basis.py",
    "bloch/hamiltonian.py",
    "bloch/row_plans.py",
    "bloch/row_cache.py",
    "bloch/prediction.py",
)


def physics_source_files() -> tuple[Path, ...]:
    """Return absolute paths of PHYSICS_SOURCE_PATHS inside the imported grond package, in manifest order."""

    package_root = Path(__file__).resolve().parents[1]
    return tuple(package_root / relative for relative in PHYSICS_SOURCE_PATHS)
