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
    "legacy/engine/dynrow_contract.py",
    "paths.py",
    "legacy/engine/bloch_engine.py",
    "legacy/engine/build_dynrows_xds.py",
    "bloch/source.py",
    "bloch/constants.py",
    "bloch/errors.py",
    "bloch/model.py",
    "bloch/numerics.py",
    "bloch/config.py",
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
    "io/raw_detector.py",
    "io/raw_pixels.py",
    "io/raw_geometry.py",
    "io/raw_contracts.py",
)


def physics_source_files() -> tuple[Path, ...]:
    package_root = Path(__file__).resolve().parents[1]
    return tuple(package_root / relative for relative in PHYSICS_SOURCE_PATHS)
