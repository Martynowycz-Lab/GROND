from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
LEGACY_ROOT = PACKAGE_ROOT / "legacy"
ENGINE_ROOT = LEGACY_ROOT / "engine"


def legacy_script(*parts: str) -> Path:
    """Return a path inside the vendored compatibility-script tree."""

    return LEGACY_ROOT.joinpath(*parts)


def engine_script(name: str) -> Path:
    """Return a path to a vendored command-line engine script."""

    return ENGINE_ROOT / name


# Backward-compatible internal alias used by the first CLI implementation.
port_script = engine_script
