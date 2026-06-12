from __future__ import annotations

import os
import sys
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


def runtime_cache_dir(*parts: str) -> Path:
    """Return a writable cache directory for generated GROND intermediates.

    Installed package directories may be read-only, so generated Bloch caches
    must live in a user/runtime location unless a command explicitly passes a
    run-local cache path. `GROND_CACHE_DIR` is the portable override for shared
    servers and scratch filesystems.
    """

    override = os.environ.get("GROND_CACHE_DIR", "").strip()
    if override:
        root = Path(override).expanduser()
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches" / "grond"
    elif os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        root = (Path(local_appdata).expanduser() if local_appdata else Path.home() / "AppData" / "Local") / "GROND" / "Cache"
    else:
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
        root = (Path(xdg_cache_home).expanduser() if xdg_cache_home else Path.home() / ".cache") / "grond"
    return root.joinpath(*parts)


# Backward-compatible internal alias used by the first CLI implementation.
port_script = engine_script
