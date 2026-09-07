from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests._support import REPO_ROOT

SRC_ROOT = REPO_ROOT / "src"


def test_every_legacy_engine_module_imports_as_part_of_the_package(tmp_path: Path) -> None:
    """Catch bare sibling imports that only work from the source directory."""

    script = r"""
import importlib
import pkgutil
import sys

import grond.legacy.engine as engine

initial_path = tuple(sys.path)
module_names = sorted(info.name for info in pkgutil.iter_modules(engine.__path__))
for module_name in module_names:
    importlib.import_module(f"{engine.__name__}.{module_name}")

if tuple(sys.path) != initial_path:
    raise AssertionError("importing legacy engine modules changed sys.path")

bare_siblings = sorted(name for name in module_names if name in sys.modules)
if bare_siblings:
    raise AssertionError(f"legacy modules were imported under bare names: {bare_siblings}")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
