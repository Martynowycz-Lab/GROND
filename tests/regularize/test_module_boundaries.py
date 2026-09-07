"""Protect callable compatibility and cache identity across regularization moves."""
from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

from grond.legacy.engine import native_row_regularize
from grond.regularize.source import REGULARIZATION_SOURCE_PATHS, regularization_source_files


@pytest.mark.parametrize(
    "adapter,module,name",
    [
        ("native_row_regularize", "regularize.contracts", "Nuisance"),
        ("native_row_regularize", "regularize.evaluation", "evaluate_trial"),
        ("native_row_regularize", "regularize.row_output", "write_row_table"),
        ("native_row_regularize", "regularize.scales", "fit_group_scales"),
        ("native_row_regularize", "refine.shelx_input", "write_endpoint_ins"),
        ("native_structure_update", "refine.native_objective", "residual_vector"),
        ("native_multi_structure_update", "refine.datasets", "DatasetState"),
        ("shelxl_feedback_loop", "refine.feedback_contract", "read_json"),
    ],
)
def test_legacy_imports_return_the_canonical_callable(adapter, module, name):
    legacy = importlib.import_module("grond.legacy.engine." + adapter)
    owner = importlib.import_module("grond." + module)
    assert getattr(legacy, name) is getattr(owner, name)
    assert Path(inspect.getfile(getattr(owner, name))).resolve() in regularization_source_files()


def test_every_declared_regularization_helper_changes_prediction_fingerprint(monkeypatch):
    paths = regularization_source_files()
    assert len(paths) == len(set(paths)) == len(REGULARIZATION_SOURCE_PATHS)
    assert all(path.is_file() for path in paths)
    original_read = Path.read_bytes
    before = native_row_regularize._prediction_engine_source_fingerprint()
    for changed in paths:
        def altered_read(path, changed=changed):
            content = original_read(path)
            return content + b"\n# isolated source identity test\n" if path == changed else content

        with monkeypatch.context() as context:
            context.setattr(Path, "read_bytes", altered_read)
            assert native_row_regularize._prediction_engine_source_fingerprint() != before, str(changed)


@pytest.mark.parametrize(
    "filename",
    [
        "native_row_regularize.py",
        "scale_compression_polish.py",
        "combine_native_nuisance_runs.py",
        "continuous_thickness_polish.py",
        "shelxl_feedback_loop.py",
        "native_structure_update.py",
        "native_multi_structure_update.py",
    ],
)
def test_legacy_entrypoints_import_their_own_checkout_without_pythonpath(tmp_path, filename):
    source = Path(native_row_regularize.__file__).resolve().with_name(filename)
    result = subprocess.run(
        [sys.executable, str(source), "--help"],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1"),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--out-dir" in result.stdout
