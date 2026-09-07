"""Behavioral boundaries for the focused Bloch modules and legacy adapters."""

from __future__ import annotations

import importlib
import inspect
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from grond.bloch.source import PHYSICS_SOURCE_PATHS, physics_source_files
from grond.legacy.engine import bloch_engine, build_dynrows_xds


@pytest.mark.parametrize(
    "module,names",
    [
        (
            "grond.bloch.model",
            ("parse_model", "parse_res", "parse_cif", "fcalc_res", "atom_dw", "alternate_hand_model"),
        ),
        ("grond.bloch.config", ("Config", "frozen_kernel_contract")),
        ("grond.bloch.backends", ("jax_bloch_available", "promoted_bethe_effective_hamiltonian")),
        ("grond.bloch.cache", ("row_plan_cache_validation_errors", "load_validated_ug_cache")),
        ("grond.bloch.splits", ("split_masks", "build_transform_split_contract")),
        ("grond.bloch.errors", ("LocalSgCrossingError", "NonPassiveAbsorptionError")),
        ("grond.io.xds", ("XDSData", "rotation_matrix_from_header")),
    ],
)
def test_legacy_engine_exports_are_the_canonical_objects(module, names):
    implementation = importlib.import_module(module)
    for name in names:
        assert getattr(bloch_engine, name) is getattr(implementation, name)


@pytest.mark.parametrize(
    "module,names",
    [
        ("grond.io.raw_detector", ("parse_xparm", "xds_detector_origin_and_beam_center")),
        ("grond.io.raw_pixels", ("FrameCache", "integrate_observation")),
        ("grond.io.raw_geometry", ("fit_geometry_calibration", "sg_for_hkl")),
        ("grond.io.raw_contracts", ("observation_record", "write_xds_like")),
    ],
)
def test_legacy_raw_adapter_exports_are_the_canonical_objects(module, names):
    implementation = importlib.import_module(module)
    for name in names:
        assert getattr(build_dynrows_xds, name) is getattr(implementation, name)


def test_source_manifest_covers_every_forward_model_method_owner():
    paths = physics_source_files()
    assert len(paths) == len(set(paths)) == len(PHYSICS_SOURCE_PATHS)
    assert all(path.is_file() for path in paths)
    package = Path(bloch_engine.__file__).resolve().parents[2]
    assert [path.relative_to(package).as_posix() for path in paths] == list(PHYSICS_SOURCE_PATHS)
    for cls in bloch_engine.XDSBlochPort.__mro__:
        if cls is object:
            continue
        assert Path(inspect.getfile(cls)).resolve() in paths
        for member in cls.__dict__.values():
            if isinstance(member, staticmethod):
                member = member.__func__
            if inspect.isfunction(member):
                assert Path(inspect.getfile(member)).resolve() in paths


def test_mixin_responsibilities_do_not_shadow_each_other():
    owners = {}
    for cls in bloch_engine.XDSBlochPort.__mro__[1:]:
        if cls is object:
            continue
        for name, member in cls.__dict__.items():
            if inspect.isfunction(member) or isinstance(member, staticmethod):
                assert name not in owners, f"{name} appears in both {owners.get(name)} and {cls.__name__}"
                owners[name] = cls.__name__
    assert owners["row_prediction_plan"] == "RowPlanMixin"
    assert owners["sg_for_hkls"] == "GeometryMixin"
    assert owners["hamiltonian_static_part"] == "HamiltonianMixin"


def test_configuration_pickle_resolves_through_legacy_import_surface():
    cfg = bloch_engine.Config(n_basis=37, n_points=11)
    restored = pickle.loads(pickle.dumps(cfg))
    assert isinstance(restored, bloch_engine.Config)
    assert restored.n_basis == 37
    assert restored.n_points == 11


@pytest.mark.parametrize("filename", ["bloch_engine.py", "build_dynrows_xds.py"])
def test_standalone_legacy_entrypoints_bootstrap_the_source_package(tmp_path, filename):
    root = Path(__file__).resolve().parents[2]
    path = root / "src" / "grond" / "legacy" / "engine" / filename
    env = dict(os.environ, PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    expected_flag = "--hkl" if filename == "bloch_engine.py" else "--integrate-hkl"
    assert expected_flag in result.stdout


def test_prediction_fingerprint_tracks_transitive_sign_and_row_contracts(monkeypatch):
    from grond.legacy.engine import native_row_regularize

    original_read = Path.read_bytes
    baseline = native_row_regularize._prediction_engine_source_fingerprint()
    package = Path(bloch_engine.__file__).resolve().parents[2]
    for relative in ("hkl_transform.py", "io/row_contract.py", "legacy/engine/dynrow_contract.py"):
        changed = package / relative

        def altered_read(path, changed=changed):
            content = original_read(path)
            return content + b"\n# isolated fingerprint test\n" if path == changed else content

        with monkeypatch.context() as context:
            context.setattr(Path, "read_bytes", altered_read)
            assert native_row_regularize._prediction_engine_source_fingerprint() != baseline
