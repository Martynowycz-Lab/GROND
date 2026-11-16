"""--prediction-cache-dir: a relative path means the directory grond was started in, not the package directory the
stages run in."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from grond.io.config import DatasetSpec
from grond.paths import PACKAGE_ROOT
from grond.workflows.nuisance_policy import prediction_cache_dir

DS = DatasetSpec(name="c1", hkl="INTEGRATE.HKL", xds_dir=".")


def _args(cache: str, out_dir: Path, scope: str = "dataset") -> argparse.Namespace:
    return argparse.Namespace(prediction_cache_dir=cache, out_dir=str(out_dir), prediction_cache_scope=scope)


def test_relative_cache_dir_is_taken_from_the_launch_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = prediction_cache_dir(_args("relcache", tmp_path / "out"), DS, "screen")
    assert path == Path(os.getcwd()) / "relcache" / "native_predictions" / "c1"
    assert path.is_absolute()
    assert PACKAGE_ROOT not in path.parents
    staged = prediction_cache_dir(_args("relcache", tmp_path / "out", scope="stage"), DS, "screen")
    assert staged == path.parent.parent / "stage_predictions" / "screen" / "c1"


def test_absolute_and_default_cache_dirs_are_unchanged(tmp_path):
    given = tmp_path / "cache"
    assert prediction_cache_dir(_args(str(given), tmp_path / "out"), DS, "screen") == given / "native_predictions" / "c1"
    out = tmp_path / "out"
    assert prediction_cache_dir(_args("", out), DS, "screen") == out / "cache" / "native_predictions" / "c1"
