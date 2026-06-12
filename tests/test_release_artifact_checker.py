from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = ROOT / "scripts" / "check_release_artifacts.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_release_artifacts", CHECKER_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_tiny_wheel(path: Path, *, include_required: bool = True) -> None:
    required = (
        "grond/__init__.py",
        "grond/cli.py",
        "grond/io/row_contract.py",
        "grond/legacy/engine/bloch_engine.py",
        "grond/schemas/grond-row-v1.schema.json",
        "grond-0.2.0.dist-info/METADATA",
        "grond-0.2.0.dist-info/entry_points.txt",
    )
    members = required if include_required else required[:-1]
    with zipfile.ZipFile(path, "w") as zf:
        for name in members:
            zf.writestr(name, "ok\n")


def write_tiny_sdist(path: Path, *, include_required: bool = True) -> None:
    required = (
        "docs/DEVELOPMENT.md",
        "docs/INSTALL.md",
        "docs/QUICKSTART.md",
        "scripts/check_release_artifacts.py",
        "tests/test_package_metadata.py",
        "tests/test_release_artifact_checker.py",
    )
    members = required if include_required else required[:-1]
    with tarfile.open(path, "w:gz") as tf:
        for name in members:
            data = b"ok\n"
            info = tarfile.TarInfo(f"grond-0.2.0/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_release_checker_requires_wheel_and_sdist(tmp_path: Path):
    checker = load_checker()
    write_tiny_wheel(tmp_path / "grond-0.2.0-py3-none-any.whl")

    _summaries, failures = checker.scan_artifacts(tmp_path)

    assert any("expects both a wheel and a source distribution" in failure for failure in failures)


def test_release_checker_accepts_complete_artifact_pair(tmp_path: Path):
    checker = load_checker()
    write_tiny_wheel(tmp_path / "grond-0.2.0-py3-none-any.whl")
    write_tiny_sdist(tmp_path / "grond-0.2.0.tar.gz")

    summaries, failures = checker.scan_artifacts(tmp_path)

    assert len(summaries) == 2
    assert failures == []


def test_release_checker_reports_missing_required_sdist_file(tmp_path: Path):
    checker = load_checker()
    write_tiny_wheel(tmp_path / "grond-0.2.0-py3-none-any.whl")
    write_tiny_sdist(tmp_path / "grond-0.2.0.tar.gz", include_required=False)

    _summaries, failures = checker.scan_artifacts(tmp_path)

    assert any("missing required source file tests/test_release_artifact_checker.py" in failure for failure in failures)


def test_release_checker_reports_missing_required_wheel_file(tmp_path: Path):
    checker = load_checker()
    write_tiny_wheel(tmp_path / "grond-0.2.0-py3-none-any.whl", include_required=False)
    write_tiny_sdist(tmp_path / "grond-0.2.0.tar.gz")

    _summaries, failures = checker.scan_artifacts(tmp_path)

    assert any("missing required wheel file dist-info/entry_points.txt" in failure for failure in failures)
