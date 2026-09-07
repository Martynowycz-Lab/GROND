from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

from tests._support import REPO_ROOT

ROOT = REPO_ROOT
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


def write_tiny_sdist(
    path: Path,
    *,
    include_required: bool = True,
    extra_members: dict[str, str] | None = None,
) -> None:
    required = (
        "ACKNOWLEDGMENTS.md",
        "DATA_POLICY.md",
        "RELEASE_CHECKLIST.md",
        "SECURITY.md",
        "docs/DEVELOPMENT.md",
        "docs/INSTALL.md",
        "docs/QUICKSTART.md",
        "docs/REFERENCES.md",
        "scripts/check_release_artifacts.py",
        "scripts/smoke_install_wheel.py",
        "tests/package/test_package_metadata.py",
        "tests/package/test_release_artifact_checker.py",
    )
    members = list(required if include_required else required[:-1])
    extra_members = extra_members or {}
    members.extend(extra_members)
    with tarfile.open(path, "w:gz") as tf:
        for name in members:
            data = extra_members.get(name, "ok\n").encode()
            info = tarfile.TarInfo(f"grond-0.2.0/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def add_wheel_member(path: Path, name: str, text: str = "ok\n") -> None:
    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr(name, text)


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

    assert any("missing required source file tests/package/test_release_artifact_checker.py" in failure for failure in failures)


def test_release_checker_reports_missing_required_wheel_file(tmp_path: Path):
    checker = load_checker()
    write_tiny_wheel(tmp_path / "grond-0.2.0-py3-none-any.whl", include_required=False)
    write_tiny_sdist(tmp_path / "grond-0.2.0.tar.gz")

    _summaries, failures = checker.scan_artifacts(tmp_path)

    assert any("missing required wheel file dist-info/entry_points.txt" in failure for failure in failures)


def test_release_checker_rejects_non_python_package_artifact_names(tmp_path: Path):
    checker = load_checker()
    wheel = tmp_path / "grond-0.2.0-py3-none-any.whl"
    write_tiny_wheel(wheel)
    add_wheel_member(wheel, "grond/package.json")
    write_tiny_sdist(tmp_path / "grond-0.2.0.tar.gz")

    _summaries, failures = checker.scan_artifacts(tmp_path)

    assert any("package.json" in failure for failure in failures)


def test_release_checker_rejects_non_python_package_text(tmp_path: Path):
    checker = load_checker()
    wheel = tmp_path / "grond-0.2.0-py3-none-any.whl"
    write_tiny_wheel(wheel)
    forbidden_text = "Run " + "n" + "pm install before using GROND.\n"
    add_wheel_member(wheel, "grond/bad_install.txt", forbidden_text)
    write_tiny_sdist(tmp_path / "grond-0.2.0.tar.gz")

    _summaries, failures = checker.scan_artifacts(tmp_path)

    assert any("n" + "pm install" in failure for failure in failures)


def test_release_checker_rejects_repository_boundary_names(tmp_path: Path):
    checker = load_checker()
    wheel = tmp_path / "grond-0.2.0-py3-none-any.whl"
    write_tiny_wheel(wheel)
    forbidden_members = (
        "exper" + "iments/probe.py",
        "Open" + "DM/solver.py",
        "open_direct_" + "methods/solver.py",
        "pa" + "per/draft.tex",
        "manu" + "script/draft.md",
        "raw_" + "data/frame_0001.cbf",
        "cache/predictions.bin",
        "outputs/run_summary.json",
        "fixtures/frame_0002.cbf",
        "fixtures/integrated.refl",
    )
    write_tiny_sdist(
        tmp_path / "grond-0.2.0.tar.gz",
        extra_members={name: "not public\n" for name in forbidden_members},
    )

    _summaries, failures = checker.scan_artifacts(tmp_path)

    for member in forbidden_members:
        assert any(member.casefold() in failure.casefold() for failure in failures), member


def test_release_checker_rejects_repository_boundary_text(tmp_path: Path):
    checker = load_checker()
    wheel = tmp_path / "grond-0.2.0-py3-none-any.whl"
    write_tiny_wheel(wheel)
    forbidden_texts = (
        "See " + "exper" + "iments/probe.py for the private run.\n",
        "Load the " + "Open" + "DM checkout.\n",
        "Import open_direct_" + "methods here.\n",
        "Read pa" + "per/draft.tex before release.\n",
        "Attach the manu" + "script text.\n",
    )
    for index, forbidden_text in enumerate(forbidden_texts):
        add_wheel_member(wheel, f"grond/boundary_text_{index}.txt", forbidden_text)
    write_tiny_sdist(tmp_path / "grond-0.2.0.tar.gz")

    _summaries, failures = checker.scan_artifacts(tmp_path)

    for index in range(len(forbidden_texts)):
        assert any(f"boundary_text_{index}.txt" in failure for failure in failures)
