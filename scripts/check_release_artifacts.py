#!/usr/bin/env python3
"""Check built GROND release artifacts for package-boundary mistakes."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

GENERATED_NAME_TERMS = (
    "__pycache__",
    ".pyc",
    ".npz",
    ".npy",
    "legacy/engine/cache",
    ".ruff_cache",
    ".pytest_cache",
)

REPOSITORY_BOUNDARY_NAME_TERMS = (
    "exper" + "iments/",
    "open" + "dm/",
    "open_direct_" + "methods/",
    "open-direct-" + "methods/",
    "pa" + "per/",
    "manu" + "script/",
    "raw_" + "data/",
    "raw-" + "data/",
    "/raw/data/",
    "/cache/",
    "outputs/",
    "grond_out/",
)

RAW_DATA_SUFFIXES = (
    ".cbf",
    ".eer",
    ".expt",
    ".h5",
    ".hdf5",
    ".hkl",
    ".img",
    ".mrc",
    ".mrcs",
    ".mtz",
    ".nxs",
    ".npy",
    ".npz",
    ".pickle",
    ".pkl",
    ".refl",
    ".smv",
)

NON_PYTHON_PACKAGE_NAME_TERMS = (
    "package.json",
    "package-lock.json",
    "pn" + "pm-lock.yaml",
    "ya" + "rn.lock",
    "node_modules/",
)

NON_PYTHON_PACKAGE_SUFFIXES = (
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
)

NON_PYTHON_PACKAGE_TEXT_TERMS = (
    "Java" + "Script",
    "java" + "script",
    "n" + "pm install",
    "ya" + "rn install",
    "pn" + "pm install",
)

PACKAGE_BOUNDARY_TERMS = (
    "pa" + "per" + "/",
    "manu" + "script",
    "Chem" + "Rxiv",
    "Bio" + "Rxiv",
    "/Users/" + "Mike",
    "/data/Data1/users/" + "mike",
    "/data/Data1/" + "mike",
    "Micro" + "ED",
    "3D " + "ED",
    "3D-" + "ED",
)

REPOSITORY_BOUNDARY_TEXT_TERMS = (
    "exper" + "iments/",
    "open" + "dm",
    "open_direct_" + "methods",
    "open-direct-" + "methods",
    "open direct " + "methods",
    "pa" + "per/",
    "manu" + "script",
)

REQUIRED_SDIST_SUFFIXES = (
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

REQUIRED_WHEEL_SUFFIXES = (
    "grond/__init__.py",
    "grond/cli.py",
    "grond/io/row_contract.py",
    "grond/legacy/engine/bloch_engine.py",
    "grond/schemas/grond-row-v1.schema.json",
    "dist-info/METADATA",
    "dist-info/entry_points.txt",
)


def text_from_bytes(data: bytes) -> str:
    """Decode archive members defensively for simple token scans."""

    return data.decode("utf-8", errors="ignore")


def scan_name(artifact: Path, name: str, failures: list[str]) -> None:
    """Check an archive member name."""

    for term in GENERATED_NAME_TERMS + NON_PYTHON_PACKAGE_NAME_TERMS + PACKAGE_BOUNDARY_TERMS:
        if term in name:
            failures.append(f"{artifact.name}: {name}: name contains {term!r}")
            return
    folded_name = name.casefold()
    for term in REPOSITORY_BOUNDARY_NAME_TERMS:
        if term.casefold() in folded_name:
            failures.append(f"{artifact.name}: {name}: name crosses repository boundary {term!r}")
            return
    if folded_name.endswith(NON_PYTHON_PACKAGE_SUFFIXES):
        failures.append(f"{artifact.name}: {name}: name has a non-Python package suffix")
    elif folded_name.endswith(RAW_DATA_SUFFIXES):
        failures.append(f"{artifact.name}: {name}: name has a raw-data suffix")


def scan_text(artifact: Path, name: str, text: str, failures: list[str]) -> None:
    """Check archive member text for private paths and package-boundary terms."""

    for term in NON_PYTHON_PACKAGE_TEXT_TERMS + PACKAGE_BOUNDARY_TERMS:
        if term in text:
            failures.append(f"{artifact.name}: {name}: text contains {term!r}")
            return
    folded_text = text.casefold()
    for term in REPOSITORY_BOUNDARY_TEXT_TERMS:
        if term.casefold() in folded_text:
            failures.append(f"{artifact.name}: {name}: text crosses repository boundary {term!r}")
            return


def scan_wheel(artifact: Path, failures: list[str]) -> int:
    """Scan one wheel and return its member count."""

    count = 0
    names: list[str] = []
    with zipfile.ZipFile(artifact) as zf:
        for name in zf.namelist():
            count += 1
            names.append(name)
            scan_name(artifact, name, failures)
            try:
                scan_text(artifact, name, text_from_bytes(zf.read(name)), failures)
            except KeyError:
                failures.append(f"{artifact.name}: {name}: could not read member")
    for suffix in REQUIRED_WHEEL_SUFFIXES:
        if not any(name.endswith(suffix) for name in names):
            failures.append(f"{artifact.name}: missing required wheel file {suffix}")
    return count


def scan_sdist(artifact: Path, failures: list[str]) -> int:
    """Scan one source distribution and return its member count."""

    count = 0
    names: list[str] = []
    with tarfile.open(artifact) as tf:
        for member in tf.getmembers():
            count += 1
            names.append(member.name)
            scan_name(artifact, member.name, failures)
            if not member.isfile():
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                failures.append(f"{artifact.name}: {member.name}: could not read member")
                continue
            scan_text(artifact, member.name, text_from_bytes(extracted.read()), failures)
    for suffix in REQUIRED_SDIST_SUFFIXES:
        if not any(name.endswith(suffix) for name in names):
            failures.append(f"{artifact.name}: missing required source file {suffix}")
    return count


def scan_artifacts(path: Path) -> tuple[list[str], list[str]]:
    """Return summary lines and failure lines for all artifacts under path."""

    artifacts = sorted(path.iterdir())
    summaries: list[str] = []
    failures: list[str] = []
    saw_wheel = False
    saw_sdist = False
    for artifact in artifacts:
        if artifact.suffix == ".whl":
            saw_wheel = True
            count = scan_wheel(artifact, failures)
        elif artifact.name.endswith(".tar.gz"):
            saw_sdist = True
            count = scan_sdist(artifact, failures)
        else:
            continue
        summaries.append(f"{artifact.name}: {count} files")
    if not summaries:
        failures.append(f"{path}: no .whl or .tar.gz artifacts found")
    elif not saw_wheel or not saw_sdist:
        failures.append(f"{path}: release check expects both a wheel and a source distribution")
    return summaries, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path, help="Directory containing built wheel and sdist artifacts.")
    args = parser.parse_args()

    summaries, failures = scan_artifacts(args.artifact_dir)
    for line in summaries:
        print(line)
    if failures:
        print("Release artifact check failed:")
        for line in failures[:50]:
            print(line)
        return 1
    print("Release artifact check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
