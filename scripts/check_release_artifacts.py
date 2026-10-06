#!/usr/bin/env python3
"""Check built GROND release artifacts for package-boundary mistakes."""

from __future__ import annotations

import argparse
import re
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

# Working directories that belong next to a checkout, never inside a release.
LOCAL_DIRECTORY_NAME_TERMS = (
    "experiments/",
    "raw_data/",
    "raw-data/",
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
    "pnpm-lock.yaml",
    "yarn.lock",
    "node_modules/",
)

NON_PYTHON_PACKAGE_SUFFIXES = (
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
)

# Text that must not ship: absolute paths into a personal home or data directory, private keys and
# instructions for a non-Python package manager. Each pattern needs a path component or keyword after the
# prefix, so the patterns do not match their own source.
FORBIDDEN_TEXT_PATTERNS = (
    ("an absolute macOS home path", re.compile(r"/Users/[\w.-]+/")),
    ("an absolute Linux home path", re.compile(r"/home/[\w.-]+/")),
    ("an absolute mounted-volume path", re.compile(r"/Volumes/[\w. -]+/")),
    ("an absolute data-volume path", re.compile(r"/data/Data\d")),
    ("a private key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("a non-Python package manager command", re.compile(r"\b(?:npm|yarn|pnpm) install\b")),
)

REQUIRED_SDIST_SUFFIXES = (
    "ACKNOWLEDGMENTS.md",
    "DATA_POLICY.md",
    "SECURITY.md",
    "docs/DEVELOPMENT.md",
    "docs/INSTALL.md",
    "docs/USAGE.md",
    "docs/REFERENCES.md",
    "scripts/check_release_artifacts.py",
    "scripts/smoke_install_wheel.py",
    "tests/test_physics.py",
    "tests/reference/reference_lib.py",
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

    for term in GENERATED_NAME_TERMS + NON_PYTHON_PACKAGE_NAME_TERMS:
        if term in name:
            failures.append(f"{artifact.name}: {name}: name contains {term!r}")
            return
    folded_name = name.casefold()
    for term in LOCAL_DIRECTORY_NAME_TERMS:
        if term in folded_name:
            failures.append(f"{artifact.name}: {name}: name is inside a local working directory {term!r}")
            return
    if folded_name.endswith(NON_PYTHON_PACKAGE_SUFFIXES):
        failures.append(f"{artifact.name}: {name}: name has a non-Python package suffix")
    elif folded_name.endswith(RAW_DATA_SUFFIXES):
        failures.append(f"{artifact.name}: {name}: name has a raw-data suffix")


def scan_text(artifact: Path, name: str, text: str, failures: list[str]) -> None:
    """Check archive member text for private paths, private keys and non-Python package instructions."""

    for label, pattern in FORBIDDEN_TEXT_PATTERNS:
        match = pattern.search(text)
        if match:
            failures.append(f"{artifact.name}: {name}: text contains {label} ({match.group(0)!r})")
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
