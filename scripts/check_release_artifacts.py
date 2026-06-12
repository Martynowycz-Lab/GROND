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


def text_from_bytes(data: bytes) -> str:
    """Decode archive members defensively for simple token scans."""

    return data.decode("utf-8", errors="ignore")


def scan_name(artifact: Path, name: str, failures: list[str]) -> None:
    """Check an archive member name."""

    for term in GENERATED_NAME_TERMS + PACKAGE_BOUNDARY_TERMS:
        if term in name:
            failures.append(f"{artifact.name}: {name}: name contains {term!r}")
            return


def scan_text(artifact: Path, name: str, text: str, failures: list[str]) -> None:
    """Check archive member text for private paths and package-boundary terms."""

    for term in PACKAGE_BOUNDARY_TERMS:
        if term in text:
            failures.append(f"{artifact.name}: {name}: text contains {term!r}")
            return


def scan_wheel(artifact: Path, failures: list[str]) -> int:
    """Scan one wheel and return its member count."""

    count = 0
    with zipfile.ZipFile(artifact) as zf:
        for name in zf.namelist():
            count += 1
            scan_name(artifact, name, failures)
            try:
                scan_text(artifact, name, text_from_bytes(zf.read(name)), failures)
            except KeyError:
                failures.append(f"{artifact.name}: {name}: could not read member")
    return count


def scan_sdist(artifact: Path, failures: list[str]) -> int:
    """Scan one source distribution and return its member count."""

    count = 0
    with tarfile.open(artifact) as tf:
        for member in tf.getmembers():
            count += 1
            scan_name(artifact, member.name, failures)
            if not member.isfile():
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                failures.append(f"{artifact.name}: {member.name}: could not read member")
                continue
            scan_text(artifact, member.name, text_from_bytes(extracted.read()), failures)
    return count


def scan_artifacts(path: Path) -> tuple[list[str], list[str]]:
    """Return summary lines and failure lines for all artifacts under path."""

    artifacts = sorted(path.iterdir())
    summaries: list[str] = []
    failures: list[str] = []
    for artifact in artifacts:
        if artifact.suffix == ".whl":
            count = scan_wheel(artifact, failures)
        elif artifact.name.endswith(".tar.gz"):
            count = scan_sdist(artifact, failures)
        else:
            continue
        summaries.append(f"{artifact.name}: {count} files")
    if not summaries:
        failures.append(f"{path}: no .whl or .tar.gz artifacts found")
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
