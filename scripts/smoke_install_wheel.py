#!/usr/bin/env python3
"""Install a built GROND wheel into a fresh venv and smoke-test entry points."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import venv
from pathlib import Path


def find_one_wheel(dist_dir: Path) -> Path:
    """Return the single wheel artifact under dist_dir."""

    wheels = sorted(dist_dir.glob("*.whl"))
    if len(wheels) != 1:
        names = ", ".join(path.name for path in wheels) or "none"
        raise SystemExit(f"expected exactly one wheel in {dist_dir}, found {names}")
    return wheels[0]


def venv_python(venv_dir: Path) -> Path:
    """Return the venv Python path for the current platform."""

    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def venv_script(venv_dir: Path, name: str) -> Path:
    """Return an installed console-script path for the current platform."""

    suffix = ".exe" if os.name == "nt" else ""
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    return venv_dir / scripts_dir / f"{name}{suffix}"


def run_checked(cmd: list[str | Path], *, cwd: Path) -> None:
    """Run one smoke command and print it for CI logs."""

    printable = " ".join(str(part) for part in cmd)
    print(f"+ {printable}")
    subprocess.run([str(part) for part in cmd], cwd=cwd, check=True)


def smoke_install_wheel(dist_dir: Path) -> None:
    """Install the built wheel and verify module plus console entry points."""

    wheel = find_one_wheel(dist_dir.resolve())
    with tempfile.TemporaryDirectory(prefix="grond-wheel-smoke-") as tmp:
        tmp_dir = Path(tmp)
        venv_dir = tmp_dir / "venv"
        venv.EnvBuilder(with_pip=True).create(venv_dir)
        python = venv_python(venv_dir)

        run_checked([python, "-m", "pip", "install", "--disable-pip-version-check", str(wheel)], cwd=tmp_dir)
        run_checked([python, "-m", "grond", "--help"], cwd=tmp_dir)
        run_checked([python, "-m", "grond", "validate"], cwd=tmp_dir)
        run_checked([python, "-m", "grond", "doctor", "--probe-bloch"], cwd=tmp_dir)
        run_checked([venv_script(venv_dir, "grond"), "--help"], cwd=tmp_dir)
        run_checked([venv_script(venv_dir, "GROND"), "--help"], cwd=tmp_dir)

    print(f"Wheel install smoke test passed for {wheel.name}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist_dir", type=Path, help="Directory containing exactly one built GROND wheel.")
    args = parser.parse_args()

    smoke_install_wheel(args.dist_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
