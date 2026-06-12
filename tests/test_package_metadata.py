import importlib.metadata
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cfg",
    ".cff",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def tracked_files() -> list[Path]:
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    return [Path(line) for line in out.splitlines()]


def test_license_metadata_is_mit():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    installed = importlib.metadata.metadata("grond")

    assert pyproject["project"]["license"] == "MIT"
    assert installed.get("License-Expression") == "MIT"
    assert (ROOT / "LICENSE").read_text().startswith("MIT License")
    assert "license: MIT" in (ROOT / "CITATION.cff").read_text()


def test_citation_title_matches_package_title():
    citation = (ROOT / "CITATION.cff").read_text()
    package_title = (ROOT / "src" / "grond" / "__init__.py").read_text()

    assert 'title: "GROND: Global Regularization Of Non-kinematical Diffraction"' in citation
    assert "Global Regularization Of Non-kinematical Diffraction" in package_title


def test_no_tracked_paper_workspace():
    paper_dir = "pa" + "per" + "/"
    assert not any(str(path).startswith(paper_dir) for path in tracked_files())


def test_tracked_text_is_portable():
    banned = (
        "/Users/" + "Mike",
        "/data/Data1/users/" + "mike",
        "/data/Data1/" + "mike",
    )
    offenders = []
    for rel_path in tracked_files():
        if rel_path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = (ROOT / rel_path).read_text(errors="ignore")
        for token in banned:
            if token in text:
                offenders.append(f"{rel_path}: {token}")

    assert offenders == []


def test_public_text_avoids_disfavored_dataset_labels():
    banned = ("Micro" + "ED", "3D " + "ED", "3D-" + "ED")
    offenders = []
    for rel_path in tracked_files():
        if rel_path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = (ROOT / rel_path).read_text(errors="ignore")
        for token in banned:
            if token in text:
                offenders.append(f"{rel_path}: {token}")

    assert offenders == []
